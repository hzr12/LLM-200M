"""Mixture-of-Experts Transformer (RMSNorm + RoPE + GQA + SwiGLU, top-k routing).

References:
  - Llama (RoPE, RMSNorm, GQA)
  - Mixtral / DeepSeek-V2 MoE routing (top-k, softmax scores, aux + z loss)
  - NanoGPT-style memmap data loading is kept in train.py

The router is shared across layers by default (single tiny tensor), which keeps
the expert-router parameters tiny. Each layer has its own experts.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm（NPU 上可选 npu_rms_norm 融合算子，Ascend C JIT 编译可用）。

    CANN 8.0.RC1 实测：npu_rms_norm 走 te_rmsnorm JIT 编译路径（fwd+bwd 可用、
    数值误差 ~2^-7 量级），而 npu_swiglu / npu_apply_rotary_pos_emb 等静态
    二进制算子缺失。首次 forward 用真实 shape 探测一次（含 backward），失败
    类级缓存回退到逐元素实现。
    """
    _npu_rms_ok: bool | None = None

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    @classmethod
    def _probe_npu_rms_norm(cls, x: torch.Tensor, w: torch.Tensor,
                            eps: float) -> bool:
        """真实 (B, T, dim) shape 探测 npu_rms_norm forward+backward。"""
        try:
            import torch_npu  # noqa: F401
            xx = torch.randn_like(x, requires_grad=True)
            out = torch_npu.npu_rms_norm(xx, w, eps)
            y = out[0] if isinstance(out, tuple) else out
            y.sum().backward()
            torch.npu.synchronize()
            return True
        except Exception:  # noqa: BLE001
            return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "npu" and x.dtype in (torch.float16, torch.bfloat16):
            if RMSNorm._npu_rms_ok is None:
                RMSNorm._npu_rms_ok = self._probe_npu_rms_norm(
                    x, self.weight.to(x.dtype), self.eps)
            if RMSNorm._npu_rms_ok:
                try:
                    import torch_npu  # noqa: F401
                    out = torch_npu.npu_rms_norm(x, self.weight.to(x.dtype),
                                                 self.eps)
                    return out[0] if isinstance(out, tuple) else out
                except Exception:  # noqa: BLE001
                    RMSNorm._npu_rms_ok = False  # 运行时失败 → 全局回退
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope(head_dim: int, seq_len: int, theta: float,
                    scaling: dict | None = None) -> torch.Tensor:
    """Return (cos, sin) of shape [seq_len, head_dim/2]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    if scaling is not None:  # NTK-aware scaling (llama3 style)
        factor = scaling.get("factor", 1.0)
        low_freq_factor = scaling.get("low_freq_factor", 1.0)
        high_freq_factor = scaling.get("high_freq_factor", 4.0)
        old_ctx_len = 4096
        low_freq_wavelen = old_ctx_len / low_freq_factor
        high_freq_wavelen = old_ctx_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq = torch.where(
            wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth = torch.max(
            torch.zeros_like(wavelen),
            (high_freq_wavelen - wavelen) / (high_freq_wavelen - low_freq_wavelen))
        inv_freq = torch.where(wavelen < high_freq_wavelen, inv_freq, inv_freq * (1 + smooth * (factor - 1)))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, heads, T, head_dim]; cos,sin: [T, head_dim/2]
    # RoPE pairs are (q[..., i], q[..., i+head_dim/2]); split first then mix.
    # cos/sin 由 Model._rope 按 q.dtype 预缓存（命中时 dtype 一致，.to 为 no-op）；
    # 仍保留 .to 作为兜底，保证 autocast 下 q.dtype != cos.dtype 时正确。
    cos = cos.to(q.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim/2]
    sin = sin.to(q.dtype).unsqueeze(0).unsqueeze(0)
    q1, q2 = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2:]
    q = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
    k1, k2 = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2:]
    k = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)
    return q, k


class Attention(nn.Module):
    # Class-level synchronous probe result for npu_fusion_attention:
    # None = untested, True = usable, False = broken on this CANN install.
    # Shared by all Attention instances (one probe per process).
    _npu_fa_ok: bool | None = None
    _npu_fa_warned: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.repeats = self.n_heads // self.n_kv_heads
        self.use_flash_attn = cfg.use_flash_attn
        self.fa_layout = cfg.fa_layout
        self._scale = self.head_dim ** -0.5
        d = cfg.d_model
        # qkv 合并为单个 GEMM（[D, (H + 2*KV)*D]）：3 个小 GEMM 变 1 个，
        # 宽度 768→1024，NPU tiling 更高效。旧 checkpoint 的 q_proj/k_proj/
        # v_proj.weight 由 _load_from_state_dict 兼容拼接。
        self.qkv_proj = nn.Linear(d, (self.n_heads + 2 * self.n_kv_heads)
                                  * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        # Cached additive causal mask (upper triangle = -inf), lazily built in
        # forward. Replacing per-call triu()+masked_fill() with a broadcast add
        # removes one O(T^2) elementwise kernel per layer per micro-batch.
        self.register_buffer("_causal_bias", None, persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """旧 checkpoint 兼容：q_proj/k_proj/v_proj.weight → qkv_proj.weight。

        qkv 合并前 state_dict 存三个独立权重；合并后只有 qkv_proj.weight。
        这里在加载前把旧三权重按 out 维拼接（q 在前、k 中、v 后），旧键被
        消费掉不会出现在 unexpected_keys。
        """
        qk, kk, vk = (prefix + "q_proj.weight", prefix + "k_proj.weight",
                      prefix + "v_proj.weight")
        if qk in state_dict and prefix + "qkv_proj.weight" not in state_dict:
            w_q = state_dict.pop(qk)
            w_k = state_dict.pop(kk)
            w_v = state_dict.pop(vk)
            state_dict[prefix + "qkv_proj.weight"] = torch.cat([w_q, w_k, w_v],
                                                               dim=0)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @staticmethod
    def _npu_fa_files_ok() -> bool:
        """纯文件系统判定：CANN OPP 的 FA 注册 config json 是否可解析。

        8.0.RC1 上 flash_attention_score_grad.json 已损坏（"Cannot parse json"）
        → 直接判定不可用，绝不调用设备算子（FA 失败会污染设备状态：
        side-stream 数据拷贝与后续算子输出野值，drain 只能防崩溃救不回
        状态——minirepro 实测 E2/E3 验证）。
        """
        import glob
        import json as _json
        roots = [os.environ.get("ASCEND_OPP_PATH", ""),
                 "/usr/local/Ascend/ascend-toolkit/latest/opp"]
        for base in roots:
            if not base:
                continue
            cfg_dir = os.path.join(base, "built-in", "op_impl", "ai_core", "tbe",
                                   "kernel", "config", "ascend910")
            jsons = sorted(glob.glob(os.path.join(cfg_dir, "flash_attention_score*.json")))
            for j in jsons:
                try:
                    with open(j) as f:
                        _json.load(f)
                except Exception:  # noqa: BLE001
                    return False   # 有文件但损坏 → 不可用
            if jsons:
                return True        # 存在且全部可解析 → 才允许设备探测
        return False

    def _npu_fa_probe(self, device: torch.device, dtype: torch.dtype,
                      seq_len: int = 8, batch: int = 1) -> bool:
        """Synchronously check whether NPU fused attention is usable.

        文件预检先行（CANN 算子注册 json 损坏/缺失 → 零设备操作判定不可用）；
        预检通过才做真实形状 forward+backward 设备探测。失败则全局禁用。
        结果缓存在类上（_npu_fa_ok），forward 不再每层重测。
        """
        if Attention._npu_fa_ok is not None:
            return Attention._npu_fa_ok
        # 文件预检：FA 在此 CANN 必然不可用（注册损坏），设备探测会污染
        # 设备状态（side-stream 拷贝返回野值、后续算子输出野值，drain 救不回），
        # 因此判定完全基于文件系统，绝不调用设备算子。
        if not self._npu_fa_files_ok():
            Attention._npu_fa_ok = False
            if not Attention._npu_fa_warned:
                Attention._npu_fa_warned = True
                print("warning: NPU fused attention unavailable on this CANN "
                      "install (FA op registration missing/corrupt: "
                      "flash_attention_score*.json under ascend910 kernel config); "
                      "disabling FlashAttention, falling back to slow attention "
                      f"(seq_len={seq_len}, batch={batch}, dtype={dtype})", flush=True)
            return False
        try:
            import torch_npu  # noqa: F401  # 确保 NPU 后端已注册
            S = seq_len
            # probe 必须镜像 forward 的真实路径：真实 batch + 真实 GQA 形状
            # + repeat + 显式 fusion_attention（与 forward 同一调用与参数）。
            q = torch.randn(batch, self.n_heads, S, self.head_dim, dtype=dtype,
                            device=device, requires_grad=True)
            k = torch.randn(batch, self.n_kv_heads, S, self.head_dim, dtype=dtype,
                            device=device, requires_grad=True)
            v = torch.randn(batch, self.n_kv_heads, S, self.head_dim, dtype=dtype,
                            device=device, requires_grad=True)
            if self.repeats > 1:
                k = k.repeat_interleave(self.repeats, dim=1)
                v = v.repeat_interleave(self.repeats, dim=1)
            out = torch_npu.npu_fusion_attention(
                q, k, v, self.n_heads, "BNSD",
                scale=self._scale,
                pre_tockens=2147483647, next_tockens=0, sparse_mode=2,
                sync=True)
            y = next(t for t in out
                     if tuple(getattr(t, "shape", ()))
                     == (batch, self.n_heads, S, self.head_dim))
            y.sum().backward()  # must exercise the fused-attention backward kernel
            torch.npu.synchronize()
            Attention._npu_fa_ok = True
        except Exception as e:  # noqa: BLE001
            Attention._npu_fa_ok = False
            # 防御性 drain（文件预检已通过才可能走到这里）：probe 触发的失败
            # kernel 会以异步错误形式残留排队，不清会导致后续设备操作崩溃。
            for _ in range(5):
                try:
                    torch.npu.synchronize()
                    torch.zeros(1, device=device).item()
                except Exception:  # noqa: BLE001
                    pass
            if not Attention._npu_fa_warned:
                Attention._npu_fa_warned = True
                print(f"warning: NPU fused attention unavailable on this CANN "
                      f"({e!r}); disabling FlashAttention, falling back to slow "
                      f"attention (seq_len={seq_len}, batch={batch}, dtype={dtype})",
                      flush=True)
        return Attention._npu_fa_ok

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # qkv 合并 GEMM → split（q 在前、k 中、v 后）
        qkv = self.qkv_proj(x)
        dq = self.n_heads * self.head_dim
        dkv = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([dq, dkv, dkv], dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        # GQA 头数对齐：
        # - CUDA 的 SDPA 原生支持 GQA broadcast，可直接传原始 k/v。
        # - NPU 的 FusedAttention 不支持 GQA 头数自动 broadcast，走融合
        #   分支前必须 repeat k/v（见下方 npu 分支）。
        # - flash-attn(CUDA) / 慢速路径 也需等头数。
        dev_type = x.device.type
        # NPU 融合分支只在启动期 check_flash_attn 探测通过后启用（_npu_fa_ok
        # 为 True）；未探测(None)/探测失败(False) 一律走慢速路径——torch_npu
        # 2.1 的 SDPA 无融合后端（math 实现），不值得走 SDPA 分支。
        use_fused = (q.dtype in (torch.float16, torch.bfloat16)
                     and (dev_type == "cuda"
                          or (dev_type == "npu" and self.use_flash_attn
                              and Attention._npu_fa_ok is True)))
        if use_fused:
            # 1) CUDA 优先 DAO flash-attn（需 repeat 后等头数 k/v；未装/失败落 SDPA）
            if dev_type == "cuda":
                k_r = k.repeat_interleave(self.repeats, dim=1)
                v_r = v.repeat_interleave(self.repeats, dim=1)
                try:
                    from flash_attn import flash_attn_func
                    y = flash_attn_func(
                        q.transpose(1, 2).contiguous(), k_r.transpose(1, 2).contiguous(),
                        v_r.transpose(1, 2).contiguous(), causal=True)
                    return self.o_proj(y.reshape(B, T, -1))
                except ImportError:
                    pass  # flash-attn 未安装 → 走 SDPA
                except Exception as e:  # noqa: BLE001
                    # flash-attn 运行失败（shape 不支持等）→ 回退 SDPA
                    if not Attention._npu_fa_warned:
                        Attention._npu_fa_warned = True
                        print(f"warning: flash_attn_func failed on cuda ({e!r}); "
                              f"falling back to SDPA", flush=True)
                # CUDA SDPA：原生支持 GQA broadcast，直接传原始 k/v 即可
                try:
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))
                except Exception as e:  # noqa: BLE001
                    if not Attention._npu_fa_warned:
                        Attention._npu_fa_warned = True
                        print(f"warning: scaled_dot_product_attention failed on cuda "
                              f"({e!r}); falling back to slow attention", flush=True)
            else:  # NPU：显式 npu_fusion_attention（SDPA 在 torch_npu 2.1 无融合后端）
                try:
                    import torch_npu  # noqa: F401
                    if self.repeats > 1:
                        kr = k.repeat_interleave(self.repeats, dim=1)
                        vr = v.repeat_interleave(self.repeats, dim=1)
                    else:
                        kr, vr = k, v
                    out = torch_npu.npu_fusion_attention(
                        q, kr, vr, self.n_heads, "BNSD",
                        scale=self._scale,
                        pre_tockens=2147483647, next_tockens=0, sparse_mode=2)
                    y = next(t for t in out
                             if tuple(getattr(t, "shape", ()))
                             == (B, self.n_heads, T, self.head_dim))
                    return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))
                except Exception as e:  # noqa: BLE001
                    # 运行时失败也全局禁用（probe 已同步验证过，这里只是兜底）
                    Attention._npu_fa_ok = False
                    if not Attention._npu_fa_warned:
                        Attention._npu_fa_warned = True
                        print(f"warning: npu_fusion_attention failed at runtime ({e!r}); "
                              f"disabling fused attention, falling back to slow "
                              f"attention (q.dtype={q.dtype})", flush=True)
        # slow fallback: cached additive causal bias instead of rebuilding
        # torch.triu(torch.ones(T,T)) + masked_fill() every forward call.
        # 慢速路径需要等头数 k/v，此处再 repeat（非 fused 路径才走到）。
        k = k.repeat_interleave(self.repeats, dim=1)
        v = v.repeat_interleave(self.repeats, dim=1)
        # slow fallback: cached additive causal bias instead of rebuilding
        # torch.triu(torch.ones(T,T)) + masked_fill() every forward call.
        if (self._causal_bias is None
                or self._causal_bias.shape[-1] < T
                or self._causal_bias.dtype != x.dtype):
            b = torch.triu(torch.full((T, T), float("-inf"),
                                      device=x.device, dtype=x.dtype), diagonal=1)
            self._causal_bias = b[None, None]
        # Fold the 1/sqrt(d) scale into q (O(T*D) work) instead of scaling the
        # full O(T^2) QK^T output, and add the causal bias in-place to avoid an
        # extra O(T^2) allocation per layer per micro-batch.
        att = (q * self._scale) @ k.transpose(-2, -1)
        att.add_(self._causal_bias[..., :T, :T])
        att = F.softmax(att, dim=-1).to(v.dtype)
        y = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Expert(nn.Module):
    """A single FFN expert (SwiGLU)."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.mlp = SwiGLU(cfg.d_model, cfg.expert_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


@dataclass
class RouterOutput:
    logits: torch.Tensor    # [B*T, n_experts]
    probs: torch.Tensor     # [B*T, n_experts] softmax(logits.float())
    top_probs: torch.Tensor # [B*T, top_k]
    top_idx: torch.Tensor   # [B*T, top_k]
    z_loss: torch.Tensor    # scalar
    aux_loss: torch.Tensor  # scalar


class Router(nn.Module):
    """Learns a per-token score over experts. Shared across layers.

    z_loss = mean(logsumexp(logits)^2): discourages huge logits
    aux_loss = n_experts * sum(freq * load)  (switch-transformer style balance)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.z_loss_coef = cfg.router_z_loss_coef
        self.aux_loss_coef = cfg.router_aux_loss_coef
        self.proj = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    def forward(self, x: torch.Tensor, use_aux_loss: bool = True) -> RouterOutput:
        logits = self.proj(x)                     # [B*T, E]
        # 同一 logits 只转 fp32 一次（softmax 与 z_loss 共用）：
        # 每层每次调用少一次 [S, E] cast，fwd+重算 12 层 = 每步少 24 次 cast。
        flogits = logits.float()
        probs = F.softmax(flogits, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=self.top_k, dim=-1)

        z_loss = torch.logsumexp(flogits, dim=-1).pow(2).mean() if use_aux_loss else torch.zeros((), device=x.device)

        aux_loss = torch.zeros((), device=x.device)
        if use_aux_loss and self.training:
            B = x.shape[0] * x.shape[1]
            # load balancing: uniform expert load
            probs_sum = probs.sum(dim=0)          # [E]
            # one-hot via expand/compare: fixed shape, no scatter_ on NPU
            onehot = F.one_hot(top_idx, num_classes=self.n_experts).float()  # [B, k, E]
            expert_load = onehot.sum(dim=(0, 1))  # [E]
            f = probs_sum / B
            p = expert_load / B
            aux_loss = self.n_experts * (f * p).sum()
        return RouterOutput(logits, probs, top_probs, top_idx, z_loss, aux_loss)


class MoEBlock(nn.Module):
    """Attention + shared FFN + MoE with top-k routing."""
    def __init__(self, cfg: Config, layer_id: int, router: Router):
        super().__init__()
        self.layer_id = layer_id
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.router = router
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.sparse_moe = cfg.sparse_moe
        self.shared_mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden) if cfg.mlp_hidden else None
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])
        self.n_shared_experts = cfg.n_shared_experts
        if cfg.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [SwiGLU(cfg.d_model, cfg.shared_expert_hidden) for _ in range(cfg.n_shared_experts)])
        else:
            self.shared_experts = nn.ModuleList()
        # fused per-expert weight matrices (W_gu/W_down), rebuilt after
        # every optimizer step instead of cat()-ing them each forward call.
        # 统一为 dense 大矩阵：[2*E*H, D]（gate/up 合并）/ [D, E*H]，sparse 与
        # dense 共用同一套固定 shape 算子（在 NPU 上只跑大 GEMM，无零散 scatter）。
        self._expert_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.norm1(x), cos, sin)
        n = self.norm2(h)
        routed, losses = self._moe(n)
        h = h + routed
        if self.shared_mlp is not None:
            h = h + self.shared_mlp(n)
        return h, losses

    def refresh_expert_cache(self) -> None:
        """Rebuild the fused per-expert weight matrices after weights change.

        Must be called after optimizer.step() (or load_state_dict) so the
        cached cat() views stay in sync with the expert parameters. The cache
        is a plain tensor built from the Parameter leaves (NOT a detached
        buffer), so gradients flow back to the experts normally.

        sparse 与 dense 共用同一份 dense 大矩阵缓存：gate/up 合并为单个
        [2*E*H, D] 权重（一次 GEMM 出 g 和 u，再 chunk——2 个 [S, E*H] 大
        GEMM 变 1 个 [S, 2*E*H]，NPU 上 kernel 数减半、tiling 更高效）；
        down 保持 [D, E*H]。sparse 靠 top-k 权重 mask 掉未选中专家（在激活上、
        down 之前乘权重），dense 用全 1 权重——两者在 NPU 上都是固定 shape
        的大 GEMM，无任何 sort/index_put/bincount 零散算子。
        """
        H = self.experts[0].mlp.gate.out_features
        W_gate = torch.cat([e.mlp.gate.weight for e in self.experts], dim=0)  # [E*H, D]
        W_up = torch.cat([e.mlp.up.weight for e in self.experts], dim=0)      # [E*H, D]
        W_gu = torch.cat([W_gate, W_up], dim=0)                               # [2*E*H, D]
        W_down = torch.cat([e.mlp.down.weight for e in self.experts], dim=1)  # [D, E*H]
        # device/shape 变化时重建，否则原地 copy_（保证 .cuda()/.to() 后缓存跟随）
        if (self._expert_cache is None
                or self._expert_cache[0].device != W_gu.device
                or self._expert_cache[0].shape != W_gu.shape):
            self._expert_cache = (W_gu, W_down)
        else:
            for dst, src in zip(self._expert_cache, (W_gu, W_down)):
                dst.copy_(src)

    def _moe(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.sparse_moe:
            return self._moe_sparse(x)
        return self._moe_dense(x)

    def _moe_sparse(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Sparse top-k MoE，优化为 dense-mask 架构（910 Pro A 友好）。

        不再走 "pad + bmm + host 侧 sort/index_put/bincount" 路线——那一路在
        Ascend 上把计算切碎成大量低效零散算子，每个新 shape 触发一次 CANN
        编译，导致前 10 分钟编译、NPU 利用率趋零。

        改为：对所有专家做一次固定 shape 的大 GEMM（与 dense 同构），top-k
        路由权重在激活 a 上、down 投影之前乘进去，未选中专家权重=0 即被
        mask 掉。数学上等价于「只算选中专家」，但 NPU 上只跑 3 次大 GEMM，
        无任何 scatter/gather，shape 恒定 → 编译一次复用，利用率拉满。

        注意：top-k 加权必须在 a 上、down 之前做，与 dense 的 (a*w)@W_down
        保持一致（线性变换不满足后加权）。
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        out = self.router(flat)
        top_probs = out.top_probs.to(x.dtype)   # [S, k]
        top_idx = out.top_idx                   # [S, k] long
        z_loss, aux_loss = out.z_loss, out.aux_loss
        S = flat.shape[0]
        E, H = self.n_experts, self.experts[0].mlp.gate.out_features

        if self._expert_cache is None or self._expert_cache[0].device != flat.device:
            self.refresh_expert_cache()
        W_gu, W_down = self._expert_cache                    # [2*E*H, D] / [D, E*H]

        # --- 固定 shape 大 GEMM：gate/up 合并一次算全部专家（autocast 下 fp16）---
        gu_all = F.linear(flat, W_gu)                        # [S, 2*E*H]
        g_all, u_all = gu_all.chunk(2, dim=-1)
        g = g_all.view(S, E, H)
        u = u_all.view(S, E, H)
        a = F.silu(g) * u                                  # [S, E, H]

        # --- top-k 路由权重：one-hot * top_probs -> [S, E]，未选中专家权重=0 ---
        onehot = F.one_hot(top_idx, num_classes=E).to(top_probs.dtype)  # [S, k, E]
        w = (onehot * top_probs.unsqueeze(-1)).sum(dim=1)   # [S, E]
        a = a * w.unsqueeze(-1)                            # mask 掉未选中专家

        y = F.linear(a.reshape(S, E * H), W_down)          # [S, D]
        y = y.reshape(B, T, D)

        if self.n_shared_experts > 0:
            for se in self.shared_experts:
                y = y + se(x)

        return y, {"router_z_loss": z_loss, "router_aux_loss": aux_loss}

    def _moe_dense(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Dense MoE 回退路径：在 910 Pro A 上与 sparse 行为一致。

        原 dense 用「全专家激活 + top-k 加权」会有 E/k 倍冗余 FLOPs，在 NPU
        上并不比 sparse 快（反而更慢）。故 dense 直接复用 dense-mask 架构：
        top-k 权重在激活上乘入，未选中专家被 mask 掉——既保留 MoE 语义，又
        只跑固定 shape 大 GEMM。两者唯一区别是 --sparse-moe 的开关含义仅用于
        路由权重是否参与（此处同样参与，等价于统一实现）。
        """
        return self._moe_sparse(x)


class MoETransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.router = Router(cfg)
        self.layers = nn.ModuleList(
            [MoEBlock(cfg, i, self.router) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._rope_cache_dtype: dict = {}
        # Rebuild fused expert-weight caches once a checkpoint has been fully
        # loaded (load_state_dict is recursive; _load_from_state_dict runs too
        # early to see the expert weights). They are plain attributes, not
        # buffers, so they are NOT part of state_dict.
        self.register_load_state_dict_post_hook(self._post_load_rebuild)

        self.apply(self._init_weights)
        # small init for output layer
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is self.lm_head:
                nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def refresh_expert_caches(self) -> None:
        """Rebuild all fused per-expert weight caches (after step / load)."""
        for layer in self.layers:
            layer.refresh_expert_cache()

    def check_flash_attn(self, device: torch.device, dtype: torch.dtype,
                         seq_len: int = 8, batch: int = 1) -> bool:
        """Pre-flight synchronous FA availability probe (Ascend only).

        Call once after model.to(device) before training, passing the real
        training sequence length (--ctx) and micro-batch. Probes forward AND
        backward with the real (batch, heads, seq) tiling shape; on failure FA
        is globally disabled inside the probe (falls back to slow attention) and
        this returns False. batch 透传真实 micro-batch，因为部分 CANN 版本的
        FusedAttention kernel 可用性依赖 batch tiling。
        """
        if len(self.layers) == 0:
            return False
        return self.layers[0].attn._npu_fa_probe(device, dtype, seq_len, batch)

    def _post_load_rebuild(self, module, incompatible_keys) -> None:
        """Called after load_state_dict finishes (all submodules loaded)."""
        self.refresh_expert_caches()

    def _rope(self, seq_len: int, device: torch.device,
              dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        # 缓存按 dtype 分桶：cos/sin 在 fp32 预计算一次，按当前激活 dtype 转换
        # 后缓存，避免每步 forward 都 .to(q.dtype)（NPU 上每步转换一次是浪费）。
        if self._rope_cache is None or self._rope_cache[0].shape[0] < seq_len:
            cos, sin = precompute_rope(self.cfg.head_dim, max(seq_len, self.cfg.max_seq_len),
                                       self.cfg.rope_theta, self.cfg.rope_scaling)
            self._rope_cache = (cos.to(device), sin.to(device))
            self._rope_cache_dtype: dict = {}
        key = (dtype, device)
        if key not in self._rope_cache_dtype:
            cos, sin = self._rope_cache
            self._rope_cache_dtype[key] = (cos.to(dtype).to(device), sin.to(dtype).to(device))
        c, s = self._rope_cache_dtype[key]
        return c[:seq_len], s[:seq_len]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """idx: [B, T]. Returns (logits, extra_losses)."""
        B, T = idx.shape
        x = self.token_embedding(idx)
        cos, sin = self._rope(T, idx.device, x.dtype)
        losses = {"router_z_loss": torch.zeros((), device=idx.device),
                  "router_aux_loss": torch.zeros((), device=idx.device)}
        n_layers = len(self.layers)
        for layer in self.layers:
            if self.cfg.gradient_checkpointing and self.training:
                # 直接把 layer 传给 checkpoint（nn.Module 可调用）。
                # 不要用 lambda 包裹——闭包捕获循环变量在反向重算时会解析成最后一层。
                x, ls = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, use_reentrant=False)
            else:
                x, ls = layer(x, cos, sin)
            for k in losses:
                losses[k] = losses[k] + ls[k]
        # average over layers so the reported value is per-layer
        for k in losses:
            losses[k] = losses[k] / n_layers
        x = self.norm(x)
        logits = self.lm_head(x)
        total_loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1), ignore_index=-1)
            total_loss = (loss
                          + self.cfg.router_z_loss_coef * losses["router_z_loss"]
                          + self.cfg.router_aux_loss_coef * losses["router_aux_loss"])
        return logits, losses | {"total": total_loss}

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 40, eos_id: int | None = None) -> torch.Tensor:
        """Autoregressive sampling (prompt-kv-cache optional; simple full-context version)."""
        self.eval()
        for _ in range(max_new_tokens):
            seq = idx if idx.shape[1] <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(seq)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        self.train()
        return idx
