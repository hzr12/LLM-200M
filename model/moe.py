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
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        # Cached additive causal mask (upper triangle = -inf), lazily built in
        # forward. Replacing per-call triu()+masked_fill() with a broadcast add
        # removes one O(T^2) elementwise kernel per layer per micro-batch.
        self.register_buffer("_causal_bias", None, persistent=False)

    def _npu_fa_probe(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Synchronously check whether npu_fusion_attention is usable.

        The real operator is ASYNC: if this CANN install has no kernel for it
        (e.g. "FlashAttentionScore does not has any binary"), the failure only
        surfaces at the next sync point (backward / copy), so a try/except
        around the call itself can never catch it and training crashes. We run
        one small-shape probe + torch.npu.synchronize() up front and globally
        disable FA on failure. Result is cached on the class.
        """
        if Attention._npu_fa_ok is not None:
            return Attention._npu_fa_ok
        try:
            import torch_npu
            S = 8
            q = torch.zeros(1, self.n_heads, S, self.head_dim, dtype=dtype, device=device)
            k = torch.zeros(1, self.n_heads, S, self.head_dim, dtype=dtype, device=device)
            v = torch.zeros(1, self.n_heads, S, self.head_dim, dtype=dtype, device=device)
            m = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool),
                           diagonal=1).view(1, 1, S, S)
            layout = self.fa_layout.strip().lower()
            if layout == "bnsd":
                y, *_ = torch_npu.npu_fusion_attention(
                    q, k, v, self.n_heads, "BNSD", atten_mask=m,
                    scale=self._scale, inner_precise=0)
            else:
                qh = q.transpose(1, 2).reshape(1, S, -1).contiguous()
                kh = k.transpose(1, 2).reshape(1, S, -1).contiguous()
                vh = v.transpose(1, 2).reshape(1, S, -1).contiguous()
                y, *_ = torch_npu.npu_fusion_attention(
                    qh, kh, vh, self.n_heads, "BSH", atten_mask=m,
                    scale=self._scale, inner_precise=0)
            torch.npu.synchronize()
            Attention._npu_fa_ok = True
        except Exception as e:  # noqa: BLE001
            Attention._npu_fa_ok = False
            if not Attention._npu_fa_warned:
                Attention._npu_fa_warned = True
                print(f"warning: npu_fusion_attention unavailable on this CANN install "
                      f"({e!r}); disabling FlashAttention, falling back to slow attention "
                      f"(layout={self.fa_layout}, dtype={dtype})", flush=True)
        return Attention._npu_fa_ok

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        # GQA: repeat kv heads
        k = k.repeat_interleave(self.repeats, dim=1)
        v = v.repeat_interleave(self.repeats, dim=1)
        # flash attention when available (CUDA flash-attn or Ascend npu_fusion_attention)
        dev_type = x.device.type
        try:
            if dev_type == "cuda" and q.dtype in (torch.float16, torch.bfloat16):
                from flash_attn import flash_attn_func
                y = flash_attn_func(
                    q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                    v.transpose(1, 2).contiguous(), causal=True)
                return self.o_proj(y.reshape(B, T, -1))
            elif (dev_type == "npu" and q.dtype in (torch.float16, torch.bfloat16)
                  and self.use_flash_attn):
                # npu_fusion_attention 通过训练脚本的 --flash-attention 参数启用。
                # 它是异步算子：若该 CANN 环境没有对应 kernel（FlashAttentionScore
                # does not has any binary），错误只会延迟到同步点爆发，try/except
                # 无法捕获。因此首次调用先做一次小 shape 同步探测，失败则全局
                # 禁用 FA 并回退下方慢速 attention。
                if not self._npu_fa_probe(x.device, q.dtype):
                    pass  # fall through to slow attention below
                else:
                    import torch_npu
                    if (getattr(self, "_npu_mask", None) is None
                            or self._npu_mask.shape[-1] < T):
                        m = torch.triu(
                            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
                        self._npu_mask = m.view(1, 1, T, T)
                    atten_mask = self._npu_mask[..., :T, :T]
                    layout = self.fa_layout.strip().lower()
                    if layout == "bnsd":
                        # q/k/v 此时都是 [B, n_heads, T, head_dim]（k/v 已 repeat）
                        y, *_ = torch_npu.npu_fusion_attention(
                            q.contiguous(), k.contiguous(), v.contiguous(),
                            self.n_heads, "BNSD",
                            atten_mask=atten_mask,
                            scale=self._scale,
                            inner_precise=0)
                        y = y.transpose(1, 2).reshape(B, T, -1)
                    else:  # BSH: [B, T, n_heads*head_dim]
                        qh = q.transpose(1, 2).reshape(B, T, -1).contiguous()
                        kh = k.transpose(1, 2).reshape(B, T, -1).contiguous()
                        vh = v.transpose(1, 2).reshape(B, T, -1).contiguous()
                        y, *_ = torch_npu.npu_fusion_attention(
                            qh, kh, vh,                             self.n_heads, "BSH",
                            atten_mask=atten_mask,
                            scale=self._scale,
                            inner_precise=0)
                        y = y.reshape(B, T, -1)
                    return self.o_proj(y)
        except Exception as e:  # noqa: BLE001
            # 兜底：同步错误（如参数非法）也全局禁用 FA 并走慢速 attention
            if dev_type == "npu":
                Attention._npu_fa_ok = False
                if not Attention._npu_fa_warned:
                    Attention._npu_fa_warned = True
                    print(f"warning: npu_fusion_attention failed ({e!r}); "
                          f"disabling FlashAttention, falling back to slow attention "
                          f"(q.dtype={q.dtype})", flush=True)
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
        probs = F.softmax(logits.float(), dim=-1)
        top_probs, top_idx = torch.topk(probs, k=self.top_k, dim=-1)

        z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean() if use_aux_loss else torch.zeros((), device=x.device)

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
        self.shared_mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden) if cfg.mlp_hidden else None
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])
        self.n_shared_experts = cfg.n_shared_experts
        if cfg.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [SwiGLU(cfg.d_model, cfg.shared_expert_hidden) for _ in range(cfg.n_shared_experts)])
        else:
            self.shared_experts = nn.ModuleList()
        # fused per-expert weight matrices (W_gate/W_up/W_down), rebuilt after
        # every optimizer step instead of cat()-ing them each forward call.
        self._expert_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

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
        """
        H = self.experts[0].mlp.gate.out_features
        W_gate = torch.cat([e.mlp.gate.weight for e in self.experts], dim=0)  # [E*H, D]
        W_up = torch.cat([e.mlp.up.weight for e in self.experts], dim=0)      # [E*H, D]
        W_down = torch.cat([e.mlp.down.weight for e in self.experts], dim=1)  # [D, E*H]
        if self._expert_cache is None:
            self._expert_cache = (W_gate, W_up, W_down)
        else:
            for dst, src in zip(self._expert_cache, (W_gate, W_up, W_down)):
                dst.copy_(src)

    def _moe(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        out = self.router(flat)
        # Router 内部已算好 softmax/topk 并缓存到 RouterOutput，这里直接复用，
        # 避免每层重复 F.softmax + torch.topk（12 层 × checkpoint 重算）。
        top_probs = out.top_probs.to(x.dtype)
        top_idx = out.top_idx
        z_loss, aux_loss = out.z_loss, out.aux_loss
        # NPU-safe routing: fixed-shape compute only.  No nonzero / index_add_ /
        # data-dependent control flow (both hang CANN graph compilation).
        # All experts run on ALL tokens; the top-k weights gate their outputs.
        # Per-expert Linear weights are concatenated into single [E*H, D] /
        # [D, E*H] matrices so the whole expert layer is 3 fused matmuls
        # instead of 16 sequential SwiGLU calls (much higher NPU utilization).
        # The cat() is cached and only rebuilt after each optimizer step
        # (refresh_expert_cache), so we don't redo it every forward call.
        if self._expert_cache is None or self._expert_cache[0].device != flat.device:
            self.refresh_expert_cache()
        W_gate, W_up, W_down = self._expert_cache
        H = self.experts[0].mlp.gate.out_features
        S = flat.shape[0]
        g_all = F.linear(flat, W_gate)                       # [S, E*H]
        u_all = F.linear(flat, W_up)                         # [S, E*H]
        g = g_all.view(S, self.n_experts, H)
        u = u_all.view(S, self.n_experts, H)
        a = F.silu(g) * u                                    # [S, E, H]
        # top-k per-expert weight: [S, k, E] one-hot * top_probs -> [S, E]
        onehot = F.one_hot(top_idx, num_classes=self.n_experts).to(top_probs.dtype)
        w = (onehot * top_probs.unsqueeze(-1)).sum(dim=1)    # [S, E]
        a = a * w.unsqueeze(-1)                              # [S, E, H]
        y = F.linear(a.reshape(S, self.n_experts * H), W_down)  # [S, D]
        y = y.reshape(B, T, D)

        if self.n_shared_experts > 0:
            for se in self.shared_experts:
                y = y + se(x)

        return y, {"router_z_loss": z_loss, "router_aux_loss": aux_loss}


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

    def check_flash_attn(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Pre-flight synchronous FA availability probe (Ascend only).

        Call once after model.to(device) before training. Returns True if
        npu_fusion_attention is usable; on failure it is globally disabled
        inside the probe (falls back to slow attention) and this returns False.
        """
        if len(self.layers) == 0:
            return False
        return self.layers[0].attn._npu_fa_probe(device, dtype)

    def _post_load_rebuild(self, module, incompatible_keys) -> None:
        """Called after load_state_dict finishes (all submodules loaded)."""
        self.refresh_expert_caches()

    def _rope(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope_cache is None or self._rope_cache[0].shape[0] < seq_len:
            cos, sin = precompute_rope(self.cfg.head_dim, max(seq_len, self.cfg.max_seq_len),
                                       self.cfg.rope_theta, self.cfg.rope_scaling)
            self._rope_cache = (cos.to(device), sin.to(device))
        return self._rope_cache[0][:seq_len], self._rope_cache[1][:seq_len]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """idx: [B, T]. Returns (logits, extra_losses)."""
        B, T = idx.shape
        x = self.token_embedding(idx)
        cos, sin = self._rope(T, idx.device)
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
