"""MindSpore Mixture-of-Experts Transformer (RMSNorm + RoPE + GQA + SwiGLU, top-k routing).

This is the MindSpore-native implementation used for Ascend training
(910B / 910 Pro A). The torch implementation is preserved in moe_torch.py.

Design rules (differ from the torch version on purpose):
  - All weights are fp32 Parameters stored in TRANSPOSED [in, out] layout, so a
    forward pass needs no per-call transpose. Per-call cast is explicit via
    _ld(x, w) = matmul(x, w.astype(x.dtype)); there is no autocast context.
  - No nn.Embedding / nn.Linear / nn.Dense: plain Parameters + ops.matmul /
    ops.gather. lm_head shares emb_w (weight tying).
  - MoE runs the dense-mask architecture with FUSED expert weights W_gu_T
    [D, 2*E*H] and W_down_T [E*H, D] as Parameters directly (no per-step cat,
    no expert cache machinery, no sort/scatter ops).
  - z/aux losses are ALWAYS computed (no self.training branch): eval and train
    share one graph, avoiding graph recompile. Val metrics therefore include
    the router aux terms (documented difference from the torch version).
  - ops.flash_attention_score is used on 910B when the startup probe succeeds;
    otherwise a slow attention path with a cached causal bias and fp16 softmax.
  - RMSNorm may use the fused ops.RmsNorm when the startup probe passes.
  - Startup probes (model.probe_fused_ops) must run in PYNATIVE mode BEFORE
    switching to GRAPH mode for training.
"""
from __future__ import annotations

import math
import os

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from .config import Config


def _ld(x: ms.Tensor, w: ms.Tensor) -> ms.Tensor:
    """Linear (no bias) with explicit weight cast to the activation dtype.

    Weights stay fp32 Parameters (fp32 gradient accumulation); the cast is
    explicit and per-call so the forward and any recomputation share one path.
    """
    return ops.matmul(x, w.astype(x.dtype))


def precompute_rope(head_dim: int, seq_len: int, theta: float,
                    scaling: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (cos, sin) as numpy [seq_len, head_dim/2] (fp32)."""
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    if scaling is not None:  # NTK-aware scaling (llama3 style)
        factor = scaling.get("factor", 1.0)
        low_freq_factor = scaling.get("low_freq_factor", 1.0)
        high_freq_factor = scaling.get("high_freq_factor", 4.0)
        old_ctx_len = 4096
        low_freq_wavelen = old_ctx_len / low_freq_factor
        high_freq_wavelen = old_ctx_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth = np.maximum(
            np.zeros_like(wavelen),
            (high_freq_wavelen - wavelen) / (high_freq_wavelen - low_freq_wavelen))
        inv_freq = np.where(wavelen < high_freq_wavelen, inv_freq,
                            inv_freq * (1 + smooth * (factor - 1)))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs), np.sin(freqs)


def apply_rope(q: ms.Tensor, k: ms.Tensor, cos: ms.Tensor,
               sin: ms.Tensor) -> tuple[ms.Tensor, ms.Tensor]:
    """q,k: [B, heads, T, head_dim]; cos,sin: [T, head_dim/2] (same dtype)."""
    cos = cos[None, None]
    sin = sin[None, None]
    d2 = q.shape[-1] // 2
    q1, q2 = q[..., :d2], q[..., d2:]
    q = ops.concat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), axis=-1)
    k1, k2 = k[..., :d2], k[..., d2:]
    k = ops.concat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), axis=-1)
    return q, k


class RMSNorm(nn.Cell):
    """RMSNorm with fixed param name `weight`; optional fused ops.RmsNorm.

    _fused_ok is decided once by the startup probe (model.probe_fused_ops) and
    cached at class level; the probe runs before graph-mode compilation, so the
    construct keeps a single static path.
    """
    _fused_ok: bool | None = None

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = ms.Parameter(ms.Tensor(np.ones(dim, np.float32), ms.float32),
                                   name="weight")
        self.eps = eps
        # fused ops.RmsNorm exists on MS 2.7 (910B) but NOT on 2.2 (910 Pro A)
        self._fused = ops.RmsNorm(epsilon=eps) if hasattr(ops, "RmsNorm") else None

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        w = self.weight.to(x.dtype)
        if self._fused is not None and RMSNorm._fused_ok \
                and x.dtype in (ms.float16, ms.bfloat16):
            out = self._fused(x, w)
            return out[0] if isinstance(out, tuple) else out
        rms = ops.rsqrt(ops.pow(x.float(), 2.0).mean(axis=-1, keep_dims=True)
                        .add(self.eps))
        return x * rms.to(x.dtype) * w


class Attention(nn.Cell):
    """GQA attention. fused FA (BNSD, sparse_mode) when the probe passed.

    _fa_ok is decided once by the startup probe (forward + grad with the real
    training shape) and cached at class level. The FA path repeats kv to
    n_heads first (uniform-head FA: the CANN GQA backward path is broken on
    this stack — aclnnReduceSum EZ1001). sparse_mode from
    LLM_SNN_FA_SPARSE_MODE (default 0 = built-in causal).
    """
    _fa_ok: bool | None = None
    _fa_warned: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.repeats = self.n_heads // self.n_kv_heads
        self._scale = self.head_dim ** -0.5
        self._sparse_mode = int(os.environ.get("LLM_SNN_FA_SPARSE_MODE", "0"))
        d = cfg.d_model
        qkv_out = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.qkv_w_T = ms.Parameter(
            ms.Tensor(np.random.randn(d, qkv_out).astype(np.float32) * 0.02, ms.float32),
            name="qkv_w_T")
        self.o_w_T = ms.Parameter(
            ms.Tensor(np.random.randn(d, d).astype(np.float32) * 0.02, ms.float32),
            name="o_w_T")
        # cached additive causal bias (upper triangle = -inf), built outside the
        # graph by ensure_bias; construct only slices with static indices.
        self._causal_bias: ms.Tensor | None = None
        self._bias_T = 0
        self._bias_dtype = None

    def ensure_bias(self, T: int, dtype) -> None:
        """(Re)build the causal bias for seq length T and activation dtype."""
        if (self._causal_bias is None or self._bias_T < T
                or self._bias_dtype != dtype):
            b = np.triu(np.full((T, T), -np.inf, np.float32), k=1)
            self._causal_bias = ms.Tensor(b, dtype)[None, None]
            self._bias_T = T
            self._bias_dtype = dtype

    def construct(self, x: ms.Tensor, cos: ms.Tensor, sin: ms.Tensor) -> ms.Tensor:
        B, T = x.shape[0], x.shape[1]
        qkv = _ld(x, self.qkv_w_T)
        dq = self.n_heads * self.head_dim
        dkv = self.n_kv_heads * self.head_dim
        q = qkv[..., :dq]
        k = qkv[..., dq: dq + dkv]
        v = qkv[..., dq + dkv:]
        q = q.reshape(B, T, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).swapaxes(1, 2)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).swapaxes(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        use_fa = (Attention._fa_ok is True
                  and x.dtype in (ms.float16, ms.bfloat16))
        if use_fa:
            if self.repeats > 1:
                k = k.repeat_interleave(self.repeats, dim=1)
                v = v.repeat_interleave(self.repeats, dim=1)
            out = ops.flash_attention_score(
                q, k, v, self.n_heads,
                keep_prob=1.0, scalar_value=self._scale,
                pre_tokens=2147483647, next_tokens=0, inner_precise=0,
                input_layout="BNSD", sparse_mode=self._sparse_mode)
            y = out[0] if isinstance(out, tuple) else out
        else:
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)
            att = ops.matmul(q * self._scale, k.swapaxes(-2, -1))
            att = att + self._causal_bias[..., :T, :T]
            p = ops.softmax(att, axis=-1)
            y = ops.matmul(p, v)
        return _ld(y.swapaxes(1, 2).reshape(B, T, -1), self.o_w_T)


class Router(nn.Cell):
    """Learns a per-token score over experts. Shared across layers.

    z_loss = mean(logsumexp(logits)^2); aux_loss = E * sum(freq * load).
    Both are always computed (single graph for train/eval).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router_w = ms.Parameter(
            ms.Tensor(np.random.randn(cfg.d_model, cfg.n_experts).astype(np.float32) * 0.02,
                      ms.float32), name="router_w")

    def construct(self, x: ms.Tensor) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor,
                                               ms.Tensor, ms.Tensor]:
        """x: [S, D]. Returns (top_probs, top_idx, z_loss, aux_loss) with
        top_probs/top_idx in fp32 / int32."""
        logits = _ld(x, self.router_w)          # [S, E] act dtype
        flogits = logits.float()
        probs = ops.softmax(flogits, axis=-1)
        top_probs, top_idx = ops.topk(probs, self.top_k, -1)
        z_loss = ops.pow(ops.logsumexp(flogits, axis=-1), 2.0).mean()
        S = probs.shape[0]
        probs_sum = probs.sum(axis=0)           # [E]
        onehot = ops.one_hot(top_idx, self.n_experts, 1.0, 0.0, axis=-1)
        expert_load = onehot.sum(axis=0).sum(axis=0)   # [E]
        f = probs_sum / S
        p = expert_load / S
        aux_loss = self.n_experts * (f * p).sum()
        return top_probs, top_idx, z_loss, aux_loss


class MoEBlock(nn.Cell):
    """Attention + MoE (dense-mask, fused expert weights)."""
    _swiglu_ok: bool | None = None  # set by probe_fused_ops (910B has ops.swiglu)

    def __init__(self, cfg: Config, layer_id: int, router: Router):
        super().__init__()
        self.layer_id = layer_id
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.router = router
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        d = cfg.d_model
        EH = cfg.n_experts * cfg.expert_hidden
        self.W_gu_T = ms.Parameter(
            ms.Tensor(np.random.randn(d, 2 * EH).astype(np.float32) * 0.02, ms.float32),
            name="W_gu_T")
        self.W_down_T = ms.Parameter(
            ms.Tensor(np.random.randn(EH, d).astype(np.float32) * 0.02, ms.float32),
            name="W_down_T")

    def construct(self, x: ms.Tensor, cos: ms.Tensor, sin: ms.Tensor) -> tuple:
        d = x.shape[-1]
        h = x + self.attn(self.norm1(x), cos, sin)
        n = self.norm2(h)
        B, T = x.shape[0], x.shape[1]
        flat = n.reshape(-1, d)
        S = flat.shape[0]
        top_probs, top_idx, z_loss, aux_loss = self.router(flat)
        EH = self.W_gu_T.shape[-1] // 2
        E, H = self.n_experts, EH // self.n_experts
        gu = _ld(flat, self.W_gu_T)            # [S, 2*EH]
        if MoEBlock._swiglu_ok is True:
            a = ops.swiglu(gu).reshape(S, E, H)   # fused silu(gate)*up (910B)
        else:
            g_all, u_all = gu[..., :EH], gu[..., EH:]
            g = g_all.reshape(S, E, H)
            u = u_all.reshape(S, E, H)
            a = g * ops.sigmoid(g) * u            # silu(g) * u; sigmoid avoids ops.silu (absent on 2.2)
        w = (ops.one_hot(top_idx, E, 1.0, 0.0, axis=-1).to(x.dtype)
             * top_probs.to(x.dtype).unsqueeze(-1)).sum(axis=1)  # [S, E]
        a = a * w.unsqueeze(-1)
        y = _ld(a.reshape(S, E * H), self.W_down_T).reshape(B, T, d)
        return h + y, z_loss, aux_loss


class MoETransformer(nn.Cell):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # activation dtype for embedding gather; set by train/chat before use
        # (default fp16; fp32 for CPU parity runs).
        self.act_dtype = ms.float16
        # tied embedding/lm_head, init matches the torch version: the shared
        # tensor is re-initialized with the scaled output-layer std.
        self.emb_w = ms.Parameter(
            ms.Tensor(np.random.randn(cfg.vocab_size, d).astype(np.float32)
                      * (0.02 / math.sqrt(2 * cfg.n_layers)), ms.float32),
            name="emb_w")
        self.router = Router(cfg)
        self.layers = nn.CellList([MoEBlock(cfg, i, self.router)
                                   for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self._cos: ms.Tensor | None = None
        self._sin: ms.Tensor | None = None
        self._rope_T = 0

    # -- rope tables & causal bias are prepared host-side before the graph ---
    def ensure_rope(self, T: int, dtype) -> None:
        if self._cos is None or self._cos.dtype != dtype or self._rope_T < T:
            max_T = max(T, self.cfg.max_seq_len)
            cos, sin = precompute_rope(self.cfg.head_dim, max_T,
                                       self.cfg.rope_theta, self.cfg.rope_scaling)
            self._cos = ms.Tensor(cos, dtype)
            self._sin = ms.Tensor(sin, dtype)
            self._rope_T = max_T

    def prepare_rope_bias(self, T: int, dtype) -> None:
        """Call once with the max sequence length before graph compilation."""
        self.ensure_rope(T, dtype)
        for layer in self.layers:
            layer.attn.ensure_bias(T, dtype)

    # ----------------------------------------------------------------------
    def probe_fused_ops(self, batch: int, seq_len: int, dtype) -> None:
        """Startup probes (PYNATIVE mode, before switching to GRAPH).

        - RMSNorm fused ops.RmsNorm: fwd + grad parity vs elementwise.
        - SwiGLU ops.swiglu: fwd parity vs silu(g)*u (910B only).
        - FlashAttention ops.flash_attention_score: fwd + grad on the real
          training shape, using exactly the model path (kv repeated to
          n_heads, sparse_mode from env, default 0).
        Sets RMSNorm._fused_ok / MoEBlock._swiglu_ok / Attention._fa_ok.
        """
        B, T, D = batch, seq_len, self.cfg.d_model
        print(f"probe: fused ops with shape B={B} T={T} dtype={dtype}", flush=True)
        # --- RmsNorm ---
        if not hasattr(ops, "RmsNorm"):
            RMSNorm._fused_ok = False
            print("probe: ops.RmsNorm unavailable -> elementwise RMSNorm", flush=True)
        else:
            try:
                x = ms.Tensor(np.random.randn(B, T, D).astype(np.float32) * 0.5, dtype)

                class RmFused(nn.Cell):
                    def __init__(self, g):
                        super().__init__()
                        self.g = g
                        self.rms = ops.RmsNorm(epsilon=1e-6)
                    def construct(self, xx):
                        out = self.rms(xx, self.g.to(xx.dtype))
                        return (out[0] if isinstance(out, tuple) else out).sum()

                class RmElem(nn.Cell):
                    def __init__(self, g):
                        super().__init__()
                        self.g = g
                    def construct(self, xx):
                        rms = ops.rsqrt(ops.pow(xx.float(), 2.0).mean(axis=-1, keep_dims=True).add(1e-6))
                        return (xx * rms.to(xx.dtype) * self.g.to(xx.dtype)).sum()

                g1 = ms.Parameter(ms.Tensor(np.ones(D, np.float32), ms.float32), name="g1")
                g2 = ms.Parameter(ms.Tensor(np.ones(D, np.float32), ms.float32), name="g2")
                l1, (dg1,) = ms.value_and_grad(RmFused(g1), grad_position=None,
                                               weights=[g1])(x)
                l2, (dg2,) = ms.value_and_grad(RmElem(g2), grad_position=None,
                                               weights=[g2])(x)
                err = max(float(abs(l1.asnumpy() - l2.asnumpy())) / max(1.0, float(abs(l2.asnumpy()))),
                          float(np.abs(dg1.asnumpy() - dg2.asnumpy()).max())
                          / (float(np.abs(dg2.asnumpy()).max()) + 1e-8))
                RMSNorm._fused_ok = err < 1e-2
                print(f"probe: RmsNorm fused rel_err={err:.4f} -> _fused_ok={RMSNorm._fused_ok}", flush=True)
            except Exception as e:  # noqa: BLE001
                RMSNorm._fused_ok = False
                print(f"probe: RmsNorm fused FAILED ({e!r}) -> _fused_ok=False", flush=True)
        # --- SwiGLU (ops.swiglu: silu(x[..., :H]) * x[..., H:]) ---
        if not hasattr(ops, "swiglu"):
            MoEBlock._swiglu_ok = False
            print("probe: ops.swiglu unavailable -> elementwise silu", flush=True)
        else:
            try:
                EH = self.cfg.n_experts * self.cfg.expert_hidden
                gu = ms.Tensor(np.random.randn(64, 2 * EH).astype(np.float32) * 0.5, dtype)
                class SwiProbe(nn.Cell):
                    def construct(self, xx):
                        return ops.swiglu(xx).sum()
                class SwiElem(nn.Cell):
                    def construct(self, xx):
                        g, u = xx[..., :EH], xx[..., EH:]
                        return (g * ops.sigmoid(g) * u).sum()
                l1 = SwiProbe()(gu)
                l2 = SwiElem()(gu)
                err = float(abs(l1.asnumpy() - l2.asnumpy())) / max(1.0, float(abs(l2.asnumpy())))
                MoEBlock._swiglu_ok = err < 1e-2
                print(f"probe: swiglu rel_err={err:.4f} -> _swiglu_ok={MoEBlock._swiglu_ok}", flush=True)
            except Exception as e:  # noqa: BLE001
                MoEBlock._swiglu_ok = False
                print(f"probe: swiglu FAILED ({e!r}) -> _swiglu_ok=False", flush=True)
        # --- FlashAttention ---
        if not (hasattr(ops, "flash_attention_score")
                and dtype in (ms.float16, ms.bfloat16)):
            Attention._fa_ok = False
            print("probe: flash_attention_score unavailable -> slow attention", flush=True)
            return
        N, KV, HD = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim
        repeats = N // KV
        q = ms.Tensor(np.random.randn(B, N, T, HD).astype(np.float32) * 0.5, dtype)
        k = ms.Tensor(np.random.randn(B, KV, T, HD).astype(np.float32) * 0.5, dtype)
        v = ms.Tensor(np.random.randn(B, KV, T, HD).astype(np.float32) * 0.5, dtype)
        scale = HD ** -0.5

        class FAProbe(nn.Cell):
            def __init__(self, smode, rpts):
                super().__init__()
                self.smode = smode
                self.rpts = rpts
            def construct(self, qq, kk, vv):
                if self.rpts > 1:
                    kk = kk.repeat_interleave(self.rpts, dim=1)
                    vv = vv.repeat_interleave(self.rpts, dim=1)
                out = ops.flash_attention_score(
                    qq, kk, vv, N, keep_prob=1.0, scalar_value=scale,
                    pre_tokens=2147483647, next_tokens=0, inner_precise=0,
                    input_layout="BNSD", sparse_mode=self.smode)
                y = out[0] if isinstance(out, tuple) else out
                return y.sum()

        smodes = [int(os.environ.get("LLM_SNN_FA_SPARSE_MODE", "0"))]
        for s in smodes + [m for m in (0,) if m not in smodes]:
            try:
                gfn = ms.value_and_grad(FAProbe(s, repeats), grad_position=(0, 1, 2))
                loss, (dq, dk, dv) = gfn(q, k, v)
                finite = (np.isfinite(loss.asnumpy()).all()
                          and np.isfinite(dq.asnumpy()).all()
                          and np.isfinite(dk.asnumpy()).all()
                          and np.isfinite(dv.asnumpy()).all())
                if finite:
                    Attention._fa_ok = True
                    for layer in self.layers:
                        layer.attn._sparse_mode = s
                    print(f"probe: FA OK sparse_mode={s} (kv repeated) -> _fa_ok=True", flush=True)
                    return
                print(f"probe: FA sparse_mode={s} non-finite grads, trying next", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"probe: FA sparse_mode={s} FAILED ({str(e)[:200]!r})", flush=True)
        Attention._fa_ok = False
        if not Attention._fa_warned:
            Attention._fa_warned = True
            print("probe: FA unusable on this platform -> slow attention", flush=True)

    # ----------------------------------------------------------------------
    def construct(self, idx: ms.Tensor, targets: ms.Tensor | None = None):
        """idx: [B, T] int32. Returns (logits, z_loss, aux_loss) — loss itself
        is computed by LossCell in train.py (graph-stable single path)."""
        B, T = idx.shape[0], idx.shape[1]
        x = ops.gather(self.emb_w, idx, 0).to(self.act_dtype)
        cos = self._cos[:T]
        sin = self._sin[:T]
        z = ops.zeros((), ms.float32)
        a = ops.zeros((), ms.float32)
        for layer in self.layers:
            x, lz, la = layer(x, cos, sin)
            z = z + lz
            a = a + la
        n = len(self.layers)
        z = z / n
        a = a / n
        x = self.norm(x)
        logits = _ld(x, self.emb_w.swapaxes(0, 1))  # tied lm_head
        return logits, z, a

    def generate(self, idx: ms.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 40, eos_id: int | None = None) -> ms.Tensor:
        """Autoregressive sampling. Runs in PYNATIVE mode (chat / eval)."""
        try:
            from mindspore import context as _ctx
            on_ascend = _ctx.get_context("device_target") == "Ascend"
        except Exception:  # noqa: BLE001
            on_ascend = False
        self.act_dtype = ms.float16 if on_ascend else ms.float32
        self.prepare_rope_bias(self.cfg.max_seq_len, self.act_dtype)
        for _ in range(max_new_tokens):
            if idx.shape[1] > self.cfg.max_seq_len:
                idx = idx[:, -self.cfg.max_seq_len:]
            logits, _, _ = self(idx)
            lf = logits[:, -1, :].float() / temperature
            if top_k > 0:
                v, _ = ops.topk(lf, min(top_k, lf.shape[-1]), -1)
                lf = ops.masked_fill(lf, lf < v[:, -1:], float("-inf"))
            probs = ops.softmax(lf, axis=-1)
            nxt = self._sample(probs)
            idx = ops.concat((idx, nxt), axis=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx

    @staticmethod
    def _sample(probs: ms.Tensor) -> ms.Tensor:
        try:
            return ops.multinomial(probs, 1)
        except Exception:  # noqa: BLE001
            p = probs.asnumpy()
            nxt = np.array([np.random.choice(p.shape[-1], p=p[i]) for i in range(p.shape[0])],
                           np.int32)
            return ms.Tensor(nxt, ms.int32).reshape(-1, 1)
