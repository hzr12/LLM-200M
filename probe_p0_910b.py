"""P0 probe for 910B (MindSpore 2.7, bf16/fp16, FlashAttention).

Run on the node:
    python probe_p0_910b.py

Prints PROBE-marked lines. Paste the full output back.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

import mindspore as ms
from mindspore import context, nn, ops


def run_npu_smi():
    candidates = ["npu-smi", "/usr/local/Ascend/driver/tools/npu-smi"]
    for c in candidates:
        try:
            r = subprocess.run([c, "info"], capture_output=True, text=True,
                               timeout=30)
            print(f"PROBE npu-smi ({c}) rc={r.returncode}")
            print(r.stdout[-4000:] if r.stdout else r.stderr[-2000:])
            return
        except Exception as e:  # noqa: BLE001
            print(f"PROBE npu-smi ({c}) unavailable: {e!r}")
    print("PROBE npu-smi: not found in PATH")


# --------------------------------------------------------------------------
# FlashAttention probe
# --------------------------------------------------------------------------
def slow_attn_np(q, k, v, scale, causal):
    """q/k/v [B, N, S, D] fp64 numpy. Returns out [B, N, S, D]."""
    B, N, S, D = q.shape
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    if causal:
        mask = np.tril(np.ones((S, S), np.float64))[None, None]
        scores = scores * mask + (1.0 - mask) * (-1e30)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    return np.matmul(p, v)


def _fa_y(out):
    if isinstance(out, tuple):
        return out[0]
    return out


def probe_fa_forward():
    B, S, N, KV, D = 2, 64, 4, 2, 64
    scale = D ** -0.5
    for dtype, dtname in ((ms.float16, "fp16"), (ms.bfloat16, "bf16")):
        q = ms.Tensor(np.random.randn(B, S, N, D).astype(np.float32) * 0.5, dtype)
        k = ms.Tensor(np.random.randn(B, S, KV, D).astype(np.float32) * 0.5, dtype)
        v = ms.Tensor(np.random.randn(B, S, KV, D).astype(np.float32) * 0.5, dtype)
        for sparse in (2, 0):
            # BNSD layout (GQA: N != KV)
            try:
                out = ops.flash_attention_score(
                    q, k, v, N, keep_prob=1.0, scalar_value=scale,
                    pre_tokens=2147483647, next_tokens=0, inner_precise=0,
                    input_layout="BNSD", sparse_mode=sparse)
                y = _fa_y(out)
                # reference on repeated kv
                kr = k.repeat_interleave(N // KV, axis=1)
                vr = v.repeat_interleave(N // KV, axis=1)
                ref = slow_attn_np(q.asnumpy().astype(np.float64),
                                   kr.asnumpy().astype(np.float64),
                                   vr.asnumpy().astype(np.float64), scale,
                                   causal=(sparse == 2))
                err = float(np.abs(y.asnumpy().astype(np.float64) - ref).max())
                rerr = err / float(np.abs(ref).max() + 1e-8)
                print(f"PROBE FA BNSD GQA OK {dtname} sparse={sparse} out_shape={tuple(y.shape)} "
                      f"ret_type={type(out).__name__} max_abs_err={err:.4f} rel_err={rerr:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"PROBE FA BNSD GQA FAILED {dtname} sparse={sparse}: {str(e)[:250]!r}", flush=True)
            # BSH layout (GQA via narrower k/v)
            qb = q.swapaxes(1, 2).reshape(B, S, N * D)
            kb = k.swapaxes(1, 2).reshape(B, S, KV * D)
            vb = v.swapaxes(1, 2).reshape(B, S, KV * D)
            try:
                out = ops.flash_attention_score(
                    qb, kb, vb, N, keep_prob=1.0, scalar_value=scale,
                    pre_tokens=2147483647, next_tokens=0, inner_precise=0,
                    input_layout="BSH", sparse_mode=sparse)
                y = _fa_y(out)
                yn = y.reshape(B, S, N, D).swapaxes(1, 2)
                kr = k.repeat_interleave(N // KV, axis=1)
                vr = v.repeat_interleave(N // KV, axis=1)
                ref = slow_attn_np(q.asnumpy().astype(np.float64),
                                   kr.asnumpy().astype(np.float64),
                                   vr.asnumpy().astype(np.float64), scale,
                                   causal=(sparse == 2))
                err = float(np.abs(yn.asnumpy().astype(np.float64) - ref).max())
                rerr = err / float(np.abs(ref).max() + 1e-8)
                print(f"PROBE FA BSH GQA OK {dtname} sparse={sparse} out_shape={tuple(y.shape)} "
                      f"ret_type={type(out).__name__} max_abs_err={err:.4f} rel_err={rerr:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"PROBE FA BSH GQA FAILED {dtname} sparse={sparse}: {str(e)[:250]!r}", flush=True)


def probe_fa_grad():
    B, S, N, KV, D = 2, 64, 4, 2, 64
    scale = D ** -0.5
    for dtype, dtname in ((ms.float16, "fp16"), (ms.bfloat16, "bf16")):
        q = ms.Tensor(np.random.randn(B, S, N, D).astype(np.float32) * 0.5, dtype)
        k = ms.Tensor(np.random.randn(B, S, KV, D).astype(np.float32) * 0.5, dtype)
        v = ms.Tensor(np.random.randn(B, S, KV, D).astype(np.float32) * 0.5, dtype)
        qr = ms.Tensor(np.random.randn(B, S, N, D).astype(np.float32) * 0.5, dtype)
        kr = ms.Tensor(np.random.randn(B, S, KV, D).astype(np.float32) * 0.5, dtype)
        vr = ms.Tensor(np.random.randn(B, S, KV, D).astype(np.float32) * 0.5, dtype)

        class FACell(nn.Cell):
            def __init__(self, sparse):
                super().__init__()
                self.sparse = sparse
            def construct(self, qq, kk, vv):
                out = ops.flash_attention_score(
                    qq, kk, vv, N, keep_prob=1.0, scalar_value=scale,
                    pre_tokens=2147483647, next_tokens=0, inner_precise=0,
                    input_layout="BNSD", sparse_mode=self.sparse)
                return _fa_y(out).sum()

        class SlowCell(nn.Cell):
            def __init__(self, causal):
                super().__init__()
                self.causal = causal
            def construct(self, qq, kk, vv):
                # repeat kv to N heads in fp32 for the reference
                kkr = kk.repeat_interleave(N // KV, axis=1).float()
                vvr = vv.repeat_interleave(N // KV, axis=1).float()
                scores = ops.matmul(qq.float(), kkr.swapaxes(-2, -1)) * scale
                if self.causal:
                    bias = np.tril(np.ones((S, S), np.float32))[None, None]
                    bias = ms.Tensor(bias, ms.float32)
                    scores = scores + (bias - 1.0) * -1e4
                p = ops.softmax(scores, axis=-1)
                return (ops.matmul(p, vvr)).sum()

        for sparse in (2, 0):
            try:
                fa_cell = FACell(sparse)
                gfn = ms.value_and_grad(fa_cell, grad_position=(0, 1, 2))
                _, (dq, dk, dv) = gfn(q, k, v)
                slow_cell = SlowCell(causal=(sparse == 2))
                sfn = ms.value_and_grad(slow_cell, grad_position=(0, 1, 2))
                _, (sq, sk, sv) = sfn(qr, kr, vr)
                errs = []
                for g, r in ((dq, sq), (dk, sk), (dv, sv)):
                    gf = g.asnumpy().astype(np.float64)
                    rf = r.asnumpy().astype(np.float64)
                    errs.append(float(np.abs(gf - rf).max()) / (float(np.abs(rf).max()) + 1e-8))
                finite = bool(np.isfinite(dq.asnumpy()).all()
                              and np.isfinite(dk.asnumpy()).all()
                              and np.isfinite(dv.asnumpy()).all())
                print(f"PROBE FA GRAD OK {dtname} sparse={sparse} finite={finite} "
                      f"rel_err q/k/v={[f'{e:.3f}' for e in errs]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"PROBE FA GRAD FAILED {dtname} sparse={sparse}: {str(e)[:250]!r}", flush=True)


# --------------------------------------------------------------------------
# RmsNorm / RoPE / swiglu probes (same as 910PA)
# --------------------------------------------------------------------------
def probe_rmsnorm():
    D = 768
    rms = ops.RmsNorm(epsilon=1e-6)
    for dtype, scale in ((ms.float32, 0.5), (ms.float16, 0.5), (ms.bfloat16, 0.5)):
        x = ms.Tensor((np.random.randn(2, 8, D) * scale).astype(np.float32), dtype)
        gamma = ms.Tensor(np.ones(D, np.float32), dtype)
        try:
            out = rms(x, gamma)
            if isinstance(out, tuple):
                out = out[0]
            xf = x.asnumpy().astype(np.float64)
            gf = gamma.asnumpy().astype(np.float64)
            r = np.sqrt((xf ** 2).mean(-1, keepdims=True) + 1e-6)
            ref = (xf / r) * gf
            err = float(np.abs(out.asnumpy().astype(np.float64) - ref).max())
            print(f"PROBE RmsNorm OK dtype={dtype} max_abs_err={err:.3e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"PROBE RmsNorm FAILED dtype={dtype}: {e!r}", flush=True)


def elem_rope_ref(q: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    D = q.shape[-1]
    q1, q2 = q[..., :D // 2], q[..., D // 2:]
    c = cos[np.newaxis, :, np.newaxis, :]
    s = sin[np.newaxis, :, np.newaxis, :]
    return np.concatenate((q1 * c - q2 * s, q1 * s + q2 * c), axis=-1)


def probe_apply_rotary():
    B, S, N, D = 1, 4, 2, 8
    q = ms.Tensor(np.random.randn(B, S, N, D).astype(np.float32) * 0.5, ms.float32)
    cos = ms.Tensor(np.cos(np.arange(S * (D // 2)).reshape(S, D // 2) * 0.1).astype(np.float32), ms.float32)
    sin = ms.Tensor(np.sin(np.arange(S * (D // 2)).reshape(S, D // 2) * 0.1).astype(np.float32), ms.float32)
    ref = elem_rope_ref(q.asnumpy(), cos.asnumpy(), sin.asnumpy())
    op = ops.ApplyRotaryPosEmb()
    for freqs, tag in (
        (ms.Tensor(np.stack([cos.asnumpy(), sin.asnumpy()], axis=-1), ms.float32), "freqs_cis=stack(cos,sin)"),
        (cos, "freqs_cis=cos only"),
    ):
        try:
            out = op(q, freqs)
            if isinstance(out, tuple):
                out = out[0]
            e = float(np.abs(out.asnumpy().astype(np.float64) - ref).max())
            print(f"PROBE ApplyRotaryPosEmb OK ({tag}) out_shape={tuple(out.shape)} max_abs_err={e:.3e}", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            print(f"PROBE ApplyRotaryPosEmb ({tag}) FAILED: {str(e)[:200]!r}", flush=True)
    print("PROBE ApplyRotaryPosEmb: no convention matched", flush=True)


def probe_swiglu():
    H = 320
    x = ms.Tensor(np.random.randn(2, 8, 2 * H).astype(np.float32) * 0.5, ms.float16)
    try:
        y = ops.swiglu(x)
        xn = x.asnumpy().astype(np.float64)
        g, u = xn[..., :H], xn[..., H:]
        ref = (g / (1.0 + np.exp(-g))) * u
        err = float(np.abs(y.asnumpy().astype(np.float64) - ref).max())
        print(f"PROBE swiglu OK out_shape={tuple(y.shape)} max_abs_err={err:.3e}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PROBE swiglu FAILED: {e!r}", flush=True)


def main():
    print(f"PROBE mindspore version: {ms.__version__}", flush=True)
    print(f"PROBE python: {sys.version}", flush=True)
    try:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
        print("PROBE set_context(PYNATIVE, Ascend) OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PROBE set_context FAILED: {e!r}", flush=True)
        return
    for k in ("device_name", "device_id", "device_target", "max_device_memory"):
        try:
            print(f"PROBE ctx[{k}] = {ms.get_context(k)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"PROBE ctx[{k}] FAILED: {e!r}", flush=True)
    try:
        print(f"PROBE hal.get_device_name = {ms.hal.get_device_name()}", flush=True)
        print(f"PROBE hal.get_device_capability = {ms.hal.get_device_capability()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PROBE hal FAILED: {e!r}", flush=True)
    for name in ("set_device",):
        print(f"PROBE hasattr ms.{name} = {hasattr(ms, name)}", flush=True)
    for name in ("RmsNorm", "ApplyRotaryPosEmb", "GroupTopk", "MultiheadAttention",
                 "swiglu", "SwigLU", "flash_attention_score", "incre_flash_attention",
                 "prompt_flash_attention", "fused_infer_attention_score",
                 "moe_token_permute", "moe_token_unpermute", "moe_distribute_dispatch",
                 "moe_distribute_combine", "moe_init_routing_v2", "clip_by_global_norm",
                 "AdamWeightDecay", "MicroBatchInterleaved", "DynamicLossScaleUpdateCell",
                 "CrossEntropyLoss", "value_and_grad", "TopK", "OneHot", "logsumexp",
                 "multinomial", "masked_fill", "repeat_interleave", "BatchMatMul"):
        print(f"PROBE hasattr ops/ms.{name} = {hasattr(ops, name) or hasattr(ms, name)}", flush=True)
    run_npu_smi()
    probe_fa_forward()
    probe_fa_grad()
    probe_rmsnorm()
    probe_apply_rotary()
    probe_swiglu()
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()