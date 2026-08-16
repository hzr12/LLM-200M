"""P0 probe for 910 Pro A (MindSpore 2.2, fp16 path).

Run on the node:
    python probe_p0_910pa.py

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
            print(r.stdout[-3000:] if r.stdout else r.stderr[-2000:])
            return
        except Exception as e:  # noqa: BLE001
            print(f"PROBE npu-smi ({c}) unavailable: {e!r}")
    print("PROBE npu-smi: not found in PATH")


def probe_rmsnorm():
    if not hasattr(ops, "RmsNorm"):
        print("PROBE RmsNorm: ops.RmsNorm NOT AVAILABLE on this version", flush=True)
        return
    D = 768
    rms = ops.RmsNorm(epsilon=1e-6)
    for dtype, scale in ((ms.float32, 0.5), (ms.float16, 0.5)):
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
            print(f"PROBE RmsNorm OK dtype={dtype} out_shape={out.shape} max_abs_err={err:.3e}")
        except Exception as e:  # noqa: BLE001
            print(f"PROBE RmsNorm FAILED dtype={dtype}: {e!r}")
    # no-param call form (some versions need gamma as second arg only)
    try:
        out = rms(ms.Tensor(np.ones((1, 4, D), np.float32), ms.float32),
                  ms.Tensor(np.ones(D, np.float32), ms.float32))
        print(f"PROBE RmsNorm 2-arg call OK shape={out.shape if not isinstance(out, tuple) else out[0].shape}")
    except Exception as e:  # noqa: BLE001
        print(f"PROBE RmsNorm 2-arg call FAILED: {e!r}")


def elem_rope_ref(q: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                  sign: float) -> np.ndarray:
    """q [B, S, N, D]; cos/sin [S, D/2]. sign=+1 → cos/-sin, -1 → cos/+sin."""
    q = q.astype(np.float64)
    D = q.shape[-1]
    q1, q2 = q[..., :D // 2], q[..., D // 2:]
    c = cos[np.newaxis, :, np.newaxis, :]
    s = sign * sin[np.newaxis, :, np.newaxis, :]
    return np.concatenate((q1 * c - q2 * s, q1 * s + q2 * c), axis=-1)


def probe_apply_rotary():
    if not hasattr(ops, "ApplyRotaryPosEmb"):
        print("PROBE ApplyRotaryPosEmb: NOT AVAILABLE on this version", flush=True)
        return
    B, S, N, D = 1, 4, 2, 8
    q = ms.Tensor(np.random.randn(B, S, N, D).astype(np.float32) * 0.5, ms.float32)
    cos = ms.Tensor(np.cos(np.arange(S * (D // 2)).reshape(S, D // 2) * 0.1).astype(np.float32), ms.float32)
    sin = ms.Tensor(np.sin(np.arange(S * (D // 2)).reshape(S, D // 2) * 0.1).astype(np.float32), ms.float32)
    ref_p = elem_rope_ref(q.asnumpy(), cos.asnumpy(), sin.asnumpy(), +1.0)
    ref_m = elem_rope_ref(q.asnumpy(), cos.asnumpy(), sin.asnumpy(), -1.0)
    op = ops.ApplyRotaryPosEmb()
    tried = 0
    # convention 1: (x, freqs_cis) with freqs_cis = cos+sin pairs [S, D/2, 2]
    for freqs, tag in (
        (ms.Tensor(np.stack([cos.asnumpy(), sin.asnumpy()], axis=-1), ms.float32), "freqs_cis=stack(cos,sin)[S,D/2,2]"),
        (cos, "freqs_cis=cos only"),
    ):
        tried += 1
        try:
            out = op(q, freqs)
            if isinstance(out, tuple):
                out = out[0]
            e_p = float(np.abs(out.asnumpy() - ref_p).max())
            e_m = float(np.abs(out.asnumpy() - ref_m).max())
            print(f"PROBE ApplyRotaryPosEmb OK conv={tried} ({tag}) out_shape={out.shape} "
                  f"err_sign_cos-sin={e_p:.3e} err_sign_cos+sin={e_m:.3e}")
            return
        except Exception as e:  # noqa: BLE001
            print(f"PROBE ApplyRotaryPosEmb conv={tried} ({tag}) FAILED: {str(e)[:200]!r}")
    # convention 2: 6-arg (q, k, sin, cos, cos_valid, sin_valid) — shapes guessed
    k = ms.Tensor(np.random.randn(B, S, N, D).astype(np.float32) * 0.5, ms.float32)
    for cshape, tag in (
        ((B, S, N, D // 2), "sin/cos [B,S,N,D/2]"),
        ((S, D // 2), "sin/cos [S,D/2]"),
    ):
        tried += 1
        try:
            cv = np.ones(cshape, np.float32)
            sv = np.zeros(cshape, np.float32)
            out = op(q, k, sin.broadcast_to(cshape) if sin.shape != cshape else sin,
                     cos.broadcast_to(cshape) if cos.shape != cshape else cos,
                     ms.Tensor(cv, ms.float32), ms.Tensor(sv, ms.float32))
            if isinstance(out, tuple):
                qo = out[0]
            else:
                qo = out
            e_p = float(np.abs(qo.asnumpy() - ref_p).max())
            e_m = float(np.abs(qo.asnumpy() - ref_m).max())
            print(f"PROBE ApplyRotaryPosEmb OK conv={tried} ({tag}) out_shape={qo.shape} "
                  f"err_cos-sin={e_p:.3e} err_cos+sin={e_m:.3e}")
            return
        except Exception as e:  # noqa: BLE001
            print(f"PROBE ApplyRotaryPosEmb conv={tried} ({tag}) FAILED: {str(e)[:200]!r}")
    print("PROBE ApplyRotaryPosEmb: all conventions failed (see above)")


def probe_train_smoke():
    class Tiny(nn.Cell):
        def __init__(self, D=64, H=32):
            super().__init__()
            self.w = ms.Parameter(ms.Tensor(np.random.randn(D, H).astype(np.float32) * 0.02, ms.float32), name="w")
        def construct(self, x):
            return ops.matmul(x, self.w)

    cell = Tiny()
    x = ms.Tensor(np.random.randn(4, 8, 64).astype(np.float32) * 0.5, ms.float32)
    try:
        gfn = ms.value_and_grad(cell, grad_position=None, weights=cell.trainable_params())
        loss, grads = gfn(x)
        print(f"PROBE value_and_grad OK loss={float(loss):.4f} n_grads={len(grads)} "
              f"grad_shape={tuple(grads[0].shape)} grad_finite={bool(np.isfinite(grads[0].asnumpy()).all())}")
    except Exception as e:  # noqa: BLE001
        print(f"PROBE value_and_grad FAILED: {e!r}")
        return
    try:
        steps = 10
        lr_arr = ms.Tensor(np.linspace(1e-4, 1e-5, steps).astype(np.float32), ms.float32)
        optim = nn.AdamWeightDecay(cell.trainable_params(), learning_rate=lr_arr,
                                   weight_decay=0.1, eps=1e-8)
        w0 = cell.w.asnumpy().copy()
        for i in range(steps):
            loss, grads = gfn(x)
            optim(grads)
        dw = float(np.abs(cell.w.asnumpy() - w0).max())
        print(f"PROBE AdamWeightDecay OK final_loss={float(loss):.4f} max_dw={dw:.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"PROBE AdamWeightDecay FAILED: {e!r}")


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
                 "prompt_flash_attention", "clip_by_global_norm", "AdamWeightDecay",
                 "MicroBatchInterleaved", "DynamicLossScaleUpdateCell", "CrossEntropyLoss",
                 "value_and_grad", "TopK", "OneHot", "logsumexp", "multinomial",
                 "masked_fill", "repeat_interleave", "BatchMatMul"):
        print(f"PROBE hasattr ops/ms.{name} = {hasattr(ops, name) or hasattr(ms, name)}", flush=True)
    run_npu_smi()
    probe_rmsnorm()
    probe_apply_rotary()
    probe_train_smoke()
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()