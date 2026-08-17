"""Numerical parity gates for the MindSpore port.

Run on the node (mindspore installed). Generates its own fixture if absent:

    python verify_ms_parity.py                     # CPU fp32 gate (1e-4)
    python verify_ms_parity.py --device Ascend --dtype fp16   # device fp16 (1e-2)
    python verify_ms_parity.py --device Ascend --dtype bf16   # device bf16 (1e-2)

Gates:
  G1  torch fixture logits vs numpy fp32 reference        (rel err < 1e-5)
  G2  MS CPU fp32 logits/z/aux vs numpy reference         (rel err < 1e-4)
  G3  numerical gradcheck on sampled params (CPU fp32)    (rel err < 5e-3)
  G4  device fp16/bf16 logits vs numpy reference          (rel err < 1e-2)
Device runs also exercise the fused paths (FA / fused RmsNorm) once
probe_fused_ops has marked them usable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import mindspore as ms
from mindspore import context, nn, ops


# --------------------------------------------------------------------------
# numpy fp32 reference (mirrors model/moe.py exactly)
# --------------------------------------------------------------------------
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def numpy_forward(cfg: dict, P: dict, ids: np.ndarray):
    """Return (logits fp32 [B,T,V], z_loss, aux_loss) — pure numpy fp32/64."""
    from model.moe import precompute_rope
    B, T = ids.shape
    d = cfg["d_model"]
    N = cfg["n_heads"]
    KV = cfg["n_kv_heads"]
    HD = cfg["head_dim"]
    E = cfg["n_experts"]
    H = cfg["expert_hidden"]
    K = cfg["top_k"]
    eps = cfg["norm_eps"]
    scale = HD ** -0.5
    cos, sin = precompute_rope(HD, max(T, cfg.get("max_seq_len", T)),
                               cfg["rope_theta"], cfg.get("rope_scaling"))
    cos, sin = cos[:T], sin[:T]

    def rms(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        xf = x.astype(np.float64)
        r = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
        return (xf / r * w.astype(np.float64)).astype(np.float32)

    def rope(x: np.ndarray) -> np.ndarray:  # [B,H,T,HD]
        d2 = HD // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        c = cos[None, None]
        s = sin[None, None]
        return np.concatenate((x1 * c - x2 * s, x1 * s + x2 * c), axis=-1).astype(np.float32)

    x = P["emb_w"][ids]                       # [B,T,d] fp32
    z = 0.0
    aux = 0.0
    for i in range(cfg["n_layers"]):
        p = f"layers.{i}."
        # attention
        xqkv = rms(x, P[p + "norm1.weight"])
        qkv = xqkv @ P[p + "attn.qkv_w_T"]    # [B,T,(N+2KV)*HD]
        dq, dkv = N * HD, KV * HD
        q = qkv[..., :dq].reshape(B, T, N, HD).transpose(0, 2, 1, 3)
        k = qkv[..., dq: dq + dkv].reshape(B, T, KV, HD).transpose(0, 2, 1, 3)
        v = qkv[..., dq + dkv:].reshape(B, T, KV, HD).transpose(0, 2, 1, 3)
        q = rope(q)
        k = rope(k)
        k = np.repeat(k, N // KV, axis=1)
        v = np.repeat(v, N // KV, axis=1)
        scores = (q * scale) @ k.transpose(0, 1, 3, 2)   # [B,N,T,T] fp32
        mask = np.tril(np.ones((T, T), np.float32))[None, None]
        scores = scores * mask + (1.0 - mask) * (-1e30)
        p_att = _softmax(scores.astype(np.float64), axis=-1).astype(np.float32)
        attn = (p_att @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        attn = attn @ P[p + "attn.o_w_T"]
        h = x + attn
        # MoE
        n = rms(h, P[p + "norm2.weight"])
        flat = n.reshape(-1, d)
        S = flat.shape[0]
        rl = flat @ P["router.router_w"]      # [S,E]
        probs = _softmax(rl.astype(np.float64), axis=-1).astype(np.float32)
        top_idx = np.argsort(-probs, axis=-1)[:, :K]         # [S,K]
        top_probs = np.take_along_axis(probs, top_idx, axis=-1)
        z += float(np.log(np.exp(rl.astype(np.float64) - rl.max(-1, keepdims=True)).sum(-1)) ** 2).mean()
        onehot = (top_idx[..., None] == np.arange(E)).astype(np.float32)
        expert_load = onehot.sum(axis=(0, 1))
        f = probs.sum(axis=0) / S
        pp = expert_load / S
        aux += E * float((f * pp).sum())
        # dense-mask MoE
        EH = E * H
        gu = flat @ P[p + "W_gu_T"]           # [S, 2EH]
        g, u = gu[..., :EH], gu[..., EH:]
        g = g.reshape(S, E, H).astype(np.float64)
        u = u.reshape(S, E, H).astype(np.float64)
        silu = g / (1.0 + np.exp(-g))
        aa = (silu * u).astype(np.float32)
        ww = (onehot * top_probs[..., None]).sum(axis=1)    # [S,E]
        aa = aa * ww[..., None]
        y = aa.reshape(S, EH) @ P[p + "W_down_T"]
        h = h + y.reshape(B, T, d)
        x = h
    x = rms(x, P["norm.weight"])
    logits = x @ P["emb_w"].T
    n_layers = cfg["n_layers"]
    return logits.astype(np.float32), z / n_layers, aux / n_layers


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(1e-8, float(np.abs(b).max()))
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) / denom


# --------------------------------------------------------------------------
def build_model(cfg: dict, P: dict):
    from model.config import Config
    from model.moe import MoETransformer
    c = Config()
    for k, v in cfg.items():
        if hasattr(c, k):
            setattr(c, k, v)
    model = MoETransformer(c)
    for name, arr in P.items():
        if name in model.parameters_dict():
            model.parameters_dict()[name].set_data(ms.Tensor(arr, ms.float32))
    return model, c


def load_fixture(out_dir: Path) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    npz = np.load(out_dir / "fixture.npz")
    cfg = json.loads((out_dir / "fixture_cfg.json").read_text(encoding="utf-8"))
    P = {k: npz[k] for k in npz.files
         if k not in ("x_ids", "targets", "logits", "z_loss", "aux_loss")}
    return cfg, P, npz["x_ids"], npz["targets"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-dir", default="runs/ms")
    ap.add_argument("--device", default="CPU", choices=["CPU", "Ascend"])
    ap.add_argument("--dtype", default=None, choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--skip-gradcheck", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.fixture_dir)
    if not (out_dir / "fixture.npz").exists():
        print("fixture missing; generating with torch (convert_ckpt --fixture)...")
        import convert_ckpt
        convert_ckpt.main()

    cfg, P, ids, targets = load_fixture(out_dir)
    print(f"fixture: B={ids.shape[0]} T={ids.shape[1]} "
          f"layers={cfg['n_layers']} d={cfg['d_model']} E={cfg['n_experts']}")

    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=args.device.lower())
    ms.set_seed(0)
    print(f"MS {ms.__version__} device_target={ms.get_context('device_target')}")

    nplogits, ref_z, ref_a = numpy_forward(cfg, P, ids)
    npz = np.load(out_dir / "fixture.npz")
    # G1: torch fixture vs numpy reference
    e1 = rel_err(npz["logits"], nplogits)
    print(f"G1 torch-vs-numpy logits rel_err = {e1:.3e}  {'PASS' if e1 < 1e-5 else 'FAIL'}")
    if e1 >= 1e-5:
        print("G1 FAILED: numpy reference does not match torch; do not trust G2+")
        sys.exit(1)

    if args.dtype is None:
        args.dtype = "fp32" if args.device == "CPU" else "fp16"
    model, c = build_model(cfg, P)
    model.act_dtype = ms.float32 if args.dtype == "fp32" else (
        ms.float16 if args.dtype == "fp16" else ms.bfloat16)
    model.prepare_rope_bias(ids.shape[1], model.act_dtype)
    if args.device != "CPU":
        model.probe_fused_ops(ids.shape[0], ids.shape[1], model.act_dtype)
    x_ids = ms.Tensor(ids, ms.int32)

    # G2: MS vs numpy (fp32 CPU: 1e-4; device fp16/bf16: 1e-2)
    logits, z, a = model(x_ids)
    e2 = rel_err(logits.asnumpy(), nplogits)
    tol2 = 1e-2 if (args.device != "CPU" or model.act_dtype != ms.float32) else 1e-4
    print(f"G2 MS-vs-numpy logits rel_err = {e2:.3e} (tol {tol2}) "
          f"{'PASS' if e2 < tol2 else 'FAIL'}")
    ez = abs(float(z.asnumpy()) - npz) / max(1e-8, abs(npz))
    ea = abs(float(a.asnumpy()) - npa) / max(1e-8, abs(npa))
    print(f"   z_loss rel_err={ez:.3e} aux_loss rel_err={ea:.3e} "
          f"{'PASS' if max(ez, ea) < tol2 * 10 else 'FAIL'}")

    # G3: numerical gradcheck (CPU fp32 only)
    if args.device == "CPU" and model.act_dtype == ms.float32 and not args.skip_gradcheck:
        class LossCell(nn.Cell):
            def __init__(self, m, cf):
                super().__init__()
                self.m = m
                self.ce = nn.CrossEntropyLoss(ignore_index=-1)
                self.zc = cf.get("router_z_loss_coef", 0.001)
                self.ac = cf.get("router_aux_loss_coef", 0.01)
            def construct(self, idx, tgt):
                lg, z, a = self.m(idx)
                ce = self.ce(lg.float().reshape(-1, lg.shape[-1]), tgt.reshape(-1))
                return ce + self.zc * z + self.ac * a

        cell = LossCell(model, cfg)
        tgt = ms.Tensor(targets, ms.int32)
        gfn = ms.value_and_grad(cell, grad_position=None, weights=model.trainable_params())
        _, grads = gfn(x_ids, tgt)
        gmap = {p.name: g.asnumpy() for p, g in zip(model.trainable_params(), grads)}
        # sample elements across param groups
        samples = [("emb_w", (3, 7)), ("norm.weight", (5,)),
                   ("router.router_w", (10, 2)),
                   ("layers.0.W_gu_T", (6, 8)), ("layers.0.W_down_T", (9, 4)),
                   ("layers.1.attn.qkv_w_T", (2, 5)), ("layers.1.norm2.weight", (1,))]
        eps = 1e-3
        worst = 0.0
        all_pass = True
        for name, idx in samples:
            p = model.parameters_dict()[name]
            base = p.value().asnumpy().copy()
            grad_analytic = gmap[name][idx]
            plus = base.copy()
            plus[idx] += eps
            minus = base.copy()
            minus[idx] -= eps
            l_plus = float(cell(x_ids, tgt).asnumpy())
            p.set_data(ms.Tensor(plus, ms.float32))
            l_plus = float(cell(x_ids, tgt).asnumpy())
            p.set_data(ms.Tensor(minus, ms.float32))
            l_minus = float(cell(x_ids, tgt).asnumpy())
            p.set_data(ms.Tensor(base, ms.float32))
            grad_num = (l_plus - l_minus) / (2 * eps)
            r = abs(grad_num - grad_analytic) / max(1e-8, abs(grad_analytic))
            worst = max(worst, float(r))
            ok = r < 5e-3
            all_pass &= ok
            print(f"G3 gradcheck {name}{idx}: analytic={grad_analytic:.6e} "
                  f"num={grad_num:.6e} rel={r:.3e} {'OK' if ok else 'BAD'}")
        print(f"G3 gradcheck worst rel = {worst:.3e}  {'PASS' if all_pass else 'FAIL'}")

    print("PARITY DONE")


if __name__ == "__main__":
    main()