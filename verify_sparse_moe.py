"""验证 sparse top-k MoE 与 dense MoE 数值等价（前向 logits + 梯度）。

用法（CUDA/CPU 环境）：
    python verify_sparse_moe.py                # fp32（最严格）
    python verify_sparse_moe.py --dtype bf16   # autocast bf16（模拟真实训练）
    python verify_sparse_moe.py --dtype fp16   # autocast fp16

原理：
  - fp32 下阈值最严（logits<1e-4, loss<1e-6, grad<1e-4），用于证明数学等价。
  - bf16/fp16 用 torch.autocast 包裹（权重仍是 fp32，与真实训练一致），
    阈值按 dtype 精度放宽：
      fp16: 尾数 10 位，logits/grad 阈值 ~1e-2
      bf16: 尾数 8 位，logits/grad 阈值 ~1e-2
  - 不要直接把模型 .half()：fp16 下初始 loss(~10) 的梯度会溢出成 inf，
    两个模型都 inf → diff = nan，与 sparse/dense 无关。
"""
import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import Config, MoETransformer


def build(dense: bool, device):
    torch.manual_seed(1234)
    cfg = Config.from_name("moe-200m")
    cfg.sparse_moe = not dense
    m = MoETransformer(cfg).to(device)
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "fp32":
        ctx = nullcontext()
    else:
        dt = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        ctx = torch.autocast(device_type=device, dtype=dt)
    print(f"device: {device} | dtype: {args.dtype} "
          f"({'fp32' if args.dtype == 'fp32' else 'autocast'})")

    m_dense, cfg = build(dense=True, device=device)
    m_sparse, _ = build(dense=False, device=device)
    m_sparse.load_state_dict(m_dense.state_dict())  # 相同权重

    x = torch.randint(0, cfg.vocab_size, (2, 64), device=device)

    # --- forward 对比（autocast 下）---
    m_dense.train()
    m_sparse.train()
    with ctx:
        logits1, losses1 = m_dense(x, x)
        logits2, losses2 = m_sparse(x, x)
    d_logits = (logits1.float() - logits2.float()).abs().max().item()
    d_loss = (losses1["total"].float() - losses2["total"].float()).abs().item()
    print(f"forward logits max diff : {d_logits:.3e}")
    print(f"forward total loss diff : {d_loss:.3e}")

    # --- 梯度对比（autocast 下；梯度累积到 fp32 权重）---
    params1 = [p for p in m_dense.parameters() if p.requires_grad]
    params2 = [p for p in m_sparse.parameters() if p.requires_grad]
    with ctx:
        g1 = torch.autograd.grad(losses1["total"], params1, allow_unused=True, retain_graph=True)
        g2 = torch.autograd.grad(losses2["total"], params2, allow_unused=True, retain_graph=True)
    diffs = []
    bad = []
    for i, (a, b) in enumerate(zip(g1, g2)):
        if a is None or b is None:
            continue
        d = (a - b).abs().max().item()
        diffs.append(d)
        if not math.isfinite(d) or d > 1e-2:
            name = list(m_dense.state_dict().keys())[i]
            bad.append((name, d))
    d_grad = max(diffs) if diffs else 0.0
    n_nonfinite = sum(1 for d in diffs if not math.isfinite(d))
    print(f"gradient max diff        : {d_grad:.3e}  (non-finite grads: {n_nonfinite})")
    if bad:
        print("non-finite / >1e-2 params:")
        for name, d in bad[:10]:
            print(f"  {name}: {d}")

    if args.dtype == "fp32":
        thr_l, thr_t, thr_g = 1e-4, 1e-6, 1e-4
    else:
        thr_l, thr_t, thr_g = 1e-2, 1e-2, 1e-2
    ok = (d_logits < thr_l and d_loss < thr_t
          and n_nonfinite == 0 and d_grad < thr_g)
    print(f"\n{'PASS' if ok else 'FAIL'}  "
          f"(thresholds: logits<{thr_l}, loss<{thr_t}, grad<{thr_g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
