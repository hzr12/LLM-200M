"""SFT fine-tune the MoE (MindSpore) on tool-calling / instruction data.

Data (produced by prepare_sft.py):
    data/sft_data.bin     : uint16 token stream (concatenated ChatML dialogues)
    data/sft_mask.bin     : uint8 mask, 1 = assistant-turn token (loss computed)
    data/sft_val_data.bin / sft_val_mask.bin : held-out dialogues

Loss ignores non-assistant tokens (ignore_index=-1) and masks the CE sum by
the assistant mask, so the model only learns assistant turns.

Usage:
    python sft_train.py --init-from runs/moe-200m/ckpt_best_model.ckpt \
                        --out-dir runs/moe-200m-sft
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

import mindspore as ms
from mindspore import context, nn, ops

sys.path.insert(0, str(Path(__file__).parent))
import device as device_lib  # noqa: E402
from model import Config, MoETransformer  # noqa: E402
from model.moe import Attention  # noqa: E402
from train import DataPrefetcher, _str2bool, LossCell, EvalCell, _save_ckpt, \
    load_npz_into_model  # noqa: E402


def estimate_val(model, vdata, vmask, cfg, bs, ctx, steps=10, seed=77):
    rng = random.Random(seed)
    eval_cell = EvalCell(model, cfg, sft=True)
    losses = []
    for _ in range(steps):
        starts = [rng.randint(0, len(vdata) - ctx - 1) for _ in range(bs)]
        x = np.stack([vdata[s:s + ctx] for s in starts]).astype(np.int32)
        y = np.stack([vdata[s + 1:s + ctx + 1] for s in starts]).astype(np.int32)
        m = np.stack([vmask[s + 1:s + ctx + 1] for s in starts]).astype(np.float32)
        losses.append(float(eval_cell(ms.Tensor(x, ms.int32),
                                      ms.Tensor(y, ms.int32),
                                      ms.Tensor(m, ms.float32)).asnumpy()))
    return sum(losses) / len(losses)


def sft_lr_schedule(steps: int, lr: float, warmup_steps: int) -> np.ndarray:
    """warmup (1-based) + cosine decay to 0 (matches the torch SFT script)."""
    lrs = np.zeros(steps, np.float32)
    for s in range(steps):
        if s < warmup_steps:
            lrs[s] = lr * (s + 1) / max(1, warmup_steps)
        else:
            prog = min(1.0, (s - warmup_steps) / max(1, steps - warmup_steps))
            lrs[s] = lr * 0.5 * (1 + math.cos(math.pi * prog))
    return lrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--init-from", default="runs/moe-200m/ckpt_best_model.ckpt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--scaler-init-scale", type=float, default=1024.0)
    ap.add_argument("--out-dir", default="runs/moe-200m-sft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--gradient-checkpointing", type=_str2bool, default=False,
                    nargs="?", const=True, metavar="BOOL",
                    help="兼容参数：MS 版无 checkpoint 重计算")
    ap.add_argument("--flash-attention", type=_str2bool, default=False,
                    nargs="?", const=True, metavar="BOOL",
                    help="910B 启用 flash_attention_score（自动探针）")
    ap.add_argument("--fa-layout", default="bnsd", choices=["bnsd", "bsh"],
                    help="兼容参数：MS 版固定 BNSD")
    ap.add_argument("--sparse-moe", type=_str2bool, default=None,
                    nargs="?", const=True, metavar="BOOL",
                    help="兼容参数：MS 版统一 dense-mask 架构")
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    ms.set_seed(args.seed)

    device_lib.init_ms(mode="pynative")
    on_ascend = ms.get_context("device_target") == "Ascend"
    dtype = device_lib.amp_dtype()
    use_scaler = device_lib.enable_loss_scaler()
    print(f"MS {ms.__version__} target={ms.get_context('device_target')} "
          f"dtype={dtype} scaler={use_scaler} 910b={device_lib.is_910b()}")

    data_dir = Path(args.data_dir)
    train_data = np.memmap(data_dir / "sft_data.bin", dtype=np.uint16, mode="r")
    train_mask = np.memmap(data_dir / "sft_mask.bin", dtype=np.uint8, mode="r")
    val_data = np.memmap(data_dir / "sft_val_data.bin", dtype=np.uint16, mode="r")
    val_mask = np.memmap(data_dir / "sft_val_mask.bin", dtype=np.uint8, mode="r")
    print(f"sft train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    init_p = str(args.init_from)
    if init_p.endswith(".pt"):
        sys.exit("torch .pt checkpoint: 先运行 convert_ckpt.py --pt ... --out <dir>，"
                 "再把生成的 .npz 传给 --init-from")
    if init_p.endswith(".npz"):
        npz_cfg = {}
    else:
        stem = Path(init_p).stem.replace("_model", "")
        meta_path = Path(init_p).parent / f"{stem}_meta.json"
        npz_cfg = (json.loads(meta_path.read_text(encoding="utf-8")).get("cfg", {})
                   if meta_path.exists() else {})

    cfg = Config(**{k: v for k, v in npz_cfg.items() if hasattr(Config, k)})
    cfg.dropout = 0.0
    cfg.use_flash_attn = args.flash_attention
    print(f"config: layers={cfg.n_layers} d_model={cfg.d_model} "
          f"experts={cfg.n_experts} top_k={cfg.top_k}")

    model = MoETransformer(cfg)
    model.act_dtype = dtype
    if on_ascend:
        if cfg.use_flash_attn:
            model.probe_fused_ops(args.micro_batch, args.ctx, dtype)
        else:
            Attention._fa_ok = False
    else:
        Attention._fa_ok = False
    model.prepare_rope_bias(args.ctx, dtype)
    if init_p.endswith(".npz"):
        load_npz_into_model(model, init_p)
    else:
        ms.load_param_into_net(model, ms.load_checkpoint(init_p))
    print(f"loaded pretrained weights from {args.init_from}")

    n_train_tokens = len(train_data)
    steps_per_epoch = n_train_tokens // (args.batch_size * args.ctx)
    total_steps = args.epochs * steps_per_epoch
    grad_accum = max(1, args.batch_size // args.micro_batch)
    print(f"steps/epoch: {steps_per_epoch} | total steps: {total_steps} "
          f"| grad_accum: {grad_accum}")

    params = []
    for p in model.trainable_params():
        wd = args.weight_decay if (p.ndim >= 2 and "norm" not in p.name) else 0.0
        params.append({"params": p, "weight_decay": wd})
    lr_arr = sft_lr_schedule(total_steps, args.lr, args.warmup_steps)
    optim = nn.AdamWeightDecay(params, learning_rate=ms.Tensor(lr_arr, ms.float32),
                               beta1=0.9, beta2=0.95, eps=1e-8)

    context.set_context(mode=context.GRAPH_MODE)
    loss_cell = LossCell(model, cfg, sft=True)
    vg = ms.value_and_grad(loss_cell, grad_position=None,
                           weights=model.trainable_params())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sft.log"

    def log(msg):
        line = f"[step {step:>5d}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    rng = np.random.default_rng(args.seed)
    prefetcher = DataPrefetcher([(train_data, 0), (train_data, 1),
                                 (train_mask, 1)],
                                args.micro_batch, args.ctx, rng)

    print("compiling train graph (first step)...", flush=True)
    x0, y0, m0 = prefetcher.next()
    vg(ms.Tensor(x0, ms.int32), ms.Tensor(y0, ms.int32),
       ms.Tensor(m0, ms.float32))
    print("graph compiled", flush=True)

    step = 0
    scale_val = args.scaler_init_scale if use_scaler else 1.0
    running_t = ms.Tensor(0.0, ms.float32)
    n_running = 0
    best_val = float("inf")
    growth_streak = 0
    t0 = time.time()

    while step < total_steps:
        if use_scaler:
            loss_cell.scale.set_data(ms.Tensor([scale_val], ms.float32))
        acc_grads = None
        for _ in range(grad_accum):
            x, y, m = prefetcher.next()
            loss, grads = vg(ms.Tensor(x, ms.int32), ms.Tensor(y, ms.int32),
                             ms.Tensor(m, ms.float32))
            if acc_grads is None:
                acc_grads = list(grads)
            else:
                acc_grads = [g1 + g2 for g1, g2 in zip(acc_grads, grads)]
        if use_scaler:
            loss_scalar = float(loss.asnumpy()) / scale_val
            if not np.isfinite(loss_scalar):
                scale_val = max(1.0, scale_val * 0.5)
                growth_streak = 0
                log(f"loss overflow (scale -> {scale_val:.0f}); step skipped")
                step += 1
                continue
            growth_streak += 1
            if growth_streak >= 2000 and scale_val < 1e8:
                scale_val *= 2.0
                growth_streak = 0
            grads = [g / scale_val for g in acc_grads]
        else:
            grads = acc_grads
        clipped, _ = ops.clip_by_global_norm(grads, args.grad_clip)
        optim(clipped)
        running_t = running_t + loss
        step += 1
        n_running += 1

        if step % args.log_every == 0:
            avg = float(running_t.asnumpy()) / max(1, n_running)
            log(f"loss {avg:.4f} | lr {lr_arr[step-1]:.2e} | "
                f"{step}/{total_steps} | {time.time()-t0:.0f}s")
            running_t = ms.Tensor(0.0, ms.float32)
            n_running = 0

        if step % 250 == 0:
            v = estimate_val(model, val_data, val_mask, cfg,
                             args.micro_batch, args.ctx)
            log(f"val loss {v:.4f} (incl. router losses)")
            if v < best_val:
                best_val = v
                _save_ckpt(out_dir, "sft_best", model, optim, step, cfg,
                           scale_val, loss=v, log=log)
                log(f"new best val loss {v:.4f}")

        if step % args.save_every == 0:
            save_loss = (float(running_t.asnumpy()) / max(1, n_running)
                         if n_running > 0 else None)
            _save_ckpt(out_dir, "sft_last", model, optim, step, cfg,
                       scale_val, loss=save_loss, log=log)

    _save_ckpt(out_dir, "sft_last", model, optim, step, cfg, scale_val, log=log)
    print(f"done. last SFT checkpoint -> {out_dir / 'sft_last_*'} "
          f"(best -> {out_dir / 'sft_best_*'})")


if __name__ == "__main__":
    main()