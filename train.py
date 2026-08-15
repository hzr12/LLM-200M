"""Pre-train the 200M-A25M MoE from scratch on data/train.bin.

Usage:
    python train.py --total-tokens 2800000000 \
                    --batch-size 32 --micro-batch 8 --ctx 2048 \
                    --out-dir runs/moe-200m

Data format (nanoGPT style):
    data/train.bin / data/val.bin : uint16 token stream (memmap)
    data/meta.json                : vocab_size, special_ids

Key details:
  - bf16 AMP (GPU) / fp32 fallback (CPU)
  - AdamW with cosine LR + warmup; min_lr = lr * 0.1
  - grad accumulation + gradient clipping
  - checkpoints every --save-every steps (also keep best-val)
  - MoE router z/aux losses are added to the cross-entropy
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import device as device_lib  # noqa: E402
from model import Config, MoETransformer  # noqa: E402


def get_batch(mmap, idx, bs, ctx, device):
    starts = torch.randint(0, len(mmap) - ctx - 1, (bs,))
    x = torch.stack([torch.from_numpy(mmap[s:s + ctx].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(mmap[s + 1:s + ctx + 1].astype(np.int64)) for s in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_val(model, val_mmap, cfg, bs, ctx, device, steps=20):
    model.eval()
    losses, router_losses = [], []
    rng = random.Random(1234)
    for _ in range(steps):
        starts = [rng.randint(0, len(val_mmap) - ctx - 1) for _ in range(bs)]
        x = torch.stack([torch.from_numpy(val_mmap[s:s + ctx].astype(np.int64)) for s in starts])
        y = torch.stack([torch.from_numpy(val_mmap[s + 1:s + ctx + 1].astype(np.int64)) for s in starts])
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with device_lib.amp_context(device):
            _, losses_d = model(x, y)
        losses.append(losses_d["total"].item())
        router_losses.append((losses_d["router_z_loss"].item(), losses_d["router_aux_loss"].item()))
    model.train()
    return (sum(losses) / len(losses),
            sum(z for z, _ in router_losses) / len(router_losses),
            sum(a for _, a in router_losses) / len(router_losses))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _save_ckpt(out_dir, name: str, model, optim, step, cfg, loss=None, log=None):
    """保存单个 checkpoint 到 out_dir/name（覆盖写，避免累积）。"""
    path = out_dir / name
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "cfg": cfg.__dict__,
        "loss": loss,
    }, path)
    msg = f"checkpoint saved -> {path}"
    if log:
        log(msg)
    else:
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ctx", type=int, default=2048)
    # training
    ap.add_argument("--total-tokens", type=float, default=2.8e9)
    ap.add_argument("--batch-size", type=int, default=32)      # total tokens per update = bs*ctx
    ap.add_argument("--micro-batch", type=int, default=8)      # forward chunks for grad accum
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-tokens", type=float, default=6e7)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    # model
    ap.add_argument("--config", default="moe-200m")
    ap.add_argument("--init-from", default=None, help="checkpoint to resume from")
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None,
                    help="激活重计算，省显存（训练变慢 ~30%）。NPU 上默认开启（32GB HBM + 全量专家激活容易 OOM）；"
                         "CUDA/CPU 默认关闭。用 --no-gradient-checkpointing 显式关闭")
    # run
    ap.add_argument("--out-dir", default="runs/moe-200m")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = device_lib.get_device()

    # NPU 默认开启 gradient checkpointing：全量专家激活很大，32GB HBM 容易 OOM。
    # 用户可用 --no-gradient-checkpointing 显式关闭（CUDA/CPU 不受影响，默认关）。
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = (device.type == "npu")
        if args.gradient_checkpointing:
            print("device is NPU: gradient-checkpointing enabled by default "
                  "(pass --no-gradient-checkpointing to disable)")

    data_dir = Path(args.data_dir)
    train_mmap = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_mmap = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"train tokens: {len(train_mmap):,} | val tokens: {len(val_mmap):,} | vocab: {meta['vocab_size']}")

    cfg = Config.from_name(args.config)
    cfg.dropout = args.dropout
    cfg.gradient_checkpointing = args.gradient_checkpointing
    if meta["vocab_size"] != cfg.vocab_size:
        print(f"note: overriding cfg.vocab_size {cfg.vocab_size} -> {meta['vocab_size']} from meta.json")
        cfg.vocab_size = meta["vocab_size"]
    print(f"config: layers={cfg.n_layers} d_model={cfg.d_model} experts={cfg.n_experts} top_k={cfg.top_k}")
    print(f"params ~ {cfg.num_parameters()}")

    model = MoETransformer(cfg).to(device)
    nparams = count_parameters(model)
    print(f"trainable params: {nparams/1e6:.2f}M (cfg estimate {cfg.num_parameters()['total']/1e6:.1f}M)")

    # --- optimizer ------------------------------------------------------
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "norm" not in name and "bias" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    optim = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(args.beta1, args.beta2), fused=device_lib.optimizer_fused(device))
    print(f"optimizer: {len(decay)} decay tensors, {len(no_decay)} no-decay tensors")

    # fp16 训练必须配 GradScaler（防梯度下溢/溢出）；bf16 不需要。
    # NPU 用 torch_npu 的 GradScaler，CUDA 用 torch.cuda.amp.GradScaler。
    scaler = None
    if device_lib.amp_dtype() == torch.float16:
        try:
            from torch.npu.amp import GradScaler
        except ImportError:
            from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("using GradScaler (LLM_SNN_AMP=fp16)")

    step = 0
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        print(f"resumed from {args.init_from} at step {step}")

    tokens_per_step = args.batch_size * args.ctx
    total_steps = int(args.total_tokens // tokens_per_step)
    grad_accum = max(1, args.batch_size // args.micro_batch)
    print(f"tokens/step: {tokens_per_step:,} | total_steps: {total_steps} | grad_accum: {grad_accum}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg):
        line = f"[step {step:>6d}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def lr_at(s):
        if s < args.warmup_tokens // tokens_per_step:
            return args.lr * (s / max(1, args.warmup_tokens // tokens_per_step))
        progress = min(1.0, (s - args.warmup_tokens // tokens_per_step) /
                       max(1, total_steps - args.warmup_tokens // tokens_per_step))
        return args.lr * 0.1 + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    model.train()
    print(f"starting training on {device}")
    t0 = time.time()
    running = 0.0
    n_running = 0
    best_val = float("inf")

    while step < total_steps:
        # micro-batch loop with gradient accumulation
        for mb in range(grad_accum):
            x, y = get_batch(train_mmap, None, args.micro_batch, args.ctx, device)
            with device_lib.amp_context(device):
                _, losses_d = model(x, y)
                loss = losses_d["total"] / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * grad_accum
            n_running += 1

        # gradient clipping + step
        if scaler is not None:
            scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for g in optim.param_groups:
            g["lr"] = lr_at(step)
        if scaler is not None:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            tps = tokens_per_step * step / max(elapsed, 1e-6)
            log(f"loss {running/max(1,n_running):.4f} | lr {optim.param_groups[0]['lr']:.2e} | "
                f"{tps/1e6:.2f}M tok/s | {elapsed/60:.1f}min | {step}/{total_steps}")
            running, n_running = 0.0, 0

        if step % args.val_every == 0:
            vloss, vz, vaux = estimate_val(model, val_mmap, cfg, args.micro_batch, args.ctx, device)
            log(f"val loss {vloss:.4f} (z {vz:.4f}, aux {vaux:.4f})")
            if vloss < best_val:
                best_val = vloss
                _save_ckpt(out_dir, "ckpt_best.pt", model, optim, step, cfg,
                           loss=vloss, log=log)
                log(f"new best val loss {vloss:.4f}")

        if step % args.save_every == 0:
            # 只保留最新的 last checkpoint，避免累积大量文件（云脑回传友好）
            _save_ckpt(out_dir, "ckpt_last.pt", model, optim, step, cfg,
                       loss=running / max(1, n_running), log=log)

    # final: last checkpoint 即训练终点，best 已在上面按需保存
    _save_ckpt(out_dir, "ckpt_last.pt", model, optim, step, cfg, log=log)
    print(f"done. last checkpoint -> {out_dir / 'ckpt_last.pt'} (best -> {out_dir / 'ckpt_best.pt'})")


if __name__ == "__main__":
    main()
