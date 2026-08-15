"""SFT fine-tune the MoE on tool-calling / instruction data.

Data (produced by prepare_sft.py):
    data/sft_data.bin     : uint16 token stream (concatenated ChatML dialogues)
    data/sft_mask.bin     : uint8 mask, 1 = token is part of an assistant turn (loss computed)
    data/sft_val_data.bin / sft_val_mask.bin : held-out dialogues
    data/sft_meta.json    : token counts

Loss uses ignore_index=-1 for non-assistant tokens, so the model only learns
to produce assistant turns (including <|tool_call|> outputs).

Usage:
    python sft_train.py --init-from runs/moe-200m/ckpt_best.pt --out-dir runs/moe-200m-sft
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
import device as device_lib
from model import Config, MoETransformer


def get_batch_masked(data_mmap, mask_mmap, idx, bs, ctx, device):
    """Sample bs random chunks; return (x, y, mask) where y = x shifted by 1."""
    starts = torch.randint(0, len(data_mmap) - ctx - 1, (bs,))
    x = torch.stack([torch.from_numpy(data_mmap[s:s + ctx].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(data_mmap[s + 1:s + ctx + 1].astype(np.int64)) for s in starts])
    m = torch.stack([torch.from_numpy(mask_mmap[s + 1:s + ctx + 1].astype(np.int64)) for s in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)


@torch.no_grad()
def estimate_val(model, vdata, vmask, cfg, bs, ctx, device, steps=10):
    model.eval()
    losses = []
    rng = random.Random(77)
    for _ in range(steps):
        starts = [rng.randint(0, len(vdata) - ctx - 1) for _ in range(bs)]
        x = torch.stack([torch.from_numpy(vdata[s:s + ctx].astype(np.int64)) for s in starts])
        y = torch.stack([torch.from_numpy(vdata[s + 1:s + ctx + 1].astype(np.int64)) for s in starts])
        m = torch.stack([torch.from_numpy(vmask[s + 1:s + ctx + 1].astype(np.int64)) for s in starts])
        x, y, m = x.to(device), y.to(device), m.to(device)
        with device_lib.amp_context(device):
            logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), y.view(-1), ignore_index=-1, reduction="none")
        loss = (loss.view(bs, -1) * m).sum() / m.sum()
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


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
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--init-from", default="runs/moe-200m/ckpt_best.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--out-dir", default="runs/moe-200m-sft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = device_lib.get_device()

    data_dir = Path(args.data_dir)
    train_data = np.memmap(data_dir / "sft_data.bin", dtype=np.uint16, mode="r")
    train_mask = np.memmap(data_dir / "sft_mask.bin", dtype=np.uint8, mode="r")
    val_data = np.memmap(data_dir / "sft_val_data.bin", dtype=np.uint16, mode="r")
    val_mask = np.memmap(data_dir / "sft_val_mask.bin", dtype=np.uint8, mode="r")
    sft_meta = json.loads((data_dir / "sft_meta.json").read_text(encoding="utf-8"))
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"sft train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    # load pretrained
    ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["cfg"].items() if k in Config().__dict__})
    cfg.dropout = 0.0  # no dropout during SFT
    model = MoETransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded pretrained weights from {args.init_from} (pretrain step {ckpt.get('step')})")

    n_train_tokens = len(train_data)
    steps_per_epoch = n_train_tokens // (args.batch_size * args.ctx)
    total_steps = args.epochs * steps_per_epoch
    grad_accum = max(1, args.batch_size // args.micro_batch)
    print(f"steps/epoch: {steps_per_epoch} | total steps: {total_steps}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "norm" not in name and "bias" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device_lib.optimizer_fused(device))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sft.log"

    def log(msg):
        line = f"[step {step:>5d}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def lr_at(s):
        if s < args.warmup_steps:
            return args.lr * (s + 1) / args.warmup_steps
        prog = min(1.0, (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    model.train()
    step = 0
    running, n_running = 0.0, 0
    t0 = time.time()
    best_val = float("inf")
    while step < total_steps:
        for mb in range(grad_accum):
            x, y, m = get_batch_masked(train_data, train_mask, None, args.micro_batch, args.ctx, device)
            with device_lib.amp_context(device):
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, cfg.vocab_size), y.view(-1), ignore_index=-1, reduction="none")
                loss = (loss.view(args.micro_batch, -1) * m).sum() / m.sum()
                loss = loss / grad_accum
            loss.backward()
            running += loss.item() * grad_accum
            n_running += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for g in optim.param_groups:
            g["lr"] = lr_at(step)
        optim.step()
        optim.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            log(f"loss {running/max(1,n_running):.4f} | lr {optim.param_groups[0]['lr']:.2e} | "
                f"{step}/{total_steps} | {time.time()-t0:.0f}s")
            running, n_running = 0.0, 0

        if step % 250 == 0:
            v = estimate_val(model, val_data, val_mask, cfg, args.micro_batch, args.ctx, device)
            log(f"val loss {v:.4f}")
            if v < best_val:
                best_val = v
                _save_ckpt(out_dir, "sft_best.pt", model, optim, step, cfg, loss=v, log=log)
                log(f"new best val loss {v:.4f}")

        if step % args.save_every == 0:
            # 只保留最新的 last checkpoint，避免累积大量文件（云脑回传友好）
            _save_ckpt(out_dir, "sft_last.pt", model, optim, step, cfg,
                       loss=running / max(1, n_running), log=log)

    # final: last checkpoint 即训练终点，best 已在上面按需保存
    _save_ckpt(out_dir, "sft_last.pt", model, optim, step, cfg, log=log)
    print(f"done. last SFT checkpoint -> {out_dir / 'sft_last.pt'} (best -> {out_dir / 'sft_best.pt'})")


if __name__ == "__main__":
    main()
