"""Pre-train the 200M-A25M MoE from scratch on data/train.bin.

Usage (NPU, full-speed):
    python train.py --total-tokens 2800000000 \
                    --batch-size 32 --micro-batch 16 --ctx 2048 \
                    --flash-attention --gradient-checkpointing 0 \
                    --out-dir runs/moe-200m
    # --fa-layout bsh to switch npu_fusion_attention layout (default bnsd)

Data format (nanoGPT style):
    data/train.bin / data/val.bin : uint16 token stream (memmap)
    data/meta.json                : vocab_size, special_ids

Key details:
  - bf16 AMP (GPU) / fp32 fallback (CPU)
  - AdamW with cosine LR + warmup; min_lr = lr * 0.1
  - grad accumulation + gradient clipping
  - checkpoints every --save-every steps (also keep best-val)
  - MoE router z/aux losses are added to the cross-entropy
  - speed tricks (all enabled by default):
      * async micro-batch prefetch (DataPrefetcher) overlaps H2D copies
        with compute
      * fused per-expert weight caches are rebuilt once per optimizer step
        instead of cat()-ed every forward call
      * NPU + --flash-attention: gradient-checkpointing auto-disables (FA frees
        HBM); override with --gradient-checkpointing 1/0
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


def _str2bool(v):
    """Parse a boolean CLI value.

    Accepts both the bare-flag style (``--flag``, handled via nargs='?'
    + const) and explicit values: ``--flag 1``, ``--flag 0``,
    ``--flag True``, ``--flag=False``, ``--flag on/off`` ... This lets
    training-platform parameter forms (which require a value) work the
    same as local command lines.
    """
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r} (use 1/0/true/false)")


def get_batch(mmap, idx, bs, ctx, device):
    starts = torch.randint(0, len(mmap) - ctx - 1, (bs,))
    x = torch.stack([torch.from_numpy(mmap[s:s + ctx].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(mmap[s + 1:s + ctx + 1].astype(np.int64)) for s in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class DataPrefetcher:
    """Prefetch the next micro-batch on a side stream.

    The host->device copies of micro-batch N+1 are queued while micro-batch N
    is still computing, hiding the transfer latency. Falls back to a plain
    synchronous fetch when side streams are unavailable (e.g. CPU).

    ``cols`` is an iterable of ``(memmap, shift)``: each column reads
    ``memmap[s+shift : s+shift+ctx]`` for a random start ``s`` shared across
    all columns. Pre-train uses ``[(train, 0), (train, 1)]`` for (x, y); SFT
    adds a third mask column ``[(data, 0), (data, 1), (mask, 1)]``.
    ``next()`` returns a list with one tensor per column.
    """

    def __init__(self, cols, bs, ctx, device):
        self.cols = list(cols)
        self.bs = bs
        self.ctx = ctx
        self.device = torch.device(device)
        self._stream = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            self._stream = torch.cuda.Stream()
        elif self.device.type == "npu":
            try:
                import torch_npu  # noqa: F401
                if hasattr(torch, "npu") and hasattr(torch.npu, "stream"):
                    self._stream = torch.npu.Stream()
            except (ImportError, AttributeError, RuntimeError):
                self._stream = None
        self._next = None
        self.prefetch()

    def _load(self):
        bs, ctx = self.bs, self.ctx
        n = len(self.cols[0][0])
        starts = torch.randint(0, n - ctx - 1, (bs,))
        out = []
        for mmap, shift in self.cols:
            out.append(torch.stack(
                [torch.from_numpy(mmap[s + shift:s + shift + ctx].astype(np.int64))
                 for s in starts.tolist()]))
        return out

    def _copy(self, cols):
        return [c.to(self.device, non_blocking=True) for c in cols]

    def prefetch(self):
        cols = self._load()  # sync CPU work (small); copy below is async
        if self._stream is None:
            self._next = self._copy(cols)
            return
        if self.device.type == "cuda":
            with torch.cuda.stream(self._stream):
                self._next = self._copy(cols)
        else:
            with torch.npu.stream(self._stream):
                self._next = self._copy(cols)

    def next(self):
        # make sure the previous prefetch finished before handing the batch out
        if self._stream is not None:
            if self.device.type == "cuda":
                torch.cuda.current_stream().wait_stream(self._stream)
            else:
                torch.npu.current_stream().wait_stream(self._stream)
        batch = self._next
        self.prefetch()  # start loading the following micro-batch now
        return batch


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


def _save_ckpt(out_dir, name: str, model, optim, step, cfg, scaler=None, loss=None, log=None):
    """保存单个 checkpoint 到 out_dir/name（覆盖写，避免累积）。"""
    path = out_dir / name
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "cfg": cfg.__dict__,
        "scaler": scaler.state_dict() if scaler is not None else None,
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
    # 布尔参数同时支持裸写（--flag）和带值（--flag 1/0/True/False）两种写法，
    # 方便训练平台以 key=value 传参。
    ap.add_argument("--gradient-checkpointing", type=_str2bool, default=None,
                    nargs="?", const=True, metavar="BOOL",
                    help="激活重计算，省显存（训练变慢约 30%%，argparse help 转义）。NPU 上默认开启（32GB HBM + 全量专家激活容易 OOM）；"
                         "CUDA/CPU 默认关闭。关闭用 --gradient-checkpointing 0 / False")
    ap.add_argument("--flash-attention", type=_str2bool, default=False,
                    nargs="?", const=True, metavar="BOOL",
                    help="启用 FlashAttention（CUDA flash-attn / Ascend npu_fusion_attention）。"
                         "可用 --flash-attention 1/0、--flash-attention True/False 显式指定；"
                         "NPU 上建议开启，可同时提速并省显存；默认关闭")
    ap.add_argument("--fa-layout", default="bnsd", choices=["bnsd", "bsh"],
                    help="npufusion_attention 输入布局：bnsd=[B,N,S,D]（默认），bsh=[B,S,H]。"
                         "部分 CANN 版本只对其中一种布局提供 kernel")
    ap.add_argument("--sparse-moe", type=_str2bool, default=None,
                    nargs="?", const=True, metavar="BOOL",
                    help="sparse MoE：只对 router 选中的 top-k 专家计算（FLOPs 约降为 k/E，"
                         "默认开启）。关闭用 --sparse-moe 0（回退全专家计算 + top-k 加权，"
                         "NPU 上若 index_add 算子不稳定可用）")
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
    # 但若 FlashAttention 已开启（--flash-attention），省下的激活内存足够关掉重计算，
    # 此时自动默认关闭（训练快 ~30%）。用户始终可用 --gradient-checkpointing 1/0
    # 显式覆盖（CUDA/CPU 不受影响，默认关）。
    if args.gradient_checkpointing is None:
        if device.type == "npu" and args.flash_attention:
            args.gradient_checkpointing = False
            print("device is NPU + --flash-attention: gradient-checkpointing disabled "
                  "(FA frees HBM; pass --gradient-checkpointing 1 to force)")
        else:
            args.gradient_checkpointing = (device.type == "npu")
            if args.gradient_checkpointing:
                print("device is NPU: gradient-checkpointing enabled by default "
                      "(pass --gradient-checkpointing 0 to disable)")

    data_dir = Path(args.data_dir)
    train_mmap = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_mmap = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"train tokens: {len(train_mmap):,} | val tokens: {len(val_mmap):,} | vocab: {meta['vocab_size']}")

    cfg = Config.from_name(args.config)
    cfg.dropout = args.dropout
    cfg.gradient_checkpointing = args.gradient_checkpointing
    cfg.use_flash_attn = args.flash_attention
    cfg.fa_layout = args.fa_layout
    if args.sparse_moe is not None:
        cfg.sparse_moe = args.sparse_moe
    if meta["vocab_size"] != cfg.vocab_size:
        print(f"note: overriding cfg.vocab_size {cfg.vocab_size} -> {meta['vocab_size']} from meta.json")
        cfg.vocab_size = meta["vocab_size"]
    print(f"config: layers={cfg.n_layers} d_model={cfg.d_model} experts={cfg.n_experts} top_k={cfg.top_k}")
    print(f"params ~ {cfg.num_parameters()}")

    model = MoETransformer(cfg).to(device)
    # FA pre-flight 同步探测：npu_fusion_attention 是异步算子，若该 CANN 环境
    # 没有 kernel，错误只会在 backward/拷贝等同步点爆发导致崩溃（try/except
    # 无法捕获）。这里训练前先探测一次，不可用则自动禁用 FA 并降级。
    if cfg.use_flash_attn and device.type == "npu":
        if not model.check_flash_attn(device, device_lib.amp_dtype(), seq_len=args.ctx):
            print("note: npu_fusion_attention unavailable -> disabling --flash-attention", flush=True)
            cfg.use_flash_attn = False
            # FA 失效后激活内存回到高占用，慢速 attention + 全量专家激活在
            # NPU 32GB HBM 上几乎必然 OOM。因此即便用户显式传过
            # --gradient-checkpointing 0（其前提是 FA 已省下激活内存）也强制
            # 开启，避免训练中途 OOM 崩溃。
            if not cfg.gradient_checkpointing:
                cfg.gradient_checkpointing = True
                print("note: forcing gradient-checkpointing=1 (slow attention without FA "
                      "needs it to fit NPU HBM)", flush=True)
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
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
            print(f"  restored GradScaler scale={ckpt['scaler']['scale'].item():.4e}")
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
    prefetcher = DataPrefetcher(
        [(train_mmap, 0), (train_mmap, 1)], args.micro_batch, args.ctx, device)
    t0 = time.time()
    running = 0.0
    n_running = 0
    best_val = float("inf")

    while step < total_steps:
        # micro-batch loop with gradient accumulation
        # 在设备上累积 loss，log 时才同步一次 .item()。原实现每 micro-batch
        # 调一次 .item()，会强制清空 NPU 算子队列、打断 prefetch 的异步重叠。
        running_acc = torch.zeros((), device=device)
        for mb in range(grad_accum):
            x, y = prefetcher.next()
            with device_lib.amp_context(device):
                _, losses_d = model(x, y)
                loss = losses_d["total"] / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_acc += loss.detach()
            n_running += 1
        running += running_acc.item() * grad_accum

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
        # 权重已更新：重建融合的专家权重缓存（cat 结果），供下一轮 forward 使用
        model.refresh_expert_caches()
        optim.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            tps = tokens_per_step * step / max(elapsed, 1e-6)
            # 附上显存占用（进程内查询），便于判断慢速/崩溃是否由显存碎片或泄漏导致
            mem = ""
            if device.type == "npu":
                mem = (f" | hbm {torch.npu.memory_allocated()/1e9:.2f}/"
                       f"{torch.npu.memory_reserved()/1e9:.2f}G")
            elif device.type == "cuda":
                mem = (f" | hbm {torch.cuda.memory_allocated()/1e9:.2f}/"
                       f"{torch.cuda.memory_reserved()/1e9:.2f}G")
            log(f"loss {running/max(1,n_running):.4f} | lr {optim.param_groups[0]['lr']:.2e} | "
                f"{tps/1e6:.2f}M tok/s | {elapsed/60:.1f}min | {step}/{total_steps}{mem}")
            running, n_running = 0.0, 0

        if step % args.val_every == 0:
            vloss, vz, vaux = estimate_val(model, val_mmap, cfg, args.micro_batch, args.ctx, device)
            log(f"val loss {vloss:.4f} (z {vz:.4f}, aux {vaux:.4f})")
            if vloss < best_val:
                best_val = vloss
                _save_ckpt(out_dir, "ckpt_best.pt", model, optim, step, cfg,
                           scaler=scaler, loss=vloss, log=log)
                log(f"new best val loss {vloss:.4f}")

        if step % args.save_every == 0:
            # 只保留最新的 last checkpoint，避免累积大量文件（云脑回传友好）
            _save_ckpt(out_dir, "ckpt_last.pt", model, optim, step, cfg,
                       scaler=scaler, loss=running / max(1, n_running), log=log)

    # final: last checkpoint 即训练终点，best 已在上面按需保存
    _save_ckpt(out_dir, "ckpt_last.pt", model, optim, step, cfg, scaler=scaler, log=log)
    print(f"done. last checkpoint -> {out_dir / 'ckpt_last.pt'} (best -> {out_dir / 'ckpt_best.pt'})")


if __name__ == "__main__":
    main()
