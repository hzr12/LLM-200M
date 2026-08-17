"""Pre-train the 200M-A25M MoE (MindSpore) on data/train.bin.

Usage (910B, full-speed):
    python train.py --total-tokens 2800000000 \
                    --batch-size 32 --micro-batch 32 --ctx 2048 \
                    --flash-attention --out-dir runs/moe-200m
    # 910 Pro A (fp16 + 手动 LossScaler，无融合算子)：
    python train.py --total-tokens 2800000000 --batch-size 32 --micro-batch 32 \
                    --ctx 2048 --out-dir runs/moe-200m

CLI 与 torch 版本保持兼容（openi_train.py 按原参数调用）。仅保留语义的开关
（--gradient-checkpointing/--fa-layout/--sparse-moe）照常接受但不改变行为：
  - MS 版无 checkpoint 重计算（图模式），内存压力靠 micro-batch 控制；
  - FA 布局固定 BNSD（910B MS 2.7 的 flash_attention_score）；
  - sparse/dense MoE 已统一为 dense-mask 大 GEMM。
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

import mindspore as ms
from mindspore import context, nn, ops

sys.path.insert(0, str(Path(__file__).parent))
import device as device_lib  # noqa: E402
from model import Config, MoETransformer  # noqa: E402
from model.moe import Attention  # noqa: E402


def _str2bool(v):
    """Parse a boolean CLI value (bare flag or key=value forms)."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
class DataPrefetcher:
    """Host-side numpy batch prefetcher (columns of (memmap, shift))."""

    def __init__(self, cols, bs, ctx, rng):
        self.cols = list(cols)
        self.bs = bs
        self.ctx = ctx
        self.rng = rng
        self._next = None
        self.prefetch()

    def _load(self):
        n = len(self.cols[0][0])
        starts = self.rng.integers(0, n - self.ctx - 1, self.bs)
        out = []
        for mmap, shift in self.cols:
            out.append(np.stack(
                [mmap[s + shift:s + shift + self.ctx] for s in starts])
                .astype(np.int32))
        return out

    def prefetch(self):
        self._next = self._load()

    def next(self):
        batch = self._next
        self.prefetch()
        return batch


# --------------------------------------------------------------------------
# loss cells
# --------------------------------------------------------------------------
class LossCell(nn.Cell):
    """CE(+router losses) with an in-graph loss scale Parameter (fp16 scaler).

    scale is a non-trainable Parameter updated host-side via set_data, so the
    train graph stays static; grads come back pre-scaled and are divided
    host-side before clipping.
    """

    def __init__(self, model: MoETransformer, cfg: Config, sft: bool = False):
        super().__init__()
        self.model = model
        self.sft = sft
        self.ce = nn.CrossEntropyLoss(ignore_index=-1,
                                      reduction="none" if sft else "mean")
        self.zc = cfg.router_z_loss_coef
        self.ac = cfg.router_aux_loss_coef
        self.scale = ms.Parameter(ms.Tensor([1.0], ms.float32),
                                  name="loss_scale", requires_grad=False)

    def construct(self, idx, targets, mask=None):
        logits, z, a = self.model(idx)
        ce = self.ce(logits.float().reshape(-1, logits.shape[-1]),
                     targets.reshape(-1))
        if self.sft:
            B = targets.shape[0]
            ce = (ce.reshape(B, -1) * mask).sum() / mask.sum()
        total = (ce + self.zc * z + self.ac * a) * self.scale
        return total


class EvalCell(nn.Cell):
    """Unscaled loss for validation (train/eval share one model graph)."""

    def __init__(self, model: MoETransformer, cfg: Config, sft: bool = False):
        super().__init__()
        self.model = model
        self.sft = sft
        self.ce = nn.CrossEntropyLoss(ignore_index=-1,
                                      reduction="none" if sft else "mean")
        self.zc = cfg.router_z_loss_coef
        self.ac = cfg.router_aux_loss_coef

    def construct(self, idx, targets, mask=None):
        logits, z, a = self.model(idx)
        ce = self.ce(logits.float().reshape(-1, logits.shape[-1]),
                     targets.reshape(-1))
        if self.sft:
            B = targets.shape[0]
            ce = (ce.reshape(B, -1) * mask).sum() / mask.sum()
        return ce + self.zc * z + self.ac * a


def estimate_val(model, val_mmap, cfg, bs, ctx, steps=20, seed=1234):
    rng = random.Random(seed)
    eval_cell = EvalCell(model, cfg)
    losses = []
    for _ in range(steps):
        starts = [rng.randint(0, len(val_mmap) - ctx - 1) for _ in range(bs)]
        x = np.stack([val_mmap[s:s + ctx] for s in starts]).astype(np.int32)
        y = np.stack([val_mmap[s + 1:s + ctx + 1] for s in starts]).astype(np.int32)
        losses.append(float(eval_cell(ms.Tensor(x, ms.int32),
                                      ms.Tensor(y, ms.int32)).asnumpy()))
    return sum(losses) / len(losses)


# --------------------------------------------------------------------------
# checkpoint helpers
# --------------------------------------------------------------------------
def load_npz_into_model(model, path: str) -> None:
    """Load convert_ckpt.py .npz (MS-layout float32 arrays) into the model."""
    npz = np.load(path)
    pd = {k: ms.Parameter(ms.Tensor(npz[k], ms.float32), name=k)
          for k in npz.files}
    ms.load_param_into_net(model, pd)
    print(f"loaded weights from {path} ({len(npz.files)} params)")


def _save_ckpt(out_dir: Path, name: str, model, optim, step, cfg,
               scale_val: float, loss=None, log=None):
    """Save model + optimizer + meta as three files (overwrite)."""
    ms.save_checkpoint(model, str(out_dir / f"{name}_model.ckpt"))
    ms.save_checkpoint(optim, str(out_dir / f"{name}_optim.ckpt"))
    meta = {"step": step, "loss": loss, "scale": scale_val, "cfg": cfg.__dict__}
    (out_dir / f"{name}_meta.json").write_text(
        json.dumps(meta, default=str), encoding="utf-8")
    msg = f"checkpoint saved -> {out_dir / name}_*"
    if log:
        log(msg)
    else:
        print(msg, flush=True)


def lr_schedule(steps: int, lr: float, warmup_steps: int,
                total_steps: int, floor: float = 0.1) -> np.ndarray:
    """warmup (0-based) + cosine decay to lr*floor; full numpy array."""
    lrs = np.zeros(steps, np.float32)
    for s in range(steps):
        if s < warmup_steps:
            lrs[s] = lr * s / max(1, warmup_steps)
        else:
            progress = min(1.0, (s - warmup_steps)
                           / max(1, total_steps - warmup_steps))
            lrs[s] = lr * floor + (1 - floor) * lr * 0.5 \
                * (1 + math.cos(math.pi * progress))
    return lrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--total-tokens", type=float, default=2.8e9)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-tokens", type=float, default=6e7)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--scaler-init-scale", type=float, default=1024.0,
                    help="fp16 手动 LossScaler 初始值（仅 910 Pro A 生效）")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--config", default="moe-200m")
    ap.add_argument("--init-from", default=None,
                    help="resume: 转换后的 .npz（模型权重）或本脚本保存的 "
                         "*_model.ckpt（含优化器/step）")
    ap.add_argument("--gradient-checkpointing", type=_str2bool, default=None,
                    nargs="?", const=True, metavar="BOOL",
                    help="MS 版无 checkpoint 重计算（图模式），此开关仅兼容 CLI，"
                         "内存压力请用 --micro-batch 控制")
    ap.add_argument("--flash-attention", type=_str2bool, default=False,
                    nargs="?", const=True, metavar="BOOL",
                    help="910B 启用 flash_attention_score（自动探针，失败降级慢注意力）；"
                         "0 强制走慢注意力")
    ap.add_argument("--fa-layout", default="bnsd", choices=["bnsd", "bsh"],
                    help="兼容参数：MS 版固定 BNSD")
    ap.add_argument("--sparse-moe", type=_str2bool, default=None,
                    nargs="?", const=True, metavar="BOOL",
                    help="兼容参数：MS 版统一 dense-mask 架构")
    ap.add_argument("--out-dir", default="runs/moe-200m")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    ms.set_seed(args.seed)

    device_lib.init_ms(mode="pynative")
    on_ascend = ms.get_context("device_target") == "Ascend"
    dtype = device_lib.amp_dtype()
    use_scaler = device_lib.enable_loss_scaler()
    print(f"MS {ms.__version__} target={ms.get_context('device_target')} "
          f"dtype={dtype} scaler={use_scaler} 910b={device_lib.is_910b()} "
          f"hbm_free={device_lib.memory_available_mb()}MB")

    data_dir = Path(args.data_dir)
    train_mmap = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_mmap = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    print(f"train tokens: {len(train_mmap):,} | val tokens: {len(val_mmap):,} "
          f"| vocab: {meta['vocab_size']}")

    cfg = Config.from_name(args.config)
    cfg.dropout = args.dropout
    cfg.use_flash_attn = args.flash_attention
    if meta["vocab_size"] != cfg.vocab_size:
        print(f"note: overriding cfg.vocab_size {cfg.vocab_size} -> {meta['vocab_size']}")
        cfg.vocab_size = meta["vocab_size"]
    print(f"config: layers={cfg.n_layers} d_model={cfg.d_model} "
          f"experts={cfg.n_experts} top_k={cfg.top_k} | params ~ {cfg.num_parameters()}")

    model = MoETransformer(cfg)
    model.act_dtype = dtype
    if on_ascend:
        if cfg.use_flash_attn:
            model.probe_fused_ops(args.micro_batch, args.ctx, dtype)
        else:
            Attention._fa_ok = False
            print("probe: --flash-attention 0 -> slow attention", flush=True)
    else:
        Attention._fa_ok = False
    model.prepare_rope_bias(args.ctx, dtype)

    step = 0
    scale_val = args.scaler_init_scale if use_scaler else 1.0
    resume_optim_pd = None
    if args.init_from:
        p = str(args.init_from)
        if p.endswith(".pt"):
            sys.exit("torch .pt checkpoint: 先运行 convert_ckpt.py --pt ... "
                     "--out <dir>，再把生成的 .npz 传给 --init-from")
        if p.endswith(".npz"):
            load_npz_into_model(model, p)
        else:  # .ckpt saved by this script
            ms.load_param_into_net(model, ms.load_checkpoint(p))
            stem = Path(p).stem.replace("_model", "")
            optim_path = Path(p).parent / f"{stem}_optim.ckpt"
            meta_path = Path(p).parent / f"{stem}_meta.json"
            if optim_path.exists() and meta_path.exists():
                resume_optim_pd = ms.load_checkpoint(str(optim_path))
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                step = int(m.get("step", 0))
                scale_val = float(m.get("scale", scale_val))
                print(f"resumed from {p} at step {step}")
            else:
                print(f"resumed model weights from {p} (no optim/meta found)")

    tokens_per_step = args.batch_size * args.ctx
    total_steps = int(args.total_tokens // tokens_per_step)
    grad_accum = max(1, args.batch_size // args.micro_batch)
    print(f"tokens/step: {tokens_per_step:,} | total_steps: {total_steps} "
          f"| grad_accum: {grad_accum}")

    # optimizer: per-param weight decay groups, order preserved from
    # trainable_params() so value_and_grad grads stay aligned.
    params = []
    for p in model.trainable_params():
        wd = args.weight_decay if (p.ndim >= 2 and "norm" not in p.name) else 0.0
        params.append({"params": p, "weight_decay": wd})
    lr_arr = lr_schedule(total_steps, args.lr,
                         int(args.warmup_tokens // tokens_per_step),
                         total_steps, floor=0.1)
    optim = nn.AdamWeightDecay(params, learning_rate=ms.Tensor(lr_arr, ms.float32),
                               beta1=args.beta1, beta2=args.beta2, eps=1e-8)
    if resume_optim_pd is not None:
        ms.load_param_into_net(optim, resume_optim_pd)
        print("optimizer state restored")

    # switch to graph mode AFTER probes / weight load / bias-prep
    context.set_context(mode=context.GRAPH_MODE)
    loss_cell = LossCell(model, cfg)
    vg = ms.value_and_grad(loss_cell, grad_position=None,
                           weights=model.trainable_params())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg):
        line = f"[step {step:>6d}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    rng = np.random.default_rng(args.seed)
    prefetcher = DataPrefetcher([(train_mmap, 0), (train_mmap, 1)],
                                args.micro_batch, args.ctx, rng)

    # graph compile dry-run (one vg call; no weight update)
    print("compiling train graph (first step)...", flush=True)
    x0, y0 = prefetcher.next()
    vg(ms.Tensor(x0, ms.int32), ms.Tensor(y0, ms.int32))
    print("graph compiled", flush=True)

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
            x, y = prefetcher.next()
            loss, grads = vg(ms.Tensor(x, ms.int32), ms.Tensor(y, ms.int32))
            if acc_grads is None:
                acc_grads = list(grads)
            else:
                acc_grads = [g1 + g2 for g1, g2 in zip(acc_grads, grads)]
        if use_scaler:
            # 每步同步一次：fp16 必须做 overflow 检查
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
            # bf16/fp32：不逐步同步，loss 仅在 log 时一次性读回
            grads = acc_grads
        clipped, _ = ops.clip_by_global_norm(grads, args.grad_clip)
        optim(clipped)
        running_t = running_t + loss   # device-side accumulation, no sync
        step += 1
        n_running += 1

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            tps = tokens_per_step * step / max(elapsed, 1e-6)
            avg_loss = float(running_t.asnumpy()) / max(1, n_running)
            mem = (f" | hbm_free {device_lib.memory_available_mb()/1e3:.1f}G"
                   if on_ascend else "")
            log(f"loss {avg_loss:.4f} | "
                f"lr {lr_arr[step-1]:.2e} | {tps/1e6:.2f}M tok/s | "
                f"{elapsed/60:.1f}min | {step}/{total_steps}{mem}")
            running_t = ms.Tensor(0.0, ms.float32)
            n_running = 0

        if step % args.val_every == 0:
            vloss = estimate_val(model, val_mmap, cfg,
                                 args.micro_batch, args.ctx)
            log(f"val loss {vloss:.4f} (incl. router losses)")
            if vloss < best_val:
                best_val = vloss
                _save_ckpt(out_dir, "ckpt_best", model, optim, step, cfg,
                           scale_val, loss=vloss, log=log)
                log(f"new best val loss {vloss:.4f}")

        if step % args.save_every == 0:
            save_loss = (float(running_t.asnumpy()) / max(1, n_running)
                         if n_running > 0 else None)
            _save_ckpt(out_dir, "ckpt_last", model, optim, step, cfg,
                       scale_val, loss=save_loss, log=log)

    _save_ckpt(out_dir, "ckpt_last", model, optim, step, cfg, scale_val, log=log)
    print(f"done. last checkpoint -> {out_dir / 'ckpt_last_*'} "
          f"(best -> {out_dir / 'ckpt_best_*'})")


if __name__ == "__main__":
    main()