"""1.py — MindSpore 训练瓶颈查验脚本（910B / 910 Pro A）。

段结构（与 torch 版 1.py 一致）：
  1. sanity: 首步 debug + 50 步 loss（顺带编译预热，图模式首步编译在此吸收）
  2. steady: 编译完成后、无 .item() 同步、无 profiler 的稳态 plain wall
     （预热 3 + 计时 10 步）——性能门只看这一段
  3. profiled: ms.Profiler 窗口（try/except，不可用则退化纯计时）

用法（节点）：
    python 1.py --steps 120 --micro-batch 32 --ctx 2048 --flash-attention
    python 1.py --steps 120 --micro-batch 8 --ctx 2048     # 910 Pro A
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import mindspore as ms
from mindspore import context, nn, ops

PROFILER_AVAILABLE = False
try:
    from mindspore.profiler import Profiler
    PROFILER_AVAILABLE = True
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
import device as device_lib  # noqa: E402
from model import Config, MoETransformer  # noqa: E402
from model.moe import Attention  # noqa: E402
from train import DataPrefetcher, LossCell  # noqa: E402


def run_steps(model, loss_cell, vg, prefetcher, cfg, n_steps, dtype,
              use_scaler, scale_val=1.0, collect_loss=False, debug=False):
    """n 个 micro-batch 的 forward+backward（无优化器）。返回 (耗时, losses)。"""
    losses = []
    t0 = time.time()
    for _ in range(n_steps):
        x, y = prefetcher.next()
        loss, _ = vg(ms.Tensor(x, ms.int32), ms.Tensor(y, ms.int32))
        if debug:
            print(f"  [debug] loss = {float(loss.asnumpy()):.6f}", flush=True)
        if collect_loss:
            losses.append(float(loss.asnumpy()) / scale_val)
    return time.time() - t0, losses


def profiled_run(model, loss_cell, vg, prefetcher, cfg, dtype, args, scale_val):
    """Profiler 窗口；不可用则退化纯计时。返回 (wall, n_steps)。"""
    n_active = max(4, min(30, args.steps))
    prof = None
    if PROFILER_AVAILABLE and os.environ.get("LLM_SNN_NO_PROFILER") != "1":
        try:
            prof = Profiler(output_path=str(args.trace_dir))
        except Exception as e:  # noqa: BLE001
            print(f"note: ms.Profiler init failed ({e!r}) -> plain timing only",
                  flush=True)
            prof = None
    n_total = 2 + 2 + n_active  # wait + warmup + active
    t0 = time.time()
    for _ in range(n_total):
        x, y = prefetcher.next()
        loss, _ = vg(ms.Tensor(x, ms.int32), ms.Tensor(y, ms.int32))
        if prof is not None:
            prof.step()
    wall = time.time() - t0
    if prof is not None:
        try:
            prof.stop()
        except Exception:  # noqa: BLE001
            pass
    return wall, n_active, prof


def analyze(wall, steps, losses, ctx, mb, args):
    rep = {"steps": steps, "ctx": ctx, "micro_batch": mb}
    print("\n=== MindSpore bottleneck report ===", flush=True)
    if losses:
        arr = np.asarray(losses, dtype=np.float64)
        nan = bool(np.isnan(arr).any())
        inf = bool(np.isinf(arr).any())
        rep["loss_stats"] = {"mean": float(arr.mean()), "std": float(arr.std()),
                             "nan": nan, "inf": inf,
                             "first3": [float(v) for v in arr[:3]]}
        print(f"loss sanity (n={len(losses)}): first3={[f'{v:.3f}' for v in arr[:3]]} "
              f"mean {arr.mean():.4f} std {arr.std():.4f} nan {nan} inf {inf}",
              flush=True)
    if wall:
        rep["wall_s"] = wall
        rep["tokens_per_s"] = steps * ctx * mb / wall
        print(f"wall {wall:.1f}s for {steps} steps -> "
              f"{rep['tokens_per_s']/1e3:.1f}K tok/s", flush=True)
    out = Path(args.report)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(description="MindSpore NPU profiler")
    ap.add_argument("--data-dir", default="/home/ma-user/work/dataset/LLM")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=120,
                    help="总步数：前 50 步 sanity + 稳态段 + 其余 profiler 窗口")
    ap.add_argument("--gradient-checkpointing", type=int, default=None,
                    help="兼容参数：MS 版无 checkpoint 重计算")
    ap.add_argument("--config", default="moe-200m")
    ap.add_argument("--flash-attention", action="store_true")
    ap.add_argument("--fa-layout", default="bnsd", choices=["bnsd", "bsh"])
    ap.add_argument("--trace-dir", default="ms_prof_trace")
    ap.add_argument("--report", default="npu_profile_report.json")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    device_lib.init_ms(mode="pynative")
    if ms.get_context("device_target") != "Ascend":
        print("error: this script is Ascend-only", flush=True)
        sys.exit(1)
    dtype = device_lib.amp_dtype()
    use_scaler = device_lib.enable_loss_scaler()
    scale_val = 1024.0 if use_scaler else 1.0
    print(f"MS {ms.__version__} dtype={dtype} scaler={use_scaler} "
          f"910b={device_lib.is_910b()} "
          f"hbm_free={device_lib.memory_available_mb()}MB", flush=True)

    cfg = Config.from_name(args.config)
    cfg.use_flash_attn = args.flash_attention
    data_dir = Path(args.data_dir)
    train_mmap = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_mmap):,}", flush=True)

    model = MoETransformer(cfg)
    model.act_dtype = dtype
    if cfg.use_flash_attn:
        model.probe_fused_ops(args.micro_batch, args.ctx, dtype)
    else:
        Attention._fa_ok = False
    model.prepare_rope_bias(args.ctx, dtype)
    context.set_context(mode=context.GRAPH_MODE)
    loss_cell = LossCell(model, cfg)
    vg = ms.value_and_grad(loss_cell, grad_position=None,
                           weights=model.trainable_params())
    rng = np.random.default_rng(1337)
    prefetcher = DataPrefetcher([(train_mmap, 0), (train_mmap, 1)],
                                args.micro_batch, args.ctx, rng)

    print("sanity: first 50 steps (debug + warmup + loss check)...", flush=True)
    sanity_wall, losses = run_steps(model, loss_cell, vg, prefetcher, cfg, 50,
                                    dtype, use_scaler, scale_val,
                                    collect_loss=True, debug=True)
    print(f"sanity wall {sanity_wall:.1f}s for 50 steps -> "
          f"{50 * args.ctx * args.micro_batch / sanity_wall / 1e3:.1f}K tok/s "
          f"(includes graph compile)", flush=True)

    print("steady segment: warmup 3 + timed 10 steps (no sync)...", flush=True)
    run_steps(model, loss_cell, vg, prefetcher, cfg, 3, dtype, use_scaler,
              scale_val)
    steady_wall, _ = run_steps(model, loss_cell, vg, prefetcher, cfg, 10,
                               dtype, use_scaler, scale_val)
    print(f"steady plain wall {steady_wall:.1f}s for 10 steps -> "
          f"{10 * args.ctx * args.micro_batch / steady_wall / 1e3:.1f}K tok/s "
          f"(compiled steady state, no sync)", flush=True)

    print(f"profiled run: {max(1, args.steps - 50)} steps...", flush=True)
    prof_wall, n_steps, prof = profiled_run(
        model, loss_cell, vg, prefetcher, cfg, dtype, args, scale_val)
    print(f"profiled wall {prof_wall:.1f}s for {n_steps} steps -> "
          f"{n_steps * args.ctx * args.micro_batch / prof_wall / 1e3:.1f}K tok/s",
          flush=True)

    analyze(prof_wall, n_steps, losses, args.ctx, args.micro_batch, args)


if __name__ == "__main__":
    main()