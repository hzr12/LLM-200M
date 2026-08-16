"""profile_npu.py v4 — 910 Pro A 训练瓶颈完整查验脚本

v4 变更（相对 v3）：
  1. wall 统计用 profiled_run 的实际步数（v3 误用 max(1,steps-50)=70 导致
     tok/s 虚低——profiled_run 实际只跑 wait+warmup+active 步）。
  2. 新增 sanity 后的稳态 plain timing 段（预热 3 + 计时 10 步，无
     loss.item() 同步、无 profiler）——编译完成后的真实基线，兼测
     "每步 .item() 同步是否拖慢 sanity"。
  3. --gradient-checkpointing 0/1 可显式覆盖 NPU 恒开逻辑（B=8 下可试关：
     算子数 ~x0.67，host 侧提交开销大减；32G 若 OOM 回退）。
  4. DataPrefetcher 构建移到 check_flash_attn 之后（设备探测不干扰拷贝）。

用法（云脑 NPU 机）：
    python 1.py --steps 120 --micro-batch 8 --ctx 2048 --flash-attention
    python 1.py --steps 120 --micro-batch 8 --ctx 2048 --flash-attention --gradient-checkpointing 0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROFILER_AVAILABLE = False
try:
    import torch_npu  # noqa: F401
    from torch_npu import profiler as npu_profiler
    PROFILER_AVAILABLE = True
except ImportError:
    pass


# --------------------------------------------------------------------------
# 1) AMP 有效性探测（fp16/bf16 都测）
# --------------------------------------------------------------------------
def probe_amp_dtype(device: torch.device, dtype: torch.dtype) -> torch.dtype:
    """autocast 上下文跑小 matmul，看输出 dtype（不支持的目标 dtype 会静默
    禁用 autocast，输出回 fp32）。"""
    a = torch.randn(64, 768, device=device)
    b = torch.randn(768, 768, device=device)
    with torch.autocast(device_type="npu", dtype=dtype):
        y = a @ b
    return y.dtype


# --------------------------------------------------------------------------
# 2) 训练段
# --------------------------------------------------------------------------
def run_steps(model, prefetcher, device, cfg, n_steps, dtype,
              collect_loss=False, debug=False):
    """n 个 micro-batch 的 forward+backward。返回 (耗时, losses 列表)。"""
    losses = []
    t0 = time.time()
    for i in range(n_steps):
        x, y = prefetcher.next()
        with device_lib.amp_context(device):
            logits, losses_d = model(x, y)
            loss = losses_d["total"]
            # backward 必须在 autocast 内：checkpoint（use_reentrant=False）
            # 的重算发生在 backward 时，若上下文已退出，重算走 fp32 路径与
            # forward 的 fp16 路径分裂 → CheckpointError 53 vs 50 张量。
            loss.backward()
        if debug and i == 0:
            print(f"  [debug] tokens: x min/max = {int(x.min())}/{int(x.max())} "
                  f"| y min/max = {int(y.min())}/{int(y.max())}", flush=True)
            lg = logits.float()
            print(f"  [debug] logits: mean {lg.mean().item():.4f} "
                  f"std {lg.std().item():.4f} max {lg.max().item():.4f}", flush=True)
            for k in ("total", "router_z_loss", "router_aux_loss"):
                v = losses_d.get(k)
                if v is None:
                    print(f"  [debug] loss[{k}] = None", flush=True)
                else:
                    print(f"  [debug] loss[{k}] = {v.item():.6f} (dtype {v.dtype})",
                          flush=True)
        if collect_loss:
            losses.append(loss.item() if loss is not None else float("nan"))
    torch.npu.synchronize()
    return time.time() - t0, losses


def profiled_run(model, prefetcher, device, cfg, dtype, args):
    """wait=2, warmup=2, active=N；profiler 不可用则退化纯计时。
    返回 (prof, wall, n_steps)：wall 与 n_steps 为实际执行步数。"""
    n_active = max(4, min(30, args.steps))
    if not (PROFILER_AVAILABLE and getattr(npu_profiler, "profile", None)):
        print("note: torch_npu.profiler unavailable -> plain timing only", flush=True)
        wall, _ = run_steps(model, prefetcher, device, cfg, n_active, dtype)
        return None, wall, n_active
    schedule = npu_profiler.schedule(wait=2, warmup=2, active=n_active, repeat=1)
    prof = npu_profiler.profile(
        activities=[npu_profiler.ProfilerActivity.CPU,
                    npu_profiler.ProfilerActivity.NPU],
        schedule=schedule,
        on_trace_ready=npu_profiler.tensorboard_trace_handler(str(args.trace_dir)),
        record_shapes=True,
        with_stack=False,
    )
    n_total = 2 + 2 + n_active  # wait + warmup + active
    prof.start()
    t0 = time.time()
    for _ in range(n_total):
        x, y = prefetcher.next()
        with device_lib.amp_context(device):
            _, losses_d = model(x, y)
            # backward 必须在 autocast 内（checkpoint 重算依赖，见 run_steps）
            losses_d["total"].backward()
        prof.step()
    torch.npu.synchronize()
    wall = time.time() - t0
    prof.stop()
    return prof, wall, n_active


# --------------------------------------------------------------------------
# 3) 分析（key_averages；跨版本字段名防御）
# --------------------------------------------------------------------------
def _ev_time(ev):
    """取事件 self device/npu 时间：扫描字段名，跳过 cpu 字段。"""
    best = 0.0
    for attr in dir(ev):
        low = attr.lower()
        if "time" not in low or low.startswith("_"):
            continue
        if "cpu" in low:
            continue
        try:
            v = getattr(ev, attr)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, (int, float)) and v > 0:
            best = max(best, float(v))
    return best


def analyze(prof, wall, steps, amp_results, expected, losses, ctx, mb, args):
    rep = {"amp_probe": {str(k): str(v) for k, v in amp_results.items()},
           "amp_expected": str(expected),
           "amp_ok": amp_results.get(expected) == expected,
           "steps": steps, "ctx": ctx, "micro_batch": mb}

    print("\n=== NPU bottleneck report ===", flush=True)
    for d, actual in amp_results.items():
        ok = actual == d
        print(f"amp autocast {d}: -> {actual} {'OK' if ok else 'FAILED (静默降级)'}",
              flush=True)

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
        if np.allclose(arr, 0.0):
            print("!!! loss 全为 0 —— 结合 [debug] 输出判断数据/CE 问题。",
                  flush=True)

    if wall:
        rep["wall_s"] = wall
        rep["tokens_per_s"] = steps * ctx * mb / wall
        print(f"wall {wall:.1f}s for {steps} steps -> "
              f"{rep['tokens_per_s']/1e3:.1f}K tok/s", flush=True)

    if prof is None:
        print("(no profiler data)", flush=True)
        rep["gap_ratio"] = None
        return rep

    try:
        kav = prof.key_averages()
    except Exception as e:  # noqa: BLE001
        print(f"key_averages() failed ({e!r}); trace at {args.trace_dir}", flush=True)
        rep["gap_ratio"] = None
        return rep

    evs = [e for e in kav if _ev_time(e) > 0]
    if not evs:
        print("(profiler produced no usable events)", flush=True)
        rep["gap_ratio"] = None
        return rep

    kernel_s = sum(_ev_time(e) for e in evs) / 1e6
    gap = max(0.0, wall - kernel_s)
    gap_ratio = gap / wall if wall > 0 else 0.0
    rep.update({"kernel_s": kernel_s, "gap_s": gap, "gap_ratio": gap_ratio})
    print(f"kernel total {kernel_s:.1f}s / wall {wall:.1f}s -> "
          f"gap {gap_ratio*100:.0f}% (host/sync 受限程度)", flush=True)

    cats = {"transdata": [], "matmul": [], "elementwise": []}
    for e in evs:
        name = (e.name or "").lower()
        t = _ev_time(e)
        if any(s in name for s in ("transdata", "transpose", "permute",
                                   "contiguous", "formattrans")):
            cats["transdata"].append((e.name, t))
        elif any(s in name for s in ("matmul", "bmm", "linear", "matmulv",
                                     "batchmatmul")):
            cats["matmul"].append((e.name, t))
        elif any(s in name for s in ("elementwise", "silu", "mul", "add",
                                     "rsqrt", "pow", "mean", "softmax",
                                     "layernorm", "rmsnorm")):
            cats["elementwise"].append((e.name, t))
    print("\n--- 类别占比 (self NPU time) ---", flush=True)
    for k, items in cats.items():
        s = sum(t for _, t in items) / 1e6
        pct = s / kernel_s * 100 if kernel_s else 0.0
        rep[f"{k}_s"] = s
        rep[f"{k}_pct"] = pct
        top = sorted(items, key=lambda x: -x[1])[:5]
        top_s = ", ".join(f"{n}({t/1e6:.2f}s)" for n, t in top)
        print(f"{k}: {pct:.0f}% ({s:.1f}s)  top: {top_s}", flush=True)

    print("\n--- Top-20 算子 (self NPU time) ---", flush=True)
    by_name = {}
    for e in evs:
        by_name.setdefault(e.name, [0, 0.0])
        by_name[e.name][0] += 1
        by_name[e.name][1] += _ev_time(e)
    top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:20]
    rep["top_ops"] = [{"name": n, "count": c, "self_us": t} for n, (c, t) in top]
    for n, (c, t) in top:
        print(f"  {t/1e6:8.2f}s x{c:<6d} {n}", flush=True)

    out = Path(args.report)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}", flush=True)
    return rep


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="NPU bottleneck profiler v4")
    ap.add_argument("--amp", choices=["fp16", "bf16"], default="fp16",
                    help="探测精度（默认 fp16：torch_npu 2.1 autocast 只支持 fp16）")
    ap.add_argument("--data-dir", default="/home/ma-user/work/dataset/LLM")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=120,
                    help="总步数：前 50 步 loss sanity + 稳态段 + 其余 profiler 窗口")
    ap.add_argument("--gradient-checkpointing", type=int, default=None,
                    help="0/1 显式覆盖 checkpointing（默认 None=NPU 恒开）")
    ap.add_argument("--config", default="moe-200m")
    ap.add_argument("--flash-attention", action="store_true")
    ap.add_argument("--fa-layout", default="bnsd", choices=["bnsd", "bsh"])
    ap.add_argument("--trace-dir", default="npu_prof_trace")
    ap.add_argument("--report", default="npu_profile_report.json")
    args = ap.parse_args()

    # 必须在 import device 之前设置，device 模块 import 时读 LLM_SNN_AMP
    os.environ["LLM_SNN_AMP"] = args.amp
    sys.path.insert(0, str(Path(__file__).parent))
    global device_lib, Config, MoETransformer, DataPrefetcher
    import device as device_lib  # noqa: E402
    from model import Config, MoETransformer  # noqa: E402
    from train import DataPrefetcher  # noqa: E402

    device = device_lib.get_device()
    if device.type != "npu":
        print(f"error: this script is NPU-only (got {device})", flush=True)
        sys.exit(1)
    device_lib.init_npu()

    # AMP 探测：fp16 与 bf16 都测
    amp_results = {d: probe_amp_dtype(device, d) for d in (torch.float16,
                                                           torch.bfloat16)}
    for d, actual in amp_results.items():
        print(f"amp probe: {d} -> autocast produced {actual}", flush=True)
    if amp_results[torch.float16] != torch.float16:
        print("!!! fp16 autocast 也失效：检查 torch_npu 安装", flush=True)
    if amp_results.get(torch.bfloat16) != torch.bfloat16:
        print("(bf16 不可用符合预期：torch_npu 2.1 autocast 只支持 fp16)", flush=True)

    dtype = device_lib.amp_dtype()
    cfg = Config.from_name(args.config)
    cfg.use_flash_attn = args.flash_attention
    cfg.fa_layout = args.fa_layout
    if args.gradient_checkpointing is None:
        cfg.gradient_checkpointing = True  # NPU 恒开（FA 不省内存，保命项）
    else:
        cfg.gradient_checkpointing = bool(args.gradient_checkpointing)

    data_dir = Path(args.data_dir)
    train_mmap = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    model = MoETransformer(cfg).to(device)

    # FA pre-flight：用真实 micro_batch 测 forward+backward（moe.py 新版带
    # 文件预检：CANN 注册 json 损坏/缺失 → 零设备操作判定不可用）
    if cfg.use_flash_attn:
        try:
            ok = model.check_flash_attn(device, dtype, args.ctx, args.micro_batch)
        except TypeError:
            ok = model.check_flash_attn(device, dtype, args.ctx)
            print("note: moe.py 未带 batch 参数 -> probe 用 B=1，可能误判 FA 可用",
                  flush=True)
        print(f"npu_fusion_attention probe ({dtype}, B={args.micro_batch}): "
              f"{'OK' if ok else 'FAILED -> FA 禁用'}", flush=True)
        if not ok:
            cfg.use_flash_attn = False
        else:
            print("note: probe 通过但 8.0.RC1 的 FA backward 仍可能在训练中 OOM；"
                  "若 sanity 段 OOM 请改用 --micro-batch 8 或去掉 --flash-attention",
                  flush=True)

    # DataPrefetcher 在 FA probe 之后构建（设备探测不干扰数据拷贝 stream）
    prefetcher = DataPrefetcher(
        [(train_mmap, 0), (train_mmap, 1)], args.micro_batch, args.ctx, device)

    print(f"cfg: flash_attn={cfg.use_flash_attn} "
          f"gradient_checkpointing={cfg.gradient_checkpointing} "
          f"amp={dtype}", flush=True)

    # sanity 段：首步 debug + 50 步 loss（顺带完成编译预热）
    print("sanity: first 50 steps (debug + warmup + loss check)...", flush=True)
    sanity_wall, losses = run_steps(model, prefetcher, device, cfg, 50, dtype,
                                    collect_loss=True, debug=True)
    print(f"sanity wall {sanity_wall:.1f}s for 50 steps -> "
          f"{50 * args.ctx * args.micro_batch / sanity_wall / 1e3:.1f}K tok/s "
          f"(plain timing, no profiler)", flush=True)

    # 稳态段：编译完成后、无 .item() 同步、无 profiler 的真实基线
    _, _ = run_steps(model, prefetcher, device, cfg, 3, dtype)
    steady_wall, _ = run_steps(model, prefetcher, device, cfg, 10, dtype)
    print(f"steady plain wall {steady_wall:.1f}s for 10 steps -> "
          f"{10 * args.ctx * args.micro_batch / steady_wall / 1e3:.1f}K tok/s "
          f"(compiled steady state, no sync)", flush=True)

    # profiler 段
    print(f"profiled run: {max(1, args.steps - 50)} steps under "
          f"torch_npu.profiler...", flush=True)
    prof, wall, n_steps = profiled_run(model, prefetcher, device, cfg, dtype, args)

    analyze(prof, wall, n_steps, amp_results, dtype, losses,
            args.ctx, args.micro_batch, args)


if __name__ == "__main__":
    main()
