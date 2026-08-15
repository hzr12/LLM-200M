"""OpenI 启智云脑 (C2NET) 训练入口。

在启智云脑上创建训练任务时，平台会通过环境变量注入数据与输出位置
(data_url / train_url / pretrain_model_url 等)。C2NET 模块负责把这些
URL 自动下载/挂载成容器内路径，并支持训练结束后把输出回传到云端。

标准接入方式（C2NET-BETA，见 https://openi.pcl.ac.cn/OpenIOSSG/c2net-pypi）：

    from c2net.context import prepare, upload_output
    ctx = prepare()
    ctx.dataset_path          # 数据集挂载目录（对应 data_url）
    ctx.output_path           # 输出目录（对应 train_url，回传时上传）
    ctx.pretrain_model_path   # 预训练模型挂载目录（可选）
    ctx.code_path             # 代码目录
    upload_output()           # 训练结束后把 output_path 回传到云端

数据集导入约定：平台挂载的 `dataset_path` 根目录下含一个数据子目录 `LLM`，
即训练数据放在 `dataset_path/LLM/`（内含 train.bin / val.bin / meta.json，
SFT 还含 sft_data.bin / sft_mask.bin / sft_val_* / sft_meta.json）。
本脚本默认把 `--data-dir` 指向 `dataset_path/LLM`。

本脚本是 train.py / sft_train.py 的 C2NET 包装：解析出真实路径后把
--data-dir / --out-dir / --init-from 等参数转发给底层训练脚本，训练结束
自动 upload_output()。

Usage (在启智云脑上，创建训练任务时的启动命令)：
    # 预训练（数据在 dataset_path/LLM/ 下）
    python openi_train.py --mode pretrain --total-tokens 2800000000 \
        --batch-size 32 --micro-batch 8 --ctx 2048
    # 若训练数据在数据集根目录而非 LLM 子目录：
    python openi_train.py --mode pretrain --c2net-data-subdir "" \
        --total-tokens 2800000000 --batch-size 32 --micro-batch 8 --ctx 2048
    # SFT（pretrain 权重作为预训练模型挂载，例如挂载的权重文件名为 ckpt_best.pt）
    python openi_train.py --mode sft --init-from ckpt_best.pt --epochs 3

本地调试（未安装 c2net 或未运行在云脑环境）：
    python openi_train.py --mode pretrain --data-dir data \
        --out-dir runs/moe-200m --total-tokens 50000000 --ctx 512
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


# grampus.py / 启智平台注入的启动参数。这些参数只属于平台侧，
# 必须过滤掉，绝不能透传给底层 train.py / sft_train.py（否则 argparse 报错）。
_PLATFORM_ARGS = {
    "multi_data_url", "pretrain_url", "train_url", "model_url", "code_url",
    "boot_file", "code_name", "grampus_code_file_name", "grampus_code_url",
    "grampus_model_file_name", "data_url", "openi_code_path", "openi_data_path",
}


def _filter_platform_args(args: list) -> list:
    """从 grampus 透传的 unknown 参数里剔除平台注入项，保留用户训练参数。

    支持两种形式：`--name=value` 和 `--name value`。
    """
    out = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("--"):
            name = a[2:].split("=", 1)[0].lower()
            if name in _PLATFORM_ARGS:
                # --name=value 形式直接跳过；--name value 形式还需跳过下一个
                if "=" not in a:
                    skip_next = True
                continue
        out.append(a)
    return out


def resolve_c2net_ctx():
    """尝试初始化 C2NET。返回 context 对象，或 None（本地/未安装）。"""
    try:
        from c2net.context import prepare
        return prepare()
    except ImportError:
        print("[openi] c2net not installed — falling back to local paths (dev mode)")
        return None


def main():
    ap = argparse.ArgumentParser(description="OpenI 启智云脑 (C2NET) 训练入口")
    ap.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain",
                    help="pretrain=预训练, sft=微调")
    # c2net / 云脑路径（仅在本地调试时使用，云脑上会被 c2net 覆盖）
    ap.add_argument("--data-dir", default="data",
                    help="本地调试用：数据目录（云脑上自动由 dataset_path 解析）")
    ap.add_argument("--c2net-data-subdir", default="LLM",
                    help="云脑数据集根目录下的数据子目录（默认 'LLM'，即 "
                         "dataset_path/LLM）。若你的训练数据就在数据集根目录，传空字符串 ''")
    ap.add_argument("--out-dir", default=None,
                    help="本地调试用：输出目录（云脑上自动由 output_path 覆盖）")
    ap.add_argument("--init-from", default=None,
                    help="SFT/续训权重文件。云脑上优先在 pretrain_model_path 下查找")
    args, unknown = ap.parse_known_args()

    ctx = resolve_c2net_ctx()

    data_dir = args.data_dir
    out_dir = args.out_dir or ("runs/moe-200m" if args.mode == "pretrain"
                               else "runs/moe-200m-sft")
    init_from = args.init_from

    if ctx is not None:
        # --- 云脑：使用 C2NET 解析出的真实路径 ---
        ds_path = getattr(ctx, "dataset_path", None) or data_dir
        # 输出路径 = c2net_context.output_path（平台回传目录）
        out_path = getattr(ctx, "output_path", None)
        if not out_path:
            print("[openi] warning: c2net_context.output_path is empty; "
                  "falling back to local --out-dir", file=sys.stderr)
        else:
            out_dir = str(out_path)
        # 数据集根目录下的数据子目录：dataset_path/LLM（与用户约定一致）
        sub = (args.c2net_data_subdir or "").strip().strip("/")
        data_dir = str(Path(ds_path) / sub) if sub else str(Path(ds_path))
        print(f"[openi] c2net dataset_path      = {ds_path}")
        print(f"[openi] c2net data_dir (=dataset_path/{sub}) = {data_dir}")
        print(f"[openi] c2net output_path       = {out_dir}")
        print(f"[openi] c2net pretrain_model_path = {getattr(ctx, 'pretrain_model_path', None)}")

        # 确保输出目录存在（云脑上 output_path 可能尚未创建）
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if init_from:
            init_from = _resolve_init_from(init_from, ctx)
    else:
        # --- 本地：兼容平台直接注入环境变量的场景 ---
        if init_from and not os.path.isabs(init_from):
            # 若平台把权重放进了 data_url 数据集，则尝试在数据目录下解析
            cand = Path(data_dir) / init_from
            if cand.is_file():
                init_from = str(cand)

    # --- 组装底层训练脚本命令 ---
    if args.mode == "pretrain":
        script = Path(__file__).with_name("train.py")
        cmd = [sys.executable, str(script), "--data-dir", data_dir, "--out-dir", out_dir]
    else:
        script = Path(__file__).with_name("sft_train.py")
        cmd = [sys.executable, str(script), "--data-dir", data_dir, "--out-dir", out_dir]
        if init_from:
            cmd += ["--init-from", init_from]

    # 透传前先剔除平台注入参数（grampus 会把 multi_data_url 等塞进 unknown）
    forward = _filter_platform_args(unknown)
    if forward != unknown:
        print(f"[openi] filtered {len(unknown) - len(forward)} platform args "
              f"({', '.join(_PLATFORM_ARGS & {a[2:].split('=', 1)[0] for a in unknown if a.startswith('--')}) or 'unknown'})",
              flush=True)
    cmd += forward
    print(f"[openi] cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)

    rc = 0
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n[openi] interrupted", file=sys.stderr)
        rc = 130
    except Exception as e:  # noqa: BLE001
        print(f"[openi] training script failed: {e}", file=sys.stderr)
        rc = 1

    # --- 无论成功/失败/中断，都尝试把 output_path 回传到云端 ---
    # 这样即使训练中途崩溃，已保存的 checkpoint 也能抢救回传，不会丢失。
    if ctx is not None:
        try:
            from c2net.context import upload_output
            print("[openi] uploading outputs to cloud (even if training failed)...", flush=True)
            upload_output()
            print("[openi] outputs uploaded", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[openi] upload_output failed (non-fatal): {e}", file=sys.stderr)

    if rc != 0:
        sys.exit(rc)


def _resolve_init_from(init_from: str, ctx) -> str:
    """在云脑上把 --init-from 解析成真实路径。

    优先用绝对路径/相对 CWD；否则依次在 pretrain_model_path、dataset_path
    下查找同名文件。
    """
    p = Path(init_from)
    if p.is_absolute():
        return init_from
    if p.is_file():  # 相对 CWD 存在
        return str(p)

    bases = []
    pm = getattr(ctx, "pretrain_model_path", None)
    ds = getattr(ctx, "dataset_path", None)
    if pm:
        bases.append(Path(pm))
    if ds:
        bases.append(Path(ds))
    for b in bases:
        cand = b / init_from
        if cand.is_file():
            print(f"[openi] resolved --init-from -> {cand}")
            return str(cand)
    # 找不到就原样传下去，让底层脚本报更明确的错
    print(f"[openi] warning: --init-from '{init_from}' not found under "
          f"pretrain_model_path or dataset_path; passing as-is", file=sys.stderr)
    return init_from


if __name__ == "__main__":
    main()
