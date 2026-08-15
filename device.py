"""Device abstraction: CUDA (NVIDIA GPU) / NPU (Ascend 910 via torch_npu) / CPU.

Usage:
    import device
    dev = device.get_device()          # torch.device("cuda"|"npu"|"cpu")
    is_npu = device.is_npu()           # bool
    with device.amp_context(dev):      # autocast for CUDA/NPU, no-op on CPU
        out = model(x)
    fused = device.optimizer_fused(dev)  # fused AdamW only on CUDA

Ascend 910 requires CANN + torch_npu installed. torch_npu is imported lazily so
this module is safe to import on machines without the NPU stack.
"""
from __future__ import annotations

import os

import torch

# AMP 精度选择：环境变量 LLM_SNN_AMP ∈ {"bf16", "fp16"}
# - 默认 bf16：与原设计一致（CUDA），无需 loss scaling。
# - fp16：NPU 首选。torch_npu 2.1 的 autocast 不支持 bf16（会被静默禁用，
#   导致全模型 fp32 计算，显存翻倍、吞吐暴跌）。因此 NPU 上若未显式设置
#   LLM_SNN_AMP，自动降级为 fp16，并由 train.py 自动启用 GradScaler。
def is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except (ImportError, AttributeError, RuntimeError):
        return False


def _pick_amp_dtype() -> torch.dtype:
    v = os.environ.get("LLM_SNN_AMP", "").strip().lower()
    if v in ("fp16", "float16"):
        return torch.float16
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    # 未显式指定：
    if is_npu_available():
        print("note: LLM_SNN_AMP unset on NPU -> using fp16 "
              "(bf16 autocast unsupported by torch_npu 2.1; "
              "train.py auto-enables GradScaler)", flush=True)
        return torch.float16
    return torch.bfloat16


_AMP_DTYPE = _pick_amp_dtype()


def amp_dtype() -> torch.dtype:
    return _AMP_DTYPE


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device() -> torch.device:
    """Pick the best available device: CUDA > NPU > CPU."""
    if is_npu_available():
        return torch.device("npu")
    if is_cuda_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_npu(device: torch.device | str = None) -> bool:
    device = device or get_device()
    return torch.device(device).type == "npu"


def amp_context(device: torch.device | str):
    """Return an autocast context; dtype follows LLM_SNN_AMP (default bf16).

    在 CANN8 + torch2.1 + torch_npu 2.1.0 上，若 bf16 autocast 报错或精度异常，
    可设 `LLM_SNN_AMP=fp16` 切换到 fp16（届时需自行配合 GradScaler）。
    """
    return amp_context_dtype(device, _AMP_DTYPE)


def amp_context_dtype(device: torch.device | str, dtype):
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if dev.type == "npu":
        return torch.autocast(device_type="npu", dtype=dtype)
    return torch.autocast(device_type="cpu", dtype=dtype)


def optimizer_fused(device: torch.device | str) -> bool:
    """`fused=True` in AdamW is only safe on CUDA; NPU does not support it."""
    return torch.device(device).type == "cuda"
