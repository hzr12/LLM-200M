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

# AMP 精度选择：环境变量 LLM_SNN_AMP ∈ {"fp16", "bf16"}
# - 默认 fp16：训练显存小、NPU/CUDA 均稳定支持，train.py 会自动启用
#   GradScaler 防止梯度下溢/溢出（本设计当前默认）。
# - bf16：仅当显式设置 LLM_SNN_AMP=bf16 时使用。注意 torch_npu 2.1 的
#   autocast 不支持 bf16（会被静默禁用导致全模型 fp32 计算），故 NPU 上
#   不推荐。
def _pick_amp_dtype() -> torch.dtype:
    v = os.environ.get("LLM_SNN_AMP", "").strip().lower()
    if v in ("fp16", "float16"):
        return torch.float16
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    # 未显式指定：默认 fp16（全局）
    print("note: LLM_SNN_AMP unset -> defaulting to fp16 "
          "(train.py auto-enables GradScaler; set LLM_SNN_AMP=bf16 to override)",
          flush=True)
    return torch.float16


_AMP_DTYPE = _pick_amp_dtype()


def amp_dtype() -> torch.dtype:
    return _AMP_DTYPE


def is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except (ImportError, AttributeError, RuntimeError):
        return False


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
    """Return an autocast context; dtype follows LLM_SNN_AMP (default fp16).

    torch_npu 2.1 的 autocast 不支持 bf16，故默认使用 fp16；bf16 需显式
    设 `LLM_SNN_AMP=bf16`（仅推荐 CUDA 上使用）。fp16 模式下 train.py
    会自动配合 GradScaler。
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
