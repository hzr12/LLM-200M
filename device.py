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

# AMP 精度选择：环境变量 LLM_SNN_AMP ∈ {"fp16", "bf16"}（默认 fp16）
#
# 硬件上 Ascend910ProA 的 Cube 单元原生支持 bf16，但实测在 CANN 8 +
# torch_npu 2.1.0 环境下 bf16 autocast 无法使用（算子不支持/回退 Vector），
# 故默认回退为 fp16。fp16 需配套 GradScaler（见 train.py 的
# torch.npu.amp.GradScaler），防梯度下溢/溢出。
#
# 仅当显式 LLM_SNN_AMP=bf16 且环境确实支持时使用 bf16（免 GradScaler）。
# CUDA 上两者皆可，默认走 fp16 以保证与本仓库 NPU 路径一致。
def _pick_amp_dtype() -> torch.dtype:
    v = os.environ.get("LLM_SNN_AMP", "").strip().lower()
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16"):
        return torch.float16
    # 未显式指定：默认 fp16（CANN 8 + torch_npu 2.1 环境下 bf16 不可用）
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

    CANN 8 + torch_npu 2.1.0 环境下 bf16 autocast 不可用，故默认 fp16；
    bf16 模式需显式设 `LLM_SNN_AMP=bf16` 且环境确实支持（否则回退慢路径）。
    fp16 下 train.py 自动配合 GradScaler。
    """
    return amp_context_dtype(device, _AMP_DTYPE)


def init_npu() -> None:
    """NPU 一次性初始化：开启 JIT 编译缓存，降低首步编译台阶。

    Ascend910ProA 上 CANN 对每个新 (算子, shape) 首次执行会编译二进制落盘。
    开启 jit_compile 后编译产物缓存复用，避免用户观察到的"前 10 分钟编译"
    卡顿；同时启用算子在线编译（CANN 8 默认已带，此处确保开启）。
    """
    if not is_npu_available():
        return
    try:
        torch.npu.set_compile_mode(jit_compile=True)
    except Exception:  # noqa: BLE001
        pass  # 老版本 torch_npu 无此 API，忽略不影响功能
    # 若使用确定性/性能模式可在此设置；910B 默认性能优先，无需额外配置。


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
