"""Device abstraction for MindSpore: Ascend 910B / 910 Pro A / CPU.

Usage:
    import device
    device.init_ms(mode="graph")      # before creating tensors/model
    is_910b = device.is_910b()        # True on 910B, False on 910 Pro A / CPU
    dtype = device.amp_dtype()        # bf16 on 910B, fp16 on 910 Pro A (LLM_SNN_AMP overrides)

Design rules (MindSpore):
  - No per-call autocast: weights stay fp32 Parameters in transposed [in, out]
    layout and are explicitly cast per call (_ld in model/moe.py). AMP is
    therefore purely a dtype-selection concern at init time, not a runtime
    context manager.
  - 910B (MS 2.7): bf16, no loss scaler needed.
  - 910 Pro A (MS 2.2): fp16, manual loss scaler in train.py.
"""
from __future__ import annotations

import os

import mindspore as ms
from mindspore import context

# AMP precision choice. Default follows the platform: bf16 on 910B (fp16 is
# also OK there), fp16 on 910 Pro A (bf16 Cube support is weak on 2.2 / A2).
# Override with LLM_SNN_AMP ∈ {"fp16", "bf16"}.
_PLATFORM_910B = os.environ.get("LLM_SNN_IS_910B", "").strip().lower()


def is_910b() -> bool:
    """Platform detection, multi-source so it works before/after init_ms.

    Sources, in order: LLM_SNN_IS_910B env override, ms context device_name,
    ms.hal device name. Ascend A2 (910B) reports 910B in the device name;
    Ascend A1 (910 Pro A) reports 910 or 910A.
    """
    if _PLATFORM_910B in ("1", "true", "yes"):
        return True
    if _PLATFORM_910B in ("0", "false", "no"):
        return False
    try:
        name = str(ms.get_context("device_name") or "")
        upper = name.upper()
        if "910B" in upper:
            return True
        if "910" in upper and "A" not in upper and "PA" not in upper:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        n = str(ms.hal.get_device_name() or "").upper()
        if "910B" in n:
            return True
        if "910" in n and "A" not in n and "PA" not in n:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def init_ms(mode: str = "graph") -> None:
    """One-time MindSpore init. mode: "graph" (training) or "pynative" (chat).

    On 910 Pro A (MS 2.2) only set_context exists (no set_device); both 2.2
    and 2.7 accept set_context. Device target: Ascend if available, else CPU
    (for parity tests / local runs).
    """
    try:
        from mindspore import hal
        hal.ascend.set_device(0)
    except Exception:  # noqa: BLE001
        pass
    target = "Ascend"
    if not is_ascend_available():
        target = "CPU"
    ms_mode = context.GRAPH_MODE if mode == "graph" else context.PYNATIVE_MODE
    context.set_context(mode=ms_mode, device_target=target)
    # deterministic-free performance defaults
    try:
        context.set_context(enable_compile_cache=False)
    except Exception:  # noqa: BLE001
        pass


def is_ascend_available() -> bool:
    try:
        from mindspore import hal
        return hal.ascend.is_device_available() or hal.is_device_available()
    except Exception:  # noqa: BLE001
        try:
            ms.get_context("device_target")
            return ms.get_context("device_target") == "Ascend"
        except Exception:  # noqa: BLE001
            return False


def amp_dtype():
    v = os.environ.get("LLM_SNN_AMP", "").strip().lower()
    if v in ("bf16", "bfloat16"):
        return ms.bfloat16
    if v in ("fp16", "float16"):
        return ms.float16
    # platform default: 910B → bf16 (no scaler), 910 Pro A → fp16 (scaler)
    return ms.bfloat16 if is_910b() else ms.float16


def enable_loss_scaler() -> bool:
    """910 Pro A fp16 needs the manual loss scaler; 910B bf16 does not."""
    return amp_dtype() == ms.float16


def memory_available_mb() -> float | None:
    """Approx. free device memory in MB, best effort. None if unknown."""
    try:
        return float(ms.get_context("max_device_memory")) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return None
