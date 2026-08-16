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

    Sources, in order: LLM_SNN_IS_910B env override, ms.hal device name
    (MS 2.7+, takes a device_id), npu-smi output parse (all versions).
    Ascend A2 (910B2) reports 910B in the name; A1 (910 Pro A) reports 910ProA.
    """
    if _PLATFORM_910B in ("1", "true", "yes"):
        return True
    if _PLATFORM_910B in ("0", "false", "no"):
        return False
    try:
        n = str(ms.hal.get_device_name(0) or "").upper()
        if "910B" in n:
            return True
        if "910" in n and "A" not in n and "PA" not in n:
            return True
    except Exception:  # noqa: BLE001
        pass
    smi = _npu_smi_text()
    if "910B" in smi:
        return True
    return False


def _npu_smi_text() -> str:
    try:
        import subprocess
        r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                           timeout=30)
        return (r.stdout or r.stderr or "")
    except Exception:  # noqa: BLE001
        return ""


def init_ms(mode: str = "graph") -> None:
    """One-time MindSpore init. mode: "graph" (training) or "pynative" (chat).

    MS 2.7 has ms.set_device (device_target via set_context is deprecated);
    MS 2.2 (910 Pro A) only has set_context. Both accept set_context(mode=...).
    """
    target = "Ascend"
    if not is_ascend_available():
        target = "CPU"
    try:
        ms.set_device(target)   # MS >= 2.3-ish; absent on 2.2
    except AttributeError:
        context.set_context(device_target=target)
    ms_mode = context.GRAPH_MODE if mode == "graph" else context.PYNATIVE_MODE
    context.set_context(mode=ms_mode)


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
    """Approx. device HBM total/free in MB via npu-smi; None if unknown."""
    try:
        best = None
        for line in _npu_smi_text().splitlines():
            import re
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
            if pairs:
                used, total = int(pairs[-1][0]), int(pairs[-1][1])
                if total > 0 and (best is None or total > best[1]):
                    best = (used, total)
        if best is None:
            return None
        return float(best[1] - best[0])
    except Exception:  # noqa: BLE001
        return None
