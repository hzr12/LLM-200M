"""Model package.

MoETransformer / precompute_rope are the MindSpore implementation (model/moe.py).
The torch implementation lives in model/moe_torch.py and is imported lazily on
purpose: torch-only tooling (convert_ckpt.py --fixture, A-route tooling) must be
able to load moe_torch without a mindspore installation.
"""
from .config import Config

__all__ = ["Config", "MoETransformer", "precompute_rope"]


def __getattr__(name):
    if name == "MoETransformer":
        from .moe import MoETransformer
        return MoETransformer
    if name == "precompute_rope":
        from .moe import precompute_rope
        return precompute_rope
    raise AttributeError(f"module 'model' has no attribute {name!r}")
