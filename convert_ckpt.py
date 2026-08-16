"""Convert a torch checkpoint (.pt) to MindSpore-compatible numpy arrays.

Only needs torch + numpy (no mindspore): output is a flat .npz keyed by the
MindSpore parameter names (already transposed to the MS [in, out] layout) plus
a meta.json. train.py loads it with ms.Tensor + load_param_into_net.

Usage:
    python convert_ckpt.py --pt runs/moe-200m/ckpt_last.pt --out runs/ms
    # torch CPU 上从旧权重生成验证 fixture（小模型、固定种子）：
    python convert_ckpt.py --fixture --out runs/ms/fixture.npz

Key mapping (torch state_dict -> MS param):
  token_embedding.weight       -> emb_w                 (tied lm_head, no copy)
  norm.weight                  -> norm.weight
  router.proj.weight           -> router.router_w       (transposed)
  layers.{i}.norm1.weight      -> layers.{i}.norm1.weight
  layers.{i}.attn.qkv_proj.w.  -> layers.{i}.attn.qkv_w_T   (transposed)
  layers.{i}.attn.o_proj.w.    -> layers.{i}.attn.o_w_T     (transposed)
  layers.{i}.norm2.weight      -> layers.{i}.norm2.weight
  layers.{i}.experts.{j}.(gate|up|down).weight
                               -> layers.{i}.W_gu_T / W_down_T (fused, transposed)

Legacy q_proj/k_proj/v_proj checkpoints are concatenated into qkv_w_T.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def build_ms_dict(ckpt: dict, cfg) -> dict[str, np.ndarray]:
    """Convert a torch checkpoint dict (keys/values) to MS-layout numpy arrays."""
    sd = ckpt["model"]
    n_layers = cfg.get("n_layers", 12)
    n_experts = cfg.get("n_experts", 16)
    expert_hidden = cfg.get("expert_hidden", 320)
    out: dict[str, np.ndarray] = {}
    t = lambda x: x.detach().cpu().numpy().astype(np.float32)

    # old-format compat: q_proj/k_proj/v_proj -> qkv_proj
    def get_attn_w(i):
        prefix = f"layers.{i}.attn."
        key = prefix + "qkv_proj.weight"
        if key in sd:
            return t(sd[key])
        q = t(sd[prefix + "q_proj.weight"])
        k = t(sd[prefix + "k_proj.weight"])
        v = t(sd[prefix + "v_proj.weight"])
        return np.concatenate([q, k, v], axis=0)

    out["emb_w"] = t(sd["token_embedding.weight"])
    out["norm.weight"] = t(sd["norm.weight"])
    out["router.router_w"] = t(sd["router.proj.weight"]).T
    for i in range(n_layers):
        p = f"layers.{i}."
        out[p + "norm1.weight"] = t(sd[p + "norm1.weight"])
        out[p + "norm2.weight"] = t(sd[p + "norm2.weight"])
        out[p + "attn.qkv_w_T"] = get_attn_w(i).T
        out[p + "attn.o_w_T"] = t(sd[p + "attn.o_proj.weight"]).T
        # fused expert weights
        w_gu = np.concatenate(
            [np.concatenate([t(sd[f"{p}experts.{j}.mlp.gate.weight"]),
                             t(sd[f"{p}experts.{j}.mlp.up.weight"])], axis=0)
             for j in range(n_experts)], axis=0)      # [2*E*H, D]
        w_down = np.concatenate(
            [t(sd[f"{p}experts.{j}.mlp.down.weight"]) for j in range(n_experts)],
            axis=1)                                    # [D, E*H]
        out[p + "W_gu_T"] = w_gu.T
        out[p + "W_down_T"] = w_down
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=None, help="torch .pt checkpoint")
    ap.add_argument("--out", default="runs/ms")
    ap.add_argument("--fixture", action="store_true",
                    help="generate a small verification fixture from a fresh "
                         "torch model instead of converting a checkpoint")
    args = ap.parse_args()

    if args.fixture:
        from model.moe_torch import MoETransformer
        from model.config import Config
        cfg = Config()
        cfg.n_layers = 2
        cfg.d_model = 64
        cfg.n_heads = 4
        cfg.n_kv_heads = 2
        cfg.head_dim = 16
        cfg.n_experts = 4
        cfg.top_k = 2
        cfg.expert_hidden = 32
        cfg.vocab_size = 128
        cfg.max_seq_len = 32
        cfg.sparse_moe = True
        torch.manual_seed(42)
        model = MoETransformer(cfg).eval()
        B, T = 2, 16
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            logits, losses = model(ids, targets)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        msd = build_ms_dict({"model": model.state_dict()}, cfg.__dict__)
        np.savez(out_dir / "fixture.npz",
                 x_ids=ids.numpy().astype(np.int32),
                 targets=targets.numpy().astype(np.int32),
                 logits=logits.numpy().astype(np.float32),
                 z_loss=np.float32(losses["router_z_loss"].item()),
                 aux_loss=np.float32(losses["router_aux_loss"].item()),
                 **msd)
        (out_dir / "fixture_cfg.json").write_text(
            json.dumps(cfg.__dict__, default=str), encoding="utf-8")
        print(f"fixture saved -> {out_dir / 'fixture.npz'}")
        return

    if not args.pt:
        ap.error("--pt or --fixture required")
    ckpt = torch.load(args.pt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    msd = build_ms_dict(ckpt, cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "ckpt.npz", **msd)
    meta = {"step": ckpt.get("step", 0), "loss": ckpt.get("loss"),
            "cfg": cfg, "n_params": len(msd),
            "source": str(args.pt)}
    (out_dir / "meta.json").write_text(json.dumps(meta, default=str, indent=2),
                                       encoding="utf-8")
    nbytes = sum(a.nbytes for a in msd.values())
    print(f"converted {len(msd)} params ({nbytes/1e6:.1f} MB) -> {out_dir} "
          f"(step={meta['step']})")


if __name__ == "__main__":
    main()