"""Model configuration for the 200M-A25M MoE (16 experts, top-2).

Parameter math (vocab 50K, d_model 768, 12 layers, expert_hidden 320):
  embedding (tied, counted once):  50_000 x 768            =  38.4M
  attention / layer:  q 768x768 + kv 768x256 + o 768x768   =   1.57M
  moe / layer:        16 x (gate 768x320 + up 768x320
                           + down 320x768)                 =  11.80M
  ---------------------------------------------------------------
  total                                  38.4 + 12 x 13.37 = 198.8M
  active / token (incl. attention):    12 x (1.57 + 1.48)  =  36.6M
  active FFN-only (top-2 experts):     12 x 1.48M          =  17.7M
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- data / vocab -------------------------------------------------
    vocab_size: int = 50000
    max_seq_len: int = 2048
    pad_id: int = 5            # "<pad>" in the retrained tokenizer
    im_start_id: int = 1
    im_end_id: int = 2
    tool_call_id: int = 3
    tool_result_id: int = 4

    # --- transformer ---------------------------------------------------
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4        # GQA
    head_dim: int = 64
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None   # e.g. {"factor": 4.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0}
    norm_eps: float = 1e-6
    mlp_hidden: int = 0        # shared dense FFN per layer; 0 = pure MoE (attention + experts only)
    dropout: float = 0.0

    # --- MoE ------------------------------------------------------------
    n_experts: int = 16
    top_k: int = 2
    expert_hidden: int = 320   # per-expert SwiGLU intermediate
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.01
    # 0 -> no shared expert; >0 -> an extra always-on expert (DeepSeek-style)
    n_shared_experts: int = 0
    shared_expert_hidden: int = 512

    # --- training -------------------------------------------------------
    gradient_checkpointing: bool = False
    # 是否启用 FlashAttention（CUDA flash-attn / Ascend npu_fusion_attention）。
    # 通过训练脚本的 --flash-attention 参数控制，不再使用环境变量。
    use_flash_attn: bool = False
    # npu_fusion_attention 的输入布局："bnsd"（[B,N,S,D]）或 "bsh"（[B,S,H]）。
    # 部分 CANN 版本只对其中一种布局提供 kernel，可在此切换。
    fa_layout: str = "bnsd"
    # sparse MoE：只对 router 选中的 top-k 专家做计算（FLOPs 约为全专家激活的
    # k/E），默认开启以提速。通过训练脚本 --sparse-moe 0/1 显式控制（默认
    # 开启；NPU 上若 index_add 算子不稳定，传 --sparse-moe 0 回退到
    # "全专家计算 + top-k 加权" 的 dense 模式）。
    sparse_moe: bool = True

    @classmethod
    def from_name(cls, name: str) -> "Config":
        cfg = cls()
        # minimal presets for later scaling; override via YAML/CLI if needed
        if name == "moe-200m":
            pass
        elif name == "moe-350m":
            cfg.n_layers = 16
            cfg.d_model = 1024
            cfg.n_heads = 16
            cfg.n_kv_heads = 4
            cfg.expert_hidden = 512
        else:
            raise ValueError(f"unknown config name: {name}")
        return cfg

    def num_parameters(self) -> dict:
        """Rough parameter count (embedding counted once, tied to lm_head)."""
        d = self.d_model
        emb = self.vocab_size * d
        attn_per = d * d * (1 + 1) + d * self.n_kv_heads * self.head_dim * 2  # q + o, kv
        shared_mlp_per = 3 * d * self.mlp_hidden if self.mlp_hidden else 0
        moe_per = self.n_experts * 3 * d * self.expert_hidden
        shared_expert_per = self.n_shared_experts * 3 * d * self.shared_expert_hidden
        per_layer = attn_per + shared_mlp_per + moe_per + shared_expert_per
        active_per = (attn_per + shared_mlp_per
                      + self.top_k * 3 * d * self.expert_hidden
                      + self.n_shared_experts * 3 * d * self.shared_expert_hidden)
        total = emb + per_layer * self.n_layers
        active = active_per * self.n_layers
        return {"total": total, "active": active, "embedding": emb,
                "per_layer": per_layer, "active_per_layer": active_per}


def save_yaml(cfg: Config, path: str | Path) -> None:
    import yaml
    Path(path).write_text(yaml.safe_dump(cfg.__dict__), encoding="utf-8")
