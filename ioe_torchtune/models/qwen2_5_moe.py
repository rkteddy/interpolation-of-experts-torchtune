"""torchtune model builder: Qwen2.5-0.5B architecture with MoE/IoE MLPs.

This file provides a torchtune-compatible builder function:

    qwen2_5_0_5b_moe(...)

It is designed to be referenced from a torchtune YAML config via:

    model:
      _component_: ioe_torchtune.models.qwen2_5_moe.qwen2_5_0_5b_moe
      num_experts: 8
      ...

Design goals:
* Reuse torchtune's Qwen2.5 implementation for everything except the MLP.
* Swap every dense FeedForward with MoELayer (Top-2 gating placeholder).

Pretraining note:
* This builder does NOT load any HF checkpoint weights.
  It returns a randomly initialized model suitable for from-scratch pretraining.
"""

from __future__ import annotations

import inspect
from typing import Optional

from torch import nn

from torchtune.models.qwen2_5 import qwen2_5_0_5b

from ioe_torchtune.moe import MoEConfig, MoELayer
from ioe_torchtune.models.moe_transformer import (
    MoETransformerDecoder,
    MoETransformerSelfAttentionLayer,
)


def qwen2_5_0_5b_moe(
    *,
    num_experts: int = 8,
    top_k: int = 2,
    capacity_factor: Optional[float] = None,
    router_z_loss_coef: float = 0.0,
    expert_dropout: float = 0.0,
) -> nn.Module:
    """Build Qwen2.5-0.5B with MoE MLP blocks.

    Args:
        num_experts: Experts per layer.
        top_k: Top-k gating (placeholder supports only top_k=2).
        capacity_factor: Optional capacity factor.
        router_z_loss_coef: Coef for router z-loss (stabilizer).
        expert_dropout: Optional dropout within each expert MLP.

    Returns:
        A MoETransformerDecoder that returns (logits, router_loss, aux_loss).
    """
    # 1) Build the *dense* torchtune Qwen2.5 0.5B model.
    dense = qwen2_5_0_5b()

    # 2) Detect whether the underlying torchtune TransformerSelfAttentionLayer supports `mask_mod`.
    #    We avoid importing TransformerSelfAttentionLayer directly (API path/version drift).
    #    MoETransformerSelfAttentionLayer inherits it, so use MRO to fetch the base class.
    base_sa_cls = MoETransformerSelfAttentionLayer.__mro__[1]
    sa_init_params = inspect.signature(base_sa_cls.__init__).parameters
    supports_mask_mod = "mask_mod" in sa_init_params

    # 3) Replace each layer with a MoE-aware layer.
    moe_layers = nn.ModuleList()

    for layer in dense.layers:
        # layer.mlp is torchtune.modules.feed_forward.FeedForward
        d_model = layer.mlp.w1.in_features
        d_ff = layer.mlp.w1.out_features

        moe_cfg = MoEConfig(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            router_z_loss_coef=router_z_loss_coef,
            expert_dropout=expert_dropout,
        )
        moe_mlp = MoELayer(moe_cfg)

        kwargs = dict(
            attn=layer.attn,
            mlp=moe_mlp,
            sa_norm=layer.sa_norm,
            mlp_norm=layer.mlp_norm,
            sa_scale=layer.sa_scale,
            mlp_scale=layer.mlp_scale,
        )
        # Only pass mask_mod if this torchtune version supports it and the layer actually has it.
        if supports_mask_mod and hasattr(layer, "mask_mod"):
            kwargs["mask_mod"] = getattr(layer, "mask_mod")

        moe_layer = MoETransformerSelfAttentionLayer(**kwargs)
        moe_layers.append(moe_layer)

    # 4) Wrap everything in a MoETransformerDecoder.
    #    Re-use dense embeddings/norm/output so overall structure stays torchtune-compatible.
    moe_model = MoETransformerDecoder(
        tok_embeddings=dense.tok_embeddings,
        layers=moe_layers,
        max_seq_len=dense.max_seq_len,
        num_heads=dense.num_heads,
        head_dim=dense.head_dim,
        norm=dense.norm,
        output=dense.output,
        num_layers=getattr(dense, "num_layers", None),
        output_hidden_states=getattr(dense, "output_hidden_states", None),
    )

    return moe_model
