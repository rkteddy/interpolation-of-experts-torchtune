"""Minimal torchtune-compatible Transformer wrappers that propagate MoE losses.

Why this exists:
- torchtune's built-in TransformerSelfAttentionLayer expects its MLP to return
  a tensor. Our MoELayer returns (out, router_loss, aux_loss).

So we provide:
- MoETransformerSelfAttentionLayer: identical to torchtune's, but expects an
  MoE MLP and returns (hidden, router_loss, aux_loss).
- MoETransformerDecoder: iterates layers and accumulates MoE losses, returning
  (logits, router_loss, aux_loss).

This keeps everything else (attention, embeddings, norms, output head) as
standard torchtune modules, which is important for compatibility and ROCm.

Attention on ROCm:
- torchtune's MultiHeadAttention uses native PyTorch SDPA (scaled_dot_product_attention)
  by default. That is ROCm-friendly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from torchtune.modules import TransformerDecoder, TransformerSelfAttentionLayer


class MoETransformerSelfAttentionLayer(TransformerSelfAttentionLayer):
    """Transformer block that expects an MoE MLP and returns MoE losses."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Self-attention (same as torchtune)
        h = self.sa_norm(x)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos, **kwargs)
        h = self.sa_scale(attn_out) + x

        # MoE MLP (differs: expects a tuple)
        mlp_in = self.mlp_norm(h)
        mlp_out, router_loss, aux_loss = self.mlp(mlp_in)
        out = h + self.mlp_scale(mlp_out)

        return out, router_loss, aux_loss


class MoETransformerDecoder(TransformerDecoder):
    """TransformerDecoder that returns (logits, router_loss, aux_loss)."""

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.tok_embeddings(tokens)

        router_loss_total = torch.zeros((), device=x.device, dtype=torch.float32)
        aux_loss_total = torch.zeros((), device=x.device, dtype=torch.float32)

        for layer in self.layers:
            x, router_loss, aux_loss = layer(x, mask=mask, input_pos=input_pos, **kwargs)
            router_loss_total = router_loss_total + router_loss.float()
            aux_loss_total = aux_loss_total + aux_loss.float()

        x = self.norm(x)
        logits = self.output(x)
        return logits, router_loss_total, aux_loss_total
