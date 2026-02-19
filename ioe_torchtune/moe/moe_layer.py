"""Mixture-of-Experts layer implementations.

This file is intentionally self-contained and heavily commented so you can
rapidly swap the routing / expert logic for your new IoE architecture.

Notes for AMD ROCm:
- This code uses only native PyTorch ops (no CUDA-only deps like flash-attn/bnb).
- Avoids custom fused kernels; BF16/FP16 should work on MI250X.

The MoELayer defined here is designed to replace torchtune's FeedForward MLP.

Interface contract:
    out, router_loss, aux_loss = moe(x)

- out: same shape/dtype as x
- router_loss: scalar tensor (regularizer on router logits; placeholder)
- aux_loss: scalar tensor (load-balancing loss; used in total_loss)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MoEConfig:
    """Configuration for MoELayer.

    Args:
        d_model: Model hidden size.
        d_ff: Expert intermediate size (e.g., SwiGLU hidden dim).
        num_experts: Number of experts.
        top_k: Number of experts selected per token. (We use Top-2 by default.)
        capacity_factor: Optional capacity factor (tokens per expert).
            If None, do not enforce capacity (simpler placeholder).
        router_z_loss_coef: Coefficient for the router z-loss.
            z-loss is a stabilizer used in some router formulations.
        expert_dropout: Optional dropout within experts.
    """

    d_model: int
    d_ff: int
    num_experts: int
    top_k: int = 2
    capacity_factor: Optional[float] = None
    router_z_loss_coef: float = 0.0
    expert_dropout: float = 0.0


class ExpertMLP(nn.Module):
    """A single expert MLP matching torchtune's SwiGLU-style FeedForward.

    For Llama/Qwen-style MLP:
        y = W2( silu(W1(x)) * W3(x) )

    We keep bias=False to match most HF Qwen/Llama checkpoints.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [tokens, d_model]
        x1 = F.silu(self.w1(x))
        x3 = self.w3(x)
        h = self.dropout(x1 * x3)
        return self.w2(h)


class MoELayer(nn.Module):
    """A standard Top-K MoE layer (Top-2 by default).

    This layer is written for hackability and correctness first.
    You can later optimize dispatch with grouped GEMMs / token sorting.

    Forward:
        out, router_loss, aux_loss = layer(x)

    Shapes:
        x: [B, S, D] or [T, D]
        out: same as x

    Implementation notes:
    - We route each token independently.
    - We do *not* enforce capacity by default (capacity_factor=None).
      This is fine for small-scale validation runs.
    - We accumulate outputs in fp32 for numerical robustness on BF16.
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        if cfg.top_k != 2:
            raise NotImplementedError(
                "This placeholder implements Top-2 gating. "
                "(Extend easily by generalizing the dispatch loop.)"
            )

        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

        # Experts live in a ModuleList for easy surgery / swapping.
        self.experts = nn.ModuleList(
            [ExpertMLP(cfg.d_model, cfg.d_ff, dropout=cfg.expert_dropout) for _ in range(cfg.num_experts)]
        )

    @staticmethod
    def _router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
        """Router z-loss (stabilizer).

        Used in PaLM / related router designs to keep router logits from
        growing too large. The typical form is:
            z_loss = mean( logsumexp(logits)^2 )

        We keep it as a scalar fp32 tensor.
        """

        z = torch.logsumexp(router_logits.float(), dim=-1)
        return (z * z).mean()

    @staticmethod
    def _load_balancing_loss(
        router_probs: torch.Tensor,
        top1_expert: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """Switch/Top-k style load balancing loss.

        One common formulation (Switch Transformer) is:
            importance = mean(router_probs) over tokens
            load       = fraction of tokens routed to each expert (top1)
            aux_loss   = num_experts * sum(importance * load)

        This encourages both probability mass and actual routing decisions
        to be balanced.
        """

        # router_probs: [T, E]
        T = router_probs.shape[0]

        # importance: [E]
        importance = router_probs.float().sum(dim=0) / max(T, 1)

        # load: [E]
        load = torch.bincount(top1_expert, minlength=num_experts).float() / max(T, 1)

        aux = (importance * load).sum() * num_experts
        return aux

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        if x.dim() == 3:
            bsz, seqlen, d = x.shape
            x_flat = x.reshape(bsz * seqlen, d)
        elif x.dim() == 2:
            x_flat = x
            bsz, seqlen, d = 1, x.shape[0], x.shape[1]
        else:
            raise ValueError(f"MoELayer expected x dim 2 or 3, got shape {x.shape}")

        # -----------------------------
        # 1) Router: compute probabilities
        # -----------------------------
        router_logits = self.router(x_flat)  # [T, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [T, E]

        # -----------------------------
        # 2) Top-2 selection
        # -----------------------------
        top2_probs, top2_idx = torch.topk(router_probs, k=2, dim=-1)  # [T, 2], [T, 2]

        # Normalize the top-2 probs so each token's weights sum to 1.
        top2_probs = top2_probs / (top2_probs.sum(dim=-1, keepdim=True) + 1e-9)

        top1_idx = top2_idx[:, 0]

        # -----------------------------
        # 3) Auxiliary losses
        # -----------------------------
        aux_loss = self._load_balancing_loss(router_probs, top1_idx, self.cfg.num_experts)

        router_z = self._router_z_loss(router_logits)
        router_loss = self.cfg.router_z_loss_coef * router_z

        # -----------------------------
        # 4) Dispatch tokens to experts
        # -----------------------------
        # Accumulate into fp32 for numeric robustness.
        out_fp32 = torch.zeros_like(x_flat, dtype=torch.float32)

        # Optional capacity (simple token dropping). This is just a placeholder.
        # For real training, implement proper capacity + combine via scatter.
        capacity: Optional[int] = None
        if self.cfg.capacity_factor is not None:
            # Capacity per expert = ceil(capacity_factor * T / num_experts)
            T = x_flat.shape[0]
            capacity = int(self.cfg.capacity_factor * (T / self.cfg.num_experts) + 0.999)

        for e, expert in enumerate(self.experts):
            # Tokens where expert e is chosen as top-1
            idx1 = (top2_idx[:, 0] == e).nonzero(as_tuple=False).squeeze(-1)
            # Tokens where expert e is chosen as top-2
            idx2 = (top2_idx[:, 1] == e).nonzero(as_tuple=False).squeeze(-1)

            if capacity is not None:
                # Drop overflow tokens (placeholder). A proper implementation would
                # keep the top-k tokens per expert by router score.
                if idx1.numel() > capacity:
                    idx1 = idx1[:capacity]
                if idx2.numel() > capacity:
                    idx2 = idx2[:capacity]

            if idx1.numel() > 0:
                y1 = expert(x_flat.index_select(0, idx1))  # [N1, D]
                w1 = top2_probs.index_select(0, idx1)[:, 0].unsqueeze(-1)  # [N1, 1]
                out_fp32.index_add_(0, idx1, y1.float() * w1.float())

            if idx2.numel() > 0:
                y2 = expert(x_flat.index_select(0, idx2))  # [N2, D]
                w2 = top2_probs.index_select(0, idx2)[:, 1].unsqueeze(-1)  # [N2, 1]
                out_fp32.index_add_(0, idx2, y2.float() * w2.float())

        out = out_fp32.to(dtype=x_flat.dtype)

        if x.dim() == 3:
            out = out.reshape(bsz, seqlen, d)
        # else already [T, D]

        assert out.shape == orig_shape, f"Output shape {out.shape} != input shape {orig_shape}"

        # router_loss and aux_loss must be scalar tensors on device.
        router_loss = router_loss.to(device=x.device)
        aux_loss = aux_loss.to(device=x.device)

        return out, router_loss, aux_loss
