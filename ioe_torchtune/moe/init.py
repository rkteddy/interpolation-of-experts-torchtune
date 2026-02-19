"""Utilities for initializing MoE weights.

Goal:
- Load the official *dense* Qwen2.5 checkpoint (HF safetensors)
- Replace each dense FeedForward (w1/w2/w3) with a MoELayer
- Initialize each expert from the corresponding dense weights

This keeps the model "mostly pretrained" while you experiment with
new routing / expert architectures.

Important:
- Call these init helpers **before sharding with FSDP** when possible.
  Copying into sharded DTensor parameters is doable but more complex.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


def _maybe_get(sd: Dict[str, Any], keys: list[str]) -> Optional[torch.Tensor]:
    """Try multiple possible state-dict keys and return the first match."""

    for k in keys:
        v = sd.get(k, None)
        if v is not None:
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"State dict value for {k} is not a Tensor: {type(v)}")
            return v
    return None


def init_moe_from_dense_state_dict(
    model: nn.Module,
    dense_state_dict: Dict[str, Any],
    *,
    num_experts: int,
    layer_prefixes: tuple[str, ...] = ("layers", "decoder.layers"),
    verbose: bool = True,
) -> None:
    """Initialize MoE experts from dense (pretrained) MLP weights.

    We look for dense keys of the form:
        {prefix}.{layer_idx}.mlp.w1.weight
        {prefix}.{layer_idx}.mlp.w2.weight
        {prefix}.{layer_idx}.mlp.w3.weight

    And copy them into each expert:
        {prefix}.{layer_idx}.mlp.experts.{e}.w1.weight
        ...

    Args:
        model: MoE model instance whose layers contain MoELayer modules.
        dense_state_dict: Full dense checkpoint state dict (from HF checkpointer).
        num_experts: Number of experts per MoE layer.
        layer_prefixes: Prefixes to try for layer naming.
            torchtune models typically use "layers.{i}.*".
        verbose: Print a short summary.

    Raises:
        RuntimeError if no layers were initialized (usually means key mismatch).
    """

    # Safety: copying into DTensors is non-trivial; warn early.
    any_dtensor = any(hasattr(p, "_local_tensor") for p in model.parameters())
    if any_dtensor:
        raise RuntimeError(
            "Detected sharded/DTensor parameters on the model. "
            "Please call init_moe_from_dense_state_dict() before FSDP sharding."
        )

    num_layers_inited = 0

    # Walk through the model modules and find MoE layers.
    for name, module in model.named_modules():
        # We identify MoE layers by the presence of `experts` and `router`.
        if not (hasattr(module, "experts") and hasattr(module, "router")):
            continue

        # Try to infer layer index from module name.
        # Examples:
        #   layers.0.mlp
        #   decoder.layers.12.mlp
        parts = name.split(".")
        layer_idx: Optional[int] = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is None:
            continue

        # Find dense weights for this layer.
        w1 = w2 = w3 = None
        for prefix in layer_prefixes:
            w1 = _maybe_get(dense_state_dict, [f"{prefix}.{layer_idx}.mlp.w1.weight"])
            w2 = _maybe_get(dense_state_dict, [f"{prefix}.{layer_idx}.mlp.w2.weight"])
            w3 = _maybe_get(dense_state_dict, [f"{prefix}.{layer_idx}.mlp.w3.weight"])
            if w1 is not None and w2 is not None and w3 is not None:
                break

        if w1 is None or w2 is None or w3 is None:
            # Not all checkpoints name identically; skip if not found.
            continue

        # Copy dense expert weights into every expert.
        # (A more sophisticated init could add small noise per expert.)
        with torch.no_grad():
            for e in range(num_experts):
                module.experts[e].w1.weight.copy_(w1.to(module.experts[e].w1.weight.dtype))
                module.experts[e].w2.weight.copy_(w2.to(module.experts[e].w2.weight.dtype))
                module.experts[e].w3.weight.copy_(w3.to(module.experts[e].w3.weight.dtype))

            # Initialize router to be near-uniform initially.
            # Small weights -> logits near 0 -> softmax near uniform.
            nn.init.normal_(module.router.weight, mean=0.0, std=1e-3)

        num_layers_inited += 1

    if verbose:
        print(f"[init] Initialized MoE experts for {num_layers_inited} layers")

    if num_layers_inited == 0:
        raise RuntimeError(
            "Failed to initialize any MoE layers from the dense state dict. "
            "This usually means the state-dict key names don't match the model. "
            "Inspect the dense checkpoint keys and update layer_prefixes / patterns."
        )
