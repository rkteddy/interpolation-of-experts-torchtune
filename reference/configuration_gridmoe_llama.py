from __future__ import annotations

from typing import Sequence, Union

try:
    from transformers import LlamaConfig
except Exception as e:
    raise RuntimeError("Please install transformers: pip install transformers") from e


class GridMoELlamaConfig(LlamaConfig):
    """LlamaConfig + GridMoE-specific fields.

    Notes on semantics (as used by the matching modeling_gridmoe_llama.py in this package):

    - config.num_experts is the TOTAL number of experts (e.g., 64 for an 8x8 grid).
    - config.moe_router_topk is the TOTAL top-k selected among all experts.

    We also keep per-grain metadata (moe_granularity, num_experts_per_grain, topk_per_grain)
    because Megatron checkpoints often express MoE hyperparams that way.
    """

    model_type = "gridmoe_llama"

    def __init__(
        self,
        # MoE granularity metadata
        moe_granularity: int = 1,
        num_experts_per_grain: int = 1,
        topk_per_grain: int = 1,

        # Grid router
        moe_grid_shape: Union[Sequence[int], str] = (1, 1),
        moe_router_num_anchors: int = 4,
        moe_router_anchor_spread_init: bool = True,
        moe_router_debug_check: bool = False,

        # Optional explicit totals (if provided, they override derived values)
        num_experts: int | None = None,
        moe_router_topk: int | None = None,
        expert_intermediate_size: int | None = None,

        # pass-through to LlamaConfig
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.moe_granularity = int(moe_granularity)
        self.num_experts_per_grain = int(num_experts_per_grain)
        self.topk_per_grain = int(topk_per_grain)

        if isinstance(moe_grid_shape, str):
            moe_grid_shape = [int(x) for x in moe_grid_shape.split(",") if x.strip()]
        self.moe_grid_shape = list(moe_grid_shape)

        # Derived totals (can be overridden below)
        derived_num_experts = 1
        for d in self.moe_grid_shape:
            derived_num_experts *= int(d)

        if num_experts is None:
            # Fallback if grid_shape is not meaningful
            if derived_num_experts <= 0 or derived_num_experts == 1 and self.moe_grid_shape == [1, 1]:
                derived_num_experts = int(self.num_experts_per_grain * max(self.moe_granularity, 1))
            self.num_experts = int(derived_num_experts)
        else:
            self.num_experts = int(num_experts)

        # IMPORTANT: In this HF implementation we interpret `moe_router_topk` as the
        # **TOTAL** top-k among all experts.
        #
        # In the original Megatron/ReMoE configs, a common convention is:
        #   - `topk_per_grain` is per-granularity-group top-k
        #   - total_topk = topk_per_grain * moe_granularity
        #
        # To be robust, if the caller does not explicitly provide `moe_router_topk`, we
        # derive it from (topk_per_grain, moe_granularity).
        if moe_router_topk is None:
            self.moe_router_topk = int(self.topk_per_grain) * int(self.moe_granularity)
        else:
            self.moe_router_topk = int(moe_router_topk)

        self.moe_router_num_anchors = int(moe_router_num_anchors)
        self.moe_router_anchor_spread_init = bool(moe_router_anchor_spread_init)
        self.moe_router_debug_check = bool(moe_router_debug_check)

        # Convenience: expert intermediate dim (if using smaller experts)
        if expert_intermediate_size is not None:
            self.expert_intermediate_size = int(expert_intermediate_size)
        elif getattr(self, "intermediate_size", None) is not None:
            # default heuristic
            self.expert_intermediate_size = int(self.intermediate_size // max(self.moe_granularity, 1))
