import math
from typing import Tuple

import torch
import torch.nn as nn

# ---- DEEPSPEED_SHIELD ----
# Avoid broken deepspeed import path on some Py3.12 stacks (distutils removed).
import os as _os
import sys as _sys
import types as _types
import importlib.util as _importlib_util
import importlib.machinery as _machinery

if _os.environ.get("GRIDMOE_DISABLE_DEEPSPEED", "1") == "1":
    for _k in list(_sys.modules.keys()):
        if _k == "deepspeed" or _k.startswith("deepspeed."):
            _sys.modules.pop(_k, None)

    _orig_find_spec = _importlib_util.find_spec

    def _find_spec_no_deepspeed(name, package=None):
        if name == "deepspeed" or name.startswith("deepspeed."):
            return None
        return _orig_find_spec(name, package)

    _importlib_util.find_spec = _find_spec_no_deepspeed

    _ds = _types.ModuleType("deepspeed")
    _ds.__dict__["__version__"] = "0.0.0"
    _ds.__path__ = []
    _ds.__spec__ = _machinery.ModuleSpec("deepspeed", loader=None, is_package=True)
    _sys.modules["deepspeed"] = _ds

    for _sub in ["ops", "comm", "runtime", "utils"]:
        _fullname = f"deepspeed.{_sub}"
        _m = _types.ModuleType(_fullname)
        _m.__path__ = []
        _m.__spec__ = _machinery.ModuleSpec(_fullname, loader=None, is_package=True)
        _sys.modules[_fullname] = _m
        setattr(_ds, _sub, _m)

try:
    from transformers import LlamaForCausalLM
except Exception as e:
    raise RuntimeError(
        "Failed to import transformers LlamaForCausalLM. "
        "Often caused by broken deepspeed install on Py3.12 (missing distutils)."
    ) from e

try:
    from .configuration_gridmoe_llama import GridMoELlamaConfig
except Exception:
    from configuration_gridmoe_llama import GridMoELlamaConfig


class GridInterpolateRouter(nn.Module):
    """
    Stable router for ROCm:
      - no bool scatter
      - return top_idx/top_w (float32 weights)
      - keep routing math in float32
      - clamp indices for safety
    """
    def __init__(self, config: GridMoELlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.eps = 1e-6

        grid_shape = list(getattr(config, "moe_grid_shape", None) or [])
        if len(grid_shape) == 0:
            raise ValueError("config.moe_grid_shape is required for GridInterpolateRouter")

        self.d = len(grid_shape)
        assert self.d >= 1
        assert all(int(x) >= 2 for x in grid_shape), "Each grid dimension must be >= 2."

        self.num_ve = 1 << self.d  # 2^d

        self.num_anchors = int(getattr(config, "moe_router_num_anchors", 4))
        assert self.num_anchors > 0

        self.k = int(getattr(config, "moe_router_topk"))
        assert self.k >= 1
        assert self.k == self.num_anchors * self.num_ve, (
            f"Requirement: moe_router_topk (K) == num_anchors (M) * 2^d. "
            f"Got K={self.k}, M={self.num_anchors}, 2^d={self.num_ve}, d={self.d}."
        )

        self.num_experts = int(getattr(config, "num_experts"))
        prod_shape = int(math.prod(int(x) for x in grid_shape))
        assert prod_shape == self.num_experts, (
            f"Product(moe_grid_shape)={prod_shape} must equal num_experts={self.num_experts}."
        )

        self.hidden_size = int(getattr(config, "hidden_size"))
        out_dim = self.num_anchors * (self.d + 1)
        self.coord_proj = nn.Linear(self.hidden_size, out_dim, bias=True)

        # Buffers (will move with model.to(device))
        self.register_buffer("grid_shape", torch.tensor(grid_shape, dtype=torch.long), persistent=False)

        bits = []
        for t in range(self.num_ve):
            bits.append([(t >> j) & 1 for j in range(self.d)])
        self.register_buffer("ve_bits", torch.tensor(bits, dtype=torch.long), persistent=False)

        strides = [1]
        for j in range(self.d - 1):
            strides.append(strides[-1] * int(grid_shape[j]))
        self.register_buffer("grid_strides", torch.tensor(strides, dtype=torch.long), persistent=False)

        # Meta-safe init: only init when params have real storage.
        if bool(getattr(config, "moe_router_anchor_spread_init", True)):
            if (self.coord_proj.bias is not None) and (not self.coord_proj.bias.is_meta):
                with torch.no_grad():
                    b = self.coord_proj.bias.view(self.num_anchors, self.d + 1)
                    b[:, self.d].zero_()

                    s = int(math.ceil(self.num_anchors ** (1.0 / self.d)))
                    vals = torch.linspace(0.2, 0.8, steps=s, dtype=b.dtype, device=b.device)
                    grid = torch.cartesian_prod(*([vals] * self.d))
                    u0 = grid[: self.num_anchors].clamp(self.eps, 1.0 - self.eps)

                    logit_u0 = torch.log(u0 / (1.0 - u0))
                    b[:, : self.d].copy_(logit_u0)

                    W = self.coord_proj.weight.view(self.num_anchors, self.d + 1, self.hidden_size)
                    W[:, self.d, :].zero_()

        self.debug_check = bool(getattr(config, "moe_router_debug_check", False))

    def gating(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 3:
            hidden = hidden.reshape(-1, hidden.size(-1))
        # keep proj in its dtype
        hidden = hidden.to(dtype=self.coord_proj.weight.dtype)
        return self.coord_proj(hidden)

    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"Expected logits [N, D], got {tuple(logits.shape)}")

        N = logits.size(0)
        device = logits.device

        expected = self.num_anchors * (self.d + 1)
        if logits.size(1) != expected:
            raise ValueError(f"Expected logits dim={expected}, got {logits.size(1)}")

        # float32 math for stability / ROCm safety
        pack = logits.view(N, self.num_anchors, self.d + 1).to(torch.float32)
        coord_logits = pack[:, :, : self.d]
        anchor_logits = pack[:, :, self.d]
        anchor_pi = torch.softmax(anchor_logits, dim=1)  # [N, M]

        m = self.grid_shape
        b = self.ve_bits
        strides = self.grid_strides
        if m.device != device:
            m = m.to(device)
        if b.device != device:
            b = b.to(device)
        if strides.device != device:
            strides = strides.to(device)

        scale = (m - 1).to(torch.float32).view(1, 1, self.d)

        u = torch.sigmoid(coord_logits).clamp(self.eps, 1.0 - self.eps)
        p = u * scale
        p = torch.minimum(p, (scale - 1e-6).clamp_min(0.0))
        p = torch.maximum(p, torch.zeros_like(p))

        a = torch.floor(p).to(torch.long)
        a_max = (m - 2).clamp_min(0).view(1, 1, self.d)
        a = torch.minimum(a, a_max)
        a = torch.maximum(a, torch.zeros_like(a))

        f = (p - a.to(torch.float32)).clamp(self.eps, 1.0 - self.eps)

        b4 = b.view(1, 1, self.num_ve, self.d)
        coords = a[:, :, None, :] + b4  # [N, M, 2^d, d]

        # safety clamp (avoid any numerical edge making coords out of bounds)
        coords = torch.maximum(coords, torch.zeros_like(coords))
        coords = torch.minimum(coords, (m - 1).view(1, 1, 1, self.d))

        strides4 = strides.view(1, 1, 1, self.d)
        idx = (coords * strides4).sum(-1)  # [N, M, 2^d]
        idx = idx.clamp_(0, self.num_experts - 1)

        fj = f[:, :, None, :]
        w = torch.where(b4.bool(), fj, 1.0 - fj).prod(-1)  # [N, M, 2^d]
        w = w / (w.sum(-1, keepdim=True) + 1e-9)
        w = w * anchor_pi[:, :, None]

        K = self.k
        idx_all = idx.reshape(N, K)
        w_all = w.reshape(N, K)

        probs = torch.zeros((N, self.num_experts), device=device, dtype=torch.float32)
        probs.scatter_add_(1, idx_all, w_all)
        probs = probs.clamp_min(0.0)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-9)

        # return top-k directly (values float32)
        top_w, top_idx = torch.topk(probs, k=K, dim=1)

        if self.debug_check:
            if top_idx.min() < 0 or top_idx.max() >= self.num_experts:
                raise RuntimeError(
                    f"Router produced out-of-range expert id: "
                    f"min={int(top_idx.min())}, max={int(top_idx.max())}, E={self.num_experts}"
                )
            if not torch.isfinite(top_w).all():
                raise RuntimeError("Non-finite routing weights detected.")

        return top_idx, top_w

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gating(hidden)
        return self.routing(logits)


class LlamaMLPExpert(nn.Module):
    def __init__(self, config: GridMoELlamaConfig):
        super().__init__()
        h = int(getattr(config, "hidden_size"))
        expert_inter = int(getattr(config, "expert_intermediate_size", getattr(config, "intermediate_size")))
        bias = bool(getattr(config, "mlp_bias", False))

        self.gate_proj = nn.Linear(h, expert_inter, bias=bias)
        self.up_proj = nn.Linear(h, expert_inter, bias=bias)
        self.down_proj = nn.Linear(expert_inter, h, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GridMoELlamaMoEBlock(nn.Module):
    """
    ROCm safer MoE:
      - router gives top_idx/top_w (float32)
      - accumulation in float32
    """
    def __init__(self, config: GridMoELlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.num_experts = int(getattr(config, "num_experts"))
        self.topk = int(getattr(config, "moe_router_topk"))

        self.router = GridInterpolateRouter(config)
        self.experts = nn.ModuleList([LlamaMLPExpert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        flat = hidden_states.reshape(-1, H)  # [N, H]
        N = flat.size(0)

        top_idx, top_w = self.router(flat)  # [N, K], [N, K] (float32 weights)

        # accumulate in fp32 to avoid ROCm bf16 index_add weirdness
        out = torch.zeros((N, H), device=flat.device, dtype=torch.float32)

        for e in range(self.num_experts):
            mask = top_idx.eq(e)  # [N, K]
            if not mask.any():
                continue
            token_ids, slot_ids = mask.nonzero(as_tuple=True)
            x_e = flat.index_select(0, token_ids)
            y_e = self.experts[e](x_e)  # dtype ~ model dtype (bf16/fp16)
            w_e = top_w[token_ids, slot_ids].unsqueeze(1)  # float32
            out.index_add_(0, token_ids, y_e.to(torch.float32) * w_e)

        return out.to(dtype=hidden_states.dtype).view(B, S, H)


class GridMoELlamaForCausalLM(LlamaForCausalLM):
    config_class = GridMoELlamaConfig

    def __init__(self, config: GridMoELlamaConfig):
        super().__init__(config)

        for layer in self.model.layers:
            layer.mlp = GridMoELlamaMoEBlock(config)

        self.post_init()
