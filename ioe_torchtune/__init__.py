"""Interpolation-of-Experts (IoE) + torchtune integration.

This repo is intentionally "hackable": most components are plain PyTorch modules
and torchtune-compatible builders/recipes that you can swap out quickly.
"""

from .models.qwen2_5_moe import qwen2_5_0_5b_moe

__all__ = ["qwen2_5_0_5b_moe"]
