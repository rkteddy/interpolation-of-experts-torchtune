"""Datasets for IoE/MoE experiments.

torchtune ships a number of SFT datasets and a *map-style* text-completion dataset.
For real **from-scratch pretraining** (e.g. The Pile), streaming is often required
to avoid downloading 100s of GB before a first smoke test.

This package provides a small, hackable streaming dataset that:

* uses Hugging Face `datasets` streaming mode
* shards deterministically across distributed ranks *and* DataLoader workers
* packs raw text into fixed-length token blocks for causal LM pretraining
"""

from .pile_streaming import pile_streaming_dataset

__all__ = ["pile_streaming_dataset"]
