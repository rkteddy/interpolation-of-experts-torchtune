"""Compatibility shim.

Some users expect a `recipes/train.py` entrypoint.

This repo now focuses on **from-scratch pretraining** (no HF weight load)
so the primary recipe is:

    `recipes/moe_pretrain_distributed.py`

Run:
    tune run --nproc_per_node 16 recipes/train.py --config <cfg.yaml>
"""

from moe_pretrain_distributed import main


if __name__ == "__main__":
    main()
