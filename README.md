# interpolation-of-experts-torchtune

End-to-end, ROCm-friendly **Interpolation-of-Experts / Mixture-of-Experts** (IoE/MoE) validation stack built on **native PyTorch + torchtune**.

This repo is intentionally **modular and hackable**:

* `ioe_torchtune/moe/moe_layer.py`: `MoELayer` (Top-2 gating placeholder you can swap with IoE)
* `ioe_torchtune/models/qwen2_5_moe.py`: torchtune-compatible model builder (Qwen2.5-0.5B arch, MLP -> MoE)
* `recipes/moe_pretrain_distributed.py`: custom **from-scratch pretraining** recipe that logs/optimizes MoE losses
* `ioe_torchtune/datasets/pile_streaming.py`: streaming **The Pile** dataset (IterableDataset, rank/worker sharding)
* `scripts/launch_mi250x_fsdp_pretrain.sh`: MI250X-specific launch (16 ranks for 8x MI250X dual-GCD)
* `configs/qwen2_5_0_5b_moe_pile_pretrain_fsdp.yaml`: quick-start config (The Pile)

## Why this works on AMD ROCm

* Uses only native PyTorch ops (no bitsandbytes, no flash-attn-2).
* torchtune attention uses PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`) under the hood.
* Distributed sharding uses torchtune's FSDP2 wrapper (`torchtune.training.shard_model` -> composable `fully_shard`).

## Important change: **pretraining from scratch**

Because the MLP is replaced by a new IoE/MoE block, we **do not** load pretrained weights.

* The model is randomly initialized.
* We pretrain on **The Pile** using a standard next-token objective.
* We still use Qwen2.5's tokenizer files for convenience.

If you later decide to warm-start from a dense checkpoint, you can do so externally by loading with `strict=False` and/or seeding experts — but this repo defaults to from-scratch training.

## Prerequisites

* PyTorch built for ROCm (e.g. `torch==2.6+rocm` or newer)
* `torchtune` installed (latest)
* A node with **8x MI250X** (ROCm exposes **16 devices** = 2 GCD per MI250X)

Python deps:

```bash
pip install -r requirements.txt
```

## Quick start (pretrain on The Pile)

1) Get tokenizer files for Qwen2.5-0.5B-Instruct.

If you already have the model repo downloaded, just point `tokenizer_dir` to it.
Otherwise, one simple approach is:

```bash
# This may download more than just tokenizer files depending on your setup.
tune download Qwen/Qwen2.5-0.5B-Instruct --output-dir /data/tokenizers/Qwen2.5-0.5B-Instruct
```

2) Edit the config:

```yaml
# configs/qwen2_5_0_5b_moe_pile_pretrain_fsdp.yaml
tokenizer_dir: /data/tokenizers/Qwen2.5-0.5B-Instruct
output_dir: /data/outputs/qwen2_5_0_5b_moe_pile_pretrain
```

3) Launch on MI250X (16 ranks):

```bash
./scripts/launch_mi250x_fsdp_pretrain.sh configs/qwen2_5_0_5b_moe_pile_pretrain_fsdp.yaml
```

4) Watch TensorBoard:

```bash
tensorboard --logdir /data/outputs/qwen2_5_0_5b_moe_pile_pretrain/tb
```

## Where to hack in your new IoE router

Replace the Top-2 gating code in:

* `ioe_torchtune/moe/moe_layer.py` (`MoELayer.forward`)

The module contract is:

```python
out, router_loss, aux_loss = moe(x)
```

The recipe uses:

```python
total_loss = ce_loss + load_balancing_loss_weight * aux_loss + router_loss
```

`router_loss` and `aux_loss` are logged separately so you can monitor router stability and load-balancing behavior.

## Notes on The Pile streaming

The Pile HF loader currently requires `trust_remote_code=True`. The launch script sets:

```bash
export HF_DATASETS_TRUST_REMOTE_CODE=1
```

For large, long-running pretraining you should strongly consider offline pre-tokenization and packing, but streaming is great for first validation runs.
