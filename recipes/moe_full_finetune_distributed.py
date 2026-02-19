"""Custom torchtune-style recipe for MoE SFT with explicit auxiliary loss handling.

Why this recipe exists:
- torchtune's built-in `full_finetune_distributed` recipe assumes the model
  forward returns logits only.
- For MoE, we need to surface and optimize router / load-balancing losses.

Key requirement from the task:
    total_loss = cross_entropy_loss + load_balancing_loss_weight * aux_loss

We also log router_loss and aux_loss to any torchtune MetricLogger
(DiskLogger, WandBLogger, TensorBoardLogger, etc.).

This recipe intentionally keeps the code "hackable" and avoids CUDA-only libs.

Launch (example for MI250X dual-GCD = 16 ranks):
    tune run --nproc_per_node 16 recipes/moe_full_finetune_distributed.py \
        --config configs/qwen2_5_0_5b_moe_alpaca_cleaned_fsdp.yaml

You can override any YAML field on the command line:
    ... load_balancing_loss_weight=0.01 optimizer.lr=5e-5
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchtune.config import instantiate
from torchtune.training import (
    MODEL_KEY,
    get_shard_conditions,
    shard_model,
)
from torchtune.training.metric_logging import MetricLoggerInterface

from ioe_torchtune.moe import init_moe_from_dense_state_dict


# -------------------------
# Utilities
# -------------------------

def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _rank0_print(*args, **kwargs) -> None:
    if _get_rank() == 0:
        print(*args, **kwargs)


def _init_distributed(device_type: str = "cuda") -> torch.device:
    """Initialize torch.distributed + set the correct device."""

    if _is_distributed():
        # ROCm uses the "nccl" backend name (provided by RCCL).
        dist.init_process_group(backend="nccl")
        local_rank = _get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device(device_type, local_rank)
    else:
        return torch.device(device_type)


def _infer_pad_id(tokenizer: Any) -> int:
    """Best-effort pad id lookup across torchtune tokenizers."""

    for attr in ("pad_id", "pad_token_id"):
        if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not None:
            return int(getattr(tokenizer, attr))

    # Qwen tokenizers often use EOS as pad.
    for attr in ("eos_id", "eos_token_id"):
        if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not None:
            return int(getattr(tokenizer, attr))

    return 0


def _sft_collate(batch: list[Dict[str, Any]], *, pad_id: int, ignore_index: int = -100) -> Dict[str, torch.Tensor]:
    """Simple SFT collate: pad tokens with pad_id and labels with ignore_index."""

    # Convert lists -> tensors.
    tokens_list = []
    labels_list = []
    for ex in batch:
        t = ex["tokens"]
        l = ex["labels"]
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.long)
        if not torch.is_tensor(l):
            l = torch.tensor(l, dtype=torch.long)
        tokens_list.append(t)
        labels_list.append(l)

    max_len = max(t.numel() for t in tokens_list)

    tokens = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), ignore_index, dtype=torch.long)

    for i, (t, l) in enumerate(zip(tokens_list, labels_list)):
        n = t.numel()
        tokens[i, :n] = t
        labels[i, :n] = l

    return {"tokens": tokens, "labels": labels}


def _decide_if_shift_labels(tokens: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> bool:
    """Heuristic to decide whether labels are already next-token aligned.

    Returns:
        True  -> compute loss with shift: logits[:, :-1] vs labels[:, 1:]
        False -> compute loss without shift: logits[:, :-1] vs labels[:, :-1]

    We check whether trainable labels match the *current* tokens or the *next* tokens.
    """

    mask = labels.ne(ignore_index)
    if mask.sum() == 0:
        return True

    direct = (labels[mask] == tokens[mask]).float().mean()

    mask2 = mask[:, :-1]
    if mask2.sum() == 0:
        return True

    next_match = (labels[:, :-1][mask2] == tokens[:, 1:][mask2]).float().mean()

    # If labels match current tokens more than next tokens, assume unshifted.
    return bool((direct >= next_match).item())


def _compute_ce_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    shift_labels: bool,
) -> torch.Tensor:
    """Compute language modeling CE loss, handling optional shifting."""

    # logits: [B, S, V]
    # Always drop the last logit position (no next-token target).
    logits = logits[:, :-1, :].contiguous()

    if shift_labels:
        target = labels[:, 1:].contiguous()
    else:
        target = labels[:, :-1].contiguous()

    vocab = logits.size(-1)
    return F.cross_entropy(
        logits.view(-1, vocab),
        target.view(-1),
        ignore_index=ignore_index,
    )


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    device = _init_distributed(device_type=getattr(cfg, "device", "cuda"))
    rank = _get_rank()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    _rank0_print(f"[dist] world_size={world_size} rank={rank} device={device}")

    # -------------------------
    # Instantiate components
    # -------------------------
    tokenizer = instantiate(cfg.tokenizer)
    pad_id = _infer_pad_id(tokenizer)
    ignore_index = int(getattr(cfg, "ignore_index", -100))

    # Dataset builders in torchtune typically take tokenizer as the first arg.
    try:
        dataset = instantiate(cfg.dataset, tokenizer=tokenizer)
    except TypeError:
        dataset = instantiate(cfg.dataset, tokenizer)

    if _is_distributed():
        sampler = DistributedSampler(dataset, shuffle=bool(getattr(cfg.dataset, "shuffle", True)))
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        sampler=sampler,
        shuffle=(sampler is None and bool(getattr(cfg.dataset, "shuffle", True))),
        num_workers=int(getattr(cfg, "num_workers", 2)),
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: _sft_collate(b, pad_id=pad_id, ignore_index=ignore_index),
    )

    # Metric logger (Disk / W&B / TensorBoard)
    metric_logger: MetricLoggerInterface = instantiate(cfg.metric_logger)
    metric_logger.log_config(cfg)

    # -------------------------
    # Build model
    # -------------------------
    model = instantiate(cfg.model)

    # Load dense checkpoint (HF) for non-MoE layers.
    checkpointer = instantiate(cfg.checkpointer)
    ckpt = checkpointer.load_checkpoint()
    if isinstance(ckpt, dict) and MODEL_KEY in ckpt:
        dense_sd = ckpt[MODEL_KEY]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        dense_sd = ckpt["model"]
    else:
        dense_sd = ckpt

    # Load non-MLP weights (strict=False because MoE params don't exist in checkpoint)
    missing, unexpected = model.load_state_dict(dense_sd, strict=False)
    _rank0_print(f"[ckpt] load_state_dict strict=False. missing={len(missing)} unexpected={len(unexpected)}")

    # Initialize experts from dense MLP weights
    init_moe_from_dense_state_dict(
        model,
        dense_sd,
        num_experts=int(getattr(cfg.model, "num_experts", cfg.get("num_experts", 8))),
        verbose=(rank == 0),
    )

    # Move to device/dtype
    dtype_str = str(getattr(cfg, "dtype", "bf16"))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(dtype_str, torch.bfloat16)
    model = model.to(device=device, dtype=dtype)

    # Shard with FSDP2 (torch.distributed._composable.fsdp.fully_shard)
    use_fsdp = bool(getattr(cfg, "enable_fsdp", True)) and _is_distributed()
    if use_fsdp:
        shard_conditions = [partial(get_shard_conditions, names_to_match=None)]
        shard_model(
            model,
            shard_conditions=shard_conditions,
            cpu_offload=bool(getattr(cfg, "fsdp_cpu_offload", False)),
            reshard_after_forward=bool(getattr(cfg, "fsdp_reshard_after_forward", True)),
        )
        _rank0_print("[fsdp] Model sharded with torchtune.training.shard_model (FSDP2)")

    # Optimizer
    try:
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
    except TypeError:
        optimizer = instantiate(cfg.optimizer, model.parameters())

    optimizer.zero_grad(set_to_none=True)

    # Loss weights
    lb_w = float(getattr(cfg, "load_balancing_loss_weight", 0.01))

    # Label shifting: True/False/auto
    shift_mode = str(getattr(cfg, "shift_labels", "auto")).lower()
    assert shift_mode in ("auto", "true", "false"), "shift_labels must be one of: auto|true|false"
    decided_shift: bool | None = None

    # -------------------------
    # Train
    # -------------------------
    model.train()

    grad_accum = int(getattr(cfg, "gradient_accumulation_steps", 1))
    log_every = int(getattr(cfg.metric_logger, "log_every_n_steps", 1))

    global_step = 0
    start_time = time.time()

    for epoch in range(int(cfg.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if decided_shift is None:
                if shift_mode == "auto":
                    decided_shift = _decide_if_shift_labels(tokens, labels, ignore_index)
                else:
                    decided_shift = shift_mode == "true"
                _rank0_print(f"[loss] shift_labels={shift_mode} => decided_shift={decided_shift}")

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype in (torch.bfloat16, torch.float16))):
                logits, router_loss, aux_loss = model(tokens)
                ce_loss = _compute_ce_loss(
                    logits,
                    tokens,
                    labels,
                    ignore_index=ignore_index,
                    shift_labels=decided_shift,
                )

                total_loss = ce_loss + lb_w * aux_loss

            (total_loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                if getattr(cfg, "clip_grad_norm", None) is not None:
                    clip_val = float(cfg.clip_grad_norm)
                    if hasattr(model, "clip_grad_norm_"):
                        grad_norm = model.clip_grad_norm_(clip_val)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                else:
                    grad_norm = None

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Logging
                if global_step % log_every == 0 and rank == 0:
                    elapsed = time.time() - start_time
                    # Approx tokens/sec/gpu (rough):
                    toks = tokens.numel() * log_every * grad_accum
                    tps = toks / max(elapsed, 1e-9) / max(world_size, 1)
                    start_time = time.time()

                    payload = {
                        "loss/total": float(total_loss.detach().float().cpu()),
                        "loss/ce": float(ce_loss.detach().float().cpu()),
                        "loss/aux": float(aux_loss.detach().float().cpu()),
                        "loss/router": float(router_loss.detach().float().cpu()),
                        "perf/tokens_per_sec_per_gpu": float(tps),
                    }
                    if grad_norm is not None:
                        payload["perf/grad_norm"] = float(grad_norm.detach().float().cpu())

                    metric_logger.log_dict(payload, step=global_step)

        _rank0_print(f"[epoch] finished epoch {epoch}")

    if _is_distributed():
        dist.barrier()

    if rank == 0:
        _rank0_print("[done] training complete")

    metric_logger.close()


if __name__ == "__main__":
    main()
