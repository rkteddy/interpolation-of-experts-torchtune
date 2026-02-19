"""Custom torchtune-style **from-scratch pretraining** recipe for MoE/IoE models.

This replaces the previous SFT-focused recipe.

Key differences vs. torchtune's stock recipes:
------------------------------------------------
* **Model forward returns 3 values**:

    logits, router_loss, aux_loss = model(tokens)

* We explicitly optimize and log the MoE auxiliary loss:

    total_loss = cross_entropy_loss + load_balancing_loss_weight * aux_loss + router_loss

  (``router_loss`` is typically a *stabilizer* such as z-loss; it can be set to
  0 via config, in which case the formula reduces to the user's requirement.)

* Uses a streaming dataset for **The Pile** (IterableDataset) and runs in a
  step-based loop (``max_steps``) which is typical for pretraining.

ROCm notes:
-----------
* On ROCm, PyTorch still uses the device type string ``cuda``.
* Distributed backend name is still ``nccl`` (provided by RCCL).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchtune.config import instantiate
from torchtune.training import get_shard_conditions, shard_model
from torchtune.training.metric_logging import MetricLoggerInterface


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
        dist.init_process_group(backend="nccl")
        local_rank = _get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device(device_type, local_rank)
    return torch.device(device_type)


def _stack_collate(batch: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate for fixed-length causal LM blocks."""

    tokens_list = []
    labels_list = []
    for ex in batch:
        t = ex["tokens"]
        l = ex.get("labels", t)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.long)
        if not torch.is_tensor(l):
            l = torch.tensor(l, dtype=torch.long)
        tokens_list.append(t)
        labels_list.append(l)

    tokens = torch.stack(tokens_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return {"tokens": tokens, "labels": labels}


def _compute_ce_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard next-token cross entropy for causal LM.

    We compute:
        loss = CE(logits[:, :-1], tokens[:, 1:])
    """

    # logits: [B, S, V]
    logits = logits[:, :-1, :].contiguous()
    target = tokens[:, 1:].contiguous()
    vocab = logits.size(-1)
    return F.cross_entropy(logits.view(-1, vocab), target.view(-1), ignore_index=ignore_index)


def _build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup.

    This is intentionally simple and self-contained.
    """

    warmup_steps = int(max(warmup_steps, 0))
    total_steps = int(max(total_steps, 1))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        # progress in [0, 1]
        denom = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)

        # cosine from 1 -> min_lr_ratio
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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

    if "dataset" in cfg and isinstance(cfg.dataset, dict) and "data_dir" not in cfg.dataset:
        import os
        env_dir = os.environ.get("PILE_DIR")
        if env_dir:
            cfg.dataset["data_dir"] = env_dir
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    device = _init_distributed(device_type=getattr(cfg, "device", "cuda"))
    rank = _get_rank()
    import os
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    _rank0_print(f"[dist] world_size={world_size} rank={rank} device={device}")

    # -------------------------
    # Instantiate components
    # -------------------------
    tokenizer = instantiate(cfg.tokenizer)

    # Streaming dataset (IterableDataset) handles its own sharding.
    try:
        dataset = instantiate(cfg.dataset, tokenizer=tokenizer)
    except TypeError:
        dataset = instantiate(cfg.dataset, tokenizer)

    num_workers = int(getattr(cfg, "num_workers", 2))
    dl_kwargs: Dict[str, Any] = {
        "batch_size": int(cfg.batch_size),
        "shuffle": False,  # IterableDataset sharding/shuffle is handled inside.
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
        "collate_fn": _stack_collate,
    }
    # Only valid when num_workers > 0.
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = bool(getattr(cfg, "persistent_workers", True))
        dl_kwargs["prefetch_factor"] = int(getattr(cfg, "prefetch_factor", 2))

    dataloader = DataLoader(dataset, **dl_kwargs)

    metric_logger: MetricLoggerInterface = instantiate(cfg.metric_logger)
    metric_logger.log_config(cfg)

    # -------------------------
    # Build model (random init; no HF weight load)
    # -------------------------
    model = instantiate(cfg.model)

    dtype_str = str(getattr(cfg, "dtype", "bf16"))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(dtype_str, torch.bfloat16)
    model = model.to(device=device, dtype=dtype)

    # FSDP2 sharding (composable fully_shard via torchtune.training.shard_model)
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

    # Scheduler (optional)
    max_steps = int(getattr(cfg, "max_steps", 1000))
    warmup_steps = int(getattr(cfg, "warmup_steps", 100))
    min_lr_ratio = float(getattr(cfg, "min_lr_ratio", 0.1))
    lr_scheduler = _build_cosine_warmup_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # Checkpointer (optional)
    checkpointer = instantiate(cfg.checkpointer) if hasattr(cfg, "checkpointer") else None
    resume_from = getattr(cfg, "resume_from_checkpoint", None)
    start_step = 0
    if checkpointer is not None and resume_from not in (None, "", "null"):
        extra = checkpointer.load(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, checkpoint_dir=resume_from)
        start_step = int(extra.get("step", 0))
        _rank0_print(f"[ckpt] resumed from {resume_from} (step={start_step})")

    # Loss weights
    lb_w = float(getattr(cfg, "load_balancing_loss_weight", 0.01))
    ignore_index = int(getattr(cfg, "ignore_index", -100))

    # Train loop knobs
    grad_accum = int(getattr(cfg, "gradient_accumulation_steps", 1))
    clip_norm = getattr(cfg, "clip_grad_norm", None)
    log_every = int(getattr(cfg.metric_logger, "log_every_n_steps", 10))
    save_every = int(getattr(cfg, "save_every_n_steps", 0))

    # -------------------------
    # Train
    # -------------------------
    model.train()

    global_step = start_step
    start_time = time.time()

    data_iter = iter(dataloader)
    while global_step < max_steps:
        for micro_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            tokens = batch["tokens"].to(device, non_blocking=True)

            # Forward
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype in (torch.bfloat16, torch.float16))):
                logits, router_loss, aux_loss = model(tokens)
                ce_loss = _compute_ce_loss(logits, tokens, ignore_index=ignore_index)

                # User requirement: CE + lb_w * aux_loss
                # We also add router_loss directly (usually already scaled inside the model).
                total_loss = ce_loss + lb_w * aux_loss + router_loss

            (total_loss / grad_accum).backward()

        # Optimizer step
        grad_norm: Optional[torch.Tensor] = None
        if clip_norm is not None:
            clip_val = float(clip_norm)
            if hasattr(model, "clip_grad_norm_"):
                grad_norm = model.clip_grad_norm_(clip_val)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        global_step += 1

        # Logging (rank0)
        if rank == 0 and global_step % log_every == 0:
            elapsed = time.time() - start_time
            toks = tokens.numel() * log_every * grad_accum
            tps = toks / max(elapsed, 1e-9) / max(world_size, 1)
            start_time = time.time()

            lr = float(optimizer.param_groups[0]["lr"])

            payload = {
                "loss/total": float(total_loss.detach().float().cpu()),
                "loss/ce": float(ce_loss.detach().float().cpu()),
                "loss/aux": float(aux_loss.detach().float().cpu()),
                "loss/router": float(router_loss.detach().float().cpu()),
                "optim/lr": lr,
                "perf/tokens_per_sec_per_gpu": float(tps),
            }
            if grad_norm is not None:
                payload["perf/grad_norm"] = float(grad_norm.detach().float().cpu())

            metric_logger.log_dict(payload, step=global_step)

        # Checkpoint (all ranks)
        if checkpointer is not None and save_every and (global_step % save_every == 0):
            extra_state = {"step": global_step}
            ckpt_path = checkpointer.save(
                step=global_step,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                extra_state=extra_state,
            )
            _rank0_print(f"[ckpt] saved {ckpt_path}")

    if _is_distributed():
        dist.barrier()

    if rank == 0:
        _rank0_print("[done] pretraining complete")

    metric_logger.close()


if __name__ == "__main__":
    main()
