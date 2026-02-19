"""Distributed checkpointing (DCP) helper for FSDP training.

Why not torchtune's checkpointers?
---------------------------------
torchtune checkpointers are primarily designed around:

* loading HF weights (finetuning / continued pretraining)
* saving in HF-compatible formats

For **from-scratch pretraining** we often want a minimal, robust, and
sharded checkpoint format that can be saved and restored quickly.

PyTorch's ``torch.distributed.checkpoint`` (DCP) is a good default for this.

Design goals:
-------------
* No dependency on CUDA-only libs.
* Works for ROCm (backend is still named ``nccl``).
* Works with FSDP2 (`torch.distributed._composable.fsdp.fully_shard`).
* Hackable: all logic is in one file.

This checkpointer saves a *folder per step*:

    <output_dir>/checkpoints/step_000000100

Each folder contains per-rank files + metadata written by DCP.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


@dataclass
class DCPCheckpoint:
    path: str
    step: int


class DCPCheckpointer:
    """A tiny distributed checkpointer built on PyTorch DCP."""

    def __init__(self, *, output_dir: str, checkpoints_subdir: str = "checkpoints") -> None:
        self.output_dir = str(output_dir)
        self.checkpoints_dir = os.path.join(self.output_dir, checkpoints_subdir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def _format_step_dir(self, step: int) -> str:
        return os.path.join(self.checkpoints_dir, f"step_{step:09d}")

    def _list_step_dirs(self) -> list[DCPCheckpoint]:
        """List checkpoints sorted by step."""

        pat = re.compile(r"^step_(\d+)$")
        out: list[DCPCheckpoint] = []
        for entry in os.scandir(self.checkpoints_dir):
            if not entry.is_dir():
                continue
            m = pat.match(entry.name)
            if not m:
                continue
            step = int(m.group(1))
            out.append(DCPCheckpoint(path=entry.path, step=step))
        out.sort(key=lambda x: x.step)
        return out

    def latest_checkpoint(self) -> Optional[DCPCheckpoint]:
        ckpts = self._list_step_dirs()
        return ckpts[-1] if ckpts else None

    # -------------------------
    # Save / load
    # -------------------------
    def save(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a distributed checkpoint.

        All ranks must call this function.
        """

        ckpt_dir = self._format_step_dir(step)

        # DCP expects an empty/non-existent directory.
        # Rank 0 creates it; others wait.
        if _get_rank() == 0:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=False)
        if _is_distributed():
            torch.distributed.barrier()

        # Import DCP lazily so the rest of the repo doesn't require it for import.
        try:
            from torch.distributed.checkpoint import FileSystemWriter, save
        except Exception:  # pragma: no cover
            from torch.distributed.checkpoint import FileSystemWriter, save_state_dict as save  # type: ignore

        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optim_state = get_state_dict(model, optimizer)

        state: Dict[str, Any] = {
            "model": model_state,
            "optimizer": optim_state,
            "extra_state": extra_state or {},
        }
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler.state_dict()

        writer = FileSystemWriter(ckpt_dir)
        save(state_dict=state, storage_writer=writer)

        if _is_distributed():
            torch.distributed.barrier()

        return ckpt_dir

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a distributed checkpoint.

        Args:
            model, optimizer: The *already-constructed* objects to load into.
            lr_scheduler: Optional scheduler to restore.
            checkpoint_dir: If None, loads the latest checkpoint under output_dir.

        Returns:
            extra_state: Arbitrary dict saved under ``extra_state``.
        """

        if checkpoint_dir is None:
            latest = self.latest_checkpoint()
            if latest is None:
                return {}
            checkpoint_dir = latest.path

        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint dir does not exist: {checkpoint_dir}")

        try:
            from torch.distributed.checkpoint import FileSystemReader, load
        except Exception:  # pragma: no cover
            from torch.distributed.checkpoint import FileSystemReader, load_state_dict as load  # type: ignore

        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        # Allocate destination state_dicts with the right structure.
        model_state, optim_state = get_state_dict(model, optimizer)
        state: Dict[str, Any] = {
            "model": model_state,
            "optimizer": optim_state,
            "extra_state": {},
        }
        if lr_scheduler is not None:
            state["lr_scheduler"] = {}

        reader = FileSystemReader(checkpoint_dir)
        load(state_dict=state, storage_reader=reader)

        # Materialize into model/optimizer.
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )

        if lr_scheduler is not None and "lr_scheduler" in state:
            lr_scheduler.load_state_dict(state["lr_scheduler"])  # type: ignore[arg-type]

        return dict(state.get("extra_state", {}))
