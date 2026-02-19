"""Streaming pretraining dataset for **The Pile** (local shards).

Fixes:
- Accepts extra kwargs (split/streaming/dataset_name/etc.) from older YAML configs
  and ignores them, so torchtune.instantiate() won't crash.

Core:
- Reads local shard files: .jsonl / .jsonl.gz / .jsonl.zst(.zstd)
- Shards deterministically by:
    - distributed rank/world_size
    - dataloader worker id/num_workers
- Packs tokens into fixed-length blocks for causal LM
"""

from __future__ import annotations

import glob
import io
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass
class PileStreamingConfig:
    # Local shards
    data_dir: str
    file_glob: str = "*.jsonl*"
    text_key: str = "text"

    # Packing
    max_seq_len: int = 1024
    add_bos: bool = False
    add_eos: bool = True

    # Shuffling
    shuffle_files: bool = True
    seed: int = 42

    # Smoke test cap
    num_samples: Optional[int] = None


def _get_rank_world() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _encode_text(tokenizer: Any, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
    # torchtune-style: encode(text, add_bos=..., add_eos=...)
    try:
        return tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    except TypeError:
        pass

    ids = tokenizer.encode(text)

    if add_bos:
        bos_id = None
        for attr in ("bos_id", "bos_token_id"):
            if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not None:
                bos_id = int(getattr(tokenizer, attr))
                break
        if bos_id is not None:
            ids = [bos_id] + ids

    if add_eos:
        eos_id = None
        for attr in ("eos_id", "eos_token_id"):
            if hasattr(tokenizer, attr) and getattr(tokenizer, attr) is not None:
                eos_id = int(getattr(tokenizer, attr))
                break
        if eos_id is not None:
            ids = ids + [eos_id]

    return ids


def _open_text_lines(path: str) -> Iterator[str]:
    """Open a possibly compressed jsonl file and yield text lines."""
    lower = path.lower()
    if lower.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
        return

    if lower.endswith(".zst") or lower.endswith(".zstd"):
        # Prefer python zstandard if available
        try:
            import zstandard as zstd  # type: ignore
            with open(path, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
                    for line in text_stream:
                        yield line
            return
        except Exception:
            # Fallback: require `zstdcat` in container
            import subprocess
            p = subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert p.stdout is not None
            for raw in p.stdout:
                yield raw.decode("utf-8", errors="ignore")
            rc = p.wait()
            if rc != 0:
                err = b""
                try:
                    err = p.stderr.read() if p.stderr is not None else b""
                except Exception:
                    pass
                raise RuntimeError(
                    f"zstdcat failed for {path} (rc={rc}). "
                    f"Install python 'zstandard' or ensure zstdcat exists. stderr={err[:2000]!r}"
                )
            return

    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


class PileStreamingDataset(IterableDataset):
    def __init__(self, tokenizer: Any, cfg: PileStreamingConfig) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._cfg = cfg
        if cfg.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2 for causal LM")
        if not cfg.data_dir or not os.path.isdir(cfg.data_dir):
            raise ValueError(f"data_dir must be an existing directory, got: {cfg.data_dir}")

    def _list_files(self) -> list[str]:
        pattern = os.path.join(self._cfg.data_dir, self._cfg.file_glob)
        files = sorted(glob.glob(pattern))
        if not files:
            raise RuntimeError(f"No files matched: {pattern}")
        return files

    def _shard_files(self, files: list[str]) -> list[str]:
        rank, world = _get_rank_world()
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1

        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world * num_workers

        return files[global_worker_id::global_num_workers] if global_num_workers > 1 else files

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = self._shard_files(self._list_files())
        rank, _world = _get_rank_world()
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        global_worker_id = rank * num_workers + worker_id

        rng = random.Random(self._cfg.seed + global_worker_id)

        max_len = int(self._cfg.max_seq_len)
        token_buffer: list[int] = []
        yielded = 0

        while True:
            if self._cfg.shuffle_files:
                rng.shuffle(files)

            for fp in files:
                for line in _open_text_lines(fp):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    text = obj.get(self._cfg.text_key, None)
                    if text is None:
                        continue
                    if not isinstance(text, str):
                        text = str(text)
                    if not text.strip():
                        continue

                    ids = _encode_text(
                        self._tokenizer,
                        text,
                        add_bos=self._cfg.add_bos,
                        add_eos=self._cfg.add_eos,
                    )
                    if not ids:
                        continue

                    token_buffer.extend(ids)

                    while len(token_buffer) >= max_len:
                        block = token_buffer[:max_len]
                        token_buffer = token_buffer[max_len:]

                        t = torch.tensor(block, dtype=torch.long)
                        yield {"tokens": t}

                        yielded += 1
                        if self._cfg.num_samples is not None and yielded >= int(self._cfg.num_samples):
                            return


def pile_streaming_dataset(
    tokenizer: Any,
    *,
    # --- Local Pile shards ---
    data_dir: Optional[str] = None,
    file_glob: str = "*.jsonl*",
    text_key: str = "text",
    # --- Packing ---
    max_seq_len: int = 1024,
    shuffle_files: bool = True,
    seed: int = 42,
    add_bos: bool = False,
    add_eos: bool = True,
    num_samples: Optional[int] = None,
    # --- Compatibility: accept HF/old-config args and ignore ---
    split: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    streaming: Optional[bool] = None,
    trust_remote_code: Optional[bool] = None,
    **_ignored: Any,
) -> "PileStreamingDataset":
    """
    torchtune-style dataset builder.

    Robust behavior:
    - If `data_dir` is not provided via YAML/CLI, fall back to environment variable `PILE_DIR`.
    - If `file_glob` is default and env `PILE_GLOB` is set, use it.
    - Accepts and ignores HF-style args (split/streaming/etc.) so old YAML won't break.
    """
    import os

    if not data_dir:
        data_dir = os.environ.get("PILE_DIR")

    if (file_glob == "*.jsonl*") and os.environ.get("PILE_GLOB"):
        file_glob = os.environ["PILE_GLOB"]

    if not data_dir:
        raise ValueError(
            "Missing Pile data_dir. Provide one of:\n"
            "  1) YAML: dataset.data_dir: /path/to/pile\n"
            "  2) CLI: dataset.data_dir=/path/to/pile\n"
            "  3) ENV: export PILE_DIR=/path/to/pile\n"
        )

    cfg = PileStreamingConfig(
        data_dir=data_dir,
        file_glob=file_glob,
        text_key=text_key,
        max_seq_len=max_seq_len,
        shuffle_files=shuffle_files,
        seed=seed,
        add_bos=add_bos,
        add_eos=add_eos,
        num_samples=num_samples,
    )
    return PileStreamingDataset(tokenizer, cfg)
