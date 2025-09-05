# src/lmdb_text_dataset.py
import os
import re
import json
import struct
from typing import Any, Dict, List, Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from composer.utils import dist as cdist
import transformers

# --------- Dataset ---------
class LMDBTextDataset(Dataset):
    """
    Minimal reader for DNABERT2-style LMDB written as:
      keys: sample:{12d}:{tokens|type|len|modality_mask}
      meta: __meta__ (JSON) with fields including {"count": N, "notes": "keys: sample:{12d}:..."}
    Returns dicts like {"input_ids": LongTensor[max_seq_len]} to work with
    transformers.DataCollatorForLanguageModeling.
    """

    def __init__(
        self,
        lmdb_path: str,
        max_seq_len: int,
        subdb: Optional[str] = None,     # named sub-database if used when writing, else None
        readonly: bool = True,
        readahead: bool = False,
        lock: bool = False,
    ):
        self.path = lmdb_path
        self.max_seq_len = int(max_seq_len)

        # LMDB handles
        self._env: Optional[lmdb.Environment] = None
        self._dbi: Optional[lmdb._Database] = None  # type: ignore[attr-defined]

        # From meta
        self._length: Optional[int] = None
        self._kv_width: int = 12  # default width if not found in meta

        # Subdb name (bytes) if specified
        self._subdb_name = subdb.encode() if isinstance(subdb, str) else None

        # Open & probe (then close; workers will reopen)
        self._open_env(readonly=readonly, readahead=readahead, lock=lock)
        self._probe_meta_and_length()
        self._close_env()

    # ----- LMDB lifecycle -----
    def _open_env(self, readonly=True, readahead=False, lock=False, map_size=0):
        if self._env is not None:
            return
        self._env = lmdb.open(
            self.path,
            readonly=readonly,
            lock=lock,
            readahead=readahead,
            max_readers=2048,
            subdir=os.path.isdir(self.path),  # file vs directory auto-detect
            map_size=map_size if map_size > 0 else 0,
            max_dbs=16,
            meminit=False,
        )
        if self._subdb_name is not None:
            # open named subdb if exists (no create)
            self._dbi = self._env.open_db(self._subdb_name, create=False)

    def _close_env(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
            self._dbi = None

    def reopen_env_for_worker(self):
        """To be used in worker_init_fn."""
        self._open_env(readonly=True, readahead=False, lock=False)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_env"] = None
        s["_dbi"] = None
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)

    # ----- Meta / length -----
    def _probe_meta_and_length(self):
        assert self._env is not None
        with self._env.begin(write=False) as txn:
            # Prefer __meta__ JSON (written by your converter)
            v = txn.get(b"__meta__", db=self._dbi)
            if v:
                try:
                    meta = json.loads(v.decode("utf-8"))
                    if isinstance(meta, dict):
                        cnt = meta.get("count", None)
                        if isinstance(cnt, int) and cnt >= 0:
                            self._length = cnt
                        notes = meta.get("notes", "")
                        m = re.search(r"sample:\{(\d+)d\}", notes)
                        if m:
                            self._kv_width = int(m.group(1))
                except Exception:
                    pass

            # Fallback to classic __len__/len keys
            if self._length is None:
                for k in (b"__len__", b"len", b"__L__"):
                    v2 = txn.get(k, db=self._dbi)
                    if v2 is None:
                        continue
                    try:
                        self._length = int(v2.decode("utf-8"))
                        break
                    except Exception:
                        try:
                            self._length = int.from_bytes(v2, "little")
                            break
                        except Exception:
                            pass

            # Final fallback: count keys with ":tokens" suffix
            if self._length is None:
                try:
                    n = 0
                    with txn.cursor(db=self._dbi) as cur:
                        for k, _ in cur:
                            # Skip meta keys
                            if k in (b"__meta__", b"__len__", b"len", b"__L__"):
                                continue
                            if k.endswith(b":tokens"):
                                n += 1
                    self._length = n
                except Exception:
                    self._length = 0

    def __len__(self):
        return int(self._length or 0)

    # ----- Helpers -----
    @staticmethod
    def _dtype_from_str(s: str) -> np.dtype:
        mp = {
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint16": np.uint16,
            "uint32": np.uint32,
        }
        return mp.get(s, np.int64)

    # ----- Item read (KV layout) -----
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._env is None:
            self.reopen_env_for_worker()
        assert self._env is not None

        kprefix = f"sample:{idx:0{self._kv_width}d}:".encode("ascii")
        with self._env.begin(write=False) as txn:
            raw_tokens = txn.get(kprefix + b"tokens", db=self._dbi)
            if raw_tokens is None:
                raise IndexError(f"Missing tokens for sample {idx}")
            raw_dtype = txn.get(kprefix + b"type", db=self._dbi) or b"int64"
            dtype_str = raw_dtype.decode("ascii") if isinstance(raw_dtype, (bytes, bytearray)) else "int64"
            raw_len = txn.get(kprefix + b"len", db=self._dbi)

            arr = np.frombuffer(raw_tokens, dtype=self._dtype_from_str(dtype_str))
            if raw_len:
                n = struct.unpack("<I", raw_len)[0]
                arr = arr[:n]
            # truncate / pad to max_seq_len (pad optional; DCLM handles padding)
            arr = arr[: self.max_seq_len].copy()
            ids = torch.from_numpy(arr).to(torch.long)
            return {"input_ids": ids}

# --------- Builder ---------
def _worker_init_fn(_):
    info = get_worker_info()
    if info is None:
        return
    ds = info.dataset  # type: ignore[attr-defined]
    if isinstance(ds, LMDBTextDataset):
        ds.reopen_env_for_worker()

def build_lmdb_text_dataloader(
    cfg,
    tokenizer: transformers.PreTrainedTokenizerBase,
    device_batch_size: int,
):
    """
    cfg structure (minimal):

    train_loader:
      name: text_lmdb
      dataset:
        lmdb_path: /path/to/data.lmdb
        max_seq_len: 128
        subdb: null                 # optional if you used a named subdb
        shuffle: true               # optional (default False)
        num_workers: 4
        mlm_probability: 0.30       # used by DataCollatorForLanguageModeling
      drop_last: true
    """
    lmdb_path = cfg.dataset.get("lmdb_path", None)
    if not lmdb_path:
        raise ValueError("cfg.dataset.lmdb_path must be set.")

    ds = LMDBTextDataset(
        lmdb_path=lmdb_path,
        max_seq_len=int(cfg.dataset.get("max_seq_len", 128)),
        subdb=cfg.dataset.get("subdb", None),
        readonly=True,
        readahead=False,
        lock=False,
    )

    print(f"[INFO] LMDB dataset ready at {lmdb_path} with {len(ds)} samples.")

    # MLM collator (uses tokenizer's mask token etc.)
    mlm_prob = cfg.dataset.get("mlm_probability", None)
    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=(mlm_prob is not None),
        mlm_probability=(mlm_prob or 0.15),
    )

    # Optional boundary wrapper (rare; keep parity with your previous code)
    eos_token_id = cfg.dataset.get("eos_token_id", None)
    bos_token_id = cfg.dataset.get("bos_token_id", None)
    if (eos_token_id is not None) or (bos_token_id is not None):
        collate_fn = _ConcatenatedSequenceCollatorWrapper(
            base_collator=collate_fn,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )

    # Distributed sampler (Composer helper)
    sampler = cdist.get_sampler(
        ds,
        drop_last=cfg.drop_last,
        shuffle=bool(cfg.dataset.get("shuffle", False)),
    )

    num_workers = int(cfg.dataset.get("num_workers", 0))
    return DataLoader(
        ds,
        sampler=sampler,          # do not also set shuffle=True on DataLoader
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=(num_workers > 0),
        timeout=cfg.get("timeout", 0),
        worker_init_fn=_worker_init_fn,
    )

# (Optional) keep a minimal version of your concatenated-sequence wrapper for parity
class _ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch, parity with your text path."""
    def __init__(self, base_collator, eos_token_id: Optional[int] = None, bos_token_id: Optional[int] = None):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError("Must supply eos_token_id or bos_token_id.")
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError("Cannot use both EOS and BOS.")
        if eos_token_id is None:
            self.split_token_id = bos_token_id
            self.bos_mode = True
        else:
            self.split_token_id = eos_token_id
            self.bos_mode = False

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        batch["sequence_id"] = self.get_sequence_id_from_batch(batch)
        return batch

    def get_sequence_id_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        is_sep = torch.eq(batch["input_ids"], self.split_token_id)
        csum = torch.cumsum(is_sep, dim=1).to(batch["input_ids"].dtype)
        if self.bos_mode:
            return csum
        left_zeros = csum.new_zeros((csum.shape[0], 1))
        return torch.cat([left_zeros, csum[:, :-1]], dim=1)

# --------- Standalone test ---------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("Quick inspect LMDB KV corpus")
    ap.add_argument("--lmdb_path", type=str, required=True)
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--subdb", type=str, default=None)
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    ds = LMDBTextDataset(
        lmdb_path=args.lmdb_path,
        max_seq_len=args.max_seq_len,
        subdb=args.subdb,
        readonly=True,
        readahead=False,
        lock=False,
    )
    print(f"[TEST] length = {len(ds)}")

    for i in range(min(args.num, len(ds))):
        ex = ds[i]
        x = ex["input_ids"]
        print(f"[TEST] sample {i}: shape={tuple(x.shape)}, dtype={x.dtype}, head={x[:20].tolist()}")
