#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import lmdb
import argparse
import numpy as np
import struct
import sys
from typing import Callable, Optional, Tuple
import time
import re

from DNATextTokenizers import DNATextUnifiedTokenizer


# -------------------- utils --------------------

def _to_1d_list(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x
    try:
        import torch
        if hasattr(torch, "Tensor") and isinstance(x, torch.Tensor):
            return x.view(-1).tolist()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()
    return list(x)


def _tokenize_line(tokenizer, line, max_length, bos_text, eos_text):
    text = f"{bos_text}<dna>{line}<dna>{eos_text}"
    tok = tokenizer(text, truncation=True, max_length=max_length)
    input_ids = _to_1d_list(tok.get("input_ids"))
    modality_mask = _to_1d_list(tok.get("modality_mask")) if "modality_mask" in tok else None
    return input_ids, modality_mask


def _write_meta(txn: lmdb.Transaction, meta: dict):
    txn.put(b"meta", json.dumps(meta, ensure_ascii=False).encode("utf-8"))


def _retry_mapfull(env: lmdb.Environment,
                   txn: Optional[lmdb.Transaction],
                   writer_fn: Callable[[lmdb.Transaction], None],
                   grow_factor: float,
                   grow_min_bytes: int) -> lmdb.Transaction:
    if txn is None:
        txn = env.begin(write=True)
    while True:
        try:
            writer_fn(txn)
            return txn
        except lmdb.MapFullError:
            try:
                txn.abort()
            except Exception:
                pass
            info = env.info()
            old = int(info.get("map_size", 0)) or int(env.stat().get("psize", 4096)) * 4 * 1024 * 1024
            new = max(int(old * grow_factor), old + grow_min_bytes)
            print(f"[LMDB] Map full -> grow mapsize: {old/(1<<30):.2f} GB -> {new/(1<<30):.2f} GB", flush=True)
            env.set_mapsize(new)
            txn = env.begin(write=True)


def _meta_payload(count: int,
                  has_mask: bool,
                  tokenizer_path: str,
                  max_length: int) -> dict:
    return {
        "version": 1,
        "count": count,
        "schema": {
            "tokens": "int64",
            "len": "uint32",
            "modality_mask": "bool(optional)",
            "type": "str"
        },
        "dtype": "int64",
        "has_mask": has_mask,
        "tokenizer": tokenizer_path,
        "max_length": max_length,
        "layout": "sample:{12d}:{tokens|len|modality_mask|type}"
    }

# ==== NEW: commit helper with meta sync + progress log ========================
def _safe_commit(env: lmdb.Environment,
                 txn: lmdb.Transaction,
                 total_written: int,
                 has_mask: bool,
                 tokenizer_path: str,
                 max_length: int,
                 grow_factor: float,
                 grow_min_bytes: int,
                 commit_range: Tuple[int, int]):
    """Atomically write meta(count=total_written) and commit."""
    start, end = commit_range
    t0 = time.time()

    def write_meta_final(_txn):
        _write_meta(_txn, _meta_payload(total_written, has_mask, tokenizer_path, max_length))

    txn = _retry_mapfull(env, txn, write_meta_final, grow_factor, grow_min_bytes)
    txn.commit()
    env.sync()
    dt = time.time() - t0
    print(f"[COMMIT] samples [{start:,}, {end:,}) committed; "
          f"total={total_written:,}; took {dt:.2f}s", flush=True)
    return env.begin(write=True)

# ==== NEW: post-verify scanner =================================================
_rx_tokens = re.compile(rb"^sample:(\d{12}):tokens$")

def post_verify_scan_present(lmdb_path: str,
                             preview_probe: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
    """Scan LMDB after build; return sorted present sample ids, max_id+1, holes."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False, readahead=True, max_readers=512)
    present = []
    with env.begin() as txn:
        cur = txn.cursor()
        n_scanned = 0
        for k, _ in cur:
            m = _rx_tokens.match(k)
            if m:
                present.append(int(m.group(1)))
            n_scanned += 1
            if preview_probe and n_scanned % preview_probe == 0:
                print(f"[VERIFY] scanned {n_scanned:,} keys...", flush=True)
    env.close()
    if not present:
        return np.array([], dtype=np.int64), 0, 0
    present = np.array(sorted(present), dtype=np.int64)
    n_present = present.size
    n_max = int(present[-1]) + 1
    holes = n_max - n_present
    return present, n_max, holes

# ==== NEW: compact (repair) writer ============================================
def compact_rewrite_lmdb(src_path: str,
                         dst_path: str,
                         present_indices: np.ndarray,
                         meta_template: dict,
                         map_size_gb: float = 512.0):
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    if os.path.exists(dst_path):
        os.remove(dst_path)

    env_src = lmdb.open(src_path, readonly=True, lock=False, subdir=False, readahead=True, max_readers=512)
    env_dst = lmdb.open(dst_path, map_size=int(map_size_gb * (1 << 30)),
                        subdir=False, lock=True, writemap=False, readahead=False, meminit=False, max_dbs=1)

    with env_src.begin() as txn_src, env_dst.begin(write=True) as txn_dst:
        cur = txn_src.cursor()
        total = 0
        for new_id, old_id in enumerate(present_indices.tolist()):
            prefix = f"sample:{old_id:012d}:".encode()
            if not cur.set_range(prefix):
                continue
            while True:
                k = cur.key()
                if k is None or not k.startswith(prefix):
                    break
                v = cur.value()
                suffix = k.split(b":")[-1]
                k_new = f"sample:{new_id:012d}:".encode() + suffix
                txn_dst.put(k_new, v)
                if not cur.next():
                    break
            total += 1
            if total % 100_000 == 0:
                _write_meta(txn_dst, {**meta_template, "count": total})
                txn_dst.commit()
                env_dst.sync()
                print(f"[REPACK] {total:,} samples...", flush=True)
                # reopen write txn
                txn_dst = env_dst.begin(write=True)

        # final meta
        _write_meta(txn_dst, {**meta_template, "count": total})
    env_dst.sync()
    env_src.close()
    env_dst.close()
    print(f"[REPACK][DONE] wrote {total:,} samples to {dst_path}", flush=True)


# -------------------- main converter --------------------

def convert_txt_to_lmdb(
    input_txt: str,
    output_lmdb: str,
    tokenizer_path: str,
    max_length: int = 512,
    bos_text: str = "",
    eos_text: str = "",
    map_size_gb: float = 8.0,
    commit_interval: int = 10000,
    compact: bool = True,
    auto_grow: bool = True,
    grow_factor: float = 1.6,
    grow_min_gb: float = 1.0,
    # ==== NEW opts =============================================================
    log_every: int = 10000,
    emit_index_mapping: Optional[str] = None,
    post_verify: bool = True,
    repair_if_needed: bool = False,
    repair_output_lmdb: Optional[str] = None,
):
    out_dir = os.path.dirname(os.path.abspath(output_lmdb))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    build_path = output_lmdb + ".build"
    compact_path = output_lmdb + ".compact"

    for p in (build_path, compact_path, build_path + "-lock", compact_path + "-lock"):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    print("[DEBUG] Step 1: Loading tokenizer...", flush=True)
    tokenizer = DNATextUnifiedTokenizer.from_pretrained(tokenizer_path)
    tokenizer.model_max_length = int(1e30)
    print("[DEBUG] Step 2: Tokenizer loaded successfully.", flush=True)

    env = lmdb.open(
        build_path,
        map_size=int(map_size_gb * (1 << 30)),
        subdir=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )

    total = 0
    has_mask = False
    grow_min_bytes = int(grow_min_gb * (1 << 30))

    with env.begin(write=True) as txn:
        _write_meta(txn, _meta_payload(0, has_mask, tokenizer_path, max_length))

    print("[DEBUG] Step 3: Converting data to LMDB...", flush=True)

    txn = env.begin(write=True)
    batch_start = 0
    t0 = time.time()
    try:
        with open(input_txt, "r", encoding="utf-8", errors="ignore") as f:
            for line_i, raw in enumerate(f):
                line = raw.strip()
                if not line:
                    continue

                if line_i % log_every == 0 and line_i > 0:
                    dt = time.time() - t0
                    rate = (total - batch_start) / max(dt, 1e-6)
                    print(f"[DEBUG] line={line_i:,} written={total:,} (+{total-batch_start}) "
                          f"~{rate:.1f} samples/s", flush=True)
                    batch_start = total
                    t0 = time.time()

                input_ids, modality_mask = _tokenize_line(
                    tokenizer, line, max_length, bos_text, eos_text
                )
                if not input_ids:
                    continue

                tok_arr = np.asarray(input_ids, dtype=np.int64)
                token_bytes = tok_arr.tobytes()

                if len(token_bytes) % 8 != 0:
                    print(f"[WARN] token bytes not divisible by 8 at line {line_i}; skip.", flush=True)
                    continue

                kprefix = f"sample:{total:012d}:".encode("ascii")

                def write_sample(_txn):
                    nonlocal has_mask
                    _txn.put(kprefix + b"tokens", token_bytes)
                    _txn.put(kprefix + b"type", b"int64")
                    _txn.put(kprefix + b"len", struct.pack("<I", tok_arr.size))

                    if modality_mask is not None:
                        mask_arr = np.asarray(modality_mask, dtype=np.bool_)
                        if mask_arr.size != tok_arr.size:
                            m = min(mask_arr.size, tok_arr.size)
                            if m == 0:
                                raise ValueError(f"Empty after aligning tokens/mask at line {line_i}")
                            mask_arr = mask_arr[:m]
                            tok_trim = tok_arr[:m]
                            _txn.replace(kprefix + b"tokens", tok_trim.tobytes())
                            _txn.replace(kprefix + b"len", struct.pack("<I", int(m)))
                        _txn.put(kprefix + b"modality_mask", mask_arr.tobytes())
                        has_mask = True

                if auto_grow:
                    txn = _retry_mapfull(env, txn, write_sample, grow_factor, grow_min_bytes)
                else:
                    write_sample(txn)

                total += 1

                if total % commit_interval == 0:
                    # atomic meta(count=total) + commit + sync + reopen
                    txn = _safe_commit(env, txn, total, has_mask, tokenizer_path, max_length,
                                       grow_factor, grow_min_bytes,
                                       commit_range=(total - commit_interval, total))

        # final commit
        txn = _safe_commit(env, txn, total, has_mask, tokenizer_path, max_length,
                           grow_factor, grow_min_bytes,
                           commit_range=(max(total - (total % commit_interval), 0), total))
        env.close()

        # compact to single file
        if compact:
            print("[DEBUG] Compact copying to final single-file LMDB ...", flush=True)
            env_src = lmdb.open(build_path, readonly=True, subdir=False, lock=False, readahead=True)
            env_src.copy(compact_path, compact=True)
            env_src.close()

            if os.path.exists(output_lmdb):
                os.remove(output_lmdb)
            os.replace(compact_path, output_lmdb)

            for p in (compact_path + "-lock", build_path, build_path + "-lock"):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        else:
            if os.path.exists(output_lmdb):
                os.remove(output_lmdb)
            os.replace(build_path, output_lmdb)
            try:
                os.remove(build_path + "-lock")
            except Exception:
                pass

        print(f"[DONE] Wrote {total:,} samples to {output_lmdb} "
              f"(dtype=int64, has_mask={has_mask}).", flush=True)

        # ==== NEW: post-verify & optional repair =================================
        if post_verify:
            print("[VERIFY] scanning present sample ids ...", flush=True)
            present, n_max, holes = post_verify_scan_present(output_lmdb, preview_probe=2_000_000)
            print(f"[VERIFY] max_id={n_max-1:,} present={present.size:,} holes={holes:,}", flush=True)
            if emit_index_mapping:
                np.save(emit_index_mapping, present)
                print(f"[VERIFY] saved present mapping -> {emit_index_mapping}", flush=True)
            if holes > 0 and repair_if_needed:
                dst = repair_output_lmdb or (os.path.splitext(output_lmdb)[0] + "_repacked.lmdb")
                print(f"[REPAIR] holes detected -> repacking to {dst}", flush=True)
                compact_rewrite_lmdb(
                    src_path=output_lmdb,
                    dst_path=dst,
                    present_indices=present,
                    meta_template=_meta_payload(0, has_mask, tokenizer_path, max_length),
                    map_size_gb=max(512.0, map_size_gb * 8),
                )
                print("[REPAIR] Done. You may point your training to the repacked LMDB.", flush=True)

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Flushing current transaction...", file=sys.stderr, flush=True)
        try:
            if txn is not None:
                txn = _retry_mapfull(env, txn,
                                     lambda _t: _write_meta(_t, _meta_payload(total, has_mask, tokenizer_path, max_length)),
                                     grow_factor, grow_min_bytes)
                txn.commit()
        except Exception:
            pass
        try:
            env.sync(); env.close()
        except Exception:
            pass
        raise
    except Exception:
        try:
            if txn is not None:
                txn.abort()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        raise


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert DNABERT2 TXT to LMDB (single-file, auto-grow, compact, verified)."
    )
    parser.add_argument("--input_txt", type=str, required=True)
    parser.add_argument("--output_lmdb", type=str, default=None, help="Output LMDB file path (single file)")
    parser.add_argument("--output_dir", type=str, default=None, help="If set, final file at {dir}/data.lmdb")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bos_text", type=str, default="")
    parser.add_argument("--eos_text", type=str, default="")
    parser.add_argument("--map_size_gb", type=float, default=8.0)
    parser.add_argument("--commit_interval", type=int, default=10000)
    parser.add_argument("--no_compact", action="store_true", help="Skip compact copy (not recommended)")
    parser.add_argument("--no_auto_grow", action="store_true", help="Disable automatic mapsize growth")
    parser.add_argument("--grow_factor", type=float, default=1.6, help="Auto-grow multiplicative factor")
    parser.add_argument("--grow_min_gb", type=float, default=1.0, help="Auto-grow minimum increment (GB)")
    # ==== NEW CLI ==============================================================
    parser.add_argument("--log_every", type=int, default=10000)
    parser.add_argument("--emit_index_mapping", type=str, default=None,
                        help="Save present sample ids to this .npy path after build")
    parser.add_argument("--no_post_verify", action="store_true", help="Skip post-build verification scan")
    parser.add_argument("--repair_if_needed", action="store_true",
                        help="If holes found, repack to a new LMDB with contiguous ids")
    parser.add_argument("--repair_output_lmdb", type=str, default=None,
                        help="Destination path for repaired LMDB (defaults to *_repacked.lmdb)")
    args = parser.parse_args()

    if args.output_lmdb:
        out_path = args.output_lmdb
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "data.lmdb")
    else:
        raise ValueError("Please provide --output_lmdb or --output_dir")

    convert_txt_to_lmdb(
        input_txt=args.input_txt,
        output_lmdb=out_path,
        tokenizer_path=args.tokenizer,
        max_length=args.max_length,
        bos_text=args.bos_text,
        eos_text=args.eos_text,
        map_size_gb=args.map_size_gb,
        commit_interval=args.commit_interval,
        compact=(not args.no_compact),
        auto_grow=(not args.no_auto_grow),
        grow_factor=args.grow_factor,
        grow_min_gb=args.grow_min_gb,
        log_every=args.log_every,
        emit_index_mapping=args.emit_index_mapping,
        post_verify=(not args.no_post_verify),
        repair_if_needed=args.repair_if_needed,
        repair_output_lmdb=args.repair_output_lmdb,
    )


if __name__ == "__main__":
    main()
