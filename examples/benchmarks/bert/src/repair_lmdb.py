#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import lmdb
import json
import struct
import argparse
import time
from typing import Optional

# -------------------- utils --------------------

def _fmt_bytes(n: int) -> str:
    return f"{n / (1<<30):.2f} GiB"

def _fmt_eta(elapsed_s: float, done: int, total: Optional[int]) -> str:
    if not total or total <= 0 or done == 0:
        return "ETA: N/A"
    rate = done / max(elapsed_s, 1e-9)
    remain = max(total - done, 0)
    eta_s = remain / max(rate, 1e-9)
    return f"ETA: {eta_s/3600:.2f} h @ {rate:.1f} items/s"

def open_env(path: str, readonly: bool, map_size: int = 0, subdir: bool | None = None):
    if subdir is None:
        subdir = os.path.isdir(path)
    return lmdb.open(
        path,
        readonly=readonly,
        lock=not readonly,
        subdir=subdir,
        map_size=map_size,
        max_dbs=1,
        readahead=False,
        meminit=False,
    )

def retry_mapfull(env: lmdb.Environment, txn: lmdb.Transaction, writer_fn,
                  grow_factor: float = 1.6, grow_min_bytes: int = 1 << 30,
                  verbose: bool = True):
    """Run writer_fn(txn); if MDB_MAP_FULL, grow mapsize and retry."""
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
            old = int(info.get("map_size", 0))
            new = max(int(old * grow_factor), old + grow_min_bytes)
            if verbose:
                print(f"[MAPFULL] Growing map_size {_fmt_bytes(old)} -> {_fmt_bytes(new)}", flush=True)
            env.set_mapsize(new)
            txn = env.begin(write=True)

def parse_width_from_meta_notes(notes: str, default_width: int = 12) -> int:
    m = re.search(r"sample:\{(\d+)d\}", notes or "")
    return int(m.group(1)) if m else default_width

def detect_width_from_first_key(txn: lmdb.Transaction) -> int:
    """If no meta notes, infer width from first ':tokens' key."""
    with txn.cursor() as cur:
        for k, _ in cur:
            if k.endswith(b":tokens") and k.startswith(b"sample:"):
                try:
                    middle = k[len(b"sample:") : k.rfind(b":")]
                    return len(middle)
                except Exception:
                    break
    return 12

def read_meta(txn: lmdb.Transaction):
    tokenizer = None
    max_length = None
    width = None
    has_mask = None
    count = None
    m = txn.get(b"__meta__")
    if m:
        try:
            meta = json.loads(m.decode("utf-8"))
            tokenizer = meta.get("tokenizer")
            max_length = meta.get("max_length")
            has_mask = meta.get("has_mask")
            count = meta.get("count")
            width = parse_width_from_meta_notes(meta.get("notes", ""), default_width=None)
        except Exception:
            pass
    return tokenizer, max_length, width, has_mask, count

def bytes_per_token(dtype_str: str) -> int:
    return {
        "int16": 2, "int32": 4, "int64": 8,
        "uint16": 2, "uint32": 4,
    }.get(dtype_str, 8)

def safe_decode_ascii(b: bytes, default="int64") -> str:
    try:
        return b.decode("ascii")
    except Exception:
        return default

def write_meta(txn: lmdb.Transaction, count: int, width: int,
               tokenizer: str | None, max_length: int | None, has_mask: bool | None):
    meta = {
        "version": 1,
        "count": int(count),
        "schema": {"tokens": "bytes", "modality_mask": "bytes(optional)", "type": "str"},
        "dtype": "int64",
        "has_mask": bool(has_mask) if has_mask is not None else None,
        "tokenizer": tokenizer,
        "max_length": max_length,
        "notes": f"keys: sample:{{{width}d}}:{{tokens|modality_mask|type|len}}",
    }
    txn.put(b"__meta__", json.dumps(meta, ensure_ascii=False).encode("utf-8"))

# -------------------- verification --------------------

def verify_lmdb(src_path: str, sample_probe: int | None = None,
                preview: int = 0, full_scan_missing: bool = False,
                max_missing: int = 10, log_every: int = 1_000_000):
    t0 = time.perf_counter()
    env = open_env(src_path, readonly=True)
    with env.begin(write=False) as txn:
        tokenizer, max_length, width, has_mask, count = read_meta(txn)
        if width is None:
            width = detect_width_from_first_key(txn)

        print(f"[VERIFY] src={src_path}")
        print(f"[VERIFY] tokenizer={tokenizer} max_length={max_length} has_mask(meta)={has_mask}")
        print(f"[VERIFY] meta.count={count} inferred width={width}")

        # preview前 N 个 tokens 键
        if preview > 0:
            print(f"[VERIFY] preview first {preview} ':tokens' keys:")
            shown = 0
            with txn.cursor() as cur:
                for k, _ in cur:
                    if k.endswith(b":tokens") and k.startswith(b"sample:"):
                        print("  ", k.decode("ascii", errors="ignore"))
                        shown += 1
                        if shown >= preview:
                            break

        # 统计实际 ':tokens' 个数
        n_tokens = 0
        with txn.cursor() as cur:
            for i, (k, _) in enumerate(cur, start=1):
                if k.endswith(b":tokens"):
                    n_tokens += 1
                if i % log_every == 0:
                    print(f"[VERIFY] scanned {i:,} keys ... found ':tokens'={n_tokens:,}", flush=True)

        print(f"[VERIFY] actual ':tokens' keys = {n_tokens:,}")
        if isinstance(count, int):
            diff = n_tokens - count
            print(f"[VERIFY] actual - meta.count = {diff:+,}")

        # 探测一个指定 index
        if sample_probe is not None:
            key = f"sample:{sample_probe:0{width}d}:tokens".encode("ascii")
            print(f"[VERIFY] exists[{sample_probe}] = {txn.get(key) is not None}")

        # 完全扫描缺失索引（很慢，慎用）
        if full_scan_missing and isinstance(count, int) and count > 0:
            print(f"[VERIFY] full scan for missing indices in [0, {count-1}] (will be slow)...")
            missing = []
            for idx in range(count):
                k = f"sample:{idx:0{width}d}:tokens".encode("ascii")
                if txn.get(k) is None:
                    missing.append(idx)
                    if len(missing) >= max_missing:
                        break
                if (idx + 1) % max(1, count // 100) == 0:
                    pct = (idx + 1) * 100.0 / count
                    print(f"  progress: {pct:.1f}%  checked={idx+1:,}  found_missing={len(missing)}", flush=True)
            if missing:
                print(f"[VERIFY] first {len(missing)} missing indices: {missing}")
            else:
                print("[VERIFY] no missing indices found in the first pass.")

    env.close()
    print(f"[VERIFY] done in {time.perf_counter() - t0:.2f}s")

# -------------------- repair --------------------

def repair_lmdb(src_path: str, dst_path: str,
                commit_interval: int = 100_000,
                grow_factor: float = 1.6,
                grow_min_gb: float = 1.0,
                log_every: int = 100_000,
                stop_after: int | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)

    # open source
    src_subdir = os.path.isdir(src_path)
    src_env = open_env(src_path, readonly=True, subdir=src_subdir)

    # read meta for tokenizer/max_length/width
    with src_env.begin(write=False) as txn_src:
        tokenizer, max_length, width, has_mask_meta, count_meta = read_meta(txn_src)
        if width is None:
            width = detect_width_from_first_key(txn_src)

    # prepare destination single-file LMDB
    if os.path.exists(dst_path):
        os.remove(dst_path)
    dst_env = open_env(dst_path, readonly=False, map_size=8 * (1 << 30), subdir=False)

    # initial meta
    with dst_env.begin(write=True) as txn_dst:
        write_meta(txn_dst, count=0, width=width, tokenizer=tokenizer, max_length=max_length, has_mask=False)

    grow_min_bytes = int(grow_min_gb * (1 << 30))
    copied = 0
    has_mask_out = False
    inferred_len_count = 0
    mask_trunc_count = 0
    mask_present_count = 0

    print(f"[REPAIR] src={src_path} (subdir={src_subdir}) -> dst={dst_path}")
    print(f"[REPAIR] width={width} tokenizer={tokenizer} max_length={max_length} meta.count={count_meta}")
    print(f"[REPAIR] commit_interval={commit_interval}  log_every={log_every}  grow_min={_fmt_bytes(grow_min_bytes)}")

    t0 = time.perf_counter()
    last_log_t = t0
    txn_dst = dst_env.begin(write=True)

    with src_env.begin(write=False) as txn_src:
        with txn_src.cursor() as cur:
            scanned_keys = 0
            for key, val in cur:
                scanned_keys += 1
                # only copy tokens entries
                if not (key.endswith(b":tokens") and key.startswith(b"sample:")):
                    continue

                # derive sibling keys in source
                base = key[:-len(b"tokens")]  # b"sample:000...:"
                k_type = base + b"type"
                k_len  = base + b"len"
                k_mask = base + b"modality_mask"

                dtype_bytes = txn_src.get(k_type) or b"int64"
                dtype_str = safe_decode_ascii(dtype_bytes, default="int64")
                len_bytes = txn_src.get(k_len)

                if len_bytes is not None and len(len_bytes) == 4:
                    n_tokens = struct.unpack("<I", len_bytes)[0]
                else:
                    # compute from bytes length and dtype
                    bpt = bytes_per_token(dtype_str)
                    n_tokens = len(val) // bpt
                    inferred_len_count += 1

                # modality_mask（可选；若过长则截断到 n_tokens）
                mask_bytes = txn_src.get(k_mask)
                if mask_bytes is not None:
                    mask_present_count += 1
                    if len(mask_bytes) > n_tokens:
                        mask_bytes = mask_bytes[:n_tokens]
                        mask_trunc_count += 1

                # new key prefix in destination
                new_prefix = f"sample:{copied:0{width}d}:".encode("ascii")

                def writer(tdst: lmdb.Transaction):
                    # tokens
                    tdst.put(new_prefix + b"tokens", val)
                    # type
                    tdst.put(new_prefix + b"type", dtype_bytes)
                    # len (always recomputed/reliable)
                    tdst.put(new_prefix + b"len", struct.pack("<I", int(n_tokens)))
                    # modality_mask (optional)
                    if mask_bytes is not None:
                        tdst.put(new_prefix + b"modality_mask", mask_bytes)

                txn_dst = retry_mapfull(dst_env, txn_dst, writer,
                                        grow_factor=grow_factor, grow_min_bytes=grow_min_bytes, verbose=True)
                copied += 1

                # logging
                if copied % log_every == 0:
                    now = time.perf_counter()
                    span = now - last_log_t
                    total_span = now - t0
                    rate = log_every / max(span, 1e-9)
                    eta_str = _fmt_eta(total_span, copied, count_meta)
                    print(f"[REPAIR] copied={copied:,}  scanned_keys={scanned_keys:,}  rate={rate:.1f} items/s  {eta_str}", flush=True)
                    last_log_t = now

                # commit
                if copied % commit_interval == 0:
                    def write_meta_partial(tdst):
                        write_meta(tdst, count=copied, width=width,
                                   tokenizer=tokenizer, max_length=max_length, has_mask=has_mask_out or (mask_present_count>0))
                    txn_dst = retry_mapfull(dst_env, txn_dst, write_meta_partial,
                                            grow_factor=grow_factor, grow_min_bytes=grow_min_bytes, verbose=True)
                    txn_dst.commit()
                    txn_dst = dst_env.begin(write=True)
                    print(f"[REPAIR] committed meta at copied={copied:,}", flush=True)

                # optional stop for小样本测试
                if stop_after is not None and copied >= stop_after:
                    print(f"[REPAIR] stop_after reached: {stop_after} samples", flush=True)
                    break

    # final meta
    def write_meta_final(tdst):
        write_meta(tdst, count=copied, width=width,
                   tokenizer=tokenizer, max_length=max_length, has_mask=(has_mask_out or (mask_present_count>0)))
    txn_dst = retry_mapfull(dst_env, txn_dst, write_meta_final,
                            grow_factor=grow_factor, grow_min_bytes=grow_min_bytes, verbose=True)
    txn_dst.commit()

    src_env.close()
    dst_env.sync()
    dst_env.close()

    elapsed = time.perf_counter() - t0
    print(f"[DONE] repaired -> {dst_path}")
    print(f"       total_copied={copied:,}  elapsed={elapsed/3600:.2f} h  avg_rate={copied/max(elapsed,1e-9):.1f} items/s")
    print(f"       inferred_len_count={inferred_len_count:,}  mask_present_count={mask_present_count:,}  mask_trunc_count={mask_trunc_count:,}")

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser("Repair DNABERT2 LMDB (sample:{Nd}:{tokens|type|len|modality_mask}) to dense indices")
    ap.add_argument("--src", required=True, help="Source LMDB path (file or directory)")
    ap.add_argument("--dst", required=False, help="Destination LMDB file (single-file)")
    ap.add_argument("--verify-only", action="store_true", help="Only verify counts and example key existence")
    ap.add_argument("--probe-index", type=int, default=None, help="Check existence of a specific index in source")
    ap.add_argument("--preview", type=int, default=0, help="Preview first N ':tokens' keys on verify")
    ap.add_argument("--full-scan-missing", action="store_true", help="Verify: full scan 0..meta.count-1 and list first missing indices (slow)")
    ap.add_argument("--max-missing", type=int, default=10, help="Verify: print at most K missing indices when full-scan-missing")
    ap.add_argument("--commit-interval", type=int, default=100000)
    ap.add_argument("--grow-factor", type=float, default=1.6)
    ap.add_argument("--grow-min-gb", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=100000, help="Print progress every N copied samples")
    ap.add_argument("--stop-after", type=int, default=None, help="Copy at most N samples (for dry-run)")
    args = ap.parse_args()

    if args.verify_only:
        verify_lmdb(args.src,
                    sample_probe=args.probe_index,
                    preview=args.preview,
                    full_scan_missing=args.full_scan_missing,
                    max_missing=args.max_missing,
                    log_every=max(args.log_every, 1_000_000))
        return

    if not args.dst:
        ap.error("--dst is required unless --verify-only is set")

    repair_lmdb(
        src_path=args.src,
        dst_path=args.dst,
        commit_interval=args.commit_interval,
        grow_factor=args.grow_factor,
        grow_min_gb=args.grow_min_gb,
        log_every=args.log_every,
        stop_after=args.stop_after,
    )

if __name__ == "__main__":
    main()
