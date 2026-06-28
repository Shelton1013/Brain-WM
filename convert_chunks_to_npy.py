"""
One-time converter: chunk_XXXX.npz → chunk_XXXX_trials.npy + chunk_XXXX_sids.npy

Why: np.load(npz, mmap_mode='r') does NOT work — accessing npz['trials']
loads the whole array into RAM. For huge chunked caches (1+ TB total),
loading every chunk per DDP rank causes OOM.

Solution: extract each .npz into two side-car .npy files. dataset_multi.py
will prefer the .npy files when present and load them with mmap_mode='r'
(true OS-level memory-mapping — pages in/out on access).

The original .npz files are left in place; remove them manually after
verifying the new training run works.

Usage:
    python convert_chunks_to_npy.py \\
        --chunk_dir /home/pxieaf/home2/dataset_cache_cbramod/tueg_a5bae231e0f17af5
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk_dir", required=True,
                   help="Directory containing chunk_XXXX.npz files")
    p.add_argument("--keep_npz", action="store_true",
                   help="If set, do NOT delete the original .npz after extract")
    args = p.parse_args()

    chunk_dir = Path(args.chunk_dir)
    if not chunk_dir.is_dir():
        print(f"ERROR: {chunk_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    npz_files = sorted(chunk_dir.glob("chunk_*.npz"))
    if not npz_files:
        print(f"ERROR: no chunk_*.npz files found in {chunk_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(npz_files)} npz chunks in {chunk_dir}")
    total_in_gb = sum(f.stat().st_size for f in npz_files) / 1e9
    print(f"Total input: {total_in_gb:.2f} GB")

    t0 = time.time()
    total_out_gb = 0.0
    n_done = 0
    n_skipped = 0
    for npz_file in npz_files:
        stem = npz_file.stem  # e.g. "chunk_0000"
        trials_npy = chunk_dir / f"{stem}_trials.npy"
        sids_npy = chunk_dir / f"{stem}_sids.npy"

        if trials_npy.exists() and sids_npy.exists():
            n_skipped += 1
            n_done += 1
            print(f"  [{n_done}/{len(npz_files)}] {stem}: already extracted, "
                  f"skipping", flush=True)
            continue

        t_start = time.time()
        npz = np.load(str(npz_file))
        trials = npz["trials"]
        sids = npz["subject_ids"]

        # np.save writes header + raw bytes, mmap-friendly
        np.save(str(trials_npy), trials)
        np.save(str(sids_npy), sids)

        out_gb = (trials_npy.stat().st_size + sids_npy.stat().st_size) / 1e9
        total_out_gb += out_gb
        dt = time.time() - t_start
        n_done += 1
        print(f"  [{n_done}/{len(npz_files)}] {stem}: "
              f"{trials.shape} → {out_gb:.2f} GB in {dt:.1f}s", flush=True)

        if not args.keep_npz:
            npz_file.unlink()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Output total: {total_out_gb:.2f} GB ({n_skipped} skipped)")
    if not args.keep_npz:
        print(f"Original .npz files deleted. Re-run prebuild if you need them back.")


if __name__ == "__main__":
    main()
