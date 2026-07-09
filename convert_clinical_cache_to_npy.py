"""Convert TUAB/TUEV .pt list caches → mmap-able .npy side-cars.

WHY: the .pt clinical cache stores trials as a Python list of ~370k small
arrays (~70 GB fully resident in RAM). Each eval loads that, and DataLoader
workers fork-copy it → ~150 GB per eval; two concurrent TUAB evals exhaust
RAM and HANG (GPU-independent). This writes, per cache:
    <stem>_trials.npy   stacked [N, T, C] float32  (loaded with mmap_mode='r')
    <stem>_meta.pt      labels / recording_ids / patient_ids / electrode_names
dataset_tuh_clinical._try_load_mmap_sidecar prefers these when present, so the
trials stay on disk, page in on access, and are SHARED across workers and
across concurrent evals → low RAM, and many TUAB evals can run at once.

Uses np.lib.format.open_memmap to stream to disk (peak RAM ≈ the .pt list, not
doubled).

Usage:
    python convert_clinical_cache_to_npy.py \\
        --cache_dir /home/pxieaf/home2/dataset_cache_labram \\
        --pattern 'tuab_*.pt'
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--pattern", default="tu*_*.pt",
                    help="glob for .pt caches (e.g. 'tuab_*.pt')")
    ap.add_argument("--keep_pt", action="store_true",
                    help="keep the original .pt after conversion")
    args = ap.parse_args()

    pts = sorted(glob.glob(os.path.join(args.cache_dir, args.pattern)))
    pts = [p for p in pts if not p.endswith("_meta.pt")]  # never re-convert meta
    if not pts:
        print(f"No .pt caches matching {args.pattern} in {args.cache_dir}")
        sys.exit(1)
    print(f"Found {len(pts)} cache(s) to convert")

    for p in pts:
        stem = p[:-3] if p.endswith(".pt") else p
        trials_npy = stem + "_trials.npy"
        meta_pt = stem + "_meta.pt"
        if os.path.exists(trials_npy) and os.path.exists(meta_pt):
            print(f"  [skip] {Path(p).name} already converted")
            continue

        t0 = time.time()
        print(f"  loading {Path(p).name} ...", flush=True)
        d = torch.load(p, weights_only=False)
        trials = d["trials"]
        n = len(trials)
        t0_arr = np.asarray(trials[0], dtype=np.float32)
        T, C = t0_arr.shape

        # Stream to disk via memmap (peak RAM ≈ the .pt list, not doubled).
        out = np.lib.format.open_memmap(
            trials_npy, mode="w+", dtype=np.float32, shape=(n, T, C))
        for i, t in enumerate(trials):
            out[i] = np.asarray(t, dtype=np.float32)
        out.flush()
        del out

        meta = {k: d.get(k) for k in
                ("labels", "recording_ids", "patient_ids", "electrode_names")}
        torch.save(meta, meta_pt)

        gb = os.path.getsize(trials_npy) / 1e9
        print(f"  → {Path(trials_npy).name}: [{n}, {T}, {C}]  {gb:.1f} GB  "
              f"in {time.time()-t0:.0f}s", flush=True)
        del d, trials
        if not args.keep_pt:
            os.remove(p)
            print(f"    removed {Path(p).name}")

    print("Done. Clinical evals will now mmap these side-cars (low RAM + "
          "concurrent-safe).")


if __name__ == "__main__":
    main()
