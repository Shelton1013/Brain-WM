"""Pre-build TUEG dataset cache in a single process to avoid DDP timeout.

Run this ONCE before launching DDP training. After it finishes, the
dataset_cache_dir contains the TUEG cache and DDP training will hit it
instantly (no 10-min handshake timeout).

Usage:
    python prebuild_tueg_cache.py \\
        --tueg_dir /home/pxieaf/home2/tuh/tuh_eeg/v2.0.1/edf \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --sample_rate 256 --trial_duration_s 4 --min_channels 19
"""

import argparse
import time
from pathlib import Path

from dataset_multi import EDFDirectoryDataset, _sid_before_underscore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tueg_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=256)
    ap.add_argument("--trial_duration_s", type=int, default=4)
    ap.add_argument("--min_channels", type=int, default=19)
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--use_ea", action="store_true", default=True)
    args = ap.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[prebuild] Loading TUEG from {args.tueg_dir}")
    print(f"[prebuild] Cache target: {args.cache_dir}")

    ds = EDFDirectoryDataset(
        data_dir=args.tueg_dir,
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        use_ea=args.use_ea,
        max_files=args.max_files,
        min_channels=args.min_channels,
        subject_id_fn=_sid_before_underscore,
        cache_dir=args.cache_dir,
        cache_tag="tueg",
    )

    elapsed = time.time() - t0
    print(f"[prebuild] Done in {elapsed/60:.1f} min")
    print(f"[prebuild] Trials: {len(ds)}")
    print(f"[prebuild] Subjects: {ds.n_subjects}")
    print(f"[prebuild] Cache files in {args.cache_dir}:")
    for f in sorted(Path(args.cache_dir).glob("tueg*.pt")):
        size_gb = f.stat().st_size / 1e9
        print(f"  {f.name}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
