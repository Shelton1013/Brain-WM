"""Pre-build TUEG dataset cache in a single process to avoid DDP timeout.

Run this ONCE before launching DDP training. After it finishes, the
dataset_cache_dir contains the TUEG cache and DDP training will hit it
instantly (no 10-min handshake timeout).

LEAKAGE PREVENTION: pass --exclude_from_dirs with one or more downstream
eval corpus paths (e.g., TUAB, TUEV). All patient IDs found in those
directories will be EXCLUDED from the TUEG cache to prevent pretraining
on downstream eval recordings. Patient ID = first 8 chars of EDF
basename (TUH anonymization scheme; matches _sid_before_underscore).

Usage (apples-to-apples with LaBraM pretrain protocol):
    python prebuild_tueg_cache.py \\
        --tueg_dir /home/pxieaf/home2/tuh/tuh_eeg/v2.0.1/edf \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --sample_rate 256 --trial_duration_s 4 --min_channels 19 \\
        --exclude_from_dirs \\
            /home/pxieaf/home2/tuh/tuh_eeg_abnormal/v3.0.1/edf \\
            /home/pxieaf/home2/tuh/tuh_eeg_events/v2.0.1/edf
"""

import argparse
import time
from pathlib import Path

from dataset_multi import EDFDirectoryDataset, _sid_before_underscore


def _collect_patient_ids(dirs):
    """Walk one or more directories, return set of patient IDs
    (first 8 chars of EDF basename, matching TUH anonymization)."""
    ids = set()
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            print(f"  [exclude] WARN: {d} not a directory, skipping")
            continue
        n_before = len(ids)
        for edf in d.rglob("*.edf"):
            pid = _sid_before_underscore(edf)
            if pid:
                ids.add(pid)
        print(f"  [exclude] {d}: {len(ids) - n_before} new patient IDs")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tueg_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=256)
    ap.add_argument("--trial_duration_s", type=int, default=4)
    ap.add_argument("--min_channels", type=int, default=19)
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--use_ea", action="store_true", default=True)
    ap.add_argument("--exclude_from_dirs", type=str, nargs="*", default=[],
                    help="Paths to downstream eval corpus EDF roots. All "
                         "patient IDs found there will be excluded from "
                         "TUEG pretrain. Pass TUAB + TUEV here.")
    ap.add_argument("--exclude_id_file", type=str, default=None,
                    help="Optional: path to a text file with one patient ID "
                         "per line. Merged with --exclude_from_dirs.")
    args = ap.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    # ── Collect exclude set ──
    exclude_ids = set()
    if args.exclude_from_dirs:
        print(f"[exclude] Collecting patient IDs from {len(args.exclude_from_dirs)} dirs")
        exclude_ids |= _collect_patient_ids(args.exclude_from_dirs)
    if args.exclude_id_file:
        with open(args.exclude_id_file) as f:
            for line in f:
                pid = line.strip()
                if pid:
                    exclude_ids.add(pid)
        print(f"[exclude] +{sum(1 for _ in open(args.exclude_id_file)) if args.exclude_id_file else 0} from file")
    if exclude_ids:
        print(f"[exclude] Total exclude set: {len(exclude_ids)} patient IDs")
    else:
        print(f"[exclude] WARN: no patients excluded — TUEG cache will overlap "
              f"with downstream eval recordings. Pass --exclude_from_dirs "
              f"for paper-grade no-leakage pretrain.")

    t0 = time.time()
    print(f"\n[prebuild] Loading TUEG from {args.tueg_dir}")
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
        exclude_patient_ids=exclude_ids,
    )

    elapsed = time.time() - t0
    print(f"\n[prebuild] Done in {elapsed/60:.1f} min")
    print(f"[prebuild] Trials: {len(ds)}")
    print(f"[prebuild] Subjects: {ds.n_subjects}")
    print(f"[prebuild] Excluded: {len(exclude_ids)} patient IDs")
    print(f"[prebuild] Cache files in {args.cache_dir}:")
    for f in sorted(Path(args.cache_dir).glob("tueg*.pt")):
        size_gb = f.stat().st_size / 1e9
        print(f"  {f.name}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
