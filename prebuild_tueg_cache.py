"""Pre-build TUEG dataset cache (parallel, unbuffered output).

Multi-process version of TUEG preprocessing. Each worker handles one
patient (all of that patient's EDFs are loaded, concatenated, EA'd,
and segmented in the same process to preserve EA semantics).

The output cache file is BIT-IDENTICAL in structure to what the original
EDFDirectoryDataset's single-process path produces, and uses the SAME
cache key, so the main training pipeline will hit this cache transparently.

Run with `python -u` (or default — we also call sys.stdout.reconfigure
internally) so progress prints reach the log file in real time.

LEAKAGE PREVENTION: pass --exclude_from_dirs with one or more downstream
eval corpus paths (e.g., TUAB, TUEV). All patient IDs found in those
directories will be EXCLUDED from the TUEG cache to prevent pretraining
on downstream eval recordings.

Usage:
    python prebuild_tueg_cache.py \\
        --tueg_dir /home/pxieaf/home2/tuh/tuh_eeg/v2.0.1/edf \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --sample_rate 256 --trial_duration_s 4 --min_channels 19 \\
        --n_workers 16 \\
        --exclude_from_dirs \\
            /home/pxieaf/home2/tuh/tuh_eeg_abnormal/v3.0.1/edf \\
            /home/pxieaf/home2/tuh/tuh_eeg_events/v2.0.1/edf
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset_multi import (
    _sid_before_underscore,
    pick_common_channels,
    _cache_key,
    _save_cache,
    _try_load_cache,
)

try:
    import mne
    mne.set_log_level("ERROR")
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# ============================================================
# Worker function — runs in a subprocess for one patient
# ============================================================

def _process_patient(args_tuple):
    """Load all EDFs for one patient, return (subject_pid, trials [list of np],
    ch_names) or (subject_pid, [], None) on failure.

    `args_tuple = (patient_id, edf_paths, sample_rate, trial_duration_s,
                   use_ea, min_channels)`
    """
    patient_id, edf_paths, sample_rate, trial_duration_s, use_ea, min_channels = args_tuple
    trial_samples = sample_rate * trial_duration_s
    min_samples_for_filter = int(40 * sample_rate)

    if not MNE_AVAILABLE:
        return patient_id, [], None

    subj_recordings = []
    matched_ch_names = None

    for edf_path in edf_paths:
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            if raw.info["sfreq"] != sample_rate:
                raw.resample(sample_rate, verbose=False)
            if raw.n_times < min_samples_for_filter:
                continue
            raw.filter(0.1, 75.0, verbose=False)
            ch_indices, ch_names = pick_common_channels(raw.ch_names)
            if len(ch_indices) < min_channels:
                continue
            if matched_ch_names is None:
                matched_ch_names = ch_names
            arr = raw.get_data()[ch_indices].T.astype(np.float32)
            subj_recordings.append(arr)
        except Exception:
            continue

    if not subj_recordings:
        return patient_id, [], None

    eeg = np.vstack(subj_recordings)

    if use_ea:
        # Euclidean Alignment per-patient
        try:
            from dataset_multi import euclidean_alignment
            eeg = euclidean_alignment(eeg)
        except Exception:
            pass  # if EA not available / fails, fall through

    # Segment into non-overlapping trials, z-score per trial
    trials = []
    n_trials = len(eeg) // trial_samples
    for t in range(n_trials):
        start = t * trial_samples
        trial = eeg[start:start + trial_samples]
        mean = trial.mean(axis=0, keepdims=True)
        std = trial.std(axis=0, keepdims=True) + 1e-8
        trials.append(((trial - mean) / std).astype(np.float32))

    return patient_id, trials, matched_ch_names


# ============================================================
# Exclude set
# ============================================================

def _collect_patient_ids(dirs):
    ids = set()
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            print(f"  [exclude] WARN: {d} not a directory, skipping", flush=True)
            continue
        n_before = len(ids)
        for edf in d.rglob("*.edf"):
            pid = _sid_before_underscore(edf)
            if pid:
                ids.add(pid)
        print(f"  [exclude] {d}: {len(ids) - n_before} new patient IDs",
              flush=True)
    return ids


# ============================================================
# Main
# ============================================================

def main():
    # Force unbuffered stdout (in case run without python -u)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--tueg_dir", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=256)
    ap.add_argument("--trial_duration_s", type=int, default=4)
    ap.add_argument("--min_channels", type=int, default=19)
    ap.add_argument("--max_patients", type=int, default=None)
    ap.add_argument("--use_ea", action="store_true", default=True)
    ap.add_argument("--exclude_from_dirs", type=str, nargs="*", default=[])
    ap.add_argument("--exclude_id_file", type=str, default=None)
    ap.add_argument("--n_workers", type=int, default=16,
                    help="Number of parallel worker processes.")
    args = ap.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    # ── Collect exclude set ──
    exclude_ids = set()
    if args.exclude_from_dirs:
        print(f"[exclude] Collecting patient IDs from "
              f"{len(args.exclude_from_dirs)} dirs", flush=True)
        exclude_ids |= _collect_patient_ids(args.exclude_from_dirs)
    if args.exclude_id_file:
        with open(args.exclude_id_file) as f:
            for line in f:
                pid = line.strip()
                if pid:
                    exclude_ids.add(pid)
        print(f"[exclude] Total after file merge: {len(exclude_ids)}",
              flush=True)
    if exclude_ids:
        print(f"[exclude] Total exclude set: {len(exclude_ids)} patient IDs",
              flush=True)
    else:
        print(f"[exclude] WARN: no patients excluded", flush=True)

    # ── Compute cache key (SAME formula as EDFDirectoryDataset) ──
    # so that this output is hit by the main training pipeline.
    exclude_hash = hashlib.md5(
        ",".join(sorted(exclude_ids)).encode()
    ).hexdigest()[:8] if exclude_ids else "none"
    cache_payload_key = _cache_key({
        "kind": "tueg",
        "data_dir": str(args.tueg_dir),
        "sample_rate": args.sample_rate,
        "trial_duration_s": args.trial_duration_s,
        "use_ea": args.use_ea,
        "max_files": None,            # we don't filter by max_files here
        "min_channels": args.min_channels,
        "grouped": True,              # we always group by patient
        "exclude": exclude_hash,
        "n_excluded": len(exclude_ids),
    })
    cache_path = Path(args.cache_dir) / f"tueg_{cache_payload_key}.pt"
    print(f"[prebuild] Cache target: {cache_path}", flush=True)

    if _try_load_cache(cache_path) is not None:
        print(f"[prebuild] Cache already exists, exiting.", flush=True)
        return

    # ── Scan EDFs, group by patient, apply exclude ──
    print(f"\n[prebuild] Scanning {args.tueg_dir} for EDF files...", flush=True)
    t_scan = time.time()
    all_edfs = sorted(Path(args.tueg_dir).rglob("*.edf"))
    print(f"[prebuild] Found {len(all_edfs)} EDF files in "
          f"{time.time()-t_scan:.1f}s", flush=True)

    patient_files: dict[str, list[Path]] = defaultdict(list)
    n_excluded_files = 0
    for f in all_edfs:
        pid = _sid_before_underscore(f)
        if not pid:
            continue
        if pid in exclude_ids:
            n_excluded_files += 1
            continue
        patient_files[pid].append(f)
    patients = sorted(patient_files.keys())
    if args.max_patients:
        patients = patients[:args.max_patients]
    print(f"[prebuild] After exclusion: {len(patients)} patients, "
          f"{sum(len(patient_files[p]) for p in patients)} EDFs "
          f"({n_excluded_files} EDFs excluded)", flush=True)

    # ── Process patients in parallel ──
    worker_args = [
        (p, patient_files[p], args.sample_rate, args.trial_duration_s,
         args.use_ea, args.min_channels)
        for p in patients
    ]

    t_proc = time.time()
    all_trials: list[np.ndarray] = []
    all_subject_ids: list[int] = []
    electrode_names: list[str] | None = None
    n_done = 0
    n_skipped = 0

    print(f"\n[prebuild] Starting parallel processing with "
          f"{args.n_workers} workers...", flush=True)

    with mp.Pool(args.n_workers) as pool:
        for patient_id, trials, ch_names in pool.imap_unordered(
                _process_patient, worker_args, chunksize=2):
            n_done += 1
            if not trials:
                n_skipped += 1
            else:
                subj_idx = len(set(all_subject_ids))  # next subject index
                for trial in trials:
                    all_trials.append(trial)
                    all_subject_ids.append(subj_idx)
                if electrode_names is None and ch_names is not None:
                    electrode_names = ch_names

            if n_done % 50 == 0 or n_done == len(worker_args):
                elapsed = time.time() - t_proc
                rate = n_done / max(elapsed, 1e-6)
                eta_min = (len(worker_args) - n_done) / max(rate, 1e-6) / 60
                print(f"  [{n_done}/{len(worker_args)}] "
                      f"trials={len(all_trials)} "
                      f"skipped={n_skipped} "
                      f"elapsed={elapsed/60:.1f}m "
                      f"rate={rate:.2f}p/s "
                      f"ETA={eta_min:.1f}m", flush=True)

    n_subjects = len(set(all_subject_ids))

    # ── Save in EDFDirectoryDataset-compatible format ──
    print(f"\n[prebuild] Saving cache...", flush=True)
    _save_cache(cache_path, {
        "trials": all_trials,
        "subject_ids": all_subject_ids,
        "electrode_names": electrode_names,
        "n_subjects": n_subjects,
    })

    elapsed_total = time.time() - t_proc
    print(f"\n[prebuild] Done in {elapsed_total/60:.1f} min", flush=True)
    print(f"[prebuild] Trials: {len(all_trials)}", flush=True)
    print(f"[prebuild] Subjects: {n_subjects}", flush=True)
    print(f"[prebuild] Skipped patients: {n_skipped}", flush=True)
    print(f"[prebuild] Excluded patient IDs: {len(exclude_ids)}", flush=True)
    print(f"[prebuild] Cache file:", flush=True)
    if cache_path.exists():
        size_gb = cache_path.stat().st_size / 1e9
        print(f"  {cache_path.name}  ({size_gb:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
