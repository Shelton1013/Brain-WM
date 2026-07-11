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

    CBraMod-style quality filters applied (when enabled via args):
      - drop_short_recording_min: skip EDF if shorter than N minutes
      - trim_start_end_sec: discard first/last N seconds of each recording
      - notch_freq: notch filter at given Hz (60 for US/TUH, 50 for EU)
      - reject_abs_uv: drop any trial where |x| exceeds N µV (before norm)

    `args_tuple = (patient_id, edf_paths, sample_rate, trial_duration_s,
                   use_ea, min_channels, normalization,
                   drop_short_recording_min, trim_start_end_sec,
                   notch_freq, reject_abs_uv)`
    """
    (patient_id, edf_paths, sample_rate, trial_duration_s,
     use_ea, min_channels, normalization,
     drop_short_recording_min, trim_start_end_sec,
     notch_freq, reject_abs_uv, highpass_hz, lowpass_hz) = args_tuple
    trial_samples = sample_rate * trial_duration_s
    min_samples_for_filter = int(40 * sample_rate)

    if not MNE_AVAILABLE:
        return patient_id, [], None

    subj_recordings = []
    matched_ch_names = None

    last_err = None
    for edf_path in edf_paths:
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            if raw.info["sfreq"] != sample_rate:
                raw.resample(sample_rate, verbose=False)

            # Drop too-short recordings (after resample so duration is consistent)
            if drop_short_recording_min and drop_short_recording_min > 0:
                if raw.n_times < drop_short_recording_min * 60 * sample_rate:
                    continue
            if raw.n_times < min_samples_for_filter:
                continue

            # Bandpass + optional notch (MNE requires array-like for freqs)
            raw.filter(highpass_hz, lowpass_hz, verbose=False)
            if notch_freq and notch_freq > 0:
                raw.notch_filter([float(notch_freq)], verbose=False)

            ch_indices, ch_names = pick_common_channels(raw.ch_names)
            if len(ch_indices) < min_channels:
                continue
            if matched_ch_names is None:
                matched_ch_names = ch_names
            arr = raw.get_data()[ch_indices].T.astype(np.float32)

            # Trim first/last N seconds (remove start/end artifacts).
            # Skip the file if trimming leaves too little data.
            if trim_start_end_sec and trim_start_end_sec > 0:
                trim_samples = trim_start_end_sec * sample_rate
                if arr.shape[0] <= 2 * trim_samples + trial_samples:
                    continue
                arr = arr[trim_samples:-trim_samples]

            subj_recordings.append(arr)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue

    if not subj_recordings:
        # Return a marker so caller can log why all EDFs for this patient failed
        return patient_id, [], last_err or "no_edfs_processed"

    eeg = np.vstack(subj_recordings)

    # Amplitude-reject reference: capture BEFORE Euclidean Alignment and BEFORE
    # normalization. EA (X @ R^-1/2) whitens the spatial covariance to identity,
    # which destroys the physical amplitude scale — a µV threshold checked on
    # post-EA data (O(1) magnitude) would never fire. This `eeg` is MNE's native
    # Volts (get_data() applies no unit conversion); the per-trial check below
    # scales the µV threshold to Volts accordingly.
    eeg_unnormed_for_amp_check = eeg

    if use_ea:
        # Euclidean Alignment per-patient
        try:
            from dataset_multi import euclidean_alignment
            eeg = euclidean_alignment(eeg)
        except Exception:
            pass  # if EA not available / fails, fall through

    # Recording-level robust normalization BEFORE segmentation (Laya-style)
    if normalization == "per_recording_robust":
        from dataset_multi import normalize_signal
        eeg = normalize_signal(eeg, "per_recording_robust")

    # Segment into non-overlapping trials.
    # Filter out: NaN/Inf trials (EA can produce them); trials exceeding
    # reject_abs_uv (CBraMod-style amplitude reject; checked on pre-norm µV).
    trials = []
    n_trials = len(eeg) // trial_samples
    for t in range(n_trials):
        start = t * trial_samples
        trial = eeg[start:start + trial_samples]

        # Amplitude reject (on pre-EA, pre-normalization data). Data is in
        # Volts (MNE native), so convert the µV threshold to Volts (×1e-6).
        if reject_abs_uv and reject_abs_uv > 0:
            raw_trial = eeg_unnormed_for_amp_check[start:start + trial_samples]
            if np.abs(raw_trial).max() > reject_abs_uv * 1e-6:
                continue

        if normalization == "per_trial_zscore":
            # Legacy: per-trial z-score
            mean = trial.mean(axis=0, keepdims=True)
            std = trial.std(axis=0, keepdims=True) + 1e-8
            normed = ((trial - mean) / std).astype(np.float32)
        else:
            # per_recording_robust already applied above
            normed = trial.astype(np.float32)
        if not np.isfinite(normed).all():
            continue
        trials.append(normed)

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
    ap.add_argument("--normalization", type=str, default="per_trial_zscore",
                    choices=["per_trial_zscore", "per_recording_robust"],
                    help="per_trial_zscore: legacy 4s window z-score (destroys "
                         "long-range amplitude). per_recording_robust: Defossez "
                         "2022 / Laya style; (x-median)/(IQR/1.349) per recording "
                         "BEFORE segmentation. Use per_recording_robust for "
                         "clinical task transfer (TUAB/TUEV).")
    ap.add_argument("--n_workers", type=int, default=16,
                    help="Number of parallel worker processes.")
    # CBraMod-style quality filters (paper §3.1)
    ap.add_argument("--drop_short_recording_min", type=float, default=0.0,
                    help="Drop recordings shorter than N minutes. "
                         "CBraMod uses 5. Default 0 = no filter (legacy).")
    ap.add_argument("--trim_start_end_sec", type=int, default=0,
                    help="Discard first/last N seconds of each recording "
                         "(start/end artifacts). CBraMod uses 60. "
                         "Default 0 = no trim.")
    ap.add_argument("--highpass_hz", type=float, default=0.1,
                    help="bandpass high-pass cutoff (0.5 recommended for v3 to "
                         "clean sub-delta drift while keeping delta)")
    ap.add_argument("--lowpass_hz", type=float, default=75.0,
                    help="bandpass low-pass cutoff (45 recommended for v3 to kill "
                         "EMG/powerline that pollute beta/gamma band targets)")
    ap.add_argument("--notch_freq", type=float, default=0.0,
                    help="Notch filter at given Hz (60 for US/TUH data, "
                         "50 for EU). Default 0 = no notch.")
    ap.add_argument("--reject_abs_uv", type=float, default=0.0,
                    help="Drop any trial where |x| exceeds N µV (pre-norm). "
                         "CBraMod uses 100. Default 0 = no reject. "
                         "WARNING: 100 is very aggressive and may drop 60-70%% "
                         "of TUH clinical trials.")
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
    # Quality-filter fields are appended ONLY when non-default to keep the
    # cache hash backward-compatible: a run with all filters off produces
    # the same hash as the legacy (pre-patch) run.
    payload = {
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
        "normalization": args.normalization,
    }
    if args.drop_short_recording_min > 0:
        payload["drop_short_min"] = args.drop_short_recording_min
    if args.trim_start_end_sec > 0:
        payload["trim_sec"] = args.trim_start_end_sec
    if args.notch_freq > 0:
        payload["notch_hz"] = args.notch_freq
    if args.reject_abs_uv > 0:
        payload["reject_uv"] = args.reject_abs_uv
    cache_payload_key = _cache_key(payload)
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

    # ── Process patients in parallel, FLUSH chunks to disk periodically
    # to avoid pickling a 180 GB list-of-numpy-arrays in one shot
    # (Python pickle of list[ndarray] needs ~2× RAM and is extremely slow).
    worker_args = [
        (p, patient_files[p], args.sample_rate, args.trial_duration_s,
         args.use_ea, args.min_channels, args.normalization,
         args.drop_short_recording_min, args.trim_start_end_sec,
         args.notch_freq, args.reject_abs_uv, args.highpass_hz, args.lowpass_hz)
        for p in patients
    ]

    CHUNK_SIZE_PATIENTS = 500   # flush a chunk every N patients
    chunk_dir = cache_path.with_suffix("")  # tueg_<hash>/
    chunk_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prebuild] Chunk dir: {chunk_dir}", flush=True)

    t_proc = time.time()
    chunk_trials: list[np.ndarray] = []
    chunk_subject_ids: list[int] = []
    all_subject_count = 0
    electrode_names: list[str] | None = None
    n_done = 0
    n_skipped = 0
    chunk_idx = 0
    total_trials = 0
    chunk_files: list[Path] = []

    def _flush_chunk():
        """Stack chunk_trials into one numpy array and save to disk."""
        nonlocal chunk_trials, chunk_subject_ids, chunk_idx
        if not chunk_trials:
            return
        # Stack (incurs 2× memory briefly, ok per chunk = ~38 GB for 500 patients)
        stacked = np.stack(chunk_trials).astype(np.float32)
        chunk_subj_ids_arr = np.array(chunk_subject_ids, dtype=np.int64)
        chunk_file = chunk_dir / f"chunk_{chunk_idx:04d}.npz"
        np.savez(str(chunk_file), trials=stacked, subject_ids=chunk_subj_ids_arr)
        chunk_files.append(chunk_file)
        size_gb = chunk_file.stat().st_size / 1e9
        print(f"    ↑ chunk {chunk_idx:04d}: {len(chunk_trials)} trials "
              f"({size_gb:.2f} GB) saved", flush=True)
        chunk_idx += 1
        # Free Python objects (helps GC reclaim memory)
        chunk_trials.clear()
        chunk_subject_ids.clear()
        del stacked
        del chunk_subj_ids_arr

    print(f"\n[prebuild] Starting parallel processing with "
          f"{args.n_workers} workers, chunked save every "
          f"{CHUNK_SIZE_PATIENTS} patients...", flush=True)

    patients_in_chunk = 0
    error_samples: list[str] = []  # collect first few skip reasons for debug

    def _iter_results():
        """Yield (pid, trials, ch_names_or_err) tuples — serial if n_workers<=1
        (drops mp.Pool overhead and exposes worker exceptions cleanly), else
        parallel via mp.Pool.imap_unordered."""
        if args.n_workers <= 1:
            for wa in worker_args:
                yield _process_patient(wa)
        else:
            with mp.Pool(args.n_workers) as pool:
                yield from pool.imap_unordered(
                    _process_patient, worker_args, chunksize=2)

    for patient_id, trials, ch_names_or_err in _iter_results():
        n_done += 1
        patients_in_chunk += 1
        if not trials:
            n_skipped += 1
            # ch_names_or_err contains the error string for skipped patients
            if isinstance(ch_names_or_err, str) and len(error_samples) < 5:
                error_samples.append(f"  [skip-sample] {patient_id}: "
                                     f"{ch_names_or_err}")
                print(error_samples[-1], flush=True)
        else:
            ch_names = ch_names_or_err   # success path: matched_ch_names
            subj_idx = all_subject_count
            all_subject_count += 1
            for trial in trials:
                chunk_trials.append(trial)
                chunk_subject_ids.append(subj_idx)
                total_trials += 1
            if electrode_names is None and ch_names is not None:
                electrode_names = ch_names

        if n_done % 50 == 0 or n_done == len(worker_args):
            elapsed = time.time() - t_proc
            rate = n_done / max(elapsed, 1e-6)
            eta_min = (len(worker_args) - n_done) / max(rate, 1e-6) / 60
            print(f"  [{n_done}/{len(worker_args)}] "
                  f"trials={total_trials} "
                  f"in_chunk={len(chunk_trials)} "
                  f"skipped={n_skipped} "
                  f"elapsed={elapsed/60:.1f}m "
                  f"rate={rate:.2f}p/s "
                  f"ETA={eta_min:.1f}m", flush=True)

        # Flush chunk every CHUNK_SIZE_PATIENTS patients
        if patients_in_chunk >= CHUNK_SIZE_PATIENTS:
            _flush_chunk()
            patients_in_chunk = 0

    # Flush final partial chunk
    _flush_chunk()

    # ── Save manifest pointing to chunks ──
    print(f"\n[prebuild] Saving manifest...", flush=True)
    manifest = {
        "format": "chunked_npz_v1",
        "chunk_files": [str(f.name) for f in chunk_files],
        "n_total_trials": total_trials,
        "n_subjects": all_subject_count,
        "electrode_names": electrode_names,
    }
    # Manifest goes alongside chunk dir; cache_path remains the manifest file
    # so EDFDirectoryDataset's cache hit logic can find it.
    _save_cache(cache_path, manifest)

    elapsed_total = time.time() - t_proc
    total_chunk_gb = sum(f.stat().st_size for f in chunk_files) / 1e9
    print(f"\n[prebuild] Done in {elapsed_total/60:.1f} min", flush=True)
    print(f"[prebuild] Trials: {total_trials}", flush=True)
    print(f"[prebuild] Subjects: {all_subject_count}", flush=True)
    print(f"[prebuild] Chunks: {len(chunk_files)} ({total_chunk_gb:.2f} GB)", flush=True)
    print(f"[prebuild] Skipped patients: {n_skipped}", flush=True)
    print(f"[prebuild] Excluded patient IDs: {len(exclude_ids)}", flush=True)
    print(f"[prebuild] Manifest: {cache_path}", flush=True)
    print(f"[prebuild] Chunk dir: {chunk_dir}", flush=True)


if __name__ == "__main__":
    main()
