"""TUAB / TUEV clinical evaluation datasets.

Both datasets keep the SAME preprocessing as our pretrain pipeline
(19 channels, 256 Hz, 4 s segments, z-score per trial) so that pretrained
encoder weights apply directly. This deliberately diverges from LaBraM's
23ch/200Hz/{10s,5s} preprocessing — labels are identical, recordings
are identical, only the temporal/spatial resolution differs.

Adds 50 Hz notch (matching LaBraM); the rest of the filter cascade
(0.1-75 Hz bandpass, 256 Hz resample) is already in the pretrain path.

Both classes cache the processed numpy payload to .pt for instant
re-loading on subsequent runs.

Usage:
    train_ds = TUABDataset(
        data_dir="/home/pxieaf/home2/tuh/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        cache_dir="/home/pxieaf/home2/dataset_cache",
    )
    eval_ds  = TUABDataset(..., split="eval", ...)

    train_ds = TUEVDataset(
        data_dir="/home/pxieaf/home2/tuh/tuh_eeg_events/v2.0.1/edf",
        split="train",
        cache_dir="/home/pxieaf/home2/dataset_cache",
    )
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_multi import (
    pick_common_channels,
    _cache_key,
    _try_load_cache,
    _save_cache,
)

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# ============================================================
# Preprocessing shared by both datasets
# ============================================================

def _load_and_preprocess_raw(
    edf_path: Path,
    sample_rate: int,
    min_channels: int,
):
    """Load an EDF and return (data [T, n_ch] float32 microvolts, ch_names).

    Returns None on any failure (file too short for filter, channel
    mismatch, read error).
    """
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        if raw.info["sfreq"] != sample_rate:
            raw.resample(sample_rate, verbose=False)
        # 0.1 Hz highpass needs ~33 s @ 256 Hz; skip too-short recordings
        if raw.n_times < int(40 * sample_rate):
            return None
        raw.notch_filter(50.0, verbose=False)
        raw.filter(0.1, 75.0, verbose=False)
        ch_indices, ch_names = pick_common_channels(raw.ch_names)
        if len(ch_indices) < min_channels:
            return None
        data_uv = raw.get_data(units="uV")[ch_indices].T.astype(np.float32)
        return data_uv, ch_names
    except Exception:
        return None


def _robust_scale_per_recording(eeg: np.ndarray) -> np.ndarray:
    """Per-recording robust scaling: (x - median) / (IQR/1.349 + eps).

    Applied to the full continuous recording (Defossez 2022 / Laya style).
    Matches the normalization used in pretraining when
    --normalization per_recording_robust.
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    median = np.median(eeg, axis=0, keepdims=True)
    q75 = np.percentile(eeg, 75, axis=0, keepdims=True)
    q25 = np.percentile(eeg, 25, axis=0, keepdims=True)
    robust_std = (q75 - q25) / 1.349 + 1e-6
    return ((eeg - median) / robust_std).astype(np.float32)


def _segment(
    eeg: np.ndarray,
    trial_samples: int,
    normalization: str = "per_trial_zscore",
) -> list[np.ndarray]:
    """Slice continuous [T, C] into list of [trial_samples, C] non-overlapping windows.

    normalization:
      - "per_trial_zscore": z-score each segment independently (legacy)
      - "per_recording_robust": skip per-segment z-score (caller must have
        already applied _robust_scale_per_recording to the full recording)
    """
    n_trials = len(eeg) // trial_samples
    out = []
    for t in range(n_trials):
        start = t * trial_samples
        trial = eeg[start:start + trial_samples]
        if normalization == "per_recording_robust":
            out.append(trial.astype(np.float32))
        else:
            mean = trial.mean(axis=0, keepdims=True)
            std = trial.std(axis=0, keepdims=True) + 1e-8
            out.append(((trial - mean) / std).astype(np.float32))
    return out


# ============================================================
# TUAB: binary classification (normal vs abnormal)
# ============================================================

class TUABDataset(Dataset):
    """TUH Abnormal EEG Corpus (TUAB) — binary classification.

    Directory layout (TUAB v3.0.1):
        data_dir/
            train/
                normal/   *.edf  (label 0)
                abnormal/ *.edf  (label 1)
            eval/
                normal/   *.edf  (label 0)
                abnormal/ *.edf  (label 1)

    Labels are determined by parent directory name. Each EDF is segmented
    into non-overlapping 4 s windows at 256 Hz (19 channels). All segments
    of a recording inherit the file-level label.

    Attributes:
        trials:   list[np.ndarray [T, C] float32]
        labels:   list[int]  (0=normal, 1=abnormal)
        electrode_names: list[str]
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train" or "eval"
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        min_channels: int = 19,
        cache_dir: str | None = None,
        normalization: str = "per_trial_zscore",
    ):
        assert split in ("train", "eval"), f"split must be 'train' or 'eval', got {split}"
        assert normalization in ("per_trial_zscore", "per_recording_robust"), \
            f"unknown normalization: {normalization}"
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")

        self.split = split
        self.normalization = normalization
        self.trial_samples = sample_rate * trial_duration_s
        self.trials: list[np.ndarray] = []
        self.labels: list[int] = []
        self.recording_ids: list[int] = []  # which EDF each trial came from
        self.patient_ids: list[str] = []    # 8-letter TUAB patient id per trial
        self.electrode_names: list[str] | None = None

        # Cache key includes split + normalization
        cache_path = None
        if cache_dir:
            key = _cache_key({
                "kind": "tuab",
                "data_dir": str(data_dir),
                "split": split,
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "min_channels": min_channels,
                "normalization": normalization,
            })
            cache_path = Path(cache_dir) / f"tuab_{split}_{key}.pt"
            cached = _try_load_cache(cache_path)
            if cached is not None:
                self.trials = cached["trials"]
                self.labels = cached["labels"]
                self.electrode_names = cached["electrode_names"]
                # Backward-compat: old caches don't have recording_ids / patient_ids
                self.recording_ids = cached.get("recording_ids", [])
                self.patient_ids = cached.get("patient_ids", [])
                return

        # Walk normal/ and abnormal/ subdirectories of the split
        split_root = Path(data_dir) / split
        recording_counter = 0
        for label_int, label_name in [(0, "normal"), (1, "abnormal")]:
            class_dir = split_root / label_name
            if not class_dir.is_dir():
                print(f"  [TUAB] missing {class_dir}, skipping")
                continue
            edf_files = sorted(class_dir.rglob("*.edf"))
            print(f"  [TUAB {split}] {label_name}: {len(edf_files)} files")
            for fi, edf_path in enumerate(edf_files):
                if fi % 100 == 0 and fi > 0:
                    print(f"    {fi}/{len(edf_files)} {label_name} processed, "
                          f"running trial count = {len(self.trials)}")
                result = _load_and_preprocess_raw(
                    edf_path, sample_rate, min_channels)
                if result is None:
                    continue
                data_uv, ch_names = result
                if self.electrode_names is None:
                    self.electrode_names = ch_names
                if normalization == "per_recording_robust":
                    data_uv = _robust_scale_per_recording(data_uv)
                segs = _segment(data_uv, self.trial_samples, normalization)
                # TUAB filename: aaaaaaaa_s001_t000.edf  → patient = "aaaaaaaa"
                patient_id = edf_path.stem.split("_")[0]
                for s in segs:
                    self.trials.append(s)
                    self.labels.append(label_int)
                    self.recording_ids.append(recording_counter)
                    self.patient_ids.append(patient_id)
                recording_counter += 1

        print(f"  [TUAB {split}] total: {len(self.trials)} trials "
              f"from {recording_counter} recordings, "
              f"{len(set(self.patient_ids))} unique patients")

        if cache_path is not None:
            _save_cache(cache_path, {
                "trials": self.trials,
                "labels": self.labels,
                "recording_ids": self.recording_ids,
                "patient_ids": self.patient_ids,
                "electrode_names": self.electrode_names,
            })

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, i):
        return torch.from_numpy(self.trials[i]), int(self.labels[i])


# ============================================================
# TUEV: 6-class event classification
# ============================================================

# .rec label index in {1,2,3,4,5,6}; we 0-index to {0..5}
TUEV_LABEL_NAMES = ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]


def _parse_rec_file(rec_path: Path) -> list[tuple[float, float, int]]:
    """Parse a TUEV .rec file into [(start_sec, end_sec, label_0_to_5), ...].

    .rec format: each row "channel,start,end,label,prob" — we drop channel
    and prob, keep one entry per (start, end, label) triple, dedupe events
    that have identical timing (multi-channel events are listed once per
    channel in the .rec file).
    """
    try:
        arr = np.genfromtxt(str(rec_path), delimiter=",", dtype=float)
        if arr.ndim == 1:  # single-line file
            arr = arr.reshape(1, -1)
        seen = set()
        out = []
        for row in arr:
            if len(row) < 4:
                continue
            start, end, label = float(row[1]), float(row[2]), int(row[3])
            if label < 1 or label > 6:
                continue
            key = (round(start, 3), round(end, 3), label)
            if key in seen:
                continue
            seen.add(key)
            out.append((start, end, label - 1))  # 0-index
        return out
    except Exception:
        return []


class TUEVDataset(Dataset):
    """TUH EEG Events Corpus (TUEV) — 6-class event classification.

    Directory layout (TUEV v2.0.1):
        data_dir/
            train/
                <patient_id>/
                    <patient_id>_<rec_id>.edf
                    <patient_id>_<rec_id>.rec   ← per-segment labels
            eval/
                <same structure>

    For each event (start, end, label) in a .rec file, we extract one
    `trial_duration_s` second window centered on the event midpoint. Each
    such window inherits the event's 6-class label (0..5).

    Classes (1..6 in .rec, 0..5 here):
        0=SPSW, 1=GPED, 2=PLED, 3=EYEM, 4=ARTF, 5=BCKG

    Attributes:
        trials:   list[np.ndarray [T, C] float32]
        labels:   list[int]  (0..5)
        electrode_names: list[str]
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        min_channels: int = 19,
        cache_dir: str | None = None,
        normalization: str = "per_trial_zscore",
    ):
        assert split in ("train", "eval"), f"split must be 'train' or 'eval', got {split}"
        assert normalization in ("per_trial_zscore", "per_recording_robust"), \
            f"unknown normalization: {normalization}"
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")

        self.split = split
        self.normalization = normalization
        self.trial_samples = sample_rate * trial_duration_s
        self.trials: list[np.ndarray] = []
        self.labels: list[int] = []
        self.recording_ids: list[int] = []  # which EDF each trial came from
        self.patient_ids: list[str] = []    # 8-letter TUEV patient id per trial
        self.electrode_names: list[str] | None = None

        cache_path = None
        if cache_dir:
            key = _cache_key({
                "kind": "tuev",
                "data_dir": str(data_dir),
                "split": split,
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "min_channels": min_channels,
                "normalization": normalization,
            })
            cache_path = Path(cache_dir) / f"tuev_{split}_{key}.pt"
            cached = _try_load_cache(cache_path)
            if cached is not None:
                self.trials = cached["trials"]
                self.labels = cached["labels"]
                self.electrode_names = cached["electrode_names"]
                # Backward-compat: old caches don't have recording_ids / patient_ids
                self.recording_ids = cached.get("recording_ids", [])
                self.patient_ids = cached.get("patient_ids", [])
                return

        split_root = Path(data_dir) / split
        edf_files = sorted(split_root.rglob("*.edf"))
        print(f"  [TUEV {split}] {len(edf_files)} EDF files")
        events_skipped_oob = 0
        events_extracted = 0
        recording_counter = 0
        for fi, edf_path in enumerate(edf_files):
            if fi % 200 == 0 and fi > 0:
                print(f"    {fi}/{len(edf_files)} files, "
                      f"{events_extracted} events extracted, "
                      f"{events_skipped_oob} skipped (OOB)")

            rec_path = edf_path.with_suffix(".rec")
            if not rec_path.exists():
                continue
            events = _parse_rec_file(rec_path)
            if not events:
                continue

            result = _load_and_preprocess_raw(
                edf_path, sample_rate, min_channels)
            if result is None:
                continue
            data_uv, ch_names = result
            if self.electrode_names is None:
                self.electrode_names = ch_names

            # Per-recording robust scaling applies to full continuous recording
            # BEFORE event extraction; per-trial z-score applies per event below.
            if normalization == "per_recording_robust":
                data_uv = _robust_scale_per_recording(data_uv)

            T_total = data_uv.shape[0]
            half = self.trial_samples // 2
            any_event_added = False
            # TUEV filename: aaaaaaaa_s001_t000.edf → patient = "aaaaaaaa"
            patient_id = edf_path.stem.split("_")[0]
            for start_s, end_s, label_0_to_5 in events:
                mid = int(round((start_s + end_s) / 2 * sample_rate))
                lo = mid - half
                hi = mid + (self.trial_samples - half)
                if lo < 0 or hi > T_total:
                    events_skipped_oob += 1
                    continue
                trial = data_uv[lo:hi]
                if normalization == "per_recording_robust":
                    self.trials.append(trial.astype(np.float32))
                else:
                    mean = trial.mean(axis=0, keepdims=True)
                    std = trial.std(axis=0, keepdims=True) + 1e-8
                    self.trials.append(((trial - mean) / std).astype(np.float32))
                self.labels.append(label_0_to_5)
                self.recording_ids.append(recording_counter)
                self.patient_ids.append(patient_id)
                events_extracted += 1
                any_event_added = True
            if any_event_added:
                recording_counter += 1

        print(f"  [TUEV {split}] total: {len(self.trials)} trials "
              f"from {recording_counter} recordings, "
              f"{len(set(self.patient_ids))} unique patients, "
              f"{events_skipped_oob} skipped (OOB)")

        # Per-class count (sanity)
        if self.labels:
            counts = np.bincount(self.labels, minlength=6)
            for ci, name in enumerate(TUEV_LABEL_NAMES):
                print(f"    {name}: {counts[ci]}")

        if cache_path is not None:
            _save_cache(cache_path, {
                "trials": self.trials,
                "labels": self.labels,
                "recording_ids": self.recording_ids,
                "patient_ids": self.patient_ids,
                "electrode_names": self.electrode_names,
            })

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, i):
        return torch.from_numpy(self.trials[i]), int(self.labels[i])
