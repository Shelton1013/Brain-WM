"""
PhysioNet MI 4-class loader matching CBraMod's protocol.

CBraMod (ICLR 2025) Table 8/9 protocol on PhysioNet-MI:
  - 4 classes: LH (left hand), RH (right hand), BF (both fists), BFt (both feet)
  - Source runs:
      4, 8, 12  → imagined LH (T1) / RH (T2)
      6, 10, 14 → imagined BF  (T1) / BFt (T2)
  - Subject-disjoint split: train S1-70, val S71-89, test S90-109
  - Window: 4 seconds starting at event onset
  - Reported FT BAcc: 0.6417 ± 0.0091 (CBraMod-base)

This loader matches that split exactly so OUR model's eval BAcc on the
S90-109 test set is directly comparable to CBraMod's published number.

Disclaimer: when our pretrain corpus includes PhysioNet (all 109 subj),
this is "in-distribution pretraining" (same as REVE; flag in paper).

Usage:
    train_ds = PhysioNetMI4ClassDataset(
        data_dir="/path/to/physionet", subjects=list(range(1, 71)),
        sample_rate=256, trial_duration_s=4,
        normalization="per_recording_robust",
        cache_dir="/path/to/cache",
    )
"""

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import mne
    from mne.io import read_raw_edf
    from mne.datasets import eegbci
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# Run → which two task labels (T1, T2) it provides
# T0 (rest) is always discarded for 4-class MI classification.
RUN_TASK_MAP = {
    4:  ("LH", "RH"),   # imagined LH / RH
    8:  ("LH", "RH"),
    12: ("LH", "RH"),
    6:  ("BF", "BFt"),  # imagined both fists / both feet
    10: ("BF", "BFt"),
    14: ("BF", "BFt"),
}

# 4-class label mapping (consistent ordering)
LABEL_MAP = {"LH": 0, "RH": 1, "BF": 2, "BFt": 3}
LABEL_NAMES = ["LH", "RH", "BF", "BFt"]


# CBraMod subject splits
CBRAMOD_SPLITS = {
    "train": list(range(1, 71)),     # S1-70
    "val":   list(range(71, 90)),    # S71-89
    "test":  list(range(90, 110)),   # S90-109
}


def _robust_scale_per_recording(eeg: np.ndarray) -> np.ndarray:
    """Per-recording robust scaling: (x - median) / (IQR/1.349 + eps)."""
    eeg = np.asarray(eeg, dtype=np.float32)
    median = np.median(eeg, axis=0, keepdims=True)
    q75 = np.percentile(eeg, 75, axis=0, keepdims=True)
    q25 = np.percentile(eeg, 25, axis=0, keepdims=True)
    robust_std = (q75 - q25) / 1.349 + 1e-6
    return ((eeg - median) / robust_std).astype(np.float32)


def _cache_key(config: dict) -> str:
    s = ",".join(f"{k}={config[k]}" for k in sorted(config))
    return hashlib.md5(s.encode()).hexdigest()[:16]


class PhysioNetMI4ClassDataset(Dataset):
    """PhysioNet MI 4-class event-aligned dataset, CBraMod protocol.

    Returns:
        trial: torch.Tensor [trial_samples, n_channels] float32
        label: int (0..3)
    Plus self.subject_ids: list[int] aligned with self.trials
    """

    def __init__(
        self,
        data_dir: str,
        subjects: list[int],
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        normalization: str = "per_recording_robust",
        cache_dir: Optional[str] = None,
    ):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")
        assert normalization in ("per_trial_zscore", "per_recording_robust"), \
            f"unknown normalization: {normalization}"

        self.sample_rate = sample_rate
        self.trial_samples = sample_rate * trial_duration_s
        self.normalization = normalization
        self.trials: list[np.ndarray] = []
        self.labels: list[int] = []
        self.subject_ids: list[int] = []
        self.electrode_names: list[str] | None = None

        # Cache check
        cache_path = None
        if cache_dir:
            key = _cache_key({
                "kind": "physio_mi_4cls",
                "data_dir": str(data_dir),
                "subjects": ",".join(str(s) for s in sorted(subjects)),
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "normalization": normalization,
            })
            cache_path = Path(cache_dir) / f"physio_mi_4cls_{key}.pt"
            if cache_path.exists():
                print(f"  ↻ loading cache: {cache_path.name}")
                cached = torch.load(cache_path, weights_only=False)
                self.trials = cached["trials"]
                self.labels = cached["labels"]
                self.subject_ids = cached["subject_ids"]
                self.electrode_names = cached["electrode_names"]
                print(f"    ← {len(self.trials)} trials, "
                      f"{len(set(self.subject_ids))} subjects, "
                      f"{len(self.electrode_names)} channels")
                self._print_class_dist()
                return

        # Build from raw
        print(f"  [PhysioNet MI 4-class] Loading subjects {min(subjects)}..{max(subjects)} "
              f"({len(subjects)} subj)")
        for subj in subjects:
            try:
                self._process_subject(subj, data_dir)
            except Exception as e:
                print(f"    skip subject {subj}: {e}")
                continue

        print(f"  [PhysioNet MI 4-class] Loaded {len(self.trials)} trials "
              f"from {len(set(self.subject_ids))} subjects")
        self._print_class_dist()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "trials": self.trials,
                "labels": self.labels,
                "subject_ids": self.subject_ids,
                "electrode_names": self.electrode_names,
            }, cache_path)
            print(f"  ↑ saved cache: {cache_path.name}")

    def _process_subject(self, subj: int, data_dir: str):
        runs = sorted(RUN_TASK_MAP.keys())
        files = eegbci.load_data(subj, runs, path=data_dir)
        # files come back ordered by runs we requested
        # Need to know which file corresponds to which run
        # eegbci.load_data returns files in same order as runs list passed
        for fpath, run_id in zip(files, runs):
            try:
                self._extract_events_from_file(fpath, run_id, subj)
            except Exception as e:
                print(f"    subject {subj} run {run_id}: skip ({e})")
                continue

    def _extract_events_from_file(
        self, fpath: str, run_id: int, subj: int,
    ):
        raw = read_raw_edf(fpath, preload=True, verbose=False)
        if raw.info["sfreq"] != self.sample_rate:
            raw.resample(self.sample_rate, verbose=False)
        raw.filter(0.1, 75.0, verbose=False)

        if self.electrode_names is None:
            self.electrode_names = raw.ch_names

        data = raw.get_data().T.astype(np.float32)   # [T_total, C]

        # Per-recording robust scaling BEFORE event extraction
        if self.normalization == "per_recording_robust":
            data = _robust_scale_per_recording(data)

        events, event_id = mne.events_from_annotations(
            raw, verbose=False,
        )  # events: [n_events, 3] with (sample, prev_id, this_id)

        # event_id maps description → integer id. PhysioNet uses
        # "T0", "T1", "T2" as descriptions.
        # Reverse: int → description
        id_to_desc = {v: k for k, v in event_id.items()}

        t1_label, t2_label = RUN_TASK_MAP[run_id]

        T_total = data.shape[0]
        for sample_idx, _, this_id in events:
            desc = id_to_desc.get(this_id, "")
            if desc == "T1":
                label_str = t1_label
            elif desc == "T2":
                label_str = t2_label
            else:
                continue   # skip T0 (rest)

            lo = int(sample_idx)
            hi = lo + self.trial_samples
            if hi > T_total:
                continue   # OOB

            trial = data[lo:hi]
            if self.normalization == "per_trial_zscore":
                mean = trial.mean(axis=0, keepdims=True)
                std = trial.std(axis=0, keepdims=True) + 1e-8
                trial = ((trial - mean) / std).astype(np.float32)

            self.trials.append(trial.astype(np.float32))
            self.labels.append(LABEL_MAP[label_str])
            self.subject_ids.append(subj)

    def _print_class_dist(self):
        if not self.labels:
            return
        bc = np.bincount(self.labels, minlength=4)
        print(f"    Class dist: LH={bc[0]} RH={bc[1]} BF={bc[2]} BFt={bc[3]}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), int(self.labels[idx])
