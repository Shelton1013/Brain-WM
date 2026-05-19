"""
Multi-dataset EEG loader for large-scale JEPA pretraining.

Supports:
  1. PhysioNet MI (109 subjects, ~100h) — already have this
  2. MOABB datasets via moabb library (multiple MI/P300/SSVEP datasets)
  3. TUH EEG Corpus (TUEG) — requires separate download + access request
  4. Custom .edf/.npy directories

All datasets are normalized to:
  - Resampled to 256 Hz
  - Filtered 0.1-75 Hz
  - Per-subject Euclidean Alignment
  - Segmented into 4s trials
  - Z-score normalized per trial
  - Mapped to a common electrode subset

Usage:
  from dataset_multi import MultiDatasetEEG

  dataset = MultiDatasetEEG(
      sources=[
          {"type": "physionet", "n_subjects": 109},
          {"type": "moabb", "name": "Cho2017"},
          {"type": "moabb", "name": "Lee2019_MI"},
          {"type": "moabb", "name": "BNCI2014001"},
          {"type": "edf_dir", "path": "/data/tueg/edf/"},
      ],
      sample_rate=256,
      trial_duration_s=4,
  )
"""

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
from dataset import euclidean_alignment, PhysioNetMIDataset

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# ============================================================
# Common electrode set (10-20 system, 19 channels minimum)
# ============================================================

COMMON_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz",
    "T3", "T4", "T5", "T6",  # or T7/T8/P7/P8
    "P3", "P4", "Pz",
    "O1", "O2",
]

# Aliases for different naming conventions
CHANNEL_ALIASES = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    "FP1": "Fp1", "FP2": "Fp2",
    "EEG Fp1": "Fp1", "EEG Fp2": "Fp2",
    "EEG F3": "F3", "EEG F4": "F4",
    "EEG C3": "C3", "EEG C4": "C4",
    "EEG P3": "P3", "EEG P4": "P4",
    "EEG O1": "O1", "EEG O2": "O2",
    "EEG Fz": "Fz", "EEG Cz": "Cz", "EEG Pz": "Pz",
    "EEG F7": "F7", "EEG F8": "F8",
    "EEG T3": "T3", "EEG T4": "T4",
    "EEG T5": "T5", "EEG T6": "T6",
    # TUH format: "EEG FP1-REF" etc.
}


def normalize_channel_name(name: str) -> str:
    """Map various channel naming conventions to standard 10-20."""
    # Strip common suffixes
    clean = name.strip()
    for suffix in ["-REF", "-LE", "-AR", "-AVG"]:
        if clean.upper().endswith(suffix):
            clean = clean[:-len(suffix)].strip()

    # Check aliases
    if clean in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[clean]
    if clean.upper() in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[clean.upper()]

    # Check if it's already standard
    for std in COMMON_CHANNELS:
        if clean.upper() == std.upper():
            return std

    return clean  # return as-is if not recognized


def pick_common_channels(ch_names: list[str]) -> tuple[list[int], list[str]]:
    """Find indices of common channels in a recording's channel list.

    Returns:
        (indices, matched_names) — indices into ch_names for the matched channels
    """
    normalized = [normalize_channel_name(n) for n in ch_names]
    indices = []
    matched = []
    for target in COMMON_CHANNELS:
        for i, norm_name in enumerate(normalized):
            if norm_name == target:
                indices.append(i)
                matched.append(target)
                break
    return indices, matched


# ============================================================
# Single-source dataset wrappers
# ============================================================

class MOABBDataset(Dataset):
    """Load any MOABB dataset as 4s EEG trials.

    Popular MOABB datasets for pretraining:
      - Cho2017: 52 subjects, 64 ch, 512Hz, left/right MI
      - Lee2019_MI: 54 subjects, 62 ch, 1000Hz, left/right MI
      - BNCI2014001: 9 subjects, 22 ch, 250Hz, 4-class MI
      - BNCI2014004: 9 subjects, 3 ch, 250Hz, 2-class MI
      - Shin2017A: 29 subjects, 30 ch, 200Hz, 2-class MI
      - Weibo2014: 10 subjects, 60 ch, 200Hz, 7-class MI
      - Zhou2016: 4 subjects, 14 ch, 250Hz, 3-class MI
    """

    def __init__(
        self,
        dataset_name: str,
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        use_ea: bool = True,
        min_channels: int = 19,
        data_dir: str = None,
    ):
        try:
            import moabb
            import moabb.datasets
        except ImportError:
            raise ImportError("moabb required: pip install moabb")

        # Set MOABB/MNE download path BEFORE creating dataset
        if data_dir:
            import os
            import mne
            moabb_path = os.path.join(data_dir, "moabb")
            os.makedirs(moabb_path, exist_ok=True)
            mne.set_config("MNE_DATA", moabb_path, set_env=True)
            # Also set MOABB-specific path
            moabb.utils.set_download_dir(moabb_path)

        self.trial_samples = sample_rate * trial_duration_s
        self.trials = []
        self.subject_ids = []
        self.electrode_names = None

        # Get dataset class
        ds_class = getattr(moabb.datasets, dataset_name)
        ds = ds_class()

        print(f"  Loading MOABB/{dataset_name} ({len(ds.subject_list)} subjects)...")

        for subj_idx, subj in enumerate(ds.subject_list):
            try:
                data = ds.get_data(subjects=[subj])
                subj_recordings = []

                for session_name, session in data[subj].items():
                    for run_name, raw in session.items():
                        # Resample
                        if raw.info["sfreq"] != sample_rate:
                            raw.resample(sample_rate, verbose=False)
                        # Filter
                        raw.filter(0.1, 75.0, verbose=False)
                        # Pick common channels
                        ch_indices, ch_names = pick_common_channels(raw.ch_names)
                        if len(ch_indices) < min_channels:
                            continue
                        if self.electrode_names is None:
                            self.electrode_names = ch_names

                        eeg = raw.get_data()[ch_indices].T.astype(np.float32)
                        subj_recordings.append(eeg)

                if not subj_recordings:
                    continue

                # Concatenate + EA
                subj_data = np.concatenate(subj_recordings, axis=0)
                if use_ea:
                    subj_data = euclidean_alignment(subj_data)

                # Segment into trials
                n_trials = len(subj_data) // self.trial_samples
                for t in range(n_trials):
                    start = t * self.trial_samples
                    trial = subj_data[start:start + self.trial_samples]
                    mean = trial.mean(axis=0, keepdims=True)
                    std = trial.std(axis=0, keepdims=True) + 1e-8
                    self.trials.append(((trial - mean) / std))
                    self.subject_ids.append(subj_idx)

            except Exception as e:
                print(f"    Skipping subject {subj}: {e}")

        self.n_subjects = len(set(self.subject_ids))
        print(f"    → {len(self.trials)} trials, {self.n_subjects} subjects, "
              f"{len(self.electrode_names or [])} channels")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), self.subject_ids[idx]


class EDFDirectoryDataset(Dataset):
    """Load EEG from a directory of .edf files (e.g., TUH EEG Corpus).

    Scans recursively for .edf files, loads each as continuous EEG,
    segments into 4s trials.
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        use_ea: bool = True,
        max_files: int = None,
        min_channels: int = 19,
    ):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")

        self.trial_samples = sample_rate * trial_duration_s
        self.trials = []
        self.subject_ids = []
        self.electrode_names = None

        edf_files = sorted(Path(data_dir).rglob("*.edf"))
        if max_files:
            edf_files = edf_files[:max_files]

        print(f"  Loading EDF dir: {data_dir} ({len(edf_files)} files)...")

        for file_idx, edf_path in enumerate(edf_files):
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
                if raw.info["sfreq"] != sample_rate:
                    raw.resample(sample_rate, verbose=False)
                raw.filter(0.1, 75.0, verbose=False)

                ch_indices, ch_names = pick_common_channels(raw.ch_names)
                if len(ch_indices) < min_channels:
                    continue
                if self.electrode_names is None:
                    self.electrode_names = ch_names

                eeg = raw.get_data()[ch_indices].T.astype(np.float32)
                if use_ea:
                    eeg = euclidean_alignment(eeg)

                n_trials = len(eeg) // self.trial_samples
                for t in range(n_trials):
                    start = t * self.trial_samples
                    trial = eeg[start:start + self.trial_samples]
                    mean = trial.mean(axis=0, keepdims=True)
                    std = trial.std(axis=0, keepdims=True) + 1e-8
                    self.trials.append(((trial - mean) / std))
                    self.subject_ids.append(file_idx)

                if file_idx % 100 == 0 and file_idx > 0:
                    print(f"    ... {file_idx}/{len(edf_files)} files, "
                          f"{len(self.trials)} trials so far")

            except Exception as e:
                if file_idx < 5:
                    print(f"    Skipping {edf_path.name}: {e}")

        self.n_subjects = len(set(self.subject_ids))
        print(f"    → {len(self.trials)} trials, {self.n_subjects} recordings, "
              f"{len(self.electrode_names or [])} channels")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), self.subject_ids[idx]


# ============================================================
# Multi-dataset combiner
# ============================================================

class MultiDatasetEEG(Dataset):
    """Combine multiple EEG datasets for large-scale pretraining.

    All sources are normalized to common format:
      - 256 Hz, 0.1-75 Hz filtered
      - Common 19-channel subset (10-20 system)
      - Per-subject EA
      - 4s trials, z-scored

    Usage:
        dataset = MultiDatasetEEG(sources=[
            {"type": "physionet", "n_subjects": 109},
            {"type": "moabb", "name": "Cho2017"},
            {"type": "moabb", "name": "Lee2019_MI"},
            {"type": "edf_dir", "path": "/data/tueg/edf/", "max_files": 5000},
            {"type": "edf_dir", "path": "/data/hbn/eeg/"},  # HBN
        ])

    Available public datasets (no restricted access):
      - PhysioNet MI: 109 subjects, 64ch, ~10h
      - MOABB (auto-download): Cho2017 (52s), Lee2019_MI (54s), BNCI2014001 (9s), etc.
      - HBN (OpenNeuro): 3000+ subjects, 128ch, free download from S3
      - TUH EEG (requires email approval): 15000+ subjects, ~60K recordings
    """

    def __init__(
        self,
        sources: list[dict],
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        physionet_data_dir: str = "/home/share/data_makchen/peng/datasets/physionet",
        download_dir: str = "/home/share/data_makchen/peng/datasets",
    ):
        datasets = []
        total_subjects = 0

        for src in sources:
            src_type = src["type"]
            print(f"\n--- Loading source: {src_type} ---")

            if src_type == "physionet":
                n_subj = src.get("n_subjects", 109)
                ds = PhysioNetMIDataset(
                    subjects=list(range(1, n_subj + 1)),
                    sample_rate=sample_rate,
                    trial_duration_s=trial_duration_s,
                    data_dir=physionet_data_dir,
                )
                # Map PhysioNet to common 19 channels
                if ds.electrode_names:
                    ch_indices, ch_names = pick_common_channels(ds.electrode_names)
                    if ch_indices:
                        print(f"  Mapping PhysioNet {len(ds.electrode_names)}ch → {len(ch_indices)}ch")
                        ds.trials = [t[:, ch_indices] for t in ds.trials]
                        ds.electrode_names = ch_names
                # Remap subject IDs to global
                for i in range(len(ds.subject_ids)):
                    ds.subject_ids[i] += total_subjects
                total_subjects += ds.n_subjects
                datasets.append(ds)

            elif src_type == "moabb":
                ds = MOABBDataset(
                    dataset_name=src["name"],
                    sample_rate=sample_rate,
                    trial_duration_s=trial_duration_s,
                    data_dir=src.get("data_dir", download_dir),
                )
                for i in range(len(ds.subject_ids)):
                    ds.subject_ids[i] += total_subjects
                total_subjects += ds.n_subjects
                datasets.append(ds)

            elif src_type == "edf_dir":
                ds = EDFDirectoryDataset(
                    data_dir=src["path"],
                    sample_rate=sample_rate,
                    trial_duration_s=trial_duration_s,
                    max_files=src.get("max_files", None),
                )
                for i in range(len(ds.subject_ids)):
                    ds.subject_ids[i] += total_subjects
                total_subjects += ds.n_subjects
                datasets.append(ds)

            else:
                print(f"  Unknown source type: {src_type}, skipping")

        # Combine
        self.datasets = datasets
        self._lengths = [len(d) for d in datasets]
        self._cumulative = []
        cum = 0
        for l in self._lengths:
            self._cumulative.append(cum)
            cum += l

        self.n_subjects = total_subjects
        self.total_trials = sum(self._lengths)

        # Use electrode names from first dataset that has them
        self.electrode_names = None
        for d in datasets:
            if hasattr(d, "electrode_names") and d.electrode_names:
                self.electrode_names = d.electrode_names
                break

        # Summary
        total_hours = self.total_trials * trial_duration_s / 3600
        print(f"\n{'='*50}")
        print(f"  Multi-dataset summary:")
        print(f"  Sources: {len(datasets)}")
        print(f"  Total trials: {self.total_trials:,}")
        print(f"  Total subjects: {self.n_subjects}")
        print(f"  Total hours: {total_hours:.1f}h")
        print(f"  Channels: {len(self.electrode_names or [])}")
        print(f"{'='*50}")

    def __len__(self):
        return self.total_trials

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, (cum, length) in enumerate(zip(self._cumulative, self._lengths)):
            if idx < cum + length:
                return self.datasets[i][idx - cum]
        raise IndexError(f"Index {idx} out of range")
