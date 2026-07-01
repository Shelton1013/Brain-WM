"""EEG dataset for BrainWM pretraining (v2: with Euclidean Alignment)."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.linalg import fractional_matrix_power

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# ============================================================
# Euclidean Alignment (EA)
# ============================================================

def euclidean_alignment(data: np.ndarray) -> np.ndarray:
    """Apply Euclidean Alignment to a single subject's EEG.

    Re-centers the covariance matrix to the identity, removing
    subject-specific signal distribution differences (impedance,
    skull conductivity, electrode placement).

    Reference: He & Wu (2020), IEEE TNSRE.

    Args:
        data: [T, C] continuous EEG from one subject
    Returns:
        [T, C] aligned EEG
    """
    # Covariance matrix: [C, C]
    R = (data.T @ data) / data.shape[0]
    # R^(-1/2) via eigendecomposition
    try:
        R_inv_sqrt = fractional_matrix_power(R, -0.5).real.astype(np.float32)
    except (ValueError, np.linalg.LinAlgError):
        # Fallback: regularize if singular
        R_reg = R + 1e-6 * np.eye(R.shape[0])
        R_inv_sqrt = fractional_matrix_power(R_reg, -0.5).real.astype(np.float32)
    # Align: X_aligned = X @ R^(-1/2)
    return data @ R_inv_sqrt


# ============================================================
# Generic EEG Dataset
# ============================================================

class EEGDataset(Dataset):
    """Dataset that loads preprocessed EEG trials for BrainWM pretraining.

    Each sample is a 4-second EEG segment across all channels.
    Returns (trial, subject_id) for adversarial training.
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        normalize: bool = True,
        use_ea: bool = True,
    ):
        self.sample_rate = sample_rate
        self.trial_samples = sample_rate * trial_duration_s
        self.normalize = normalize
        self.electrode_names = None

        self.trials = []        # list of [trial_samples, C]
        self.subject_ids = []   # subject id per trial

        data_path = Path(data_dir)
        subject_id = 0

        for f in sorted(data_path.glob("*.npy")):
            arr = np.load(f, allow_pickle=True).astype(np.float32)
            if arr.ndim == 2:
                # Single recording: [T_total, C]
                if use_ea:
                    arr = euclidean_alignment(arr)
                segs = self._segment_trials(arr)
            elif arr.ndim == 3:
                # Already segmented: [n_trials, T, C]
                if use_ea:
                    # EA on concatenated data then re-segment
                    flat = arr.reshape(-1, arr.shape[-1])
                    flat = euclidean_alignment(flat)
                    segs = flat.reshape(arr.shape)
                else:
                    segs = arr
            else:
                continue

            for seg in segs:
                self.trials.append(seg)
                self.subject_ids.append(subject_id)
            subject_id += 1

        self.n_subjects = subject_id

        # Load electrode names if available
        names_file = data_path / "electrode_names.txt"
        if names_file.exists():
            with open(names_file) as f_:
                self.electrode_names = [line.strip() for line in f_.readlines()]

        print(f"Loaded {len(self.trials)} trials from {self.n_subjects} subjects (EA={use_ea})")

    def _segment_trials(self, recording: np.ndarray) -> np.ndarray:
        T, C = recording.shape
        n_trials = T // self.trial_samples
        if n_trials == 0:
            return np.zeros((0, self.trial_samples, C), dtype=np.float32)
        trimmed = recording[:n_trials * self.trial_samples]
        return trimmed.reshape(n_trials, self.trial_samples, C)

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx].astype(np.float32)
        if self.normalize:
            mean = trial.mean(axis=0, keepdims=True)
            std = trial.std(axis=0, keepdims=True) + 1e-8
            trial = (trial - mean) / std
        subject_id = self.subject_ids[idx]
        return torch.from_numpy(trial), subject_id


# ============================================================
# PhysioNet Motor Imagery Dataset
# ============================================================

class PhysioNetMIDataset(Dataset):
    """PhysioNet Motor Movement/Imagery dataset (v2: with EA + subject_id)."""

    def __init__(
        self,
        subjects: list[int] | None = None,
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        data_dir: str = "./data/physionet",
        use_ea: bool = True,
    ):
        if not MNE_AVAILABLE:
            raise ImportError("mne is required: pip install mne")

        from mne.datasets import eegbci
        from mne.io import read_raw_edf

        self.sample_rate = sample_rate
        self.trial_samples = sample_rate * trial_duration_s
        self.trials = []
        self.subject_ids = []
        self.electrode_names = None

        if subjects is None:
            subjects = list(range(1, 110))

        task_runs = [4, 6, 8, 10, 12, 14]

        for subj_idx, subj in enumerate(subjects):
            try:
                # update_path=False: prevent the interactive prompt
                #     "Do you want to set the path ... [y]/n?"
                # which hangs any nohup / DDP job with no stdin.
                files = eegbci.load_data(
                    subj, task_runs, path=data_dir, update_path=False,
                )
                # Concatenate all runs for this subject (for EA)
                subj_data_list = []
                for f in files:
                    raw = read_raw_edf(f, preload=True, verbose=False)
                    if raw.info["sfreq"] != sample_rate:
                        raw.resample(sample_rate, verbose=False)
                    raw.filter(0.1, 75.0, verbose=False)
                    if self.electrode_names is None:
                        self.electrode_names = raw.ch_names
                    subj_data_list.append(raw.get_data().T.astype(np.float32))

                # Concatenate all runs: [T_total, C]
                subj_data = np.concatenate(subj_data_list, axis=0)

                # Euclidean Alignment per subject
                if use_ea:
                    subj_data = euclidean_alignment(subj_data)

                # Segment into trials
                n_trials = len(subj_data) // self.trial_samples
                for t in range(n_trials):
                    start = t * self.trial_samples
                    end = start + self.trial_samples
                    trial = subj_data[start:end]
                    # Z-score normalize
                    mean = trial.mean(axis=0, keepdims=True)
                    std = trial.std(axis=0, keepdims=True) + 1e-8
                    trial = (trial - mean) / std
                    self.trials.append(trial)
                    self.subject_ids.append(subj_idx)

            except Exception as e:
                print(f"Skipping subject {subj}: {e}")

        self.n_subjects = len(subjects)
        print(f"Loaded {len(self.trials)} trials from {self.n_subjects} subjects (EA={use_ea})")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = torch.from_numpy(self.trials[idx])
        subject_id = self.subject_ids[idx]
        return trial, subject_id  # [T, C], int
