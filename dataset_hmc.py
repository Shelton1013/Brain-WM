"""HMC (Haaglanden Medisch Centrum) sleep-staging loader — 5-class.

Source: PhysioNet "HMC Sleep Staging" — 154 PSG recordings, AASM 5-class
(W, N1, N2, N3, REM) scored in 30-s epochs.

Layout:
    <root>/recordings/SN001.edf                 # PSG
                     /SN001_sleepscoring.txt     # CSV, one row per annotation
                     /SN001_sleepscoring.edf     # same as EDF+ (we use .txt)

EEG channels (mastoid-referenced): 'EEG F4-M1','EEG C4-M1','EEG O2-M1',
'EEG C3-M2'. We place them into our 19-ch 10-20 layout by NAME (never
positional) and zero-pad the rest; EMG/EOG/ECG are ignored.

Protocol mirrors our ISRUC eval: 30-s epochs → middle 10 s (matches the 10 s
pretrain window), per-recording robust norm, subject-disjoint split.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import mne
    from mne.io import read_raw_edf
    mne.set_log_level("ERROR")
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# 4 HMC EEG channels → index in our canonical 19-ch 10-20 order
# ['Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz','T3','T4','T5','T6','P3','P4','Pz','O1','O2']
HMC_EEG_CHANNELS = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2"]
HMC_CH_TO_19CH_IDX = {
    "EEG F4-M1": 3,    # F4
    "EEG C4-M1": 8,    # C4
    "EEG O2-M1": 18,   # O2
    "EEG C3-M2": 7,    # C3
}
HMC_LABEL_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES = 5
EPOCH_S = 30
HMC_NATIVE_SR = 256


def _robust_scale_per_recording(eeg: np.ndarray) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=np.float32)
    median = np.median(eeg, axis=0, keepdims=True)
    q75 = np.percentile(eeg, 75, axis=0, keepdims=True)
    q25 = np.percentile(eeg, 25, axis=0, keepdims=True)
    robust_std = (q75 - q25) / 1.349 + 1e-6
    return ((eeg - median) / robust_std).astype(np.float32)


def _cache_key(config: dict) -> str:
    s = ",".join(f"{k}={config[k]}" for k in sorted(config))
    return hashlib.md5(s.encode()).hexdigest()[:16]


def list_hmc_recordings(data_dir):
    """Return sorted recording ids (e.g. 'SN001') found under <root>/recordings."""
    rec_dir = Path(data_dir) / "recordings"
    if not rec_dir.is_dir():
        rec_dir = Path(data_dir)   # allow pointing directly at recordings/
    ids = []
    for p in sorted(rec_dir.glob("SN*.edf")):
        if "sleepscoring" in p.name:
            continue
        ids.append(p.stem)   # 'SN001'
    return sorted(set(ids)), rec_dir


def make_hmc_split(data_dir, train_frac=0.7, val_frac=0.15, seed=42):
    """Deterministic subject-disjoint split over recording ids."""
    ids, _ = list_hmc_recordings(data_dir)
    rng = np.random.RandomState(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": sorted(ids[:n_tr]),
        "val":   sorted(ids[n_tr:n_tr + n_val]),
        "test":  sorted(ids[n_tr + n_val:]),
    }


def _pick_hmc_eeg(raw):
    """Return raw channel names for the 4 HMC EEG channels, in canonical order.
    Name-based (case/space tolerant); requires all 4."""
    available = list(raw.ch_names)
    upper_map = {ch.upper().replace(" ", ""): ch for ch in available}
    picked = []
    for want in HMC_EEG_CHANNELS:
        key = want.upper().replace(" ", "")
        if want in available:
            picked.append(want)
        elif key in upper_map:
            picked.append(upper_map[key])
        else:
            raise ValueError(f"HMC EEG channel {want} not in {available}")
    return picked


def _parse_hmc_scoring(txt_path: Path) -> dict:
    """Parse _sleepscoring.txt → {epoch_index: label} for 30-s sleep-stage rows.

    CSV columns: Date, Time, Recording onset, Duration, Annotation, Linked channel
    We key by round(onset/30) so 'Lights off' (duration 0, off-grid) rows and
    any gaps are handled robustly.
    """
    epochs = {}
    for raw_line in Path(txt_path).read_text(errors="ignore").splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) < 5 or not parts[4].startswith("Sleep stage"):
            continue
        try:
            onset = float(parts[2])
        except ValueError:
            continue
        stage = parts[4].replace("Sleep stage", "").strip()
        if stage in HMC_LABEL_MAP:
            epochs[int(round(onset / EPOCH_S))] = HMC_LABEL_MAP[stage]
    return epochs


class HMCDataset(Dataset):
    """HMC sleep-staging, 5-class. Returns (trial [T,19] float32, label int)."""

    def __init__(
        self,
        data_dir: str,
        recordings: list[str],
        sample_rate: int = 256,
        trial_duration_s: int = 10,
        normalization: str = "per_recording_robust",
        cache_dir: Optional[str] = None,
    ):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")
        self.sample_rate = sample_rate
        self.trial_samples = sample_rate * trial_duration_s
        self.trial_duration_s = trial_duration_s
        self.normalization = normalization
        self.trials: list[np.ndarray] = []
        self.labels: list[int] = []
        self.subject_ids: list[int] = []
        self.electrode_names = None

        _, rec_dir = list_hmc_recordings(data_dir)

        cache_path = None
        if cache_dir:
            key = _cache_key({
                "kind": "hmc",
                "data_dir": str(data_dir),
                "recordings": ",".join(sorted(recordings)),
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "normalization": normalization,
                "channels": "eeg4_named_v1",
            })
            cache_path = Path(cache_dir) / f"hmc_{key}.pt"
            if cache_path.exists():
                print(f"  ↻ loading cache: {cache_path.name}")
                c = torch.load(cache_path, weights_only=False)
                self.trials = c["trials"]; self.labels = c["labels"]
                self.subject_ids = c["subject_ids"]
                print(f"    ← {len(self.trials)} trials, "
                      f"{len(set(self.subject_ids))} recordings")
                self._print_dist()
                return

        print(f"  [HMC] Loading {len(recordings)} recordings from {rec_dir}")
        for si, rid in enumerate(sorted(recordings)):
            try:
                self._process(rec_dir, rid, si)
            except Exception as e:
                print(f"    skip {rid}: {type(e).__name__}: {e}")
        print(f"  [HMC] Loaded {len(self.trials)} trials from "
              f"{len(set(self.subject_ids))} recordings")
        self._print_dist()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"trials": self.trials, "labels": self.labels,
                        "subject_ids": self.subject_ids}, cache_path)
            print(f"  ↑ saved cache: {cache_path.name}")

    def _process(self, rec_dir: Path, rid: str, si: int):
        psg = rec_dir / f"{rid}.edf"
        txt = rec_dir / f"{rid}_sleepscoring.txt"
        if not txt.exists():
            raise FileNotFoundError(f"no scoring txt for {rid}")
        raw = read_raw_edf(str(psg), preload=True, verbose=False)
        ch = _pick_hmc_eeg(raw)
        ch_idx = [raw.ch_names.index(c) for c in ch]
        if int(round(raw.info["sfreq"])) != self.sample_rate:
            raw.resample(self.sample_rate, verbose=False)
        sig = raw.get_data()[ch_idx].T.astype(np.float32) * 1e6   # [T, 4] µV
        if self.normalization == "per_recording_robust":
            sig = _robust_scale_per_recording(sig)
        if self.electrode_names is None:
            self.electrode_names = [
                "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
                "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
                "P3", "P4", "Pz", "O1", "O2"]

        epochs = _parse_hmc_scoring(txt)      # {epoch_idx: label}
        ep_samp = self.sample_rate * EPOCH_S
        offset = (EPOCH_S - self.trial_duration_s) // 2 * self.sample_rate
        n_ep_sig = sig.shape[0] // ep_samp
        for i in range(n_ep_sig):
            if i not in epochs:
                continue
            s = i * ep_samp + offset
            e = s + self.trial_samples
            if e > sig.shape[0]:
                break
            trial_4 = sig[s:e]
            if self.normalization == "per_trial_zscore":
                trial_4 = ((trial_4 - trial_4.mean(0)) / (trial_4.std(0) + 1e-8)).astype(np.float32)
            trial_19 = np.zeros((self.trial_samples, 19), dtype=np.float32)
            for src_idx, cname in enumerate(HMC_EEG_CHANNELS):
                trial_19[:, HMC_CH_TO_19CH_IDX[cname]] = trial_4[:, src_idx]
            self.trials.append(trial_19)
            self.labels.append(epochs[i])
            self.subject_ids.append(si)

    def _print_dist(self):
        if not self.labels:
            print("    Class dist: (empty)"); return
        bc = np.bincount(self.labels, minlength=5)
        print(f"    Class dist: " +
              "  ".join(f"{n}={c}" for n, c in zip(LABEL_NAMES, bc)))

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, i):
        return torch.from_numpy(self.trials[i]), int(self.labels[i])


def hmc_inspect(data_dir):
    ids, rec_dir = list_hmc_recordings(data_dir)
    print(f"Found {len(ids)} HMC recordings under {rec_dir}")
    sp = make_hmc_split(data_dir)
    print(f"split: train {len(sp['train'])} / val {len(sp['val'])} / test {len(sp['test'])}")
    # parse one scoring file as a sanity check
    if ids:
        ep = _parse_hmc_scoring(rec_dir / f"{ids[0]}_sleepscoring.txt")
        bc = np.bincount(list(ep.values()), minlength=5)
        print(f"{ids[0]}: {len(ep)} scored epochs, dist " +
              " ".join(f"{n}={c}" for n, c in zip(LABEL_NAMES, bc)))


if __name__ == "__main__":
    import sys
    hmc_inspect(sys.argv[1] if len(sys.argv) > 1
                else "/home/pxieaf/home2/datasets/HMC")
