"""
ISRUC-Sleep Subgroup I loader matching CBraMod protocol.

CBraMod (ICLR 2025) Table on ISRUC: BA 0.6655, κ 0.5567, F1 0.6499
  Source: 100 healthy subjects, 6 EEG channels @ 200 Hz, 30-sec epochs,
          5-class sleep stage (W, N1, N2, N3, REM)
  Split:  Subject-disjoint fixed — train 1-80, val 81-90, test 91-100

Our adaptation:
  - ISRUC's 6 channels (F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1)
    are placed into our standard 19-channel 10-20 positions; the other
    13 channels are zero-padded (same convention as TUH eval).
  - 30-second epochs are truncated/cropped to the MIDDLE 10 seconds to
    match our pretrain window (10 s, max_seq_len=128 tokens at 256 Hz).
  - per_recording_robust normalization (Defossez/Laya-style) applied
    on the full recording before epoch segmentation, matching pretrain.

ISRUC file layout (Subgroup I):
    isruc_root/{N}/{N}.rec       # PSG signal (EDF inside despite .rec ext)
              /{N}/{N}_1.txt     # 1 sleep stage per line, expert 1
              /{N}/{N}_2.txt     # expert 2 (we use _1)

Usage:
    train_ds = ISRUCDataset(
        data_dir="/path/to/isruc_root",
        subjects=list(range(1, 81)),
        sample_rate=256,
        trial_duration_s=10,        # middle 10s of 30s epoch
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
    mne.set_log_level("ERROR")
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# 6 ISRUC EEG channels (referenced to contralateral mastoid)
ISRUC_EEG_CHANNELS = [
    "F3-A2", "C3-A2", "O1-A2",
    "F4-A1", "C4-A1", "O2-A1",
]

# Mapping into our 19-channel 10-20 standard order
# ['Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz','T3','T4','T5','T6','P3','P4','Pz','O1','O2']
ISRUC_CH_TO_19CH_IDX = {
    "F3-A2": 2,    # F3
    "C3-A2": 7,    # C3
    "O1-A2": 17,   # O1
    "F4-A1": 3,    # F4
    "C4-A1": 8,    # C4
    "O2-A1": 18,   # O2
}

# Raw label code -> 0-indexed class id (W, N1, N2, N3, REM)
# Note: REM is coded as '5' in ISRUC files, not 4
LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}
LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CLASSES = 5

EPOCH_S = 30
ISRUC_NATIVE_SR = 200

# CBraMod's fixed subject-level split
CBRAMOD_ISRUC_SPLITS = {
    "train": list(range(1, 81)),
    "val":   list(range(81, 91)),
    "test":  list(range(91, 101)),
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


def _resolve_subject_dir(data_dir: Path, subj: int) -> Path:
    """Find subject's directory: could be {data_dir}/{N}/ or {data_dir}/Subject{N}/."""
    candidates = [
        data_dir / str(subj),
        data_dir / f"Subject{subj}",
        data_dir / "subgroupI" / str(subj),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"ISRUC subject {subj}: tried {[str(c) for c in candidates]}, "
        f"none exist. Inspect data_dir={data_dir}.")


def _find_rec_and_labels(subj_dir: Path, subj: int):
    """Locate the .rec and _1.txt for this subject."""
    # Try standard naming
    rec = subj_dir / f"{subj}.rec"
    txt = subj_dir / f"{subj}_1.txt"
    if not rec.exists():
        # Fall back to first .rec in directory
        recs = sorted(subj_dir.glob("*.rec"))
        if not recs:
            recs = sorted(subj_dir.glob("*.edf"))
        if not recs:
            raise FileNotFoundError(f"No .rec/.edf in {subj_dir}")
        rec = recs[0]
    if not txt.exists():
        txts = sorted(subj_dir.glob("*_1.txt"))
        if not txts:
            txts = sorted(subj_dir.glob("*_1.csv"))
        if not txts:
            txts = sorted(subj_dir.glob("*.txt"))
        if not txts:
            raise FileNotFoundError(f"No labels in {subj_dir}")
        txt = txts[0]
    return rec, txt


def _pick_isruc_eeg_channels(raw):
    """Pick the 6 EEG channels from a raw object; handle naming variants."""
    available = [ch for ch in raw.ch_names]
    # Try exact match
    found = {ch: ch for ch in ISRUC_EEG_CHANNELS if ch in available}
    if len(found) == 6:
        return [found[ch] for ch in ISRUC_EEG_CHANNELS]
    # Try with case insensitivity and "M" instead of "A"
    upper_map = {ch.upper(): ch for ch in available}
    found = []
    for want in ISRUC_EEG_CHANNELS:
        cands = [
            want.upper(),
            want.replace("A2", "M2").upper(),
            want.replace("A1", "M1").upper(),
        ]
        match = None
        for c in cands:
            if c in upper_map:
                match = upper_map[c]
                break
        if match is None:
            raise ValueError(
                f"Could not find ISRUC channel {want} in {available}")
        found.append(match)
    return found


class ISRUCDataset(Dataset):
    """ISRUC sleep stage dataset, 5-class (W/N1/N2/N3/REM).

    Returns:
        trial: torch.Tensor [trial_samples, 19] float32 (ISRUC's 6 EEG
               channels placed at correct 10-20 positions; other 13 zero)
        label: int (0..4)
    Plus self.subject_ids tracking source subject.
    """

    def __init__(
        self,
        data_dir: str,
        subjects: list[int],
        sample_rate: int = 256,
        trial_duration_s: int = 10,
        normalization: str = "per_recording_robust",
        cache_dir: Optional[str] = None,
    ):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")
        assert normalization in ("per_trial_zscore", "per_recording_robust")
        assert trial_duration_s <= EPOCH_S, (
            f"trial_duration_s={trial_duration_s} must be <= ISRUC epoch "
            f"length {EPOCH_S}s")

        self.sample_rate = sample_rate
        self.trial_samples = sample_rate * trial_duration_s
        self.normalization = normalization
        self.trial_duration_s = trial_duration_s
        self.trials: list[np.ndarray] = []
        self.labels: list[int] = []
        self.subject_ids: list[int] = []
        self.electrode_names = None    # set after first subject

        data_dir = Path(data_dir)

        # Cache
        cache_path = None
        if cache_dir:
            key = _cache_key({
                "kind": "isruc",
                "data_dir": str(data_dir),
                "subjects": ",".join(str(s) for s in sorted(subjects)),
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "normalization": normalization,
            })
            cache_path = Path(cache_dir) / f"isruc_{key}.pt"
            if cache_path.exists():
                print(f"  ↻ loading cache: {cache_path.name}")
                cached = torch.load(cache_path, weights_only=False)
                self.trials = cached["trials"]
                self.labels = cached["labels"]
                self.subject_ids = cached["subject_ids"]
                self.electrode_names = cached["electrode_names"]
                print(f"    ← {len(self.trials)} trials, "
                      f"{len(set(self.subject_ids))} subjects")
                self._print_class_dist()
                return

        print(f"  [ISRUC] Loading subjects {subjects[:3]}..{subjects[-2:]} "
              f"({len(subjects)} subj) from {data_dir}")
        for subj in subjects:
            try:
                self._process_subject(subj, data_dir)
            except Exception as e:
                print(f"    skip subject {subj}: {type(e).__name__}: {e}")
                continue

        print(f"  [ISRUC] Loaded {len(self.trials)} trials "
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

    def _process_subject(self, subj: int, data_dir: Path):
        subj_dir = _resolve_subject_dir(data_dir, subj)
        rec, txt = _find_rec_and_labels(subj_dir, subj)

        # ── Read EDF (.rec) ──
        raw = read_raw_edf(str(rec), preload=True, verbose=False)
        # CBraMod-style filter
        raw.filter(0.3, 35.0, fir_design="firwin", verbose=False)
        raw.notch_filter([50.0], verbose=False)
        if raw.info["sfreq"] != self.sample_rate:
            raw.resample(self.sample_rate, verbose=False)

        # ── Pick 6 EEG channels in canonical order ──
        ch_names = _pick_isruc_eeg_channels(raw)
        ch_idx = [raw.ch_names.index(ch) for ch in ch_names]
        sig = raw.get_data()[ch_idx].T.astype(np.float32)   # [T, 6], V
        sig = sig * 1e6     # → µV (matches our other loaders)

        if self.electrode_names is None:
            # Report the 19-channel 10-20 names (electrode_names is the
            # downstream-visible name list; ISRUC 6 ch slot into 19)
            self.electrode_names = [
                "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
                "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
                "P3", "P4", "Pz", "O1", "O2",
            ]

        # ── Per-recording robust normalization (before epoch segmentation) ──
        if self.normalization == "per_recording_robust":
            sig = _robust_scale_per_recording(sig)

        # ── Read labels (one per 30-s epoch, expert 1) ──
        labels_raw: list[int] = []
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line in LABEL_MAP:
                    labels_raw.append(LABEL_MAP[line])
                # ignore unknown codes (sometimes MT, ?, etc.)

        if not labels_raw:
            raise ValueError(f"No valid labels in {txt}")

        # ── Segment into 30-s epochs, then take MIDDLE trial_duration_s ──
        epoch_samples = self.sample_rate * EPOCH_S
        offset = (EPOCH_S - self.trial_duration_s) // 2 * self.sample_rate

        n_epochs_available = sig.shape[0] // epoch_samples
        n_epochs = min(n_epochs_available, len(labels_raw))

        for ep in range(n_epochs):
            ep_start = ep * epoch_samples
            mid_start = ep_start + offset
            mid_end = mid_start + self.trial_samples
            if mid_end > sig.shape[0]:
                break
            trial_6ch = sig[mid_start:mid_end]    # [trial_samples, 6]

            if self.normalization == "per_trial_zscore":
                mean = trial_6ch.mean(axis=0, keepdims=True)
                std = trial_6ch.std(axis=0, keepdims=True) + 1e-8
                trial_6ch = ((trial_6ch - mean) / std).astype(np.float32)

            # ── Place 6 channels into 19-channel layout, zero-pad rest ──
            trial_19 = np.zeros(
                (self.trial_samples, 19), dtype=np.float32)
            for src_idx, src_name in enumerate(ISRUC_EEG_CHANNELS):
                dst_idx = ISRUC_CH_TO_19CH_IDX[src_name]
                trial_19[:, dst_idx] = trial_6ch[:, src_idx]

            self.trials.append(trial_19)
            self.labels.append(labels_raw[ep])
            self.subject_ids.append(subj)

    def _print_class_dist(self):
        if not self.labels:
            return
        bc = np.bincount(self.labels, minlength=N_CLASSES)
        print(f"    Class dist: " + " ".join(
            f"{LABEL_NAMES[i]}={bc[i]}" for i in range(N_CLASSES)))

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), int(self.labels[idx])
