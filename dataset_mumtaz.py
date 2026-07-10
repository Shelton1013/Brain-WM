"""Mumtaz2016 Depression EEG dataset loader.

Source: Figshare EEG_Data_New (Mumtaz et al., 2016 IEEE TNSRE)
  DOI: 10.6084/M9.FIGSHARE.4244171.V2
  https://figshare.com/articles/dataset/EEG_Data_New/4244171

Files: MDD_S{N}(_)*_(EC|EO|TASK).edf  or  H_S{N}(_)*_(EC|EO|TASK).edf
  MDD: Major Depressive Disorder patient
  H:   Healthy control
  EC:  Eyes closed (~5 min)
  EO:  Eyes open (~5 min)
  TASK: P300 auditory task — DROP (different modality)

Native format: 19 channels 10-20 with -LE (linked-ear) reference @ 256 Hz.
Matches our model exactly — no channel padding or resampling needed.

Protocol matched to CBraMod's `preprocessing_mumtaz.py`:
  - Drop TASK files
  - 19 channels in the -LE reference form
  - Subject-disjoint fixed split
Our differences (safer for our pretrained model):
  - 256 Hz native (CBraMod uses 200 Hz)
  - 10 s window (CBraMod uses 5 s at 200 Hz)
  - Per-recording robust normalization (matches our pretrain)
"""
from __future__ import annotations
import hashlib
import re
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


# 19 EEG channels in the -LE (linked-ear) reference, matching CBraMod
# https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_mumtaz.py
MUMTAZ_EEG_CHANNELS_LE = [
    "EEG Fp1-LE", "EEG Fp2-LE",
    "EEG F3-LE",  "EEG F4-LE",
    "EEG F7-LE",  "EEG F8-LE",
    "EEG Fz-LE",
    "EEG C3-LE",  "EEG C4-LE",  "EEG Cz-LE",
    "EEG T3-LE",  "EEG T4-LE",
    "EEG T5-LE",  "EEG T6-LE",
    "EEG P3-LE",  "EEG P4-LE",  "EEG Pz-LE",
    "EEG O1-LE",  "EEG O2-LE",
]
# Bare-name aliases (some recordings just use "Fp1", no prefix / no ref)
MUMTAZ_EEG_CHANNELS_BARE = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
    "P3", "P4", "Pz", "O1", "O2",
]

# Order these into the 19-ch 10-20 layout our model expects
# ['Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz','T3','T4','T5','T6','P3','P4','Pz','O1','O2']
OUR_19CH_ORDER = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
    "P3", "P4", "Pz", "O1", "O2",
]
LABEL_NAMES = ["Healthy", "MDD"]
N_CLASSES = 2
NATIVE_SR = 256


# Match filename patterns (real Mumtaz2016 Figshare format):
#   "H S13 EO.edf"            -> group=H, subj=13, cond=EO
#   "MDD S11  EC.edf"         -> group=MDD, subj=11, cond=EC (double space OK)
#   "H S1 EC.edf"             -> subj=1
# Skip artifacts like "6921143_H S15 EO.edf" (Figshare duplicate downloads).
_FNAME_RE = re.compile(
    r"^(?P<group>MDD|H)\s+S(?P<subj>\d+)\s+(?P<cond>EC|EO|TASK)\.edf$",
    re.IGNORECASE,
)


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


def _pick_mumtaz_channels(raw) -> list[str]:
    """Return the 19 channel names in our canonical 10-20 order."""
    available = raw.ch_names
    upper_map = {ch.upper(): ch for ch in available}

    picked_by_std_name: dict[str, str] = {}   # {"Fp1": actual_ch_name, ...}

    for std_name in OUR_19CH_ORDER:
        # Try known aliases in preferred order
        candidates = [
            f"EEG {std_name}-LE",
            f"EEG {std_name}-REF",
            f"EEG {std_name}",
            std_name,
        ]
        matched = None
        for c in candidates:
            if c in available:
                matched = c; break
            if c.upper() in upper_map:
                matched = upper_map[c.upper()]; break
        if matched is None:
            raise ValueError(
                f"Cannot find channel {std_name} in {available[:10]}...")
        picked_by_std_name[std_name] = matched

    # Return the actual channel names in our canonical order
    return [picked_by_std_name[std] for std in OUR_19CH_ORDER]


def _list_subjects(data_dir):
    """Return list of (subject_id_str, group, edf_paths_for_EC_and_EO)."""
    data_dir = Path(data_dir)
    subjects: dict[tuple[str, int], list[Path]] = {}
    for p in sorted(data_dir.glob("*.edf")):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        group = m.group("group").upper()
        subj = int(m.group("subj"))
        cond = m.group("cond").upper()
        if cond == "TASK":
            continue   # Drop P300 TASK files
        subjects.setdefault((group, subj), []).append(p)
    return subjects


def make_subject_split(data_dir: str,
                       seed: int = 42,
                       val_counts: dict | None = None,
                       test_counts: dict | None = None) -> dict:
    """Subject-disjoint split with CSBrain/CBraMod's fixed COUNTS (not fractions).

    CSBrain (NeurIPS 2025) Mumtaz2016 protocol: 24 MDD + 19 HC train,
    5 MDD + 4 HC val, 5 MDD + 5 HC test. We fix val/test to those exact
    held-out counts (so the evaluation sets are composition-matched to their
    Table 17) and put every remaining subject into train. `seed` only permutes
    WHICH subjects land in each bucket (counts stay fixed) for multi-seed
    robustness; CSBrain report a single split.
    """
    if val_counts is None:
        val_counts = {"MDD": 5, "H": 4}
    if test_counts is None:
        test_counts = {"MDD": 5, "H": 5}
    subjects = _list_subjects(Path(data_dir))
    grouped = {"H": sorted(sid for grp, sid in subjects if grp == "H"),
               "MDD": sorted(sid for grp, sid in subjects if grp == "MDD")}

    rng = np.random.RandomState(seed)
    splits = {"train": {}, "val": {}, "test": {}}
    for grp, sids in grouped.items():
        sids = list(sids)
        rng.shuffle(sids)
        n_te = test_counts.get(grp, 0)
        n_va = val_counts.get(grp, 0)
        splits["test"][grp]  = sids[:n_te]
        splits["val"][grp]   = sids[n_te:n_te + n_va]
        splits["train"][grp] = sids[n_te + n_va:]
    # Transparency: print actual vs CSBrain reference counts.
    print("  [mumtaz split] "
          + " ".join(f"{s}:MDD{len(splits[s]['MDD'])}/H{len(splits[s]['H'])}"
                     for s in ("train", "val", "test"))
          + "  (CSBrain ref: train MDD24/H19 val MDD5/H4 test MDD5/H5)")
    return splits


class MumtazDataset(Dataset):
    """Mumtaz2016 depression EEG dataset, 2-class (MDD/Healthy).

    Returns:
        trial: torch.Tensor [trial_samples, 19] float32
        label: int (0=Healthy, 1=MDD)
    Also self.subject_ids: list[int] tracking source subject index
    (encoded as +sid for Healthy, -sid for MDD to avoid collision).
    """

    def __init__(
        self,
        data_dir: str,
        subjects: dict,             # {"H": [1,2,3,...], "MDD": [1,2,...]}
        sample_rate: int = 256,
        trial_duration_s: int = 10,
        normalization: str = "per_recording_robust",
        cache_dir: Optional[str] = None,
    ):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")
        assert normalization in ("per_trial_zscore", "per_recording_robust")
        self.sample_rate = sample_rate
        self.trial_samples = sample_rate * trial_duration_s
        self.normalization = normalization
        self.trial_duration_s = trial_duration_s
        self.trials: list[np.ndarray] = []
        self.labels: list[int] = []
        self.subject_ids: list[int] = []

        data_dir = Path(data_dir)

        # Cache
        cache_path = None
        if cache_dir:
            key = _cache_key({
                "kind": "mumtaz",
                "data_dir": str(data_dir),
                "subjects": ",".join(
                    f"{grp}:{','.join(str(s) for s in sorted(subjects[grp]))}"
                    for grp in sorted(subjects)),
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "normalization": normalization,
            })
            cache_path = Path(cache_dir) / f"mumtaz_{key}.pt"
            if cache_path.exists():
                print(f"  ↻ loading cache: {cache_path.name}")
                cached = torch.load(cache_path, weights_only=False)
                self.trials = cached["trials"]
                self.labels = cached["labels"]
                self.subject_ids = cached["subject_ids"]
                print(f"    ← {len(self.trials)} trials, "
                      f"{len(set(self.subject_ids))} subjects")
                self._print_class_dist()
                return

        all_files = _list_subjects(data_dir)
        wanted = {("H", sid) for sid in subjects.get("H", [])} | \
                 {("MDD", sid) for sid in subjects.get("MDD", [])}
        print(f"  [Mumtaz] Loading H {sorted(subjects.get('H', []))}, "
              f"MDD {sorted(subjects.get('MDD', []))} "
              f"({len(wanted)} subjects) from {data_dir}")

        for (group, sid), edf_paths in sorted(all_files.items()):
            if (group, sid) not in wanted:
                continue
            label = 1 if group == "MDD" else 0
            # Encode subject_id uniquely: +sid for H, -sid for MDD
            enc_sid = (-sid) if group == "MDD" else sid
            for edf in sorted(edf_paths):
                try:
                    self._process_edf(edf, label, enc_sid)
                except Exception as e:
                    print(f"    skip {edf.name}: {type(e).__name__}: {e}")

        print(f"  [Mumtaz] Loaded {len(self.trials)} trials from "
              f"{len(set(self.subject_ids))} subjects")
        self._print_class_dist()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "trials": self.trials,
                "labels": self.labels,
                "subject_ids": self.subject_ids,
            }, cache_path)
            print(f"  ↑ saved cache: {cache_path.name}")

    def _process_edf(self, edf: Path, label: int, enc_sid: int):
        raw = read_raw_edf(str(edf), preload=True, verbose=False)
        ch_picked = _pick_mumtaz_channels(raw)
        raw.pick(ch_picked)                # order preserved
        # Resample only if native != target
        if int(round(raw.info["sfreq"])) != self.sample_rate:
            raw = raw.resample(self.sample_rate, verbose=False)
        # Get data as (samples, channels)
        sig = raw.get_data().T.astype(np.float32)       # (T, 19)

        # Per-recording robust scaling BEFORE trial segmentation
        if self.normalization == "per_recording_robust":
            sig = _robust_scale_per_recording(sig)

        # Non-overlapping trials
        n_trials = sig.shape[0] // self.trial_samples
        for i in range(n_trials):
            start = i * self.trial_samples
            trial = sig[start:start + self.trial_samples]
            if self.normalization == "per_trial_zscore":
                trial = (trial - trial.mean(0)) / (trial.std(0) + 1e-6)
            self.trials.append(trial.astype(np.float32))
            self.labels.append(int(label))
            self.subject_ids.append(int(enc_sid))

    def _print_class_dist(self):
        if not self.labels:
            print("    Class dist: (empty)")
            return
        arr = np.asarray(self.labels)
        n_h = int((arr == 0).sum())
        n_m = int((arr == 1).sum())
        print(f"    Class dist: Healthy={n_h}  MDD={n_m}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), int(self.labels[idx])
