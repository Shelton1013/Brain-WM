"""Mental Arithmetic EEG dataset (PhysioNet 'eegmat', Zyma et al. 2019).

Source: https://physionet.org/content/eegmat/1.0.0/
Files: Subject{NN}_1.edf (resting baseline, label 0) and Subject{NN}_2.edf
       (during serial subtraction / mental arithmetic, label 1). 36 subjects.
Native: 21 ch (19-ch 10-20 with 'EEG ' prefix + A2-A1 ref + ECG), 500 Hz.

Task: binary rest vs mental-arithmetic. The discriminative signal is spectral
(alpha suppression + frontal-midline theta during arithmetic) — in scope for a
spectral-prediction model. Maps cleanly to our 19-ch 10-20 (all present).

Reuses Mumtaz preprocessing (name-based 19-ch pick, robust scale, 10-s windows).
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_mumtaz import (
    _robust_scale_per_recording, _cache_key, _pick_mumtaz_channels, MNE_AVAILABLE,
)

if MNE_AVAILABLE:
    from mne.io import read_raw_edf

LABEL_NAMES = ["rest", "arithmetic"]
N_CLASSES = 2
NATIVE_SR = 500

# Subject00_1.edf -> sid 0, cond 1 (rest) ; _2 -> arithmetic
_FNAME_RE = re.compile(r"^Subject(?P<sid>\d+)_(?P<cond>[12])\.edf$")


def _list_subjects(data_dir):
    """{sid: {1: path_rest, 2: path_task}}."""
    out: dict[int, dict[int, Path]] = {}
    for p in sorted(Path(data_dir).rglob("*.edf")):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        out.setdefault(int(m.group("sid")), {})[int(m.group("cond"))] = p
    return out


def make_subject_split(data_dir, seed=42, n_test=8, n_val=4):
    """Subject-disjoint fixed-count split (seed permutes which subjects)."""
    sids = sorted(_list_subjects(data_dir).keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(sids)
    split = {"test": sorted(sids[:n_test]),
             "val":  sorted(sids[n_test:n_test + n_val]),
             "train": sorted(sids[n_test + n_val:])}
    print(f"  [MA split] train {len(split['train'])} / val {len(split['val'])} "
          f"/ test {len(split['test'])} subjects")
    return split


class MentalArithmeticDataset(Dataset):
    """rest (0) vs arithmetic (1); returns (trial [T,19], label)."""

    def __init__(self, data_dir, subjects, sample_rate=256, trial_duration_s=10,
                 normalization="per_recording_robust", cache_dir: Optional[str] = None):
        if not MNE_AVAILABLE:
            raise ImportError("mne required")
        assert normalization in ("per_trial_zscore", "per_recording_robust")
        self.sample_rate = sample_rate
        self.trial_samples = sample_rate * trial_duration_s
        self.normalization = normalization
        self.trials: list[np.ndarray] = []
        self.labels: list[int] = []
        self.subject_ids: list[int] = []
        data_dir = Path(data_dir)

        cache_path = None
        if cache_dir:
            key = _cache_key({"kind": "mental_arithmetic", "data_dir": str(data_dir),
                              "subjects": ",".join(str(s) for s in sorted(subjects)),
                              "sample_rate": sample_rate,
                              "trial_duration_s": trial_duration_s,
                              "normalization": normalization})
            cache_path = Path(cache_dir) / f"mentalarith_{key}.pt"
            if cache_path.exists():
                print(f"  ↻ loading cache: {cache_path.name}")
                c = torch.load(cache_path, weights_only=False)
                self.trials, self.labels, self.subject_ids = (
                    c["trials"], c["labels"], c["subject_ids"])
                print(f"    ← {len(self.trials)} trials, {len(set(self.subject_ids))} subj")
                self._dist(); return

        allsub = _list_subjects(data_dir)
        print(f"  [MA] loading subjects {sorted(subjects)} from {data_dir}")
        for sid in subjects:
            for cond, label in ((1, 0), (2, 1)):   # _1 rest=0, _2 task=1
                f = allsub.get(sid, {}).get(cond)
                if f is None:
                    continue
                try:
                    self._process(f, label, sid)
                except Exception as e:
                    print(f"    skip {f.name}: {type(e).__name__}: {e}")
        print(f"  [MA] Loaded {len(self.trials)} trials")
        self._dist()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"trials": self.trials, "labels": self.labels,
                        "subject_ids": self.subject_ids}, cache_path)
            print(f"  ↑ saved cache: {cache_path.name}")

    def _process(self, edf: Path, label: int, sid: int):
        raw = read_raw_edf(str(edf), preload=True, verbose=False)
        raw.pick(_pick_mumtaz_channels(raw))
        if int(round(raw.info["sfreq"])) != self.sample_rate:
            raw = raw.resample(self.sample_rate, verbose=False)
        sig = raw.get_data().T.astype(np.float32)          # (T, 19)
        if self.normalization == "per_recording_robust":
            sig = _robust_scale_per_recording(sig)
        n = sig.shape[0] // self.trial_samples
        for i in range(n):
            tr = sig[i * self.trial_samples:(i + 1) * self.trial_samples]
            if self.normalization == "per_trial_zscore":
                tr = (tr - tr.mean(0)) / (tr.std(0) + 1e-6)
            self.trials.append(tr.astype(np.float32))
            self.labels.append(int(label))
            self.subject_ids.append(int(sid))

    def _dist(self):
        if not self.labels:
            print("    (empty)"); return
        a = np.asarray(self.labels)
        print(f"    Class dist: rest={int((a==0).sum())}  arithmetic={int((a==1).sum())}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), int(self.labels[idx])
