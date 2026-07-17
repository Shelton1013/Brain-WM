"""Schizophrenia resting-state EEG dataset loader (Olejarczyk & Jernajczyk 2017).

Source: RepOD "EEG in schizophrenia"
  DOI: 10.18150/repod.0107441
  https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441
  Paper: Olejarczyk & Jernajczyk, PLOS ONE 12(11):e0188629 (2017)

Files: h01.edf ... h14.edf  (14 healthy controls)
       s01.edf ... s14.edf  (14 paranoid schizophrenia)
Native format: 19-ch standard 10-20 (Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4
  T5 P3 Pz P4 T6 O1 O2), 250 Hz, ~15 min eyes-closed rest. Matches our 19-ch
  montage exactly (only 250->256 Hz resample needed).

Why this task for a frequency-native model: schizophrenia's core oscillopathy
is disrupted theta-phase / gamma-amplitude coupling and dysregulated gamma
synchrony at rest (Uhlhaas & Singer 2010; Won et al. 2018) — a cross-frequency
phenomenon, NOT purely marginal band power. This dataset is in NO competitor
benchmark suite (LaBraM/BIOT/CBraMod/EEGPT/CSBrain).

Reuses the Mumtaz preprocessing exactly (name-based 19-ch pick, per-recording
robust scale, non-overlapping windows) so it drops into the same eval pipeline.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_mumtaz import (
    _robust_scale_per_recording, _cache_key, _pick_mumtaz_channels,
    MNE_AVAILABLE,
)

if MNE_AVAILABLE:
    from mne.io import read_raw_edf

LABEL_NAMES = ["Healthy", "Schizophrenia"]
N_CLASSES = 2
NATIVE_SR = 250

# h<NN>.edf -> healthy, s<NN>.edf -> schizophrenia
_FNAME_RE = re.compile(r"^(?P<group>[hs])(?P<subj>\d+)\.edf$", re.IGNORECASE)


def _list_subjects(data_dir):
    """Return {(group, sid): [edf_path]}; group in {'H','SZ'}."""
    data_dir = Path(data_dir)
    subjects: dict[tuple[str, int], list[Path]] = {}
    for p in sorted(data_dir.rglob("*.edf")):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        group = "SZ" if m.group("group").lower() == "s" else "H"
        sid = int(m.group("subj"))
        subjects.setdefault((group, sid), []).append(p)
    return subjects


def make_subject_split(data_dir: str, seed: int = 42,
                       val_counts: dict | None = None,
                       test_counts: dict | None = None) -> dict:
    """Subject-disjoint split with fixed COUNTS (seed permutes which subjects).

    No competitor uses this dataset, so we define a balanced subject-disjoint
    split: test 4 SZ/4 HC, val 2/2, train = rest (8/8). Multi-seed averages
    over the small held-out set (only 28 subjects total).
    """
    if val_counts is None:
        val_counts = {"SZ": 2, "H": 2}
    if test_counts is None:
        test_counts = {"SZ": 4, "H": 4}
    subjects = _list_subjects(Path(data_dir))
    grouped = {"H": sorted(sid for grp, sid in subjects if grp == "H"),
               "SZ": sorted(sid for grp, sid in subjects if grp == "SZ")}
    rng = np.random.RandomState(seed)
    splits = {"train": {}, "val": {}, "test": {}}
    for grp, sids in grouped.items():
        sids = list(sids); rng.shuffle(sids)
        n_te = test_counts.get(grp, 0); n_va = val_counts.get(grp, 0)
        splits["test"][grp]  = sids[:n_te]
        splits["val"][grp]   = sids[n_te:n_te + n_va]
        splits["train"][grp] = sids[n_te + n_va:]
    print("  [SZ split] " + " ".join(
        f"{s}:SZ{len(splits[s]['SZ'])}/H{len(splits[s]['H'])}"
        for s in ("train", "val", "test")))
    return splits


class SchizophreniaDataset(Dataset):
    """Olejarczyk schizophrenia EEG, 2-class (SZ/Healthy).

    Returns (trial [trial_samples, 19] float32, label int). Also self.labels /
    self.subject_ids (enc: +sid for H, -sid for SZ). Preprocessing identical to
    MumtazDataset.
    """

    def __init__(self, data_dir: str, subjects: dict,
                 sample_rate: int = 256, trial_duration_s: int = 10,
                 normalization: str = "per_recording_robust",
                 cache_dir: Optional[str] = None):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")
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
            key = _cache_key({
                "kind": "schizophrenia", "data_dir": str(data_dir),
                "subjects": ",".join(
                    f"{grp}:{','.join(str(s) for s in sorted(subjects[grp]))}"
                    for grp in sorted(subjects)),
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "normalization": normalization})
            cache_path = Path(cache_dir) / f"schizophrenia_{key}.pt"
            if cache_path.exists():
                print(f"  ↻ loading cache: {cache_path.name}")
                cached = torch.load(cache_path, weights_only=False)
                self.trials = cached["trials"]; self.labels = cached["labels"]
                self.subject_ids = cached["subject_ids"]
                print(f"    ← {len(self.trials)} trials, "
                      f"{len(set(self.subject_ids))} subjects")
                self._print_class_dist(); return

        all_files = _list_subjects(data_dir)
        wanted = {("H", sid) for sid in subjects.get("H", [])} | \
                 {("SZ", sid) for sid in subjects.get("SZ", [])}
        print(f"  [SZ] Loading H {sorted(subjects.get('H', []))}, "
              f"SZ {sorted(subjects.get('SZ', []))} "
              f"({len(wanted)} subjects) from {data_dir}")

        for (group, sid), edf_paths in sorted(all_files.items()):
            if (group, sid) not in wanted:
                continue
            label = 1 if group == "SZ" else 0
            enc_sid = (-sid) if group == "SZ" else sid
            for edf in sorted(edf_paths):
                try:
                    self._process_edf(edf, label, enc_sid)
                except Exception as e:
                    print(f"    skip {edf.name}: {type(e).__name__}: {e}")

        print(f"  [SZ] Loaded {len(self.trials)} trials from "
              f"{len(set(self.subject_ids))} subjects")
        self._print_class_dist()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"trials": self.trials, "labels": self.labels,
                        "subject_ids": self.subject_ids}, cache_path)
            print(f"  ↑ saved cache: {cache_path.name}")

    def _process_edf(self, edf: Path, label: int, enc_sid: int):
        raw = read_raw_edf(str(edf), preload=True, verbose=False)
        raw.pick(_pick_mumtaz_channels(raw))            # 19-ch, our order
        if int(round(raw.info["sfreq"])) != self.sample_rate:
            raw = raw.resample(self.sample_rate, verbose=False)
        sig = raw.get_data().T.astype(np.float32)       # (T, 19)
        if self.normalization == "per_recording_robust":
            sig = _robust_scale_per_recording(sig)
        n_trials = sig.shape[0] // self.trial_samples
        for i in range(n_trials):
            trial = sig[i * self.trial_samples:(i + 1) * self.trial_samples]
            if self.normalization == "per_trial_zscore":
                trial = (trial - trial.mean(0)) / (trial.std(0) + 1e-6)
            self.trials.append(trial.astype(np.float32))
            self.labels.append(int(label))
            self.subject_ids.append(int(enc_sid))

    def _print_class_dist(self):
        if not self.labels:
            print("    Class dist: (empty)"); return
        arr = np.asarray(self.labels)
        print(f"    Class dist: Healthy={int((arr==0).sum())}  "
              f"SZ={int((arr==1).sum())}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), int(self.labels[idx])
