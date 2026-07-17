"""Parkinson's resting-EEG dataset loader (UNM, Cavanagh — OpenNeuro ds003490).

BIDS layout:
  sub-XXX/ses-01/eeg/sub-XXX_ses-01_task-Rest_eeg.set (+ .fdt)
  sub-XXX/ses-02/eeg/...                                (PD only; CTL have ses-01)
participants.tsv:
  participant_id  Original_ID  Group(PD/CTL)  sess1_Med  sess2_Med  sex  age
  -> sess1_Med / sess2_Med are ON / OFF (PD) or n/a / "no s2" (CTL).

Two label modes:
  on_off   : WITHIN-PD medication contrast (ON=0 vs OFF=1). Each PD subject
             contributes BOTH sessions; LOSO holds out a subject (both sessions)
             so train never sees that subject. This is the CLEAN cross-frequency
             test: the within-subject L-DOPA contrast controls out the subject's
             baseline spectrum, so marginal band power should struggle while
             beta-gamma PAC (modulated by L-DOPA; Swann 2015) should not.
  pd_vs_hc : PD (OFF session) vs CTL (ses-01). Between-group; partly band-power
             separable (a documented risk to the coupling argument).

64-ch 10-10 montage -> mapped to our 19-ch 10-20 by name, with old<->new aliases
(T3=T7, T4=T8, T5=P7, T6=P8).
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_mumtaz import (
    _robust_scale_per_recording, _cache_key, OUR_19CH_ORDER, MNE_AVAILABLE,
)

if MNE_AVAILABLE:
    import mne

LABEL_NAMES = {"on_off": ["ON", "OFF"], "pd_vs_hc": ["CTL", "PD"]}
N_CLASSES = 2

# old 10-20 name -> acceptable aliases in a 10-10 montage
_ALIASES = {"T3": ["T3", "T7"], "T4": ["T4", "T8"],
            "T5": ["T5", "P7"], "T6": ["T6", "P8"]}


def _pick_19(raw) -> list[str]:
    avail = raw.ch_names
    up = {c.upper(): c for c in avail}
    picked = []
    for std in OUR_19CH_ORDER:
        cands = _ALIASES.get(std, [std])
        m = None
        for c in cands:
            if c in avail:
                m = c; break
            if c.upper() in up:
                m = up[c.upper()]; break
        if m is None:
            raise ValueError(f"channel {std} (tried {cands}) not in {avail}")
        picked.append(m)
    return picked


def _read_participants(data_dir):
    """participant_id -> dict(group, sess_med={1:'ON'/..,2:'OFF'/..})."""
    out = {}
    with open(Path(data_dir) / "participants.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["participant_id"]] = {
                "group": row.get("Group", "").strip().upper(),
                1: row.get("sess1_Med", "").strip().upper(),
                2: row.get("sess2_Med", "").strip().upper(),
            }
    return out


def list_all_subjects(data_dir):
    return sorted(_read_participants(data_dir).keys())


def _sid_int(participant_id: str) -> int:
    return int("".join(ch for ch in participant_id if ch.isdigit()))


class ParkinsonsDataset(Dataset):
    """ds003490 PD resting EEG -> (trial [T,19], label). self.subject_ids groups
    LOSO folds (both sessions of a PD subject share one id)."""

    def __init__(self, data_dir: str, subjects: list[str], label: str = "on_off",
                 sample_rate: int = 256, trial_duration_s: int = 10,
                 normalization: str = "per_recording_robust",
                 cache_dir: Optional[str] = None):
        if not MNE_AVAILABLE:
            raise ImportError("mne required")
        assert label in ("on_off", "pd_vs_hc")
        assert normalization in ("per_trial_zscore", "per_recording_robust")
        self.sample_rate = sample_rate
        self.trial_samples = sample_rate * trial_duration_s
        self.normalization = normalization
        self.label_mode = label
        self.trials: list[np.ndarray] = []
        self.labels: list[int] = []
        self.subject_ids: list[int] = []
        data_dir = Path(data_dir)

        cache_path = None
        if cache_dir:
            key = _cache_key({"kind": "parkinsons_ds003490", "label": label,
                              "data_dir": str(data_dir),
                              "subjects": ",".join(sorted(subjects)),
                              "sample_rate": sample_rate,
                              "trial_duration_s": trial_duration_s,
                              "normalization": normalization})
            cache_path = Path(cache_dir) / f"parkinsons_{key}.pt"
            if cache_path.exists():
                print(f"  ↻ loading cache: {cache_path.name}")
                c = torch.load(cache_path, weights_only=False)
                self.trials, self.labels, self.subject_ids = (
                    c["trials"], c["labels"], c["subject_ids"])
                print(f"    ← {len(self.trials)} trials, "
                      f"{len(set(self.subject_ids))} subjects")
                self._dist(); return

        meta = _read_participants(data_dir)
        recordings = self._plan_recordings(subjects, meta)   # (sub, ses, label)
        print(f"  [PD/{label}] {len(recordings)} recordings from "
              f"{len(set(s for s,_,_ in recordings))} subjects")
        for sub, ses, lab in recordings:
            setf = (data_dir / sub / f"ses-{ses:02d}" / "eeg" /
                    f"{sub}_ses-{ses:02d}_task-Rest_eeg.set")
            if not setf.exists():
                print(f"    skip {setf.name}: missing"); continue
            try:
                self._process(setf, lab, _sid_int(sub))
            except Exception as e:
                print(f"    skip {setf.name}: {type(e).__name__}: {e}")

        print(f"  [PD/{label}] Loaded {len(self.trials)} trials")
        self._dist()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"trials": self.trials, "labels": self.labels,
                        "subject_ids": self.subject_ids}, cache_path)
            print(f"  ↑ saved cache: {cache_path.name}")

    def _plan_recordings(self, subjects, meta):
        """Return list of (participant_id, session_int, label_int)."""
        rec = []
        for sub in subjects:
            m = meta.get(sub)
            if m is None:
                continue
            if self.label_mode == "on_off":
                if m["group"] != "PD":
                    continue
                for ses in (1, 2):
                    med = m[ses]
                    if med == "ON":
                        rec.append((sub, ses, 0))
                    elif med == "OFF":
                        rec.append((sub, ses, 1))
            else:  # pd_vs_hc
                if m["group"] == "PD":
                    off_ses = 1 if m[1] == "OFF" else (2 if m[2] == "OFF" else 1)
                    rec.append((sub, off_ses, 1))
                elif m["group"] in ("CTL", "HC", "CONTROL"):
                    rec.append((sub, 1, 0))
        return rec

    def _process(self, setf: Path, label: int, sid: int):
        raw = mne.io.read_raw_eeglab(str(setf), preload=True, verbose="ERROR")
        raw.pick(_pick_19(raw))
        if int(round(raw.info["sfreq"])) != self.sample_rate:
            raw = raw.resample(self.sample_rate, verbose="ERROR")
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
        names = LABEL_NAMES[self.label_mode]
        print(f"    Class dist: {names[0]}={int((a==0).sum())}  "
              f"{names[1]}={int((a==1).sum())}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), int(self.labels[idx])
