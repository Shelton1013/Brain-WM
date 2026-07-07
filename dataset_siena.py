"""Siena Scalp EEG Database loader — binary seizure detection.

Source: PhysioNet "Siena Scalp EEG Database" v1.0.0
  https://physionet.org/content/siena-scalp-eeg/1.0.0/
  Detti et al., 2020. 14 epilepsy patients (PN00..PN17), scalp EEG @ 512 Hz,
  10-20 montage (+ EKG / extra channels we drop).

Layout on disk (one folder per patient):
    <root>/PN00/PN00-1.edf
    <root>/PN00/PN00-2.edf
    <root>/PN00/Seizures-list-PN00.txt
    ...

Task (matches CBraMod's CHB-MIT seizure protocol, the closest analog since
CBraMod does not ship a Siena preprocessor):
  - Non-overlapping 10 s windows.
  - label = 1 (ictal) if the window overlaps a registered seizure interval,
    else 0 (interictal). We use interval OVERLAP (stricter/cleaner than
    CBraMod's "boundary falls inside window" test, which misses windows fully
    contained in a long seizure).
  - Subject-disjoint split.
  - 19-ch 10-20 in our canonical order, 256 Hz, per-recording robust norm.

Seizures are rare → the negative class dominates. `negative_per_positive`
caps interictal windows per patient at N× that patient's ictal count so FT
isn't swamped (set to 0 to keep ALL negatives; balanced_accuracy is the
headline metric either way).

⚠ ANNOTATION PARSER: the `Seizures-list-PNxx.txt` format (clock HH.MM.SS
times) is Siena-specific and parsed defensively below. VALIDATE against a real
file before trusting labels — see inspect command printed by parse_seizure_list
on failure, and dataset_siena_inspect() helper.
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


# Canonical 19-ch 10-20 order our model expects (same as dataset_mumtaz).
OUR_19CH_ORDER = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
    "P3", "P4", "Pz", "O1", "O2",
]
LABEL_NAMES = ["Interictal", "Ictal"]
N_CLASSES = 2
NATIVE_SR = 512


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


def _pick_1020_channels(raw) -> list[str]:
    """Return 19 channel names in our canonical 10-20 order.

    Siena names channels like 'EEG Fp1', 'EEG F3', sometimes bare 'Fp1'.
    T3/T4/T5/T6 may appear as T7/T8/P7/P8 (newer nomenclature)."""
    available = raw.ch_names
    upper_map = {ch.upper().strip(): ch for ch in available}
    # newer<->older 10-20 temporal aliases
    aliases = {
        "T3": ["T3", "T7"], "T4": ["T4", "T8"],
        "T5": ["T5", "P7"], "T6": ["T6", "P8"],
    }
    picked: dict[str, str] = {}
    for std in OUR_19CH_ORDER:
        names = aliases.get(std, [std])
        cands = []
        for nm in names:
            cands += [f"EEG {nm}", f"EEG {nm}-REF", f"EEG {nm}-LE", nm]
        matched = None
        for c in cands:
            if c in available:
                matched = c; break
            if c.upper() in upper_map:
                matched = upper_map[c.upper()]; break
        if matched is None:
            raise ValueError(
                f"Cannot find channel {std} (tried {cands[:4]}...) "
                f"in {available[:12]}")
        picked[std] = matched
    return [picked[std] for std in OUR_19CH_ORDER]


# ── Seizure annotation parsing ──────────────────────────────────────────────
_CLOCK_RE = re.compile(r"(\d{1,2})[.:](\d{2})[.:](\d{2})")


def _clock_to_sec(s: str) -> Optional[int]:
    m = _CLOCK_RE.search(s)
    if not m:
        return None
    h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mm * 60 + ss


def parse_seizure_list(txt_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Parse Seizures-list-PNxx.txt → {edf_filename: [(start_s, end_s), ...]}.

    Offsets are seconds from that registration's start. Siena gives clock
    times (HH.MM.SS); we compute seizure_start - registration_start, handling
    midnight wraparound. Defensive to blank lines / casing / '.' or ':' sep.
    """
    seizures: dict[str, list[tuple[int, int]]] = {}
    if not txt_path.exists():
        return seizures
    cur_file = None
    reg_start = None
    sz_start = None
    for raw_line in txt_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        low = line.lower()
        if low.startswith("file name") or low.startswith("registration file"):
            m = re.search(r"(PN\d+[-_]?\d*\.edf)", line, re.IGNORECASE)
            cur_file = m.group(1) if m else None
            reg_start = sz_start = None
        elif low.startswith("registration start"):
            reg_start = _clock_to_sec(line)
        elif low.startswith("seizure start") or (
                low.startswith("start time") and "registration" not in low):
            sz_start = _clock_to_sec(line)
        elif low.startswith("seizure end") or (
                low.startswith("end time") and "registration" not in low):
            sz_end = _clock_to_sec(line)
            if cur_file and reg_start is not None and \
               sz_start is not None and sz_end is not None:
                a = sz_start - reg_start
                b = sz_end - reg_start
                if a < 0:  # crossed midnight
                    a += 24 * 3600
                    b += 24 * 3600
                if b < a:
                    b += 24 * 3600
                seizures.setdefault(cur_file, []).append((int(a), int(b)))
            sz_start = None
    return seizures


def _list_subjects(data_dir) -> dict[str, dict]:
    """Return {PNxx: {'edfs': [Path,...], 'seizures': {edf_name:[(s,e)]}}}."""
    data_dir = Path(data_dir)
    out: dict[str, dict] = {}
    # subject folders PNxx (fallback: flat dir of PNxx-*.edf)
    subj_dirs = [p for p in sorted(data_dir.glob("PN*")) if p.is_dir()]
    if not subj_dirs:
        subj_dirs = [data_dir]  # flat layout
    for d in subj_dirs:
        edfs = sorted(d.glob("PN*.edf")) or sorted(d.glob("*.edf"))
        if not edfs:
            continue
        # subject id = PNxx prefix of the folder or first edf
        sid = d.name if d.name.upper().startswith("PN") else \
            re.match(r"(PN\d+)", edfs[0].name, re.IGNORECASE).group(1)
        sid = sid.upper()
        txt = next(iter(d.glob(f"Seizures-list-{sid}*.txt")), None) or \
            next(iter(d.glob("Seizures-list*.txt")), None)
        seiz = parse_seizure_list(txt) if txt else {}
        out.setdefault(sid, {"edfs": [], "seizures": {}})
        out[sid]["edfs"].extend(edfs)
        out[sid]["seizures"].update(seiz)
    return out


def make_subject_split(data_dir: str, train_frac: float = 0.6,
                       val_frac: float = 0.2, seed: int = 42) -> dict:
    """Deterministic subject-disjoint split over PNxx ids."""
    subs = sorted(_list_subjects(data_dir).keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_tr = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": subs[:n_tr],
        "val":   subs[n_tr:n_tr + n_val],
        "test":  subs[n_tr + n_val:],
    }


class SienaDataset(Dataset):
    """Siena scalp-EEG seizure detection, 2-class (interictal / ictal).

    Returns (trial [T,19] float32, label int). self.subject_ids tracks the
    integer index of the source PNxx patient (for recording-level metrics).
    """

    def __init__(
        self,
        data_dir: str,
        subjects: list[str],            # ["PN00", "PN01", ...]
        sample_rate: int = 256,
        trial_duration_s: int = 10,
        normalization: str = "per_recording_robust",
        negative_per_positive: float = 5.0,  # cap negatives at N× positives/subj; 0=keep all
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")
        assert normalization in ("per_trial_zscore", "per_recording_robust")
        self.sample_rate = sample_rate
        self.trial_samples = sample_rate * trial_duration_s
        self.trial_duration_s = trial_duration_s
        self.normalization = normalization
        self.negative_per_positive = negative_per_positive
        self.trials: list[np.ndarray] = []
        self.labels: list[int] = []
        self.subject_ids: list[int] = []

        data_dir = Path(data_dir)
        subjects = [s.upper() for s in subjects]

        cache_path = None
        if cache_dir:
            key = _cache_key({
                "kind": "siena",
                "data_dir": str(data_dir),
                "subjects": ",".join(sorted(subjects)),
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "normalization": normalization,
                "neg_per_pos": negative_per_positive,
                "seed": seed,
            })
            cache_path = Path(cache_dir) / f"siena_{key}.pt"
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

        all_subs = _list_subjects(data_dir)
        rng = np.random.RandomState(seed)
        print(f"  [Siena] Loading {sorted(subjects)} from {data_dir}")
        for si, sid in enumerate(sorted(subjects)):
            if sid not in all_subs:
                print(f"    WARN: {sid} not found on disk, skipping")
                continue
            info = all_subs[sid]
            pos, neg = self._load_subject(info, si)
            self._balance_and_store(pos, neg, si, rng)

        print(f"  [Siena] Loaded {len(self.trials)} trials from "
              f"{len(set(self.subject_ids))} subjects")
        self._print_class_dist()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"trials": self.trials, "labels": self.labels,
                        "subject_ids": self.subject_ids}, cache_path)
            print(f"  ↑ saved cache: {cache_path.name}")

    def _load_subject(self, info: dict, si: int):
        """Return (pos_trials, neg_trials) lists of [T,19] arrays."""
        pos, neg = [], []
        for edf in sorted(info["edfs"]):
            try:
                sig = self._read_edf(edf)               # (T,19)
            except Exception as e:
                print(f"    skip {edf.name}: {type(e).__name__}: {e}")
                continue
            intervals = info["seizures"].get(edf.name, [])
            # convert seizure seconds → sample ranges
            sz_ranges = [(int(a * self.sample_rate), int(b * self.sample_rate))
                         for (a, b) in intervals]
            n_trials = sig.shape[0] // self.trial_samples
            for i in range(n_trials):
                s = i * self.trial_samples
                e = s + self.trial_samples
                trial = sig[s:e]
                if self.normalization == "per_trial_zscore":
                    trial = (trial - trial.mean(0)) / (trial.std(0) + 1e-6)
                # overlap test: window [s,e) vs any seizure [a,b)
                label = 0
                for (a, b) in sz_ranges:
                    if s < b and a < e:
                        label = 1; break
                (pos if label == 1 else neg).append(trial.astype(np.float32))
        return pos, neg

    def _balance_and_store(self, pos, neg, si, rng):
        if self.negative_per_positive and self.negative_per_positive > 0 and pos:
            cap = int(len(pos) * self.negative_per_positive)
            if len(neg) > cap:
                idx = rng.choice(len(neg), size=cap, replace=False)
                neg = [neg[k] for k in idx]
        for t in pos:
            self.trials.append(t); self.labels.append(1)
            self.subject_ids.append(int(si))
        for t in neg:
            self.trials.append(t); self.labels.append(0)
            self.subject_ids.append(int(si))

    def _read_edf(self, edf: Path) -> np.ndarray:
        raw = read_raw_edf(str(edf), preload=True, verbose=False)
        ch = _pick_1020_channels(raw)
        raw.pick(ch)
        if int(round(raw.info["sfreq"])) != self.sample_rate:
            raw = raw.resample(self.sample_rate, verbose=False)
        sig = raw.get_data().T.astype(np.float32)       # (T,19)
        if self.normalization == "per_recording_robust":
            sig = _robust_scale_per_recording(sig)
        return sig

    def _print_class_dist(self):
        if not self.labels:
            print("    Class dist: (empty)"); return
        arr = np.asarray(self.labels)
        print(f"    Class dist: Interictal={int((arr==0).sum())}  "
              f"Ictal={int((arr==1).sum())}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), int(self.labels[idx])


def dataset_siena_inspect(data_dir: str):
    """Print discovered subjects + parsed seizures — run to VALIDATE the
    annotation parser against real files before trusting labels."""
    subs = _list_subjects(data_dir)
    print(f"Found {len(subs)} subjects under {data_dir}")
    for sid, info in sorted(subs.items()):
        n_edf = len(info["edfs"])
        n_sz = sum(len(v) for v in info["seizures"].values())
        print(f"  {sid}: {n_edf} edf, {n_sz} seizures parsed "
              f"across {len(info['seizures'])} files")
        for f, ivs in info["seizures"].items():
            print(f"      {f}: {ivs}")


if __name__ == "__main__":
    import sys
    dataset_siena_inspect(sys.argv[1] if len(sys.argv) > 1
                          else "/home/pxieaf/home2/datasets/siena-scalp-eeg")
