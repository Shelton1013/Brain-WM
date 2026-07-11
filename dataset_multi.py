"""
Multi-dataset EEG loader for large-scale JEPA pretraining.

Supports:
  1. PhysioNet MI                                    type='physionet'
  2. MOABB datasets, auto-download                   type='moabb', name=<ClassName>
       Common adds: Cho2017, Lee2019_MI, BNCI2014001 (BCIC-IV-2a),
       BNCI2014004 (BCIC-IV-2b), Schirrmeister2017 (HGD), Weibo2014, Shin2017A
  3. Healthy Brain Network (.set), grouped by sub-*   type='hbn'
  4. TUH EEG Corpus (TUEG / TUAB / TUEV / TUSZ)       type='tueg'
       Subject grouped by filename prefix before '_' (00000003_s002_t000.edf → 00000003)
  5. CHB-MIT pediatric scalp EEG (PhysioNet)          type='chb_mit'
       Subject grouped by chbXX prefix (chb01_03.edf → chb01)
  6. Siena Scalp EEG (PhysioNet)                      type='siena'
       Subject grouped by PNXX prefix (PN00-1.edf → PN00)
  7. HMC Sleep PSG (PhysioNet)                        type='hmc'
       One file per subject; min_channels relaxed to 4 (PSG has few EEG chans)
  8. CAP Sleep Database (PhysioNet)                   type='cap'
       One file per subject; min_channels relaxed to 4
  9. Generic EDF directory (legacy)                   type='edf_dir'
       Each file treated as a separate 'subject' (use only for unstructured dumps)

Not yet supported (need custom loaders):
  - Sleep-EDF / Sleep-EDFx: bipolar derivations (Fpz-Cz, Pz-Oz) don't match
    the standard 10-20 monopolar COMMON_CHANNELS. Extend COMMON_CHANNELS or
    add a bipolar-aware loader to enable.
  - SEED / SEED-IV / SEED-V: MATLAB .mat format, needs dedicated reader.

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
          {"type": "moabb", "name": "Schirrmeister2017"},     # HGD
          {"type": "moabb", "name": "BNCI2014001"},           # BCIC-IV-2a
          {"type": "tueg",   "path": "/data/tuh_eeg/edf/"},
          {"type": "chb_mit","path": "/data/chb-mit/"},
          {"type": "siena",  "path": "/data/siena-scalp-eeg/"},
          {"type": "hbn",    "path": "/data/hbn/eeg/"},
      ],
      sample_rate=256,
      trial_duration_s=4,
  )
"""

import hashlib
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
from dataset import euclidean_alignment, PhysioNetMIDataset


# ============================================================
# Cache helpers (avoid 10+ hour upfront preprocessing every run)
# ============================================================

def _cache_key(config: dict) -> str:
    """Stable 16-char hash of a preprocessing config dict."""
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _try_load_cache(cache_path: Path):
    """Load (trials, subject_ids, electrode_names, n_subjects) or None on miss."""
    if not cache_path.exists():
        return None
    print(f"  ↻ loading cache: {cache_path.name}")
    t0 = time.time()
    cached = torch.load(str(cache_path), weights_only=False)
    elapsed = time.time() - t0
    n_trials = len(cached.get("trials", []))
    n_subj = cached.get("n_subjects", "?")
    ch = cached.get("electrode_names") or []
    print(f"    ← {n_trials} trials, {n_subj} subjects, {len(ch)} channels "
          f"(cached, loaded in {elapsed:.1f}s)")
    return cached


def _save_cache(cache_path: Path, payload: dict):
    """Persist preprocessed dataset payload to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↑ saving cache: {cache_path.name}")
    t0 = time.time()
    torch.save(payload, str(cache_path))
    size_mb = cache_path.stat().st_size / 1024**2
    print(f"    saved {size_mb:.1f} MB in {time.time() - t0:.1f}s")

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# ============================================================
# Normalization (Défossez 2022-style robust scaling)
# ============================================================

def robust_scale_per_recording(eeg: np.ndarray) -> np.ndarray:
    """Per-recording robust scaling: (x - median) / (IQR / 1.349 + eps).

    Computes median and IQR (75th - 25th percentile) over the time
    dimension per channel, then normalizes. Preserves long-range
    amplitude structure (unlike per-trial z-score). 1.349 = Phi^-1(0.75)
    - Phi^-1(0.25), making IQR/1.349 a robust std estimator for Gaussian.

    Args:
        eeg: [T, C] float array (one full recording).

    Returns:
        [T, C] float32 scaled.
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    median = np.median(eeg, axis=0, keepdims=True)
    q75 = np.percentile(eeg, 75, axis=0, keepdims=True)
    q25 = np.percentile(eeg, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    robust_std = iqr / 1.349 + 1e-6
    return ((eeg - median) / robust_std).astype(np.float32)


def normalize_signal(eeg: np.ndarray, mode: str) -> np.ndarray:
    """Apply normalization to a full recording before segmentation.

    Args:
        eeg: [T, C] full recording.
        mode: "per_recording_robust" (Laya/Défossez style) or
              "none" (no recording-level normalization; per-trial
              z-score is applied later during segmentation).
    """
    if mode == "per_recording_robust":
        return robust_scale_per_recording(eeg)
    elif mode == "none":
        return eeg
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


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


# Case-insensitive alias lookup table (keys all uppercase). TUEG uses
# "EEG FP1-REF" (uppercase P) while our alias dict was written with
# "EEG Fp1" — without this we missed Fp1, Fp2, Fz, Cz, Pz on every TUEG
# recording, dropping match count below the 19-channel threshold.
_CHANNEL_ALIASES_UPPER = {k.upper(): v for k, v in CHANNEL_ALIASES.items()}


def normalize_channel_name(name: str) -> str:
    """Map various channel naming conventions to standard 10-20."""
    # Strip common suffixes and trailing dots (PhysioNet uses "C3..", "Fp1." etc.)
    clean = name.strip().rstrip(".")
    for suffix in ["-REF", "-LE", "-AR", "-AVG"]:
        if clean.upper().endswith(suffix):
            clean = clean[:-len(suffix)].strip()

    # Case-insensitive alias lookup
    if clean.upper() in _CHANNEL_ALIASES_UPPER:
        return _CHANNEL_ALIASES_UPPER[clean.upper()]

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
# Subject-id extractors for grouped EDF loading
# (Used so per-subject Euclidean Alignment works correctly when one subject
#  spans multiple .edf files, e.g. CHB-MIT, TUH, Siena.)
# ============================================================

def _sid_before_underscore(path):
    """Prefix before first underscore.
    TUH:     '00000003_s002_t000.edf' → '00000003'
    CHB-MIT: 'chb01_03.edf'           → 'chb01'
    """
    return path.stem.split("_")[0]


def _sid_before_dash(path):
    """Prefix before first dash.
    Siena: 'PN00-1.edf' → 'PN00'
    """
    return path.stem.split("-")[0]


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
        cache_dir: str = None,
    ):
        try:
            import moabb
            import moabb.datasets
        except ImportError:
            raise ImportError("moabb required: pip install moabb")

        # Set MOABB/MNE download path BEFORE creating dataset.
        # In DDP jobs, only rank 0 writes to the shared mne-python.json to
        # avoid concurrent writes corrupting the config (each other rank
        # would race to also write the same value, but the JSON file
        # can end up half-written). Other ranks wait via barrier, then
        # only set the env var (no disk write).
        if data_dir:
            import os
            import mne
            moabb_path = os.path.join(data_dir, "moabb")
            os.makedirs(moabb_path, exist_ok=True)

            # Detect DDP rank
            rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

            if rank == 0:
                # Persistent config write (only rank 0)
                mne.set_config("MNE_DATA", moabb_path, set_env=True)
                try:
                    moabb.utils.set_download_dir(moabb_path)
                except Exception as e:
                    print(f"  [MOABB] set_download_dir failed on rank 0: {e}")

            # All ranks: wait for rank 0 to finish, then set env var
            # (in-process, no file write).
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            except Exception:
                pass
            os.environ["MNE_DATA"] = moabb_path

        self.trial_samples = sample_rate * trial_duration_s
        self.trials = []
        self.subject_ids = []
        self.electrode_names = None

        # ── Cache check (avoid re-processing MOABB raw files on every run) ──
        cache_path = None
        if cache_dir:
            key = _cache_key({
                "kind": "moabb",
                "dataset_name": dataset_name,
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "use_ea": use_ea,
                "min_channels": min_channels,
            })
            cache_path = Path(cache_dir) / f"moabb_{dataset_name}_{key}.pt"
            cached = _try_load_cache(cache_path)
            if cached is not None:
                self.trials = cached["trials"]
                self.subject_ids = cached["subject_ids"]
                self.electrode_names = cached["electrode_names"]
                self.n_subjects = cached["n_subjects"]
                return

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
                        # Skip recordings too short for 0.1 Hz highpass
                        if raw.n_times < int(40 * sample_rate):
                            continue
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

        # ── Save cache for next run ──
        if cache_path is not None:
            _save_cache(cache_path, {
                "trials": self.trials,
                "subject_ids": self.subject_ids,
                "electrode_names": self.electrode_names,
                "n_subjects": self.n_subjects,
            })

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), self.subject_ids[idx]


class HBNDataset(Dataset):
    """Load HBN EEG data from downloaded .set (EEGLAB) files.

    HBN uses 128-channel EGI system at 500Hz. We map to common 19ch subset.
    Each .set file is one recording session (resting state, movie watching, etc.).
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        use_ea: bool = True,
        max_subjects: int = None,
        min_channels: int = 19,
        cache_dir: str = None,
    ):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")

        self.trial_samples = sample_rate * trial_duration_s
        self.trials = []
        self.subject_ids = []
        self.electrode_names = None

        # ── Cache check (avoid re-preprocessing on every run) ──
        cache_path = None
        if cache_dir:
            key = _cache_key({
                "kind": "hbn",
                "data_dir": str(data_dir),
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "use_ea": use_ea,
                "max_subjects": max_subjects,
                "min_channels": min_channels,
            })
            cache_path = Path(cache_dir) / f"hbn_{key}.pt"
            cached = _try_load_cache(cache_path)
            if cached is not None:
                self.trials = cached["trials"]
                self.subject_ids = cached["subject_ids"]
                self.electrode_names = cached["electrode_names"]
                self.n_subjects = cached["n_subjects"]
                return

        # Find all .set files, grouped by subject
        from collections import defaultdict
        subject_files = defaultdict(list)
        for f in sorted(Path(data_dir).rglob("*.set")):
            # Extract subject ID from path (e.g., sub-NDARAB514MAJ)
            parts = f.parts
            sub_id = None
            for part in parts:
                if part.startswith("sub-"):
                    sub_id = part
                    break
            if sub_id:
                subject_files[sub_id].append(f)

        subjects = sorted(subject_files.keys())
        if max_subjects:
            subjects = subjects[:max_subjects]

        print(f"  Loading HBN: {data_dir} ({len(subjects)} subjects)...")

        for subj_idx, sub_id in enumerate(subjects):
            try:
                subj_recordings = []
                for set_file in subject_files[sub_id]:
                    try:
                        raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
                        if raw.info["sfreq"] != sample_rate:
                            raw.resample(sample_rate, verbose=False)
                        # Skip recordings too short for 0.1 Hz highpass
                        if raw.n_times < int(40 * sample_rate):
                            continue
                        raw.filter(0.1, 75.0, verbose=False)

                        ch_indices, ch_names = pick_common_channels(raw.ch_names)
                        if len(ch_indices) < min_channels:
                            continue
                        if self.electrode_names is None:
                            self.electrode_names = ch_names

                        eeg = raw.get_data()[ch_indices].T.astype(np.float32)
                        subj_recordings.append(eeg)
                    except Exception:
                        continue

                if not subj_recordings:
                    continue

                subj_data = np.concatenate(subj_recordings, axis=0)
                if use_ea:
                    subj_data = euclidean_alignment(subj_data)

                n_trials = len(subj_data) // self.trial_samples
                for t in range(n_trials):
                    start = t * self.trial_samples
                    trial = subj_data[start:start + self.trial_samples]
                    mean = trial.mean(axis=0, keepdims=True)
                    std = trial.std(axis=0, keepdims=True) + 1e-8
                    self.trials.append(((trial - mean) / std))
                    self.subject_ids.append(subj_idx)

            except Exception as e:
                if subj_idx < 3:
                    print(f"    Skipping {sub_id}: {e}")

            if (subj_idx + 1) % 50 == 0:
                print(f"    ... {subj_idx+1}/{len(subjects)} subjects, "
                      f"{len(self.trials)} trials")

        self.n_subjects = len(set(self.subject_ids))
        print(f"    → {len(self.trials)} trials, {self.n_subjects} subjects, "
              f"{len(self.electrode_names or [])} channels")

        # Persist to cache for next run
        if cache_path is not None:
            _save_cache(cache_path, {
                "trials": self.trials,
                "subject_ids": self.subject_ids,
                "electrode_names": self.electrode_names,
                "n_subjects": self.n_subjects,
            })

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), self.subject_ids[idx]


class EDFDirectoryDataset(Dataset):
    """Load EEG from a directory of .edf files (e.g., TUH EEG Corpus).

    Recursively scans for .edf files, picks the common 10-20 subset, applies
    Euclidean Alignment, and segments into 4s trials.

    Two subject-counting modes:
      - subject_id_fn=None (default, legacy): each .edf file is one "subject".
        Use for unstructured EDF dumps where one file = one recording.
      - subject_id_fn=callable(Path)->str: group files by the returned id,
        apply EA on the concatenated per-subject data, and count distinct
        subjects. Required for datasets where one subject spans multiple files
        (TUH, CHB-MIT, Siena, etc.).
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 256,
        trial_duration_s: int = 4,
        use_ea: bool = True,
        max_files: int = None,
        min_channels: int = 19,
        subject_id_fn=None,
        cache_dir: str = None,
        cache_tag: str = "edf",
        exclude_patient_ids: set | None = None,
        normalization: str = "per_trial_zscore",
        # CBraMod-style quality filter args (cache-key only; actual filtering
        # is done by prebuild_tueg_cache.py — these args must match prebuild's
        # values for the cache to hit).
        drop_short_recording_min: float = 0.0,
        trim_start_end_sec: int = 0,
        notch_freq: float = 0.0,
        reject_abs_uv: float = 0.0,
    ):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")

        self.trial_samples = sample_rate * trial_duration_s
        self.trials = []
        self.subject_ids = []
        self.electrode_names = None
        self.normalization = normalization
        # Normalize exclude set; use frozenset for hash stability in cache key.
        exclude_set = frozenset(exclude_patient_ids or [])

        # ── Cache check ──
        cache_path = None
        if cache_dir:
            # Hash the exclude set into the cache key so different exclusion
            # lists produce different cache files (no silent stale-cache reuse).
            exclude_hash = hashlib.md5(
                ",".join(sorted(exclude_set)).encode()
            ).hexdigest()[:8] if exclude_set else "none"
            # Filter-key payload identical to prebuild_tueg_cache.py: only
            # append filter keys when non-default so legacy hashes are
            # back-compat.
            payload = {
                "kind": cache_tag,
                "data_dir": str(data_dir),
                "sample_rate": sample_rate,
                "trial_duration_s": trial_duration_s,
                "use_ea": use_ea,
                "max_files": max_files,
                "min_channels": min_channels,
                "grouped": subject_id_fn is not None,
                "exclude": exclude_hash,
                "n_excluded": len(exclude_set),
                "normalization": normalization,
            }
            if drop_short_recording_min > 0:
                payload["drop_short_min"] = drop_short_recording_min
            if trim_start_end_sec > 0:
                payload["trim_sec"] = trim_start_end_sec
            if notch_freq > 0:
                payload["notch_hz"] = notch_freq
            if reject_abs_uv > 0:
                payload["reject_uv"] = reject_abs_uv
            key = _cache_key(payload)
            cache_path = Path(cache_dir) / f"{cache_tag}_{key}.pt"
            cached = _try_load_cache(cache_path)
            if cached is not None:
                # Chunked manifest format from prebuild_tueg_cache.py:
                # iterate chunk files and concatenate. Each chunk is .npz with
                # (trials [N,T,C] float32, subject_ids [N] int64).
                if isinstance(cached, dict) and cached.get("format") == "chunked_npz_v1":
                    chunk_dir = cache_path.with_suffix("")
                    print(f"  Loading chunked cache: {len(cached['chunk_files'])} chunks "
                          f"from {chunk_dir}", flush=True)
                    self.trials = []
                    self.subject_ids = []
                    n_dropped_bad = 0
                    n_loaded = 0
                    for fname in cached["chunk_files"]:
                        # Prefer mmap-friendly side-car .npy if present
                        # (see convert_chunks_to_npy.py). Falls back to .npz.
                        stem = fname.rsplit(".", 1)[0]
                        trials_npy = chunk_dir / f"{stem}_trials.npy"
                        sids_npy = chunk_dir / f"{stem}_sids.npy"
                        if trials_npy.exists() and sids_npy.exists():
                            trials_arr = np.load(str(trials_npy), mmap_mode="r")
                            sids = np.load(str(sids_npy))
                        else:
                            npz = np.load(str(chunk_dir / fname))
                            trials_arr = npz["trials"]
                            sids = npz["subject_ids"]
                        # Vectorized NaN/Inf filter — Euclidean Alignment can
                        # produce non-finite values on ill-conditioned R
                        # matrices for some TUEG patients (overflow in
                        # fractional_matrix_power). Drop bad trials here
                        # rather than re-running prebuild.
                        # Block-wise finite check to bound peak RAM: with mmap
                        # trials_arr, np.isfinite over the WHOLE chunk would
                        # materialize a ~20 GB bool array per DDP rank (×8 = OOM).
                        # Materialize only a small block at a time for the check;
                        # the stored trials stay zero-copy mmap views.
                        n_rows = len(trials_arr)
                        if os.environ.get("SKIP_FINITE_CHECK") == "1":
                            # Skip the per-block isfinite READ (the NFS-heavy part
                            # that makes startup take hours on a huge cache). Store
                            # all rows as lazy mmap views — no data read here. Any
                            # rare NaN (EA ill-conditioning) is caught by the
                            # per-batch guard in train_one_epoch.
                            self.trials.extend(trials_arr)          # lazy views
                            self.subject_ids.extend(sids.tolist())
                            n_loaded += n_rows
                        else:
                            _BLK = 4096
                            n_good_chunk = 0
                            for b0 in range(0, n_rows, _BLK):
                                b1 = min(b0 + _BLK, n_rows)
                                block = np.asarray(trials_arr[b0:b1])  # RAM: BLK×T×C
                                fm = np.isfinite(block).all(axis=(1, 2))
                                for j in np.nonzero(fm)[0]:
                                    gi = b0 + int(j)
                                    # zero-copy mmap view (block freed next iter)
                                    self.trials.append(trials_arr[gi])
                                    self.subject_ids.append(int(sids[gi]))
                                n_good_chunk += int(fm.sum())
                            n_dropped_bad += n_rows - n_good_chunk
                            n_loaded += n_good_chunk
                    self.electrode_names = cached["electrode_names"]
                    self.n_subjects = cached["n_subjects"]
                    if n_dropped_bad:
                        print(f"    [chunked-cache] dropped {n_dropped_bad} "
                              f"trials with NaN/Inf values "
                              f"({100*n_dropped_bad/(n_loaded+n_dropped_bad):.2f}%)",
                              flush=True)
                    return
                # Legacy single-payload format
                self.trials = cached["trials"]
                self.subject_ids = cached["subject_ids"]
                self.electrode_names = cached["electrode_names"]
                self.n_subjects = cached["n_subjects"]
                return

        edf_files = sorted(Path(data_dir).rglob("*.edf"))

        # ── Patient exclusion (only meaningful when subject_id_fn is given;
        # otherwise we have no way to extract patient ids from filenames) ──
        if exclude_set and subject_id_fn is not None:
            before = len(edf_files)
            edf_files = [
                f for f in edf_files
                if subject_id_fn(f) not in exclude_set
            ]
            n_dropped = before - len(edf_files)
            print(f"  [exclude] dropped {n_dropped}/{before} EDFs "
                  f"({len(exclude_set)} excluded patient IDs)")
        elif exclude_set and subject_id_fn is None:
            print(f"  [exclude] WARN: exclude_patient_ids given but "
                  f"subject_id_fn is None — no filtering performed")

        if max_files:
            edf_files = edf_files[:max_files]

        def _load_one(edf_path):
            """Return [T, C] float32 array, or None on failure / channel-mismatch."""
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
                if raw.info["sfreq"] != sample_rate:
                    raw.resample(sample_rate, verbose=False)
                # 0.1 Hz highpass needs ~8449 samples (~33 s @ 256 Hz) of
                # filter length; signals shorter than this produce severely
                # distorted output. Skip them.
                min_samples_for_filter = int(40 * sample_rate)
                if raw.n_times < min_samples_for_filter:
                    return None
                raw.filter(0.1, 75.0, verbose=False)
                ch_indices, ch_names = pick_common_channels(raw.ch_names)
                if len(ch_indices) < min_channels:
                    return None
                if self.electrode_names is None:
                    self.electrode_names = ch_names
                return raw.get_data()[ch_indices].T.astype(np.float32)
            except Exception:
                return None

        def _segment_and_store(eeg, subj_idx):
            """Apply normalization at recording level OR per-trial, then segment."""
            # Recording-level normalization (Laya/Défossez style):
            # apply ONCE per concatenated recording before segmentation.
            # Preserves long-range amplitude / variance structure across trials.
            if self.normalization == "per_recording_robust":
                eeg = normalize_signal(eeg, "per_recording_robust")
            n_trials = len(eeg) // self.trial_samples
            for t in range(n_trials):
                start = t * self.trial_samples
                trial = eeg[start:start + self.trial_samples]
                if self.normalization == "per_trial_zscore":
                    # Legacy: per-trial z-score (4s window) — destroys long-range
                    mean = trial.mean(axis=0, keepdims=True)
                    std = trial.std(axis=0, keepdims=True) + 1e-8
                    trial = (trial - mean) / std
                # If per_recording_robust was applied above, trial is already scaled.
                self.trials.append(trial.astype(np.float32))
                self.subject_ids.append(subj_idx)

        if subject_id_fn is None:
            # Per-file mode (legacy): one "subject" per file, per-file EA.
            print(f"  Loading EDF dir: {data_dir} ({len(edf_files)} files, per-file, "
                  f"norm={self.normalization})...")
            for file_idx, edf_path in enumerate(edf_files):
                eeg = _load_one(edf_path)
                if eeg is None:
                    if file_idx < 5:
                        print(f"    Skipping {edf_path.name} (load failed / channel mismatch)")
                    continue
                if use_ea:
                    eeg = euclidean_alignment(eeg)
                _segment_and_store(eeg, file_idx)
                if file_idx % 100 == 0 and file_idx > 0:
                    print(f"    ... {file_idx}/{len(edf_files)} files, "
                          f"{len(self.trials)} trials so far")
        else:
            # Grouped mode: bucket files by subject_id_fn, per-subject EA.
            from collections import defaultdict
            subject_files = defaultdict(list)
            for f in edf_files:
                try:
                    sid = subject_id_fn(f)
                    if sid is not None:
                        subject_files[sid].append(f)
                except Exception:
                    pass
            subjects = sorted(subject_files.keys())
            print(f"  Loading EDF dir: {data_dir} "
                  f"({len(edf_files)} files, {len(subjects)} subjects, grouped, "
                  f"norm={self.normalization})...")
            for subj_idx, sid in enumerate(subjects):
                subj_recordings = [
                    eeg for eeg in (_load_one(p) for p in subject_files[sid])
                    if eeg is not None
                ]
                if not subj_recordings:
                    continue
                subj_data = np.concatenate(subj_recordings, axis=0)
                if use_ea:
                    subj_data = euclidean_alignment(subj_data)
                _segment_and_store(subj_data, subj_idx)
                if (subj_idx + 1) % 50 == 0:
                    print(f"    ... {subj_idx+1}/{len(subjects)} subjects, "
                          f"{len(self.trials)} trials so far")

        self.n_subjects = len(set(self.subject_ids))
        unit = "subjects" if subject_id_fn is not None else "recordings"
        print(f"    → {len(self.trials)} trials, {self.n_subjects} {unit}, "
              f"{len(self.electrode_names or [])} channels")

        # Persist to cache for next run
        if cache_path is not None:
            _save_cache(cache_path, {
                "trials": self.trials,
                "subject_ids": self.subject_ids,
                "electrode_names": self.electrode_names,
                "n_subjects": self.n_subjects,
            })

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
        cache_dir: str = None,
        normalization: str = "per_trial_zscore",
        # CBraMod-style quality-filter args propagated to EDF-based sources
        # (TUEG, CHB-MIT, etc.). Must match prebuild_tueg_cache.py values
        # for the cache to hit.
        drop_short_recording_min: float = 0.0,
        trim_start_end_sec: int = 0,
        notch_freq: float = 0.0,
        reject_abs_uv: float = 0.0,
    ):
        datasets = []
        total_subjects = 0
        self.normalization = normalization
        self._filter_kwargs = dict(
            drop_short_recording_min=drop_short_recording_min,
            trim_start_end_sec=trim_start_end_sec,
            notch_freq=notch_freq,
            reject_abs_uv=reject_abs_uv,
        )

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
                        ds.trials = [t[:, ch_indices].copy() for t in ds.trials]
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
                    cache_dir=cache_dir,
                )
                for i in range(len(ds.subject_ids)):
                    ds.subject_ids[i] += total_subjects
                total_subjects += ds.n_subjects
                datasets.append(ds)

            elif src_type == "hbn":
                ds = HBNDataset(
                    data_dir=src["path"],
                    sample_rate=sample_rate,
                    trial_duration_s=trial_duration_s,
                    max_subjects=src.get("max_subjects", None),
                    cache_dir=cache_dir,
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
                    cache_dir=cache_dir,
                    cache_tag="edf",
                    normalization=normalization,
                    **self._filter_kwargs,
                )
                for i in range(len(ds.subject_ids)):
                    ds.subject_ids[i] += total_subjects
                total_subjects += ds.n_subjects
                datasets.append(ds)

            elif src_type in ("tueg", "chb_mit", "siena", "hmc", "cap"):
                # Convenience EDF loaders with per-dataset subject grouping
                # and sensible defaults. Override with src['min_channels'] /
                # src['max_files'] / src['subject_id_fn'] if needed.
                presets = {
                    "tueg":    dict(subject_id_fn=_sid_before_underscore, min_channels=19),
                    "chb_mit": dict(subject_id_fn=_sid_before_underscore, min_channels=19),
                    "siena":   dict(subject_id_fn=_sid_before_dash,       min_channels=19),
                    # Sleep PSG datasets usually have only a few EEG channels:
                    "hmc":     dict(subject_id_fn=None,                   min_channels=4),
                    "cap":     dict(subject_id_fn=None,                   min_channels=4),
                }[src_type]
                ds = EDFDirectoryDataset(
                    data_dir=src["path"],
                    sample_rate=sample_rate,
                    trial_duration_s=trial_duration_s,
                    max_files=src.get("max_files", None),
                    min_channels=src.get("min_channels", presets["min_channels"]),
                    subject_id_fn=src.get("subject_id_fn", presets["subject_id_fn"]),
                    cache_dir=cache_dir,
                    cache_tag=src_type,  # e.g., "tueg", "chb_mit", "siena"
                    exclude_patient_ids=src.get("exclude_patient_ids"),
                    normalization=normalization,
                    **self._filter_kwargs,
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
