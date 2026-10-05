"""CBraMod's downstream samples + splits as one npz per task (for linear probes).

Each builder follows the matching CBraMod `preprocessing/preprocessing_<task>.py`
line for line (file ordering, split slices, channels, filtering, resampling,
window length, labels); only the container changes (npz instead of LMDB), so
every model is probed on exactly the samples CBraMod is evaluated on.

Output <out_dir>/<task>.npz:
  X [N, C, T] float32 µV at 200 Hz,  y [N] int64,  rec [N] recording id,
  split [N] in {train, val, test},  ch [C] 10-20 channel names (upper case).

    python build_task_npz.py --task ma --out_dir ~/home2/datasets/lp_tasks
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

D = Path("/home/pxieaf/home2/datasets")


def _pack(rows, ch, out):
    X = np.stack([r[0] for r in rows]).astype(np.float32)
    tmp = Path(str(out) + ".tmp.npz")                          # atomic: job done = file exists
    np.savez(tmp, X=X, y=np.asarray([r[1] for r in rows], np.int64),
             rec=np.asarray([r[2] for r in rows]), split=np.asarray([r[3] for r in rows]),
             ch=np.asarray([c.upper() for c in ch]))
    tmp.replace(out)
    print(out, X.shape, {s: int((np.asarray([r[3] for r in rows]) == s).sum())
                         for s in ("train", "val", "test")}, flush=True)


def build_ma(out):
    """preprocessing_stress.py: sorted edf, 56/8/rest, 20 ch (incl. A2-A1),
    200 Hz, 5-s windows, label = file suffix (_1 rest -> 0, _2 task -> 1)."""
    import mne
    root = D / "mental_arithmetic/eeg-during-mental-arithmetic-tasks-1.0.0"
    files = sorted(f for f in os.listdir(root) if f.endswith(".edf"))
    parts = {"train": files[:56], "val": files[56:64], "test": files[64:]}
    sel = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4',
           'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2',
           'EEG Fz', 'EEG Cz', 'EEG Pz', 'EEG A2-A1']
    rows = []
    for sp, fl in parts.items():
        for f in fl:
            raw = mne.io.read_raw_edf(root / f, preload=True, verbose=False)
            raw.pick(sel); raw.reorder_channels(sel); raw.resample(200, verbose=False)
            eeg = raw.get_data(units="uV")
            a = eeg.shape[1] % 1000
            if a:
                eeg = eeg[:, :-a]
            eeg = eeg.reshape(20, -1, 1000).transpose(1, 0, 2)
            for s in eeg:
                rows.append((s, int(f[-5]) - 1, f[:-4], sp))
    _pack(rows, [c.replace("EEG ", "") for c in sel], out)


def build_mumtaz(out):
    """preprocessing_mumtaz.py: non-TASK edf split by 'MDD', sorted, H 40/8/rest,
    MDD 42/10/rest; 19 -LE ch, 200 Hz, 0.3-75 Hz, 50 Hz notch, 5-s windows."""
    import mne
    root = D / "mumtaz2016_full"
    allf = [f for f in os.listdir(root) if f.endswith(".edf") and "TASK" not in f]
    H = sorted(f for f in allf if "MDD" not in f)
    M = sorted(f for f in allf if "MDD" in f)
    parts = {"train": H[:40] + M[:42], "val": H[40:48] + M[42:52], "test": H[48:] + M[52:]}
    sel = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE',
           'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE',
           'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE']
    rows = []
    for sp, fl in parts.items():
        for f in fl:
            raw = mne.io.read_raw_edf(root / f, preload=True, verbose=False)
            raw.pick_channels(sel, ordered=True)
            raw.resample(200, verbose=False)
            raw.filter(l_freq=0.3, h_freq=75, verbose=False)
            raw.notch_filter(50, verbose=False)
            eeg = raw.to_data_frame().values[:, 1:]          # µV (mne default scaling)
            a = eeg.shape[0] % 1000
            if a:
                eeg = eeg[:-a]
            eeg = eeg.reshape(-1, 1000, eeg.shape[1]).transpose(0, 2, 1)
            for s in eeg:
                rows.append((s, int("MDD" in f), f[:-4], sp))
    _pack(rows, [c.replace("EEG ", "").replace("-LE", "") for c in sel], out)


def build_physio(out):
    """preprocessing_physio.py: subject dirs sorted 70/19/20, MI runs 04/06/08/10/
    12/14, 64 ch, average ref, 0.3 Hz HP, 60 Hz notch, 200 Hz, 4-s epochs from
    each cue, rest (T0) dropped; labels L/R fist (runs 4/8/12) and fists/feet."""
    import mne
    root = D / "physionet"
    root = next((p for p in (root, root / "files") if (p / "S001").exists()), root)
    files = sorted(f for f in os.listdir(root) if f.startswith("S") and (root / f).is_dir())
    parts = {"train": files[:70], "val": files[70:89], "test": files[89:109]}
    sel = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..',
           'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.',
           'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..',
           'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.',
           'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
           'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
    rows = []
    for sp, fl in parts.items():
        for f in fl:
            for task in ['04', '06', '08', '10', '12', '14']:
                raw = mne.io.read_raw_edf(root / f / f"{f}R{task}.edf", preload=True, verbose=False)
                raw.pick_channels(sel, ordered=True)
                if raw.info["bads"]:
                    raw.interpolate_bads()
                raw.set_eeg_reference(ref_channels="average", verbose=False)
                raw.filter(l_freq=0.3, h_freq=None, verbose=False)
                raw.notch_filter(60, verbose=False)
                raw.resample(200, verbose=False)
                ev, _ = mne.events_from_annotations(raw, verbose=False)
                ep = mne.Epochs(raw, ev, event_id=None, tmin=0, tmax=4. - 1. / raw.info["sfreq"],
                                baseline=None, preload=True, verbose=False)
                data = ep.get_data(units="uV")[:, :, -800:]
                for s, e in zip(data, ep.events[:, 2]):
                    if e != 1:
                        rows.append((s, int(e - 2 if task in ['04', '08', '12'] else e),
                                     f"{f}R{task}", sp))
    _pack(rows, [c.strip(".") for c in sel], out)


def build_faced(out):
    """preprocessing_faced.py on our rebuilt Processed_data (see build_faced_lmdb_
    cbramod.py): sorted subjects 80/20/23, 250->200 Hz FFT resample, 3 x 10 s."""
    from scipy import signal
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
    root = D / "FACED_proc"
    files = sorted(f for f in os.listdir(root) if f.startswith("sub-") and f.endswith(".npy"))
    parts = {"train": files[:80], "val": files[80:100], "test": files[100:]}
    ch = json.loads((root / "channels.json").read_text())
    rows = []
    for sp, fl in parts.items():
        for f in fl:
            eeg = signal.resample(np.load(root / f), 6000, axis=2)          # [28, 32, 6000]
            for i, lab in enumerate(labels):
                for j in range(3):
                    rows.append((eeg[i, :, j * 2000:(j + 1) * 2000], lab, f[:-4], sp))
    _pack(rows, ch, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["ma", "mumtaz", "physio", "faced"], required=True)
    p.add_argument("--out_dir", default=str(D / "lp_tasks"))
    a = p.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    globals()[f"build_{a.task}"](Path(a.out_dir) / f"{a.task}.npz")
