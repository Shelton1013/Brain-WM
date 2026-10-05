"""TUAB samples for linear probes, following CBraMod `preprocessing_tuab.py`.

CBraMod: per recording resample 200 Hz, 0.3-75 Hz, 60 Hz notch, 16 TCP bipolar
channels from the -REF leads, non-overlapping 10-s windows (2000 samples), label
abnormal = 1; split: train/val = first 80% / last 20% of the SORTED train-folder
subject ids (per class), test = the official eval folder. Files missing a lead
are skipped (as theirs).

We store, per 10-s window and in the SAME window order:
  mono200 [N, 19, 2000] float16 µV : CBraMod's filtered signal, 19 monopolar leads
           (CBraMod's 16 bipolar channels are differences of these; LaBraM is fed
           monopolar leads by name)
  ours256 [N, 2560, 19] float16    : our input contract on the same windows --
           0.5-45 Hz on the whole recording, 256 Hz, per-recording robust scaling
plus y / rec / split. TUAB is part of the TUEG pretraining corpus for every model
compared here (CBraMod's main TUAB result also pretrains on TUEG incl. TUAB).

    python build_tuab_lp.py --out_dir ~/home2/datasets/lp_tasks/tuab --workers 24
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("/home/pxieaf/home2/tuh/tuh_eeg_abnormal/v3.0.1/edf")
MONO = ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ", "C3", "C4", "CZ",
        "T3", "T4", "T5", "T6", "P3", "P4", "PZ", "O1", "O2"]      # = OUR_19CH_ORDER
# CBraMod's 16 TCP pairs, as indices into MONO
TCP = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("FP2", "F8"), ("F8", "T4"),
       ("T4", "T6"), ("T6", "O2"), ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
       ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")]
TCP_IDX = [(MONO.index(a), MONO.index(b)) for a, b in TCP]
WIN = 2000


def split_files():
    out = []
    for cls, lab in (("abnormal", 1), ("normal", 0)):
        d = ROOT / "train" / cls / "01_tcp_ar"
        subs = sorted({f.split("_")[0] for f in os.listdir(d)})
        cut = int(len(subs) * 0.8)
        part = {s: ("train" if i < cut else "val") for i, s in enumerate(subs)}
        out += [(d / f, lab, part[f.split("_")[0]]) for f in sorted(os.listdir(d)) if f.endswith(".edf")]
        e = ROOT / "eval" / cls / "01_tcp_ar"
        out += [(e / f, lab, "test") for f in sorted(os.listdir(e)) if f.endswith(".edf")]
    return out


def process(args):
    import mne
    from scipy.signal import butter, resample_poly, sosfiltfilt
    path, lab, sp, tmp = args
    o = Path(tmp) / (path.stem + ".npz")
    if o.exists():
        return str(o)
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.resample(200, verbose=False)
    raw.filter(l_freq=0.3, h_freq=75, verbose=False)
    raw.notch_filter(60, verbose=False)
    names = [c.upper() for c in raw.ch_names]
    try:
        idx = [names.index(f"EEG {c}-REF") for c in MONO]
    except ValueError:
        return None                                              # CBraMod skips these too
    x = raw.get_data(units="uV")[idx]                            # [19, T] µV @200 Hz
    n = x.shape[1] // WIN
    if n == 0:
        return None
    mono = x[:, :n * WIN].reshape(19, n, WIN).transpose(1, 0, 2)
    # ours: whole-recording 0.5-45 Hz, 256 Hz, robust scaling, same 10-s windows
    sos = butter(4, (0.5, 45.0), btype="band", fs=200, output="sos")
    y = resample_poly(sosfiltfilt(sos, x[:, :n * WIN], axis=1), 32, 25, axis=1)   # [19, n*2560]
    med = np.median(y, axis=1, keepdims=True)
    iqr = np.percentile(y, 75, axis=1, keepdims=True) - np.percentile(y, 25, axis=1, keepdims=True)
    y = (y - med) / (iqr / 1.349 + 1e-6)
    ours = y.reshape(19, n, 2560).transpose(1, 2, 0)
    np.savez(o, mono=mono.astype(np.float16), ours=ours.astype(np.float16), lab=lab, sp=sp,
             rec=path.stem)
    return str(o)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/home/pxieaf/home2/datasets/lp_tasks/tuab")
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    out = Path(a.out_dir); tmp = out / "per_rec"; tmp.mkdir(parents=True, exist_ok=True)
    files = split_files()
    print(f"{len(files)} recordings", flush=True)
    with Pool(a.workers) as pool:
        done = [r for r in pool.imap(process, [(f, l, s, tmp) for f, l, s in files], chunksize=4) if r]
    print(f"{len(done)} usable", flush=True)
    # concatenate (sorted = deterministic) into single arrays for fast full loads
    parts = [np.load(f) for f in sorted(done)]
    N = sum(len(q["mono"]) for q in parts)
    mono = np.lib.format.open_memmap(out / "mono200.tmp.npy", "w+", np.float16, (N, 19, WIN))
    ours = np.lib.format.open_memmap(out / "ours256.tmp.npy", "w+", np.float16, (N, 2560, 19))
    meta = {"y": [], "rec": [], "split": []}
    i = 0
    for q in parts:
        n = len(q["mono"])
        mono[i:i + n] = q["mono"]; ours[i:i + n] = q["ours"]
        meta["y"] += [int(q["lab"])] * n; meta["rec"] += [str(q["rec"])] * n
        meta["split"] += [str(q["sp"])] * n
        i += n
    mono.flush(); ours.flush(); del mono, ours
    np.savez(out / "meta.npz", y=np.asarray(meta["y"], np.int64), rec=np.asarray(meta["rec"]),
             split=np.asarray(meta["split"]), mono_ch=np.asarray(MONO), tcp=np.asarray(TCP_IDX))
    os.replace(out / "mono200.tmp.npy", out / "mono200.npy")
    os.replace(out / "ours256.tmp.npy", out / "ours256.npy")
    sp = np.asarray(meta["split"])
    print(json.dumps({"N": N, **{s: int((sp == s).sum()) for s in ("train", "val", "test")}}))
    (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
