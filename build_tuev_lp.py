"""TUEV samples for linear probes, following CBraMod `preprocessing_tuev.py`.

CBraMod: per recording resample 200 Hz, 0.3-75 Hz, 60 Hz notch, 16 TCP bipolar
channels from the -REF leads; one sample per annotated event in the .rec file:
the 1-s event plus 2 s on each side (5 s = 1000 samples), taken from the
recording concatenated three times (circular padding at the edges); label =
.rec class 1..6 -> 0..5 (SPSW, GPED, PLED, EYEM, ARTF, BCKG). Split: train/val =
first 80% / last 20% of the SORTED train-folder subject ids, test = eval folder.
Files missing one of the 16 TCP leads are skipped (as theirs).

Stored per sample, in the same order (same layout as build_tuab_lp.py, so
eval_linear_probe_tuab.py --task tuev probes it):
  mono200 [N, 19, 1000] float16 µV : CBraMod's signal, 19 monopolar leads (Fz/Cz/Pz
           are zero when absent -- CBraMod's 16 bipolar channels never use them)
  ours256 [N, 1280, 19] float16    : our input contract on the same 5-s windows --
           0.5-45 Hz on the whole recording, 256 Hz, per-recording robust scaling
TUEV recordings come from the TUH corpus; every compared model pretrains on TUEG.

    python build_tuev_lp.py --out_dir ~/home2/datasets/lp_tasks/tuev --workers 24
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from build_tuab_lp import MONO, TCP, TCP_IDX

ROOT = Path("/home/pxieaf/home2/tuh/tuh_eeg_events/v2.0.1/edf")
FS, OFS = 200, 256
WIN = 5 * FS
NEED = sorted({c for pair in TCP for c in pair})          # the 16 leads CBraMod needs


def split_files():
    train = sorted(ROOT.joinpath("train").rglob("*.edf"))
    subs = sorted({p.name.split("_")[0] for p in train})
    cut = int(len(subs) * 0.8)
    part = {s: ("train" if i < cut else "val") for i, s in enumerate(subs)}
    out = [(p, part[p.name.split("_")[0]]) for p in train]
    out += [(p, "test") for p in sorted(ROOT.joinpath("eval").rglob("*.edf"))]
    return out


def process(args):
    import mne
    from scipy.signal import butter, resample_poly, sosfiltfilt
    path, sp, tmp = args
    o = Path(tmp) / (path.stem + ".npz")
    if o.exists():
        return str(o)
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.resample(FS, verbose=False)
    raw.filter(l_freq=0.3, h_freq=75, verbose=False)
    raw.notch_filter(60, verbose=False)
    names = [c.upper() for c in raw.ch_names]
    if any(f"EEG {c}-REF" not in names for c in NEED):
        return None
    times = raw.times
    data = raw.get_data(units="uV")
    x = np.zeros((19, data.shape[1]))
    for i, c in enumerate(MONO):
        if f"EEG {c}-REF" in names:
            x[i] = data[names.index(f"EEG {c}-REF")]
    ev = np.genfromtxt(str(path)[:-3] + "rec", delimiter=",")
    ev = ev.reshape(-1, ev.shape[-1])
    T = x.shape[1]
    # ours: whole-recording 0.5-45 Hz, 256 Hz, robust scaling
    sos = butter(4, (0.5, 45.0), btype="band", fs=FS, output="sos")
    y = resample_poly(sosfiltfilt(sos, x, axis=1), OFS, FS, axis=1)
    med = np.median(y, axis=1, keepdims=True)
    iqr = np.percentile(y, 75, axis=1, keepdims=True) - np.percentile(y, 25, axis=1, keepdims=True)
    y = (y - med) / (iqr / 1.349 + 1e-6)
    x3, y3 = np.concatenate([x, x, x], 1), np.concatenate([y, y, y], 1)
    Ty = y.shape[1]
    mono, ours, lab = [], [], []
    for chan, t0, t1, cls in ev[:, :4]:
        s = int(np.where(times >= t0)[0][0])
        e = int(np.where(times >= t1)[0][0])
        seg = x3[:, T + s - 2 * FS: T + e + 2 * FS]
        if seg.shape[1] != WIN:                              # CBraMod assumes 1-s events
            continue
        so = int(round(s * OFS / FS))
        mono.append(seg)
        ours.append(y3[:, Ty + so - 2 * OFS: Ty + so - 2 * OFS + 5 * OFS].T)
        lab.append(int(cls) - 1)
    if not lab:
        return None
    np.savez(o, mono=np.stack(mono).astype(np.float16), ours=np.stack(ours).astype(np.float16),
             lab=np.asarray(lab), sp=sp, rec=path.stem)
    return str(o)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/home/pxieaf/home2/datasets/lp_tasks/tuev")
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    out = Path(a.out_dir); tmp = out / "per_rec"; tmp.mkdir(parents=True, exist_ok=True)
    files = split_files()
    print(f"{len(files)} recordings", flush=True)
    with Pool(a.workers) as pool:
        done = [r for r in pool.imap(process, [(f, s, tmp) for f, s in files], chunksize=2) if r]
    print(f"{len(done)} usable", flush=True)
    parts = sorted(done)
    N = 0
    for f in parts:
        with np.load(f) as q:
            N += q["mono"].shape[0]
    mono = np.lib.format.open_memmap(out / "mono200.tmp.npy", "w+", np.float16, (N, 19, WIN))
    ours = np.lib.format.open_memmap(out / "ours256.tmp.npy", "w+", np.float16, (N, 5 * OFS, 19))
    meta = {"y": [], "rec": [], "split": []}
    i = 0
    for f in parts:
        with np.load(f) as q:
            n = q["mono"].shape[0]
            mono[i:i + n] = q["mono"]; ours[i:i + n] = q["ours"]
            meta["y"] += [int(v) for v in q["lab"]]; meta["rec"] += [str(q["rec"])] * n
            meta["split"] += [str(q["sp"])] * n
        i += n
    mono.flush(); ours.flush(); del mono, ours
    np.savez(out / "meta.npz", y=np.asarray(meta["y"], np.int64), rec=np.asarray(meta["rec"]),
             split=np.asarray(meta["split"]), mono_ch=np.asarray(MONO), tcp=np.asarray(TCP_IDX))
    os.replace(out / "mono200.tmp.npy", out / "mono200.npy")
    os.replace(out / "ours256.tmp.npy", out / "ours256.npy")
    sp = np.asarray(meta["split"]); yy = np.asarray(meta["y"])
    print(json.dumps({"N": N, **{s: int((sp == s).sum()) for s in ("train", "val", "test")},
                      "classes": np.bincount(yy).tolist()}))
    (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
