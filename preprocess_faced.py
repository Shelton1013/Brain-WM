"""FACED raw BIDS (NEMAR nm000112) -> official-style Processed_data arrays.

The official FACED `Processed_data` (Synapse syn50614194, used by CBraMod and
CSBrain) is not reachable from our region, so we rebuild it from the raw BDF
following the FACED paper (Chen et al., Sci. Data 2023): for each of the 28
video clips keep the LAST 30 s, 0.05-47 Hz band-pass, common-average reference,
250 Hz, with bad channels (electrode pops / dead leads, detected inside the
kept clips) replaced by spherical-spline interpolation BEFORE re-referencing so
they cannot leak into every channel through the average. Difference from the
official release: no ICA (not reproducible from the paper) — stated in the paper.

Output: <out_dir>/sub-XXX.npy, float32 [28 videos, 32 ch, 7500] in µV, videos
ordered by video_index 1..28 (the order CBraMod's label table assumes), plus
channels.json with the channel order.

    python preprocess_faced.py --raw_dir ~/home2/datasets/FACED_raw \
        --out_dir ~/home2/datasets/FACED_proc --workers 16
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SFREQ_OUT = 250
CLIP_S = 30
N_VIDEOS = 28
# bad channel: peak |x| > BAD_PEAK x median channel peak, or std > BAD_STD x
# median channel std (inside the kept clips). Loose on purpose: blink-dominated
# Fp1/Fp2 sit at ~5x peak and must stay.
BAD_PEAK, BAD_STD = 20.0, 8.0

# FACED was recorded on two montages: 61 subjects name the temporal leads
# T3/T4/T5/T6 and carry ear references A1/A2; 62 subjects name them T7/T8/P7/P8
# and carry horizontal EOG HEOR/HEOL instead. Both are laid out in one 32-slot
# order: 30 scalp channels + 2 auxiliary (non-scalp) slots, kept out of bad-
# channel handling and the average reference.
RENAME = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
AUX = {"A1": "AUX1", "A2": "AUX2", "HEOR": "AUX1", "HEOL": "AUX2"}
CANON32 = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
           "Cz", "C3", "C4", "T3", "T4", "AUX1", "AUX2", "CP1", "CP2", "CP5", "CP6",
           "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2"]


def _load_uv(bdf: Path):
    import mne
    raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose=False)
    raw.rename_channels({c: RENAME.get(c, AUX.get(c, c)) for c in raw.ch_names})
    raw.pick(CANON32)                                  # also fixes the order
    raw.set_channel_types({"AUX1": "misc", "AUX2": "misc"})
    data = raw.get_data(picks="all")                   # volts per BDF header
    # Some recordings store µV values under a V unit (see FACED README): a
    # scalp-EEG std is ~1e-5 V, so values ~1e6 larger mean the unit is µV.
    scale = 1.0 if np.median(np.std(data, axis=1)) > 1e-2 else 1e6
    raw._data = data * scale                           # now in µV
    return raw


def _find_bad_channels(raw, ev):
    """Scalp channels whose peak or std inside the kept clips is extreme."""
    sf = raw.info["sfreq"]
    eeg = [c for c in raw.ch_names if not c.startswith("AUX")]
    x = raw.get_data(picks=eeg)
    keep = np.zeros(x.shape[1], bool)
    for _, r in ev.iterrows():
        end = int(round((float(r["onset"]) + float(r["duration"])) * sf))
        keep[max(0, end - int(CLIP_S * sf)):end] = True
    xc = x[:, keep]
    peak, sd = np.abs(xc).max(axis=1), xc.std(axis=1)
    bad = (peak > BAD_PEAK * np.median(peak)) | (sd > BAD_STD * np.median(sd))
    return [eeg[i] for i in np.flatnonzero(bad)]


def process_subject(args):
    bdf, out_dir = args
    sub = bdf.name.split("_")[0]
    out = Path(out_dir) / f"{sub}.npy"
    if out.exists():
        return sub, "exists", None
    try:
        raw = _load_uv(bdf)
        raw.filter(0.05, 47.0, picks="all", verbose=False)
        ev = pd.read_csv(str(bdf).replace("_eeg.bdf", "_events.tsv"), sep="\t")
        ev = ev[pd.to_numeric(ev["video_index"], errors="coerce").notna()]
        ev = ev.assign(video_index=ev["video_index"].astype(int))
        bads = _find_bad_channels(raw, ev)
        if bads:
            raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
            raw.info["bads"] = bads
            raw.interpolate_bads(reset_bads=True, verbose=False)
        raw.set_eeg_reference("average", verbose=False)   # scalp channels only
        raw.resample(SFREQ_OUT, verbose=False)
        x = raw.get_data(picks=CANON32)                  # [32, T] µV
        clips = np.zeros((N_VIDEOS, x.shape[0], CLIP_S * SFREQ_OUT), np.float32)
        seen = set()
        for _, r in ev.iterrows():
            v = int(r["video_index"])
            end = int(round((float(r["onset"]) + float(r["duration"])) * SFREQ_OUT))
            start = end - CLIP_S * SFREQ_OUT
            if not 1 <= v <= N_VIDEOS or start < 0 or end > x.shape[1] or v in seen:
                continue
            clips[v - 1] = x[:, start:end]
            seen.add(v)
        missing = sorted(set(range(1, N_VIDEOS + 1)) - seen)
        if missing:
            return sub, f"missing videos {missing}", None
        np.save(out, clips)
        return sub, f"ok bads={bads}", CANON32
    except Exception as e:                               # keep going, report
        return sub, f"error {type(e).__name__}: {e}", None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    out_dir = Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    bdfs = sorted(Path(a.raw_dir).expanduser().glob("sub-*/eeg/*_eeg.bdf"))
    print(f"{len(bdfs)} subjects -> {out_dir}", flush=True)
    chans = None
    with Pool(a.workers) as pool:
        for sub, status, ch in pool.imap_unordered(
                process_subject, [(b, out_dir) for b in bdfs]):
            print(f"  {sub}: {status}", flush=True)
            if ch and chans is None:
                chans = ch
    if chans:
        (out_dir / "channels.json").write_text(json.dumps(chans))
    n_ok = len(list(out_dir.glob("sub-*.npy")))
    print(f"done: {n_ok}/{len(bdfs)} subjects written", flush=True)


if __name__ == "__main__":
    main()
