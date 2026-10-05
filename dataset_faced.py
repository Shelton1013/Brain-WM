"""FACED emotion recognition (9 classes) for our model.

Reads the official-style arrays written by preprocess_faced.py
(sub-XXX.npy, [28 videos, 32 ch, 30 s x 250 Hz] µV) — the SAME arrays the
CBraMod baseline is built from — and adapts them to our input contract only:
the 19 canonical 10-20 channels (FACED names T3/T4/T5/T6 directly), 256 Hz,
per-recording robust scaling over the subject's 28 clips, 10-s windows
(3 per clip, as CBraMod).

Split = CBraMod/CSBrain: sorted subjects [:80] train, [80:100] val, [100:] test.
Labels by video index: anger, disgust, fear, sadness, neutral(4 clips),
amusement, inspiration, joy, tenderness.
Ref: CSBrain BA 0.5752 / CBraMod BA 0.5509 (on the official Processed_data).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from dataset_mumtaz import OUR_19CH_ORDER, _robust_scale_per_recording

LABEL_NAMES = ["anger", "disgust", "fear", "sadness", "neutral",
               "amusement", "inspiration", "joy", "tenderness"]
N_CLASSES = 9
VIDEO_LABELS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4,
                         5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8])
SRC_SR = 250
WIN_S = 10


def cbramod_split(proc_dir):
    subs = sorted(p.stem for p in Path(proc_dir).glob("sub-*.npy"))
    return {"train": subs[:80], "val": subs[80:100], "test": subs[100:]}


def load_xy(proc_dir, subjects, sample_rate=256):
    """-> X [N, T, 19] float32 (robust-scaled), y [N], subject ids [N]."""
    proc_dir = Path(proc_dir)
    chans = json.loads((proc_dir / "channels.json").read_text())
    idx = [chans.index(c) for c in OUR_19CH_ORDER]
    win = WIN_S * sample_rate
    X, y, sid = [], [], []
    for s in subjects:
        clips = np.load(proc_dir / f"{s}.npy")[:, idx, :]          # [28,19,7500]
        if sample_rate != SRC_SR:
            clips = resample_poly(clips, sample_rate, SRC_SR, axis=-1)
        # per-recording robust scale: statistics over all 28 clips of the subject
        flat = clips.transpose(0, 2, 1).reshape(-1, len(idx))        # [28*T,19]
        clips = _robust_scale_per_recording(flat).reshape(
            clips.shape[0], clips.shape[2], len(idx))                # [28,T,19]
        for v in range(clips.shape[0]):
            for j in range(clips.shape[1] // win):
                X.append(clips[v, j * win:(j + 1) * win])
                y.append(VIDEO_LABELS[v])
                sid.append(s)
    return (np.stack(X).astype(np.float32), np.asarray(y, np.int64),
            np.asarray(sid))
