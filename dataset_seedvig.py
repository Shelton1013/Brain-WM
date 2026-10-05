"""SEED-VIG vigilance estimation (regression on PERCLOS) for our model.

Raw_Data/*.mat: EEG.data [1,416,000 x 17] µV at 200 Hz = 885 x 8-s samples per
session; perclos_labels/*.mat: perclos [885, 1] in [0, 1].

Protocol = CBraMod (paper split, see cbramod_split): the 23 session
files ordered by subject number, [:15] train,
[15:19] val, [19:] test; one sample = one 8-s window, label = its PERCLOS.
Ref (CBraMod Table 11, FT): CBraMod Pearson 0.6459 / R2 0.3365 / RMSE 0.2587.

Adapted to our input contract only: 0.5-45 Hz band-pass (= pretraining band),
200 -> 256 Hz, per-recording robust scaling over the whole session.

Channels: SEED-VIG has 17 temporal/posterior leads, our encoder was pretrained
on 19 fixed 10-20 slots. `channel_slot_weights` maps each SEED-VIG lead onto
the pretrained slot embeddings by inverse-distance weighting of its two nearest
10-20 slots (standard_1005 positions); an exact position match takes that slot.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

from dataset_mumtaz import OUR_19CH_ORDER, _robust_scale_per_recording

SRC_SR = 200
WIN_S = 8
N_PER_SESSION = 885
VIG_CHANNELS = ["FT7", "FT8", "T7", "T8", "TP7", "TP8", "CP1", "CP2", "P1", "PZ",
                "P2", "PO3", "POZ", "PO4", "O1", "OZ", "O2"]
_ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}   # old 10-20 names


def cbramod_split(root):
    """Experiments ordered by subject NUMBER, [:15] / [15:19] / [19:] — the split
    the CBraMod paper states (1-15 / 16-19 / 20-23) and the one its reported
    numbers come from: their R2/RMSE imply a test-label std of 0.3176 (BIOT row:
    0.3175); this split gives 0.318. Its released script sorts file names
    lexicographically instead ('10_' < '1_'), whose test set (subjects 6-9) has
    std 0.175 and cannot produce those R2/RMSE pairs."""
    files = sorted((p.name for p in (Path(root) / "Raw_Data").glob("*.mat")),
                   key=lambda f: (int(f.split("_")[0]), f))
    assert len(files) == 23, files
    return {"train": files[:15], "val": files[15:19], "test": files[19:]}


def channel_slot_weights(k=2):
    """[17, 19] row-stochastic weights: SEED-VIG lead -> pretrained 10-20 slots."""
    import mne
    pos = {n.upper(): v for n, v in
           mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"].items()}
    slots = np.array([pos[_ALIAS.get(c, c).upper()] for c in OUR_19CH_ORDER])
    W = np.zeros((len(VIG_CHANNELS), len(OUR_19CH_ORDER)), np.float32)
    for i, c in enumerate(VIG_CHANNELS):
        d = np.linalg.norm(slots - pos[c], axis=1)
        j = np.argsort(d)[:k]
        if d[j[0]] < 1e-6:
            W[i, j[0]] = 1.0
        else:
            w = 1.0 / d[j]
            W[i, j] = w / w.sum()
    return W


def load_xy(root, files, sample_rate=256, bandpass=(0.5, 45.0), robust=True):
    """-> X [N, 8*sr, 17] float32, y [N] float32, session ids [N].
    sample_rate=200, bandpass=None, robust=False = CBraMod's input (raw µV)."""
    root = Path(root)
    sos = butter(4, bandpass, btype="band", fs=SRC_SR, output="sos") if bandpass else None
    win = WIN_S * sample_rate
    X, y, sid = [], [], []
    for f in files:
        eeg = loadmat(root / "Raw_Data" / f)["EEG"]
        chn = [str(c[0]).upper() for c in eeg["chn"][0][0][0]]
        assert chn == VIG_CHANNELS, chn
        x = np.asarray(eeg["data"][0][0], np.float64)               # [T, 17] µV
        lab = loadmat(root / "perclos_labels" / f)["perclos"][:, 0].astype(np.float32)
        assert x.shape[0] == N_PER_SESSION * WIN_S * SRC_SR and len(lab) == N_PER_SESSION
        if sos is not None:
            x = sosfiltfilt(sos, x, axis=0)
        if sample_rate != SRC_SR:
            x = resample_poly(x, sample_rate, SRC_SR, axis=0)
        if robust:
            x = _robust_scale_per_recording(x)                        # whole session
        X.append(x[:N_PER_SESSION * win].reshape(N_PER_SESSION, win, -1))
        y.append(lab)
        sid += [f[:-4]] * N_PER_SESSION
    return (np.concatenate(X).astype(np.float32), np.concatenate(y),
            np.asarray(sid))
