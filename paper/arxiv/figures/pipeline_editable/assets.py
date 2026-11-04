"""Image assets + bar values for the editable pipeline slide (same signals as make_pipeline_ref.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)
FS, P, T = 256, 200, 2560
NP = T // P
BANDS = [("δ", 0.5, 4), ("θ", 4, 8), ("α", 8, 13), ("β", 13, 30), ("γ", 30, 45)]
BCOL = ["#5b5bd6", "#8e5bd0", "#1f9e89", "#e0a030", "#d1495b"]
MASK = "#f4b183"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
rng = np.random.default_rng(3)
t = np.arange(T) / FS


def eeg(seed, alpha_amp=1.0, delta_amp=1.0):
    r = np.random.default_rng(seed)
    pink = np.cumsum(r.standard_normal(T)) * 0.02
    pink -= np.convolve(pink, np.ones(256) / 256, "same")
    x = (delta_amp * 1.2 * np.sin(2 * np.pi * 1.8 * t + r.uniform(0, 6))
         + 0.6 * np.sin(2 * np.pi * 6 * t + r.uniform(0, 6))
         + alpha_amp * 0.9 * np.sin(2 * np.pi * 10.2 * t + r.uniform(0, 6)) * (1 + 0.4 * np.sin(2 * np.pi * 0.3 * t))
         + 0.25 * np.sin(2 * np.pi * 20 * t + r.uniform(0, 6))
         + 0.1 * np.sin(2 * np.pi * 38 * t + r.uniform(0, 6))
         + 0.12 * r.standard_normal(T) + pink)
    return x / x.std()


X = np.stack([eeg(1, 0.5, 1.3), eeg(2, 0.8, 1.0), eeg(3, 1.4, 0.7)])
MASKED = rng.random((3, NP)) < 0.3
MASKED[0, 2] = MASKED[1, 5] = MASKED[2, 8] = True


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, transparent=True)
    plt.close(fig)


def eeg_png(name, chans, mask, w, h, mcol=MASK, lw=0.45):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0.035, 0.18, 0.93, 0.82])
    for k, c in enumerate(chans):
        ax.plot(t, X[c] * 0.2 - k, color="#333333", lw=lw)
        if mask:
            for j in range(NP):
                if MASKED[c, j]:
                    ax.add_patch(Rectangle((j * P / FS, -k - 0.45), P / FS, 0.9, color=mcol, alpha=0.55, lw=0))
    for j in range(1, NP):
        ax.axvline(j * P / FS, color="#bdbdbd", lw=0.3, ls=":")
    ax.set_xlim(0, T / FS); ax.set_ylim(-len(chans) + 0.4, 0.6)
    ax.set_yticks([]); ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.tick_params(labelsize=6, length=2, pad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, name)


eeg_png("eeg_raw.png", [0, 1, 2], False, 2.25, 1.5)
eeg_png("eeg_masked_2ch.png", [1, 2], True, 1.95, 1.15)
eeg_png("eeg_masked_3ch_gray.png", [0, 1, 2], True, 3.0, 1.75, mcol="#a9a9a9", lw=0.5)
eeg_png("eeg_raw_wide.png", [0, 1, 2], False, 3.0, 1.1, lw=0.45)

patch = X[1, 5 * P:6 * P]
spec = np.abs(np.fft.rfft(patch)) ** 2
f = np.fft.rfftfreq(P, 1 / FS)
y_tgt = np.array([np.log(2 * spec[(f >= lo) & (f < hi)].sum() / P ** 2 + 1e-3) for _, lo, hi in BANDS])
y_hat = y_tgt + np.array([0.35, -0.25, 0.3, -0.35, 0.2])

fig = plt.figure(figsize=(2.25, 0.95))
ax = fig.add_axes([0.035, 0.22, 0.93, 0.78])
ax.plot(np.arange(P) / FS, patch, color="#333333", lw=0.8)
ax.set_yticks([]); ax.tick_params(labelsize=6, length=2, pad=1)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
save(fig, "patch.png")

fig = plt.figure(figsize=(3.0, 1.5))
ax = fig.add_axes([0.14, 0.2, 0.84, 0.75])
m = (f > 0) & (f <= 50)
ax.semilogy(f[m], spec[m] + 1e-6, color="#333333", lw=0.9, marker="o", ms=2)
for (nm, lo, hi), c in zip(BANDS, BCOL):
    ax.axvspan(lo, hi, color=c, alpha=0.18, lw=0)
    ax.text((lo + hi) / 2, 0.88, nm, ha="center", transform=ax.get_xaxis_transform(), color=c,
            fontsize=8, fontweight="bold")
ax.set_xlim(0, 47); ax.set_xlabel("frequency (Hz)", fontsize=6.5, labelpad=1)
ax.set_ylabel("$|X_k|^2$", fontsize=7); ax.tick_params(labelsize=6, length=2, pad=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
save(fig, "periodogram.png")

spec_c = np.fft.rfft(patch)
for name, base, col in (("waves_same_power.png", 10, None), ("waves_many.png", 20, None)):
    fig = plt.figure(figsize=(1.35, 1.2))
    ax = fig.add_axes([0, 0, 1, 1])
    for k in range(3):
        ph = np.exp(1j * np.random.default_rng(base + k).uniform(0, 2 * np.pi, len(spec_c)))
        w = np.fft.irfft(np.abs(spec_c) * ph, n=P)
        ax.plot(np.arange(P) / FS, w / patch.std() * 0.3 - k, color=["#2b2b2b", "#555555", "#888888"][k], lw=0.7)
    ax.axis("off")
    save(fig, name)
fig = plt.figure(figsize=(1.0, 1.1))
ax = fig.add_axes([0, 0, 1, 1]); ax.plot(np.arange(P) / FS, patch, color="#b5475b", lw=0.8); ax.axis("off")
save(fig, "recon_patch.png")

json.dump({"y_hat": y_hat.tolist(), "y_tgt": y_tgt.tolist(), "masked": MASKED.astype(int).tolist()},
          open(OUT / "values.json", "w"))
print("assets ->", OUT)

# mini power spectrum for the teaser (no tick labels): the masked patch's periodogram, bands shaded
fig = plt.figure(figsize=(1.1, 0.8))
ax = fig.add_axes([0.06, 0.2, 0.92, 0.78])
m = (f > 0) & (f <= 46)
ax.semilogy(f[m], spec[m] + 1e-6, color="#333333", lw=0.8)
for (nm, lo, hi), c in zip(BANDS, BCOL):
    ax.axvspan(lo, hi, color=c, alpha=0.28, lw=0)
ax.set_xlim(0, 46); ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()
ax.set_xlabel("frequency", fontsize=6.5, labelpad=1, family="serif")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
save(fig, "spectrum_mini.png")
