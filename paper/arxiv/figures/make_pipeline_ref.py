"""Reference drawing of the SpecBraM pipeline (linear patch embedding, parameter-free
periodogram band-power target, no SIGReg). Synthetic EEG; the target panels are computed
exactly as in the paper (per-patch FFT, classical bands, log P).

    python paper/arxiv/figures/make_pipeline_ref.py
-> pipeline_ref.pdf / pipeline_ref.png next to this script
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
FS, P, T = 256, 200, 2560
NP = T // P                                   # 12 patches of 0.78 s
BANDS = [("δ", 0.5, 4), ("θ", 4, 8), ("α", 8, 13), ("β", 13, 30), ("γ", 30, 45)]
BCOL = ["#5b5bd6", "#8e5bd0", "#1f9e89", "#e0a030", "#d1495b"]
INK, MUTED, EDGE = "#222222", "#666666", "#bdbdbd"
ENC, ENC_EDGE = "#8fc1a9", "#5f9e82"
MASK = "#f4b183"
GREEN, PURPLE, RED = "#4f9a5f", "#8a5fa8", "#b5475b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans"})

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


CH = ["Fp1", "Cz", "O1"]
X = np.stack([eeg(1, 0.5, 1.3), eeg(2, 0.8, 1.0), eeg(3, 1.4, 0.7)])
MASKED = rng.random((3, NP)) < 0.3
MASKED[0, 2] = MASKED[1, 5] = MASKED[2, 8] = True        # make sure every channel shows a mask


def band_logpow(x):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return np.array([np.log(2 * spec[(f >= lo) & (f < hi)].sum() / len(x) ** 2 + 1e-3) for _, lo, hi in BANDS]), f, spec


def plot_eeg(ax, mask=True, lw=0.35, chans=range(3), labels=True):
    for k, c in enumerate(chans):
        y = X[c] * 0.2 - k
        ax.plot(t, y, color="#333333", lw=lw)
        if labels:
            ax.text(-0.25, -k, CH[c], ha="right", va="center", fontsize=7, color=MUTED)
        if mask:
            for j in range(NP):
                if MASKED[c, j]:
                    ax.add_patch(Rectangle((j * P / FS, -k - 0.45), P / FS, 0.9, color=MASK, alpha=0.55, lw=0))
    for j in range(1, NP):
        ax.axvline(j * P / FS, color=EDGE, lw=0.3, ls=":")
    ax.set_xlim(0, T / FS); ax.set_ylim(-len(list(chans)) + 0.4, 0.6)
    ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.tick_params(labelsize=6, length=2)
    ax.set_yticks([]); ax.set_xlabel("time (s)", fontsize=6, labelpad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def bars(ax, vals, target=None, title=None, ylab=True):
    xs = np.arange(5)
    lo = min(vals.min(), target.min() if target is not None else vals.min()) - 0.6
    if target is not None:
        for i in xs:
            ax.bar(i, target[i] - lo, bottom=lo, width=0.62, fill=False, edgecolor=BCOL[i], lw=1.2, ls="--")
    ax.bar(xs, vals - lo, bottom=lo, width=0.42 if target is not None else 0.6, color="#16848f")
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in BANDS], fontsize=8)
    for lab, c in zip(ax.get_xticklabels(), BCOL):
        lab.set_color(c)
    ax.set_yticks([]); ax.tick_params(length=0)
    if ylab:
        ax.set_ylabel("log band power", fontsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=7.5, color=MUTED, pad=2)


def box(fig, x, y, w, h, text, fc=ENC, ec=ENC_EDGE, fs=10, bold=True, color="white"):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                      transform=fig.transFigure, fc=fc, ec=ec, lw=1.0))
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
             fontweight="bold" if bold else "normal", color=color)


def flame(fig, x, y):
    fig.text(x, y, "trainable", fontsize=7, ha="center", color="white", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="#e4572e", ec="none"))


def arrow(fig, x0, y0, x1, y1, color=GREEN, cs="arc3"):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=11, lw=1.4, color=color, connectionstyle=cs))


fig = plt.figure(figsize=(15, 9.2))
fig.text(0.008, 0.985, "(a) Pretraining", fontsize=11, fontweight="bold", va="top")

# ---------------- encoder branch (top) ----------------
ax = fig.add_axes([0.035, 0.73, 0.15, 0.17]); plot_eeg(ax, mask=False)
ax.set_title("Multichannel EEG, 0.78-s patches", fontsize=8, color=MUTED, pad=2)
box(fig, 0.215, 0.775, 0.095, 0.075, "Patchify +\nlinear\nembedding", fs=8.5)
flame(fig, 0.262, 0.862)
arrow(fig, 0.188, 0.815, 0.213, 0.815)
# token grid: masking acts on the patch EMBEDDINGS (masked tokens -> learnable [M] token),
# not on the raw signal; the target is computed from the uncorrupted patch
ax = fig.add_axes([0.345, 0.735, 0.16, 0.15])
for k in range(3):
    for j in range(NP):
        if MASKED[k, j]:
            ax.add_patch(FancyBboxPatch((j + 0.08, 2 - k + 0.1), 0.84, 0.8, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc="#9e9e9e", ec="#6f6f6f", lw=0.6))
            ax.text(j + 0.5, 2 - k + 0.5, "M", ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold")
        else:
            ax.add_patch(FancyBboxPatch((j + 0.08, 2 - k + 0.1), 0.84, 0.8, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=ENC, ec=ENC_EDGE, lw=0.6))
    ax.text(-0.3, 2 - k + 0.5, CH[k], ha="right", va="center", fontsize=7, color=MUTED)
ax.set_xlim(0, NP); ax.set_ylim(0, 3); ax.axis("off")
ax.set_title("token grid (channel $\\times$ time patch)", fontsize=8, color=MUTED, pad=2)
fig.text(0.425, 0.712, "masked tokens (p = 0.3) $\\rightarrow$ learnable [M] token", fontsize=7.5,
         ha="center", color=INK)
fig.text(0.425, 0.694, "the raw signal itself is not altered", fontsize=6.8, ha="center", color=MUTED)
arrow(fig, 0.312, 0.815, 0.326, 0.815)
box(fig, 0.535, 0.765, 0.12, 0.095, "Transformer\nencoder", fs=10)
fig.text(0.595, 0.772, "criss-cross: channel / time attention", fontsize=6.3, ha="center", color="white")
flame(fig, 0.595, 0.872)
arrow(fig, 0.508, 0.815, 0.533, 0.815)
box(fig, 0.68, 0.775, 0.09, 0.075, "Band-power\nhead", fs=9.5)
flame(fig, 0.725, 0.862)
arrow(fig, 0.657, 0.815, 0.678, 0.815)
patch = X[1, 5 * P:6 * P]
y_tgt, f, spec = band_logpow(patch)
y_hat = y_tgt + np.array([0.35, -0.25, 0.3, -0.35, 0.2])
ax = fig.add_axes([0.80, 0.735, 0.085, 0.15]); bars(ax, y_hat, title="prediction $\\hat y_{c,t,b}$")
arrow(fig, 0.772, 0.815, 0.798, 0.815)

# ---------------- target branch (bottom of panel a) ----------------
ax = fig.add_axes([0.035, 0.47, 0.15, 0.11])
ax.plot(np.arange(P) / FS, patch, color="#2b2b2b", lw=0.8)
ax.set_title("one masked patch $x_{c,t}$ (200 samples)", fontsize=7.5, color=MUTED, pad=2)
ax.set_yticks([]); ax.tick_params(labelsize=6, length=2); ax.set_xlabel("time (s)", fontsize=6, labelpad=1)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
arrow(fig, 0.11, 0.725, 0.11, 0.6, color=PURPLE)
fig.text(0.118, 0.66, "same, uncorrupted\nsignal", fontsize=6.5, color=PURPLE)

ax = fig.add_axes([0.235, 0.45, 0.2, 0.15])
m = (f > 0) & (f <= 50)
ax.semilogy(f[m], spec[m] + 1e-6, color="#2b2b2b", lw=0.9, marker="o", ms=2)
for (nm, lo, hi), c in zip(BANDS, BCOL):
    ax.axvspan(lo, hi, color=c, alpha=0.18, lw=0)
    ax.text((lo + hi) / 2, 0.9, nm, ha="center", transform=ax.get_xaxis_transform(),
            color=c, fontsize=8, fontweight="bold")
ax.set_xlim(0, 47); ax.set_xlabel("frequency (Hz)", fontsize=6.5, labelpad=1)
ax.set_ylabel("$|X_k|^2$", fontsize=7); ax.tick_params(labelsize=6, length=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("per-patch periodogram (bins every 1.28 Hz)", fontsize=7.5, color=MUTED, pad=2)
arrow(fig, 0.187, 0.525, 0.233, 0.525, color=PURPLE)
fig.text(0.21, 0.54, "FFT", fontsize=7.5, color=PURPLE, ha="center")

fig.text(0.455, 0.585, "Parameter-free target", fontsize=9, fontweight="bold", color=PURPLE)
fig.text(0.455, 0.545, r"1. Periodogram   $|X_k|^2$,  $X_k=\sum_\tau x_{c,t,\tau}e^{-i2\pi k\tau/P}$", fontsize=8.5)
fig.text(0.455, 0.51, r"2. Band power   $P_{c,t,b}=\frac{2}{P^2}\sum_{f_k\in B_b}|X_k|^2$", fontsize=8.5)
fig.text(0.455, 0.475, r"3. Log   $y_{c,t,b}=\log(P_{c,t,b}+\varepsilon)$", fontsize=8.5)
fig.text(0.455, 0.445, "δ 0.5–4, θ 4–8, α 8–13, β 13–30, γ 30–45 Hz", fontsize=7.5, color=MUTED)
arrow(fig, 0.437, 0.525, 0.453, 0.525, color=PURPLE)

ax = fig.add_axes([0.70, 0.45, 0.085, 0.15]); bars(ax, y_tgt, target=y_tgt * 0 + y_tgt, title="target $y_{c,t,b}$")
ax.patches[-5:]  # (target drawn as dashed outline only)
for b in list(ax.patches)[5:]:
    b.set_visible(False)
arrow(fig, 0.665, 0.525, 0.698, 0.525, color=PURPLE)

ax = fig.add_axes([0.83, 0.45, 0.1, 0.15]); bars(ax, y_hat, target=y_tgt, title="masked patches only", ylab=False)
arrow(fig, 0.787, 0.525, 0.828, 0.525, color=PURPLE)
arrow(fig, 0.93, 0.80, 0.90, 0.61, color=GREEN, cs="arc3,rad=-0.3")
fig.text(0.70, 0.395, r"$\mathcal{L}=\frac{1}{5|\mathcal{M}|}\;\sum_{(c,t)\in\mathcal{M}}\;\;\sum_{b=1}^{5}\;"
                      r"\left(\hat y_{c,t,b}-y_{c,t,b}\right)^2$", fontsize=10.5)
fig.text(0.70, 0.365, "no reconstruction term, no regularizer", fontsize=7.5, color=MUTED)

# legend
flame(fig, 0.33, 0.376)
fig.patches.append(Rectangle((0.385, 0.377), 0.012, 0.012, transform=fig.transFigure, fc="#16848f"))
fig.text(0.40, 0.375, "predicted band power", fontsize=8.5)
fig.patches.append(Rectangle((0.505, 0.377), 0.012, 0.012, transform=fig.transFigure, fill=False, ec=INK, ls="--"))
fig.text(0.52, 0.375, "target band power", fontsize=8.5)
fig.patches.append(Rectangle((0.61, 0.377), 0.016, 0.012, transform=fig.transFigure, fc=MASK, alpha=0.55))
fig.text(0.63, 0.375, "masked patch", fontsize=8.5)
fig.lines.append(plt.Line2D([0.01, 0.99], [0.345, 0.345], transform=fig.transFigure, color="#2a6f97", lw=1.2))

# ---------------- (b) ours vs alternative ----------------
fig.text(0.008, 0.335, "(b) What the target asks the encoder to learn", fontsize=11, fontweight="bold", va="top")
for x0, col, tag, title in ((0.02, "#79c9c4", "ours", "Predict band power"),
                            (0.51, RED, "alternative", "Reconstruct the raw waveform")):
    fig.patches.append(FancyBboxPatch((x0, 0.02), 0.47, 0.285, boxstyle="round,pad=0.004,rounding_size=0.015",
                                      transform=fig.transFigure, fill=False, ec=col, lw=1.4))
    box(fig, x0 + 0.012, 0.268, 0.06, 0.028, tag, fc=col, ec=col, fs=9)
    fig.text(x0 + 0.17, 0.282, title, fontsize=10.5, va="center")
    fig.lines.append(plt.Line2D([x0 + 0.235] * 2, [0.04, 0.255], transform=fig.transFigure, color=col, lw=1))

ax = fig.add_axes([0.035, 0.10, 0.13, 0.13]); plot_eeg(ax, mask=True, chans=[1, 2], labels=False)
ax = fig.add_axes([0.19, 0.10, 0.045, 0.12]); bars(ax, y_hat, target=y_tgt, ylab=False)
arrow(fig, 0.166, 0.165, 0.187, 0.165, color="#4fb3ad")
fig.text(0.13, 0.05, "infer the hidden patch's power from its context", ha="center", fontsize=8, style="italic")

# many waveforms, same power: same amplitude spectrum, random phases
spec_c = np.fft.rfft(patch)
ax = fig.add_axes([0.27, 0.10, 0.09, 0.13])
for k in range(3):
    ph = np.exp(1j * np.random.default_rng(10 + k).uniform(0, 2 * np.pi, len(spec_c)))
    w = np.fft.irfft(np.abs(spec_c) * ph, n=P)
    ax.plot(np.arange(P) / FS, w / patch.std() * 0.3 - k, color=["#2b2b2b", "#555555", "#888888"][k], lw=0.7)
ax.axis("off"); ax.set_title("many waveforms", fontsize=8, pad=1)
ax = fig.add_axes([0.40, 0.10, 0.06, 0.12]); bars(ax, y_tgt, ylab=False)
ax.set_title("the same power", fontsize=8, pad=1)
arrow(fig, 0.362, 0.165, 0.397, 0.165, color="#4fb3ad")
fig.text(0.37, 0.05, "phase is unpredictable and irrelevant:\nlearn the spectral state", ha="center",
         fontsize=8, fontweight="bold")

ax = fig.add_axes([0.525, 0.10, 0.13, 0.13]); plot_eeg(ax, mask=True, chans=[1, 2], labels=False)
ax = fig.add_axes([0.665, 0.10, 0.065, 0.12])
ax.plot(np.arange(P) / FS, patch, color=RED, lw=0.8); ax.axis("off")
ax.set_title("every sample of\nthe masked patch", fontsize=7.5, pad=1)
arrow(fig, 0.656, 0.165, 0.663, 0.165, color=RED)
fig.text(0.62, 0.05, "infer the hidden patch's waveform from its context", ha="center", fontsize=8, style="italic")
ax = fig.add_axes([0.76, 0.10, 0.05, 0.12]); bars(ax, y_tgt, ylab=False)
ax.set_title("one power", fontsize=8, pad=1)
ax = fig.add_axes([0.85, 0.10, 0.09, 0.13])
for k in range(3):
    ph = np.exp(1j * np.random.default_rng(20 + k).uniform(0, 2 * np.pi, len(spec_c)))
    w = np.fft.irfft(np.abs(spec_c) * ph, n=P)
    ax.plot(np.arange(P) / FS, w / patch.std() * 0.3 - k, color=["#2b2b2b", "#555555", "#888888"][k], lw=0.7)
ax.axis("off"); ax.set_title("many possible waveforms", fontsize=8, pad=1)
arrow(fig, 0.812, 0.165, 0.847, 0.165, color=RED)
fig.text(0.86, 0.05, "capacity spent on phase and noise", ha="center", fontsize=8, fontweight="bold")

fig.savefig(OUT / "pipeline_ref.pdf")
fig.savefig(OUT / "pipeline_ref.png", dpi=150)
print(OUT / "pipeline_ref.pdf")
