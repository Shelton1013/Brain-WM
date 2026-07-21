#!/usr/bin/env python3
"""Label-efficiency figure -> label_efficiency.pdf.

Frozen-feature probe trained on a fraction of downstream labels: pretrained
(ours, ~2500h subset, converged ckpt) vs random-init encoder, identical splits.
Four panels: Mumtaz + Mental Arithmetic (linear probe), ISRUC + HMC (seq2seq).

Numbers are from eval_label_efficiency{,_seq2seq}.py on
sub2100_2500h_matched/checkpoint_ep16. Re-run and paste here to update.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FR = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
DATA = {
    "Mumtaz (depression)": dict(
        pre=[0.899, 0.926, 0.926, 0.932, 0.929, 0.917],
        rnd=[0.637, 0.714, 0.736, 0.785, 0.807, 0.820]),
    "Mental Arithmetic": dict(
        pre=[0.579, 0.621, 0.635, 0.641, 0.629, 0.607],
        rnd=[0.522, 0.516, 0.514, 0.518, 0.528, 0.516]),
    "ISRUC (sleep)": dict(
        pre=[0.229, 0.224, 0.652, 0.710, 0.733, 0.751],
        rnd=[0.199, 0.200, 0.252, 0.386, 0.432, 0.554]),
    "HMC (sleep)": dict(
        pre=[0.203, 0.198, 0.556, 0.642, 0.664, 0.703],
        rnd=[0.206, 0.200, 0.235, 0.379, 0.432, 0.492]),
}

ACC, RND, INK = "#0A8F9E", "#8795A1", "#1B2A38"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": "#5b6b78", "axes.linewidth": 0.7,
})

fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True)
for ax, (title, d) in zip(axes.ravel(), DATA.items()):
    pre, rnd = np.array(d["pre"]), np.array(d["rnd"])
    ax.fill_between(FR, rnd, pre, where=pre >= rnd, color=ACC, alpha=0.13, lw=0)
    ax.axhline(rnd[-1], ls=":", lw=0.9, color=RND, zorder=1)
    ax.text(0.011, rnd[-1] + 0.008, "random @100%", color=RND,
            fontsize=6.6, va="bottom")
    ax.plot(FR, rnd, "--", color=RND, lw=1.3, marker="o", ms=3.4,
            mfc="white", mec=RND, mew=1.0, label="random init", zorder=3)
    ax.plot(FR, pre, "-", color=ACC, lw=1.6, marker="o", ms=3.6,
            mfc=ACC, mec=ACC, label="pretrained (ours)", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(FR)
    ax.set_xticklabels(["1", "5", "10", "25", "50", "100"], fontsize=7)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK, fontweight="bold")
    ax.grid(True, which="major", axis="y", color="#E3E9ED", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.03)
    ax.set_ylim(0.15, 1.0)

for ax in axes[:, 0]:
    ax.set_ylabel("balanced accuracy", fontsize=8)
for ax in axes[1, :]:
    ax.set_xlabel("% of downstream training labels (log)", fontsize=8)
axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False, handlelength=1.8)
fig.tight_layout(pad=0.6)
fig.savefig("label_efficiency.pdf", bbox_inches="tight")
print("wrote label_efficiency.pdf")
