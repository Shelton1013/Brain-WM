#!/usr/bin/env python3
"""Label-efficiency figure -> label_efficiency.pdf.

Frozen-feature probe trained on a fraction of downstream labels: pretrained
(ours) vs random-init encoder, on identical splits. Two panels: Mumtaz (linear
probe) and ISRUC (seq2seq head).

DATA below is provisional (measured on an earlier model checkpoint). Re-run
eval_label_efficiency.py on the final ~2500h model and paste the numbers here,
then `python make_label_efficiency.py` to regenerate before submission.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- data (fraction of training labels; balanced accuracy) -----------------
FR = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
DATA = {
    "Mumtaz (linear probe)": dict(
        pre=[0.912, 0.953, 0.949, 0.957, 0.956, 0.949],
        rnd=[0.624, 0.707, 0.737, 0.784, 0.799, 0.821]),
    "ISRUC (seq2seq head)": dict(
        pre=[0.200, 0.215, 0.615, 0.703, 0.724, 0.737],
        rnd=[0.198, 0.199, 0.245, 0.369, 0.437, 0.564]),
}

ACC, RND, INK = "#0A8F9E", "#8795A1", "#1B2A38"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": "#5b6b78", "axes.linewidth": 0.7,
})

fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.1), sharex=True)
for ax, (title, d) in zip(axes, DATA.items()):
    pre, rnd = np.array(d["pre"]), np.array(d["rnd"])
    # gap shading (pretrained advantage)
    ax.fill_between(FR, rnd, pre, where=pre >= rnd, color=ACC, alpha=0.13, lw=0)
    # random @ 100% reference
    ax.axhline(rnd[-1], ls=":", lw=0.9, color=RND, zorder=1)
    ax.text(0.011, rnd[-1] + 0.006, "random @100%", color=RND,
            fontsize=6.6, va="bottom")
    # curves
    ax.plot(FR, rnd, "--", color=RND, lw=1.3, marker="o", ms=3.6,
            mfc="white", mec=RND, mew=1.0, label="random init", zorder=3)
    ax.plot(FR, pre, "-", color=ACC, lw=1.6, marker="o", ms=3.8,
            mfc=ACC, mec=ACC, label="pretrained (ours)", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(FR)
    ax.set_xticklabels(["1%", "5%", "10%", "25%", "50%", "100%"], fontsize=7)
    ax.set_ylabel("balanced accuracy", fontsize=8)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK, fontweight="bold")
    ax.grid(True, which="major", axis="y", color="#E3E9ED", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.03)

axes[0].legend(loc="lower right", fontsize=7, frameon=False, handlelength=1.8)
axes[-1].set_xlabel("fraction of downstream training labels (log scale)",
                    fontsize=8)
fig.tight_layout(pad=0.5)
fig.savefig("label_efficiency.pdf", bbox_inches="tight")
print("wrote label_efficiency.pdf")
