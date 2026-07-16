"""Generate result figures for the paper. Run: python make_figures.py
Outputs survives_ft.pdf and filter_ablation.pdf in this directory.
Numbers are hard-coded from current evals; update as full results arrive.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
RAND, OURS = "#9AA7B2", "#0A8F9E"

# ---------------------------------------------------------------- Fig 1
# task -> metric label, [rand_frozen, ours_frozen, rand_ft, ours_ft]  (None = TBD)
survive = {
    "ISRUC\n(BA)":   [0.566, 0.717, 0.738, 0.769],
    "Mumtaz\n(BA)":  [0.733, 0.837, 0.798, 0.869],
    "Siena\n(AUC-PR)": [0.081, 0.239, 0.053, 0.654],
}
fig, ax = plt.subplots(figsize=(6.6, 2.9))
tasks = list(survive); x = np.arange(len(tasks)); w = 0.19
groups = [("Random·frozen", RAND, 0, .55), ("Ours·frozen", OURS, 1, .55),
          ("Random·FT", RAND, 2, 1.0), ("Ours·FT", OURS, 3, 1.0)]
for label, color, i, alpha in groups:
    vals = [survive[t][i] for t in tasks]
    ax.bar(x + (i-1.5)*w, vals, w, label=label, color=color, alpha=alpha,
           edgecolor="white", linewidth=.5,
           hatch="//" if i in (2, 3) else None)
ax.set_xticks(x); ax.set_xticklabels(tasks)
ax.set_ylabel("score"); ax.set_ylim(0, 1)
ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper left")
ax.set_title("Pretraining advantage survives fine-tuning", fontsize=11, loc="left")
ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig("survives_ft.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2
# filter vs no-filter (frozen). ISRUC=BA(helps), Siena=AUC-PR(hurts)
filt = {"ISRUC (BA)\nrhythmic": (0.676, 0.717),
        "Siena (AUC-PR)\ntransient": (0.311, 0.239)}
fig, ax = plt.subplots(figsize=(4.0, 2.9))
labels = list(filt); x = np.arange(len(labels)); w = 0.34
ax.bar(x - w/2, [filt[l][0] for l in labels], w, label="no filter",
       color=RAND, edgecolor="white")
ax.bar(x + w/2, [filt[l][1] for l in labels], w, label="0.5–45 Hz filter",
       color=OURS, edgecolor="white")
for i, l in enumerate(labels):
    a, b = filt[l]
    ax.annotate("↑" if b > a else "↓", (i + w/2, b + .02),
                ha="center", color="#0F9D76" if b > a else "#D1541F", fontsize=13)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("frozen score"); ax.set_ylim(0, .85)
ax.legend(fontsize=8, frameon=False)
ax.set_title("Filtering: task-dependent", fontsize=11, loc="left")
ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig("filter_ablation.pdf"); plt.close(fig)

print("wrote survives_ft.pdf, filter_ablation.pdf")
