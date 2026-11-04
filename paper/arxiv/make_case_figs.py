"""Case-analysis figures for the eight downstream datasets, from the saved test predictions
(results/bootstrap/preds; strict linear probe, one checkpoint per model).

A1  confusion matrices: SpecBraM (main checkpoint), REVE-Base and their difference
A2  per-subject balanced accuracy: SpecBraM vs. each external model

    python paper/arxiv/make_case_figs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "results/bootstrap/preds"
OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})

DATASETS = {
    "isruc": ("ISRUC", ["W", "N1", "N2", "N3", "REM"]),
    "hmc": ("HMC", ["W", "N1", "N2", "N3", "REM"]),
    "tuab": ("TUAB", ["Normal", "Abnormal"]),
    "tuev": ("TUEV", ["SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG"]),
    "ma": ("MA", ["Rest", "Task"]),
    "mumtaz": ("Mumtaz", ["Healthy", "MDD"]),
    "faced": ("FACED", ["Anger", "Disgust", "Fear", "Sadness", "Neutral", "Amusement",
                        "Inspiration", "Joy", "Tenderness"]),
    "physio": ("PhysioNet-MI", ["Left fist", "Right fist", "Both fists", "Both feet"]),
}
NAMES = {"specbram": "SpecBraM", "cbramod": "CBraMod", "labram": "LaBraM-Base", "biot": "BIOT",
         "csbrain": "CSBrain", "reve_base": "REVE-Base"}


def load(ds, name):
    """npz of one model; SpecBraM = main checkpoint, REVE = readout chosen on validation."""
    if name == "specbram":
        f = sorted(PRED.glob(f"{ds}__m_s42*.npz"))
    elif name == "reve_base":
        c = {r: sorted(PRED.glob(f"{ds}__reve_base_{r}*.npz")) for r in ("mean", "attn")}
        c = {r: np.load(v[0]) for r, v in c.items() if v}
        # tie -> mean pooling, as in the tables
        r = max(c, key=lambda k: (float(c[k]["val_ba"]), k == "mean"))
        return c[r]
    else:
        f = sorted(PRED.glob(f"{ds}__{name}.npz")) + sorted(PRED.glob(f"{ds}__{name}__*.npz"))
    return np.load(f[0]) if f else None


def cm_rows(d, k):
    return confusion_matrix(d["y"], d["pred"], labels=list(range(k)), normalize="true")


def a1():
    for ds, (title, cls) in DATASETS.items():
        k = len(cls)
        a, b = load(ds, "specbram"), load(ds, "reve_base")
        ca, cb = cm_rows(a, k), cm_rows(b, k)
        wide = k > 6
        fig, axes = plt.subplots(1, 3, figsize=(7.0 if wide else 3.4 + 0.25 * k, 2.6 if wide else 1.25 + 0.17 * k))
        panels = [(ca, f"SpecBraM (BA {balanced_accuracy_score(a['y'], a['pred']):.3f})", "Blues", (0, 1)),
                  (cb, f"REVE-Base (BA {balanced_accuracy_score(b['y'], b['pred']):.3f})", "Blues", (0, 1)),
                  (ca - cb, "SpecBraM $-$ REVE", "RdBu", None)]
        for ax, (m, name, cmap, lim) in zip(axes, panels):
            if lim is None:
                v = max(0.05, np.abs(m).max())
                lim = (-v, v)
            ax.imshow(m, cmap=cmap, vmin=lim[0], vmax=lim[1])
            fs = 4.2 if wide else (5.5 if k > 4 else 6.5)
            for i in range(k):
                for j in range(k):
                    val = m[i, j]
                    txt = f"{100 * val:+.0f}" if cmap == "RdBu" else f"{100 * val:.0f}"
                    dark = abs(val) > 0.6 * max(abs(lim[0]), abs(lim[1]))
                    ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                            color="white" if dark else "#222222")
            ax.set_xticks(range(k)); ax.set_yticks(range(k))
            ax.set_xticklabels(cls, rotation=90 if k > 2 else 0, fontsize=5.5 if wide else 6)
            ax.set_yticklabels(cls if ax is axes[0] else [], fontsize=5.5 if wide else 6)
            ax.set_title(name, fontsize=6.5)
            ax.set_xlabel("predicted", fontsize=6)
            ax.tick_params(length=0)
        axes[0].set_ylabel("true", fontsize=6)
        fig.tight_layout(pad=0.3, w_pad=0.6)
        fig.savefig(OUT / f"cm_{ds}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print("A1", ds, "diag SpecBraM-REVE:", np.round(100 * (np.diag(ca) - np.diag(cb)), 1))




def per_subject(d):
    out = {}
    for s_ in np.unique(d["subj"]):
        m = d["subj"] == s_
        y, p = d["y"][m], d["pred"][m]
        # balanced accuracy within the subject (= accuracy when the subject has a single class)
        out[str(s_)] = float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))
    return out


def a2():
    import json
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.7))
    frac = {}
    for ax, (ds, (title, _)) in zip(axes.flat, DATASETS.items()):
        ours = per_subject(load(ds, "specbram"))
        frac[ds] = {}
        for name in ("cbramod", "labram", "biot", "csbrain", "reve_base"):
            d = load(ds, name)
            if d is None:
                continue
            other = per_subject(d)
            diff = np.array([ours[k] - other[k] for k in ours])
            frac[ds][name] = {"better": float((diff > 0).mean()), "tie": float((diff == 0).mean()),
                              "n": int(len(diff))}
        rv = per_subject(load(ds, "reve_base"))
        x = np.array([rv[k] for k in ours]); y = np.array([ours[k] for k in ours])
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max()); pad = 0.04 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999999", lw=0.8, ls="--", zorder=1)
        ax.scatter(x, y, s=9 if len(x) < 60 else 4, color="#1f6f78", alpha=0.75, lw=0, zorder=2)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")
        b = frac[ds]["reve_base"]
        unit = "rec." if ds == "tuev" else "subj."
        worse = 1 - b["better"] - b["tie"]
        ax.set_title(f"{title} ({b['n']} {unit}): better {100 * b['better']:.0f}%, worse {100 * worse:.0f}%",
                     fontsize=6)
        ax.tick_params(labelsize=5.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("REVE-Base", fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("SpecBraM", fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.6)
    fig.savefig(OUT / "per_subject.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    json.dump(frac, open(ROOT / "results/bootstrap/per_subject_frac.json", "w"), indent=1)
    # win / tie / loss table against every external model
    cols = [("cbramod", "CBraMod"), ("labram", "LaBraM-Base"), ("biot", "BIOT"), ("csbrain", "CSBrain"),
            ("reve_base", "REVE-Base")]
    rows = []
    for ds, (title, _) in DATASETS.items():
        cells = []
        for k, _ in cols:
            v = frac[ds].get(k)
            if v is None:
                cells.append("--")
            else:
                w, t = round(100 * v["better"]), round(100 * v["tie"])
                cells.append(f"{w}/{t}/{100 - w - t}")
        unit = "rec." if ds == "tuev" else "subj."
        rows.append(f"{title} ({frac[ds]['reve_base']['n']} {unit}) & " + " & ".join(cells) + r" \\")
    head = " & ".join(n for _, n in cols)
    body = "\n".join(rows)
    tex = (r"""% auto-generated by make_case_figs.py -- do not edit
\begin{table}[h]\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Per-subject comparison}: percentage of test subjects (TUEV: recordings) on
which SpecBraM's balanced accuracy is higher / equal / lower than that of each external model
(strict linear probe, one checkpoint per model).}
\label{tab:per_subject}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l """ + "c" * len(cols) + r"""}
\toprule
Dataset & """ + head + r""" \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    (ROOT / "paper/arxiv/tables/per_subject.tex").write_text(tex, encoding="utf-8")
    for ds, r in frac.items():
        print("A2", ds, {k: f"{100 * v['better']:.0f}%/{100 * v['tie']:.0f}% (n={v['n']})" for k, v in r.items()})


STAGE_ROW = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}            # W, REM, N1, N2, N3 from top to bottom
STAGE_TICKS = ["N3", "N2", "N1", "REM", "W"]


def b1():
    """Whole-night hypnograms: expert vs. model predictions for the test subject whose SpecBraM
    balanced accuracy is closest to the dataset median (chosen by rule, not by eye)."""
    models = [("specbram", "SpecBraM", "#1f6f78"), ("reve_base", "REVE-Base", "#d08a2a"),
              ("cbramod", "CBraMod", "#8a5ab0")]
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 3.9), sharex="col")
    picked = {}
    for col, ds in enumerate(("isruc", "hmc")):
        D = {k: load(ds, k) for k, _, _ in models}
        ref = D["specbram"]
        subs = list(dict.fromkeys(ref["subj"].tolist()))
        ba = {u: balanced_accuracy_score(ref["y"][ref["subj"] == u], ref["pred"][ref["subj"] == u])
              for u in subs}
        med = float(np.median(list(ba.values())))
        u = min(subs, key=lambda k: abs(ba[k] - med))
        m = ref["subj"] == u
        y = ref["y"][m]
        t = np.arange(len(y)) * 30 / 3600                 # hours
        rows = [("Expert", y, "#222222")] + [(nm, D[k]["pred"][D[k]["subj"] == u], c) for k, nm, c in models]
        picked[ds] = (u, len(y), med)
        for r, (nm, z, c) in enumerate(rows):
            ax = axes[r, col]
            zz = np.array([STAGE_ROW[int(v)] for v in z])
            ax.step(t, zz, where="post", color=c, lw=0.7)
            if r:
                wrong = z != y
                ax.scatter(t[wrong], zz[wrong], s=1.5, color="#c0392b", lw=0, zorder=3)
                acc = balanced_accuracy_score(y, z)
                ax.text(1.0, 1.02, f"BA {acc:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
            ax.set_yticks(range(5)); ax.set_yticklabels(STAGE_TICKS if col == 0 else [], fontsize=5.5)
            ax.set_ylim(-0.6, 4.6)
            ax.text(0.0, 1.02, nm, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
                    color=c, fontweight="bold")
            ax.tick_params(labelsize=5.5, length=2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[-1, col].set_xlabel(f"{DATASETS[ds][0]}: hours since start of recording", fontsize=6.5)
    fig.tight_layout(pad=0.3, h_pad=0.9, w_pad=1.0)
    fig.savefig(OUT / "hypnogram.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    for ds, (u, n, med) in picked.items():
        print("B1", ds, "subject", u, "epochs", n, "median BA", round(med, 3))


BANDS = [(0.5, 4, "#8e7cc3"), (4, 8, "#76a5af"), (8, 13, "#93c47d"), (13, 30, "#f6b26b"), (30, 45, "#e06666")]


def b2():
    """TUEV examples: windows that one model classifies correctly and the other not (chosen by
    rule: the most confident correct prediction of the model that is right)."""
    import json
    from scipy.signal import welch
    cases = json.load(open(ROOT / "results/bootstrap/tuev_cases.json"))
    raw = np.load(ROOT / "results/bootstrap/tuev_cases.npz")
    ch = [str(c) for c in raw["mono_ch"]]
    order = ["SPSW", "EYEM", "GPED", "PLED"]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.4), gridspec_kw={"height_ratios": [3.2, 1]})
    for col, name in enumerate(order):
        c, x = cases[name], raw[name]                        # [19, 1000] uV, 200 Hz
        t = np.arange(x.shape[1]) / 200
        ax = axes[0, col]
        sc = 4 * np.median(np.abs(x - np.median(x, 1, keepdims=True))) + 1e-6
        bar = 100 if 100 / sc <= 2.5 else (50 if 50 / sc <= 2.5 else 20)
        for i in range(x.shape[0]):
            ax.plot(t, np.clip((x[i] - np.median(x[i])) / sc, -1.3, 1.3) - i, color="#222222", lw=0.35)
        ax.set_yticks(-np.arange(len(ch))); ax.set_yticklabels(ch if col == 0 else [], fontsize=4.5)
        ax.set_ylim(-len(ch), 1.2); ax.set_xlim(0, t[-1])
        ax.axvspan(2.0, 3.0, color="#fde9c8", zorder=0)        # the annotated 1-s event
        ax.plot([0.15, 0.15], [-len(ch) + 0.6, -len(ch) + 0.6 + bar / sc], color="#c0392b", lw=1.2)
        ax.text(0.22, -len(ch) + 0.6, f"{bar} $\\mu$V", fontsize=4.5, color="#c0392b", va="bottom")
        ok = {"SpecBraM": c["spec"], "REVE": c["reve"]}
        lines = [f"true: {name}"] + [f"{k}: {v} " + ("(correct)" if v == name else "(wrong)") for k, v in ok.items()]
        ax.set_title("\n".join(lines), fontsize=5.8, loc="left")
        ax.tick_params(labelsize=5, length=2)
        ax.set_xlabel("time (s)", fontsize=5.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax2 = axes[1, col]
        f, P = welch(x, fs=200, nperseg=200)
        P = P.mean(0)
        for lo, hi, colr in BANDS:
            ax2.axvspan(lo, hi, color=colr, alpha=0.25, lw=0)
        ax2.semilogy(f, P, color="#222222", lw=0.8)
        ax2.set_xlim(0.5, 45)
        sel = (f >= 0.5) & (f <= 45)
        ax2.set_ylim(P[sel].min() / 2, P[sel].max() * 2)
        ax2.set_xlabel("frequency (Hz)", fontsize=5.5)
        if col == 0:
            ax2.set_ylabel("PSD ($\\mu$V$^2$/Hz)", fontsize=5.5)
        ax2.tick_params(labelsize=5, length=2)
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.8)
    fig.savefig(OUT / "tuev_cases.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("B2 done")


FEAT = ROOT / "results/feats/preds"
PAL = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f", "#bab0ac"]


def load_feat(ds, name):
    if name == "specbram":
        f = sorted(FEAT.glob(f"{ds}__m_s42*.npz"))
        return np.load(f[0]) if f else None
    if name == "reve_base":
        c = {r: sorted(FEAT.glob(f"{ds}__reve_base_{r}*.npz")) for r in ("mean", "attn")}
        c = {r: np.load(v[0]) for r, v in c.items() if v}
        if not c:
            return None
        r = max(c, key=lambda k: (float(c[k]["val_ba"]), k == "mean"))
        return c[r]
    f = sorted(FEAT.glob(f"{ds}__{name}.npz")) + sorted(FEAT.glob(f"{ds}__{name}__*.npz"))
    return np.load(f[0]) if f else None


def c1(per_class=300):
    """t-SNE of the frozen test features (the standardized features the linear probe sees), on a
    class-balanced subsample; plus the 10-NN accuracy in the full feature space."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    models = [("specbram", "SpecBraM"), ("reve_base", "REVE-Base"), ("cbramod", "CBraMod")]
    fig, axes = plt.subplots(8, 3, figsize=(7.0, 17.5))
    stats = {}
    for r, (ds, (title, cls)) in enumerate(DATASETS.items()):
        ref = load_feat(ds, "specbram")
        y_all = ref["y"]
        rng = np.random.default_rng(0)
        idx = np.concatenate([rng.permutation(np.flatnonzero(y_all == c))[:per_class]
                              for c in range(len(cls))])
        stats[ds] = {}
        for col, (k, nm) in enumerate(models):
            ax = axes[r, col]
            d = load_feat(ds, k)
            if d is None or "feat" not in d.files:
                ax.axis("off")
                continue
            X = np.nan_to_num(d["feat"][idx].astype(np.float32))
            y = d["y"][idx]
            assert (y == y_all[idx]).all()
            Z = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
            knn = float(cross_val_score(KNeighborsClassifier(10), Z, y, cv=5,
                                        scoring="balanced_accuracy").mean())
            E = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
            stats[ds][k] = knn
            for c in range(len(cls)):
                m = y == c
                ax.scatter(E[m, 0], E[m, 1], s=1.2, color=PAL[c], lw=0, label=cls[c], rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#999999")
            ax.set_title(f"{nm}  (10-NN BA {knn:.2f})", fontsize=6.5)
            if col == 0:
                ax.set_ylabel(title, fontsize=7.5, fontweight="bold")
        h, l = axes[r, 0].get_legend_handles_labels()
        axes[r, 2].legend(h, l, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=5.5,
                          markerscale=4, frameon=False, handletextpad=0.2)
        print("C1", ds, {k: round(v, 3) for k, v in stats[ds].items()}, flush=True)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.3)
    fig.savefig(OUT / "tsne.pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    import json
    json.dump(stats, open(ROOT / "results/feats/knn_stats.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    steps = sys.argv[1:] or ["a1", "a2", "b1", "b2"]
    for st in steps:
        globals()[st]()
