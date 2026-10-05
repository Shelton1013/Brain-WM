"""Linear-probe tables + figures for the paper, straight from the result JSONs.

    python paper/figures/make_lp_artifacts.py
reads   results/linear_probe_v2/lp2_*.json   (eval_linear_probe_v2.py output)
writes  paper/tables_linear_probe.tex, paper/figures/lp_label_efficiency.{pdf,png},
        paper/figures/lp_compute.{pdf,png}
Primary protocol = "avg3" (three 10-s windows per 30-s epoch, features averaged).
Rerun whenever new result files arrive; missing models are simply skipped.
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/linear_probe_v2"
PAPER = ROOT / "paper"
MODE = "avg3"
DS = ("isruc", "hmc")
FRACS = ("0.01", "0.05", "0.10", "0.25", "0.50", "1.00")

# REVE numbers are quoted from its paper (Table 4, linear probing, 30-s input,
# attention pooling) — not re-measured; marked with a dagger.
REVE = {"REVE-Base": (0.697, 0.647), "REVE-Large": (0.758, 0.710)}
# main model from 10-01: linear tokenizer + band-power target (option B); its
# matched comparator is linear tokenizer + raw-waveform reconstruction
MAIN, MAIN_RAW = "linear_bandpower_ep16", "linear_rawrecon_ep16"

R = {d: {} for d in DS}
for f in glob.glob(str(RES / "lp2_*.json")):
    d = json.load(open(f))
    R[d["dataset"]].update(d["results"])


def get(ds, key, metric="balanced_accuracy"):
    v = R[ds].get(key, {}).get(MODE)
    return None if v is None else v["test"][metric]


def le(ds, key):
    v = R[ds].get(key, {}).get(MODE)
    return None if v is None else [v["label_eff"][f]["balanced_accuracy"][0] for f in FRACS]


f4 = lambda x: "--" if x is None else f"{x:.4f}"

# ------------------------------- tables -------------------------------------
MODELS = [("random_init", "Random init (our arch.)", "--"),
          ("biot", "BIOT", "mixed"), ("biot_sleep", "BIOT (sleep ckpt)", "mixed"),
          ("cbramod", "CBraMod", r"${\sim}9$k"), ("labram", "LaBraM-Base", r"${\sim}2.5$k")]
rows = []
for key, name, hrs in MODELS:
    if all(get(ds, key) is None for ds in DS):
        continue
    rows.append(f"{name} & {hrs} & " + " & ".join(
        f"{f4(get(ds, key))} & {f4(get(ds, key, 'cohen_kappa'))}" for ds in DS) + r" \\")
for name, (a, b) in REVE.items():
    rows.append(f"{name}$^\\dagger$ & ${{\\sim}}60$k & {a:.3f} & -- & {b:.3f} & -- \\\\")
rows.append(r"\textbf{SpecBraM (ours)} & $2{,}388$ & " + " & ".join(
    f"\\textbf{{{f4(get(ds, MAIN))}}} & \\textbf{{{f4(get(ds, MAIN, 'cohen_kappa'))}}}"
    for ds in DS) + r" \\")
t1 = (r"""\begin{table}[t]\centering\small
\caption{\textbf{Linear probe of frozen EEG foundation models.} Per-epoch pooled
features, logistic regression (C chosen on validation), 30-s epochs as the mean of
three 10-s windows; identical splits and protocol for every model we ran.
$^\dagger$Quoted from the REVE paper (its own linear-probing protocol).}
\label{tab:lp_main}
\begin{tabular}{l c cc cc}
\toprule
& & \multicolumn{2}{c}{ISRUC} & \multicolumn{2}{c}{HMC}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Model & Pretrain (h) & BA & $\kappa$ & BA & $\kappa$\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

le_models = [("labram", "LaBraM-Base"), ("cbramod", "CBraMod"), ("biot", "BIOT"),
             (MAIN_RAW, "Raw-waveform recon (same arch.)"), (MAIN, r"\textbf{SpecBraM}")]
rows = []
for ds in DS:
    rows.append(r"\multicolumn{7}{l}{\emph{" + ds.upper() + r"}}\\")
    for key, name in le_models:
        c = le(ds, key)
        if c:
            rows.append(f"{name} & " + " & ".join(f"{x:.3f}" for x in c) + r" \\")
t2 = (r"""\begin{table}[t]\centering\small
\caption{\textbf{Label efficiency under the linear probe} (balanced accuracy vs.\
fraction of labelled training epochs; mean of 5 subsets below 100\%).}
\label{tab:lp_label_eff}
\begin{tabular}{l cccccc}
\toprule
Model & 1\% & 5\% & 10\% & 25\% & 50\% & 100\%\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

TARGETS = [("ours_ep16", "Band power (ours)"), ("patchlocal_ep16", "\\quad leak-free filtering"),
           ("fft_ep16", "\\quad exact-FFT bands"), ("freqmask_ep16", "\\quad frequency masking"),
           ("raw_ep16", "Raw-waveform reconstruction"),
           ("bandrecon_ep16", "Band-waveform reconstruction")]
rows = [f"{name} & " + " & ".join(f"{f4(get(ds, k))} & {f4(get(ds, k, 'cohen_kappa'))}"
                                  for ds in DS) + r" \\" for k, name in TARGETS]
two = [("Filterbank", "ours_ep16", "raw_ep16"),
       ("Linear", "linear_bandpower_ep16", "linear_rawrecon_ep16")]
rows2 = []
for tok, bp, raw in two:
    cells = []
    for ds in DS:
        a, b = get(ds, bp), get(ds, raw)
        cells.append(f"{f4(a)} & {f4(b)} & " + ("--" if None in (a, b) else f"{(a - b) * 100:+.1f}"))
    rows2.append(f"{tok} & " + " & ".join(cells) + r" \\")
t3 = (r"""\begin{table}[t]\centering\small
\caption{\textbf{Target study, matched backbone / data / steps} (linear probe).
Top: pretext targets. Bottom: tokenizer $\times$ target; the target effect is
the same with and without the spectral tokenizer.}
\label{tab:lp_targets}
\begin{tabular}{l cc cc}
\toprule
& \multicolumn{2}{c}{ISRUC} & \multicolumn{2}{c}{HMC}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Pretext target & BA & $\kappa$ & BA & $\kappa$\\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{5}{l}{\emph{Tokenizer $\times$ target (BA): band power / raw waveform / gap}}\\
\end{tabular}\\[2pt]
\begin{tabular}{l ccc ccc}
& \multicolumn{3}{c}{ISRUC} & \multicolumn{3}{c}{HMC}\\
Tokenizer & BP & Raw & $\Delta$ & BP & Raw & $\Delta$\\
\midrule
""" + "\n".join(rows2) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

rows = []
for e in (4, 8, 16):
    cells = []
    for ds in DS:
        a, b = get(ds, f"ours_ep{e}"), get(ds, f"raw_ep{e}")
        cells.append(f"{f4(a)} & {f4(b)}")
    rows.append(f"{e} & " + " & ".join(cells) + r" \\")
t4 = (r"""\begin{table}[t]\centering\small
\caption{\textbf{Pretraining efficiency} (linear probe BA along pretraining):
band power at epoch 4 already exceeds raw-waveform reconstruction at epoch 16.}
\label{tab:lp_compute}
\begin{tabular}{c cc cc}
\toprule
& \multicolumn{2}{c}{ISRUC} & \multicolumn{2}{c}{HMC}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Epoch & Band power & Raw waveform & Band power & Raw waveform\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

(PAPER / "tables_linear_probe.tex").write_text(
    "% Auto-generated by paper/figures/make_lp_artifacts.py — do not edit by hand.\n"
    + "\n".join([t1, t2, t3, t4]), encoding="utf-8")

# ------------------------------- figures ------------------------------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
SERIES = [(MAIN, "SpecBraM (ours)", "#2a78d6", "o", "-", 3.0),
          ("labram", "LaBraM-Base", "#1baf7a", "^", "--", 1.8),
          ("cbramod", "CBraMod", "#eb6834", "s", "--", 1.8),
          ("biot", "BIOT", "#eda100", "D", ":", 1.8),
          ("random_init", "Random init", "#9a9994", "x", ":", 1.4)]
xs = [1, 5, 10, 25, 50, 100]


def style(ax, title):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURF)
for ax, ds in zip(axes, DS):
    style(ax, f"{ds.upper()} — linear probe vs. labels")
    for key, lab, c, m, ls, lw in SERIES:
        y = le(ds, key)
        if y:
            ax.plot(xs, y, color=c, ls=ls, lw=lw, marker=m, ms=7, mec=SURF, mew=1.5,
                    label=lab, zorder=4 if key == MAIN else 3)
    reve = REVE["REVE-Large"][DS.index(ds)]
    ax.scatter([100], [reve], marker="*", s=180, color="#008300", zorder=5,
               label="REVE-Large (paper, 100%)")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x}%" for x in xs])
    ax.minorticks_off(); ax.set_xlabel("Labelled training epochs")
    ax.set_ylabel("Balanced accuracy")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(PAPER / f"figures/lp_label_efficiency.{ext}", dpi=200, facecolor=SURF)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURF)
for ax, ds in zip(axes, DS):
    style(ax, f"{ds.upper()} — linear probe along pretraining")
    for key, lab, c, m, ls in (("ours", "Band power (ours)", "#2a78d6", "o", "-"),
                               ("raw", "Raw-waveform recon", "#eb6834", "s", "--")):
        ep = [e for e in (4, 8, 16) if get(ds, f"{key}_ep{e}") is not None]
        ax.plot(ep, [get(ds, f"{key}_ep{e}") for e in ep], color=c, ls=ls, lw=2.4,
                marker=m, ms=8, mec=SURF, mew=2, label=lab)
    ax.set_xticks([4, 8, 16]); ax.set_xlabel("Pretraining epoch (same data, same steps)")
    ax.set_ylabel("Balanced accuracy")
axes[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(PAPER / f"figures/lp_compute.{ext}", dpi=200, facecolor=SURF)
print("wrote", PAPER / "tables_linear_probe.tex", "and 2 figures")
