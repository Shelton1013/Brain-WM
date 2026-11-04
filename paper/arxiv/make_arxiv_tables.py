"""Tables + figures for the arXiv version, generated from the result JSONs only.

    python paper/arxiv/make_arxiv_tables.py
reads   results/arxiv_data/  (copied from the server: eval_results/*, model/*ft30s*)
writes  paper/arxiv/tables/*.tex, paper/arxiv/figures/*.pdf
Every number in the tables comes from these files, except rows marked with a
dagger, which are quoted from the cited papers (constants below).

Only the theory-aligned target is reported: classical bands (0.5-4/4-8/8-13/13-30/
30-45 Hz), per-patch periodogram, log P. Runs of an earlier implementation (narrow
Hann-filter bands, log(1+P)) are kept in the result files but not tabulated.
In every table the best value of each column among comparable rows is bold
(rows quoted under a different protocol are excluded from the comparison).
Results of runs still in progress are included automatically once present.
"""
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/arxiv_data"
OUT = Path(__file__).resolve().parent
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

DS = ("isruc", "hmc")
# subject-level bootstrap (paper/arxiv/bootstrap_metrics.py on the saved test predictions):
# BOOT[ds]["models"][model][metric] = {"point", "se"}
_BF = ROOT / "results/bootstrap/boot.json"
BOOT = json.load(open(_BF)) if _BF.exists() else {}
BOOT_KEY = {"m": "specbram_s42"}          # main released checkpoint vs. single external ckpts


def boot(t, model, m):
    """(point, (lo, hi)) with the 95% bootstrap CI, or None."""
    r = BOOT.get(t, {}).get("models", {}).get(BOOT_KEY.get(model, model), {}).get(m)
    return (r["point"], (r["lo"], r["hi"])) if r else None


def boot_seeds(t, m):
    """SpecBraM over its three pretraining seeds: (mean, std)."""
    runs = BOOT.get(t, {}).get("models", {}).get("specbram", {}).get("per_run", [])
    return (st.mean([r[m] for r in runs]), st.stdev([r[m] for r in runs])) if len(runs) > 1 else None
FR = ("0.01", "0.05", "0.10", "0.25", "0.50", "1.00")
SEEDS = (42, 43, 44)
M3 = ("balanced_accuracy", "cohen_kappa", "weighted_f1")

# ------------------------------------------------------------------ loading: sleep LP
LP = {d: {} for d in DS}
for fj in glob.glob(str(D / "eval_results/linear_probe_v2/lp2_*.json")):
    j = json.load(open(fj))
    LP[j["dataset"]].update(j["results"])

# handcrafted spectral features under the same strict probe (eval_linear_probe_handcrafted.py);
# mean over the three 10-s windows ("avg3"), exactly as our encoder is probed
HC = {d: {} for d in DS}
for fj in glob.glob(str(D / "eval_results/linear_probe_v2/lp2hc_*.json")):
    j = json.load(open(fj))
    HC[j["dataset"]] = j["results"]
HCMODE = "avg3"


def hc(ds, kind, m="balanced_accuracy"):
    v = HC[ds].get(f"hc_{kind}", {}).get(HCMODE)
    return None if v is None else v["test"][m]


def hcle(ds, kind, fr):
    v = HC[ds].get(f"hc_{kind}", {}).get(HCMODE)
    return None if v is None else v["label_eff"][fr]["balanced_accuracy"][0]


# seed-indexed probe keys of each pretrained model family
FAM = {
    "d1": {s: f"linfftstdlog_s{s}_ep16" for s in SEEDS},          # main target + SIGReg 0.05 (old)
    # main model: no SIGReg; seed 42 re-pretrained on 4 GPUs x 128 like seeds 43/44
    "m": {42: "g_s42_ep16", 43: "linfftstdlog_nosig_s43_ep16", 44: "linfftstdlog_nosig_s44_ep16"},
    "d2n": {42: "linfftstdmt_nosig_s42_ep16", 43: "mt_s43_ep16", 44: "mt_s44_ep16"},  # multitaper
    "g": {s: f"linfftstdlog_nosig_s{s}_ep16" for s in SEEDS},     # main, no SIGReg
    "d2": {s: f"linfftstdmt_s{s}_ep16" for s in SEEDS},           # main, multitaper
    "nd": {42: "nd_s42_ep16"},                                     # main without delta band
    "linraw_sig": {42: "linear_rawrecon_ep16", 43: "linraw_s43_ep16", 44: "linraw_s44_ep16"},
    "linraw": {s: f"lr_s{s}_ep16" for s in SEEDS},                # raw recon, no SIGReg
    "fbd1": {s: f"fbd1_s{s}_ep16" for s in SEEDS},                # main target, filterbank tok
    "fbraw_sig": {42: "raw_ep16", 43: "raw_s43_ep16", 44: "raw_s44_ep16"},
    "fbraw": {s: f"fr_s{s}_ep16" for s in SEEDS},                 # filterbank raw, no SIGReg
    "bandrecon_sig": {s: ("bandrecon_ep16" if s == 42 else f"bandrecon_s{s}_ep16") for s in SEEDS},
    "bandrecon": {s: f"br_s{s}_ep16" for s in SEEDS},             # band-waveform, no SIGReg
    "sbp": {s: f"sbp_s{s}_ep16" for s in SEEDS},                  # 4.2M params, band power
    "srr": {s: f"srr_s{s}_ep16" for s in SEEDS},                  # 4.2M params, raw recon
}
MAIN = "m"


def has(ds, key):
    return key in LP[ds]


def lp(ds, key, m="balanced_accuracy"):
    return LP[ds][key]["avg3"]["test"][m]


def le(ds, key, fr):
    return LP[ds][key]["avg3"]["label_eff"][fr]["balanced_accuracy"][0]


def seeds_of(name):
    """seeds whose results exist for both datasets."""
    return [s for s, k in FAM[name].items() if all(has(ds, k) for ds in DS)]


def fam(ds, name, getter):
    return [getter(ds, FAM[name][s]) for s in seeds_of(name)]


def write(name, body):
    (OUT / "tables" / f"{name}.tex").write_text(
        "% auto-generated by make_arxiv_tables.py -- do not edit\n" + body, encoding="utf-8")


# ------------------------------------------------------------------ table rendering with bold
def agg(xs):
    """list of seed values -> (mean, std) ; single value -> (value, None)."""
    if xs is None:
        return None
    if isinstance(xs, (int, float)):
        return (float(xs), None)
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else None)


def render(rows, dp):
    """rows: list of "\\midrule"-strings or dicts {label, cells:[(m,s)|None|str], cmp:bool}.
    dp: decimals per column. Bold = column maximum among rows with cmp=True."""
    ncol = len(dp)
    best = [None] * ncol
    for r in rows:
        if isinstance(r, dict) and r.get("cmp", True):
            for j, c in enumerate(r["cells"]):
                if isinstance(c, tuple):
                    v = round(c[0], dp[j])
                    best[j] = v if best[j] is None else max(best[j], v)
    out = []
    for r in rows:
        if isinstance(r, str):
            out.append(r)
            continue
        cells = []
        for j, c in enumerate(r["cells"]):
            if c is None:
                cells.append("--")
            elif isinstance(c, str):
                cells.append(c)
            else:
                m, s = c
                txt = f"{m:.{dp[j]}f}"
                if r.get("cmp", True) and best[j] is not None and round(m, dp[j]) == best[j]:
                    txt = rf"\textbf{{{txt}}}"
                if isinstance(s, tuple):
                    cells.append(rf"\ci{{{txt}}}{{{s[0]:.{dp[j]}f}}}{{{s[1]:.{dp[j]}f}}}")
                else:
                    cells.append(rf"\val{{{txt}}}{{{s:.{dp[j]}f}}}" if s is not None else txt)
        out.append(f"{r['label']} & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def R(label, cells, cmp=True):
    return {"label": label, "cells": cells, "cmp": cmp}


# ------------------------------------------------------------------ T: LP vs FMs (sleep)
# REVE is measured with our probe; its readout (mean pooling, or its own pretrained
# attention pooling -- both frozen and label-free) is chosen on the validation split
REVE_SIZES = (("base", "REVE-Base", r"${\sim}60$k"),)


def reve_sleep_key(ds, size):
    ks = [f"reve_{size}_{p}" for p in ("mean", "attn") if f"reve_{size}_{p}" in LP[ds]]
    return max(ks, key=lambda k: LP[ds][k]["avg3"]["val_ba"]) if ks else None


rows = []
for key, name, hrs in (("random_init", "Random init (our arch.)", "0"),
                       ("biot", "BIOT (six-dataset)", "mixed"),
                       ("biot_sleep", "BIOT (SHHS+PREST)", "mixed"),
                       ("cbramod", "CBraMod", r"${\sim}9$k"),
                       ("labram", "LaBraM-Base", r"${\sim}2.5$k"),
                       ("csbrain", "CSBrain", r"${\sim}9$k")):
    if all(key in LP[ds] for ds in DS):
        rows.append(R(f"{name} & {hrs}", [agg(lp(ds, key, m)) for ds in DS for m in M3]))
for size, name, hrs in REVE_SIZES:
    if all(reve_sleep_key(ds, size) for ds in DS):
        rows.append(R(f"{name} & {hrs}", [agg(lp(ds, reve_sleep_key(ds, size), m)) for ds in DS for m in M3]))
if False:
    rows.append(r"\midrule")
    for kind, name in (("basic", "Handcrafted: 5-band log/relative power"),
                       ("rich", "Handcrafted: rich spectral set$^\\ast$")):
        rows.append(R(f"{name} & --", [agg(hc(ds, kind, m)) for ds in DS for m in M3]))
rows.append(r"\midrule")
rows.append(R(r"\textbf{SpecBraM} & $2{,}388$",
              [agg(fam(ds, MAIN, lambda d, k: lp(d, k, m))) for ds in DS for m in M3]))
write("lp_main", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Strict linear probe on sleep staging.} Frozen encoder, mean-pooled
features, logistic regression with $C$ selected on the validation split, test scored once
(deterministic); a 30-s epoch is the mean of its three 10-s window features. Every model we
ran sees the identical epochs and splits, each in its own input convention
(Sec.~\ref{sec:lpproto}). Our rows: mean${\pm}$std over three pretraining seeds. Bold: best
per column. REVE: its readout (our mean pooling or its own pretrained attention pooling, both
frozen and label-free) is selected on the validation split, like $C$.""" + (r""" Handcrafted rows: features averaged over the three 10-s windows of each
epoch (as for every encoder; better than whole-epoch features), same samples and probe;
$^\ast$1-Hz log PSD (0.5--45\,Hz), relative power in six bands, Hjorth parameters, spectral
entropy, 95\% spectral edge, line length and kurtosis per channel.""" if False else "") + r"""}
\label{tab:lpmain}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c ccc ccc}
\toprule
& & \multicolumn{3}{c}{ISRUC} & \multicolumn{3}{c}{HMC}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Model & Pretrain (h) & BA & $\kappa$ & F1 & BA & $\kappa$ & F1\\
\midrule
""" + render(rows, [4] * 6) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")


# ------------------------------------------------------------------ T: target ablations
def four(name):
    """BA 100% ISRUC/HMC, BA 1% ISRUC/HMC (mean, std over available seeds)."""
    return ([agg(fam(ds, name, lp)) for ds in DS]
            + [agg(fam(ds, name, lambda d, k: le(d, k, "0.01"))) for ds in DS])


def gap(name, base):
    if not seeds_of(name) or not seeds_of(base):
        return [None, None]
    return [f"{(st.mean(fam(ds, name, lp)) - st.mean(fam(ds, base, lp))) * 100:+.1f}" for ds in DS]


def nseed(name):
    n = len(seeds_of(name))
    return "" if n == 3 else rf" ({n} seed{'s' if n > 1 else ''})"


rows = [r"\multicolumn{7}{l}{\emph{Linear patch embedding, 51.3M parameters (main model)}}\\"]
for k, nm in ((MAIN, r"\textbf{Band power} (ours)"),
              ("linraw", "Raw-waveform reconstruction")):
    if seeds_of(k):
        rows.append(R(nm + nseed(k), four(k) + (["--", "--"] if k == MAIN else gap(k, MAIN))))
if seeds_of("sbp"):
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{7}{l}{\emph{Linear patch embedding, 4.2M parameters (CBraMod's size)}}\\")
    for k, nm in (("sbp", "Band power (ours)"), ("srr", "Raw-waveform reconstruction")):
        if seeds_of(k):
            rows.append(R(nm + nseed(k), four(k) + (["--", "--"] if k == "sbp" else gap(k, "sbp"))))
rows.append(r"\midrule")
rows.append(r"\multicolumn{7}{l}{\emph{Filterbank tokenizer}}\\")
fbbase = "fbd1" if seeds_of("fbd1") else None
for k, nm in (("fbd1", r"Band power (ours)"),
              ("fbraw", "Raw-waveform reconstruction"),
              ("bandrecon", "Band-waveform reconstruction")):
    if seeds_of(k):
        rows.append(R(nm + nseed(k), four(k) + (["--", "--"] if k == fbbase or not fbbase else gap(k, fbbase))))
write("ablation", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Pretext-target study} at matched backbone, data ($2{,}388$\,h), steps (16
epochs) and launcher; only the target (or the tokenizer) changes. Strict linear probe,
mean${\pm}$std over three pretraining seeds unless stated. ``1\%'': balanced accuracy with
$1\%$ of the labelled training epochs (mean of five fixed subsets). $\Delta$: difference to the
band-power row of the same block, in balanced-accuracy points; paired by pretraining seed, band
power exceeds raw-waveform reconstruction on every seed in every block. Bold: best per column.}
\label{tab:ablation}
\begin{tabular}{l cc cc cc}
\toprule
& \multicolumn{2}{c}{BA (100\% labels)} & \multicolumn{2}{c}{BA (1\% labels)} & \multicolumn{2}{c}{$\Delta$ BA (pts)}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Target & ISRUC & HMC & ISRUC & HMC & ISRUC & HMC\\
\midrule
""" + render(rows, [4, 4, 3, 3, 1, 1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ------------------------------------------------------------------ T (appendix): target variants
rows = []
for k, nm in ((MAIN, r"\textbf{Band power} (main: periodogram, five classical bands)"),
              ("d2n", r"\quad multitaper estimate (3 DPSS tapers, $NW{=}2$)")):
    if seeds_of(k):
        rows.append(R(nm + nseed(k), four(k) + (["--", "--"] if k == MAIN else gap(k, MAIN))))
write("variants", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Spectral estimator of the band-power target} (linear patch embedding; everything else as
the main model). Strict linear probe on sleep staging, mean${\pm}$std over three pretraining
seeds unless stated; columns as Table~\ref{tab:ablation}. Bold: best per column.}
\label{tab:variants}
\begin{tabular}{l cc cc cc}
\toprule
& \multicolumn{2}{c}{BA (100\% labels)} & \multicolumn{2}{c}{BA (1\% labels)} & \multicolumn{2}{c}{$\Delta$ BA (pts)}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Target & ISRUC & HMC & ISRUC & HMC & ISRUC & HMC\\
\midrule
""" + render(rows, [4, 4, 3, 3, 1, 1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ------------------------------------------------------------------ T: paired band power vs raw
def paired(bp, raw, getter, dp):
    ss = [s for s in seeds_of(bp) if s in seeds_of(raw)]
    cells = []
    for ds in DS:
        a = [getter(ds, FAM[bp][s]) for s in ss]
        b = [getter(ds, FAM[raw][s]) for s in ss]
        g = [(x - y) * 100 for x, y in zip(a, b)]
        sd = rf"\,{{\scriptsize$\pm${st.stdev(g):.1f}}}" if len(g) > 1 else ""
        cells += [agg(a), agg(b), f"{st.mean(g):+.1f}{sd}", f"{sum(x > 0 for x in g)}/{len(g)}"]
    return cells


rows = []
for lab, getter, dp in (("100\\% labels", lp, 4), ("1\\% labels", lambda d, k: le(d, k, "0.01"), 3)):
    for tok, bp, raw in (("Linear", MAIN, "linraw"), ("Filterbank", "fbd1", "fbraw")):
        if seeds_of(bp):
            rows.append(R(f"{tok}, {lab}" + nseed(bp), paired(bp, raw, getter, dp), cmp=False))
    rows.append(r"\midrule")
rows = rows[:-1]
write("twobytwo", r"""\begin{table}[t]\centering\small
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Band power (BP) vs.\ raw-waveform reconstruction (Raw)}, paired by
pretraining seed, under both tokenizers. $\Delta$: mean${\pm}$std of the same-seed difference in
balanced-accuracy points; ``wins'': seeds with BP $>$ Raw.}
\label{tab:twobytwo}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l cccc cccc}
\toprule
& \multicolumn{4}{c}{ISRUC} & \multicolumn{4}{c}{HMC}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Tokenizer & BP & Raw & $\Delta$ & wins & BP & Raw & $\Delta$ & wins\\
\midrule
""" + render(rows, [4, 4, 0, 0, 4, 4, 0, 0]) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
# (paired rows are not bolded: each row compares two models)

# ------------------------------------------------------------------ T: along pretraining (main)
CURVE_EP = (1, 2, 4, 8, 12, 16)


def curve_key(kind, e):
    if e == 16:
        return FAM[MAIN][42] if kind == "bp" else FAM["linraw"][42]
    return f"m_s42_ep{e}" if kind == "bp" else f"lr_s42_ep{e}"


HAS_CURVE = all(has(ds, curve_key(k, e)) for ds in DS for k in ("bp", "raw") for e in CURVE_EP)
if HAS_CURVE:
    rows = [R(f"{e}", [agg(lp(ds, curve_key(k, e))) for ds in DS for k in ("bp", "raw")]) for e in CURVE_EP]
    # bold per epoch row is more useful here: mark the better target at each epoch
    body = []
    for e in CURVE_EP:
        cells = []
        for ds in DS:
            b, r = lp(ds, curve_key("bp", e)), lp(ds, curve_key("raw", e))
            cells += [rf"\textbf{{{b:.4f}}}" if b >= r else f"{b:.4f}",
                      rf"\textbf{{{r:.4f}}}" if r > b else f"{r:.4f}"]
        body.append(f"{e} & " + " & ".join(cells) + r" \\")
    write("compute", r"""\begin{table}[t]\centering\small
\caption{\textbf{Along pretraining} (linear patch embedding, seed 42, strict linear probe BA;
bold: better target at that epoch).}
\label{tab:compute}
\begin{tabular}{c cc cc}
\toprule
& \multicolumn{2}{c}{ISRUC} & \multicolumn{2}{c}{HMC}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Epoch & BP & Raw & BP & Raw\\
\midrule
""" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
else:
    (OUT / "tables" / "compute.tex").unlink(missing_ok=True)

# ------------------------------------------------------------------ T: label efficiency
LEM = [("biot", "BIOT"), ("cbramod", "CBraMod"), ("labram", "LaBraM-Base"), ("csbrain", "CSBrain")]
rows = []
for ds in DS:
    rows.append(r"\multicolumn{7}{l}{\emph{" + ds.upper() + r"}}\\")
    blk = [R(nm, [agg(le(ds, k, fr)) for fr in FR]) for k, nm in LEM if k in LP[ds]]
    for size, name, _ in REVE_SIZES:
        if reve_sleep_key(ds, size):
            blk.append(R(name, [agg(le(ds, reve_sleep_key(ds, size), fr)) for fr in FR]))
    if False:  # handcrafted -> appendix table
        blk += [R(nm, [agg(hcle(ds, kind, fr)) for fr in FR])
                for kind, nm in (("basic", "Handcrafted, 5-band"), ("rich", "Handcrafted, rich"))]
    blk.append(R(r"\textbf{SpecBraM}", [(le(ds, FAM[MAIN][42], fr), None) for fr in FR]))
    rows.append(render(blk, [3] * 6))
write("labeleff", r"""\begin{table}[t]\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Label efficiency} (strict linear probe, balanced accuracy vs.\ fraction of
labelled training epochs; ${<}100\%$: mean of five fixed random subsets shared by all
models). SpecBraM: main checkpoint. Bold: best per column.}
\label{tab:labeleff}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l cccccc}
\toprule
Model & 1\% & 5\% & 10\% & 25\% & 50\% & 100\%\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------------ T: 30-s fine-tuning
FT = {}
for fj in glob.glob(str(D / "model/*/checkpoint_ep16_*_ft30s_rep0.json")):
    run = Path(fj).parent.name
    ds = "isruc" if "_isruc_" in Path(fj).name else "hmc"
    FT[(run, ds)] = json.load(open(fj))["jepa_finetune"]


MAIN_FT = {42: "abl_linear_fftstdlog_nosig4g_s42_sub2100", 43: "abl_linear_fftstdlog_nosig_s43_sub2100",
           44: "abl_linear_fftstdlog_nosig_s44_sub2100"}


def ftv(tgt, ds, m):
    if tgt == "main":
        return [FT[(MAIN_FT[s], ds)][m]["mean"] for s in SEEDS if (MAIN_FT[s], ds) in FT]
    if tgt == "rawnosig":
        runs = [f"abl_linear_rawrecon_nosig_s{s}_sub2100" for s in SEEDS]
        return [FT[(r, ds)][m]["mean"] for r in runs if (r, ds) in FT]
    out = []
    for s in SEEDS:
        run = (f"abl_linear_{tgt}_sub2100" if s == 42 and tgt != "fftstdlog"
               else f"abl_linear_{tgt}_s{s}_sub2100")
        out.append(FT[(run, ds)][m]["mean"])
    return out


PAPER_FT = [  # (group, name, hours, ISRUC BA,k,F1 (val/std), HMC ...) quoted from CSBrain / REVE
    ("task", r"EEGNet~\cite{lawhern2018eegnet}", "--", ["0.7154/.0121", "0.7040/.0173", "0.7513/.0124", "0.6534/.0122", "0.5886/.0201", "0.6536/.0168"]),
    ("task", r"EEGConformer~\cite{song2023conformer}", "--", ["0.7400/.0133", "0.7143/.0162", "0.7634/.0151", "0.7149/.0086", "0.6432/.0055", "0.7080/.0039"]),
    ("task", r"SPaRCNet~\cite{jing2023sparcnet}", "--", ["0.7487/.0075", "0.7097/.0132", "0.7624/.0092", "0.4756/.1109", "0.3147/.1315", "0.4108/.1310"]),
    ("task", r"ContraWR~\cite{yang2023contrawr}", "--", ["0.7402/.0126", "0.7178/.0156", "0.7610/.0137", "0.4242/.0541", "0.2340/.0554", "0.2987/.0288"]),
    ("task", r"CNN-Transformer~\cite{peh2022cnntransformer}", "--", ["0.7363/.0087", "0.7129/.0121", "0.7719/.0153", "0.6573/.0141", "0.5961/.0105", "0.6896/.0065"]),
    ("task", r"FFCL~\cite{li2022ffcl}", "--", ["0.7277/.0182", "0.7016/.0292", "0.7614/.0197", "0.4427/.0702", "0.2542/.0654", "0.2902/.0485"]),
    ("task", r"ST-Transformer~\cite{song2021sttransformer}", "--", ["0.7381/.0205", "0.7013/.0352", "0.7681/.0175", "0.2559/.0141", "0.0503/.0183", "0.1428/.0122"]),
    ("fm", r"BIOT~\cite{yang2023biot}", "mixed", ["0.7527/.0121", "0.7291/.0230", "0.7790/.0146", "0.6862/.0041", "0.6295/.0113", "0.7091/.0147"]),
    ("fm", r"LaBraM-Base~\cite{jiang2024labram}", r"${\sim}2.5$k", ["0.7633/.0102", "0.7231/.0182", "0.7810/.0133", "0.7277/.0101", "0.6813/.0053", "0.7454/.0027"]),
    ("fm", r"CBraMod~\cite{wang2025cbramod}", r"${\sim}9$k", ["0.7865/.0110", "0.7442/.0152", "0.8011/.0099", "0.7269/.0041", "0.6685/.0104", "0.7395/.0089"]),
    ("fm", r"CSBrain~\cite{csbrain}", r"${\sim}9$k", ["0.7925/.0030", "0.7406/.0102", "0.7990/.0091", "0.7345/.0047", "0.6818/.0046", "0.7506/.0042"]),
    ("fm", r"REVE~\cite{reve2025}", r"${\sim}60$k", ["0.7819/", "0.7500/", "0.8005/", "--", "--", "--"]),
]


def pv(s):
    if s == "--":
        return None
    a, b = s.split("/")
    return (float(a), float("0" + b) if b else None)


rows = [r"\multicolumn{8}{l}{\emph{Task-specific models}}\\"]
rows += [R(f"{n} & {h}", [pv(x) for x in v]) for g, n, h, v in PAPER_FT if g == "task"]
rows += [r"\midrule", r"\multicolumn{8}{l}{\emph{Foundation models}}\\"]
rows += [R(f"{n} & {h}", [pv(x) for x in v]) for g, n, h, v in PAPER_FT if g == "fm"]
rows.append(r"\midrule")
for tgt, nm in (("rawnosig","Raw-waveform recon.\\ (same arch.)"), ("main", r"\textbf{SpecBraM}")):
    rows.append(R(f"{nm} & $2{{,}}388$", [agg(ftv(tgt, ds, m)) for ds in DS for m in M3]))
write("ft30", r"""\begin{table*}[t]\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Sleep staging, fine-tuned} (5-class, 30-s epochs in sequences of 20, one
shared 1-layer sequence transformer; identical subject-disjoint splits as
\citet{csbrain}). Baselines as reported by \citet{csbrain} (REVE: its paper, ISRUC only).
Our rows: mean${\pm}$std over three \emph{pretraining} seeds (one fine-tuning run each), so
the spread includes pretraining variance, unlike the baselines' fine-tuning-seed spread.
Bold: best per column.}
\label{tab:ft30}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c ccc ccc}
\toprule
& & \multicolumn{3}{c}{ISRUC} & \multicolumn{3}{c}{HMC}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Method & Pretrain (h) & BA & $\kappa$ & Weighted F1 & BA & $\kappa$ & Weighted F1\\
\midrule
""" + render(rows, [4] * 6) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")

# ------------------------------------------------------------------ loading: other tasks
TK = {}
LPT = D / "eval_results/linear_probe_tasks"


def _load(p):
    return json.load(open(p))["results"] if Path(p).exists() else {}


for t in ("ma", "mumtaz", "faced", "physio"):
    TK[t] = _load(LPT / f"lp_{t}.json")
    TK[t].update(_load(LPT / f"lp_{t}_d1.json"))
for t in ("tuab", "tuev"):
    TK[t] = {}
    for g in ("lin", "base", "d1"):
        TK[t].update(_load(LPT / f"lp_{t}_{g}.json"))
for t in TK:
    TK[t].update(_load(LPT / f"lp_{t}_hc.json"))                  # hc_basic / hc_rich
    TK[t].update(_load(LPT / f"lp_{t}_reve.json"))                # reve_{base,large}_{mean,attn}
    TK[t].update(_load(LPT / f"lp_{t}_csbrain.json"))
    for k in ["nd_s42"] + [f"{f}_s{s}" for f in ("g", "fbd1", "lr") for s in SEEDS]:
        TK[t].update(_load(LPT / f"lp_{t}_{k}.json"))
TFAM = {"d1": {s: f"d1_s{s}" for s in SEEDS}, "m": {s: f"g_s{s}" for s in SEEDS},
        "linraw_sig": {s: f"linraw_s{s}" for s in SEEDS}, "linraw": {s: f"lr_s{s}" for s in SEEDS}}


def tk(t, key, m):
    return TK[t][key]["test"][m] if key in TK[t] else None


def tkf(t, k, m):
    return agg([tk(t, TFAM[k][s], m) for s in SEEDS])


BIN = ("balanced_accuracy", "auc_pr", "auroc")
MUL = ("balanced_accuracy", "cohen_kappa", "weighted_f1")
OURS = [("m", r"\textbf{SpecBraM}"), ("linraw", "Raw-waveform recon.\\ (same arch.)")]
EXT = [("cbramod", "CBraMod"), ("labram", "LaBraM-Base"),
       ("csbrain", "CSBrain"), ("reve_base", "REVE-Base"),
       ("random_init", "Random init (our arch.)")]


def reve_task_key(t, base):
    ks = [f"{base}_{p}" for p in ("mean", "attn") if f"{base}_{p}" in TK[t]]
    return max(ks, key=lambda k: TK[t][k]["val_ba"]) if ks else base


def task_rows(spec):
    """spec: [(task, metrics), ...] -> comparable rows."""
    rows = [R(nm, [tkf(t, k, m) for t, ms_ in spec for m in ms_]) for k, nm in OURS]
    rows.append(r"\midrule")
    for k, nm in EXT:
        cells = [agg(tk(t, reve_task_key(t, k) if k.startswith("reve") else k, m))
                 for t, ms_ in spec for m in ms_]
        if any(c is not None for c in cells):
            rows.append(R(nm, cells))
    return rows


rows = task_rows([("ma", BIN), ("mumtaz", BIN)])
rows += [r"\midrule", R(r"\emph{CBraMod, fine-tuned}$^\dagger$",
                        [(v, None) for v in (0.726, 0.627, 0.791, 0.956, 0.992, 0.992)], cmp=False)]
write("tasks_binary", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{Strict linear probe beyond sleep: binary tasks.} Samples, channels and
subject-disjoint splits are produced by CBraMod's own preprocessing scripts, so every model
is probed on identical samples; external models use their publicly released weights. Mental
Arithmetic (MA) has only 4 test subjects. Bold: best per
column among linear probes. $^\dagger$CBraMod's \emph{fine-tuned} result from its paper, for
reference.}
\label{tab:taskbin}
\begin{tabular}{l ccc ccc}
\toprule
& \multicolumn{3}{c}{MA (stress, 2-class)} & \multicolumn{3}{c}{Mumtaz (MDD, 2-class)}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & BA & AUC-PR & AUROC & BA & AUC-PR & AUROC\\
\midrule
""" + render(rows, [3] * 6) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

rows = task_rows([("faced", MUL), ("physio", MUL)])
rows += [r"\midrule",
         R(r"\emph{CBraMod, fine-tuned}$^\dagger$", [(v, None) for v in (0.551, 0.504, 0.562, 0.642, 0.522, 0.643)], cmp=False)]
write("tasks_multi", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{Strict linear probe beyond sleep: multi-class tasks} (protocol as
Table~\ref{tab:taskbin}). FACED: 9-class emotion (chance $0.111$); PhysioNet-MI: 4-class
motor imagery (chance $0.25$). Bold: best per column among linear probes.
$^\dagger$CBraMod fine-tuned, from its paper.}
\label{tab:taskmulti}
\begin{tabular}{l ccc ccc}
\toprule
& \multicolumn{3}{c}{FACED (emotion, 9-class)} & \multicolumn{3}{c}{PhysioNet-MI (4-class)}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & BA & $\kappa$ & F1 & BA & $\kappa$ & F1\\
\midrule
""" + render(rows, [3] * 6) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

rows = task_rows([("tuab", BIN), ("tuev", MUL)])
rows += [r"\midrule", R(r"\emph{CBraMod, fine-tuned}$^\dagger$",
                        [(v, None) for v in (0.829, 0.926, 0.923, 0.667, 0.677, 0.834)], cmp=False)]
write("tuh", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{Strict linear probe on TUH clinical tasks.} TUAB: normal vs.\ abnormal
recording, 10-s windows; TUEV: six event classes (spike/sharp wave, GPED, PLED, eye movement,
artifact, background), 5-s windows around each annotated event. Samples, 16-channel bipolar
inputs (for CBraMod) and splits follow CBraMod's preprocessing scripts; every
model sees the same windows. Test patients are disjoint from training/validation; TUAB's
per-class preparation shares three patients between training and validation.
Both corpora are part of TUEG, on which all compared models
pretrain. Bold: best per column among linear probes. $^\dagger$CBraMod fine-tuned, from its paper.}
\label{tab:tuh}
\begin{tabular}{l ccc ccc}
\toprule
& \multicolumn{3}{c}{TUAB (abnormal, 2-class)} & \multicolumn{3}{c}{TUEV (events, 6-class)}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & BA & AUC-PR & AUROC & BA & $\kappa$ & F1\\
\midrule
""" + render(rows, [3] * 6) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ------------------------------------------------------------------ T: SEED-VIG
VIG = D / "eval_results/seedvig"
SV = {"0.5": _load(VIG / "lp_all.json"), "4": _load(VIG / "lp_all_hp4.json")}
SV["0.5"].update(_load(VIG / "lp_reve.json"))                     # raw REVE input, 0.5 Hz column
SV["0.5"].update(_load(VIG / "lp_csbrain.json"))
for b in ("0.5", "4"):
    SV[b].update(_load(VIG / f"lp_d1_bp{b}.json"))
    SV[b].update(_load(VIG / f"lp_nd_s42_bp{b}.json"))
    for s in SEEDS:
        SV[b].update(_load(VIG / f"lp_g_s{s}_bp{b}.json"))
        SV[b].update(_load(VIG / f"lp_lr_s{s}_bp{b}.json"))
SVFT = {}                                                          # 0.5-45 Hz FT only
for fj in glob.glob(str(VIG / "ft_lin*.json")) + glob.glob(str(VIG / "ft_d1*.json")) + glob.glob(str(VIG / "ft_g_s*.json")) + glob.glob(str(VIG / "ft_lr_s*.json")):
    SVFT.update(json.load(open(fj))["results"])


def svm(k, b, m):
    return SV[b][k]["test"][m] if k in SV[b] else None


rows = []
for k, nm in OURS:
    ks = [TFAM[k][s] for s in SEEDS]
    cells = [agg([svm(kk, b, m) for kk in ks]) for b in ("0.5", "4") for m in ("pearson", "r2")]
    have = [kk for kk in ks if kk in SVFT]
    cells += [agg([SVFT[kk][m] for kk in have]) if have else None for m in ("pearson", "r2")]
    rows.append(R(nm, cells))
if "nd_s42" in SV["0.5"]:
    rows.append(R(r"\quad SpecBraM without $\delta$-band target (1 seed)",
                  [agg(svm("nd_s42", b, m)) for b in ("0.5", "4") for m in ("pearson", "r2")] + [None, None]))
rows.append(r"\midrule")
for size, name, _ in REVE_SIZES:
    ks = [f"reve_{size}_{p}" for p in ("mean", "attn") if f"reve_{size}_{p}" in SV["0.5"]]
    if ks:
        k = max(ks, key=lambda kk: SV["0.5"][kk]["val"]["r2"])
        rows.append(R(f"{name}", [agg(svm(k, "0.5", m)) for m in ("pearson", "r2")] + [None] * 4))
for k, nm in (("labram", "LaBraM-Base"), ("cbramod", "CBraMod"), ("csbrain", "CSBrain")):
    if k in SV["0.5"]:
        rows.append(R(nm, [agg(svm(k, b, m)) for b in ("0.5", "4") for m in ("pearson", "r2")] + [None, None]))
write("seedvig", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{SEED-VIG vigilance regression (PERCLOS)} on the split stated in the CBraMod
paper (experiments 1--15/16--19/20--23 in subject order). Linear probe = frozen encoder +
ridge regression (regularization on validation $R^2$); FT = full fine-tuning (MSE, epoch
selected on validation $R^2$). Input band 0.5--45\,Hz (standard) or 4--45\,Hz (high-passed
probe input). Our rows: mean${\pm}$std over three pretraining seeds unless stated. Bold: best
per column.}
\label{tab:seedvig}
\begin{tabular}{l cc cc cc}
\toprule
& \multicolumn{2}{c}{LP, 0.5--45\,Hz} & \multicolumn{2}{c}{LP, 4--45\,Hz} & \multicolumn{2}{c}{FT, 0.5--45\,Hz}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Model & Pearson & $R^2$ & Pearson & $R^2$ & Pearson & $R^2$\\
\midrule
""" + render(rows, [3] * 6) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")



# ------------------------------------------------------------------ T: overview of the strict probe by task type
# rows: our model and the external foundation models only (no random init / own controls)
CATS = [("Sleep staging", [("isruc", "ISRUC"), ("hmc", "HMC")]),
        ("Clinical EEG", [("tuab", "TUAB"), ("tuev", "TUEV")]),
        ("Workload", [("ma", "MA")]),
        ("Depression", [("mumtaz", "Mumtaz")]),
        ("Emotion", [("faced", "FACED")]),
        ("Motor imagery", [("physio", "PhysioNet-MI")])]
OV_COLS = [t for _, ts in CATS for t in ts]


def ov_cell(model, t):
    if t in DS:
        if model == "m":
            return agg(fam(t, MAIN, lp))
        if model.startswith("reve"):
            k = reve_sleep_key(t, model.split("_")[1])
            return agg(lp(t, k)) if k else None
        return agg(lp(t, model)) if model in LP[t] else None
    if model == "m":
        return tkf(t, "m", "balanced_accuracy")
    k = reve_task_key(t, model) if model.startswith("reve") else model
    return agg(tk(t, k, "balanced_accuracy"))


OV_ROWS = [("m", r"\textbf{SpecBraM}", r"$2{,}388$\,h"), ("cbramod", "CBraMod", r"${\sim}9$k\,h"),
           ("labram", "LaBraM-Base", r"${\sim}2.5$k\,h"), ("biot", "BIOT", "mixed"),
           ("csbrain", "CSBrain", r"${\sim}9$k\,h"), ("reve_base", "REVE-Base", r"${\sim}60$k\,h")]
rows = [R(f"{nm} & {hrs}", [boot(t, k, "balanced_accuracy") or ov_cell(k, t) for t, _ in OV_COLS])
        for k, nm, hrs in OV_ROWS]
head1 = " & ".join(rf"\multicolumn{{{len(ts)}}}{{c}}{{{c}}}" for c, ts in CATS)
rules, col = [], 3
for c, ts in CATS:
    rules.append(rf"\cmidrule(lr){{{col}-{col + len(ts) - 1}}}")
    col += len(ts)
write("overview", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{Strict linear probe across task types} (balanced accuracy). Frozen encoder,
mean-pooled tokens, logistic regression with $C$ selected on the validation split, test scored
once; every model sees the identical samples and splits in its own input convention
(Sec.~\ref{sec:lpproto}), and every number is measured by us with the publicly released
weights. Every model is one checkpoint: SpecBraM is our main released checkpoint, the
baselines are their released ones. Brackets: 95\% confidence interval from 1{,}000 bootstrap
resamples of the test subjects (TUEV: recordings); the spread over our three pretraining seeds
is given in Appendix~\ref{app:fullmetrics}. REVE: readout selected on the validation split. BIOT's
released channel tokens cover only the sleep montage. Per-dataset results with Cohen's $\kappa$,
weighted F1 and AUROC/AUC-PR: Appendix~\ref{app:fullmetrics}. Bold: best per column.}
\label{tab:overview}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c """ + " ".join("c" * len(ts) for _, ts in CATS) + r"""}
\toprule
& & """ + head1 + r"""\\
""" + "".join(rules) + r"""
Model & Pretraining & """ + " & ".join(n for _, n in OV_COLS) + r"""\\
\midrule
""" + render(rows, [3] * 8) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")

# ------------------------------------------------------------------ T (appendix): full metrics, by task type
FULL_ROWS = [("m", r"\textbf{SpecBraM}"), ("cbramod", "CBraMod"), ("labram", "LaBraM-Base"),
             ("biot", "BIOT (six-dataset)"), ("biot_sleep", "BIOT (SHHS+PREST)"),
             ("csbrain", "CSBrain"), ("reve_base", "REVE-Base")]


def full_cell(model, t, m):
    if t in DS:
        if model == "m":
            return agg(fam(t, MAIN, lambda d, k: lp(d, k, m)))
        if model.startswith("reve"):
            k = reve_sleep_key(t, model.split("_")[1])
            return agg(lp(t, k, m)) if k else None
        return agg(lp(t, model, m)) if model in LP[t] else None
    if model == "m":
        return tkf(t, "m", m)
    k = reve_task_key(t, model) if model.startswith("reve") else model
    return agg(tk(t, k, m))


SHORTM = {"balanced_accuracy": "BA", "cohen_kappa": r"$\kappa$", "weighted_f1": "F1",
          "auc_pr": "AUC-PR", "auroc": "AUROC"}


def full_table(name, label, caption, groups):
    """groups: [(category, [(task, title, metrics)])]"""
    cols = [(t, m) for _, ts in groups for t, _, ms_ in ts for m in ms_]
    rows = []
    for k, nm in FULL_ROWS:
        cells = [full_cell(k, t, m) for t, m in cols]
        if any(c is not None for c in cells):
            rows.append(R(nm, cells))
    h_cat = " & ".join(rf"\multicolumn{{{sum(len(ms_) for _, _, ms_ in ts)}}}{{c}}{{{c}}}" for c, ts in groups)
    h_task = " & ".join(rf"\multicolumn{{{len(ms_)}}}{{c}}{{{ti}}}" for _, ts in groups for _, ti, ms_ in ts)
    rules, col = [], 2
    for _, ts in groups:
        for _, _, ms_ in ts:
            rules.append(rf"\cmidrule(lr){{{col}-{col + len(ms_) - 1}}}")
            col += len(ms_)
    write(name, r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{2.5pt}
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l """ + "c" * len(cols) + r"""}
\toprule
& """ + h_cat + r"""\\
& """ + h_task + r"""\\
""" + "".join(rules) + r"""
Model & """ + " & ".join(SHORTM[m] for _, m in cols) + r"""\\
\midrule
""" + render(rows, [4 if t in DS else 3 for t, _ in cols]) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")


# one table per dataset (REVE-style detailed results): every model, all metrics, bootstrap SE
DET_ROWS = [("m", r"\textbf{SpecBraM}"), ("cbramod", "CBraMod"), ("labram", "LaBraM-Base"),
            ("biot", "BIOT"), ("csbrain", "CSBrain"), ("reve_base", "REVE-Base")]
DET_M = {2: ("balanced_accuracy", "cohen_kappa", "weighted_f1", "auroc", "auc_pr"),
         0: ("balanced_accuracy", "cohen_kappa", "weighted_f1")}
DET_NAME = {"balanced_accuracy": "Balanced Acc.", "cohen_kappa": r"Cohen's $\kappa$",
            "weighted_f1": "Weighted F1", "auroc": "AUROC", "auc_pr": "AUC-PR"}
DETAIL_DS = (("isruc", "ISRUC"), ("hmc", "HMC"), ("tuab", "TUAB"), ("tuev", "TUEV"), ("ma", "MA"),
             ("mumtaz", "Mumtaz"), ("faced", "FACED"), ("physio", "PhysioNet-MI"))


def seed_std(t, m):
    """std over the three pretraining seeds of SpecBraM's score (no bootstrap)."""
    runs = BOOT.get(t, {}).get("models", {}).get("specbram", {}).get("per_run", [])
    return st.stdev([r[m] for r in runs]) if len(runs) > 1 else None


for t, title in DETAIL_DS:
    if t not in BOOT:
        print("no bootstrap yet:", t)
        continue
    nc = BOOT[t]["n_classes"]
    ms_ = DET_M[2] if nc == 2 else DET_M[0]
    dp = 3
    rows = []
    for k, nm in DET_ROWS:
        cells = [boot(t, k, m) for m in ms_]
        if any(cells):
            rows.append(R(nm, cells))
    rows.insert(1, R(r"\quad{\footnotesize three seeds, mean${\pm}$std}",
                     [boot_seeds(t, m) for m in ms_], cmp=False))
    unit = "recordings" if t == "tuev" else "subjects"
    env, box = ("table*", "") if nc == 2 else ("table", r"\resizebox{\columnwidth}{!}{%")
    write(f"detail_{t}", r"""\begin{""" + env + r"""}[""" + ("t" if nc == 2 else "h") + r"""]\centering\small
\setlength{\tabcolsep}{""" + ("8pt" if nc == 2 else "3pt") + r"""}
\caption{\textbf{Detailed results on """ + title + r"""} (strict linear probe). One checkpoint
per model, with the 95\% confidence interval from 1{,}000 bootstrap resamples of the
""" + f"{BOOT[t]['n_units']:,}".replace(",", "{,}") + " test " + unit + r""". Below SpecBraM's main
checkpoint: mean${\pm}$std over its three pretraining seeds. Bold: best per column.}
\label{tab:det_""" + t + r"""}
""" + box + r"""
\begin{tabular}{l """ + "c" * len(ms_) + r"""}
\toprule
Model & """ + " & ".join(DET_NAME[m] for m in ms_) + r"""\\
\midrule
""" + render(rows, [dp] * len(ms_)) + "\n\\bottomrule\n\\end{tabular}" + ("}" if box else "")
          + "\n\\end{" + env + "}\n")

# ------------------------------------------------------------------ T (appendix): target effect beyond sleep
rows = []
for k, nm in (("m", r"\textbf{Band power} (ours)"), ("linraw", "Raw-waveform reconstruction")):
    rows.append(R(nm, [tkf(t, k, "balanced_accuracy") for t, _ in OV_COLS if t not in DS]))
write("target_tasks", r"""\begin{table}[t]\centering\small
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Band power vs.\ raw-waveform reconstruction beyond sleep} (same architecture,
data and steps; strict linear probe, balanced accuracy, mean${\pm}$std over three pretraining
seeds). Bold: best per column.}
\label{tab:target_tasks}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l """ + "c" * 6 + r"""}
\toprule
Target & """ + " & ".join(n for t, n in OV_COLS if t not in DS) + r"""\\
\midrule
""" + render(rows, [3] * 6) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------------ T: label-free clustering
CL = D / "eval_results/cluster"


def cl(ds, m):
    return _load(CL / f"cl_{ds}_{m}.json")


rows = []
for fam_, nm in (("m", r"\textbf{SpecBraM}"), ("lr", "Raw-waveform recon.\\ (same arch.)")):
    cells = []
    for ds in DS:
        r = cl(ds, "specbram")
        ks = [k for k in r if k == fam_ + "_s42"]
        cells += [agg([r[k]["test"][q][0] for k in ks]) if ks else None for q in ("nmi", "ari")]
    rows.append(R(nm, cells))
rows.append(r"\midrule")
for m, nm in (("cbramod", "CBraMod"), ("labram", "LaBraM-Base"), ("biot", "BIOT"),
              ("csbrain", "CSBrain"), ("reve_base", "REVE-Base")):
    cells = []
    for ds in DS:
        r = cl(ds, m)
        if not r:
            cells += [None, None]
            continue
        k = max(r, key=lambda kk: r[kk]["val"]["nmi"][0])
        cells += [agg(r[k]["test"][q][0]) for q in ("nmi", "ari")]
    rows.append(R(nm, cells))
write("cluster", r"""\begin{table}[t]\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{Clustering evaluation}: k-means ($k{=}5$) on frozen sleep features of the
test split (standardized, 64 PCA dimensions; cluster fitting uses no labels), agreement with the
expert stages. Mean over five k-means seeds; our rows: seed-42 checkpoints.
REVE: readout selected by validation NMI using labels. Bold: best per column.}
\label{tab:cluster}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l cc cc}
\toprule
& \multicolumn{2}{c}{ISRUC} & \multicolumn{2}{c}{HMC}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & NMI & ARI & NMI & ARI\\
\midrule
""" + render(rows, [3] * 4) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------------ T (appendix): handcrafted features
TCOLS = (("ma", "MA"), ("mumtaz", "Mumtaz"), ("faced", "FACED"), ("physio", "PhysioNet-MI"),
         ("tuab", "TUAB"), ("tuev", "TUEV"))
rows = []
for k, nm in (("m", r"\textbf{SpecBraM}"), ("linraw", "Raw-waveform recon.\\ (same arch.)")):
    cells = [agg(fam(ds, k, lp)) for ds in DS] + [agg(fam(ds, k, lambda d, kk: le(d, kk, "0.01"))) for ds in DS]
    cells += [tkf(t, k, "balanced_accuracy") for t, _ in TCOLS]
    rows.append(R(nm, cells))
rows.append(r"\midrule")
for kind, nm in (("basic", "Handcrafted: 5-band log/relative power"), ("rich", "Handcrafted: rich spectral set")):
    cells = [agg(hc(ds, kind)) for ds in DS] + [agg(hcle(ds, kind, "0.01")) for ds in DS]
    cells += [agg(tk(t, f"hc_{kind}", "balanced_accuracy")) for t, _ in TCOLS]
    rows.append(R(nm, cells))
write("handcrafted", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Handcrafted spectral features under the same strict linear probe} (balanced
accuracy). Basic: log absolute and relative power in the five classical bands per channel.
Rich: 1-Hz log PSD (0.5--45\,Hz), relative power in six bands, Hjorth parameters, spectral
entropy, 95\% spectral edge, line length and kurtosis per channel. Features are computed on our
input of each window (sleep: averaged over the three 10-s windows of an epoch, as for every
encoder). Bold: best per column.}
\label{tab:handcrafted}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l cc cc cccccc}
\toprule
& \multicolumn{2}{c}{Sleep, 100\%} & \multicolumn{2}{c}{Sleep, 1\%} & \multicolumn{6}{c}{Beyond sleep}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-11}
Features & ISRUC & HMC & ISRUC & HMC & """ + " & ".join(n for _, n in TCOLS) + r"""\\
\midrule
""" + render(rows, [4, 4, 3, 3] + [3] * 6) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")

# ------------------------------------------------------------------ T (appendix): confounds
LPC = {ds: {} for ds in DS}
for ds in DS:
    for fj in glob.glob(str(D / f"eval_results/linear_probe_v2/lp2chan_{ds}_*.json")):
        LPC[ds].update(json.load(open(fj))["results"])


def chan(ds, k):
    return LPC[ds][k]["avg3"]["test"]["balanced_accuracy"] if k in LPC[ds] else None


def both(a_fn, b_fn):
    return [a_fn("isruc"), b_fn("isruc"), a_fn("hmc"), b_fn("hmc")]


rows = [r"\multicolumn{5}{l}{\emph{Readout: (a) mean over all tokens (main protocol), (b) per-channel means}}\\"]
rows.append(R(r"\textbf{SpecBraM}", both(lambda ds: agg(fam(ds, MAIN, lp)),
                                         lambda ds: agg([chan(ds, f"m_s{s}") for s in SEEDS])), cmp=False))
rows.append(R("Raw-waveform recon.", both(lambda ds: agg(fam(ds, "linraw", lp)),
                                          lambda ds: agg([chan(ds, f"lr_s{s}") for s in SEEDS])), cmp=False))
for k, nm in (("cbramod", "CBraMod"), ("labram", "LaBraM-Base"), ("csbrain", "CSBrain")):
    if all(k in LP[ds] for ds in DS):
        rows.append(R(nm, both(lambda ds, k=k: agg(lp(ds, k)), lambda ds, k=k: agg(chan(ds, k))), cmp=False))
for size, nm in (("base", "REVE-Base"),):
    rows.append(R(nm, both(lambda ds, size=size: agg(lp(ds, reve_sleep_key(ds, size))),
                           lambda ds, size=size: agg(chan(ds, f"reve_{size}_chan"))), cmp=False))
rows.append(r"\midrule")
rows.append(r"\multicolumn{5}{l}{\emph{Input: (a) the model's own convention, (b) our per-recording normalization}}\\")
for k, nm in (("cbramod", "CBraMod"), ("labram", "LaBraM-Base")):
    rows.append(R(nm, both(lambda ds, k=k: agg(lp(ds, k)), lambda ds, k=k: agg(lp(ds, k + "_rn"))), cmp=False))
rows.append(r"\midrule")
rows.append(r"\multicolumn{5}{l}{\emph{Scale: (a) 51.3M parameters, (b) 4.2M parameters (CBraMod: 4.0M)}}\\")
for big, small, nm in ((MAIN, "sbp", "Band power"), ("linraw", "srr", "Raw-waveform recon.")):
    rows.append(R(nm, both(lambda ds, big=big: agg(fam(ds, big, lp)),
                           lambda ds, small=small: agg(fam(ds, small, lp)) if seeds_of(small) else None), cmp=False))
write("confounds", r"""\begin{table}[t]\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Factors besides the target in the comparison with external models} (strict
linear probe, balanced accuracy; our rows: mean${\pm}$std over pretraining seeds). Per-channel
readout: tokens averaged over time within each channel and concatenated; still a frozen encoder
and a logistic regression. Our normalization for CBraMod/LaBraM: per-recording median/IQR
scaling, rescaled to the dataset's typical amplitude.}
\label{tab:confounds}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l cc cc}
\toprule
& \multicolumn{2}{c}{ISRUC} & \multicolumn{2}{c}{HMC}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & (a) & (b) & (a) & (b)\\
\midrule
""" + render(rows, [4] * 4) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


# ------------------------------------------------------------------ T (appendix): fine-tuning beyond sleep
FTT = D / "eval_results/finetune_tasks"


def ftt(t, fam_, m="balanced_accuracy"):
    vals = []
    for s_ in SEEDS:
        fj = FTT / f"ft_{t}_{fam_}_s{s_}.json"
        if fj.exists():
            vals.append(json.load(open(fj))["results"][f"{fam_}_s{s_}"]["test"][m])
    return agg(vals) if vals else None


FT_TASKS = (("ma", "MA"), ("mumtaz", "Mumtaz"), ("faced", "FACED"), ("physio", "PhysioNet-MI"),
            ("tuab", "TUAB"), ("tuev", "TUEV"))
rows = [R(r"\textbf{SpecBraM}", [ftt(t, "m") for t, _ in FT_TASKS]),
        R("Raw-waveform recon.\\ (same arch.)", [ftt(t, "lr") for t, _ in FT_TASKS])]
n_lr = {t: len([s_ for s_ in SEEDS if (FTT / f"ft_{t}_lr_s{s_}.json").exists()]) for t, _ in FT_TASKS}
note = "".join(f" Raw recon.\\ on {n}: {k} seeds." for (t, n), k in zip(FT_TASKS, n_lr.values()) if k < 3) if False else ""
write("ft_tasks", r"""\begin{table}[t]\centering\small
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Full fine-tuning beyond sleep} (balanced accuracy, mean${\pm}$std over three
pretraining seeds). Same samples and splits as the linear probe; encoder and a linear head on
mean-pooled tokens are trained with layer-wise learning-rate decay, and the epoch is selected on
validation balanced accuracy. Bold: best per column.}
\label{tab:ft_tasks}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l """ + "c" * len(FT_TASKS) + r"""}
\toprule
Model & """ + " & ".join(n for _, n in FT_TASKS) + r"""\\
\midrule
""" + render(rows, [3] * len(FT_TASKS)) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------------ figure: label efficiency
GRID = "#e5e5e5"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
xs = [1, 5, 10, 25, 50, 100]
SER = [(MAIN, "SpecBraM (band power)", "#1f6f78", "o", "-", True),
       ("labram", "LaBraM-Base", "#5a8f3e", "^", ":", False),
       ("cbramod", "CBraMod", "#8a5ab0", "D", ":", False),
       ("biot", "BIOT", "#b08a2a", "v", ":", False),
       ("csbrain", "CSBrain", "#c0504d", "s", ":", False),
       ("reve_base", "REVE-Base", "#d08a2a", "P", "--", False),
       ("random_init", "Random init", "#999999", "x", ":", False)]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, ds in zip(axes, DS):
    for k, lab, c, mk, ls, is_fam in SER:
        if is_fam:
            m = [le(ds, FAM[k][42], fr) for fr in FR]
        else:
            kk = reve_sleep_key(ds, "base") if k == "reve_base" else k
            if kk not in LP[ds]:
                continue
            m = [le(ds, kk, fr) for fr in FR]
        ax.plot(xs, m, color=c, marker=mk, ls=ls, lw=1.8 if is_fam else 1.2, ms=4, label=lab)
    if False:  # handcrafted -> appendix table
        ax.plot(xs, [hcle(ds, "rich", fr) for fr in FR], color="#222222", marker="*", ls="-.",
                lw=1.4, ms=5, label="Handcrafted spectral (rich)")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x}%" for x in xs]); ax.minorticks_off()
    ax.set_title(ds.upper(), fontsize=9, loc="left"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_xlabel("labelled training epochs"); ax.set_ylabel("balanced accuracy")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(OUT / "figures/label_efficiency_lp.pdf")

# stale tables of the earlier implementation
for old in ("targets", "targetdef"):
    (OUT / "tables" / f"{old}.tex").unlink(missing_ok=True)
print("tables:", sorted(p.name for p in (OUT / "tables").glob("*.tex")),
      "| pending:", [k for k in ("fbd1", "nd") if len(seeds_of(k)) < len(FAM[k])],
      "| curve:", HAS_CURVE, "| hc tasks:", [t for t in TK if "hc_rich" in TK[t]],
      "| vig FT d1:", all(f"d1_s{s}" in SVFT for s in SEEDS))

# ------------------------------------------------------------------ appendix: per-dataset descriptions + tables
_CF = ROOT / "results/bootstrap/dataset_counts.json"
if _CF.exists():
    CNT = json.load(open(_CF))
    txt = (OUT / "appendix_detail.tex").read_text(encoding="utf-8")
    for t in ("isruc", "hmc", "tuab", "tuev", "ma", "mumtaz", "faced", "physio"):
        c = CNT[t]
        n = "/".join(f"{c[k]['n']:,}".replace(",", "{,}") for k in ("train", "val", "test"))
        txt = txt.replace(f"@{t.upper()}_N@", n)
    assert "@" not in txt, "unfilled count"
    write("appendix_detail", txt)

# ------------------------------------------------------------------ appendix: paired bootstrap differences
PAIR_M = [("cbramod", "CBraMod"), ("labram", "LaBraM-Base"), ("biot", "BIOT"), ("csbrain", "CSBrain"),
          ("reve_base", "REVE-Base")]
if all("paired" in BOOT.get(t, {}) for t, _ in DETAIL_DS):
    body = []
    for t, title in DETAIL_DS:
        cells = []
        for k, _ in PAIR_M:
            r = BOOT[t]["paired"].get(k, {}).get("balanced_accuracy")
            if r is None:
                cells.append("--")
                continue
            txt = f"{100 * r['diff']:+.1f}"
            if r["lo"] > 0 or r["hi"] < 0:                     # CI excludes zero
                txt = rf"\textbf{{{txt}}}"
            cells.append(rf"\ci{{{txt}}}{{{100 * r['lo']:+.2f}}}{{{100 * r['hi']:+.2f}}}")
        body.append(f"{title} & " + " & ".join(cells) + r" \\")
    write("paired", r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{6pt}
\caption{\textbf{Paired bootstrap of the difference} in balanced accuracy (points) between
SpecBraM's main checkpoint and each external model, computed on the same 1{,}000 resamples of
the test subjects (TUEV: recordings), with its nominal, unadjusted 95\% confidence interval.
Positive: SpecBraM better. Bold: the interval excludes zero; this is not an equivalence test.}
\label{tab:paired}
\begin{tabular}{l """ + "c" * len(PAIR_M) + r"""}
\toprule
Dataset & """ + " & ".join(n for _, n in PAIR_M) + r"""\\
\midrule
""" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ------------------------------------------------------------------ T: how much of its target each model predicts
_RF = D / "eval_results/target_r2.json"
if _RF.exists():
    RR = json.load(open(_RF))
    body = []
    for e in (1, 2, 4, 8, 12, 16):
        bp, raw = RR.get(f"bp_s42_ep{e}"), RR.get(f"raw_s42_ep{e}")
        if bp and raw:
            body.append(f"{e} & {bp['r2']:.3f} & {raw['r2']:.3f} \\\\")
    s3 = {k: agg([RR[f"{k}_s{x}_ep16"]["r2"] for x in SEEDS if f"{k}_s{x}_ep16" in RR]) for k in ("bp", "raw")}
    body.append(r"\midrule")
    body.append(r"16, three seeds & " + " & ".join(rf"\val{{{m:.3f}}}{{{sd:.3f}}}" for m, sd in
                                                    (s3["bp"], s3["raw"])) + r" \\")
    bp16 = RR["bp_s42_ep16"]
    band = " / ".join(f"{x:.2f}" for x in bp16["r2_band"])
    write("target_r2", r"""\begin{table}[t]\centering\small
\caption{\textbf{How much of its pretraining target each model predicts}: fraction of the
target variance explained, $R^2=1-\mathrm{MSE}/\mathrm{Var}$, on the masked patches of 4{,}000
held-out windows (TUAB evaluation split, not excluded from TUEG pretraining; masks as in
pretraining), along pretraining (seed 42) and at the end over three seeds.
BP: mean of the five per-band $R^2$ values; waveform: pooled-sample $R^2$.
These are achieved scores, not estimates of Bayes ceilings. Band power, per band
$\delta/\theta/\alpha/\beta/\gamma$ at epoch 16: """ + band + r""".}
\label{tab:target_r2}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{l cc}
\toprule
Epoch & Band power & Raw waveform\\
\midrule
""" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
