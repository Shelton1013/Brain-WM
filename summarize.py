"""Aggregate run_evals.sh JSON outputs into a table + LaTeX rows.

Mumtaz/Siena: one JSON per seed -> aggregate the per-seed means to mean±std over
seeds. ISRUC/HMC: one JSON per mode (frozen/ft) with 3 reps -> report its mean±std.
TUAB: one JSON, --mode both.

    python summarize.py <model_dir> [--ckpt checkpoint_ep2]

e.g. python summarize.py /home/pxieaf/home2/model/lejepa_v3_filter_full_e15 --ckpt checkpoint_ep2
"""
import argparse
import glob
import json
import os

import numpy as np

# task -> (metrics to show, [(mode, jepa_key, random_key, filename_glob, multiseed)])
JK, RK = "jepa", "random"
SPECS = {
    "Mumtaz": (["balanced_accuracy", "roc_auc", "pr_auc"], [
        ("frozen", "jepa_frozen", "random_frozen", "{c}_mumtaz_seed*.json", True),
        ("ft",     "jepa_finetune", "random_finetune", "{c}_mumtaz_seed*.json", True)]),
    "Siena": (["balanced_accuracy", "roc_auc", "pr_auc"], [
        ("frozen", "jepa_frozen", "random_frozen", "{c}_siena_seed*.json", True),
        ("ft",     "jepa_finetune", "random_finetune", "{c}_siena_seed*.json", True)]),
    "ISRUC": (["balanced_accuracy", "cohen_kappa", "weighted_f1"], [
        ("frozen", "jepa_finetune", "random_finetune", "{c}_isruc_frozen.json", False),
        ("ft",     "jepa_finetune", "random_finetune", "{c}_isruc_ft.json", False)]),
    "HMC": (["balanced_accuracy", "cohen_kappa", "weighted_f1"], [
        ("frozen", "jepa_finetune", "random_finetune", "{c}_hmc_frozen.json", False),
        ("ft",     "jepa_finetune", "random_finetune", "{c}_hmc_ft.json", False)]),
    "TUAB": (["balanced_accuracy", "roc_auc", "pr_auc"], [
        ("frozen", "jepa_frozen", "random_frozen", "{c}_tuab.json", False),
        ("ft",     "jepa_finetune", "random_finetune", "{c}_tuab.json", False)]),
}
SHORT = {"balanced_accuracy": "BA", "roc_auc": "ROC", "pr_auc": "PR",
         "cohen_kappa": "kappa", "weighted_f1": "wF1"}


def point(entry, metric):
    """Per-file point value of a metric (mean if aggregated, else scalar)."""
    v = entry.get(metric)
    if isinstance(v, dict):
        return v.get("mean", float("nan")), v.get("std", float("nan"))
    return (float(v), float("nan")) if v is not None else (float("nan"), float("nan"))


def aggregate(files, key, metrics, multiseed):
    """Return {metric: (mean, std, n)}."""
    res = {}
    per = {m: [] for m in metrics}
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        e = d.get(key)
        if not e:
            continue
        for m in metrics:
            mu, sd = point(e, m)
            if mu == mu:                       # not NaN
                per[m].append((mu, sd))
    for m in metrics:
        vals = per[m]
        if not vals:
            res[m] = (float("nan"), float("nan"), 0)
        elif multiseed:                        # std ACROSS seeds
            mus = [v[0] for v in vals]
            res[m] = (float(np.mean(mus)), float(np.std(mus)), len(mus))
        else:                                  # single file: its own mean±std
            res[m] = (vals[0][0], vals[0][1], 1)
    return res


def fmt(mu, sd, n=None):
    if mu != mu:
        return "  -  "
    return f"{mu:.3f}±{sd:.3f}" if sd == sd else f"{mu:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--ckpt", default="checkpoint_ep2",
                    help="checkpoint tag prefix on the json files")
    args = ap.parse_args()

    latex = []
    print(f"\n{'='*78}\n  {args.model_dir}  [{args.ckpt}]\n{'='*78}")
    for task, (metrics, modes) in SPECS.items():
        printed_task = False
        for mode, jkey, rkey, pat, multiseed in modes:
            files = sorted(glob.glob(os.path.join(args.model_dir, pat.format(c=args.ckpt))))
            if not files:
                continue
            j = aggregate(files, jkey, metrics, multiseed)
            r = aggregate(files, rkey, metrics, multiseed)
            n = j[metrics[0]][2]
            if n == 0:
                continue
            if not printed_task:
                print(f"\n{task}  ({'seeds' if multiseed else 'reps'}={n})")
                hdr = "  " + " ".join(f"{SHORT[m]:>14s}" for m in metrics)
                print(f"  {'':10s}{hdr}")
                printed_task = True
            for who, agg in [("JEPA " + mode, j), ("Rand " + mode, r)]:
                cells = " ".join(f"{fmt(*agg[m]):>14s}" for m in metrics)
                print(f"  {who:10s}  {cells}")
            # LaTeX row for the primary metric of this task
            pm = metrics[2] if task == "Siena" else metrics[0]      # Siena -> PR-AUC
            jmu, jsd, _ = j[pm]; rmu, rsd, _ = r[pm]
            latex.append(f"% {task} {mode} ({SHORT[pm]})")
            latex.append(f"Random ({mode}) & \\val{{{rmu:.3f}}}{{{rsd:.3f}}} \\\\  % {task}")
            latex.append(f"\\textbf{{Ours}} ({mode}) & \\best{{\\val{{{jmu:.3f}}}{{{jsd:.3f}}}}} \\\\  % {task}")

    print(f"\n{'-'*78}\n  LaTeX rows (primary metric; Siena=PR-AUC, else BA)\n{'-'*78}")
    print("\n".join(latex))


if __name__ == "__main__":
    main()
