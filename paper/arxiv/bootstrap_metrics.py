"""Subject-level bootstrap of the strict linear probe.

Reads the test predictions saved with --pred_dir (one npz per dataset and model: y, pred,
[proba,] subj, val_ba), resamples test subjects with replacement and reports, for every
model and metric, the point estimate and the bootstrap standard error. For SpecBraM the score is
the mean over the three pretraining seeds, recomputed on the same resample for each seed; for
REVE the readout (mean / attention pooling) is chosen on validation balanced accuracy.

    python bootstrap_metrics.py <pred_dir> <out.json> [B]
"""
import json
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def conf(y, p, k):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)


def metrics(y, p, proba, k):
    c = conf(y, p, k)
    n, row, col = c.sum(), c.sum(1), c.sum(0)
    rec = np.divide(np.diag(c), row, out=np.zeros(k), where=row > 0)
    prec = np.divide(np.diag(c), col, out=np.zeros(k), where=col > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=prec + rec > 0)
    po, pe = np.trace(c) / n, (row @ col) / n ** 2
    out = {"balanced_accuracy": rec[row > 0].mean(),
           "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else 0.0,
           "weighted_f1": (f1 * row).sum() / n}
    if k == 2 and proba is not None:
        out["auc_pr"] = average_precision_score(y, proba[:, 1])
        out["auroc"] = roc_auc_score(y, proba[:, 1])
    return out


def load(f):
    with np.load(f, allow_pickle=True) as d:
        return {"y": d["y"].astype(int), "pred": d["pred"].astype(int), "subj": d["subj"].astype(str),
                "proba": d["proba"] if "proba" in d.files else None, "val_ba": float(d["val_ba"])}


def bootstrap_draws(y, subj, B=1000, seed=0):
    """Cluster bootstrap, conditioned on class coverage.

    Only binary tasks with one diagnosis per subject are label-stratified.
    Majority-label strata for mixed-label subjects can freeze singleton strata
    (e.g. a sleep subject) and are therefore deliberately not used.
    """
    if B < 2:
        raise ValueError("At least two bootstrap replicates are required")
    y, subj = np.asarray(y), np.asarray(subj)
    uniq, inv = np.unique(subj, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    classes = np.unique(y)
    homogeneous = all(len(np.unique(y[m])) == 1 for m in members)
    stratified = len(classes) == 2 and homogeneous
    if stratified:
        labels = np.array([y[m[0]] for m in members])
        strata = [np.flatnonzero(labels == c) for c in classes]
    else:
        strata = [np.arange(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for attempt in range(B * 100):
        idx = np.concatenate([members[i] for st in strata for i in rng.choice(st, len(st))])
        if len(np.unique(y[idx])) == len(classes):
            draws.append(idx)
        if len(draws) == B:
            break
    if len(draws) != B:
        raise ValueError("Could not obtain bootstrap resamples containing all classes")
    scheme = "binary_subject_label_stratified" if stratified else "ordinary_subject_class_coverage"
    return draws, scheme, {"attempted_draws": attempt + 1, "rejected_missing_class": attempt + 1 - len(draws)}


def main(pred_dir, out, B=1000, sleep_mode="avg3"):
    files = defaultdict(dict)                       # ds -> name -> npz
    for f in sorted(Path(pred_dir).glob("*.npz")):
        parts = f.stem.split("__")
        if len(parts) not in (2, 3):
            raise ValueError(f"Unrecognized prediction filename: {f.name}")
        ds, name = parts[:2]
        if ds in ("isruc", "hmc") and len(parts) == 3 and parts[2] != sleep_mode:
            continue
        if name in files[ds]:
            raise ValueError(f"Duplicate prediction identity for {ds}/{name}: {f.name}")
        files[ds][name] = load(f)
    res = {}
    for ds, M in files.items():
        groups = defaultdict(list)                  # model -> runs (seeds / readouts)
        for name, d in M.items():
            if re.fullmatch(r"(m|specbram|g)_s4\d(_ep16)?|linfftstdlog_nosig_s4\d_ep16", name):
                groups["specbram"].append(d)
                if "s42" in name:                   # the main (released) checkpoint, scored alone
                    groups["specbram_s42"].append(d)
            elif name.startswith("reve_base"):
                groups["reve_base"].append({**d, "mean_pool": name.endswith("mean")})
            else:
                groups[name].append(d)
        if len(groups["reve_base"]) > 1:            # readout chosen on validation
            groups["reve_base"] = [max(groups["reve_base"], key=lambda d: (d["val_ba"], d["mean_pool"]))]  # tie: mean
        if not groups["reve_base"]:
            del groups["reve_base"]
        ref = next(iter(groups.values()))[0]
        k = int(max(d["y"].max() for g in groups.values() for d in g)) + 1
        subj = ref["subj"]
        for g in groups.values():
            for d in g:
                assert (d["y"] == ref["y"]).all() and (d["subj"] == subj).all(), ds
        uniq = np.unique(subj)
        draws, scheme, sampling_info = bootstrap_draws(ref["y"], subj, B=B)

        def score(runs, idx=None):
            per = [metrics(d["y"] if idx is None else d["y"][idx], d["pred"] if idx is None else d["pred"][idx],
                           None if d["proba"] is None else (d["proba"] if idx is None else d["proba"][idx]), k)
                   for d in runs]
            return {m: float(np.mean([p[m] for p in per])) for m in per[0]}

        res[ds] = {"n_test": int(len(subj)), "n_units": int(len(uniq)), "n_classes": k,
                   "bootstrap_scheme": scheme, "bootstrap_replicates": B,
                   "sampling_info": sampling_info,
                   "sleep_mode": sleep_mode if ds in ("isruc", "hmc") else None, "models": {}}
        allboots = {}
        for name, runs in groups.items():
            point = score(runs)
            boots = allboots[name] = [score(runs, idx) for idx in draws]
            res[ds]["models"][name] = {
                "n_runs": len(runs),
                "per_run": [{m: float(v) for m, v in score([d]).items()} for d in runs],
                **{m: {"point": point[m], "se": float(np.std([b[m] for b in boots], ddof=1)),
                     "lo": float(np.percentile([b[m] for b in boots], 2.5)),
                     "hi": float(np.percentile([b[m] for b in boots], 97.5))} for m in point}}
            print(ds, name, len(runs), {m: f"{v['point']:.4f}±{v['se']:.4f}"
                                        for m, v in res[ds]["models"][name].items() if m not in ("n_runs", "per_run")})
        # paired bootstrap: main checkpoint minus each external model on the SAME resamples
        res[ds]["paired"] = {}
        ref = "specbram_s42"
        for name in groups:
            if name in (ref, "specbram"):
                continue
            pr = {}
            for m in res[ds]["models"][ref]:
                if m in ("n_runs", "per_run") or m not in res[ds]["models"][name]:
                    continue
                d = np.array([a[m] - b[m] for a, b in zip(allboots[ref], allboots[name])])
                pr[m] = {"diff": res[ds]["models"][ref][m]["point"] - res[ds]["models"][name][m]["point"],
                         "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                         "p_le0": float((d <= 0).mean())}      # bootstrap tail fraction, not a calibrated p-value
            res[ds]["paired"][name] = pr
    with open(out, "w") as stream:
        json.dump(res, stream, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pred_dir")
    parser.add_argument("out")
    parser.add_argument("B", type=int, nargs="?", default=1000)
    parser.add_argument("--sleep-mode", choices=("avg3", "center10"), default="avg3")
    args = parser.parse_args()
    main(args.pred_dir, args.out, args.B, args.sleep_mode)
