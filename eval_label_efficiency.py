"""Label-efficiency curve: frozen features + LogReg on SUBSAMPLED train labels.

The FM selling point when absolute numbers only tie SOTA: our pretrained frozen
features stay accurate with very few downstream labels, while a random-init
encoder needs many. We extract frozen features ONCE (encoder fixed), then train
LogReg on {1,5,10,25,50,100}% of the train labels (stratified, n_reps subsamples)
and evaluate on the FULL test set. Pretrained vs random-init on identical splits.

Claim to look for: at 1-10% labels, pretrained >> random (gap = label efficiency);
the two converge only at 100%.

    python eval_label_efficiency.py --checkpoint .../checkpoint_ep2.pt \\
        --dataset mumtaz --data_dir /home/pxieaf/home2/datasets/mumtaz2016_full \\
        --output le_mumtaz.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
)

from eval_tuh_clinical import load_pretrained, build_random_init
from eval_mumtaz import dataset_to_xy, extract_features

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def load_split(dataset, args):
    """Return X_tr, y_tr, X_te, y_te for the chosen dataset."""
    dsk = dict(sample_rate=args.sample_rate, trial_duration_s=args.trial_duration_s,
               normalization=args.normalization, cache_dir=args.cache_dir)
    if dataset == "mumtaz":
        from dataset_mumtaz import MumtazDataset, make_cbramod_split
        sp = make_cbramod_split(args.data_dir)
        tr = MumtazDataset(data_dir=args.data_dir, subjects=sp["train"], **dsk)
        te = MumtazDataset(data_dir=args.data_dir, subjects=sp["test"], **dsk)
    elif dataset == "schizophrenia":
        from dataset_schizophrenia import SchizophreniaDataset, make_subject_split
        sp = make_subject_split(args.data_dir, seed=42)
        tr = SchizophreniaDataset(data_dir=args.data_dir, subjects=sp["train"], **dsk)
        te = SchizophreniaDataset(data_dir=args.data_dir, subjects=sp["test"], **dsk)
    elif dataset == "mental_arithmetic":
        from dataset_mental_arithmetic import MentalArithmeticDataset, make_subject_split
        sp = make_subject_split(args.data_dir, seed=42)
        tr = MentalArithmeticDataset(data_dir=args.data_dir, subjects=sp["train"], **dsk)
        te = MentalArithmeticDataset(data_dir=args.data_dir, subjects=sp["test"], **dsk)
    else:
        raise ValueError(dataset)
    X_tr, y_tr = dataset_to_xy(tr)
    X_te, y_te = dataset_to_xy(te)
    return X_tr, y_tr, X_te, y_te


def _strat_idx(y, frac, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        n = max(1, int(round(len(ci) * frac)))
        idx.extend(rng.choice(ci, min(n, len(ci)), replace=False))
    return np.array(sorted(idx))


def _probe(feat_tr, y_tr, feat_te, y_te, idx):
    sc = StandardScaler().fit(feat_tr[idx])
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(sc.transform(feat_tr[idx]), y_tr[idx])
    proba = clf.predict_proba(sc.transform(feat_te))[:, 1]
    pred = clf.predict(sc.transform(feat_te))
    out = {"balanced_accuracy": float(balanced_accuracy_score(y_te, pred))}
    if len(np.unique(y_te)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_te, proba))
        out["pr_auc"] = float(average_precision_score(y_te, proba))
    return out


def curve(feat_tr, y_tr, feat_te, y_te, n_reps):
    """{frac: {metric: (mean,std)}} over n_reps stratified subsamples."""
    res = {}
    for frac in FRACTIONS:
        reps = []
        for r in range(n_reps if frac < 1.0 else 1):
            idx = _strat_idx(y_tr, frac, seed=100 + r)
            reps.append(_probe(feat_tr, y_tr, feat_te, y_te, idx))
        agg = {}
        for m in reps[0]:
            v = [x[m] for x in reps]
            agg[m] = (float(np.mean(v)), float(np.std(v)))
        agg["n_train"] = int(len(_strat_idx(y_tr, frac, 100)))
        res[f"{frac:.2f}"] = agg
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="mumtaz",
                   choices=["mumtaz", "schizophrenia", "mental_arithmetic"])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust")
    p.add_argument("--n_reps", type=int, default=10,
                   help="subsamples per fraction (variance at low-label)")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    print(f"\n{'='*72}\n  Label efficiency — {args.dataset}\n{'='*72}")
    model, model_cls, mtype, n_ch, ckpt_args = load_pretrained(args.checkpoint, device)
    X_tr, y_tr, X_te, y_te = load_split(args.dataset, args)
    print(f"  train {X_tr.shape}  test {X_te.shape}  classes {np.bincount(y_tr)}")

    t = time.time()
    fp_tr = extract_features(model, X_tr, device); fp_te = extract_features(model, X_te, device)
    rm = build_random_init(model_cls, n_ch, ckpt_args, device)
    fr_tr = extract_features(rm, X_tr, device); fr_te = extract_features(rm, X_te, device)
    print(f"  features extracted ({time.time()-t:.0f}s)")

    pre = curve(fp_tr, y_tr, fp_te, y_te, args.n_reps)
    rnd = curve(fr_tr, y_tr, fr_te, y_te, args.n_reps)

    print(f"\n  {'frac':>6} {'n_tr':>6} {'pretrained BA':>16} {'random BA':>16} {'gap':>7}")
    for f in [f"{x:.2f}" for x in FRACTIONS]:
        pb = pre[f]["balanced_accuracy"]; rb = rnd[f]["balanced_accuracy"]
        print(f"  {f:>6} {pre[f]['n_train']:>6} {pb[0]:>8.3f}±{pb[1]:.3f}   "
              f"{rb[0]:>8.3f}±{rb[1]:.3f}   {pb[0]-rb[0]:>+.3f}")

    out = {"checkpoint": args.checkpoint, "dataset": args.dataset,
           "model_type": mtype, "fractions": FRACTIONS,
           "pretrained": pre, "random": rnd}
    path = args.output or f"/home/pxieaf/home2/eval_results/le_{args.dataset}.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n→ Saved: {path}")


if __name__ == "__main__":
    main()
