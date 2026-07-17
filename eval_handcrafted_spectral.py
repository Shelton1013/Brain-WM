"""Handcrafted spectral baseline — the make-or-break control for our claim.

Our pretext task predicts cross-band spectral power, so the honest question is:
does the pretrained encoder learn anything BEYOND what a handcrafted band-power
feature vector already exposes?  This script computes classical Welch band-power
features (absolute log-power + relative power, per channel per band) and runs
the IDENTICAL frozen-probe classifier (StandardScaler + class-weighted LogReg)
on the IDENTICAL subject-disjoint splits used by eval_mumtaz.py / eval_siena.py.

Three-way comparison the paper needs:
    random-init encoder  <  handcrafted band power  <  our pretrained encoder ?
If pretrained does NOT beat handcrafted, the "learns cross-frequency structure"
story collapses (it only recovered marginal band power). If it does, pretraining
captures conditional cross-band/-channel/-time dependencies that hand features
miss — which is exactly the motivation.

Usage:
    python eval_handcrafted_spectral.py --dataset mumtaz \\
        --data_dir /home/pxieaf/home2/datasets/mumtaz2016 \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --seed 42 --output out.json

    python eval_handcrafted_spectral.py --dataset siena \\
        --data_dir /home/pxieaf/home2/datasets/Siena/1.0.0 ...
"""
import argparse
import json
import random as _pyrandom
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
)

# 5 canonical bands (Hz) — matches our v3 filterbank init bands.
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0)]


def dataset_to_xy(ds):
    X = np.stack(ds.trials).astype(np.float32)      # (N, T, C)
    y = np.array(ds.labels, dtype=np.int64)
    return X, y


def bandpower_features(X, fs, chunk=20000, verbose=True):
    """X (N, T, C) -> handcrafted spectral features (N, 2*n_bands*C).

    Per channel per band: log absolute power and relative power (fraction of
    total in-band power). Welch is computed in CHUNKS over the flattened
    (epoch x channel) signals so the intermediate never blows up memory, with
    progress prints (a single monolithic welch on ~1e5 epochs stalls).
    """
    N, T, C = X.shape
    nperseg = min(T, int(fs))                               # ~1 s window
    sig = np.ascontiguousarray(X.transpose(0, 2, 1)).reshape(N * C, T)  # [N*C, T]
    M = sig.shape[0]
    freqs = welch(sig[:1], fs=fs, nperseg=nperseg, axis=1)[0]
    masks = [((freqs >= lo) & (freqs < hi)) for _, lo, hi in BANDS]
    bp = np.empty((M, len(BANDS)), dtype=np.float32)        # [N*C, n_bands]
    t0 = time.time()
    for ci, s in enumerate(range(0, M, chunk)):
        e = min(M, s + chunk)
        _, psd = welch(sig[s:e], fs=fs, nperseg=nperseg, axis=1)   # [chunk, F]
        for bi, m in enumerate(masks):
            bp[s:e, bi] = psd[:, m].sum(axis=1)
        if verbose and ci % 5 == 0:
            print(f"    bandpower {e}/{M} signals ({time.time()-t0:.0f}s)", flush=True)
    bp = bp.reshape(N, C, len(BANDS))                       # [N, C, n_bands]
    total = bp.sum(axis=2, keepdims=True) + 1e-10           # [N, C, 1]
    log_abs = np.log(bp + 1e-10)
    rel = bp / total
    return np.concatenate([log_abs.reshape(N, -1), rel.reshape(N, -1)],
                          axis=1).astype(np.float32)


def compute_metrics(y_true, y_pred, y_proba):
    """BA always; ROC/PR guarded for single-class test folds (Siena)."""
    y_true = np.asarray(y_true)
    out = {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def run_frozen_probe(feat_tr, y_tr, feat_te, y_te, n_reps=5):
    """Identical protocol to eval_mumtaz.run_frozen_probe."""
    metrics_by_rep = []
    for seed in range(n_reps):
        scaler = StandardScaler()
        tr_s = scaler.fit_transform(feat_tr)
        te_s = scaler.transform(feat_te)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 random_state=42 + seed, class_weight="balanced")
        clf.fit(tr_s, y_tr)
        preds = clf.predict(te_s)
        proba = clf.predict_proba(te_s)[:, 1]
        metrics_by_rep.append(compute_metrics(y_te, preds, proba))
    agg = {}
    for k in metrics_by_rep[0]:
        vals = [m[k] for m in metrics_by_rep]
        agg[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    agg["_per_rep"] = metrics_by_rep
    return agg


# ---- per-dataset split loaders (mirror the eval_*.py scripts exactly) -------

def load_mumtaz(args):
    from dataset_mumtaz import MumtazDataset, make_subject_split
    splits = make_subject_split(args.data_dir, seed=args.seed)
    print(f"  Train: H {splits['train']['H']}  MDD {splits['train']['MDD']}")
    print(f"  Test:  H {splits['test']['H']}  MDD {splits['test']['MDD']}")

    def _ds(subjects):
        return MumtazDataset(
            data_dir=args.data_dir, subjects=subjects,
            sample_rate=args.sample_rate, trial_duration_s=args.trial_duration_s,
            normalization=args.normalization, cache_dir=args.cache_dir)

    X_tr, y_tr = dataset_to_xy(_ds(splits["train"]))
    X_te, y_te = dataset_to_xy(_ds(splits["test"]))
    meta = {"split": {k: {g: list(v) for g, v in d.items()}
                      for k, d in splits.items()}}
    return X_tr, y_tr, X_te, y_te, meta


def load_siena(args):
    from dataset_siena import SienaDataset, make_csbrain_split
    splits = make_csbrain_split(args.data_dir)
    print(f"  Train+Val pool: {splits['trainval']}")
    print(f"  Test:           {splits['test']}")

    def _ds(subjects):
        return SienaDataset(
            data_dir=args.data_dir, subjects=subjects,
            sample_rate=args.sample_rate, trial_duration_s=args.trial_duration_s,
            normalization=args.normalization,
            negative_per_positive=args.negative_per_positive,
            seed=args.seed, cache_dir=args.cache_dir)

    X_tr, y_tr = dataset_to_xy(_ds(splits["trainval"]))   # full pool for probe
    X_te, y_te = dataset_to_xy(_ds(splits["test"]))
    meta = {"split": {k: list(v) for k, v in splits.items()}}
    return X_tr, y_tr, X_te, y_te, meta


LOADERS = {"mumtaz": load_mumtaz, "siena": load_siena}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(LOADERS))
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust",
                   choices=["per_trial_zscore", "per_recording_robust"])
    p.add_argument("--negative_per_positive", type=float, default=0.0,
                   help="Siena only; 0 = keep all interictal (match eval_siena).")
    p.add_argument("--frozen_reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    _pyrandom.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*72}")
    print(f"  Handcrafted spectral baseline — {args.dataset}  (seed {args.seed})")
    print(f"  Features: log-abs + relative band power  "
          f"({2*len(BANDS)}×n_ch dims), LogReg × {args.frozen_reps}")
    print(f"{'='*72}")

    t0 = time.time()
    X_tr, y_tr, X_te, y_te, meta = LOADERS[args.dataset](args)
    print(f"  Data loaded in {(time.time()-t0)/60:.1f} min")
    print(f"  Shapes: train {X_tr.shape}, test {X_te.shape}")
    print(f"  Train class counts: {np.bincount(y_tr)}   "
          f"Test class counts: {np.bincount(y_te)}")

    feat_tr = bandpower_features(X_tr, args.sample_rate)
    feat_te = bandpower_features(X_te, args.sample_rate)
    print(f"  Handcrafted feature dim: {feat_tr.shape[1]}")

    m = run_frozen_probe(feat_tr, y_tr, feat_te, y_te, n_reps=args.frozen_reps)
    print(f"\n  Handcrafted-spectral frozen probe:")
    for k in ("balanced_accuracy", "roc_auc", "pr_auc"):
        print(f"    {k:20s} {m[k]['mean']:.4f} ± {m[k]['std']:.4f}")

    results = {
        "dataset": args.dataset,
        "feature": "handcrafted_bandpower_logabs+rel",
        "feature_dim": int(feat_tr.shape[1]),
        "seed": int(args.seed),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "handcrafted_frozen": m,
        **meta,
    }
    out_path = args.output or (f"/home/pxieaf/home2/eval_results/"
                               f"handcrafted_{args.dataset}_seed{args.seed}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved: {out_path}")


if __name__ == "__main__":
    main()
