"""Leave-one-subject-out (LOSO) three-way frozen probe for small clinical sets.

Small subject-level tasks (SZ n=28, PD n=50) are too noisy under random splits
(one unlucky test set flips the story). LOSO removes split-luck entirely: each
subject is held out exactly once, predictions are POOLED across all folds, and a
single BA/ROC/PR is computed on the pool. No seeds, one stable number.

Three (optionally four) feature sets, identical LogReg probe on identical folds:
    random-init encoder  <  handcrafted band power  <  our pretrained encoder ?
    (+ PAC-oracle: explicit phase-amplitude coupling features. If PAC >> band
     power, the task GENUINELY needs cross-frequency structure -> our win is not
     luck. If our model ~ PAC-oracle, it learned that coupling without being told.)

Usage:
    python eval_loso.py --dataset schizophrenia \\
        --data_dir /home/pxieaf/home2/datasets/schizophrenia \\
        --checkpoint .../checkpoint_ep2.pt --pac --output loso_sz.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
)

from eval_handcrafted_spectral import bandpower_features, BANDS


# ---------------- feature extractors ----------------

def _bandpass(x, lo, hi, fs, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=1)


def _tort_mi(phase, amp, nbins=18):
    """Tort modulation index per (trial, channel). phase/amp: [N,T,C] -> [N,C]."""
    N, T, C = phase.shape
    idx = np.clip(((phase + np.pi) / (2 * np.pi) * nbins).astype(np.int64), 0, nbins - 1)
    means = np.zeros((N, nbins, C), dtype=np.float64)
    for b in range(nbins):
        m = (idx == b)
        cnt = m.sum(axis=1)
        means[:, b, :] = np.where(cnt > 0, (amp * m).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    P = means / (means.sum(axis=1, keepdims=True) + 1e-12)
    P = np.clip(P, 1e-12, None)
    H = -(P * np.log(P)).sum(axis=1)
    return ((np.log(nbins) - H) / np.log(nbins)).astype(np.float32)   # [N,C]


def pac_features(X, fs, chunk=256, verbose=True):
    """Explicit phase-amplitude coupling (Tort MI) over band pairs -> [N, pairs*C].

    Phase bands: delta/theta/alpha/beta. Amp bands: beta/gamma (amp band strictly
    above the phase band). This is the 'oracle' for the cross-frequency hypothesis.
    """
    phase_bands = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30)]
    amp_bands = [("beta", 13, 30), ("gamma", 30, 45)]
    pairs = [((pl, ph_), (al, ah)) for (_, pl, ph_) in phase_bands
             for (_, al, ah) in amp_bands if al >= ph_]
    N = X.shape[0]
    out = np.empty((N, len(pairs), X.shape[2]), dtype=np.float32)
    t0 = time.time()
    for ci, s in enumerate(range(0, N, chunk)):
        e = min(N, s + chunk)
        xb = X[s:e]
        for pi, ((pl, ph_), (al, ah)) in enumerate(pairs):
            phase = np.angle(hilbert(_bandpass(xb, pl, ph_, fs), axis=1))
            amp = np.abs(hilbert(_bandpass(xb, al, ah, fs), axis=1))
            out[s:e, pi, :] = _tort_mi(phase, amp)
        if verbose and ci % 4 == 0:
            print(f"    pac {e}/{N} trials ({time.time()-t0:.0f}s)", flush=True)
    return out.reshape(N, -1)


def model_features(ckpt, X, device, random_init=False):
    from eval_tuh_clinical import load_pretrained, build_random_init
    from eval_mumtaz import extract_features
    model, model_cls, mtype, n_ch, ckpt_args = load_pretrained(ckpt, device)
    if random_init:
        model = build_random_init(model_cls, n_ch, ckpt_args, device)
    return extract_features(model, X, device), mtype


# ---------------- LOSO probe ----------------

def _metrics(yt, yp, ys):
    out = {"balanced_accuracy": float(balanced_accuracy_score(yt, yp))}
    if len(np.unique(yt)) > 1:
        out["roc_auc"] = float(roc_auc_score(yt, ys))
        out["pr_auc"] = float(average_precision_score(yt, ys))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return out


def loso_probe(feat, y, sids):
    """Leave-one-subject-out LogReg; POOL held-out predictions -> one metric set."""
    uniq = np.unique(sids)
    yt_all, yp_all, ys_all = [], [], []
    n_folds = 0
    for s in uniq:
        te = sids == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(feat[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 class_weight="balanced").fit(sc.transform(feat[tr]), y[tr])
        yp_all.append(clf.predict(sc.transform(feat[te])))
        ys_all.append(clf.predict_proba(sc.transform(feat[te]))[:, 1])
        yt_all.append(y[te])
        n_folds += 1
    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all); ys = np.concatenate(ys_all)
    m = _metrics(yt, yp, ys)
    m["n_folds"] = n_folds
    m["n_test_trials"] = int(len(yt))
    return m


# ---------------- per-dataset all-subject loaders ----------------

def load_all(dataset, args):
    if dataset == "schizophrenia":
        from dataset_schizophrenia import SchizophreniaDataset, _list_subjects
        subs = _list_subjects(args.data_dir)
        want = {"H": sorted(s for g, s in subs if g == "H"),
                "SZ": sorted(s for g, s in subs if g == "SZ")}
        ds = SchizophreniaDataset(data_dir=args.data_dir, subjects=want,
                                  sample_rate=args.sample_rate,
                                  trial_duration_s=args.trial_duration_s,
                                  normalization=args.normalization,
                                  cache_dir=args.cache_dir)
    elif dataset == "mumtaz":
        from dataset_mumtaz import MumtazDataset, _list_subjects
        subs = _list_subjects(args.data_dir)
        want = {"H": sorted(s for g, s in subs if g == "H"),
                "MDD": sorted(s for g, s in subs if g == "MDD")}
        ds = MumtazDataset(data_dir=args.data_dir, subjects=want,
                           sample_rate=args.sample_rate,
                           trial_duration_s=args.trial_duration_s,
                           normalization=args.normalization,
                           cache_dir=args.cache_dir)
    elif dataset == "parkinsons":
        from dataset_parkinsons import ParkinsonsDataset, list_all_subjects
        ds = ParkinsonsDataset(data_dir=args.data_dir,
                               subjects=list_all_subjects(args.data_dir),
                               label=args.pd_label,
                               sample_rate=args.sample_rate,
                               trial_duration_s=args.trial_duration_s,
                               normalization=args.normalization,
                               cache_dir=args.cache_dir)
    else:
        raise ValueError(dataset)
    X = np.stack(ds.trials).astype(np.float32)
    y = np.array(ds.labels, dtype=np.int64)
    sids = np.array(ds.subject_ids, dtype=np.int64)
    return X, y, sids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["schizophrenia", "mumtaz", "parkinsons"])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="if given, also run our-model + random-init LOSO")
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust")
    p.add_argument("--pac", action="store_true", help="also compute PAC-oracle")
    p.add_argument("--pd_label", default="pd_vs_hc",
                   choices=["pd_vs_hc", "on_off"], help="parkinsons only")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    print(f"\n{'='*72}\n  LOSO three-way — {args.dataset}\n{'='*72}")
    t0 = time.time()
    X, y, sids = load_all(args.dataset, args)
    print(f"  Loaded {len(y)} trials, {len(np.unique(sids))} subjects, "
          f"classes {np.bincount(y)}  ({(time.time()-t0)/60:.1f} min)")

    results = {"dataset": args.dataset, "n_trials": int(len(y)),
               "n_subjects": int(len(np.unique(sids))),
               "class_counts": [int(c) for c in np.bincount(y)]}

    feats = {}
    print("  computing handcrafted band power ...", flush=True)
    feats["handcrafted"] = bandpower_features(X, args.sample_rate)
    if args.pac:
        print("  computing PAC-oracle (Tort MI) ...", flush=True)
        feats["pac_oracle"] = pac_features(X, args.sample_rate)
    if args.checkpoint:
        print("  extracting pretrained encoder features ...", flush=True)
        feats["jepa"], mtype = model_features(args.checkpoint, X, device)
        results["model_type"] = mtype
        print("  extracting random-init encoder features ...", flush=True)
        feats["random"], _ = model_features(args.checkpoint, X, device, random_init=True)

    order = ["random", "handcrafted", "pac_oracle", "jepa"]
    print(f"\n  LOSO pooled metrics ({len(np.unique(sids))} folds):")
    print(f"  {'feature':14s} {'dim':>6s} {'BA':>8s} {'ROC':>8s} {'PR':>8s}")
    for name in order:
        if name not in feats:
            continue
        m = loso_probe(feats[name], y, sids)
        results[name] = m
        print(f"  {name:14s} {feats[name].shape[1]:>6d} "
              f"{m['balanced_accuracy']:>8.4f} {m.get('roc_auc',float('nan')):>8.4f} "
              f"{m.get('pr_auc',float('nan')):>8.4f}")

    out = args.output or f"/home/pxieaf/home2/eval_results/loso_{args.dataset}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n→ Saved: {out}")


if __name__ == "__main__":
    main()
