"""Strict linear probe for sleep staging (ISRUC / HMC).

Frozen encoder -> one 512-d mean-pooled feature per 30-s epoch -> multinomial
logistic regression (no hidden layer, no sequence context). The regulariser C
is chosen on the validation set, then the test set is scored once. Fully
deterministic. Measures only the linear accessibility of the representation;
absolute numbers are below the sequence-head protocol and are for comparing
pretext targets, not for comparing with published baselines.

Loads the data once and probes many checkpoints:
    python eval_linear_probe_sleep.py --dataset isruc \
        --data_dir /home/pxieaf/home2/datasets/isruc/subgroupI_official \
        --checkpoints name1=/path/a.pt name2=/path/b.pt --random_init \
        --output lp_isruc.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

from eval_tuh_clinical import load_pretrained, build_random_init
from eval_sleep_seq2seq import precompute_seq_features
from eval_label_efficiency_seq2seq import load_seqs

C_GRID = (0.01, 0.1, 1.0)


def epoch_features(model, X, device):
    """[N, 20, T, C] sequences -> [N*20, d] per-epoch features."""
    F = precompute_seq_features(model, X, device)            # [N, 20, d]
    return F.reshape(-1, F.shape[-1]).numpy()


def probe(Ftr, ytr, Fva, yva, Fte, yte):
    sc = StandardScaler().fit(Ftr)
    Ftr, Fva, Fte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(Ftr, ytr)
        va = balanced_accuracy_score(yva, clf.predict(Fva))
        if best is None or va > best[0]:
            best = (va, C, clf)
    va, C, clf = best
    p = clf.predict(Fte)
    return {"balanced_accuracy": float(balanced_accuracy_score(yte, p)),
            "cohen_kappa": float(cohen_kappa_score(yte, p)),
            "weighted_f1": float(f1_score(yte, p, average="weighted")),
            "val_ba": float(va), "C": C}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["isruc", "hmc"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    ap.add_argument("--sample_rate", type=int, default=256)
    ap.add_argument("--trial_duration_s", type=int, default=10)
    ap.add_argument("--normalization", default="per_recording_robust")
    ap.add_argument("--checkpoints", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--random_init", action="store_true",
                    help="also probe the same architecture, randomly initialised")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, Xva, yva, Xte, yte, _ = load_seqs(a.dataset, a)
    ytr, yva, yte = ytr.reshape(-1), yva.reshape(-1), yte.reshape(-1)
    print(f"epochs: train {len(ytr)} val {len(yva)} test {len(yte)}", flush=True)

    out = Path(a.output)
    res = json.load(open(out)) if out.exists() else {}
    todo = [c.split("=", 1) for c in a.checkpoints]
    if a.random_init:
        todo.append(("random_init", todo[0][1]))
    for name, path in todo:
        if name in res or not Path(path).exists():
            print(f"skip {name} ({'done' if name in res else 'missing'})", flush=True)
            continue
        t = time.time()
        model, cls, _, n_ch, ck_args = load_pretrained(path, device)
        if name == "random_init":
            torch.manual_seed(0)
            model = build_random_init(cls, n_ch, ck_args, device)
        feats = [epoch_features(model, X, device) for X in (Xtr, Xva, Xte)]
        res[name] = probe(feats[0], ytr, feats[1], yva, feats[2], yte)
        res[name]["checkpoint"] = path
        print(f"{name:32s} BA {res[name]['balanced_accuracy']:.4f} "
              f"k {res[name]['cohen_kappa']:.4f} (C={res[name]['C']}, "
              f"{(time.time() - t) / 60:.1f} min)", flush=True)
        json.dump(res, open(out, "w"), indent=2)            # checkpoint progress
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
