"""Strict linear probe on CBraMod's downstream tasks (MA, Mumtaz, PhysioNet-MI,
FACED) for SpecBraM checkpoints AND official foundation-model weights, on the
identical samples / split (npz from build_task_npz.py).

Protocol = eval_linear_probe_v2.py: frozen encoder, token mean-pool, features
z-scored on train, class-balanced logistic regression with C chosen on the
validation split (balanced accuracy), scored once on test. Deterministic.
Metrics: binary -> BA / AUC-PR / AUROC; multi-class -> BA / kappa / weighted F1
(the metrics CBraMod reports for each task).

Input contract per model (same samples, each model's own convention):
  specbram : 200 -> 256 Hz, 0.5-45 Hz band-pass, per-recording robust scaling,
             the 19 canonical 10-20 channels (all four tasks contain them)
  cbramod  : raw µV / 100, every channel of the sample, 1-s patches
  labram   : raw µV / 100, every channel present in LaBraM's standard_1020 list

    python eval_linear_probe_tasks.py --task ma --random_init --baselines cbramod labram \
        --checkpoints linbp_s42=.../checkpoint_ep16.pt --output lp_ma.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from dataset_mumtaz import OUR_19CH_ORDER

C_GRID = (0.01, 0.1, 1.0)
_ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}      # new -> old 10-20 names


def our_input(X, ch, rec, bandpass=(0.5, 45.0)):
    """[N, C, T] µV @200 Hz -> [N, T', 19] float32 in our input contract."""
    names = [_ALIAS.get(c, c) for c in ch]
    canon = [c.upper() for c in OUR_19CH_ORDER]
    idx = [names.index(c) for c in canon]                      # KeyError = missing lead
    x = X[:, idx, :].astype(np.float64)
    sos = butter(4, bandpass, btype="band", fs=200, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    x = resample_poly(x, 32, 25, axis=-1)                      # 200 -> 256 Hz
    out = np.empty_like(x, dtype=np.float32)
    for r in np.unique(rec):                                   # per-recording robust scale
        m = rec == r
        flat = x[m].transpose(1, 0, 2).reshape(len(idx), -1)   # [19, n*T]
        med = np.median(flat, axis=1)[None, :, None]
        iqr = (np.percentile(flat, 75, axis=1) - np.percentile(flat, 25, axis=1))[None, :, None]
        out[m] = (x[m] - med) / (iqr / 1.349 + 1e-6)
    return out.transpose(0, 2, 1)                              # [N, T', 19]


@torch.no_grad()
def feats_ours(model, X, device, bs=128):
    model.eval()
    return np.concatenate([model._encode(model._tokenize(torch.from_numpy(X[i:i + bs]).to(device)))
                           .mean(1).float().cpu().numpy() for i in range(0, len(X), bs)])


@torch.no_grad()
def feats_baseline(name, X, ch, device, bs=64):
    from eval_linear_probe_v2 import cbramod_encoder, labram_encoder
    if name == "cbramod":
        enc = cbramod_encoder(device)[0]
        chans, kw = list(range(len(ch))), {}
    else:
        import sys
        enc = labram_encoder(device)[0]
        import utils as lutils                                  # LaBraM repo (on path now)
        chans = [i for i, c in enumerate(ch) if c in lutils.standard_1020]
        kw = {"names": [ch[i] for i in chans]}
        print(f"    LaBraM channels: {len(chans)}/{len(ch)}", flush=True)
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + bs].transpose(0, 2, 1))).to(device)
        out.append(enc(x, chans, **kw).float().cpu().numpy())
    return np.concatenate(out)


def score(y, pred, proba, n_cls):
    if n_cls == 2:
        return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "auc_pr": float(average_precision_score(y, proba[:, 1])),
                "auroc": float(roc_auc_score(y, proba[:, 1]))}
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "cohen_kappa": float(cohen_kappa_score(y, pred)),
            "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def probe(F, y, n_cls):
    sc = StandardScaler().fit(F["train"])
    F = {k: sc.transform(v) for k, v in F.items()}
    fit = lambda C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(F["train"], y["train"])
    val_ba, C = max((balanced_accuracy_score(y["val"], fit(C).predict(F["val"])), C) for C in C_GRID)
    clf = fit(C)
    res = {"C": C, "val_ba": float(val_ba),
           "test": score(y["test"], clf.predict(F["test"]), clf.predict_proba(F["test"]), n_cls)}
    print(f"    C={C} val_ba={val_ba:.4f} test {res['test']}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["ma", "mumtaz", "physio", "faced"])
    p.add_argument("--data_dir", default="/home/pxieaf/home2/datasets/lp_tasks")
    p.add_argument("--checkpoints", nargs="*", default=[], help="name=path ...")
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--baselines", nargs="*", default=[], choices=["cbramod", "labram"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = np.load(Path(a.data_dir) / f"{a.task}.npz")
    X, y_all, rec, sp, ch = d["X"], d["y"], d["rec"], d["split"], [str(c) for c in d["ch"]]
    parts = {k: sp == k for k in ("train", "val", "test")}
    y = {k: y_all[m] for k, m in parts.items()}
    n_cls = int(y_all.max()) + 1
    print(f"{a.task}: X {X.shape}, classes {n_cls}, "
          + " ".join(f"{k}={m.sum()}" for k, m in parts.items()), flush=True)

    out = Path(a.output)
    res = json.load(open(out)) if out.exists() else {"task": a.task, "results": {}}
    save = lambda: json.dump(res, open(out, "w"), indent=2)

    todo = [c.split("=", 1) for c in a.checkpoints]
    if todo:
        Xo = our_input(X, ch, rec)
        print(f"  our input {Xo.shape}", flush=True)
        from eval_tuh_clinical import load_pretrained, build_random_init
        for name, ck in todo + ([("random_init", todo[0][1])] if a.random_init else []):
            if name in res["results"]:
                continue
            model, cls, _, n_ch, args = load_pretrained(ck, dev)
            if name == "random_init":
                torch.manual_seed(0)
                model = build_random_init(cls, n_ch, args, dev)
            print(f"  [{name}]", flush=True)
            F = feats_ours(model, Xo, dev)
            res["results"][name] = probe({k: F[m] for k, m in parts.items()}, y, n_cls)
            res["results"][name]["checkpoint"] = None if name == "random_init" else ck
            save(); del model; torch.cuda.empty_cache()
    for b in a.baselines:
        if b in res["results"]:
            continue
        print(f"  [{b}]", flush=True)
        F = feats_baseline(b, X, ch, dev)
        res["results"][b] = probe({k: F[m] for k, m in parts.items()}, y, n_cls)
        save()
    save()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
