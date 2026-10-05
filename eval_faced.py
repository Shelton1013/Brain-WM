"""FACED 9-class emotion recognition — frozen probe + fine-tuning.

Same protocol family as eval_mental_arithmetic.py (LaBraM-style FT with
layer-wise LR decay, fixed epoch budget, best-val-BA selection), generalised
to multi-class. Metrics follow CBraMod/CSBrain: balanced accuracy, Cohen's
kappa, weighted F1.

    python eval_faced.py --checkpoint .../checkpoint_ep16.pt \
        --proc_dir /home/pxieaf/home2/datasets/FACED_proc --mode both --seed 0
"""
import argparse
import copy
import json
import random as _pyrandom
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

from dataset_faced import N_CLASSES, cbramod_split, load_xy
from eval_mumtaz import extract_features
from eval_tuh_clinical import (
    load_pretrained, build_random_init,
    _build_labram_optimizer, _build_cosine_warmup_scheduler,
)


def metrics(y, pred):
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "cohen_kappa": float(cohen_kappa_score(y, pred)),
            "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def aggregate(per_rep):
    out = {k: {"mean": float(np.mean([m[k] for m in per_rep])),
               "std": float(np.std([m[k] for m in per_rep]))} for k in per_rep[0]}
    out["_per_rep"] = per_rep
    return out


def run_frozen_probe(f_tr, y_tr, f_te, y_te, n_reps=5):
    reps = []
    for r in range(n_reps):
        sc = StandardScaler().fit(f_tr)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42 + r,
                                 class_weight="balanced")
        clf.fit(sc.transform(f_tr), y_tr)
        reps.append(metrics(y_te, clf.predict(sc.transform(f_te))))
    return aggregate(reps)


@torch.no_grad()
def _predict(model, head, X, device, bs=128):
    model.eval(); head.eval()
    out = []
    for i in range(0, len(X), bs):
        b = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(head(model._encode(model._tokenize(b)).mean(1)).argmax(-1).cpu())
    return torch.cat(out).numpy()


def run_finetune(base_model, X_tr, y_tr, X_val, y_val, X_te, y_te, device,
                 max_epochs=50, batch_size=64, base_lr=5e-4, weight_decay=0.05,
                 layer_decay=0.65, warmup_epochs=5):
    model = copy.deepcopy(base_model)
    head = nn.Sequential(nn.BatchNorm1d(model.d_model),
                         nn.Linear(model.d_model, N_CLASSES)).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    counts = torch.bincount(torch.from_numpy(y_tr), minlength=N_CLASSES).float()
    w = 1.0 / counts.clamp(min=1)
    criterion = nn.CrossEntropyLoss(weight=(w / w.sum() * N_CLASSES).to(device))
    optimizer = _build_labram_optimizer(model, head, base_lr=base_lr,
                                        weight_decay=weight_decay,
                                        layer_decay=layer_decay, head_lr_mult=1.0)
    scheduler = _build_cosine_warmup_scheduler(
        optimizer, steps_per_epoch=len(loader), warmup_epochs=warmup_epochs,
        total_epochs=max_epochs)
    best, best_state = -1.0, None
    for ep in range(max_epochs):
        model.train(); head.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = criterion(head(model._encode(model._tokenize(bx)).mean(1)), by)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 3.0)
            optimizer.step(); scheduler.step()
        val_ba = balanced_accuracy_score(y_val, _predict(model, head, X_val, device))
        if val_ba > best:
            best = val_ba
            best_state = (copy.deepcopy(model.state_dict()),
                          copy.deepcopy(head.state_dict()))
        print(f"      ep{ep + 1:03d}{'*' if val_ba >= best else ' '} "
              f"val_ba={val_ba:.4f} best={best:.4f}", flush=True)
    model.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    m = metrics(y_te, _predict(model, head, X_te, device))
    m["best_val_ba"] = float(best)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--proc_dir", default="/home/pxieaf/home2/datasets/FACED_proc")
    p.add_argument("--mode", choices=["frozen", "finetune", "both"], default="both")
    p.add_argument("--frozen_reps", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--include_random_baseline", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    _pyrandom.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cls, mtype, n_ch, ckpt_args = load_pretrained(a.checkpoint, device)

    split = cbramod_split(a.proc_dir)
    X_tr, y_tr, _ = load_xy(a.proc_dir, split["train"])
    X_val, y_val, _ = load_xy(a.proc_dir, split["val"])
    X_te, y_te, _ = load_xy(a.proc_dir, split["test"])
    print(f"FACED train {X_tr.shape} val {X_val.shape} test {X_te.shape}", flush=True)

    res = {"checkpoint": a.checkpoint, "seed": a.seed, "split": split,
           "n_train": int(len(y_tr)), "n_val": int(len(y_val)), "n_test": int(len(y_te))}
    models = [("jepa", model)]
    if a.include_random_baseline:
        models.append(("random", build_random_init(model_cls, n_ch, ckpt_args, device)))
    for name, m in models:
        if a.mode in ("frozen", "both"):
            res[f"{name}_frozen"] = run_frozen_probe(
                extract_features(m, X_tr, device), y_tr,
                extract_features(m, X_te, device), y_te, a.frozen_reps)
            print(f"  {name} frozen BA={res[f'{name}_frozen']['balanced_accuracy']['mean']:.4f}",
                  flush=True)
        if a.mode in ("finetune", "both"):
            res[f"{name}_finetune"] = run_finetune(m, X_tr, y_tr, X_val, y_val, X_te, y_te,
                                                   device, a.max_epochs, a.batch_size)
            print(f"  {name} FT {res[f'{name}_finetune']}", flush=True)

    out = a.output or str(Path(a.checkpoint).with_name(
        f"{Path(a.checkpoint).stem}_faced_seed{a.seed}.json"))
    json.dump(res, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
