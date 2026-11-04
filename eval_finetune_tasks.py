"""Full fine-tuning on the non-sleep classification tasks (appendix only).

Same samples, splits and metrics as the strict linear probe (eval_linear_probe_tasks.py /
eval_linear_probe_tuab.py); same recipe as the SEED-VIG fine-tuning (eval_seedvig.py):
encoder + BatchNorm/linear head on mean-pooled tokens, AdamW with layer-wise LR decay,
cosine schedule with warm-up, gradient clipping. The epoch is selected on validation
balanced accuracy (early stopping), the test split is scored once.

    python eval_finetune_tasks.py --task ma --checkpoint .../checkpoint_ep16.pt \
        --name m_s42 --output ft_ma_m_s42.json
"""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from eval_linear_probe_tasks import our_input, score
from eval_tuh_clinical import (load_pretrained, _build_labram_optimizer,
                               _build_cosine_warmup_scheduler)

ROOT = Path(os.environ.get("SPECBRAM_TASK_DATA", "/home/pxieaf/home2/datasets/lp_tasks"))
# (max epochs, early-stopping patience): the TUH tasks have ~10^5 windows per epoch
SCHED = {"tuab": (20, 4), "tuev": (30, 6)}


class Windows(Dataset):
    def __init__(self, X, idx, y):
        self.X, self.idx, self.y = X, idx, y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return (torch.from_numpy(np.asarray(self.X[self.idx[i]], np.float32)),
                torch.tensor(int(self.y[i])))


def load(task):
    if task in ("tuab", "tuev"):
        meta = np.load(ROOT / task / "meta.npz")
        return np.load(ROOT / task / "ours256.npy", mmap_mode="r"), meta["y"], meta["split"]
    d = np.load(ROOT / f"{task}.npz")
    return our_input(d["X"], [str(c) for c in d["ch"]], d["rec"]), d["y"], d["split"]


@torch.no_grad()
def predict(model, head, X, idx, device, bs=256):
    model.eval(); head.eval()
    out = []
    for i in range(0, len(idx), bs):
        x = torch.from_numpy(np.asarray(X[idx[i:i + bs]], np.float32)).to(device)
        out.append(torch.softmax(head(model._encode(model._tokenize(x)).mean(1)), -1).float().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["ma", "mumtaz", "physio", "faced", "tuab", "tuev"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--base_lr", type=float, default=5e-4)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    X, y_all, sp = load(a.task)
    idx = {k: np.where(sp == k)[0] for k in ("train", "val", "test")}
    n_cls = int(y_all.max()) + 1
    max_ep, patience = SCHED.get(a.task, (50, 10))
    print(f"{a.task}: {len(X)} windows, {n_cls} classes, "
          + " ".join(f"{k}={len(v)}" for k, v in idx.items()), flush=True)

    model, *_ = load_pretrained(a.checkpoint, dev)
    head = nn.Sequential(nn.BatchNorm1d(model.d_model), nn.Linear(model.d_model, n_cls)).to(dev)
    loader = DataLoader(Windows(X, idx["train"], y_all[idx["train"]]), batch_size=a.batch_size,
                        shuffle=True, drop_last=True, num_workers=4)
    opt = _build_labram_optimizer(model, head, base_lr=a.base_lr, weight_decay=0.05,
                                  layer_decay=0.65, head_lr_mult=1.0)
    sched = _build_cosine_warmup_scheduler(opt, steps_per_epoch=len(loader), warmup_epochs=min(5, max_ep // 4),
                                           total_epochs=max_ep)
    # class-balanced loss: the tasks are imbalanced and the metric is balanced accuracy
    w = np.bincount(y_all[idx["train"]], minlength=n_cls).astype(np.float32)
    crit = nn.CrossEntropyLoss(weight=torch.from_numpy(w.sum() / (n_cls * np.maximum(w, 1))).to(dev))
    best, best_state, bad = -1.0, None, 0
    for ep in range(max_ep):
        model.train(); head.train()
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            loss = crit(head(model._encode(model._tokenize(bx)).mean(1)), by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 3.0)
            opt.step(); sched.step()
        pv = predict(model, head, X, idx["val"], dev)
        v = balanced_accuracy_score(y_all[idx["val"]], pv.argmax(1))
        print(f"  ep{ep + 1:03d} val BA {v:.4f} (best {max(best, v):.4f})", flush=True)
        if v > best:
            best, bad = v, 0
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    pt = predict(model, head, X, idx["test"], dev)
    res = {"task": a.task, "results": {a.name: {"test": score(y_all[idx["test"]], pt.argmax(1), pt, n_cls),
                                                  "val_ba": float(best), "epochs": ep + 1,
                                                  "checkpoint": a.checkpoint}}}
    print(f"-> test {res['results'][a.name]['test']}", flush=True)
    json.dump(res, open(a.output, "w"), indent=2)


if __name__ == "__main__":
    main()
