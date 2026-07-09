"""HMC sleep-staging 5-class evaluation (mirrors eval_isruc).

  - 154 PSG recordings, AASM 5-class (W, N1, N2, N3, REM), 30-s epochs
  - Input: middle 10 s of each epoch (matches our pretrain window)
  - Split: subject(recording)-disjoint 70/15/15, deterministic (--seed)
  - Reference (VERIFY): CBraMod / Conformer ~0.71 BA on HMC

Usage:
    python eval_hmc.py \\
        --checkpoint /path/to/checkpoint_ep8.pt \\
        --hmc_dir /home/pxieaf/home2/datasets/HMC \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --n_reps 3 --include_random_baseline
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, f1_score,
)

from dataset_hmc import HMCDataset, make_hmc_split, LABEL_NAMES, N_CLASSES
from eval_tuh_clinical import load_pretrained, build_random_init


def dataset_to_xy(ds):
    X = np.stack(ds.trials).astype(np.float32)
    y = np.array(ds.labels, dtype=np.int64)
    return X, y


def compute_metrics(y_true, y_pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "weighted_f1":       float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro")),
    }


def run_finetune(base_model, X_tr_np, y_tr_np, X_val_np, y_val_np,
                 X_te_np, y_te_np, device, max_epochs: int = 50) -> dict:
    model = copy.deepcopy(base_model)
    head = nn.Sequential(
        nn.BatchNorm1d(model.d_model),
        nn.Linear(model.d_model, N_CLASSES),
    ).to(device)

    X_tr  = torch.from_numpy(X_tr_np)
    y_tr  = torch.from_numpy(y_tr_np).long()
    X_val = torch.from_numpy(X_val_np)
    y_val = torch.from_numpy(y_val_np).long()
    X_te  = torch.from_numpy(X_te_np).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=128, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=256, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    class_counts = torch.bincount(y_tr, minlength=N_CLASSES)
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    class_weights = (class_weights / class_weights.sum() * N_CLASSES).to(device)

    steps_per_epoch = max(1, len(train_loader))
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1e-6},
        {"params": head.parameters(),  "lr": 1e-6},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[4e-4, 4e-3],
        steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_ba = 0.0
    best_state = None
    patience = 10
    no_improve = 0
    ep = 0

    for ep in range(max_epochs):
        model.train(); head.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            feats = model._encode(model._tokenize(bx)).mean(1)
            loss = criterion(head(feats), by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 3.0)
            optimizer.step()
            scheduler.step()

        model.eval(); head.eval()
        vp, vl = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                feats = model._encode(model._tokenize(bx)).mean(1)
                vp.append(head(feats).argmax(-1).cpu()); vl.append(by)
        val_ba = balanced_accuracy_score(torch.cat(vl).numpy(),
                                         torch.cat(vp).numpy())
        improved = val_ba > best_val_ba
        if improved:
            best_val_ba = val_ba
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "head":  {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1
        print(f"      ep{ep+1:03d}{'*' if improved else ' '} val_ba={val_ba:.4f} "
              f"best={best_val_ba:.4f} no_improve={no_improve}/{patience}", flush=True)
        if no_improve >= patience:
            print(f"      early stop at epoch {ep+1} (best_val_ba={best_val_ba:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"]); model.to(device)
        head.load_state_dict(best_state["head"]);   head.to(device)
    model.eval(); head.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 128):
            feats = model._encode(model._tokenize(X_te[i:i+128])).mean(1)
            preds.append(head(feats).argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    metrics = compute_metrics(y_te_np, preds)
    metrics["best_val_ba"] = float(best_val_ba)
    metrics["epochs"] = int(ep + 1)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hmc_dir", required=True,
                   help="HMC root (contains recordings/SN*.edf + "
                        "SN*_sleepscoring.txt)")
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust",
                   choices=["per_trial_zscore", "per_recording_robust"])
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--n_reps", type=int, default=3)
    p.add_argument("--include_random_baseline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu")

    splits = make_hmc_split(args.hmc_dir, seed=args.seed)
    print(f"\n{'='*72}\n  HMC Sleep 5-class eval\n{'='*72}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Split (seed {args.seed}): train {len(splits['train'])} / "
          f"val {len(splits['val'])} / test {len(splits['test'])} recordings")

    model, model_cls, model_type_name, n_channels, ckpt_args = \
        load_pretrained(args.checkpoint, device)

    print(f"\n--- Loading HMC splits ---")
    t0 = time.time()
    dsk = dict(sample_rate=args.sample_rate, trial_duration_s=args.trial_duration_s,
               normalization=args.normalization, cache_dir=args.cache_dir)
    train_ds = HMCDataset(args.hmc_dir, splits["train"], **dsk)
    val_ds   = HMCDataset(args.hmc_dir, splits["val"],   **dsk)
    test_ds  = HMCDataset(args.hmc_dir, splits["test"],  **dsk)
    print(f"Data loaded in {(time.time()-t0)/60:.1f} min")

    X_tr, y_tr = dataset_to_xy(train_ds)
    X_val, y_val = dataset_to_xy(val_ds)
    X_te, y_te = dataset_to_xy(test_ds)
    print(f"\nShapes: train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")

    results = {
        "checkpoint": args.checkpoint, "model_type": model_type_name,
        "seed": int(args.seed), "split": splits,
        "n_train": int(len(y_tr)), "n_val": int(len(y_val)), "n_test": int(len(y_te)),
        "n_classes": N_CLASSES,
        "reference": {"CBraMod_BA": 0.71},   # VERIFY
    }

    def _agg(reps):
        agg = {}
        for k in reps[0]:
            vals = [m[k] for m in reps if isinstance(m.get(k), (int, float))]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        agg["_per_rep"] = reps
        return agg

    print(f"\n{'='*72}\n  JEPA Fine-tune ({args.n_reps} reps × {args.max_epochs} ep)\n{'='*72}")
    jreps = []
    for rep in range(args.n_reps):
        torch.manual_seed(args.seed + rep); np.random.seed(args.seed + rep)
        print(f"  Rep {rep+1}/{args.n_reps}")
        m = run_finetune(model, X_tr, y_tr, X_val, y_val, X_te, y_te, device, args.max_epochs)
        print(f"    BA={m['balanced_accuracy']:.4f}  κ={m['cohen_kappa']:.4f}  "
              f"wF1={m['weighted_f1']:.4f}")
        jreps.append(m)
    results["jepa_finetune"] = _agg(jreps)

    if args.include_random_baseline:
        print(f"\n{'='*72}\n  Random init Fine-tune ({args.n_reps} reps)\n{'='*72}")
        rreps = []
        for rep in range(args.n_reps):
            torch.manual_seed(args.seed + rep); np.random.seed(args.seed + rep)
            print(f"  Rep {rep+1}/{args.n_reps}")
            rm = build_random_init(model_cls, n_channels, ckpt_args, device)
            m = run_finetune(rm, X_tr, y_tr, X_val, y_val, X_te, y_te, device, args.max_epochs)
            print(f"    BA={m['balanced_accuracy']:.4f}  κ={m['cohen_kappa']:.4f}  "
                  f"wF1={m['weighted_f1']:.4f}")
            rreps.append(m)
        results["random_finetune"] = _agg(rreps)

    print(f"\n{'='*72}\n  SUMMARY (HMC 5-class)\n{'='*72}")
    print(f"  {'Model':24s} {'BA':>10s} {'κ':>10s} {'wF1':>10s}")
    if "random_finetune" in results:
        r = results["random_finetune"]
        print(f"  {'Random init (Ours)':24s} {r['balanced_accuracy']['mean']:>10.4f} "
              f"{r['cohen_kappa']['mean']:>10.4f} {r['weighted_f1']['mean']:>10.4f}")
    j = results["jepa_finetune"]
    print(f"  {'JEPA (Ours)':24s} {j['balanced_accuracy']['mean']:>10.4f} "
          f"{j['cohen_kappa']['mean']:>10.4f} {j['weighted_f1']['mean']:>10.4f}")

    out = (Path(args.checkpoint).parent / "hmc_eval.json"
           if args.output is None else Path(args.output))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved: {out}")


if __name__ == "__main__":
    main()
