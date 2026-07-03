"""
ISRUC-Sleep 5-class evaluation (CBraMod-comparable protocol).

  - Subjects: 1-80 train, 81-90 val, 91-100 test (subject-disjoint, fixed)
  - 5 classes: W, N1, N2, N3, REM
  - Input: middle 10 s of each 30 s epoch (matches our pretrain window)
  - Reference: CBraMod BA 0.6655, κ 0.5567, F1 0.6499

Usage:
    python eval_isruc.py \\
        --checkpoint /path/to/best_model.pt \\
        --isruc_dir /home/pxieaf/home2/datasets/isruc/subgroupI \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --normalization per_recording_robust \\
        --trial_duration_s 10 \\
        --include_random_baseline
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

from dataset_isruc import (
    ISRUCDataset, CBRAMOD_ISRUC_SPLITS, CBRAMOD_ISRUC_III_SPLITS,
    LABEL_NAMES, N_CLASSES,
)
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

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=64, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=128, shuffle=False,
    )

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
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs, pct_start=0.2,
    )
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
            logits = head(feats)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 3.0)
            optimizer.step()
            scheduler.step()

        model.eval(); head.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                feats = model._encode(model._tokenize(bx)).mean(1)
                val_preds.append(head(feats).argmax(-1).cpu())
                val_labels.append(by)
        val_ba = balanced_accuracy_score(
            torch.cat(val_labels).numpy(),
            torch.cat(val_preds).numpy(),
        )

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "head":  {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1
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
            batch = X_te[i:i+128]
            feats = model._encode(model._tokenize(batch)).mean(1)
            logits = head(feats)
            preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)

    metrics = compute_metrics(y_te_np, preds)
    metrics["best_val_ba"] = float(best_val_ba)
    metrics["epochs"] = int(ep + 1)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--isruc_dir", required=True,
                   help="Path to ISRUC subgroup root "
                        "(contains subject dirs 1/, 2/, ... — "
                        "e.g. subgroupIII_official/ISRUC-Sleep-III/)")
    p.add_argument("--subgroup", choices=["I", "III"], default="III",
                   help="ISRUC cohort: I (100 subj, 80/10/10) or III "
                        "(10 healthy, 6/2/2). Default III = standard "
                        "LaBraM/CBraMod/CSBrain benchmark.")
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10,
                   help="Middle window length per 30-s epoch.")
    p.add_argument("--normalization", default="per_recording_robust",
                   choices=["per_trial_zscore", "per_recording_robust"])
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--n_reps", type=int, default=3)
    p.add_argument("--include_random_baseline", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    # Select subject-disjoint split based on subgroup
    if args.subgroup == "III":
        splits = CBRAMOD_ISRUC_III_SPLITS
    else:
        splits = CBRAMOD_ISRUC_SPLITS

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu")

    print(f"\n{'='*72}")
    print(f"  ISRUC Sleep 5-class eval (subgroup {args.subgroup})")
    print(f"{'='*72}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Split: train {splits['train']} / val {splits['val']} / test {splits['test']}")
    print(f"  Reference (subgroup III, published):")
    print(f"    LaBraM 0.7527 / CBraMod 0.7865 / CSBrain 0.7925 (BA)")

    model, model_cls, model_type_name, n_channels, ckpt_args = \
        load_pretrained(args.checkpoint, device)

    print(f"\n--- Loading ISRUC splits ---")
    t0 = time.time()
    train_ds = ISRUCDataset(
        data_dir=args.isruc_dir,
        subjects=splits["train"],
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        normalization=args.normalization,
        cache_dir=args.cache_dir,
    )
    val_ds = ISRUCDataset(
        data_dir=args.isruc_dir,
        subjects=splits["val"],
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        normalization=args.normalization,
        cache_dir=args.cache_dir,
    )
    test_ds = ISRUCDataset(
        data_dir=args.isruc_dir,
        subjects=splits["test"],
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        normalization=args.normalization,
        cache_dir=args.cache_dir,
    )
    print(f"Data loaded in {(time.time()-t0)/60:.1f} min")

    X_tr,  y_tr  = dataset_to_xy(train_ds)
    X_val, y_val = dataset_to_xy(val_ds)
    X_te,  y_te  = dataset_to_xy(test_ds)
    print(f"\nShapes: train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")

    results = {
        "checkpoint": args.checkpoint,
        "model_type": model_type_name,
        "n_channels": int(n_channels),
        "split": splits,
        "n_train": int(len(y_tr)),
        "n_val":   int(len(y_val)),
        "n_test":  int(len(y_te)),
        "n_classes": N_CLASSES,
        "ckpt_args": {k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
                          else str(v))
                      for k, v in ckpt_args.items()},
        "reference": {
            "CBraMod": {"BAcc": 0.6655, "kappa": 0.5567, "F1": 0.6499},
        },
    }

    def _agg(reps):
        keys = list(reps[0].keys())
        agg = {}
        for k in keys:
            vals = [m[k] for m in reps if isinstance(m.get(k), (int, float))]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)),
                          "std":  float(np.std(vals))}
        agg["_per_rep"] = reps
        return agg

    # ─── JEPA FT ───
    print(f"\n{'='*72}")
    print(f"  JEPA Fine-tune ({args.n_reps} reps × {args.max_epochs} epochs)")
    print(f"{'='*72}")
    jepa_reps = []
    for rep in range(args.n_reps):
        torch.manual_seed(42 + rep)
        np.random.seed(42 + rep)
        print(f"  Rep {rep+1}/{args.n_reps}")
        m = run_finetune(model, X_tr, y_tr, X_val, y_val, X_te, y_te,
                         device, max_epochs=args.max_epochs)
        print(f"    BA={m['balanced_accuracy']:.4f}  κ={m['cohen_kappa']:.4f}  "
              f"wF1={m['weighted_f1']:.4f}  best_val_ba={m['best_val_ba']:.4f}  "
              f"ep={m['epochs']}")
        jepa_reps.append(m)
    results["jepa_finetune"] = _agg(jepa_reps)

    # ─── Random ───
    if args.include_random_baseline:
        print(f"\n{'='*72}")
        print(f"  Random init Fine-tune ({args.n_reps} reps)")
        print(f"{'='*72}")
        rand_reps = []
        for rep in range(args.n_reps):
            torch.manual_seed(42 + rep)
            np.random.seed(42 + rep)
            print(f"  Rep {rep+1}/{args.n_reps}")
            rand_model = build_random_init(model_cls, n_channels, ckpt_args, device)
            m = run_finetune(rand_model, X_tr, y_tr, X_val, y_val, X_te, y_te,
                             device, max_epochs=args.max_epochs)
            print(f"    BA={m['balanced_accuracy']:.4f}  κ={m['cohen_kappa']:.4f}  "
                  f"wF1={m['weighted_f1']:.4f}  best_val_ba={m['best_val_ba']:.4f}  "
                  f"ep={m['epochs']}")
            rand_reps.append(m)
        results["random_finetune"] = _agg(rand_reps)

    # ─── Summary ───
    print(f"\n{'='*72}")
    print(f"  SUMMARY (ISRUC 5-class, CBraMod split)")
    print(f"{'='*72}")
    print(f"  {'Model':24s} {'BA':>10s} {'κ':>10s} {'wF1':>10s}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'CBraMod (published)':24s} {0.6655:>10.4f} {0.5567:>10.4f} {0.6499:>10.4f}")
    if "random_finetune" in results:
        ba = results["random_finetune"]["balanced_accuracy"]["mean"]
        k  = results["random_finetune"]["cohen_kappa"]["mean"]
        f1 = results["random_finetune"]["weighted_f1"]["mean"]
        print(f"  {'Random init (Ours)':24s} {ba:>10.4f} {k:>10.4f} {f1:>10.4f}")
    ba = results["jepa_finetune"]["balanced_accuracy"]["mean"]
    k  = results["jepa_finetune"]["cohen_kappa"]["mean"]
    f1 = results["jepa_finetune"]["weighted_f1"]["mean"]
    print(f"  {'JEPA (Ours)':24s} {ba:>10.4f} {k:>10.4f} {f1:>10.4f}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  vs CBraMod published: Δ BA = {ba - 0.6655:+.4f}")

    # ─── Save ───
    if args.output is None:
        out = Path(args.checkpoint).parent / "isruc_eval.json"
    else:
        out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n→ Saved: {out}")


if __name__ == "__main__":
    main()
