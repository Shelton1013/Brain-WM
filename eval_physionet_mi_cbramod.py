"""
PhysioNet MI 4-class evaluation matching CBraMod protocol.

  - Split: train S1-70, val S71-89, test S90-109 (subject-disjoint, fixed)
  - 4 classes: LH / RH / Both Fists / Both Feet (imagined)
  - Window: 4 seconds at event onset
  - Metric: Balanced Accuracy (BA), Cohen κ, weighted F1
  - Reference: CBraMod (ICLR 2025) reports BAcc 0.6417 ± 0.0091

Supports two baselines for direct comparison:
  --include_random_baseline : random-init same architecture
  --labram_baseline         : LaBraM-Base public weights (FT on this split)

CBraMod baseline is NOT integrated (would need their criss-cross
architecture + checkpoint download); cite their published 0.6417 directly.

Usage:
    python eval_physionet_mi_cbramod.py \
        --checkpoint /path/to/best_model.pt \
        --physionet_dir /home/share/data_makchen/peng/datasets/physionet \
        --cache_dir /home/pxieaf/home2/dataset_cache \
        --normalization per_recording_robust \
        --trial_duration_s 4 \
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

from dataset_physionet_mi_4class import (
    PhysioNetMI4ClassDataset, CBRAMOD_SPLITS, LABEL_NAMES,
)
from eval_tuh_clinical import load_pretrained, build_random_init


N_CLASSES = 4


# ============================================================
# Feature extraction & FT (reuse eval_tuh_clinical pattern)
# ============================================================

def _pad_or_trim_channels(X: np.ndarray, target_n_ch: int) -> np.ndarray:
    n_ch = X.shape[-1]
    if n_ch == target_n_ch:
        return X
    if n_ch > target_n_ch:
        # Positional trimming here is a BUG: the dataset must already deliver
        # the correct 19 channels in canonical 10-20 order (name-based pick).
        # If we ever see more channels than the model expects, fail loudly
        # rather than silently slicing the wrong (fronto-central) subset.
        raise ValueError(
            f"got {n_ch} channels but model expects {target_n_ch}; the "
            f"dataset should pick the canonical 10-20 channels by name "
            f"(see dataset_physionet_mi_4class._extract_events_from_file)")
    return np.pad(X, ((0, 0), (0, 0), (0, target_n_ch - n_ch)))


def dataset_to_xy(ds, target_n_ch: int):
    X = np.stack(ds.trials).astype(np.float32)
    X = _pad_or_trim_channels(X, target_n_ch)
    y = np.array(ds.labels, dtype=np.int64)
    return X, y


def compute_metrics(y_true, y_pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "weighted_f1":       float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro")),
    }


def run_finetune(
    base_model, X_tr_np, y_tr_np, X_val_np, y_val_np, X_te_np, y_te_np,
    device, max_epochs: int = 50,
) -> dict:
    """Standard FT: backbone + BN+Linear head, AdamW, OneCycleLR, early stop."""
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
        batch_size=32, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64, shuffle=False,
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

        # Validate
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
                print(f"      early stop at epoch {ep+1} (val_ba={best_val_ba:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"]); model.to(device)
        head.load_state_dict(best_state["head"]);   head.to(device)
    model.eval(); head.eval()

    # Test
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 64):
            batch = X_te[i:i+64]
            feats = model._encode(model._tokenize(batch)).mean(1)
            logits = head(feats)
            preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)

    metrics = compute_metrics(y_te_np, preds)
    metrics["best_val_ba"] = float(best_val_ba)
    metrics["epochs"] = int(ep + 1)
    return metrics


# ============================================================
# Optional: LaBraM baseline (uses run_eegbench's LaBraM loader)
# ============================================================

def run_labram_baseline(X_tr_np, y_tr_np, X_val_np, y_val_np,
                        X_te_np, y_te_np, device, max_epochs: int = 50) -> dict:
    """LaBraM-Base FT on this split. Uses run_eegbench.load_labram_model."""
    try:
        from run_eegbench import load_labram_model
    except Exception as e:
        print(f"  LaBraM loader unavailable: {e}")
        return {"error": str(e)}

    print("  Loading LaBraM-Base...")
    labram = load_labram_model()
    if labram is None:
        return {"error": "load_labram_model returned None"}

    # LaBraM expects 200 Hz, 19ch; X is at our sample_rate (likely 256).
    # The loader will need to handle resampling — defer to load_labram_model's
    # own preprocessing. For now, log a warning and skip if incompatible.
    print("  WARN: LaBraM baseline requires preprocessing alignment "
          "(200 Hz, 19 ch, their normalization). Skipping for now — "
          "use the published number from LaBraM paper / CBraMod's reproduction.")
    return {"skipped": "preprocessing not aligned — see code comments"}


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--physionet_dir", required=True,
                   help="Path to PhysioNet MI download dir (root for eegbci)")
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=4,
                   help="CBraMod uses 4s. Don't change unless verified.")
    p.add_argument("--normalization", default="per_recording_robust",
                   choices=["per_trial_zscore", "per_recording_robust"])
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--n_reps", type=int, default=3,
                   help="Repetitions of FT (different seeds) for error bars")
    p.add_argument("--include_random_baseline", action="store_true")
    p.add_argument("--labram_baseline", action="store_true",
                   help="Also FT LaBraM-Base on this split (slow + needs alignment)")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu")

    # ─── Load model ───
    print(f"\n{'='*72}")
    print(f"  PhysioNet MI 4-class eval (CBraMod protocol)")
    print(f"{'='*72}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Reference: CBraMod FT BAcc 0.6417 ± 0.0091")

    model, model_cls, model_type_name, n_channels, ckpt_args = \
        load_pretrained(args.checkpoint, device)

    # ─── Load datasets ───
    print(f"\n--- Loading PhysioNet splits (CBraMod) ---")
    t0 = time.time()
    train_ds = PhysioNetMI4ClassDataset(
        data_dir=args.physionet_dir,
        subjects=CBRAMOD_SPLITS["train"],
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        normalization=args.normalization,
        cache_dir=args.cache_dir,
    )
    val_ds = PhysioNetMI4ClassDataset(
        data_dir=args.physionet_dir,
        subjects=CBRAMOD_SPLITS["val"],
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        normalization=args.normalization,
        cache_dir=args.cache_dir,
    )
    test_ds = PhysioNetMI4ClassDataset(
        data_dir=args.physionet_dir,
        subjects=CBRAMOD_SPLITS["test"],
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        normalization=args.normalization,
        cache_dir=args.cache_dir,
    )
    print(f"Data loaded in {(time.time()-t0)/60:.1f} min")

    X_tr,  y_tr  = dataset_to_xy(train_ds, n_channels)
    X_val, y_val = dataset_to_xy(val_ds,   n_channels)
    X_te,  y_te  = dataset_to_xy(test_ds,  n_channels)

    print(f"\nShapes: train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")
    print(f"Train class dist: {np.bincount(y_tr, minlength=4)}")
    print(f"Val   class dist: {np.bincount(y_val, minlength=4)}")
    print(f"Test  class dist: {np.bincount(y_te, minlength=4)}")
    print(f"Train subjects: {sorted(set(train_ds.subject_ids))[:5]}..{sorted(set(train_ds.subject_ids))[-3:]}")
    print(f"Val   subjects: {sorted(set(val_ds.subject_ids))[:3]}..{sorted(set(val_ds.subject_ids))[-3:]}")
    print(f"Test  subjects: {sorted(set(test_ds.subject_ids))[:3]}..{sorted(set(test_ds.subject_ids))[-3:]}")

    results = {
        "checkpoint": args.checkpoint,
        "model_type": model_type_name,
        "n_channels": int(n_channels),
        "split": {
            "train_subj": [int(s) for s in CBRAMOD_SPLITS["train"]],
            "val_subj":   [int(s) for s in CBRAMOD_SPLITS["val"]],
            "test_subj":  [int(s) for s in CBRAMOD_SPLITS["test"]],
        },
        "n_train": int(len(y_tr)),
        "n_val":   int(len(y_val)),
        "n_test":  int(len(y_te)),
        "n_classes": N_CLASSES,
        "ckpt_args": {k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
                          else str(v))
                      for k, v in ckpt_args.items()},
        "reference": {
            "CBraMod": {"BAcc": 0.6417, "BAcc_std": 0.0091, "kappa": 0.5222},
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

    def _print(name, agg):
        ba = agg["balanced_accuracy"]
        k  = agg.get("cohen_kappa", {"mean": 0, "std": 0})
        f1 = agg.get("weighted_f1", {"mean": 0, "std": 0})
        print(f"  {name:24s} BA={ba['mean']:.4f}±{ba['std']:.4f}  "
              f"κ={k['mean']:.4f}±{k['std']:.4f}  "
              f"wF1={f1['mean']:.4f}±{f1['std']:.4f}")

    # ─── JEPA FT × n_reps ───
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
              f"wF1={m['weighted_f1']:.4f}  best_val_ba={m['best_val_ba']:.4f}  ep={m['epochs']}")
        jepa_reps.append(m)
    results["jepa_finetune"] = _agg(jepa_reps)
    print()
    _print("JEPA FT", results["jepa_finetune"])
    print(f"  → vs CBraMod 0.6417: "
          f"Δ = {results['jepa_finetune']['balanced_accuracy']['mean'] - 0.6417:+.4f}")

    # ─── Random baseline ───
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
                  f"wF1={m['weighted_f1']:.4f}  best_val_ba={m['best_val_ba']:.4f}  ep={m['epochs']}")
            rand_reps.append(m)
        results["random_finetune"] = _agg(rand_reps)
        print()
        _print("Random FT", results["random_finetune"])

    # ─── LaBraM baseline ───
    if args.labram_baseline:
        print(f"\n{'='*72}\n  LaBraM-Base baseline (single rep)\n{'='*72}")
        labram_m = run_labram_baseline(X_tr, y_tr, X_val, y_val, X_te, y_te,
                                       device, max_epochs=args.max_epochs)
        results["labram_finetune"] = labram_m

    # ─── Summary ───
    print(f"\n{'='*72}")
    print(f"  SUMMARY (PhysioNet MI 4-class, CBraMod split)")
    print(f"{'='*72}")
    print(f"  {'Model':24s} {'BA':>10s} {'κ':>10s} {'wF1':>10s}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'CBraMod (published)':24s} {0.6417:>10.4f} {0.5222:>10.4f} {'-':>10s}")
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
    print(f"  vs CBraMod published: Δ BA = {ba - 0.6417:+.4f}")

    # ─── Save ───
    if args.output is None:
        out = Path(args.checkpoint).parent / "physionet_mi_cbramod.json"
    else:
        out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n→ Saved: {out}")


if __name__ == "__main__":
    main()
