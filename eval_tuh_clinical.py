"""TUAB / TUEV downstream evaluation for our EEG-LeJEPA family.

Mirrors run_eegbench.py structure (model loading, frozen probe via
LogisticRegression on pooled encoder features, fine-tune with BN+Linear
head + OneCycleLR + class-weighted CE + early stopping on val BA), but
runs on TUAB or TUEV instead of EEG-Bench BCI tasks.

Frozen-probe mode is unique to us (LaBraM does not report it on
TUAB/TUEV); fine-tune mode mirrors LaBraM's protocol for direct
comparison to LaBraM Table 4 (TUAB) and Table 5 (TUEV).

Metrics:
    TUAB (binary):  Balanced Accuracy, ROC-AUC, PR-AUC
    TUEV (6-class): Balanced Accuracy, Cohen's Kappa, weighted F1

Usage:
    # Frozen probe + fine-tune on TUAB
    CUDA_VISIBLE_DEVICES=0 python eval_tuh_clinical.py \\
        --dataset tuab \\
        --checkpoint /home/pxieaf/home2/model/eeg_lejepa_outputcf_sigreg_l01_w1/best_model.pt \\
        --tuh_dir /home/pxieaf/home2/tuh/tuh_eeg_abnormal/v3.0.1/edf \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --output /home/pxieaf/home2/eval/outputcf_sigreg_l01_tuab.json

    # Same for TUEV
    CUDA_VISIBLE_DEVICES=0 python eval_tuh_clinical.py \\
        --dataset tuev \\
        --checkpoint <ckpt> \\
        --tuh_dir /home/pxieaf/home2/tuh/tuh_eeg_events/v2.0.1/edf \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --output /home/pxieaf/home2/eval/outputcf_sigreg_l01_tuev.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

BRAIN_WM_DIR = os.environ.get("BRAIN_WM_DIR", str(Path(__file__).parent))
sys.path.insert(0, BRAIN_WM_DIR)

from eeg_jepa import EEGJEPA
from eeg_mae import EEGMAE
from eeg_lejepa import EEGLeJEPA
from eeg_lejepa_spectral import EEGLeJEPASpectral
from eeg_lejepa_region import EEGLeJEPARegion
from eeg_lejepa_full import EEGLeJEPAFull
from eeg_lejepa_crossfreq import EEGLeJEPACrossFreq
from eeg_lejepa_multistream import EEGLeJEPAMultiStream
from eeg_lejepa_outputcf import EEGLeJEPAOutputCF

from dataset_tuh_clinical import TUABDataset, TUEVDataset, TUEV_LABEL_NAMES


# ============================================================
# Model loading (mirrors run_eegbench.py logic)
# ============================================================

TYPE_MAP = {
    "mae":                (EEGMAE,                "EEG-MAE"),
    "lejepa_full":        (EEGLeJEPAFull,         "EEG-LeJEPA+Full"),
    "lejepa_crossfreq":   (EEGLeJEPACrossFreq,    "EEG-LeJEPA+CrossFreq"),
    "lejepa_multistream": (EEGLeJEPAMultiStream,  "EEG-LeJEPA+MultiStream"),
    "lejepa_outputcf":    (EEGLeJEPAOutputCF,     "EEG-LeJEPA+OutputCF"),
    "lejepa_spectral":    (EEGLeJEPASpectral,     "EEG-LeJEPA+Spectral"),
    "lejepa_region":      (EEGLeJEPARegion,       "EEG-LeJEPA+Region"),
    "lejepa":             (EEGLeJEPA,             "EEG-LeJEPA"),
    "jepa":               (EEGJEPA,               "EEG-JEPA"),
}


def load_pretrained(checkpoint_path: str, device: torch.device):
    """Returns (model, model_cls, model_type_name, n_channels, ckpt_args)."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Detect n_channels from any channel-embed-like weight
    n_channels = 64
    for key, val in ckpt["model_state_dict"].items():
        if "channel_embed" in key:
            n_channels = val.shape[0]
            break

    model_type = ckpt_args.get("model", "jepa")
    if model_type in TYPE_MAP:
        model_cls, model_type_name = TYPE_MAP[model_type]
    else:
        # Fallback by inspecting keys (rare; old checkpoints)
        keys_str = str(ckpt["model_state_dict"].keys())
        if "reconstruction_head" in keys_str:
            model_cls, model_type_name = EEGMAE, "EEG-MAE"
        elif "band_head" in keys_str:
            model_cls, model_type_name = EEGLeJEPAOutputCF, "EEG-LeJEPA+OutputCF"
        elif "freq_predictor" in keys_str:
            model_cls, model_type_name = EEGLeJEPACrossFreq, "EEG-LeJEPA+CrossFreq"
        else:
            model_cls, model_type_name = EEGLeJEPA, "EEG-LeJEPA"

    model_kwargs = dict(
        n_channels=n_channels,
        d_model=ckpt_args.get("d_model", 256),
        encoder_layers=ckpt_args.get("encoder_layers", 6),
    )
    if "n_queries" in ckpt_args:
        model_kwargs["n_queries"] = ckpt_args["n_queries"]
    if model_type in ("lejepa_crossfreq", "lejepa_full",
                      "lejepa_multistream", "lejepa_outputcf"):
        model_kwargs["cf_band_conditioned"] = bool(
            ckpt_args.get("cf_band_conditioned", 0))
        model_kwargs["cf_preserve_spatial"] = bool(
            ckpt_args.get("cf_preserve_spatial", 0))
        if "cf_d_band" in ckpt_args:
            model_kwargs["cf_d_band"] = ckpt_args["cf_d_band"]

    model = model_cls(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model: {model_type_name}, {n_channels}ch, "
          f"d={ckpt_args.get('d_model', 256)}")
    return model, model_cls, model_type_name, n_channels, ckpt_args


def build_random_init(model_cls, n_channels, ckpt_args, device):
    """Build an UNTRAINED model with the same config for the random baseline."""
    model_kwargs = dict(
        n_channels=n_channels,
        d_model=ckpt_args.get("d_model", 256),
        encoder_layers=ckpt_args.get("encoder_layers", 6),
    )
    if "n_queries" in ckpt_args:
        model_kwargs["n_queries"] = ckpt_args["n_queries"]
    if "cf_band_conditioned" in ckpt_args:
        model_kwargs["cf_band_conditioned"] = bool(ckpt_args["cf_band_conditioned"])
    if "cf_preserve_spatial" in ckpt_args:
        model_kwargs["cf_preserve_spatial"] = bool(ckpt_args["cf_preserve_spatial"])
    if "cf_d_band" in ckpt_args:
        model_kwargs["cf_d_band"] = ckpt_args["cf_d_band"]
    return model_cls(**model_kwargs).to(device)


# ============================================================
# Feature extraction & channel adaptation
# ============================================================

def _pad_or_trim_channels(X: np.ndarray, target_n_ch: int) -> np.ndarray:
    """X: [N, T, C] → [N, T, target_n_ch] via zero-pad or trim."""
    n_ch = X.shape[-1]
    if n_ch == target_n_ch:
        return X
    if n_ch > target_n_ch:
        return X[..., :target_n_ch]
    return np.pad(X, ((0, 0), (0, 0), (0, target_n_ch - n_ch)))


def dataset_to_xy(ds, target_n_ch: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert {trials, labels} dataset to (X [N,T,C] float32, y [N] int)."""
    X = np.stack(ds.trials).astype(np.float32)
    X = _pad_or_trim_channels(X, target_n_ch)
    y = np.array(ds.labels, dtype=np.int64)
    return X, y


def extract_features(model, X_np, device, batch_size=64):
    """Run encoder over X (mean-pool over tokens) → [N, d_model]."""
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            batch = torch.from_numpy(X_np[i:i+batch_size]).to(device)
            tokens = model._tokenize(batch)
            encoded = model._encode(tokens)
            feats.append(encoded.mean(dim=1).cpu().numpy())
    return np.concatenate(feats)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred, y_proba=None, dataset: str = "tuab"):
    """Returns dict of {metric_name: float}.

    y_pred:  class predictions [N]
    y_proba: class probabilities [N, n_classes] (only used by TUAB AUC)
    """
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if dataset == "tuab":
        if y_proba is not None:
            # Binary case: y_proba[:, 1] is prob of class 1 (abnormal)
            pos_prob = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, pos_prob))
                out["pr_auc"] = float(average_precision_score(y_true, pos_prob))
            except ValueError:
                out["roc_auc"] = float("nan")
                out["pr_auc"] = float("nan")
    elif dataset == "tuev":
        out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        out["weighted_f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return out


# ============================================================
# Frozen probe
# ============================================================

def run_frozen_probe(
    feat_tr, y_tr, feat_te, y_te, n_classes: int, dataset: str, n_reps: int,
) -> dict:
    """LogisticRegression × n_reps, returns mean metrics + per-rep list."""
    metrics_by_rep = []
    for seed in range(n_reps):
        scaler = StandardScaler()
        tr_s = scaler.fit_transform(feat_tr)
        te_s = scaler.transform(feat_te)
        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            multi_class="multinomial" if n_classes > 2 else "auto",
            random_state=42 + seed,
            class_weight="balanced",
        )
        clf.fit(tr_s, y_tr)
        preds = clf.predict(te_s)
        proba = clf.predict_proba(te_s)
        metrics_by_rep.append(
            compute_metrics(y_te, preds, proba, dataset=dataset))

    # Aggregate mean ± std for each metric
    keys = list(metrics_by_rep[0].keys())
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_by_rep]
        agg[k] = {"mean": float(np.mean(vals)),
                  "std": float(np.std(vals))}
    agg["_per_rep"] = metrics_by_rep
    return agg


# ============================================================
# Fine-tune (matches run_eegbench.py protocol)
# ============================================================

def run_finetune(
    base_model, X_tr_np, y_tr_np, X_te_np, y_te_np,
    n_classes: int, dataset: str, device, max_epochs: int = 50,
) -> dict:
    """Fine-tune backbone + new BN+Linear head, return test metrics.

    Protocol mirrors run_eegbench.py:
      - 85/15 train/val split for early stopping (patience=10)
      - OneCycleLR with max_lr=[4e-4 backbone, 4e-3 head]
      - AdamW(wd=0.01), grad clip 3.0
      - Class-weighted CE
      - Eval on test set with best-val checkpoint
    """
    model = copy.deepcopy(base_model)
    head = nn.Sequential(
        nn.BatchNorm1d(model.d_model),
        nn.Linear(model.d_model, n_classes),
    ).to(device)

    X_tr = torch.from_numpy(X_tr_np)
    y_tr = torch.from_numpy(y_tr_np).long()
    X_te = torch.from_numpy(X_te_np).to(device)

    n_total = len(X_tr)
    n_val = max(1, int(n_total * 0.15))
    n_train = n_total - n_val
    perm = torch.randperm(n_total)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_tr[train_idx], y_tr[train_idx]),
        batch_size=32, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_tr[val_idx], y_tr[val_idx]),
        batch_size=64, shuffle=False,
    )

    class_counts = torch.bincount(y_tr[train_idx], minlength=n_classes)
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    class_weights = (class_weights / class_weights.sum() * n_classes).to(device)

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
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state["model"]); model.to(device)
        head.load_state_dict(best_state["head"]);   head.to(device)
    model.eval(); head.eval()

    # Test
    preds, probas = [], []
    with torch.no_grad():
        for i in range(0, len(X_te), 64):
            batch = X_te[i:i+64]
            feats = model._encode(model._tokenize(batch)).mean(1)
            logits = head(feats)
            probas.append(torch.softmax(logits, dim=-1).cpu().numpy())
            preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    proba = np.concatenate(probas)

    out = compute_metrics(y_te_np, preds, proba, dataset=dataset)
    out["best_val_ba"] = float(best_val_ba)
    out["epochs_trained"] = int(ep + 1)

    del model, head, best_state
    torch.cuda.empty_cache()
    return out


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tuab", "tuev"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tuh_dir", type=str, required=True,
                   help="Path to TUAB edf/ or TUEV edf/ root (contains train/ and eval/)")
    p.add_argument("--cache_dir", type=str,
                   default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--mode", choices=["frozen", "finetune", "both"], default="both")
    p.add_argument("--n_reps", type=int, default=5,
                   help="Repetitions for frozen probe (different LR seeds)")
    p.add_argument("--max_epochs", type=int, default=50,
                   help="Fine-tune max epochs")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default=None,
                   help="JSON output path; if None, derive from checkpoint")
    p.add_argument("--include_random_baseline", action="store_true",
                   help="Also eval a fresh-random-init model with same config")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu")

    n_classes = 2 if args.dataset == "tuab" else 6

    # ── Load pretrained ──
    model, model_cls, model_type_name, n_channels, ckpt_args = \
        load_pretrained(args.checkpoint, device)

    # ── Load datasets ──
    DSCls = TUABDataset if args.dataset == "tuab" else TUEVDataset

    print(f"\n--- Loading {args.dataset.upper()} train ---")
    t0 = time.time()
    train_ds = DSCls(
        data_dir=args.tuh_dir, split="train",
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        cache_dir=args.cache_dir,
    )
    print(f"--- Loading {args.dataset.upper()} eval ---")
    eval_ds = DSCls(
        data_dir=args.tuh_dir, split="eval",
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        cache_dir=args.cache_dir,
    )
    print(f"Data loaded in {(time.time()-t0)/60:.1f} min")

    X_tr, y_tr = dataset_to_xy(train_ds, n_channels)
    X_te, y_te = dataset_to_xy(eval_ds,  n_channels)
    print(f"\nShapes: train {X_tr.shape}, eval {X_te.shape}")
    print(f"Class counts (train): {np.bincount(y_tr, minlength=n_classes)}")
    print(f"Class counts (eval):  {np.bincount(y_te, minlength=n_classes)}")

    results = {
        "checkpoint": args.checkpoint,
        "model_type": model_type_name,
        "dataset": args.dataset,
        "n_classes": n_classes,
        "n_train": int(len(y_tr)),
        "n_eval":  int(len(y_te)),
        "n_channels": int(n_channels),
        "ckpt_args": {k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
                          else str(v))
                      for k, v in ckpt_args.items()},
    }

    # ── Frozen probe ──
    if args.mode in ("frozen", "both"):
        print(f"\n{'='*70}\n  Frozen probe ({args.n_reps} reps)\n{'='*70}")
        print("  [JEPA] Extracting features...")
        feat_tr = extract_features(model, X_tr, device)
        feat_te = extract_features(model, X_te, device)
        results["jepa_frozen"] = run_frozen_probe(
            feat_tr, y_tr, feat_te, y_te, n_classes, args.dataset, args.n_reps)
        _print_metric_line("  JEPA frozen ", results["jepa_frozen"], args.dataset)

        if args.include_random_baseline:
            print("  [Random] Extracting features (untrained encoder)...")
            random_model = build_random_init(model_cls, n_channels, ckpt_args, device)
            r_tr = extract_features(random_model, X_tr, device)
            r_te = extract_features(random_model, X_te, device)
            del random_model; torch.cuda.empty_cache()
            results["random_frozen"] = run_frozen_probe(
                r_tr, y_tr, r_te, y_te, n_classes, args.dataset, args.n_reps)
            _print_metric_line("  Rand frozen ", results["random_frozen"], args.dataset)

    # ── Fine-tune ──
    if args.mode in ("finetune", "both"):
        print(f"\n{'='*70}\n  Fine-tune ({args.max_epochs} epochs)\n{'='*70}")
        print("  [JEPA] Fine-tuning...")
        results["jepa_finetune"] = run_finetune(
            model, X_tr, y_tr, X_te, y_te,
            n_classes, args.dataset, device, args.max_epochs)
        _print_metric_line_ft("  JEPA-FT ", results["jepa_finetune"], args.dataset)

        if args.include_random_baseline:
            print("  [Random] Fine-tuning from scratch...")
            random_model = build_random_init(model_cls, n_channels, ckpt_args, device)
            results["random_finetune"] = run_finetune(
                random_model, X_tr, y_tr, X_te, y_te,
                n_classes, args.dataset, device, args.max_epochs)
            _print_metric_line_ft("  Rand-FT ", results["random_finetune"], args.dataset)
            del random_model; torch.cuda.empty_cache()

    # ── Save ──
    out_path = args.output
    if out_path is None:
        ckpt_stem = Path(args.checkpoint).parent.name  # use model dir name
        out_path = f"/home/pxieaf/home2/eval/{ckpt_stem}_{args.dataset}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved: {out_path}")


def _print_metric_line(prefix, agg, dataset):
    """Pretty-print frozen-probe aggregate dict."""
    if dataset == "tuab":
        ba = agg["balanced_accuracy"]
        roc = agg["roc_auc"]
        pr = agg["pr_auc"]
        print(f"{prefix}BA={ba['mean']:.3f}±{ba['std']:.3f}  "
              f"ROC-AUC={roc['mean']:.3f}±{roc['std']:.3f}  "
              f"PR-AUC={pr['mean']:.3f}±{pr['std']:.3f}")
    else:
        ba = agg["balanced_accuracy"]
        ck = agg["cohen_kappa"]
        f1 = agg["weighted_f1"]
        print(f"{prefix}BA={ba['mean']:.3f}±{ba['std']:.3f}  "
              f"κ={ck['mean']:.3f}±{ck['std']:.3f}  "
              f"wF1={f1['mean']:.3f}±{f1['std']:.3f}")


def _print_metric_line_ft(prefix, out, dataset):
    """Pretty-print fine-tune single-run dict."""
    if dataset == "tuab":
        print(f"{prefix}BA={out['balanced_accuracy']:.3f}  "
              f"ROC-AUC={out['roc_auc']:.3f}  PR-AUC={out['pr_auc']:.3f}  "
              f"(best_val_ba={out['best_val_ba']:.3f}, "
              f"epochs={out['epochs_trained']})")
    else:
        print(f"{prefix}BA={out['balanced_accuracy']:.3f}  "
              f"κ={out['cohen_kappa']:.3f}  wF1={out['weighted_f1']:.3f}  "
              f"(best_val_ba={out['best_val_ba']:.3f}, "
              f"epochs={out['epochs_trained']})")


if __name__ == "__main__":
    main()
