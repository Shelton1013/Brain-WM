"""Mumtaz2016 Depression EEG evaluation — binary classification.

  - 34 MDD + 30 Healthy (approx) subjects, 19-ch 10-20 EEG @ 256 Hz
  - Task: binary (MDD vs Healthy control)
  - Split: subject-disjoint, deterministic 60/20/20 per class (--seed)
  - Metrics: BA, ROC-AUC, PR-AUC (matches TUAB reporting)

References (published):
  LaBraM-Base   BA 0.9409
  CBraMod       BA 0.9560
  CSBrain       BA 0.9643
  BIOT          BA 0.9358

Usage:
    python eval_mumtaz.py \\
        --checkpoint /path/to/best_model.pt \\
        --mumtaz_dir /home/pxieaf/home2/datasets/mumtaz \\
        --seed 42 \\
        --include_random_baseline
"""
import argparse
import copy
import json
import random as _pyrandom
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
)

from dataset_mumtaz import (
    MumtazDataset, make_subject_split, make_cbramod_split, LABEL_NAMES, N_CLASSES,
)
from eval_tuh_clinical import (
    load_pretrained, build_random_init,
    _build_labram_optimizer, _build_cosine_warmup_scheduler,
    _inject_drop_path,
)


def dataset_to_xy(ds):
    X = np.stack(ds.trials).astype(np.float32)
    y = np.array(ds.labels, dtype=np.int64)
    return X, y


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc":           float(roc_auc_score(y_true, y_proba)),
        "pr_auc":            float(average_precision_score(y_true, y_proba)),
    }


def extract_features(model, X_np, device, batch_size=64):
    """Run encoder over X (mean-pool tokens) → [N, d_model]."""
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            batch = torch.from_numpy(X_np[i:i+batch_size]).to(device)
            tokens = model._tokenize(batch)
            encoded = model._encode(tokens)
            feats.append(encoded.mean(dim=1).cpu().numpy())
    return np.concatenate(feats)


def run_frozen_probe(feat_tr, y_tr, feat_te, y_te, n_reps: int = 5) -> dict:
    """LogisticRegression × n_reps, returns mean metrics + per-rep list.

    For binary depression detection: features standardized + LogReg,
    reports BA + ROC-AUC + PR-AUC.
    """
    metrics_by_rep = []
    for seed in range(n_reps):
        scaler = StandardScaler()
        tr_s = scaler.fit_transform(feat_tr)
        te_s = scaler.transform(feat_te)
        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            random_state=42 + seed,
            class_weight="balanced",
        )
        clf.fit(tr_s, y_tr)
        preds = clf.predict(te_s)
        proba = clf.predict_proba(te_s)[:, 1]   # prob of MDD
        metrics_by_rep.append(compute_metrics(y_te, preds, proba))

    keys = list(metrics_by_rep[0].keys())
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_by_rep]
        agg[k] = {"mean": float(np.mean(vals)),
                  "std": float(np.std(vals))}
    agg["_per_rep"] = metrics_by_rep
    return agg


def run_finetune(base_model, X_tr_np, y_tr_np, X_val_np, y_val_np,
                 X_te_np, y_te_np, device, max_epochs: int = 50,
                 batch_size: int = 64,
                 patience: int = 10,
                 ft_protocol: str = "onecycle",
                 ft_base_lr: float = 5e-4,
                 ft_weight_decay: float = 0.05,
                 ft_layer_decay: float = 0.65,
                 ft_warmup_epochs: int = 5,
                 ft_drop_path: float = 0.0,
                 ft_head_lr_mult: float = 1.0) -> dict:
    model = copy.deepcopy(base_model)
    # Inject drop_path if requested
    if ft_drop_path > 0:
        _inject_drop_path(model, ft_drop_path)
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
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size * 2, shuffle=False,
    )

    class_counts = torch.bincount(y_tr, minlength=N_CLASSES)
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    class_weights = (class_weights / class_weights.sum() * N_CLASSES).to(device)

    steps_per_epoch = max(1, len(train_loader))
    if ft_protocol == "labram":
        # Cosine + warmup + layer_decay (protective for pretrained)
        optimizer = _build_labram_optimizer(
            model, head, base_lr=ft_base_lr,
            weight_decay=ft_weight_decay,
            layer_decay=ft_layer_decay,
            head_lr_mult=ft_head_lr_mult,
        )
        scheduler = _build_cosine_warmup_scheduler(
            optimizer, steps_per_epoch=steps_per_epoch,
            warmup_epochs=ft_warmup_epochs, total_epochs=max_epochs,
        )
        print(f"      [FT] LaBraM protocol: base_lr={ft_base_lr:.1e} "
              f"layer_decay={ft_layer_decay} wd={ft_weight_decay} "
              f"warmup={ft_warmup_epochs}ep cosine patience={patience} "
              f"drop_path={ft_drop_path} head_mult={ft_head_lr_mult} "
              f"batch={batch_size}", flush=True)
    else:
        # OneCycleLR (aggressive, favors random init)
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
        marker = "*" if improved else " "
        print(f"      ep{ep+1:03d}{marker} val_ba={val_ba:.4f} "
              f"best={best_val_ba:.4f} no_improve={no_improve}/{patience}",
              flush=True)
        if no_improve >= patience:
            print(f"      early stop at epoch {ep+1} (best_val_ba={best_val_ba:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"]); model.to(device)
        head.load_state_dict(best_state["head"]);   head.to(device)
    model.eval(); head.eval()

    preds, probas = [], []
    with torch.no_grad():
        for i in range(0, len(X_te), 128):
            batch = X_te[i:i+128]
            feats = model._encode(model._tokenize(batch)).mean(1)
            logits = head(feats)
            preds.append(logits.argmax(-1).cpu().numpy())
            probas.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    preds = np.concatenate(preds)
    probas = np.concatenate(probas)

    metrics = compute_metrics(y_te_np, preds, probas)
    metrics["best_val_ba"] = float(best_val_ba)
    metrics["epochs"] = int(ep + 1)
    return metrics


def _print_ft(prefix, m):
    print(f"{prefix}BA={m['balanced_accuracy']:.4f}  "
          f"ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"(best_val_ba={m['best_val_ba']:.4f}, epochs={m['epochs']})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mumtaz_dir", required=True,
                   help="Directory containing MDD_S{N}_(EC|EO).edf and "
                        "H_S{N}_(EC|EO).edf files (TASK auto-dropped)")
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--mode", choices=["frozen", "finetune", "both"],
                   default="finetune",
                   help="frozen: linear probe on frozen features (fast "
                        "diagnostic). finetune: full FT (default). "
                        "both: run both.")
    p.add_argument("--frozen_reps", type=int, default=5,
                   help="Number of LogReg reps for frozen probe (mean±std).")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust",
                   choices=["per_trial_zscore", "per_recording_robust"])
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--ft_batch_size", type=int, default=64)
    p.add_argument("--ft_patience", type=int, default=10)
    p.add_argument("--ft_protocol", choices=["onecycle", "labram"],
                   default="onecycle",
                   help="onecycle (default, aggressive, favors random init) "
                        "or labram (cosine+warmup+layer_decay, protective).")
    p.add_argument("--ft_base_lr", type=float, default=5e-4)
    p.add_argument("--ft_weight_decay", type=float, default=0.05)
    p.add_argument("--ft_layer_decay", type=float, default=0.65,
                   help="labram only. 0.65 LaBraM default; higher = less "
                        "protective. 1.0 = no decay.")
    p.add_argument("--ft_warmup_epochs", type=int, default=5)
    p.add_argument("--ft_drop_path", type=float, default=0.0)
    p.add_argument("--ft_head_lr_mult", type=float, default=1.0)
    p.add_argument("--n_reps", type=int, default=1)
    p.add_argument("--split_mode", choices=["cbramod", "counts"], default="cbramod",
                   help="cbramod: exact CBraMod/CSBrain deterministic split "
                        "(test=14 subj, --seed only varies training). "
                        "counts: old seed-randomized subject split.")
    p.add_argument("--include_random_baseline", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Global seeding
    _pyrandom.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu")

    print(f"\n{'='*72}")
    print(f"  Mumtaz2016 Depression Binary Classification")
    print(f"{'='*72}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Reference: LaBraM 0.9409 / CBraMod 0.9560 / CSBrain 0.9643 (BA)")
    print(f"  Seed: {args.seed}")

    model, model_cls, model_type_name, n_channels, ckpt_args = \
        load_pretrained(args.checkpoint, device)

    print(f"\n--- Building splits ({args.split_mode}) ---")
    if args.split_mode == "cbramod":
        # exact CBraMod/CSBrain deterministic split (no seed); --seed only
        # affects TRAINING (init/data order) for honest mean±std reporting.
        splits = make_cbramod_split(args.mumtaz_dir)
    else:
        splits = make_subject_split(args.mumtaz_dir, seed=args.seed)
    print(f"  Train: H {splits['train']['H']}  MDD {splits['train']['MDD']}")
    print(f"  Val:   H {splits['val']['H']}  MDD {splits['val']['MDD']}")
    print(f"  Test:  H {splits['test']['H']}  MDD {splits['test']['MDD']}")

    print(f"\n--- Loading Mumtaz splits ---")
    t0 = time.time()
    train_ds = MumtazDataset(
        data_dir=args.mumtaz_dir,
        subjects=splits["train"],
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        normalization=args.normalization,
        cache_dir=args.cache_dir,
    )
    val_ds = MumtazDataset(
        data_dir=args.mumtaz_dir,
        subjects=splits["val"],
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        normalization=args.normalization,
        cache_dir=args.cache_dir,
    )
    test_ds = MumtazDataset(
        data_dir=args.mumtaz_dir,
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
    print(f"Class counts (train): H={int((y_tr==0).sum())}  MDD={int((y_tr==1).sum())}")
    print(f"Class counts (test):  H={int((y_te==0).sum())}  MDD={int((y_te==1).sum())}")

    results = {
        "checkpoint": args.checkpoint,
        "model_type": model_type_name,
        "seed": int(args.seed),
        "n_train": int(len(y_tr)),
        "n_val":   int(len(y_val)),
        "n_test":  int(len(y_te)),
        "n_classes": N_CLASSES,
        "split": {k: {g: list(v) for g, v in d.items()}
                  for k, d in splits.items()},
        "reference": {
            "LaBraM_BA": 0.9409, "CBraMod_BA": 0.9560, "CSBrain_BA": 0.9643,
        },
    }

    # ─── Frozen probe (fast feature-quality diagnostic) ───
    if args.mode in ("frozen", "both"):
        print(f"\n{'='*72}\n  [JEPA] Frozen probe (LogReg × {args.frozen_reps} reps)\n{'='*72}")
        print("  Extracting JEPA features from train/test ...")
        feat_tr = extract_features(model, X_tr, device)
        feat_te = extract_features(model, X_te, device)
        print(f"  Feature shape: train {feat_tr.shape}, test {feat_te.shape}")
        m_jepa_frozen = run_frozen_probe(feat_tr, y_tr, feat_te, y_te,
                                          n_reps=args.frozen_reps)
        print(f"  JEPA frozen  BA={m_jepa_frozen['balanced_accuracy']['mean']:.4f}"
              f"±{m_jepa_frozen['balanced_accuracy']['std']:.4f}  "
              f"ROC-AUC={m_jepa_frozen['roc_auc']['mean']:.4f}"
              f"±{m_jepa_frozen['roc_auc']['std']:.4f}")
        results["jepa_frozen"] = m_jepa_frozen

        if args.include_random_baseline:
            print(f"\n{'='*72}\n  [Random] Frozen probe (untrained encoder)\n{'='*72}")
            random_model = build_random_init(model_cls, n_channels, ckpt_args, device)
            r_feat_tr = extract_features(random_model, X_tr, device)
            r_feat_te = extract_features(random_model, X_te, device)
            m_rand_frozen = run_frozen_probe(r_feat_tr, y_tr, r_feat_te, y_te,
                                              n_reps=args.frozen_reps)
            print(f"  Rand frozen  BA={m_rand_frozen['balanced_accuracy']['mean']:.4f}"
                  f"±{m_rand_frozen['balanced_accuracy']['std']:.4f}  "
                  f"ROC-AUC={m_rand_frozen['roc_auc']['mean']:.4f}"
                  f"±{m_rand_frozen['roc_auc']['std']:.4f}")
            results["random_frozen"] = m_rand_frozen
            del random_model; torch.cuda.empty_cache()

    # ─── Fine-tune ───
    if args.mode in ("finetune", "both"):
        ft_kwargs = dict(
            ft_protocol=args.ft_protocol,
            ft_base_lr=args.ft_base_lr,
            ft_weight_decay=args.ft_weight_decay,
            ft_layer_decay=args.ft_layer_decay,
            ft_warmup_epochs=args.ft_warmup_epochs,
            ft_drop_path=args.ft_drop_path,
            ft_head_lr_mult=args.ft_head_lr_mult,
        )
        print(f"\n{'='*72}\n  [JEPA] Fine-tune (protocol={args.ft_protocol})\n{'='*72}")
        m_jepa = run_finetune(model, X_tr, y_tr, X_val, y_val, X_te, y_te,
                              device, args.max_epochs, args.ft_batch_size,
                              args.ft_patience, **ft_kwargs)
        _print_ft("  JEPA-FT ", m_jepa)
        results["jepa_finetune"] = m_jepa

        if args.include_random_baseline:
            print(f"\n{'='*72}\n  [Random] Fine-tune from scratch\n{'='*72}")
            random_model = build_random_init(model_cls, n_channels, ckpt_args, device)
            m_rand = run_finetune(random_model, X_tr, y_tr, X_val, y_val, X_te, y_te,
                                  device, args.max_epochs, args.ft_batch_size,
                                  args.ft_patience, **ft_kwargs)
            _print_ft("  Rand-FT ", m_rand)
            results["random_finetune"] = m_rand
            del random_model; torch.cuda.empty_cache()

    print(f"\n{'='*72}")
    print(f"  SUMMARY (Mumtaz2016 Depression, seed {args.seed})")
    print(f"{'='*72}")
    print(f"  {'Model':<28} {'BA':>8} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print(f"  {'-'*28} {'-'*8} {'-'*10} {'-'*10}")
    ref = results["reference"]
    print(f"  {'CBraMod (published)':<28} {ref['CBraMod_BA']:>8.4f}")
    print(f"  {'LaBraM (published)':<28} {ref['LaBraM_BA']:>8.4f}")
    if "random_frozen" in results:
        m = results["random_frozen"]
        print(f"  {'Random frozen (Ours)':<28} "
              f"{m['balanced_accuracy']['mean']:>8.4f} "
              f"{m['roc_auc']['mean']:>10.4f} "
              f"{m['pr_auc']['mean']:>10.4f}")
    if "jepa_frozen" in results:
        m = results["jepa_frozen"]
        print(f"  {'JEPA frozen (Ours)':<28} "
              f"{m['balanced_accuracy']['mean']:>8.4f} "
              f"{m['roc_auc']['mean']:>10.4f} "
              f"{m['pr_auc']['mean']:>10.4f}")
    if "random_finetune" in results:
        m = results["random_finetune"]
        print(f"  {'Random FT (Ours)':<28} "
              f"{m['balanced_accuracy']:>8.4f} "
              f"{m['roc_auc']:>10.4f} "
              f"{m['pr_auc']:>10.4f}")
    if "jepa_finetune" in results:
        m = results["jepa_finetune"]
        print(f"  {'JEPA FT (Ours)':<28} "
              f"{m['balanced_accuracy']:>8.4f} "
              f"{m['roc_auc']:>10.4f} "
              f"{m['pr_auc']:>10.4f}")

    # Save
    out_path = args.output
    if out_path is None:
        stem = Path(args.checkpoint).parent.name
        out_path = f"/home/pxieaf/home2/eval_results/{stem}_mumtaz_seed{args.seed}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved: {out_path}")


if __name__ == "__main__":
    main()
