"""Schizophrenia (Olejarczyk RepOD) evaluation — our model, frozen + FT.

Same per-sample protocol as eval_mumtaz.py (subject-disjoint split, LogReg
frozen probe + full FT, BA/ROC-AUC/PR-AUC), just swapping the dataset. This is
a mechanism-justified "signature" task: SZ's biomarker is cross-frequency
(theta-gamma coupling / gamma dysregulation), NOT in any competitor benchmark.

Compare against the handcrafted band-power baseline (eval_handcrafted_spectral.py
--dataset schizophrenia) on the SAME splits:
    Random-frozen  <  handcrafted band power  <  JEPA-frozen ?

Usage:
    python eval_schizophrenia.py --checkpoint .../checkpoint_ep2.pt \\
        --sz_dir /home/pxieaf/home2/datasets/schizophrenia \\
        --mode both --seed 42 --include_random_baseline
"""
import argparse
import json
import random as _pyrandom
import time
from pathlib import Path

import numpy as np
import torch

from dataset_schizophrenia import (
    SchizophreniaDataset, make_subject_split, N_CLASSES,
)
from eval_tuh_clinical import load_pretrained, build_random_init
from eval_mumtaz import (
    dataset_to_xy, extract_features, run_frozen_probe, run_finetune, _print_ft,
)


def _load(args, subjects):
    return SchizophreniaDataset(
        data_dir=args.sz_dir, subjects=subjects,
        sample_rate=args.sample_rate, trial_duration_s=args.trial_duration_s,
        normalization=args.normalization, cache_dir=args.cache_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sz_dir", required=True,
                   help="Dir with h01..h14.edf and s01..s14.edf")
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--mode", choices=["frozen", "finetune", "both"], default="both")
    p.add_argument("--frozen_reps", type=int, default=5)
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust",
                   choices=["per_trial_zscore", "per_recording_robust"])
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--ft_batch_size", type=int, default=64)
    p.add_argument("--ft_patience", type=int, default=10)
    p.add_argument("--ft_protocol", choices=["onecycle", "labram"], default="onecycle")
    p.add_argument("--ft_base_lr", type=float, default=5e-4)
    p.add_argument("--ft_weight_decay", type=float, default=0.05)
    p.add_argument("--ft_layer_decay", type=float, default=0.65)
    p.add_argument("--ft_warmup_epochs", type=int, default=5)
    p.add_argument("--ft_drop_path", type=float, default=0.0)
    p.add_argument("--ft_head_lr_mult", type=float, default=1.0)
    p.add_argument("--include_random_baseline", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _pyrandom.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    print(f"\n{'='*72}\n  Schizophrenia (Olejarczyk RepOD) — binary (SZ vs Healthy)")
    print(f"  Checkpoint: {args.checkpoint}   seed {args.seed}\n{'='*72}")

    model, model_cls, mtype, n_channels, ckpt_args = load_pretrained(args.checkpoint, device)

    splits = make_subject_split(args.sz_dir, seed=args.seed)
    t0 = time.time()
    X_tr, y_tr = dataset_to_xy(_load(args, splits["train"]))
    X_val, y_val = dataset_to_xy(_load(args, splits["val"]))
    X_te, y_te = dataset_to_xy(_load(args, splits["test"]))
    print(f"Data loaded in {(time.time()-t0)/60:.1f} min")
    print(f"Shapes: train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")
    print(f"Train classes: H={int((y_tr==0).sum())} SZ={int((y_tr==1).sum())}   "
          f"Test: H={int((y_te==0).sum())} SZ={int((y_te==1).sum())}")

    results = {"checkpoint": args.checkpoint, "model_type": mtype,
               "seed": int(args.seed), "n_train": int(len(y_tr)),
               "n_test": int(len(y_te)), "n_classes": N_CLASSES,
               "split": {k: {g: list(v) for g, v in d.items()}
                         for k, d in splits.items()}}

    if args.mode in ("frozen", "both"):
        print(f"\n  [JEPA] Frozen probe (LogReg × {args.frozen_reps})")
        fj_tr = extract_features(model, X_tr, device)
        fj_te = extract_features(model, X_te, device)
        results["jepa_frozen"] = run_frozen_probe(fj_tr, y_tr, fj_te, y_te,
                                                  n_reps=args.frozen_reps)
        mj = results["jepa_frozen"]
        print(f"  JEPA frozen  BA={mj['balanced_accuracy']['mean']:.4f}"
              f"±{mj['balanced_accuracy']['std']:.4f}  "
              f"ROC={mj['roc_auc']['mean']:.4f}  PR={mj['pr_auc']['mean']:.4f}")
        if args.include_random_baseline:
            rm = build_random_init(model_cls, n_channels, ckpt_args, device)
            fr_tr = extract_features(rm, X_tr, device)
            fr_te = extract_features(rm, X_te, device)
            results["random_frozen"] = run_frozen_probe(fr_tr, y_tr, fr_te, y_te,
                                                        n_reps=args.frozen_reps)
            mr = results["random_frozen"]
            print(f"  Rand frozen  BA={mr['balanced_accuracy']['mean']:.4f}"
                  f"±{mr['balanced_accuracy']['std']:.4f}  "
                  f"ROC={mr['roc_auc']['mean']:.4f}  PR={mr['pr_auc']['mean']:.4f}")
            del rm; torch.cuda.empty_cache()

    if args.mode in ("finetune", "both"):
        ft_kwargs = dict(ft_protocol=args.ft_protocol, ft_base_lr=args.ft_base_lr,
                         ft_weight_decay=args.ft_weight_decay,
                         ft_layer_decay=args.ft_layer_decay,
                         ft_warmup_epochs=args.ft_warmup_epochs,
                         ft_drop_path=args.ft_drop_path,
                         ft_head_lr_mult=args.ft_head_lr_mult)
        print(f"\n  [JEPA] Fine-tune (protocol={args.ft_protocol})")
        results["jepa_finetune"] = run_finetune(
            model, X_tr, y_tr, X_val, y_val, X_te, y_te, device,
            args.max_epochs, args.ft_batch_size, args.ft_patience, **ft_kwargs)
        _print_ft("  JEPA-FT ", results["jepa_finetune"])
        if args.include_random_baseline:
            rm = build_random_init(model_cls, n_channels, ckpt_args, device)
            results["random_finetune"] = run_finetune(
                rm, X_tr, y_tr, X_val, y_val, X_te, y_te, device,
                args.max_epochs, args.ft_batch_size, args.ft_patience, **ft_kwargs)
            _print_ft("  Rand-FT ", results["random_finetune"])
            del rm; torch.cuda.empty_cache()

    out_path = args.output or (f"/home/pxieaf/home2/eval_results/"
                               f"{Path(args.checkpoint).parent.name}_sz_seed{args.seed}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n→ Saved: {out_path}")


if __name__ == "__main__":
    main()
