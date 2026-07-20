"""Mental Arithmetic (eegmat) evaluation — frozen probe + FT (rest vs task).

In-scope spectral clinical task (alpha suppression / frontal-midline theta during
arithmetic). Same per-sample protocol as eval_mumtaz/eval_schizophrenia.

    python eval_mental_arithmetic.py --checkpoint .../checkpoint_ep8.pt \\
        --data_dir /home/pxieaf/home2/datasets/mental_arithmetic/eeg-during-mental-arithmetic-tasks-1.0.0 \\
        --mode both --seed 42 --include_random_baseline
"""
import argparse
import json
import random as _pyrandom
import time
from pathlib import Path

import numpy as np
import torch

from dataset_mental_arithmetic import (
    MentalArithmeticDataset, make_subject_split, N_CLASSES,
)
from eval_tuh_clinical import load_pretrained, build_random_init
from eval_mumtaz import (
    dataset_to_xy, extract_features, run_frozen_probe, run_finetune, _print_ft,
)


def _load(args, subjects):
    return MentalArithmeticDataset(
        data_dir=args.data_dir, subjects=subjects,
        sample_rate=args.sample_rate, trial_duration_s=args.trial_duration_s,
        normalization=args.normalization, cache_dir=args.cache_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--mode", choices=["frozen", "finetune", "both"], default="both")
    p.add_argument("--frozen_reps", type=int, default=5)
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust")
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

    _pyrandom.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    print(f"\n{'='*72}\n  Mental Arithmetic (eegmat) — rest vs arithmetic")
    print(f"  Checkpoint: {args.checkpoint}  seed {args.seed}\n{'='*72}")
    model, model_cls, mtype, n_channels, ckpt_args = load_pretrained(args.checkpoint, device)

    splits = make_subject_split(args.data_dir, seed=args.seed)
    X_tr, y_tr = dataset_to_xy(_load(args, splits["train"]))
    X_val, y_val = dataset_to_xy(_load(args, splits["val"]))
    X_te, y_te = dataset_to_xy(_load(args, splits["test"]))
    print(f"Shapes: train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")
    print(f"Train rest/task: {np.bincount(y_tr)}   Test: {np.bincount(y_te)}")

    results = {"checkpoint": args.checkpoint, "model_type": mtype, "seed": int(args.seed),
               "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "n_classes": N_CLASSES,
               "split": {k: list(v) for k, v in splits.items()}}

    if args.mode in ("frozen", "both"):
        print("\n  [JEPA] Frozen probe")
        fj_tr = extract_features(model, X_tr, device); fj_te = extract_features(model, X_te, device)
        results["jepa_frozen"] = run_frozen_probe(fj_tr, y_tr, fj_te, y_te, n_reps=args.frozen_reps)
        m = results["jepa_frozen"]
        print(f"  JEPA frozen BA={m['balanced_accuracy']['mean']:.4f}±{m['balanced_accuracy']['std']:.4f}")
        if args.include_random_baseline:
            rm = build_random_init(model_cls, n_channels, ckpt_args, device)
            fr_tr = extract_features(rm, X_tr, device); fr_te = extract_features(rm, X_te, device)
            results["random_frozen"] = run_frozen_probe(fr_tr, y_tr, fr_te, y_te, n_reps=args.frozen_reps)
            mr = results["random_frozen"]
            print(f"  Rand frozen BA={mr['balanced_accuracy']['mean']:.4f}")
            del rm; torch.cuda.empty_cache()

    if args.mode in ("finetune", "both"):
        ft = dict(ft_protocol=args.ft_protocol, ft_base_lr=args.ft_base_lr,
                  ft_weight_decay=args.ft_weight_decay, ft_layer_decay=args.ft_layer_decay,
                  ft_warmup_epochs=args.ft_warmup_epochs, ft_drop_path=args.ft_drop_path,
                  ft_head_lr_mult=args.ft_head_lr_mult)
        print(f"\n  [JEPA] Fine-tune ({args.ft_protocol})")
        results["jepa_finetune"] = run_finetune(model, X_tr, y_tr, X_val, y_val, X_te, y_te,
                                                device, args.max_epochs, args.ft_batch_size,
                                                args.ft_patience, **ft)
        _print_ft("  JEPA-FT ", results["jepa_finetune"])
        if args.include_random_baseline:
            rm = build_random_init(model_cls, n_channels, ckpt_args, device)
            results["random_finetune"] = run_finetune(rm, X_tr, y_tr, X_val, y_val, X_te, y_te,
                                                      device, args.max_epochs, args.ft_batch_size,
                                                      args.ft_patience, **ft)
            _print_ft("  Rand-FT ", results["random_finetune"])
            del rm; torch.cuda.empty_cache()

    out = args.output or (f"/home/pxieaf/home2/eval_results/"
                          f"{Path(args.checkpoint).parent.name}_mentalarith_seed{args.seed}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n→ Saved: {out}")


if __name__ == "__main__":
    main()
