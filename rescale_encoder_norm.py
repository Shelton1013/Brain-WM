"""
Quick test: rescale encoder_norm gamma/beta to restore unit-std output.

Hypothesis: pretrain underperformed because encoder_norm.weight (gamma)
collapsed to ~0.155 mean abs, making encoder output 6.5x smaller than
random init. This rescales weight + bias multiplicatively to restore
unit-std output, then saves the modified checkpoint for re-eval.

Note on interpretation:
  - Linear classifiers are SCALE-INVARIANT in theory, but with L2 reg
    (sklearn LogReg C) and FT learning rate dynamics, scale DOES matter.
  - Effective rank is NOT changed by rescaling (just multiplicative).
  - If rescaled ckpt eval is much better → scale was a real bottleneck.
  - If rescaled ckpt eval is same → real feature collapse (rank<rank),
    rescaling won't help, must re-pretrain with stronger reg.

Usage:
    python rescale_encoder_norm.py \
        --checkpoint /home/pxieaf/home2/model/outputcf_vicreg_tueg_60s_v1/best_model.pt \
        --output_checkpoint /home/pxieaf/home2/model/outputcf_vicreg_tueg_60s_v1/best_model_rescaled.pt \
        --tuh_dir /home/pxieaf/home2/tuh/tuh_eeg_events/v2.0.1/edf
"""

import argparse
import torch
import numpy as np

from eval_tuh_clinical import load_pretrained
from dataset_tuh_clinical import TUABDataset, TUEVDataset


def measure_enc_std(model, X_t):
    model.eval()
    with torch.no_grad():
        toks = model._tokenize(X_t)
        enc = model._encode(toks)
    return float(enc.std().item()), float(enc.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_checkpoint", required=True)
    p.add_argument("--task", choices=["tuab", "tuev"], default="tuev")
    p.add_argument("--tuh_dir", required=True)
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--trial_duration_s", type=int, default=5)
    p.add_argument("--normalization", default="per_recording_robust")
    p.add_argument("--n_batch", type=int, default=64)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu")

    # ─── Load eval batch ───
    print(f"\n--- Loading {args.task.upper()} eval batch ({args.n_batch} samples) ---")
    DSCls = TUABDataset if args.task == "tuab" else TUEVDataset
    eval_ds = DSCls(
        data_dir=args.tuh_dir, split="eval",
        sample_rate=256, trial_duration_s=args.trial_duration_s,
        cache_dir=args.cache_dir, normalization=args.normalization,
    )
    X = np.stack(eval_ds.trials[:args.n_batch]).astype(np.float32)

    # ─── Inspect original encoder_norm ───
    print(f"\n--- Loading original checkpoint ---")
    model, _, _, n_channels, _ = load_pretrained(args.checkpoint, device)
    X_t = torch.from_numpy(X).to(device)
    if X_t.shape[-1] < n_channels:
        X_t = torch.nn.functional.pad(X_t, (0, n_channels - X_t.shape[-1]))
    elif X_t.shape[-1] > n_channels:
        X_t = X_t[..., :n_channels]

    sd_orig = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n=== ORIGINAL encoder_norm stats ===")
    w_key = "encoder_norm.weight"
    b_key = "encoder_norm.bias"
    if w_key not in sd_orig:
        print(f"  Available norm keys: {[k for k in sd_orig if 'norm' in k][:10]}")
        raise ValueError(f"{w_key} not in state_dict")

    w = sd_orig[w_key]
    b = sd_orig[b_key]
    print(f"  weight: shape={tuple(w.shape)} mean={w.mean():.4f} "
          f"std={w.std():.4f} min={w.min():.4f} max={w.max():.4f} "
          f"abs_mean={w.abs().mean():.4f}")
    print(f"  bias:   shape={tuple(b.shape)} mean={b.mean():.4f} "
          f"std={b.std():.4f} abs_mean={b.abs().mean():.4f}")

    std_orig, mean_orig = measure_enc_std(model, X_t)
    print(f"\n  Encoder output std (measured on batch): {std_orig:.4f}")
    print(f"  Encoder output mean (measured on batch): {mean_orig:.4f}")

    # ─── Compute scale to restore unit std ───
    target_std = 1.0
    scale = target_std / max(std_orig, 1e-6)
    print(f"\n=== RESCALING encoder_norm by factor {scale:.4f} ===")

    # Modify state_dict in-place
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt["model_state_dict"][w_key] = w * scale
    ckpt["model_state_dict"][b_key] = b * scale

    # ─── Save and reload ───
    torch.save(ckpt, args.output_checkpoint)
    print(f"  Saved: {args.output_checkpoint}")

    print(f"\n--- Loading rescaled checkpoint ---")
    model_rs, _, _, _, _ = load_pretrained(args.output_checkpoint, device)
    std_new, mean_new = measure_enc_std(model_rs, X_t)
    print(f"\n=== POST-RESCALE encoder output ===")
    print(f"  std={std_new:.4f}  (target ≈ 1.0)")
    print(f"  mean={mean_new:.4f}")

    w_new = model_rs.state_dict()[w_key]
    print(f"\n  New encoder_norm.weight abs_mean: {w_new.abs().mean():.4f}")

    # ─── Print summary for user ───
    print(f"\n{'='*70}")
    print(f"  RESCALE COMPLETE")
    print(f"{'='*70}")
    print(f"  Original encoder output std: {std_orig:.4f}")
    print(f"  Rescaled encoder output std: {std_new:.4f}  ({scale:.3f}x scaling)")
    print(f"  Rescaled checkpoint: {args.output_checkpoint}")
    print(f"\n  Next step: re-eval on the rescaled ckpt:")
    print(f"    python eval_tuh_clinical.py \\")
    print(f"      --checkpoint {args.output_checkpoint} \\")
    print(f"      --dataset {args.task} \\")
    print(f"      --tuh_dir {args.tuh_dir} \\")
    print(f"      --trial_duration_s {args.trial_duration_s} \\")
    print(f"      --normalization {args.normalization} \\")
    print(f"      --mode both --n_reps 3 --max_epochs 50 \\")
    print(f"      --include_random_baseline")
    print(f"\n  If rescaled eval ≈ original eval: scale was NOT the bottleneck,")
    print(f"    real feature collapse (low effective rank). Need to re-pretrain")
    print(f"    with stronger reg (sigreg_lambda 0.05 → 0.5 or 1.0).")
    print(f"  If rescaled eval >> original eval: scale was the bottleneck.")
    print(f"    Can use rescaled ckpt for paper, or re-pretrain with better init.")


if __name__ == "__main__":
    main()
