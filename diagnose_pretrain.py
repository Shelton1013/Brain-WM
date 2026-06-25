"""
Diagnostic script: audit pretrain-eval mismatch.

Loads pretrained + random-init model, runs forward on a TUEV (or TUAB)
eval batch, and prints six checks to isolate algorithm-issue vs code-bug:

  1. Data normalization sanity:   mean/std/median/IQR of input X
                                  (should be ~0 / ~1.5 after robust scale)
  2. Channel order:               electrode names of eval dataset
                                  vs pretrained model's expectation
  3. Strict state_dict load:      re-load with strict=True, must not error
  4. Encoder output stats:        mean/std of pretrained vs random encoder
                                  output — should differ meaningfully
  5. Feature degeneracy:          effective rank + per-sample variance
                                  (if all samples have ~same feature → bad)
  6. Pretrained vs Random sim:    cosine similarity between pretrained
                                  and random encoder outputs on same batch
                                  (should NOT be ~0 and should NOT be ~1)

Usage:
    python diagnose_pretrain.py \
        --checkpoint /path/to/best_model.pt \
        --task tuev \
        --tuh_dir /path/to/tuh/edf \
        --n_batch 32
"""

import argparse
import sys
import torch
import numpy as np
import torch.nn.functional as F

from eval_tuh_clinical import (
    load_pretrained, build_random_init,
    dataset_to_xy, extract_features,
)
from dataset_tuh_clinical import TUABDataset, TUEVDataset


def stat_line(name, x):
    """Print stats of a numpy array."""
    print(f"  {name:30s}  shape={tuple(x.shape)} "
          f"mean={x.mean():+.4f}  std={x.std():.4f}  "
          f"median={np.median(x):+.4f}  "
          f"q25={np.percentile(x,25):+.3f}  q75={np.percentile(x,75):+.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", choices=["tuab", "tuev"], default="tuev")
    p.add_argument("--tuh_dir", required=True)
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=5,
                   help="Match LaBraM: 5 for TUEV, 10 for TUAB")
    p.add_argument("--normalization", default="per_recording_robust",
                   choices=["per_trial_zscore", "per_recording_robust"])
    p.add_argument("--n_batch", type=int, default=32,
                   help="Number of samples to use for diagnostics")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu")

    print(f"\n{'='*72}\n  DIAGNOSTIC: {args.checkpoint}\n{'='*72}\n")

    # ─── 1. Load eval dataset (small subset) ───
    print(f"--- Loading {args.task.upper()} eval split ---")
    DSCls = TUABDataset if args.task == "tuab" else TUEVDataset
    eval_ds = DSCls(
        data_dir=args.tuh_dir, split="eval",
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        cache_dir=args.cache_dir,
        normalization=args.normalization,
    )
    print(f"  Loaded {len(eval_ds)} trials, "
          f"{len(eval_ds.electrode_names) if eval_ds.electrode_names else '?'} channels")

    # ─── 2. Load pretrained model ───
    print(f"\n--- Loading pretrained model ---")
    model_pt, model_cls, model_type_name, n_channels, ckpt_args = \
        load_pretrained(args.checkpoint, device)
    print(f"  ckpt_args: model={ckpt_args.get('model')} "
          f"d_model={ckpt_args.get('d_model')} "
          f"max_seq_len={ckpt_args.get('max_seq_len')} "
          f"normalization={ckpt_args.get('normalization')} "
          f"trial_duration_s={ckpt_args.get('trial_duration_s')}")

    # ─── 3. Build random-init (same architecture) ───
    print(f"\n--- Building random-init model (same architecture) ---")
    model_rd = build_random_init(model_cls, n_channels, ckpt_args, device)
    model_rd.eval()

    # ─── CHECK 1: Data normalization sanity ───
    print(f"\n{'─'*72}\n  CHECK 1: Eval data X stats "
          f"(normalization={args.normalization})\n{'─'*72}")
    X, y, _ = dataset_to_xy(eval_ds, n_channels)
    X_batch = X[:args.n_batch]
    print(f"  Sampled first {args.n_batch} trials: X shape {X_batch.shape}")
    stat_line("X (all channels, all time)", X_batch)
    stat_line("X channel 0", X_batch[:, :, 0])
    stat_line("X channel 9 (mid)", X_batch[:, :, 9])
    stat_line("X channel 18 (last)", X_batch[:, :, 18])
    print(f"\n  Expected for per_recording_robust:")
    print(f"    median ≈ 0, q75-q25 ≈ 1.349 → IQR/1.349 ≈ 1.0 ⇒ x roughly Normal(0,1)")
    print(f"  Expected for per_trial_zscore:")
    print(f"    mean ≈ 0, std ≈ 1 per trial (aggregate may differ slightly)")
    if abs(X_batch.mean()) > 2.0 or X_batch.std() > 50:
        print(f"  ⚠ X stats look UN-normalized (raw μV scale?). Check normalization wiring.")
    else:
        print(f"  ✓ X stats look normalized.")

    # ─── CHECK 2: Channel order ───
    print(f"\n{'─'*72}\n  CHECK 2: Eval electrode names\n{'─'*72}")
    print(f"  Eval ds electrode_names ({len(eval_ds.electrode_names)} ch):")
    print(f"    {eval_ds.electrode_names}")
    print(f"\n  Pretrain channel order: NOT recorded in checkpoint (would need to")
    print(f"  rebuild dataset_multi with same MOABB/PhysioNet config).")
    print(f"  Visually verify: this should be standard 10-20 order, e.g.")
    print(f"  ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4',")
    print(f"   'T5','P3','Pz','P4','T6','O1','O2'] or close variant.")

    # ─── CHECK 3: Strict state_dict re-load ───
    print(f"\n{'─'*72}\n  CHECK 3: strict state_dict re-load (must not error)\n{'─'*72}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    try:
        miss, unexp = model_pt.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"  ✓ strict=True load OK. missing={list(miss)}  unexpected={list(unexp)}")
    except RuntimeError as e:
        print(f"  ⚠ strict=True load FAILED:")
        print(f"  {e}")

    # ─── Forward pass on the same batch ───
    print(f"\n{'─'*72}\n  Forward pass: pretrained + random on same batch\n{'─'*72}")
    X_t = torch.from_numpy(X_batch).to(device)
    model_pt.eval()
    with torch.no_grad():
        toks_pt = model_pt._tokenize(X_t)
        enc_pt = model_pt._encode(toks_pt)         # [B, N, D]
        feat_pt = enc_pt.mean(dim=1)               # [B, D]
        toks_rd = model_rd._tokenize(X_t)
        enc_rd = model_rd._encode(toks_rd)
        feat_rd = enc_rd.mean(dim=1)
    print(f"  enc_pt shape: {tuple(enc_pt.shape)} (B, N_tokens, D)")
    print(f"  feat_pt shape: {tuple(feat_pt.shape)} (B, D) after mean-pool over tokens")

    # ─── CHECK 4: Encoder output stats ───
    print(f"\n{'─'*72}\n  CHECK 4: Encoder output stats\n{'─'*72}")
    stat_line("Pretrained enc output", enc_pt.cpu().numpy())
    stat_line("Random enc output    ", enc_rd.cpu().numpy())
    stat_line("Pretrained feat (pool)", feat_pt.cpu().numpy())
    stat_line("Random feat (pool)    ", feat_rd.cpu().numpy())
    print(f"\n  Healthy: nonzero std, finite mean, no NaN/Inf.")
    if not torch.isfinite(enc_pt).all():
        print(f"  ⚠ Pretrained encoder has NaN/Inf!")
    if enc_pt.std().item() < 1e-4:
        print(f"  ⚠ Pretrained encoder output near-zero std (collapsed?)")

    # ─── CHECK 5: Feature degeneracy (effective rank + sample variance) ───
    print(f"\n{'─'*72}\n  CHECK 5: Feature matrix rank + sample diversity\n{'─'*72}")
    for name, F_t in [("Pretrained", feat_pt), ("Random   ", feat_rd)]:
        Fmat = F_t.cpu().numpy()                   # [B, D]
        # Singular values
        s = np.linalg.svd(Fmat - Fmat.mean(0, keepdims=True),
                          compute_uv=False)
        s_norm = s / (s.sum() + 1e-12)
        # Effective rank = exp(entropy of normalized singular values)
        eff_rank = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-12))))
        # Per-sample feature variance — high diversity if samples differ a lot
        sample_pairwise = float(
            (Fmat[:, None, :] - Fmat[None, :, :]).reshape(-1, Fmat.shape[1])
            .std())
        print(f"  {name}: D={Fmat.shape[1]} eff_rank={eff_rank:.1f}  "
              f"top3 sv ratio: {s[:3] / (s[0]+1e-12)}  "
              f"pairwise sample std={sample_pairwise:.4f}")
    print(f"\n  Healthy: eff_rank ≳ 10-50 (much smaller → features collapsed).")

    # ─── CHECK 6: Pretrained vs Random cosine similarity ───
    print(f"\n{'─'*72}\n  CHECK 6: cos_sim(pretrained_feat, random_feat) per sample\n{'─'*72}")
    cos_sim = F.cosine_similarity(feat_pt, feat_rd, dim=-1).cpu().numpy()
    print(f"  cos_sim stats: mean={cos_sim.mean():.4f}  "
          f"std={cos_sim.std():.4f}  "
          f"min={cos_sim.min():.4f}  max={cos_sim.max():.4f}")
    print(f"\n  Interpretation:")
    print(f"    ≈ 0 (orthogonal):    pretrained features differ from random → good (training did something)")
    print(f"    ≈ 1 (identical):     pretrained ≈ random init → weights didn't change much during training (BAD: dead pretrain)")
    print(f"    high but < 1 (>0.8): possible partial training / similar inductive bias (sub-optimal)")
    if cos_sim.mean() > 0.9:
        print(f"  ⚠ Pretrained features nearly identical to random — pretrain may be ineffective.")
    elif abs(cos_sim.mean()) < 0.1:
        print(f"  ✓ Pretrained features clearly different from random.")

    print(f"\n{'='*72}\n  DIAGNOSTIC DONE\n{'='*72}\n")


if __name__ == "__main__":
    main()
