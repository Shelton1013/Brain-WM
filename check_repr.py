"""
Check EEG-JEPA representation quality.

Diagnoses:
  1. Are representations collapsed? (all outputs similar)
  2. Do different MI classes produce different representations?
  3. How does pretrained compare to random init?

Usage:
  CUDA_VISIBLE_DEVICES=3 python check_repr.py \
      --checkpoint /home/share/data_makchen/peng/models/eeg_jepa/best_model.pt \
      --data_dir /home/share/data_makchen/peng/datasets/physionet
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_jepa import EEGJEPA
from evaluate import PhysioNetMI_Labeled


def extract_representations(model, loader, device):
    """Extract frozen encoder representations for all samples."""
    model.eval()
    all_reprs = []
    all_labels = []

    with torch.no_grad():
        for eeg, labels in loader:
            eeg = eeg.to(device)
            tokens = model._tokenize(eeg)
            encoded = model._encode(tokens)    # [B, N, D]
            pooled = encoded.mean(dim=1)       # [B, D]
            all_reprs.append(pooled.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_reprs), np.concatenate(all_labels)


def analyze(reprs, labels, label=""):
    """Print diagnostic statistics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    B, D = reprs.shape
    print(f"\n  Shape: {B} samples × {D} dims")

    # 1. Global statistics
    print(f"\n  --- Global Stats ---")
    print(f"  Mean of means (per dim):  {reprs.mean():.6f}")
    print(f"  Mean of stds  (per dim):  {reprs.std(axis=0).mean():.6f}")
    print(f"  Min std across dims:      {reprs.std(axis=0).min():.6f}")
    print(f"  Max std across dims:      {reprs.std(axis=0).max():.6f}")
    n_dead = (reprs.std(axis=0) < 1e-5).sum()
    print(f"  Dead dimensions (std<1e-5): {n_dead}/{D} ({n_dead/D*100:.1f}%)")

    # 2. Effective dimensionality (via PCA explained variance)
    centered = reprs - reprs.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    variance_explained = (S ** 2) / (S ** 2).sum()
    cum_var = np.cumsum(variance_explained)
    dim_90 = np.searchsorted(cum_var, 0.90) + 1
    dim_99 = np.searchsorted(cum_var, 0.99) + 1
    print(f"\n  --- Effective Dimensionality ---")
    print(f"  Dims for 90% variance: {dim_90}/{D}")
    print(f"  Dims for 99% variance: {dim_99}/{D}")
    print(f"  Top-1 singular value %: {variance_explained[0]*100:.1f}%")
    print(f"  Top-5 singular values %: {cum_var[4]*100:.1f}%")

    # 3. Inter-sample similarity
    norms = np.linalg.norm(reprs, axis=1, keepdims=True) + 1e-8
    reprs_norm = reprs / norms
    # Random 1000 pairs
    n_pairs = min(1000, B * (B-1) // 2)
    idx1 = np.random.randint(0, B, n_pairs)
    idx2 = np.random.randint(0, B, n_pairs)
    cosines = (reprs_norm[idx1] * reprs_norm[idx2]).sum(axis=1)
    print(f"\n  --- Inter-sample Cosine Similarity ---")
    print(f"  Mean: {cosines.mean():.4f}")
    print(f"  Std:  {cosines.std():.4f}")
    print(f"  Min:  {cosines.min():.4f}")
    print(f"  Max:  {cosines.max():.4f}")
    if cosines.mean() > 0.95:
        print(f"  ⚠ COLLAPSED: all representations nearly identical!")
    elif cosines.mean() > 0.80:
        print(f"  ⚠ WARNING: representations are very similar")
    else:
        print(f"  ✓ OK: representations have diversity")

    # 4. Per-class analysis
    unique_labels = np.unique(labels)
    class_names = {0: "left_fist", 1: "right_fist", 2: "both_fists", 3: "both_feet"}
    class_means = {}

    print(f"\n  --- Per-class Mean Representations ---")
    for c in unique_labels:
        mask = labels == c
        class_mean = reprs[mask].mean(axis=0)
        class_means[c] = class_mean
        class_std = reprs[mask].std(axis=0).mean()
        name = class_names.get(c, f"class_{c}")
        print(f"  {name:12s} (n={mask.sum():4d}): "
              f"norm={np.linalg.norm(class_mean):.4f}, "
              f"mean_std={class_std:.6f}")

    # 5. Inter-class distances
    print(f"\n  --- Inter-class Cosine Distance ---")
    print(f"  {'':12s}", end="")
    for c in unique_labels:
        print(f"  {class_names.get(c, f'c{c}'):>10s}", end="")
    print()

    for c1 in unique_labels:
        name1 = class_names.get(c1, f"c{c1}")
        print(f"  {name1:12s}", end="")
        m1 = class_means[c1]
        n1 = m1 / (np.linalg.norm(m1) + 1e-8)
        for c2 in unique_labels:
            m2 = class_means[c2]
            n2 = m2 / (np.linalg.norm(m2) + 1e-8)
            cos = np.dot(n1, n2)
            print(f"  {cos:10.4f}", end="")
        print()

    # Check: are class means distinguishable?
    all_cos = []
    for i, c1 in enumerate(unique_labels):
        for c2 in unique_labels[i+1:]:
            m1 = class_means[c1] / (np.linalg.norm(class_means[c1]) + 1e-8)
            m2 = class_means[c2] / (np.linalg.norm(class_means[c2]) + 1e-8)
            all_cos.append(np.dot(m1, m2))
    mean_inter = np.mean(all_cos)
    if mean_inter > 0.99:
        print(f"\n  ⚠ Class means are IDENTICAL (cos={mean_inter:.4f})")
        print(f"    → Encoder does not distinguish motor imagery classes at all")
    elif mean_inter > 0.95:
        print(f"\n  ⚠ Class means are very similar (cos={mean_inter:.4f})")
        print(f"    → Weak class separation")
    else:
        print(f"\n  ✓ Class means differ (mean cos={mean_inter:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets/physionet")
    parser.add_argument("--n_subjects", type=int, default=109)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    # Load labeled data (use val subjects for analysis)
    subjects = list(range(1, args.n_subjects + 1))
    n_train_subj = int(len(subjects) * 0.8)

    print("Loading val set (subjects 88-109)...")
    val_ds = PhysioNetMI_Labeled(
        subjects[n_train_subj:], data_dir=args.data_dir, use_ea=True,
    )
    # Remove rest
    mask = val_ds.labels != 0
    val_ds.trials = val_ds.trials[mask]
    val_ds.labels = val_ds.labels[mask] - 1
    print(f"  {len(val_ds.trials)} MI trials (4 classes)")

    n_channels = len(val_ds.electrode_names)

    # Remap channels if checkpoint uses fewer (e.g., 19ch multi-dataset)
    ckpt_n_channels = n_channels
    if args.checkpoint:
        ckpt_tmp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        for key, val in ckpt_tmp["model_state_dict"].items():
            if "channel_embed" in key:
                ckpt_n_channels = val.shape[0]
                break
        del ckpt_tmp
        if ckpt_n_channels != n_channels:
            from dataset_multi import pick_common_channels
            ch_indices, _ = pick_common_channels(val_ds.electrode_names)
            if len(ch_indices) == ckpt_n_channels:
                print(f"  Remapping {n_channels}ch → {ckpt_n_channels}ch")
                val_ds.trials = val_ds.trials[:, :, ch_indices].copy()
                n_channels = ckpt_n_channels

    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    # 1. Random init
    print("\nExtracting random init representations...")
    random_model = EEGJEPA(n_channels=n_channels).to(device)
    random_reprs, labels = extract_representations(random_model, loader, device)
    analyze(random_reprs, labels, label="Random Init Encoder")
    del random_model

    # 2. Pretrained
    if args.checkpoint:
        print("\nExtracting pretrained representations...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        pretrained = EEGJEPA(
            n_channels=ckpt_n_channels,
            d_model=ckpt_args.get("d_model", 256),
            encoder_layers=ckpt_args.get("encoder_layers", 6),
        ).to(device)
        pretrained.load_state_dict(ckpt["model_state_dict"])
        pre_reprs, labels = extract_representations(pretrained, loader, device)
        analyze(pre_reprs, labels, label="Pretrained EEG-JEPA Encoder")


if __name__ == "__main__":
    main()
