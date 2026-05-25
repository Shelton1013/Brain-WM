"""
PCA/UMAP visualization of EEG encoder representations.

Generates publication-ready figures showing:
  1. PCA 2D scatter (colored by MI class)
  2. UMAP 2D scatter (colored by MI class)
  3. Random vs Pretrained side-by-side comparison

Usage:
  CUDA_VISIBLE_DEVICES=8 python visualize_repr.py \
      --checkpoint /home/share/data_makchen/peng/models/eeg_lejepa_multi/best_model.pt \
      --data_dir /home/share/data_makchen/peng/datasets/physionet \
      --output_dir /home/share/data_makchen/peng/models/eeg_jepa/results/figures

  # Also works with 64ch checkpoint
  CUDA_VISIBLE_DEVICES=8 python visualize_repr.py \
      --checkpoint /home/share/data_makchen/peng/models/eeg_lejepa_64ch/best_model.pt
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. Install with: pip install umap-learn")

from eeg_jepa import EEGJEPA
from eeg_mae import EEGMAE
from eeg_lejepa import EEGLeJEPA
from evaluate import PhysioNetMI_Labeled


# ============================================================
# Feature extraction
# ============================================================

def extract_representations(model, loader, device):
    """Extract frozen encoder representations."""
    model.eval()
    all_reprs, all_labels = [], []
    with torch.no_grad():
        for eeg, labels in loader:
            eeg = eeg.to(device)
            tokens = model._tokenize(eeg)
            encoded = model._encode(tokens)
            pooled = encoded.mean(dim=1)
            all_reprs.append(pooled.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_reprs), np.concatenate(all_labels)


def load_model_from_checkpoint(checkpoint_path, n_channels, device):
    """Auto-detect and load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    keys_str = str(ckpt["model_state_dict"].keys())

    model_type = ckpt_args.get("model", "jepa")
    if model_type == "mae" or "reconstruction_head" in keys_str:
        model_cls = EEGMAE
        model_name = "EEG-MAE"
    elif model_type == "lejepa" or ("pred_head" in keys_str and "predictor" not in keys_str):
        model_cls = EEGLeJEPA
        model_name = "EEG-LeJEPA"
    else:
        model_cls = EEGJEPA
        model_name = "EEG-JEPA"

    # Detect n_channels from checkpoint
    for key, val in ckpt["model_state_dict"].items():
        if "channel_embed" in key:
            n_channels = val.shape[0]
            break

    model = model_cls(
        n_channels=n_channels,
        d_model=ckpt_args.get("d_model", 256),
        encoder_layers=ckpt_args.get("encoder_layers", 6),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, model_name, n_channels


# ============================================================
# Plotting
# ============================================================

CLASS_NAMES = {0: "Left Fist", 1: "Right Fist", 2: "Both Fists", 3: "Both Feet"}
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
CMAP = ListedColormap(CLASS_COLORS)


def plot_scatter(ax, coords, labels, title, show_legend=True):
    """Plot 2D scatter with class colors."""
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=CLASS_COLORS[i % len(CLASS_COLORS)],
            label=CLASS_NAMES.get(label, f"Class {label}"),
            alpha=0.5, s=8, edgecolors='none',
        )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend:
        ax.legend(loc='best', fontsize=8, markerscale=3, framealpha=0.8)


def make_figures(random_reprs, random_labels, pretrained_reprs, pretrained_labels,
                 model_name, output_dir):
    """Generate all visualization figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- PCA ----
    print("Computing PCA...")
    pca = PCA(n_components=2)
    # Fit on combined data for consistent axes
    all_reprs = np.vstack([random_reprs, pretrained_reprs])
    pca.fit(all_reprs)
    random_pca = pca.transform(random_reprs)
    pretrained_pca = pca.transform(pretrained_reprs)
    var_explained = pca.explained_variance_ratio_

    # Side-by-side PCA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_scatter(ax1, random_pca, random_labels,
                 f"Random Init\nPC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")
    plot_scatter(ax2, pretrained_pca, pretrained_labels,
                 f"{model_name} (Pretrained)\nPC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}")
    fig.suptitle("PCA of Encoder Representations (Motor Imagery)", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "pca_comparison.png", dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / "pca_comparison.pdf", bbox_inches='tight')
    print(f"  Saved: {output_dir / 'pca_comparison.png'}")
    plt.close()

    # PCA variance explained bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    pca_full = PCA(n_components=min(50, pretrained_reprs.shape[1]))
    pca_full.fit(pretrained_reprs)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ax.bar(range(len(cumvar)), pca_full.explained_variance_ratio_, alpha=0.7, label="Individual")
    ax.plot(range(len(cumvar)), cumvar, 'r-o', markersize=3, label="Cumulative")
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label="90%")
    dims_90 = np.searchsorted(cumvar, 0.9) + 1
    ax.axvline(x=dims_90, color='orange', linestyle='--', alpha=0.5,
               label=f"90% at dim {dims_90}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title(f"{model_name}: PCA Variance (90% at {dims_90} dims)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "pca_variance.png", dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'pca_variance.png'}")
    plt.close()

    # ---- UMAP ----
    if HAS_UMAP:
        print("Computing UMAP (this may take a minute)...")
        umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        random_umap = umap.fit_transform(random_reprs)
        umap2 = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        pretrained_umap = umap2.fit_transform(pretrained_reprs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        plot_scatter(ax1, random_umap, random_labels, "Random Init")
        plot_scatter(ax2, pretrained_umap, pretrained_labels, f"{model_name} (Pretrained)")
        fig.suptitle("UMAP of Encoder Representations (Motor Imagery)", fontsize=15, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / "umap_comparison.png", dpi=200, bbox_inches='tight')
        fig.savefig(output_dir / "umap_comparison.pdf", bbox_inches='tight')
        print(f"  Saved: {output_dir / 'umap_comparison.png'}")
        plt.close()

    # ---- Per-class mean distance heatmap ----
    print("Computing class distance heatmap...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, reprs, labels, title in [
        (ax1, random_reprs, random_labels, "Random Init"),
        (ax2, pretrained_reprs, pretrained_labels, f"{model_name}"),
    ]:
        unique = np.unique(labels)
        n_cls = len(unique)
        # Cosine distance matrix between class means
        means = np.array([reprs[labels == c].mean(axis=0) for c in unique])
        norms = np.linalg.norm(means, axis=1, keepdims=True) + 1e-8
        means_norm = means / norms
        cos_sim = means_norm @ means_norm.T

        im = ax.imshow(1 - cos_sim, cmap='YlOrRd', vmin=0, vmax=0.05)
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        class_labels = [CLASS_NAMES.get(c, f"C{c}") for c in unique]
        ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        # Annotate values
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, f"{1-cos_sim[i,j]:.4f}", ha='center', va='center', fontsize=8)
    fig.suptitle("Inter-class Cosine Distance (higher = more separable)", fontsize=13)
    plt.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="1 - cosine similarity")
    plt.tight_layout()
    fig.savefig(output_dir / "class_distance_heatmap.png", dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'class_distance_heatmap.png'}")
    plt.close()

    print(f"\nAll figures saved to: {output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize EEG encoder representations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets/physionet")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_subjects", type=int, default=109)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent / "figures")

    # Load labeled data (val subjects)
    subjects = list(range(1, args.n_subjects + 1))
    n_train_subj = int(len(subjects) * 0.8)

    print("Loading val set (subjects 88-109)...")
    val_ds = PhysioNetMI_Labeled(
        subjects[n_train_subj:], data_dir=args.data_dir, use_ea=True,
    )
    # Remove rest, remap labels
    mask = val_ds.labels != 0
    val_ds.trials = val_ds.trials[mask]
    val_ds.labels = val_ds.labels[mask] - 1
    n_channels = len(val_ds.electrode_names)
    print(f"  {len(val_ds.trials)} MI trials, {len(np.unique(val_ds.labels))} classes")

    # Load pretrained model (auto-detect type and n_channels)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, model_name, ckpt_n_channels = load_model_from_checkpoint(
        args.checkpoint, n_channels, device,
    )
    print(f"  Model: {model_name}, {ckpt_n_channels}ch")

    # Remap channels if needed
    if ckpt_n_channels != n_channels:
        from dataset_multi import pick_common_channels
        ch_indices, _ = pick_common_channels(val_ds.electrode_names)
        if len(ch_indices) == ckpt_n_channels:
            print(f"  Remapping {n_channels}ch → {ckpt_n_channels}ch")
            val_ds.trials = val_ds.trials[:, :, ch_indices].copy()
            n_channels = ckpt_n_channels

    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    # Extract pretrained representations
    print("\nExtracting pretrained representations...")
    pre_reprs, pre_labels = extract_representations(model, loader, device)
    print(f"  Shape: {pre_reprs.shape}")

    # Extract random representations
    print("Extracting random init representations...")
    random_model = type(model)(n_channels=n_channels).to(device)
    rand_reprs, rand_labels = extract_representations(random_model, loader, device)
    del random_model

    # Generate figures
    print("\nGenerating figures...")
    make_figures(rand_reprs, rand_labels, pre_reprs, pre_labels,
                 model_name, args.output_dir)


if __name__ == "__main__":
    main()
