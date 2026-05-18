"""
Supervised baselines for PhysioNet Motor Imagery.

Establishes upper bound: if supervised can't do well, self-supervised won't either.

Baselines:
  1. Linear: raw EEG → flatten → linear → 4 classes
  2. JEPA-arch: same JEPA encoder (not frozen) + linear head, end-to-end
  3. Simple CNN: 1D convnet baseline

Usage:
  CUDA_VISIBLE_DEVICES=3 python eval_supervised.py \
      --data_dir /home/share/data_makchen/peng/datasets/physionet
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluate import PhysioNetMI_Labeled


class LinearBaseline(nn.Module):
    """Flatten raw EEG and classify with linear layer."""
    def __init__(self, trial_samples, n_channels, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(trial_samples * n_channels),
            nn.Linear(trial_samples * n_channels, n_classes),
        )
    def forward(self, x):
        return self.classifier(x)


class CNNBaseline(nn.Module):
    """Simple 1D CNN over time, then pool + classify."""
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            # [B, C, T]
            nn.Conv1d(n_channels, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


class JEPAArchSupervised(nn.Module):
    """Same arch as EEG-JEPA encoder, but trained end-to-end supervised."""
    def __init__(self, n_channels, state_samples, d_model, n_layers, n_classes):
        super().__init__()
        from eeg_jepa import TransformerBlock
        token_dim = state_samples * n_channels
        self.state_samples = state_samples
        self.patch_proj = nn.Linear(token_dim, d_model)
        self.patch_norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, 8) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, eeg):
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S
        windows = eeg[:, :N*S, :].reshape(B, N, S, C).reshape(B, N, S * C)
        x = self.patch_norm(self.patch_proj(windows))
        x = x + self.pos_embed[:, :N, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)  # global avg pool
        return self.classifier(x)


def train_and_eval(model, train_loader, val_loader, device, n_epochs=100,
                   lr=1e-3, label=""):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        correct, total = 0, 0
        for eeg, labels in train_loader:
            eeg, labels = eeg.to(device), labels.to(device).long()
            logits = model(eeg)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.shape[0]
        train_acc = correct / max(total, 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for eeg, labels in val_loader:
                eeg, labels = eeg.to(device), labels.to(device).long()
                logits = model(eeg)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.shape[0]
        val_acc = correct / max(total, 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch}: train={train_acc:.4f} "
                  f"val={val_acc:.4f} (best={best_acc:.4f} @ep{best_epoch})")

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets/physionet")
    parser.add_argument("--n_subjects", type=int, default=109)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    # Load data (cross-subject split, no rest class)
    subjects = list(range(1, args.n_subjects + 1))
    n_train_subj = int(len(subjects) * 0.8)

    print("Loading train set...")
    train_ds = PhysioNetMI_Labeled(subjects[:n_train_subj], data_dir=args.data_dir)
    print("Loading val set...")
    val_ds = PhysioNetMI_Labeled(subjects[n_train_subj:], data_dir=args.data_dir)

    # Remove rest
    for ds in [train_ds, val_ds]:
        mask = ds.labels != 0
        ds.trials = ds.trials[mask]
        ds.labels = ds.labels[mask] - 1
        unique, counts = np.unique(ds.labels, return_counts=True)
        print(f"  {len(ds.trials)} trials, classes: {dict(zip(unique, counts))}")

    n_channels = len(train_ds.electrode_names)
    trial_samples = train_ds.trials.shape[1]
    n_classes = len(np.unique(train_ds.labels))
    print(f"\nChannels={n_channels}, Samples={trial_samples}, Classes={n_classes}")
    print(f"Chance level: {1/n_classes:.4f}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    results = {}

    # 1. Linear baseline
    print("=" * 50)
    print("Baseline 1: Linear (flatten + linear)")
    model = LinearBaseline(trial_samples, n_channels, n_classes)
    acc = train_and_eval(model, train_loader, val_loader, device,
                         n_epochs=args.epochs, lr=1e-3, label="Linear")
    results["linear"] = acc
    print(f"  → Best: {acc:.4f}\n")

    # 2. CNN baseline
    print("=" * 50)
    print("Baseline 2: 1D CNN")
    model = CNNBaseline(n_channels, n_classes)
    acc = train_and_eval(model, train_loader, val_loader, device,
                         n_epochs=args.epochs, lr=1e-3, label="CNN")
    results["cnn"] = acc
    print(f"  → Best: {acc:.4f}\n")

    # 3. JEPA-arch supervised
    print("=" * 50)
    print("Baseline 3: JEPA-arch (ViT, supervised end-to-end)")
    model = JEPAArchSupervised(n_channels, 26, 256, 6, n_classes)
    acc = train_and_eval(model, train_loader, val_loader, device,
                         n_epochs=args.epochs, lr=3e-4, label="ViT-sup")
    results["vit_supervised"] = acc
    print(f"  → Best: {acc:.4f}\n")

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print(f"  Chance:          {1/n_classes:.4f}")
    for name, acc in results.items():
        print(f"  {name:20s} {acc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
