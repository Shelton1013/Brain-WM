"""
Evaluate EEG-JEPA: linear probe on PhysioNet Motor Imagery.

Compares:
  1. Pretrained EEG-JEPA (frozen encoder + linear head)
  2. Random init baseline (same architecture, no pretraining)

Usage:
  python eval_jepa.py \
      --checkpoint /home/share/data_makchen/peng/models/eeg_jepa/best_model.pt \
      --data_dir /home/share/data_makchen/peng/datasets/physionet

  # Random-only baseline (no checkpoint)
  python eval_jepa.py --data_dir /home/share/data_makchen/peng/datasets/physionet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_jepa import EEGJEPA
from evaluate import PhysioNetMI_Labeled  # reuse labeled dataset


# ============================================================
# Linear Probe for EEG-JEPA
# ============================================================

class JEPALinearProbe(nn.Module):
    """Frozen EEG-JEPA encoder + trainable BN + linear classifier."""

    def __init__(self, jepa: EEGJEPA, n_classes: int):
        super().__init__()
        self.jepa = jepa
        for p in self.jepa.parameters():
            p.requires_grad = False
        self.bn = nn.BatchNorm1d(jepa.d_model)
        self.classifier = nn.Linear(jepa.d_model, n_classes)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.jepa.eval()
            # Tokenize + encode ALL positions (no masking during eval)
            tokens = self.jepa._tokenize(eeg)    # [B, N, D]
            encoded = self.jepa._encode(tokens)   # [B, N, D]
            pooled = encoded.mean(dim=1)          # [B, D]
        return self.classifier(self.bn(pooled))


def run_probe(jepa, train_loader, val_loader, n_classes, device,
              n_epochs=100, lr=1e-3, label=""):
    """Train linear probe and return best val accuracy."""
    probe = JEPALinearProbe(jepa, n_classes).to(device)
    optimizer = torch.optim.Adam(
        list(probe.bn.parameters()) + list(probe.classifier.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        # Train
        probe.train()
        correct, total = 0, 0
        for eeg, labels in train_loader:
            eeg, labels = eeg.to(device), labels.to(device).long()
            logits = probe(eeg)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.shape[0]
        train_acc = correct / max(total, 1)

        # Validate
        probe.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for eeg, labels in val_loader:
                eeg, labels = eeg.to(device), labels.to(device).long()
                logits = probe(eeg)
                preds = logits.argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.shape[0]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / max(total, 1)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch}: train={train_acc:.4f} "
                  f"val={val_acc:.4f} (best={best_acc:.4f} @ep{best_epoch})")

    # Per-class accuracy at final epoch
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    per_class = {}
    for c in np.unique(all_labels):
        mask = all_labels == c
        per_class[int(c)] = float((all_preds[mask] == c).mean())

    return best_acc, per_class


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate EEG-JEPA linear probe")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets/physionet")
    parser.add_argument("--output_dir", type=str,
                        default="/home/share/data_makchen/peng/models/eeg_jepa/results")
    parser.add_argument("--n_subjects", type=int, default=109)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    # ---- Load labeled data (cross-subject split) ----
    print("\n===== Linear Probe: PhysioNet Motor Imagery =====")
    subjects = list(range(1, args.n_subjects + 1))
    n_train_subj = int(len(subjects) * 0.8)
    train_subjects = subjects[:n_train_subj]
    val_subjects = subjects[n_train_subj:]
    print(f"Train subjects: 1-{n_train_subj} ({n_train_subj})")
    print(f"Val subjects: {n_train_subj+1}-{args.n_subjects} ({len(val_subjects)})")

    print("Loading train set...")
    train_dataset = PhysioNetMI_Labeled(
        train_subjects, data_dir=args.data_dir, use_ea=True,
    )
    print("Loading val set...")
    val_dataset = PhysioNetMI_Labeled(
        val_subjects, data_dir=args.data_dir, use_ea=True,
    )
    n_channels = len(train_dataset.electrode_names)
    n_classes = len(np.unique(train_dataset.labels))
    print(f"Channels: {n_channels}, Classes: {n_classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    results = {}

    # ---- Random init baseline ----
    print("\n--- Random Init Baseline ---")
    # Build model with same arch but no pretraining
    random_model = EEGJEPA(n_channels=n_channels).to(device)
    random_acc, random_pc = run_probe(
        random_model, train_loader, val_loader, n_classes, device,
        n_epochs=args.probe_epochs, lr=args.probe_lr, label="Random",
    )
    results["random_init"] = {"accuracy": random_acc, "per_class": random_pc}
    print(f"  Random Init Best Acc: {random_acc:.4f} (chance={1/n_classes:.4f})")
    del random_model
    torch.cuda.empty_cache()

    # ---- Pretrained ----
    if args.checkpoint:
        print("\n--- Pretrained EEG-JEPA ---")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})

        pretrained = EEGJEPA(
            n_channels=n_channels,
            d_model=ckpt_args.get("d_model", 256),
            encoder_layers=ckpt_args.get("encoder_layers", 6),
        ).to(device)
        pretrained.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
              f"val_loss={ckpt.get('val_loss', '?')}")

        pre_acc, pre_pc = run_probe(
            pretrained, train_loader, val_loader, n_classes, device,
            n_epochs=args.probe_epochs, lr=args.probe_lr, label="Pretrained",
        )
        results["pretrained"] = {"accuracy": pre_acc, "per_class": pre_pc}

        improvement = pre_acc - random_acc
        print(f"\n===== RESULTS =====")
        print(f"  Chance level:  {1/n_classes:.4f}")
        print(f"  Random Init:   {random_acc:.4f}")
        print(f"  Pretrained:    {pre_acc:.4f}")
        print(f"  Improvement:   {improvement:+.4f} ({improvement*100:+.1f}%)")
        print(f"  Gate (>5%):    {'PASS' if improvement > 0.05 else 'FAIL'}")
        results["improvement"] = improvement
    else:
        print(f"\nNo checkpoint. Random acc: {random_acc:.4f} (chance={1/n_classes:.4f})")
        print("Run with --checkpoint to compare pretrained vs random.")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "linear_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {output_dir / 'linear_probe_results.json'}")


if __name__ == "__main__":
    main()
