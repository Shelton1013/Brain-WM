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
            tokens = self.jepa._tokenize(eeg)
            encoded = self.jepa._encode(tokens)
            pooled = encoded.mean(dim=1)
        return self.classifier(self.bn(pooled))


class JEPAFinetune(nn.Module):
    """Unfrozen EEG-JEPA encoder + classifier. End-to-end fine-tuning."""

    def __init__(self, jepa: EEGJEPA, n_classes: int):
        super().__init__()
        self.jepa = jepa  # NOT frozen
        self.bn = nn.BatchNorm1d(jepa.d_model)
        self.classifier = nn.Linear(jepa.d_model, n_classes)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        tokens = self.jepa._tokenize(eeg)
        encoded = self.jepa._encode(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(self.bn(pooled))


def _train_eval_loop(model, train_loader, val_loader, device, optimizer,
                     n_epochs, label):
    """Shared train/eval loop for both probe and finetune."""
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
        all_preds, all_labels = [], []
        with torch.no_grad():
            for eeg, labels in val_loader:
                eeg, labels = eeg.to(device), labels.to(device).long()
                logits = model(eeg)
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

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    per_class = {}
    for c in np.unique(all_labels):
        mask = all_labels == c
        per_class[int(c)] = float((all_preds[mask] == c).mean())

    return best_acc, per_class


def run_probe(jepa, train_loader, val_loader, n_classes, device,
              n_epochs=100, lr=1e-3, label=""):
    """Frozen encoder + trainable linear head."""
    probe = JEPALinearProbe(jepa, n_classes).to(device)
    optimizer = torch.optim.Adam(
        list(probe.bn.parameters()) + list(probe.classifier.parameters()),
        lr=lr,
    )
    return _train_eval_loop(probe, train_loader, val_loader, device,
                            optimizer, n_epochs, label)


def run_finetune(jepa, train_loader, val_loader, n_classes, device,
                 n_epochs=100, lr=1e-4, label=""):
    """Unfrozen encoder + classifier. End-to-end with lower LR."""
    model = JEPAFinetune(jepa, n_classes).to(device)
    # Layer-wise LR: encoder gets smaller LR, head gets larger
    encoder_params = list(model.jepa.parameters())
    head_params = list(model.bn.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr},           # encoder: small LR
        {"params": head_params, "lr": lr * 10},          # head: 10x LR
    ], weight_decay=0.01)
    return _train_eval_loop(model, train_loader, val_loader, device,
                            optimizer, n_epochs, label)


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

    # Remove rest class (label=0) — it dominates and makes accuracy misleading
    # Only evaluate on motor imagery: left(1), right(2), both_fists(3), both_feet(4)
    def filter_rest(dataset):
        mask = dataset.labels != 0
        dataset.trials = dataset.trials[mask]
        dataset.labels = dataset.labels[mask]
        # Remap: 1->0, 2->1, 3->2, 4->3
        dataset.labels = dataset.labels - 1
        unique, counts = np.unique(dataset.labels, return_counts=True)
        print(f"  After removing rest: {len(dataset.trials)} trials, "
              f"classes: {dict(zip(unique, counts))}")

    filter_rest(train_dataset)
    filter_rest(val_dataset)

    n_classes = len(np.unique(train_dataset.labels))
    print(f"Channels: {n_channels}, Classes: {n_classes} (motor imagery only)")

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

    # ---- Pretrained (frozen probe) ----
    if args.checkpoint:
        print("\n--- Pretrained EEG-JEPA (frozen probe) ---")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})

        def load_pretrained():
            m = EEGJEPA(
                n_channels=n_channels,
                d_model=ckpt_args.get("d_model", 256),
                encoder_layers=ckpt_args.get("encoder_layers", 6),
            ).to(device)
            m.load_state_dict(ckpt["model_state_dict"])
            return m

        pretrained = load_pretrained()
        print(f"  Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
              f"val_loss={ckpt.get('val_loss', '?')}")

        pre_acc, pre_pc = run_probe(
            pretrained, train_loader, val_loader, n_classes, device,
            n_epochs=args.probe_epochs, lr=args.probe_lr, label="Frozen",
        )
        results["pretrained_frozen"] = {"accuracy": pre_acc, "per_class": pre_pc}
        del pretrained
        torch.cuda.empty_cache()

        # ---- Fine-tune from pretrained ----
        print("\n--- Pretrained EEG-JEPA (fine-tune) ---")
        pretrained_ft = load_pretrained()
        ft_acc, ft_pc = run_finetune(
            pretrained_ft, train_loader, val_loader, n_classes, device,
            n_epochs=args.probe_epochs, lr=1e-4, label="Finetune-pre",
        )
        results["pretrained_finetune"] = {"accuracy": ft_acc, "per_class": ft_pc}
        del pretrained_ft
        torch.cuda.empty_cache()

        # ---- Fine-tune from random init ----
        print("\n--- Random Init (fine-tune) ---")
        random_ft = EEGJEPA(n_channels=n_channels).to(device)
        rft_acc, rft_pc = run_finetune(
            random_ft, train_loader, val_loader, n_classes, device,
            n_epochs=args.probe_epochs, lr=1e-4, label="Finetune-rand",
        )
        results["random_finetune"] = {"accuracy": rft_acc, "per_class": rft_pc}
        del random_ft
        torch.cuda.empty_cache()

        # ---- Summary ----
        print(f"\n{'='*50}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"  Chance level:           {1/n_classes:.4f}")
        print(f"  Random frozen probe:    {random_acc:.4f}")
        print(f"  Pretrained frozen probe:{pre_acc:.4f} ({pre_acc-random_acc:+.4f})")
        print(f"  Random fine-tune:       {rft_acc:.4f}")
        print(f"  Pretrained fine-tune:   {ft_acc:.4f} ({ft_acc-rft_acc:+.4f})")
        print(f"{'='*50}")
        print(f"  Pretraining value (frozen):   {pre_acc-random_acc:+.4f}")
        print(f"  Pretraining value (finetune): {ft_acc-rft_acc:+.4f}")
        results["improvement_frozen"] = pre_acc - random_acc
        results["improvement_finetune"] = ft_acc - rft_acc
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
