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
    train_dataset_full = PhysioNetMI_Labeled(
        train_subjects, data_dir=args.data_dir, use_ea=True,
    )
    print("Loading val set...")
    val_dataset_full = PhysioNetMI_Labeled(
        val_subjects, data_dir=args.data_dir, use_ea=True,
    )
    n_channels = len(train_dataset_full.electrode_names)
    electrode_names = train_dataset_full.electrode_names

    # Detect checkpoint n_channels for channel remapping
    ckpt_n_channels = n_channels
    if args.checkpoint:
        ckpt_tmp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        for key, val in ckpt_tmp["model_state_dict"].items():
            if "channel_embed" in key:
                ckpt_n_channels = val.shape[0]
                break
        del ckpt_tmp

    ch_indices = None
    if ckpt_n_channels != n_channels:
        from dataset_multi import pick_common_channels
        ch_indices, _ = pick_common_channels(electrode_names)
        if len(ch_indices) == ckpt_n_channels:
            print(f"Remapping eval data: {n_channels}ch → {ckpt_n_channels}ch")
            n_channels = ckpt_n_channels

    # ---- Define MI sub-tasks (matching Laya Table 1) ----
    # Original labels: 0=rest, 1=left_fist, 2=right_fist, 3=both_fists, 4=both_feet
    MI_TASKS = {
        "LH_vs_RH":    {"keep": [1, 2], "remap": {1: 0, 2: 1}, "chance": 0.5},
        "RH_vs_Feet":  {"keep": [2, 4], "remap": {2: 0, 4: 1}, "chance": 0.5},
        "4_Class_MI":  {"keep": [1, 2, 3, 4], "remap": {1: 0, 2: 1, 3: 2, 4: 3}, "chance": 0.25},
    }

    def make_task_datasets(train_full, val_full, keep_labels, remap):
        """Filter and remap labels for a specific MI sub-task."""
        import copy
        def filter_ds(ds):
            mask = np.isin(ds.labels, keep_labels)
            trials = ds.trials[mask].copy()
            labels = ds.labels[mask].copy()
            # Remap
            new_labels = np.zeros_like(labels)
            for old, new in remap.items():
                new_labels[labels == old] = new
            # Channel remap if needed
            if ch_indices is not None:
                trials = trials[:, :, ch_indices].copy()
            return trials, new_labels
        tr_trials, tr_labels = filter_ds(train_full)
        va_trials, va_labels = filter_ds(val_full)
        return tr_trials, tr_labels, va_trials, va_labels

    # ---- Load checkpoint once ----
    ckpt = None
    ckpt_args = {}
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
              f"val_loss={ckpt.get('val_loss', '?'):.6f}")

    def load_pretrained():
        m = EEGJEPA(
            n_channels=ckpt_n_channels,
            d_model=ckpt_args.get("d_model", 256),
            encoder_layers=ckpt_args.get("encoder_layers", 6),
        ).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        return m

    # ---- Run all MI sub-tasks ----
    all_results = {}

    for task_name, task_cfg in MI_TASKS.items():
        print(f"\n{'='*60}")
        print(f"  Task: {task_name} (chance={task_cfg['chance']})")
        print(f"{'='*60}")

        tr_trials, tr_labels, va_trials, va_labels = make_task_datasets(
            train_dataset_full, val_dataset_full,
            task_cfg["keep"], task_cfg["remap"],
        )
        n_classes = len(task_cfg["remap"])
        unique, counts = np.unique(tr_labels, return_counts=True)
        print(f"  Train: {len(tr_labels)} trials, classes: {dict(zip(unique, counts))}")
        unique, counts = np.unique(va_labels, return_counts=True)
        print(f"  Val:   {len(va_labels)} trials, classes: {dict(zip(unique, counts))}")

        # Wrap as simple dataset
        class SimpleDS:
            def __init__(self, trials, labels):
                self.trials = trials
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                return torch.from_numpy(self.trials[idx]), self.labels[idx]

        train_loader = DataLoader(SimpleDS(tr_trials, tr_labels),
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, drop_last=True)
        val_loader = DataLoader(SimpleDS(va_trials, va_labels),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=4)

        task_results = {}

        # Random frozen probe
        print(f"\n  --- Random frozen probe ---")
        random_model = EEGJEPA(n_channels=n_channels).to(device)
        r_acc, _ = run_probe(random_model, train_loader, val_loader, n_classes,
                             device, n_epochs=args.probe_epochs, lr=args.probe_lr,
                             label=f"{task_name}/Random")
        task_results["random_frozen"] = r_acc
        del random_model; torch.cuda.empty_cache()

        if ckpt is not None:
            # Pretrained frozen probe
            print(f"\n  --- Pretrained frozen probe ---")
            pre_model = load_pretrained()
            p_acc, _ = run_probe(pre_model, train_loader, val_loader, n_classes,
                                 device, n_epochs=args.probe_epochs, lr=args.probe_lr,
                                 label=f"{task_name}/Frozen")
            task_results["pretrained_frozen"] = p_acc
            del pre_model; torch.cuda.empty_cache()

            # Pretrained fine-tune
            print(f"\n  --- Pretrained fine-tune ---")
            pre_ft = load_pretrained()
            pft_acc, _ = run_finetune(pre_ft, train_loader, val_loader, n_classes,
                                      device, n_epochs=args.probe_epochs, lr=1e-4,
                                      label=f"{task_name}/FT-pre")
            task_results["pretrained_finetune"] = pft_acc
            del pre_ft; torch.cuda.empty_cache()

            # Random fine-tune
            print(f"\n  --- Random fine-tune ---")
            rand_ft = EEGJEPA(n_channels=n_channels).to(device)
            rft_acc, _ = run_finetune(rand_ft, train_loader, val_loader, n_classes,
                                      device, n_epochs=args.probe_epochs, lr=1e-4,
                                      label=f"{task_name}/FT-rand")
            task_results["random_finetune"] = rft_acc
            del rand_ft; torch.cuda.empty_cache()

        all_results[task_name] = task_results

    # ---- Final Summary Table ----
    print(f"\n{'='*70}")
    print(f"  FULL RESULTS (balanced accuracy)")
    print(f"{'='*70}")
    print(f"  {'Task':<16} {'Chance':>7} {'Rand-F':>7} {'Pre-F':>7} {'Rand-FT':>8} {'Pre-FT':>8}")
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

    for task_name, task_cfg in MI_TASKS.items():
        r = all_results[task_name]
        chance = task_cfg["chance"]
        rf = r.get("random_frozen", 0)
        pf = r.get("pretrained_frozen", 0)
        rft = r.get("random_finetune", 0)
        pft = r.get("pretrained_finetune", 0)
        print(f"  {task_name:<16} {chance:>7.3f} {rf:>7.3f} {pf:>7.3f} {rft:>8.3f} {pft:>8.3f}")

    # Mean across tasks
    if ckpt is not None:
        tasks = list(all_results.keys())
        mean_rf = np.mean([all_results[t]["random_frozen"] for t in tasks])
        mean_pf = np.mean([all_results[t].get("pretrained_frozen", 0) for t in tasks])
        mean_rft = np.mean([all_results[t].get("random_finetune", 0) for t in tasks])
        mean_pft = np.mean([all_results[t].get("pretrained_finetune", 0) for t in tasks])
        print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
        print(f"  {'Mean':<16} {'':>7} {mean_rf:>7.3f} {mean_pf:>7.3f} {mean_rft:>8.3f} {mean_pft:>8.3f}")
        print(f"\n  Pretraining value (frozen):   {mean_pf-mean_rf:+.4f}")
        print(f"  Pretraining value (finetune): {mean_pft-mean_rft:+.4f}")
    print(f"{'='*70}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "mi_probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {output_dir / 'mi_probe_results.json'}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "linear_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {output_dir / 'linear_probe_results.json'}")


if __name__ == "__main__":
    main()
