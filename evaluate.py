"""BrainWM v2 evaluation: linear probe, oddball, imagination.

Usage:
  # Linear probe on PhysioNet motor imagery (pretrained vs random)
  python evaluate.py --task linear_probe \
      --checkpoint /home/share/data_makchen/peng/models/brainwm/best_model.pt \
      --data_dir /home/share/data_makchen/peng/datasets/physionet

  # Oddball prediction error analysis
  python evaluate.py --task oddball \
      --checkpoint /home/share/data_makchen/peng/models/brainwm/best_model.pt \
      --data_dir /path/to/oddball_data

  # Imagination rollout quality
  python evaluate.py --task imagination \
      --checkpoint /home/share/data_makchen/peng/models/brainwm/best_model.pt \
      --data_dir /home/share/data_makchen/peng/datasets/physionet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from config import BrainWMConfig
from model import BrainWM

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# ============================================================
# 1. PhysioNet Motor Imagery Dataset (with labels)
# ============================================================

class PhysioNetMI_Labeled(Dataset):
    """PhysioNet Motor Imagery with task labels for linear probe evaluation.

    Labels: 0=rest, 1=left_fist, 2=right_fist, 3=both_fists, 4=both_feet
    Uses event annotations from the EDF files.
    """

    def __init__(self, subjects, sample_rate=256, trial_duration_s=4,
                 data_dir="/home/share/data_makchen/peng/datasets/physionet",
                 use_ea=True):
        if not MNE_AVAILABLE:
            raise ImportError("mne required: pip install mne")

        from mne.datasets import eegbci
        from mne.io import read_raw_edf
        from dataset import euclidean_alignment

        self.trials = []
        self.labels = []
        self.electrode_names = None
        self.trial_samples = sample_rate * trial_duration_s

        # Task mapping: run_id -> (event_label_mapping)
        # Runs 4,8,12: left fist (T1) vs right fist (T2)
        # Runs 6,10,14: both fists (T1) vs both feet (T2)
        run_groups = {
            "lr": [4, 8, 12],   # left/right fist
            "bf": [6, 10, 14],  # both fists/feet
        }
        label_map = {
            "lr": {"T0": 0, "T1": 1, "T2": 2},  # rest, left, right
            "bf": {"T0": 0, "T1": 3, "T2": 4},  # rest, both_fists, both_feet
        }

        for subj in subjects:
            try:
                # Load all runs for EA
                all_data = []
                all_runs_info = []  # (raw, group_key)

                for group_key, runs in run_groups.items():
                    files = eegbci.load_data(subj, runs, path=data_dir)
                    for f in files:
                        raw = read_raw_edf(f, preload=True, verbose=False)
                        if raw.info["sfreq"] != sample_rate:
                            raw.resample(sample_rate, verbose=False)
                        raw.filter(0.1, 75.0, verbose=False)
                        if self.electrode_names is None:
                            self.electrode_names = raw.ch_names
                        all_data.append(raw.get_data().T.astype(np.float32))
                        all_runs_info.append((raw, group_key))

                # EA on concatenated subject data
                subj_concat = np.concatenate(all_data, axis=0)
                if use_ea:
                    subj_concat = euclidean_alignment(subj_concat)

                # Re-split and extract labeled events
                offset = 0
                for (raw, group_key), run_data in zip(all_runs_info, all_data):
                    run_len = len(run_data)
                    run_aligned = subj_concat[offset:offset + run_len]
                    offset += run_len

                    # Extract events from annotations
                    events, event_id = mne.events_from_annotations(raw, verbose=False)
                    # Map event descriptions to our labels
                    desc_to_id = {}
                    for desc, eid in event_id.items():
                        for key in label_map[group_key]:
                            if key in desc:
                                desc_to_id[eid] = label_map[group_key][key]

                    for event in events:
                        onset_sample = int(event[0] * sample_rate / raw.info["sfreq"])
                        event_code = event[2]
                        if event_code not in desc_to_id:
                            continue
                        label = desc_to_id[event_code]

                        # Extract trial window
                        start = onset_sample
                        end = start + self.trial_samples
                        if end > len(run_aligned):
                            continue

                        trial = run_aligned[start:end].copy()
                        # Z-score
                        mean = trial.mean(axis=0, keepdims=True)
                        std = trial.std(axis=0, keepdims=True) + 1e-8
                        trial = (trial - mean) / std

                        self.trials.append(trial)
                        self.labels.append(label)

            except Exception as e:
                print(f"  Skipping subject {subj}: {e}")

        self.trials = np.array(self.trials, dtype=np.float32) if self.trials else np.zeros((0, self.trial_samples, 1))
        self.labels = np.array(self.labels, dtype=np.int64) if self.labels else np.zeros(0, dtype=np.int64)

        # Filter to classes with enough samples
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"  Loaded {len(self.trials)} labeled trials, classes: {dict(zip(unique, counts))}")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trials[idx]), self.labels[idx]


# ============================================================
# 2. Linear Probe
# ============================================================

class LinearProbe(nn.Module):
    """Frozen BrainWM encoder + trainable BN + linear classifier."""

    def __init__(self, brainwm: BrainWM, n_classes: int):
        super().__init__()
        self.brainwm = brainwm
        for p in self.brainwm.parameters():
            p.requires_grad = False
        self.bn = nn.BatchNorm1d(brainwm.config.brain_state_dim)
        self.classifier = nn.Linear(brainwm.config.brain_state_dim, n_classes)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.brainwm.eval()
            outputs = self.brainwm(eeg, return_predictions=False)
            pooled = outputs["brain_states"].mean(dim=1)  # [B, D]
        return self.classifier(self.bn(pooled))


def run_linear_probe(brainwm, train_loader, val_loader, n_classes, device,
                     n_epochs=100, lr=1e-3, label=""):
    """Train linear probe and return best accuracy."""
    probe = LinearProbe(brainwm, n_classes).to(device)
    optimizer = torch.optim.Adam(
        list(probe.bn.parameters()) + list(probe.classifier.parameters()), lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        # Train
        probe.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0.0
        for eeg, labels in train_loader:
            eeg, labels = eeg.to(device), labels.to(device).long()
            logits = probe(eeg)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(-1) == labels).sum().item()
            train_total += labels.shape[0]
            train_loss_sum += loss.item() * labels.shape[0]

        # Validate
        probe.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for eeg, labels in val_loader:
                eeg, labels = eeg.to(device), labels.to(device).long()
                logits = probe(eeg)
                preds = logits.argmax(-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} (best={best_acc:.4f} @ep{best_epoch})")

    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    per_class = {}
    for c in np.unique(all_labels):
        mask = all_labels == c
        per_class[int(c)] = float((all_preds[mask] == c).mean()) if mask.sum() > 0 else 0.0

    return best_acc, per_class


# ============================================================
# 3. Oddball Prediction Error Analysis
# ============================================================

def oddball_analysis(model, standard_trials, deviant_trials, device):
    """Core Nature experiment: PE comparison between standard and deviant."""
    model.eval()
    model = model.to(device)
    state_ms = model.config.state_duration_ms

    def batch_pe(trials, batch_size=32):
        global_pe, regional_pe = [], {r: [] for r in model.config.region_names}
        for i in range(0, len(trials), batch_size):
            batch = trials[i:i+batch_size].to(device)
            pe = model.compute_prediction_error(batch).cpu()
            global_pe.append(pe)
            rpe = model.compute_regional_prediction_error(batch)
            for r in rpe:
                regional_pe[r].append(rpe[r].cpu())
        global_pe = torch.cat(global_pe, dim=0)
        for r in regional_pe:
            regional_pe[r] = torch.cat(regional_pe[r], dim=0)
        return global_pe, regional_pe

    with torch.no_grad():
        std_pe, std_rpe = batch_pe(standard_trials)
        dev_pe, dev_rpe = batch_pe(deviant_trials)

    n_steps = std_pe.shape[1]
    time_ms = np.arange(n_steps) * state_ms + state_ms

    results = {
        "time_ms": time_ms,
        "state_duration_ms": state_ms,
        "standard_pe_mean": std_pe.mean(0).numpy(),
        "standard_pe_std": std_pe.std(0).numpy(),
        "deviant_pe_mean": dev_pe.mean(0).numpy(),
        "deviant_pe_std": dev_pe.std(0).numpy(),
        "pe_difference": (dev_pe.mean(0) - std_pe.mean(0)).numpy(),
        "regional_standard": {r: std_rpe[r].mean(0).numpy() for r in std_rpe},
        "regional_deviant": {r: dev_rpe[r].mean(0).numpy() for r in dev_rpe},
        "regional_difference": {
            r: (dev_rpe[r].mean(0) - std_rpe[r].mean(0)).numpy() for r in dev_rpe
        },
    }

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(dev_pe.numpy(), std_pe.numpy(), axis=0, equal_var=False)
    results["ttest_t"] = t_stat
    results["ttest_p"] = p_val

    mmn_idx = 200 // state_ms - 1
    p300_idx = 400 // state_ms - 1
    if mmn_idx < n_steps:
        results["mmn_pe_diff"] = results["pe_difference"][mmn_idx]
        results["mmn_frontal_diff"] = results["regional_difference"]["frontal"][mmn_idx]
    if p300_idx < n_steps:
        results["p300_pe_diff"] = results["pe_difference"][p300_idx]
        results["p300_parietal_diff"] = results["regional_difference"]["parietal"][p300_idx]

    return results


# ============================================================
# 4. Imagination Rollout
# ============================================================

def evaluate_imagination(model, trials, n_context_states=20, n_predict_states=20, device="cpu"):
    model.eval()
    model = model.to(torch.device(device))
    config = model.config
    context_samples = n_context_states * config.state_samples
    total_samples = (n_context_states + n_predict_states) * config.state_samples

    per_step_cos = {s: [] for s in range(n_predict_states)}

    with torch.no_grad():
        for i in range(0, len(trials), 32):
            batch = trials[i:i+32, :total_samples, :].to(device)
            context = batch[:, :context_samples, :]
            imagined = model.predict_future(context, n_future_steps=n_predict_states)

            full_out = model(batch, return_predictions=False)
            actual = full_out["brain_states"][:, n_context_states:n_context_states+n_predict_states, :]

            for s in range(min(n_predict_states, imagined.shape[1], actual.shape[1])):
                cos = F.cosine_similarity(imagined[:, s], actual[:, s], dim=-1)
                per_step_cos[s].append(cos.mean().item())

    return {
        "cosine_per_step": {s: np.mean(v) for s, v in per_step_cos.items() if v},
    }


# ============================================================
# 5. Main
# ============================================================

def load_model(checkpoint_path, device):
    """Load BrainWM from checkpoint."""
    print(f"Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = BrainWMConfig()

    # Determine n_subjects from checkpoint
    # Try to find subject_adversary weight shape
    n_subjects = 109  # default
    for key in ckpt["model_state_dict"]:
        if "subject_adversary.classifier" in key and "weight" in key:
            n_subjects = ckpt["model_state_dict"][key].shape[0]
            break

    model = BrainWM(config, n_subjects=n_subjects)
    model.initialize_electrodes(ckpt["electrode_names"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Model loaded (n_subjects={n_subjects})")
    return model, ckpt["electrode_names"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate BrainWM v2")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint. If not provided, tests random init baseline.")
    parser.add_argument("--task", type=str, required=True,
                        choices=["linear_probe", "oddball", "imagination"])
    parser.add_argument("--data_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets/physionet")
    parser.add_argument("--output_dir", type=str,
                        default="/home/share/data_makchen/peng/models/brainwm/results")
    parser.add_argument("--n_subjects", type=int, default=109,
                        help="Number of subjects for linear probe evaluation")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Linear Probe ----
    if args.task == "linear_probe":
        print("\n===== Linear Probe: PhysioNet Motor Imagery =====")

        # Load labeled dataset
        print("Loading labeled PhysioNet MI data...")
        subjects = list(range(1, args.n_subjects + 1))

        # Split: 80% subjects for train, 20% for val (cross-subject!)
        n_train_subj = int(len(subjects) * 0.8)
        train_subjects = subjects[:n_train_subj]
        val_subjects = subjects[n_train_subj:]
        print(f"  Train subjects: {train_subjects[0]}-{train_subjects[-1]} ({len(train_subjects)})")
        print(f"  Val subjects: {val_subjects[0]}-{val_subjects[-1]} ({len(val_subjects)})")

        train_dataset = PhysioNetMI_Labeled(
            train_subjects, data_dir=args.data_dir, use_ea=True,
        )
        val_dataset = PhysioNetMI_Labeled(
            val_subjects, data_dir=args.data_dir, use_ea=True,
        )
        electrode_names = train_dataset.electrode_names

        n_classes = len(np.unique(train_dataset.labels))
        print(f"  Classes: {n_classes}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)

        results = {}

        # --- Run 1: Random init baseline ---
        print("\n--- Random Init Baseline ---")
        config = BrainWMConfig()
        random_model = BrainWM(config, n_subjects=args.n_subjects)
        random_model.initialize_electrodes(electrode_names)
        random_model = random_model.to(device)
        random_acc, random_per_class = run_linear_probe(
            random_model, train_loader, val_loader, n_classes, device,
            n_epochs=args.probe_epochs, lr=args.probe_lr, label="Random",
        )
        results["random_init"] = {"accuracy": random_acc, "per_class": random_per_class}
        print(f"  Random Init Best Acc: {random_acc:.4f}")
        del random_model
        torch.cuda.empty_cache()

        # --- Run 2: Pretrained ---
        if args.checkpoint:
            print("\n--- Pretrained Model ---")
            pretrained_model, _ = load_model(args.checkpoint, device)
            pretrained_model.initialize_electrodes(electrode_names)
            pre_acc, pre_per_class = run_linear_probe(
                pretrained_model, train_loader, val_loader, n_classes, device,
                n_epochs=args.probe_epochs, lr=args.probe_lr, label="Pretrained",
            )
            results["pretrained"] = {"accuracy": pre_acc, "per_class": pre_per_class}
            print(f"  Pretrained Best Acc: {pre_acc:.4f}")
            del pretrained_model
            torch.cuda.empty_cache()

            # --- Summary ---
            improvement = pre_acc - random_acc
            print(f"\n===== RESULTS =====")
            print(f"  Random Init:  {random_acc:.4f}")
            print(f"  Pretrained:   {pre_acc:.4f}")
            print(f"  Improvement:  {improvement:+.4f} ({improvement*100:+.1f}%)")
            print(f"  Gate (>5%):   {'PASS' if improvement > 0.05 else 'FAIL'}")
            results["improvement"] = improvement
        else:
            print(f"\n  No checkpoint provided. Random init acc: {random_acc:.4f}")
            print(f"  Run with --checkpoint to compare pretrained vs random.")

        # Save results
        with open(output_dir / "linear_probe_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {output_dir / 'linear_probe_results.json'}")

    # ---- Oddball ----
    elif args.task == "oddball":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for oddball analysis")

        model, _ = load_model(args.checkpoint, device)
        data_path = Path(args.data_dir)
        standard = torch.from_numpy(np.load(data_path / "standard.npy"))
        deviant = torch.from_numpy(np.load(data_path / "deviant.npy"))
        results = oddball_analysis(model, standard, deviant, device)

        print(f"\n===== Oddball Results (100ms resolution) =====")
        print(f"  Time axis: {results['time_ms']}")
        print(f"  PE difference: {results['pe_difference']}")
        if "mmn_frontal_diff" in results:
            print(f"  MMN (~200ms) frontal: {results['mmn_frontal_diff']:.4f}")
        if "p300_parietal_diff" in results:
            print(f"  P300 (~400ms) parietal: {results['p300_parietal_diff']:.4f}")

        np.savez(output_dir / "oddball_results.npz", **{
            k: v for k, v in results.items() if isinstance(v, np.ndarray)
        })
        print(f"Results saved → {output_dir}")

    # ---- Imagination ----
    elif args.task == "imagination":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for imagination evaluation")

        model, _ = load_model(args.checkpoint, device)

        # Use PhysioNet data for imagination test
        from dataset import PhysioNetMIDataset
        dataset = PhysioNetMIDataset(
            subjects=list(range(1, min(args.n_subjects + 1, 20))),
            data_dir=args.data_dir, use_ea=True,
        )
        trials = torch.stack([dataset[i][0] for i in range(min(200, len(dataset)))])
        results = evaluate_imagination(model, trials, device=str(device))

        print(f"\n===== Imagination Rollout =====")
        for step, cos in results["cosine_per_step"].items():
            print(f"  Step {step} (+{(step+1)*100}ms): cosine={cos:.4f}")

        with open(output_dir / "imagination_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved → {output_dir}")


if __name__ == "__main__":
    main()
