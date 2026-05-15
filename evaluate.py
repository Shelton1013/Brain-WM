"""BrainWM v2 evaluation: downstream tasks + predictive coding analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import BrainWMConfig
from model import BrainWM


# ============================================================
# 1. Linear Probe
# ============================================================

class LinearProbe(nn.Module):
    """Frozen BrainWM encoder + trainable linear classifier."""

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
            pooled = outputs["brain_states"].mean(dim=1)
        return self.classifier(self.bn(pooled))


def train_linear_probe(brainwm, train_loader, val_loader, n_classes, device,
                       n_epochs=50, lr=1e-3):
    probe = LinearProbe(brainwm, n_classes).to(device)
    optimizer = torch.optim.Adam(
        list(probe.bn.parameters()) + list(probe.classifier.parameters()), lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        probe.train()
        for eeg, labels in train_loader:
            eeg, labels = eeg.to(device), labels.to(device)
            loss = criterion(probe(eeg), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for eeg, labels in val_loader:
                eeg, labels = eeg.to(device), labels.to(device)
                correct += (probe(eeg).argmax(-1) == labels).sum().item()
                total += labels.shape[0]
        acc = correct / total
        best_acc = max(best_acc, acc)
        if epoch % 10 == 0:
            print(f"  Probe ep{epoch}: val_acc={acc:.4f}")

    return best_acc


# ============================================================
# 2. Oddball Prediction Error Analysis (100ms resolution)
# ============================================================

def oddball_analysis(model, standard_trials, deviant_trials, device):
    """Core Nature experiment: compare model PE between standard and deviant trials.

    With 100ms resolution, we can distinguish:
      - MMN at ~200ms (state 2) → frontal PE
      - P300 at ~400ms (state 4) → parietal PE

    Args:
        standard_trials: [N_std, T, C]
        deviant_trials:  [N_dev, T, C]
    Returns:
        dict with PE timecourses, regional breakdown, stats
    """
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
        # Global PE
        "standard_pe_mean": std_pe.mean(0).numpy(),
        "standard_pe_std": std_pe.std(0).numpy(),
        "deviant_pe_mean": dev_pe.mean(0).numpy(),
        "deviant_pe_std": dev_pe.std(0).numpy(),
        "pe_difference": (dev_pe.mean(0) - std_pe.mean(0)).numpy(),
        # Regional PE
        "regional_standard": {r: std_rpe[r].mean(0).numpy() for r in std_rpe},
        "regional_deviant": {r: dev_rpe[r].mean(0).numpy() for r in dev_rpe},
        "regional_difference": {
            r: (dev_rpe[r].mean(0) - std_rpe[r].mean(0)).numpy() for r in dev_rpe
        },
    }

    # Stats
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(dev_pe.numpy(), std_pe.numpy(), axis=0, equal_var=False)
    results["ttest_t"] = t_stat
    results["ttest_p"] = p_val

    # Key timepoints
    mmn_idx = 200 // state_ms - 1   # ~200ms → state index
    p300_idx = 400 // state_ms - 1  # ~400ms → state index
    if mmn_idx < n_steps:
        results["mmn_pe_diff"] = results["pe_difference"][mmn_idx]
        results["mmn_frontal_diff"] = results["regional_difference"]["frontal"][mmn_idx]
    if p300_idx < n_steps:
        results["p300_pe_diff"] = results["pe_difference"][p300_idx]
        results["p300_parietal_diff"] = results["regional_difference"]["parietal"][p300_idx]

    return results


# ============================================================
# 3. Imagination Rollout
# ============================================================

def evaluate_imagination(model, trials, n_context_states=20, n_predict_states=20, device="cpu"):
    """Evaluate rollout quality: predict future and compare to actual."""
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate BrainWM v2")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        choices=["oddball", "imagination", "linear_probe"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/share/data_makchen/peng/models/brainwm/results")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = BrainWMConfig()
    model = BrainWM(config)
    model.initialize_electrodes(ckpt["electrode_names"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "oddball":
        data_path = Path(args.data_dir)
        standard = torch.from_numpy(np.load(data_path / "standard.npy"))
        deviant = torch.from_numpy(np.load(data_path / "deviant.npy"))
        results = oddball_analysis(model, standard, deviant, device)

        print(f"\n=== Oddball Results (100ms resolution) ===")
        print(f"Time axis: {results['time_ms']}")
        print(f"PE difference (dev-std): {results['pe_difference']}")
        if "mmn_frontal_diff" in results:
            print(f"MMN (~200ms) frontal PE diff: {results['mmn_frontal_diff']:.4f}")
        if "p300_parietal_diff" in results:
            print(f"P300 (~400ms) parietal PE diff: {results['p300_parietal_diff']:.4f}")

        np.savez(output_dir / "oddball_results.npz", **{
            k: v for k, v in results.items() if isinstance(v, np.ndarray)
        })

    elif args.task == "imagination":
        trials = torch.from_numpy(np.load(Path(args.data_dir) / "trials.npy"))
        results = evaluate_imagination(model, trials, device=str(device))
        with open(output_dir / "imagination.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Cosine per step:", results["cosine_per_step"])

    print(f"Results → {output_dir}")


if __name__ == "__main__":
    main()
