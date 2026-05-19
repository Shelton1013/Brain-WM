"""
Evaluate EEG-JEPA using EEG-Bench tasks directly.

Bypasses benchmark_console.py to avoid HuggingFace downloads.
Only imports EEG-Bench's task/dataset loaders, not other models.

Usage:
  cd /home/share/data_makchen/peng/EEG-Bench

  # Run all MI tasks
  python /import/home/pxieaf/Brain-WM/eval_eegbench.py \
      --checkpoint /home/share/data_makchen/peng/models/eeg_jepa/best_model.pt \
      --tasks lr rf 4class

  # Run all available tasks
  python /import/home/pxieaf/Brain-WM/eval_eegbench.py \
      --checkpoint /home/share/data_makchen/peng/models/eeg_jepa/best_model.pt \
      --tasks all
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

# Add Brain-WM to path
BRAIN_WM_DIR = os.environ.get("BRAIN_WM_DIR", "/import/home/pxieaf/Brain-WM")
sys.path.insert(0, BRAIN_WM_DIR)

# Add EEG-Bench to path
EEGBENCH_DIR = os.environ.get("EEGBENCH_DIR", "/home/share/data_makchen/peng/datasets/EEG-Bench")
sys.path.insert(0, EEGBENCH_DIR)

from eeg_jepa import EEGJEPA


# ============================================================
# Task loading (import only data loaders, not models)
# ============================================================

def load_tasks():
    """Import EEG-Bench tasks without importing models."""
    tasks = {}

    # BCI tasks
    try:
        from eeg_bench.tasks.bci.left_hand_right_hand_mi_task import LeftHandvRightHandMITask
        tasks["lr"] = ("LH vs RH MI", LeftHandvRightHandMITask)
    except Exception as e:
        print(f"  Skip lr: {e}")
    try:
        from eeg_bench.tasks.bci.right_hand_feet_mi_task import RightHandvFeetMITask
        tasks["rf"] = ("RH vs Feet MI", RightHandvFeetMITask)
    except Exception as e:
        print(f"  Skip rf: {e}")
    try:
        from eeg_bench.tasks.bci.left_hand_right_hand_feet_tongue_mi_task import LeftHandvRightHandvFeetvTongueMITask
        tasks["4class"] = ("4-Class MI", LeftHandvRightHandvFeetvTongueMITask)
    except Exception as e:
        print(f"  Skip 4class: {e}")
    try:
        from eeg_bench.tasks.bci.five_fingers_mi_task import FiveFingersMITask
        tasks["5finger"] = ("5-Finger MI", FiveFingersMITask)
    except Exception as e:
        print(f"  Skip 5finger: {e}")

    # Clinical tasks
    try:
        from eeg_bench.tasks.clinical.abnormal_clinical_task import AbnormalClinicalTask
        tasks["abnormal"] = ("Abnormal EEG", AbnormalClinicalTask)
    except Exception as e:
        print(f"  Skip abnormal: {e}")
    try:
        from eeg_bench.tasks.clinical.epilepsy_clinical_task import EpilepsyClinicalTask
        tasks["epilepsy"] = ("Epilepsy", EpilepsyClinicalTask)
    except Exception as e:
        print(f"  Skip epilepsy: {e}")
    try:
        from eeg_bench.tasks.clinical.seizure_clinical_task import SeizureClinicalTask
        tasks["seizure"] = ("Seizure", SeizureClinicalTask)
    except Exception as e:
        print(f"  Skip seizure: {e}")
    try:
        from eeg_bench.tasks.clinical.binary_artifact_clinical_task import ArtifactBinaryClinicalTask
        tasks["artifact"] = ("Artifact (Binary)", ArtifactBinaryClinicalTask)
    except Exception as e:
        print(f"  Skip artifact: {e}")
    try:
        from eeg_bench.tasks.clinical.sleep_stages_clinical_task import SleepStagesClinicalTask
        tasks["sleep"] = ("Sleep Stages", SleepStagesClinicalTask)
    except Exception as e:
        print(f"  Skip sleep: {e}")
    try:
        from eeg_bench.tasks.clinical.mtbi_clinical_task import MTBIClinicalTask
        tasks["mtbi"] = ("mTBI", MTBIClinicalTask)
    except Exception as e:
        print(f"  Skip mtbi: {e}")
    try:
        from eeg_bench.tasks.clinical.parkinsons_clinical_task import ParkinsonsClinicalTask
        tasks["parkinsons"] = ("Parkinson's", ParkinsonsClinicalTask)
    except Exception as e:
        print(f"  Skip parkinsons: {e}")
    try:
        from eeg_bench.tasks.clinical.schizophrenia_clinical_task import SchizophreniaClinicalTask
        tasks["schizophrenia"] = ("Schizophrenia", SchizophreniaClinicalTask)
    except Exception as e:
        print(f"  Skip schizophrenia: {e}")
    try:
        from eeg_bench.tasks.clinical.ocd_clinical_task import OCDClinicalTask
        tasks["ocd"] = ("OCD", OCDClinicalTask)
    except Exception as e:
        print(f"  Skip ocd: {e}")

    return tasks


# ============================================================
# Feature extraction
# ============================================================

def preprocess_eegbench_data(X_list, n_channels, target_len=1024):
    """Convert EEG-Bench format to our [B, T, C] format.

    EEG-Bench provides X as a list (one per dataset), each [n_samples, C, T].
    """
    from scipy.signal import resample as scipy_resample

    all_trials = []
    for dataset_X in X_list:
        if isinstance(dataset_X, list):
            # Raw MNE objects — extract data
            for raw in dataset_X:
                try:
                    data = raw.get_data().T.astype(np.float32)  # [T, C]
                    if data.shape[0] != target_len:
                        data = scipy_resample(data, target_len, axis=0).astype(np.float32)
                    all_trials.append(data)
                except Exception:
                    continue
        else:
            # Numpy array: [n_samples, C, T] or [n_samples, T, C]
            for trial in dataset_X:
                if trial.ndim == 2:
                    # [C, T] → [T, C]
                    if trial.shape[0] < trial.shape[1]:
                        trial = trial.T
                    t = trial.astype(np.float32)
                else:
                    continue

                # Resample to target_len
                if t.shape[0] != target_len:
                    t = scipy_resample(t, target_len, axis=0).astype(np.float32)

                # Channel handling
                n_ch = t.shape[1]
                if n_ch > n_channels:
                    t = t[:, :n_channels]
                elif n_ch < n_channels:
                    pad = np.zeros((target_len, n_channels - n_ch), dtype=np.float32)
                    t = np.concatenate([t, pad], axis=1)

                # Z-score
                mean = t.mean(axis=0, keepdims=True)
                std = t.std(axis=0, keepdims=True) + 1e-8
                t = (t - mean) / std
                all_trials.append(t)

    if not all_trials:
        return None
    return np.stack(all_trials)


def extract_features(model, X_np, device, batch_size=64):
    """Extract frozen encoder features from preprocessed EEG."""
    model.eval()
    X_tensor = torch.from_numpy(X_np)
    features = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            tokens = model._tokenize(batch)
            encoded = model._encode(tokens)
            pooled = encoded.mean(dim=1)
            features.append(pooled.cpu().numpy())

    return np.concatenate(features)


# ============================================================
# Evaluation
# ============================================================

def evaluate_task(task_name, task_label, task_class, model, n_channels, device,
                  n_reps=5):
    """Run linear probe evaluation on one EEG-Bench task."""
    from enum import Enum

    print(f"\n{'='*60}")
    print(f"  Task: {task_label} ({task_name})")
    print(f"{'='*60}")

    try:
        task = task_class()
    except Exception as e:
        print(f"  Failed to create task: {e}")
        return None

    # Get train/test split
    try:
        from eeg_bench.enums.split import Split
        X_train, y_train, meta_train = task.get_data(Split.TRAIN)
        X_test, y_test, meta_test = task.get_data(Split.TEST)
    except Exception as e:
        print(f"  Failed to load data: {e}")
        return None

    # Concatenate labels
    y_train_cat = np.concatenate(y_train) if isinstance(y_train, list) else y_train
    y_test_cat = np.concatenate(y_test) if isinstance(y_test, list) else y_test

    # Handle string labels
    if y_train_cat.dtype.kind in ('U', 'S', 'O'):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_cat = le.fit_transform(y_train_cat)
        y_test_cat = le.transform(y_test_cat)

    n_classes = len(np.unique(y_train_cat))
    print(f"  Train: {len(y_train_cat)} samples, Test: {len(y_test_cat)} samples, "
          f"Classes: {n_classes}")

    # Preprocess
    X_train_np = preprocess_eegbench_data(X_train, n_channels)
    X_test_np = preprocess_eegbench_data(X_test, n_channels)

    if X_train_np is None or X_test_np is None:
        print(f"  Failed to preprocess data")
        return None

    print(f"  Preprocessed: train {X_train_np.shape}, test {X_test_np.shape}")

    # Extract features
    train_features = extract_features(model, X_train_np, device)
    test_features = extract_features(model, X_test_np, device)

    # Run multiple reps with different seeds
    accs = []
    for rep in range(n_reps):
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)

        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            multi_class="multinomial", random_state=42 + rep,
        )
        clf.fit(train_scaled, y_train_cat)
        preds = clf.predict(test_scaled)
        acc = balanced_accuracy_score(y_test_cat, preds)
        accs.append(acc)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"  Result: {mean_acc:.3f} ± {std_acc:.3f} (balanced accuracy, {n_reps} reps)")
    return {"mean": mean_acc, "std": std_acc, "reps": accs}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate EEG-JEPA on EEG-Bench tasks")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", default=["lr", "rf", "4class"],
                        help="Tasks to evaluate. Use 'all' for all available.")
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    n_channels = 64
    for key, val in ckpt["model_state_dict"].items():
        if "channel_embed" in key:
            n_channels = val.shape[0]
            break

    model = EEGJEPA(
        n_channels=n_channels,
        d_model=ckpt_args.get("d_model", 256),
        encoder_layers=ckpt_args.get("encoder_layers", 6),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model loaded: {n_channels}ch, d_model={ckpt_args.get('d_model', 256)}")

    # Load available tasks
    available_tasks = load_tasks()
    print(f"Available tasks: {list(available_tasks.keys())}")

    # Select tasks
    if args.tasks == ["all"]:
        task_keys = list(available_tasks.keys())
    else:
        task_keys = [t for t in args.tasks if t in available_tasks]

    # Run evaluation
    all_results = {}
    for task_key in task_keys:
        task_label, task_class = available_tasks[task_key]
        result = evaluate_task(
            task_key, task_label, task_class, model, n_channels, device,
            n_reps=args.n_reps,
        )
        if result:
            all_results[task_key] = result

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY (EEG-Bench, frozen linear probe)")
    print(f"{'='*60}")
    print(f"  {'Task':<20} {'Accuracy':>12}")
    print(f"  {'-'*20} {'-'*12}")
    for task_key, result in all_results.items():
        task_label = available_tasks[task_key][0]
        print(f"  {task_label:<20} {result['mean']:.3f}±{result['std']:.3f}")
    if all_results:
        mean_all = np.mean([r["mean"] for r in all_results.values()])
        print(f"  {'-'*20} {'-'*12}")
        print(f"  {'Mean':<20} {mean_all:.3f}")
    print(f"{'='*60}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
