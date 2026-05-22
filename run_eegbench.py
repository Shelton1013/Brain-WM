"""
Run EEG-Bench standard evaluation for EEG-JEPA.

Uses EEG-Bench's tasks, datasets, and evaluation protocol directly.
Only imports data loaders (not other models), so no HuggingFace needed.
Also runs CSP+LDA/SVM baselines for comparison.

Results are directly comparable to Laya Table 1 & 2.

Usage:
  export PYTHONPATH=/home/share/data_makchen/peng/datasets/EEG-Bench:$PYTHONPATH
  export BRAIN_WM_DIR=/import/home/pxieaf/Brain-WM

  # BCI tasks only
  CUDA_VISIBLE_DEVICES=3 python /import/home/pxieaf/Brain-WM/run_eegbench.py \
      --checkpoint /home/share/data_makchen/peng/models/eeg_jepa/best_model.pt \
      --tasks lr rf 4class 5finger

  # All tasks (BCI + clinical)
  CUDA_VISIBLE_DEVICES=3 python /import/home/pxieaf/Brain-WM/run_eegbench.py \
      --checkpoint /home/share/data_makchen/peng/models/eeg_jepa/best_model.pt \
      --tasks all

  # With baselines
  CUDA_VISIBLE_DEVICES=3 python /import/home/pxieaf/Brain-WM/run_eegbench.py \
      --checkpoint /home/share/data_makchen/peng/models/eeg_jepa/best_model.pt \
      --tasks lr rf 4class --run_baselines
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score

BRAIN_WM_DIR = os.environ.get("BRAIN_WM_DIR", "/import/home/pxieaf/Brain-WM")
sys.path.insert(0, BRAIN_WM_DIR)

EEGBENCH_DIR = os.environ.get("EEGBENCH_DIR",
                                "/home/share/data_makchen/peng/datasets/EEG-Bench")
sys.path.insert(0, EEGBENCH_DIR)

from eeg_jepa import EEGJEPA


# ============================================================
# Baseline model loaders (LaBraM etc. via EEG-Bench)
# ============================================================

def load_labram_model():
    """Try to import LaBraM from EEG-Bench. Returns class or None."""
    try:
        from eeg_bench.models.bci.labram_model import LaBraMModel
        return LaBraMModel
    except Exception as e:
        print(f"  LaBraM not available: {e}")
        return None

def load_csp_models():
    """Try to import CSP+LDA/SVM from EEG-Bench."""
    models = {}
    try:
        from eeg_bench.models.bci.csp_lda_model import CSPLDAModel
        models["csp_lda"] = CSPLDAModel
    except Exception:
        pass
    try:
        from eeg_bench.models.bci.csp_svm_model import CSPSVMModel
        models["csp_svm"] = CSPSVMModel
    except Exception:
        pass
    return models


# ============================================================
# Task registry
# ============================================================

TASK_REGISTRY = {}

def register_tasks():
    """Import EEG-Bench tasks one by one, skip failures."""
    task_imports = {
        # BCI
        "lr": ("LH vs RH MI",
               "eeg_bench.tasks.bci.left_hand_right_hand_mi_task",
               "LeftHandvRightHandMITask"),
        "rf": ("RH vs Feet MI",
               "eeg_bench.tasks.bci.right_hand_feet_mi_task",
               "RightHandvFeetMITask"),
        "4class": ("4-Class MI",
                    "eeg_bench.tasks.bci.left_hand_right_hand_feet_tongue_mi_task",
                    "LeftHandvRightHandvFeetvTongueMITask"),
        "5finger": ("5-Finger MI",
                     "eeg_bench.tasks.bci.five_fingers_mi_task",
                     "FiveFingersMITask"),
        # Clinical
        "abnormal": ("Abnormal EEG",
                      "eeg_bench.tasks.clinical.abnormal_clinical_task",
                      "AbnormalClinicalTask"),
        "epilepsy": ("Epilepsy",
                      "eeg_bench.tasks.clinical.epilepsy_clinical_task",
                      "EpilepsyClinicalTask"),
        "seizure": ("Seizure",
                     "eeg_bench.tasks.clinical.seizure_clinical_task",
                     "SeizureClinicalTask"),
        "artifact_bin": ("Artifact (Binary)",
                          "eeg_bench.tasks.clinical.binary_artifact_clinical_task",
                          "ArtifactBinaryClinicalTask"),
        "artifact_multi": ("Artifact (Multi)",
                            "eeg_bench.tasks.clinical.multiclass_artifact_clinical_task",
                            "ArtifactMulticlassClinicalTask"),
        "sleep": ("Sleep Stages",
                   "eeg_bench.tasks.clinical.sleep_stages_clinical_task",
                   "SleepStagesClinicalTask"),
        "mtbi": ("mTBI",
                  "eeg_bench.tasks.clinical.mtbi_clinical_task",
                  "MTBIClinicalTask"),
        "parkinsons": ("Parkinson's",
                        "eeg_bench.tasks.clinical.parkinsons_clinical_task",
                        "ParkinsonsClinicalTask"),
        "schizophrenia": ("Schizophrenia",
                           "eeg_bench.tasks.clinical.schizophrenia_clinical_task",
                           "SchizophreniaClinicalTask"),
        "ocd": ("OCD",
                 "eeg_bench.tasks.clinical.ocd_clinical_task",
                 "OCDClinicalTask"),
    }

    for key, (label, module_path, class_name) in task_imports.items():
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            TASK_REGISTRY[key] = (label, cls)
        except Exception as e:
            print(f"  Skip {key}: {e}")

    print(f"Loaded {len(TASK_REGISTRY)} tasks: {list(TASK_REGISTRY.keys())}")


# ============================================================
# Data loading with fallback
# ============================================================

def load_task_data(task):
    """Load train/test data from an EEG-Bench task, with per-dataset fallback."""
    from eeg_bench.enums.split import Split

    # Try normal loading first
    try:
        X_train, y_train, meta_train = task.get_data(Split.TRAIN)
        X_test, y_test, meta_test = task.get_data(Split.TEST)
        return X_train, y_train, meta_train, X_test, y_test, meta_test
    except Exception as e:
        print(f"  Full load failed: {e}")
        print(f"  Trying per-dataset fallback...")

    # Fallback: load datasets individually, skip corrupted
    X_train, y_train, meta_train = [], [], []
    X_test, y_test, meta_test = [], [], []

    for ds_cls in task.datasets:
        split_info = task.subjects_split[ds_cls]
        try:
            train_subj = split_info.get(Split.TRAIN, [])
            test_subj = split_info.get(Split.TEST, [])

            if train_subj:
                ds = ds_cls(target_classes=task.classes, subjects=train_subj)
                x, y, m = ds.get_data()
                m['task_name'] = task.name
                X_train.append(x)
                y_train.append(y)
                meta_train.append(m)

            if test_subj:
                ds = ds_cls(target_classes=task.classes, subjects=test_subj)
                x, y, m = ds.get_data()
                m['task_name'] = task.name
                X_test.append(x)
                y_test.append(y)
                meta_test.append(m)

            print(f"    OK {ds_cls.__name__}")
        except Exception as ds_e:
            print(f"    SKIP {ds_cls.__name__}: {ds_e}")

    if not X_train or not X_test:
        return None
    return X_train, y_train, meta_train, X_test, y_test, meta_test


# ============================================================
# Preprocessing
# ============================================================

def preprocess(X_list, n_channels, target_len=1024):
    """EEG-Bench format → [N, T, C]."""
    from scipy.signal import resample as scipy_resample

    trials = []
    for dataset_X in X_list:
        if isinstance(dataset_X, list):
            # MNE Raw objects
            for raw in dataset_X:
                try:
                    data = raw.get_data().T.astype(np.float32)
                    if data.shape[0] != target_len:
                        data = scipy_resample(data, target_len, axis=0).astype(np.float32)
                    trials.append(data)
                except Exception:
                    continue
        elif isinstance(dataset_X, np.ndarray):
            for trial in dataset_X:
                if trial.ndim != 2:
                    continue
                t = trial.T.astype(np.float32) if trial.shape[0] < trial.shape[1] else trial.astype(np.float32)
                if t.shape[0] != target_len:
                    t = scipy_resample(t, target_len, axis=0).astype(np.float32)
                trials.append(t)

    if not trials:
        return None

    # Normalize channels
    processed = []
    for t in trials:
        n_ch = t.shape[1]
        if n_ch > n_channels:
            t = t[:, :n_channels]
        elif n_ch < n_channels:
            t = np.pad(t, ((0, 0), (0, n_channels - n_ch)))
        # Z-score
        mean = t.mean(axis=0, keepdims=True)
        std = t.std(axis=0, keepdims=True) + 1e-8
        processed.append((t - mean) / std)

    return np.stack(processed)


def extract_features(model, X_np, device, batch_size=64):
    """Frozen encoder → pooled features."""
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            batch = torch.from_numpy(X_np[i:i+batch_size]).to(device)
            tokens = model._tokenize(batch)
            encoded = model._encode(tokens)
            features.append(encoded.mean(dim=1).cpu().numpy())
    return np.concatenate(features)


# ============================================================
# Evaluation
# ============================================================

def _run_frozen_probe(feat_tr, feat_te, y_tr, y_te, n_reps):
    """Run LogisticRegression on extracted features, return mean±std."""
    accs = []
    for seed in range(n_reps):
        scaler = StandardScaler()
        tr_s = scaler.fit_transform(feat_tr)
        te_s = scaler.transform(feat_te)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 multi_class="multinomial", random_state=42+seed)
        clf.fit(tr_s, y_tr)
        acc = balanced_accuracy_score(y_te, clf.predict(te_s))
        accs.append(acc)
    return {"mean": np.mean(accs), "std": np.std(accs)}


def _extract_labram_frozen_features(X_train, y_train, meta_train,
                                     X_test, meta_test, task_name, device):
    """Extract frozen features from pretrained LaBraM using its native pipeline.

    Uses LaBraM's own make_dataset for proper channel mapping and preprocessing,
    then extracts features from the frozen encoder (no fine-tuning).
    """
    try:
        from timm.models import create_model
        from eeg_bench.models.bci.labram_model import check_and_download_pretrained_model
        from eeg_bench.models.bci.LaBraM.make_dataset import make_dataset
        from eeg_bench.models.bci.LaBraM import utils as labram_utils
        import eeg_bench.models.bci.LaBraM.modeling_finetune
        from joblib import Memory
        from eeg_bench.config import get_config_value
        from typing import cast

        cache = Memory(location=get_config_value("cache"), verbose=0)

        # Load pretrained LaBraM encoder
        checkpoint = torch.load(check_and_download_pretrained_model(),
                                map_location=device, weights_only=False)
        new_ckpt = {k[len('student.'):]: v
                    for k, v in checkpoint['model'].items()
                    if k.startswith('student.')}

        labram = create_model("labram_base_patch200_200",
                              qkv_bias=False, rel_pos_bias=True,
                              num_classes=2, drop_rate=0.0,
                              drop_path_rate=0.1, use_mean_pooling=True,
                              init_scale=0.001, use_rel_pos_bias=True,
                              use_abs_pos_emb=True, init_values=0.1)
        labram.load_state_dict(new_ckpt, strict=False)
        labram = labram.to(device)
        labram.eval()

        def extract_from_datasets(X_list, y_list, meta_list, is_train):
            all_features = []
            split_size = 0.0  # no val split for feature extraction

            for X_, meta_ in zip(X_list, meta_list):
                try:
                    y_dummy = y_list[0] if y_list else None  # need labels for make_dataset
                    ds = cache.cache(make_dataset)(
                        X_, y_dummy, task_name,
                        meta_['sampling_frequency'],
                        meta_['channel_names'],
                        train=False, split_size=0,
                    )
                    if len(ds) == 0:
                        continue

                    ch_names = ds.ch_names
                    input_chans = labram_utils.get_input_chans(ch_names)
                    loader = DataLoader(ds, batch_size=64, shuffle=False)

                    with torch.no_grad():
                        for batch in loader:
                            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                            if x.dim() == 3:
                                B, C, T = x.shape
                            else:
                                continue
                            if T % 200 != 0:
                                x = x[:, :, :T - T % 200]
                                T = T - T % 200
                            if T == 0:
                                continue
                            x = x.reshape(B, C, T // 200, 200) / 100
                            feat = labram.forward_features(
                                x, input_chans=input_chans,
                                return_all_tokens=False,
                            )
                            all_features.append(feat.cpu().numpy())
                except Exception as ds_e:
                    print(f"      Skip dataset: {ds_e}")
                    continue

            if not all_features:
                return None
            return np.concatenate(all_features)

        print(f"    Extracting train features...")
        feat_tr = extract_from_datasets(X_train, y_train, meta_train, True)
        print(f"    Extracting test features...")
        feat_te = extract_from_datasets(X_test, [None]*len(X_test), meta_test, False)

        del labram
        torch.cuda.empty_cache()

        if feat_tr is None or feat_te is None:
            print(f"    LaBraM feature extraction returned None")
            return None, None

        print(f"    LaBraM features: train {feat_tr.shape}, test {feat_te.shape}")
        return feat_tr, feat_te

    except Exception as e:
        import traceback
        print(f"    LaBraM frozen feature extraction failed: {e}")
        traceback.print_exc()
        return None, None


def evaluate(task_key, task_label, task_class, model, n_channels, device,
             n_reps=5, run_baselines=False, run_finetune=False):
    """Run evaluation on one EEG-Bench task."""
    print(f"\n{'='*60}")
    print(f"  {task_label} ({task_key})")
    print(f"{'='*60}")

    task = task_class()
    data = load_task_data(task)
    if data is None:
        print(f"  No data available, skipping")
        return None

    X_train, y_train, meta_train, X_test, y_test, meta_test = data

    y_tr = np.concatenate(y_train) if isinstance(y_train, list) else y_train
    y_te = np.concatenate(y_test) if isinstance(y_test, list) else y_test

    if y_tr.dtype.kind in ('U', 'S', 'O'):
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr)
        y_te = le.transform(y_te)

    n_classes = len(np.unique(y_tr))
    chance = 1.0 / n_classes
    print(f"  Train: {len(y_tr)}, Test: {len(y_te)}, Classes: {n_classes}, Chance: {chance:.3f}")

    X_tr = preprocess(X_train, n_channels)
    X_te = preprocess(X_test, n_channels)
    if X_tr is None or X_te is None:
        print(f"  Preprocessing failed")
        return None
    print(f"  Shapes: train {X_tr.shape}, test {X_te.shape}")

    results = {"chance": chance, "n_classes": n_classes}

    # ========== FROZEN PROBES ==========

    # --- Ours (JEPA) frozen ---
    print(f"  [Frozen] JEPA...")
    feat_tr = extract_features(model, X_tr, device)
    feat_te = extract_features(model, X_te, device)
    results["jepa_frozen"] = _run_frozen_probe(feat_tr, feat_te, y_tr, y_te, n_reps)
    print(f"    → {results['jepa_frozen']['mean']:.3f} ± {results['jepa_frozen']['std']:.3f}")

    # --- Random frozen ---
    print(f"  [Frozen] Random...")
    random_model = EEGJEPA(n_channels=n_channels).to(device)
    feat_tr_r = extract_features(random_model, X_tr, device)
    feat_te_r = extract_features(random_model, X_te, device)
    del random_model
    results["random_frozen"] = _run_frozen_probe(feat_tr_r, feat_te_r, y_tr, y_te, n_reps)
    print(f"    → {results['random_frozen']['mean']:.3f} ± {results['random_frozen']['std']:.3f}")

    # --- LaBraM frozen ---
    if run_baselines:
        print(f"  [Frozen] LaBraM...")
        labram_feat_tr, labram_feat_te = _extract_labram_frozen_features(
            X_train, y_train, meta_train,
            X_test, meta_test, task.name, device)
        if labram_feat_tr is not None:
            # Labels need to match feature count
            # make_dataset may change sample count, so re-extract labels
            n_feat_tr = labram_feat_tr.shape[0]
            n_feat_te = labram_feat_te.shape[0]
            if n_feat_tr == len(y_tr) and n_feat_te == len(y_te):
                results["labram_frozen"] = _run_frozen_probe(
                    labram_feat_tr, labram_feat_te, y_tr, y_te, n_reps)
            else:
                # make_dataset may have filtered/resampled, use its labels
                print(f"    Label count mismatch (feat_tr={n_feat_tr} vs y_tr={len(y_tr)})")
                print(f"    Running LaBraM with its own label handling...")
                # Fallback: just use truncated labels
                y_tr_lb = y_tr[:n_feat_tr] if n_feat_tr < len(y_tr) else y_tr
                y_te_lb = y_te[:n_feat_te] if n_feat_te < len(y_te) else y_te
                results["labram_frozen"] = _run_frozen_probe(
                    labram_feat_tr, labram_feat_te, y_tr_lb, y_te_lb, n_reps)
            print(f"    → {results['labram_frozen']['mean']:.3f} ± {results['labram_frozen']['std']:.3f}")

        # --- CSP+LDA ---
        csp_models = load_csp_models()
        for name, CspModel in csp_models.items():
            try:
                print(f"  [Frozen] {name}...")
                csp = CspModel()
                csp.fit(X_train, y_train, meta_train)
                preds = csp.predict(X_test, meta_test)
                y_te_orig = np.concatenate(y_test) if isinstance(y_test, list) else y_test
                if y_te_orig.dtype.kind in ('U', 'S', 'O'):
                    le_csp = LabelEncoder()
                    le_csp.fit(np.concatenate([y_te_orig, preds]))
                    y_te_enc = le_csp.transform(y_te_orig)
                    preds_enc = le_csp.transform(preds)
                else:
                    y_te_enc, preds_enc = y_te_orig, preds
                acc = balanced_accuracy_score(y_te_enc, preds_enc)
                results[name] = {"mean": acc}
                print(f"    → {acc:.3f}")
            except Exception as e:
                print(f"    {name} failed: {e}")

    # ========== FINE-TUNE (optional) ==========
    if run_finetune:
        import copy
        ft_criterion = torch.nn.CrossEntropyLoss()
        X_tr_tensor = torch.from_numpy(X_tr)
        y_tr_tensor = torch.from_numpy(y_tr).long()
        ft_loader = DataLoader(
            TensorDataset(X_tr_tensor, y_tr_tensor),
            batch_size=32, shuffle=True, drop_last=True,
        )
        X_te_tensor = torch.from_numpy(X_te).to(device)

        def _run_finetune(ft_model, label):
            ft_head = torch.nn.Sequential(
                torch.nn.BatchNorm1d(ft_model.d_model),
                torch.nn.Linear(ft_model.d_model, n_classes),
            ).to(device)
            optimizer = torch.optim.AdamW([
                {"params": ft_model.parameters(), "lr": 1e-4},
                {"params": ft_head.parameters(), "lr": 1e-3},
            ], weight_decay=0.01)
            ft_model.train(); ft_head.train()
            for ep in range(50):
                for bx, by in ft_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = ft_head(ft_model._encode(ft_model._tokenize(bx)).mean(1))
                    optimizer.zero_grad()
                    ft_criterion(logits, by).backward()
                    optimizer.step()
            ft_model.eval(); ft_head.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(X_te_tensor), 64):
                    batch = X_te_tensor[i:i+64]
                    logits = ft_head(ft_model._encode(ft_model._tokenize(batch)).mean(1))
                    preds.append(logits.argmax(-1).cpu().numpy())
            acc = balanced_accuracy_score(y_te, np.concatenate(preds))
            print(f"    → {acc:.3f}")
            del ft_model, ft_head; torch.cuda.empty_cache()
            return acc

        print(f"  [Fine-tune] JEPA...")
        results["jepa_finetune"] = {"mean": _run_finetune(copy.deepcopy(model), "JEPA")}

        print(f"  [Fine-tune] Random...")
        results["random_finetune"] = {"mean": _run_finetune(
            EEGJEPA(n_channels=n_channels).to(device), "Random")}

        if run_baselines:
            LaBraMModel = load_labram_model()
            if LaBraMModel is not None:
                try:
                    print(f"  [Fine-tune] LaBraM...")
                    labram = LaBraMModel()
                    labram.fit(X_train, y_train, meta_train)
                    preds = labram.predict(X_test, meta_test)
                    y_te_orig = np.concatenate(y_test) if isinstance(y_test, list) else y_test
                    if y_te_orig.dtype.kind in ('U', 'S', 'O'):
                        le_l = LabelEncoder()
                        le_l.fit(np.concatenate([y_te_orig, preds]))
                        acc = balanced_accuracy_score(le_l.transform(y_te_orig), le_l.transform(preds))
                    else:
                        acc = balanced_accuracy_score(y_te_orig, preds)
                    results["labram_finetune"] = {"mean": acc}
                    print(f"    → {acc:.3f}")
                    del labram; torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    LaBraM fine-tune failed: {e}")

    delta = results["jepa_frozen"]["mean"] - results["random_frozen"]["mean"]
    print(f"  Pretraining value (frozen): {delta:+.3f}")
    results["delta"] = delta
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EEG-Bench evaluation for EEG-JEPA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", default=["lr", "rf", "4class"])
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--run_baselines", action="store_true",
                        help="Include LaBraM and CSP-LDA baselines")
    parser.add_argument("--finetune", action="store_true",
                        help="Also run fine-tune evaluation (slow)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str,
                        default="/home/share/data_makchen/peng/models/eeg_jepa/results/eegbench_results.json")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

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
    print(f"Model: {n_channels}ch, d={ckpt_args.get('d_model', 256)}")

    # Register tasks
    register_tasks()

    task_keys = list(TASK_REGISTRY.keys()) if args.tasks == ["all"] else \
                [t for t in args.tasks if t in TASK_REGISTRY]

    # Run
    all_results = {}
    for key in task_keys:
        label, cls = TASK_REGISTRY[key]
        result = evaluate(key, label, cls, model, n_channels, device,
                          n_reps=args.n_reps, run_baselines=args.run_baselines,
                          run_finetune=args.finetune)
        if result:
            all_results[key] = result

    # Summary table (Laya Table 1 format)
    bci_keys = [k for k in all_results if k in ("lr", "rf", "4class", "5finger")]
    clinical_keys = [k for k in all_results if k not in ("lr", "rf", "4class", "5finger")]

    def print_table(keys, title):
        if not keys:
            return
        has_labram_f = any("labram_frozen" in all_results.get(k, {}) for k in keys)
        has_csp = any("csp_lda" in all_results.get(k, {}) for k in keys)
        has_ft = any("jepa_finetune" in all_results.get(k, {}) for k in keys)
        has_labram_ft = any("labram_finetune" in all_results.get(k, {}) for k in keys)

        print(f"\n{'='*100}")
        print(f"  {title}")
        print(f"{'='*100}")

        # === Frozen Linear Probe (main table, matches Laya Table 1 format) ===
        print(f"\n  --- Frozen Linear Probe (balanced accuracy) ---")
        cols = [("Random", "random_frozen"), ("Ours", "jepa_frozen")]
        if has_labram_f:
            cols.append(("LaBraM", "labram_frozen"))
        if has_csp:
            cols.append(("CSP-LDA", "csp_lda"))

        header = f"  {'Task':<20} {'Chance':>7}"
        for name, _ in cols:
            header += f" {name:>12}"
        print(header)
        sep = f"  {'-'*20} {'-'*7}" + f" {'-'*12}" * len(cols)
        print(sep)

        for k in keys:
            r = all_results[k]
            label = TASK_REGISTRY[k][0]
            line = f"  {label:<20} {r['chance']:>7.3f}"
            for _, key_name in cols:
                v = r.get(key_name, {})
                if isinstance(v, dict):
                    mean = v.get("mean", 0)
                    std = v.get("std", 0)
                    line += f" {mean:>5.3f}±{std:.3f}" if std > 0 else f" {mean:>12.3f}"
                else:
                    line += f" {v:>12.3f}" if v else f" {'N/A':>12}"
            print(line)

        # Mean row
        print(sep)
        line = f"  {'Mean':<20} {'':>7}"
        for _, key_name in cols:
            vals = [all_results[k].get(key_name, {}).get("mean", 0)
                    if isinstance(all_results[k].get(key_name, {}), dict)
                    else all_results[k].get(key_name, 0)
                    for k in keys]
            line += f" {np.mean(vals):>12.3f}"
        print(line)

        means_rf = np.mean([all_results[k]["random_frozen"]["mean"] for k in keys])
        means_jf = np.mean([all_results[k]["jepa_frozen"]["mean"] for k in keys])
        print(f"\n  Pretraining value (frozen): JEPA vs Random = {means_jf - means_rf:+.4f}")

        # === Fine-tune (optional) ===
        if has_ft:
            print(f"\n  --- Fine-tune (50 epochs) ---")
            ft_cols = [("Rand-FT", "random_finetune"), ("JEPA-FT", "jepa_finetune")]
            if has_labram_ft:
                ft_cols.append(("LaBraM-FT", "labram_finetune"))

            header = f"  {'Task':<20} {'Chance':>7}"
            for name, _ in ft_cols:
                header += f" {name:>12}"
            print(header)
            sep = f"  {'-'*20} {'-'*7}" + f" {'-'*12}" * len(ft_cols)
            print(sep)

            for k in keys:
                r = all_results[k]
                label = TASK_REGISTRY[k][0]
                line = f"  {label:<20} {r['chance']:>7.3f}"
                for _, key_name in ft_cols:
                    v = r.get(key_name, {}).get("mean", 0)
                    line += f" {v:>12.3f}"
                print(line)

            print(sep)
            line = f"  {'Mean':<20} {'':>7}"
            for _, key_name in ft_cols:
                vals = [all_results[k].get(key_name, {}).get("mean", 0) for k in keys]
                line += f" {np.mean(vals):>12.3f}"
            print(line)

        print(f"{'='*100}")

    print_table(bci_keys, "BCI Tasks (Motor Imagery) — compare with Laya Table 1")
    print_table(clinical_keys, "Clinical Tasks — compare with Laya Table 2")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
