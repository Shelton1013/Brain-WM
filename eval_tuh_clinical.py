"""TUAB / TUEV downstream evaluation for our EEG-LeJEPA family.

Mirrors run_eegbench.py structure (model loading, frozen probe via
LogisticRegression on pooled encoder features, fine-tune with BN+Linear
head + OneCycleLR + class-weighted CE + early stopping on val BA), but
runs on TUAB or TUEV instead of EEG-Bench BCI tasks.

Frozen-probe mode is unique to us (LaBraM does not report it on
TUAB/TUEV); fine-tune mode mirrors LaBraM's protocol for direct
comparison to LaBraM Table 4 (TUAB) and Table 5 (TUEV).

Metrics:
    TUAB (binary):  Balanced Accuracy, ROC-AUC, PR-AUC
    TUEV (6-class): Balanced Accuracy, Cohen's Kappa, weighted F1

Usage:
    # Frozen probe + fine-tune on TUAB
    CUDA_VISIBLE_DEVICES=0 python eval_tuh_clinical.py \\
        --dataset tuab \\
        --checkpoint /home/pxieaf/home2/model/eeg_lejepa_outputcf_sigreg_l01_w1/best_model.pt \\
        --tuh_dir /home/pxieaf/home2/tuh/tuh_eeg_abnormal/v3.0.1/edf \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --output /home/pxieaf/home2/eval/outputcf_sigreg_l01_tuab.json

    # Same for TUEV
    CUDA_VISIBLE_DEVICES=0 python eval_tuh_clinical.py \\
        --dataset tuev \\
        --checkpoint <ckpt> \\
        --tuh_dir /home/pxieaf/home2/tuh/tuh_eeg_events/v2.0.1/edf \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --output /home/pxieaf/home2/eval/outputcf_sigreg_l01_tuev.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================
# DDP helpers
# ============================================================

def _ddp_active():
    return dist.is_available() and dist.is_initialized()

def _ddp_rank():
    return dist.get_rank() if _ddp_active() else 0

def _ddp_world():
    return dist.get_world_size() if _ddp_active() else 1

def _is_main():
    return _ddp_rank() == 0

def _rank0_print(*args, **kwargs):
    if _is_main():
        print(*args, **kwargs)

def _ddp_barrier():
    if _ddp_active():
        dist.barrier()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

BRAIN_WM_DIR = os.environ.get("BRAIN_WM_DIR", str(Path(__file__).parent))
sys.path.insert(0, BRAIN_WM_DIR)

from eeg_jepa import EEGJEPA
from eeg_mae import EEGMAE
from eeg_lejepa import EEGLeJEPA
from eeg_lejepa_spectral import EEGLeJEPASpectral
from eeg_lejepa_region import EEGLeJEPARegion
from eeg_lejepa_full import EEGLeJEPAFull
from eeg_lejepa_crossfreq import EEGLeJEPACrossFreq
from eeg_lejepa_multistream import EEGLeJEPAMultiStream
from eeg_lejepa_outputcf import EEGLeJEPAOutputCF
from eeg_lejepa_outputcf_pajr import EEGLeJEPAOutputCFPAJR
from eeg_lejepa_v2 import EEGLeJEPA_v2

from dataset_tuh_clinical import TUABDataset, TUEVDataset, TUEV_LABEL_NAMES


# ============================================================
# Model loading (mirrors run_eegbench.py logic)
# ============================================================

TYPE_MAP = {
    "mae":                (EEGMAE,                "EEG-MAE"),
    "lejepa_full":        (EEGLeJEPAFull,         "EEG-LeJEPA+Full"),
    "lejepa_crossfreq":   (EEGLeJEPACrossFreq,    "EEG-LeJEPA+CrossFreq"),
    "lejepa_multistream": (EEGLeJEPAMultiStream,  "EEG-LeJEPA+MultiStream"),
    "lejepa_outputcf":    (EEGLeJEPAOutputCF,     "EEG-LeJEPA+OutputCF"),
    "lejepa_outputcf_pajr": (EEGLeJEPAOutputCFPAJR, "EEG-LeJEPA+OutputCF+PAJR"),
    "lejepa_v2":          (EEGLeJEPA_v2,          "EEG-LeJEPA v2"),
    "lejepa_spectral":    (EEGLeJEPASpectral,     "EEG-LeJEPA+Spectral"),
    "lejepa_region":      (EEGLeJEPARegion,       "EEG-LeJEPA+Region"),
    "lejepa":             (EEGLeJEPA,             "EEG-LeJEPA"),
    "jepa":               (EEGJEPA,               "EEG-JEPA"),
}


def load_pretrained(checkpoint_path: str, device: torch.device):
    """Returns (model, model_cls, model_type_name, n_channels, ckpt_args)."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Detect n_channels from any channel-embed-like weight
    n_channels = 64
    for key, val in ckpt["model_state_dict"].items():
        if "channel_embed" in key:
            n_channels = val.shape[0]
            break

    model_type = ckpt_args.get("model", "jepa")
    # Auto-detect v2 by state_dict signature (v2 has encoder_blocks, patch_embed)
    keys_str = str(ckpt["model_state_dict"].keys())
    if "encoder_blocks" in keys_str and "patch_embed" in keys_str:
        model_type = "lejepa_v2"
    if model_type in TYPE_MAP:
        model_cls, model_type_name = TYPE_MAP[model_type]
    else:
        # Fallback by inspecting keys (rare; old checkpoints)
        if "reconstruction_head" in keys_str:
            model_cls, model_type_name = EEGMAE, "EEG-MAE"
        elif "band_head" in keys_str:
            model_cls, model_type_name = EEGLeJEPAOutputCF, "EEG-LeJEPA+OutputCF"
        elif "freq_predictor" in keys_str:
            model_cls, model_type_name = EEGLeJEPACrossFreq, "EEG-LeJEPA+CrossFreq"
        else:
            model_cls, model_type_name = EEGLeJEPA, "EEG-LeJEPA"

    # v2 has a totally different constructor — handle separately
    if model_type == "lejepa_v2":
        model_kwargs = dict(
            n_channels=n_channels,
            patch_len=ckpt_args.get("patch_len", 200),
            d_model=ckpt_args.get("d_model", 512),
            d_decoder=ckpt_args.get("d_decoder", 256),
            encoder_layers=ckpt_args.get("encoder_layers", 12),
            decoder_layers=ckpt_args.get("decoder_layers", 2),
            mask_ratio=ckpt_args.get("mask_ratio", 0.5),
            jepa_weight=ckpt_args.get("jepa_weight", 1.0),
            mae_weight=ckpt_args.get("mae_weight", 0.5),
            cf_weight=ckpt_args.get("cf_weight", 0.3),
            sigreg_lambda=ckpt_args.get("sigreg_lambda", 0.05),
            pajr_weight=ckpt_args.get("pajr_weight", 0.0),
            reg_type=ckpt_args.get("reg_type", "sigreg"),
            d_band_view=ckpt_args.get("cf_d_band", 64),
            cf_band_conditioned=bool(ckpt_args.get("cf_band_conditioned", 1)),
            max_time_patches=ckpt_args.get("max_time_patches", 64),
            max_channels=ckpt_args.get("max_channels", 32),
            n_subjects=ckpt_args.get("n_subjects_actual",
                                     ckpt_args.get("n_subjects", 2000)),
        )
        model = model_cls(**model_kwargs).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"Model: {model_type_name}, {n_channels}ch, "
              f"d={ckpt_args.get('d_model', 512)}")
        return model, model_cls, model_type_name, n_channels, ckpt_args

    model_kwargs = dict(
        n_channels=n_channels,
        d_model=ckpt_args.get("d_model", 256),
        encoder_layers=ckpt_args.get("encoder_layers", 6),
    )
    if "n_queries" in ckpt_args:
        model_kwargs["n_queries"] = ckpt_args["n_queries"]
    # max_seq_len drives pos_embed shape — must match the checkpoint or
    # state_dict load fails with a shape mismatch on pos_embed.
    if "max_seq_len" in ckpt_args:
        model_kwargs["max_seq_len"] = ckpt_args["max_seq_len"]
    else:
        # Infer from pos_embed in state_dict (handles old ckpts without args)
        pe = ckpt["model_state_dict"].get("pos_embed")
        if pe is not None:
            model_kwargs["max_seq_len"] = int(pe.shape[1])
    if model_type in ("lejepa_crossfreq", "lejepa_full",
                      "lejepa_multistream", "lejepa_outputcf",
                      "lejepa_outputcf_pajr"):
        model_kwargs["cf_band_conditioned"] = bool(
            ckpt_args.get("cf_band_conditioned", 0))
        model_kwargs["cf_preserve_spatial"] = bool(
            ckpt_args.get("cf_preserve_spatial", 0))
        if "cf_d_band" in ckpt_args:
            model_kwargs["cf_d_band"] = ckpt_args["cf_d_band"]
    if model_type == "lejepa_outputcf_pajr":
        # n_patients is encoded in the checkpoint's discriminator weight shape
        n_patients = 4000  # fallback
        for k, v in ckpt["model_state_dict"].items():
            if k.endswith("patient_disc.mlp.4.weight"):  # last linear out_features
                n_patients = v.shape[0]
                break
        model_kwargs["n_patients"] = n_patients
        if "par_lambda" in ckpt_args:
            model_kwargs["par_lambda"] = ckpt_args["par_lambda"]
        if "par_weight" in ckpt_args:
            model_kwargs["par_weight"] = ckpt_args["par_weight"]
        if "par_disc_hidden" in ckpt_args:
            model_kwargs["par_disc_hidden"] = ckpt_args["par_disc_hidden"]

    model = model_cls(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model: {model_type_name}, {n_channels}ch, "
          f"d={ckpt_args.get('d_model', 256)}")
    return model, model_cls, model_type_name, n_channels, ckpt_args


def build_random_init(model_cls, n_channels, ckpt_args, device):
    """Build an UNTRAINED model with the same config for the random baseline."""
    model_type = ckpt_args.get("model", "jepa")

    # v2 uses different constructor
    if model_type == "lejepa_v2":
        model_kwargs = dict(
            n_channels=n_channels,
            patch_len=ckpt_args.get("patch_len", 200),
            d_model=ckpt_args.get("d_model", 512),
            d_decoder=ckpt_args.get("d_decoder", 256),
            encoder_layers=ckpt_args.get("encoder_layers", 12),
            decoder_layers=ckpt_args.get("decoder_layers", 2),
            mask_ratio=ckpt_args.get("mask_ratio", 0.5),
            jepa_weight=ckpt_args.get("jepa_weight", 1.0),
            mae_weight=ckpt_args.get("mae_weight", 0.5),
            cf_weight=ckpt_args.get("cf_weight", 0.3),
            sigreg_lambda=ckpt_args.get("sigreg_lambda", 0.05),
            pajr_weight=ckpt_args.get("pajr_weight", 0.0),
            reg_type=ckpt_args.get("reg_type", "sigreg"),
            d_band_view=ckpt_args.get("cf_d_band", 64),
            cf_band_conditioned=bool(ckpt_args.get("cf_band_conditioned", 1)),
            max_time_patches=ckpt_args.get("max_time_patches", 64),
            max_channels=ckpt_args.get("max_channels", 32),
            n_subjects=ckpt_args.get("n_subjects_actual",
                                     ckpt_args.get("n_subjects", 2000)),
        )
        return model_cls(**model_kwargs).to(device)

    model_kwargs = dict(
        n_channels=n_channels,
        d_model=ckpt_args.get("d_model", 256),
        encoder_layers=ckpt_args.get("encoder_layers", 6),
    )
    if "n_queries" in ckpt_args:
        model_kwargs["n_queries"] = ckpt_args["n_queries"]
    if "max_seq_len" in ckpt_args:
        model_kwargs["max_seq_len"] = ckpt_args["max_seq_len"]
    # CF args only valid for CF model types; base lejepa/jepa/mae reject them.
    if model_type in ("lejepa_crossfreq", "lejepa_full",
                      "lejepa_multistream", "lejepa_outputcf",
                      "lejepa_outputcf_pajr"):
        if "cf_band_conditioned" in ckpt_args:
            model_kwargs["cf_band_conditioned"] = bool(ckpt_args["cf_band_conditioned"])
        if "cf_preserve_spatial" in ckpt_args:
            model_kwargs["cf_preserve_spatial"] = bool(ckpt_args["cf_preserve_spatial"])
        if "cf_d_band" in ckpt_args:
            model_kwargs["cf_d_band"] = ckpt_args["cf_d_band"]
    return model_cls(**model_kwargs).to(device)


# ============================================================
# Feature extraction & channel adaptation
# ============================================================

def _pad_or_trim_channels(X: np.ndarray, target_n_ch: int) -> np.ndarray:
    """X: [N, T, C] OR [T, C] → matching shape with target_n_ch via zero-pad/trim."""
    n_ch = X.shape[-1]
    if n_ch == target_n_ch:
        return X
    if n_ch > target_n_ch:
        return X[..., :target_n_ch]
    pad_dims = [(0, 0)] * (X.ndim - 1) + [(0, target_n_ch - n_ch)]
    return np.pad(X, pad_dims)


class _TrialDataset(torch.utils.data.Dataset):
    """Streaming wrapper over ds.trials (list[np.ndarray]) + labels.

    Avoids the np.stack RAM doubling. For TUAB (372k trials × 2560 × 19 ×
    4 bytes = ~70 GB), np.stack + fancy indexing copy = ~200 GB peak;
    this wrapper stays at ~70 GB (just the trials list already in RAM).
    """
    def __init__(self, trials, labels, target_n_ch):
        self.trials = trials
        self.labels = labels
        self.target_n_ch = target_n_ch

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        x = self.trials[idx]
        if x.shape[-1] != self.target_n_ch:
            x = _pad_or_trim_channels(x, self.target_n_ch)
        # Ensure contiguous + float32; copy is cheap for one trial
        x = np.ascontiguousarray(x, dtype=np.float32)
        return torch.from_numpy(x), int(self.labels[idx])


def dataset_to_xy(ds, target_n_ch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Convert {trials, labels, recording_ids} dataset to (X, y, rec_ids).

    LEGACY path used by frozen probe; this DOES stack to a single array
    (RAM-heavy for TUAB-scale). Prefer dataset_to_streaming for FT.

    Returns:
        X: [N, T, C] float32 (single contiguous array)
        y: [N] int (per-trial labels)
        rec_ids: [N] int or None
    """
    X = np.stack(ds.trials).astype(np.float32)
    X = _pad_or_trim_channels(X, target_n_ch)
    y = np.array(ds.labels, dtype=np.int64)
    rec_ids = None
    if hasattr(ds, "recording_ids") and ds.recording_ids:
        rec_ids = np.array(ds.recording_ids, dtype=np.int64)
    return X, y, rec_ids


def dataset_to_streaming(ds, target_n_ch: int):
    """Build a streaming _TrialDataset + label/rec_id/patient_id arrays.

    Used by run_finetune to avoid RAM doubling on large datasets.

    Returns rec_ids AND patient_ids (patient_ids preferred for val split
    to match LaBraM/CBraMod/CSBrain subject-disjoint 80/20 protocol).
    """
    stream = _TrialDataset(ds.trials, ds.labels, target_n_ch)
    y = np.array(ds.labels, dtype=np.int64)
    rec_ids = None
    if hasattr(ds, "recording_ids") and ds.recording_ids:
        rec_ids = np.array(ds.recording_ids, dtype=np.int64)
    patient_ids = None
    if hasattr(ds, "patient_ids") and ds.patient_ids:
        # Keep as np string array (patient IDs are 8-letter strings)
        patient_ids = np.array(ds.patient_ids, dtype=object)
    return stream, y, rec_ids, patient_ids


def aggregate_per_recording(features, labels, rec_ids):
    """Mean-pool per-trial features within each recording.

    Args:
        features: [N_trials, d_model] per-trial embeddings
        labels:   [N_trials] per-trial labels (all trials in a recording share label)
        rec_ids:  [N_trials] recording index per trial

    Returns:
        agg_features: [N_recordings, d_model]
        agg_labels:   [N_recordings]
    """
    unique_recs = np.unique(rec_ids)
    agg_features = np.zeros((len(unique_recs), features.shape[1]), dtype=features.dtype)
    agg_labels = np.zeros(len(unique_recs), dtype=labels.dtype)
    for i, rec in enumerate(unique_recs):
        mask = rec_ids == rec
        agg_features[i] = features[mask].mean(axis=0)
        # All trials in a recording share the same label (per dataset construction)
        agg_labels[i] = labels[mask][0]
    return agg_features, agg_labels


def extract_features(model, X_np, device, batch_size=64):
    """Run encoder over X (mean-pool over tokens) → [N, d_model]."""
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            batch = torch.from_numpy(X_np[i:i+batch_size]).to(device)
            tokens = model._tokenize(batch)
            encoded = model._encode(tokens)
            feats.append(encoded.mean(dim=1).cpu().numpy())
    return np.concatenate(feats)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred, y_proba=None, dataset: str = "tuab"):
    """Returns dict of {metric_name: float}.

    y_pred:  class predictions [N]
    y_proba: class probabilities [N, n_classes] (only used by TUAB AUC)
    """
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if dataset == "tuab":
        if y_proba is not None:
            # Binary case: y_proba[:, 1] is prob of class 1 (abnormal)
            pos_prob = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, pos_prob))
                out["pr_auc"] = float(average_precision_score(y_true, pos_prob))
            except ValueError:
                out["roc_auc"] = float("nan")
                out["pr_auc"] = float("nan")
    elif dataset == "tuev":
        out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        out["weighted_f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return out


# ============================================================
# Frozen probe
# ============================================================

def run_frozen_probe(
    feat_tr, y_tr, feat_te, y_te, n_classes: int, dataset: str, n_reps: int,
) -> dict:
    """LogisticRegression × n_reps, returns mean metrics + per-rep list."""
    metrics_by_rep = []
    for seed in range(n_reps):
        scaler = StandardScaler()
        tr_s = scaler.fit_transform(feat_tr)
        te_s = scaler.transform(feat_te)
        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            multi_class="multinomial" if n_classes > 2 else "auto",
            random_state=42 + seed,
            class_weight="balanced",
        )
        clf.fit(tr_s, y_tr)
        preds = clf.predict(te_s)
        proba = clf.predict_proba(te_s)
        metrics_by_rep.append(
            compute_metrics(y_te, preds, proba, dataset=dataset))

    # Aggregate mean ± std for each metric
    keys = list(metrics_by_rep[0].keys())
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_by_rep]
        agg[k] = {"mean": float(np.mean(vals)),
                  "std": float(np.std(vals))}
    agg["_per_rep"] = metrics_by_rep
    return agg


# ============================================================
# Fine-tune (matches run_eegbench.py protocol)
# ============================================================

def _build_labram_optimizer(model, head, base_lr: float, weight_decay: float,
                             layer_decay: float = 0.65,
                             head_lr_mult: float = 1.0):
    """LaBraM-style optimizer with layer-wise LR decay.

    head_lr_mult (default 1.0 matches LaBraM). Setting 10 gives the head
    10× higher LR than backbone base_lr — closer to run_eegbench's
    [4e-4 backbone, 4e-3 head] split, useful when head_lr = base_lr
    leaves the classifier undertrained.
    """
    param_groups = []
    # v2 uses `encoder_blocks` + `patch_embed`; v1 uses `encoder` + `tokenizer`
    is_v2 = hasattr(model, "encoder_blocks")
    encoder_module = model.encoder_blocks if is_v2 else model.encoder
    n_layers = len(encoder_module)

    # Deepest: tokenizer/patch_embed + pos embeds
    deepest = []
    if is_v2:
        deepest.extend(model.patch_embed.parameters())
        deepest.append(model.pos_time)
        deepest.append(model.pos_channel)
    else:
        deepest.extend(model.tokenizer.parameters())
        if hasattr(model, "pos_embed"):
            deepest.append(model.pos_embed)
    param_groups.append({
        "params": deepest,
        "lr": base_lr * (layer_decay ** (n_layers + 1)),
        "weight_decay": weight_decay,
        "name": "tokenizer_posembed",
    })

    # Encoder blocks: block 0 (near input) = deepest decay
    for i, blk in enumerate(encoder_module):
        depth_from_output = n_layers - i    # block 0 = n_layers, block n-1 = 1
        param_groups.append({
            "params": list(blk.parameters()),
            "lr": base_lr * (layer_decay ** depth_from_output),
            "weight_decay": weight_decay,
            "name": f"encoder_block_{i}",
        })

    # Shallow (near output): encoder_norm + pred_head + any CF apparatus
    shallow = list(model.encoder_norm.parameters())
    for attr in ("pred_head", "jepa_predictor", "band_head", "cf_predictor",
                 "band_embed_view", "cf_band_mask_tokens",
                 # v2-specific: decoder is auxiliary MAE branch (frozen at FT)
                 ):
        if hasattr(model, attr):
            v = getattr(model, attr)
            if isinstance(v, nn.Module):
                shallow.extend(v.parameters())
            elif isinstance(v, nn.Parameter):
                shallow.append(v)
    param_groups.append({
        "params": shallow,
        "lr": base_lr,
        "weight_decay": weight_decay,
        "name": "shallow",
    })

    # Head: (base_lr × head_lr_mult), no layer decay
    param_groups.append({
        "params": list(head.parameters()),
        "lr": base_lr * head_lr_mult,
        "weight_decay": weight_decay,
        "name": "head",
    })

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def _build_cosine_warmup_scheduler(optimizer, steps_per_epoch: int,
                                    warmup_epochs: int, total_epochs: int,
                                    min_lr_ratio: float = 1e-3):
    """Cosine decay with linear warmup, matching LaBraM.

    Warmup: linear 0 → 1× base_lr over warmup_epochs.
    Decay:  cosine 1× base_lr → min_lr_ratio × base_lr over remaining.
    """
    import math
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _inject_drop_path(model, drop_prob: float):
    """Monkey-patch model.encoder blocks with stochastic depth (drop path).

    Wraps each block's forward: with prob drop_prob during training,
    the block is skipped (residual only). At eval, always active.
    Only affects model in place; use on deepcopy to avoid side effects.

    Following LaBraM: drop_prob 0.1 for Base/Large, 0.2 for Huge.
    """
    if drop_prob <= 0:
        return

    def _wrap(block, prob):
        original_forward = block.forward

        def new_forward(x):
            if block.training and torch.rand(1, device=x.device).item() < prob:
                return x   # skip entire block, keep only identity
            return original_forward(x)

        block.forward = new_forward

    for blk in model.encoder:
        _wrap(blk, drop_prob)


def run_finetune(
    base_model, train_stream, y_tr_np, test_stream, y_te_np,
    n_classes: int, dataset: str, device, max_epochs: int = 50,
    num_workers: int = 4,
    ft_protocol: str = "onecycle",
    ft_base_lr: float = 5e-4,
    ft_weight_decay: float = 0.05,
    ft_layer_decay: float = 0.65,
    ft_warmup_epochs: int = 5,
    ft_patience: int = 10,
    ft_batch_size: int = 32,
    ft_drop_path: float = 0.0,   # stochastic depth (LaBraM Base: 0.1)
    ft_head_lr_mult: float = 1.0,   # head_lr = base_lr × mult
    train_rec_ids_np=None,   # recording_ids fallback for val split
    train_patient_ids_np=None,   # patient_ids (preferred, matches LaBraM/CBraMod)
    ft_monitor_test_every: int = 0,   # 0 disables debug test monitor
    seed: int = 42,   # RNG for val split + DDP sampler + torch (multi-seed paper)
) -> dict:
    """Fine-tune backbone + new BN+Linear head, return test metrics.

    Args:
        train_stream: _TrialDataset wrapping training trials (streaming)
        y_tr_np: [N_train] int labels (for class-weight computation)
        test_stream: _TrialDataset wrapping test trials
        y_te_np: [N_test] int labels

    Protocol mirrors run_eegbench.py:
      - 85/15 train/val split for early stopping (patience=10)
      - OneCycleLR with max_lr=[4e-4 backbone, 4e-3 head]
      - AdamW(wd=0.01), grad clip 3.0
      - Class-weighted CE
      - Eval on test set with best-val checkpoint

    Streaming impl: data stays as list[np.ndarray] in RAM; DataLoader
    workers stream batches on demand. No np.stack of 70+ GB arrays.
    """
    ddp = _ddp_active()
    world = _ddp_world()
    rank = _ddp_rank()

    model = copy.deepcopy(base_model).to(device)
    # Inject drop path (stochastic depth) into encoder blocks — LaBraM style.
    # Must be applied BEFORE DDP wrapping.
    if ft_drop_path > 0:
        _inject_drop_path(model, ft_drop_path)
    head = nn.Sequential(
        nn.BatchNorm1d(model.d_model),
        nn.Linear(model.d_model, n_classes),
    ).to(device)

    # DDP wrap
    if ddp:
        # find_unused_parameters=True: some CF apparatus (band_head, cf_predictor)
        # is not used at FT time; DDP would error otherwise.
        model = DDP(model, device_ids=[device.index] if device.type == "cuda"
                    else None, find_unused_parameters=True)
        head = DDP(head, device_ids=[device.index] if device.type == "cuda"
                   else None)

    # Access underlying model for _tokenize/_encode calls
    core_model = model.module if ddp else model
    core_head = head.module if ddp else head

    y_tr = torch.from_numpy(y_tr_np).long()
    n_total = len(train_stream)

    if train_patient_ids_np is not None:
        # Patient-disjoint 80/20 val split matching LaBraM/CBraMod/CSBrain
        # (verified from their source code: they use filename.split("_")[0]
        # on TUAB naming convention `aaaaaaaa_s001_t000.edf` to get 8-letter
        # patient prefix, then 80/20 split of unique patients).
        # We use LaBraM-style: seed-parameterized shuffle (LaBraM uses unseeded;
        # we seed for reproducibility + multi-seed paper reporting).
        rng = np.random.RandomState(seed)
        unique_patients = np.unique(train_patient_ids_np)
        rng.shuffle(unique_patients)
        n_val_pat = max(1, int(len(unique_patients) * 0.20))
        val_patients = set(unique_patients[:n_val_pat].tolist())
        train_patients = set(unique_patients[n_val_pat:].tolist())
        train_idx = [i for i, p in enumerate(train_patient_ids_np) if p in train_patients]
        val_idx   = [i for i, p in enumerate(train_patient_ids_np) if p in val_patients]
        _rank0_print(f"      [FT] patient-disjoint val split (80/20, LaBraM/CBraMod-style): "
                     f"{len(train_patients)} train patients ({len(train_idx)} trials) / "
                     f"{len(val_patients)} val patients ({len(val_idx)} trials)")
    elif train_rec_ids_np is not None:
        # Fallback: recording-disjoint val split (approximation of patient-
        # disjoint when patient_ids are not available in the cache).
        rng = np.random.RandomState(seed)
        unique_recs = np.unique(train_rec_ids_np)
        rng.shuffle(unique_recs)
        n_val_recs = max(1, int(len(unique_recs) * 0.20))
        val_recs = set(unique_recs[:n_val_recs].tolist())
        train_recs = set(unique_recs[n_val_recs:].tolist())
        train_idx = [i for i, r in enumerate(train_rec_ids_np) if r in train_recs]
        val_idx   = [i for i, r in enumerate(train_rec_ids_np) if r in val_recs]
        _rank0_print(f"      [FT] recording-disjoint val split (fallback, no patient_ids): "
                     f"{len(train_recs)} train recs ({len(train_idx)} trials) / "
                     f"{len(val_recs)} val recs ({len(val_idx)} trials)")
    else:
        n_val = max(1, int(n_total * 0.15))
        n_train = n_total - n_val
        # Same seed across ranks
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_total).tolist()
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        _rank0_print(f"      [FT] trial-level val split (no rec/patient_ids provided): "
                     f"{n_train} train / {n_val} val")

    train_subset = torch.utils.data.Subset(train_stream, train_idx)
    val_subset = torch.utils.data.Subset(train_stream, val_idx)

    # DDP: use DistributedSampler on train so each rank sees 1/world_size.
    # Val/test are small enough to replicate on each rank (each rank
    # computes full metric; identical result on all ranks).
    if ddp:
        train_sampler = DistributedSampler(
            train_subset, num_replicas=world, rank=rank,
            shuffle=True, drop_last=True, seed=seed,
        )
        train_loader = DataLoader(
            train_subset, batch_size=ft_batch_size, sampler=train_sampler,
            drop_last=True, num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_subset,
            batch_size=ft_batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    val_loader = DataLoader(
        val_subset,
        batch_size=ft_batch_size * 2, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    class_counts = torch.bincount(y_tr[torch.tensor(train_idx, dtype=torch.long)],
                                    minlength=n_classes)
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    class_weights = (class_weights / class_weights.sum() * n_classes).to(device)

    steps_per_epoch = max(1, len(train_loader))
    if ft_protocol == "labram":
        optimizer = _build_labram_optimizer(
            core_model, core_head,
            base_lr=ft_base_lr,
            weight_decay=ft_weight_decay,
            layer_decay=ft_layer_decay,
            head_lr_mult=ft_head_lr_mult,
        )
        scheduler = _build_cosine_warmup_scheduler(
            optimizer,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=ft_warmup_epochs,
            total_epochs=max_epochs,
            min_lr_ratio=1e-3,
        )
        eff_batch = ft_batch_size * world
        _rank0_print(f"      [FT] LaBraM protocol: base_lr={ft_base_lr:.1e} "
                     f"layer_decay={ft_layer_decay} wd={ft_weight_decay} "
                     f"warmup={ft_warmup_epochs}ep cosine patience={ft_patience} "
                     f"drop_path={ft_drop_path} head_mult={ft_head_lr_mult} "
                     f"batch={ft_batch_size}×{world}={eff_batch}")
    else:
        optimizer = torch.optim.AdamW([
            {"params": core_model.parameters(), "lr": 1e-6},
            {"params": core_head.parameters(),  "lr": 1e-6},
        ], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[4e-4, 4e-3],
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs, pct_start=0.2,
        )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_ba = 0.0
    best_state = None
    patience = ft_patience
    no_improve = 0

    # Debug: test_ba monitor loader (persistent across epochs).
    # Used ONLY for logging/debugging — never for ckpt selection.
    # Reported final BA is still the best-val ckpt's test BA.
    monitor_test_loader = DataLoader(
        test_stream, batch_size=ft_batch_size * 2, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
        persistent_workers=num_workers > 0,
    ) if ft_monitor_test_every > 0 else None
    y_te_tensor = torch.from_numpy(y_te_np)

    def _compute_test_ba_debug():
        """Compute current test BA WITHOUT modifying model state. Debug only.
        Replicated across ranks (each rank does full test, identical result)."""
        was_training_model = core_model.training
        was_training_head = core_head.training
        core_model.eval(); core_head.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in monitor_test_loader:
                bx = bx.to(device, non_blocking=True)
                feats = core_model._encode(core_model._tokenize(bx)).mean(1)
                preds.append(core_head(feats).argmax(-1).cpu().numpy())
        preds = np.concatenate(preds)
        ba = balanced_accuracy_score(y_te_np, preds)
        if was_training_model: core_model.train()
        if was_training_head:  core_head.train()
        return float(ba)

    best_test_ba_seen = 0.0        # for oracle upper bound (debug only)
    best_test_ep_seen = 0

    for ep in range(max_epochs):
        # DDP: ensure each rank shuffles the same way each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        model.train(); head.train()
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            feats = core_model._encode(core_model._tokenize(bx)).mean(1)
            logits = head(feats)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(core_model.parameters()) + list(core_head.parameters()), 3.0)
            optimizer.step()
            scheduler.step()

        model.eval(); head.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                feats = core_model._encode(core_model._tokenize(bx)).mean(1)
                val_preds.append(core_head(feats).argmax(-1).cpu())
                val_labels.append(by)
        val_ba = balanced_accuracy_score(
            torch.cat(val_labels).numpy(),
            torch.cat(val_preds).numpy(),
        )

        improved = val_ba > best_val_ba
        if improved:
            best_val_ba = val_ba
            # Only rank 0 keeps best_state (saves RAM); other ranks skip.
            # At end, rank 0 broadcasts the state to all ranks.
            if _is_main():
                best_state = {
                    "model": {k: v.cpu().clone() for k, v in core_model.state_dict().items()},
                    "head":  {k: v.cpu().clone() for k, v in core_head.state_dict().items()},
                }
            no_improve = 0
        else:
            no_improve += 1

        # Debug: monitor test_ba every N epochs (DOES NOT affect selection)
        test_ba_str = ""
        if ft_monitor_test_every > 0 and (
            (ep + 1) % ft_monitor_test_every == 0 or ep == 0
        ):
            cur_test_ba = _compute_test_ba_debug()
            if cur_test_ba > best_test_ba_seen:
                best_test_ba_seen = cur_test_ba
                best_test_ep_seen = ep + 1
            test_ba_str = (f" TEST_ba={cur_test_ba:.4f} "
                           f"(oracle_best={best_test_ba_seen:.4f}@ep{best_test_ep_seen})")

        cur_lr = optimizer.param_groups[-1]["lr"]
        star = "*" if improved else " "
        _rank0_print(f"      ep{ep+1:03d}{star} val_ba={val_ba:.4f} "
                     f"best={best_val_ba:.4f} no_improve={no_improve}/{patience} "
                     f"lr={cur_lr:.2e}{test_ba_str}", flush=True)

        if no_improve >= patience:
            _rank0_print(f"      early stop at ep{ep+1} (best_val_ba={best_val_ba:.4f})",
                         flush=True)
            break

    # Restore best (rank 0 has best_state; broadcast to all ranks via DDP)
    if ddp:
        # Broadcast best_state from rank 0 by loading state on rank 0 then
        # all ranks sync via DDP. Simplest: rank 0 saves best_state file,
        # then all ranks load. But since best_state lives in RAM only, we
        # do an in-place broadcast of parameters.
        if _is_main() and best_state is not None:
            core_model.load_state_dict(best_state["model"])
            core_head.load_state_dict(best_state["head"])
        # Broadcast parameters from rank 0 to all
        for p in core_model.parameters():
            dist.broadcast(p.data, src=0)
        for p in core_head.parameters():
            dist.broadcast(p.data, src=0)
        for b in core_model.buffers():
            dist.broadcast(b.data, src=0)
        for b in core_head.buffers():
            dist.broadcast(b.data, src=0)
    else:
        if best_state is not None:
            core_model.load_state_dict(best_state["model"])
            core_head.load_state_dict(best_state["head"])
    model.eval(); head.eval()

    # Test — replicated on each rank (small, ~40k trials)
    test_loader = DataLoader(
        test_stream, batch_size=max(64, ft_batch_size), shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
    )
    preds, probas = [], []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device, non_blocking=True)
            feats = core_model._encode(core_model._tokenize(bx)).mean(1)
            logits = core_head(feats)
            probas.append(torch.softmax(logits, dim=-1).cpu().numpy())
            preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    proba = np.concatenate(probas)

    out = compute_metrics(y_te_np, preds, proba, dataset=dataset)
    out["best_val_ba"] = float(best_val_ba)
    out["epochs_trained"] = int(ep + 1)
    if ft_monitor_test_every > 0:
        out["oracle_best_test_ba"] = float(best_test_ba_seen)
        out["oracle_best_test_ep"] = int(best_test_ep_seen)
        _rank0_print(f"      [debug] oracle_best_test_ba={best_test_ba_seen:.4f} "
                     f"@ep{best_test_ep_seen}  "
                     f"val-selected test_ba={out['balanced_accuracy']:.4f}  "
                     f"gap={best_test_ba_seen - out['balanced_accuracy']:+.4f}",
                     flush=True)

    del model, head, best_state, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    return out


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tuab", "tuev"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tuh_dir", type=str, required=True,
                   help="Path to TUAB edf/ or TUEV edf/ root (contains train/ and eval/)")
    p.add_argument("--cache_dir", type=str,
                   default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--mode", choices=["frozen", "finetune", "both"], default="both")
    p.add_argument("--n_reps", type=int, default=5,
                   help="Repetitions for frozen probe (different LR seeds)")
    p.add_argument("--max_epochs", type=int, default=50,
                   help="Fine-tune max epochs")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=4)
    p.add_argument("--normalization", type=str, default="per_trial_zscore",
                   choices=["per_trial_zscore", "per_recording_robust"],
                   help="Must match the pretrained checkpoint's normalization. "
                        "If checkpoint was trained with per_recording_robust, "
                        "use per_recording_robust here or features will mismatch.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default=None,
                   help="JSON output path; if None, derive from checkpoint")
    p.add_argument("--include_random_baseline", action="store_true",
                   help="Also eval a fresh-random-init model with same config")
    p.add_argument("--aggregate", action="store_true",
                   help="Aggregate per-trial features to recording-level "
                        "(mean-pool over trials in same recording) before "
                        "linear probe / FT. Reduces patient-shortcut and "
                        "matches Laya/LaBraM-style recording-level eval. "
                        "Requires fresh-built dataset cache with recording_ids; "
                        "old caches (without recording_ids) fall back to per-trial.")
    # LaBraM-style FT protocol overrides
    p.add_argument("--ft_protocol", choices=["onecycle", "labram"],
                   default="onecycle",
                   help="onecycle: legacy OneCycleLR + wd=0.01 + no layer_decay "
                        "(matches run_eegbench.py). "
                        "labram: LaBraM-style — layer-wise LR decay 0.65, cosine "
                        "schedule + linear warmup, weight_decay=0.05, larger "
                        "patience by default.")
    p.add_argument("--ft_base_lr", type=float, default=5e-4,
                   help="Base LR for labram protocol; used for the head and "
                        "shallow layers, decayed for deeper layers.")
    p.add_argument("--ft_weight_decay", type=float, default=0.05,
                   help="AdamW weight decay when ft_protocol=labram.")
    p.add_argument("--ft_layer_decay", type=float, default=0.65,
                   help="Per-layer LR decay factor when ft_protocol=labram.")
    p.add_argument("--ft_warmup_epochs", type=int, default=5,
                   help="Linear warmup epochs when ft_protocol=labram.")
    p.add_argument("--ft_patience", type=int, default=10,
                   help="Early-stopping patience on val BA. Set high "
                        "(e.g. 50) to effectively disable when using labram "
                        "protocol.")
    p.add_argument("--ft_batch_size", type=int, default=32)
    p.add_argument("--ft_monitor_test_every", type=int, default=0,
                   help="If > 0, compute test_ba every N epochs and log it "
                        "(along with an 'oracle_best_test_ba' upper bound). "
                        "For DEBUG ONLY — reported final test_ba is still "
                        "based on the best-val ckpt selection. Setting to 5 "
                        "or 10 gives visibility into val vs test dynamics.")
    p.add_argument("--ft_drop_path", type=float, default=0.0,
                   help="Stochastic depth prob (LaBraM Base 0.1, Huge 0.2). "
                        "Randomly skip entire encoder blocks during training "
                        "as regularization. 0 disables.")
    p.add_argument("--ft_head_lr_mult", type=float, default=1.0,
                   help="head_lr = base_lr × mult. LaBraM uses 1.0 (same LR "
                        "for backbone and head). Set 10 to give head 10× "
                        "higher LR, matching run_eegbench's OneCycleLR split "
                        "(head 4e-3 vs backbone 4e-4). Useful when head "
                        "undertrains under labram protocol.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for val split (patient/rec/trial), "
                        "torch/numpy/random RNGs, and DDP DistributedSampler. "
                        "Vary across runs for mean±std reporting (paper).")
    args = p.parse_args()

    # ── Global seeding (paper-critical: reproducibility + multi-seed) ──
    import random as _pyrandom
    _pyrandom.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # DDP init if torchrun set WORLD_SIZE > 1
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        _rank0_print(f"[DDP] world_size={dist.get_world_size()} "
                     f"rank={dist.get_rank()} local_rank={local_rank}", flush=True)
    else:
        device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else args.device if args.device != "auto" else "cpu")

    n_classes = 2 if args.dataset == "tuab" else 6

    # ── Load pretrained ──
    model, model_cls, model_type_name, n_channels, ckpt_args = \
        load_pretrained(args.checkpoint, device)

    # ── Load datasets ──
    DSCls = TUABDataset if args.dataset == "tuab" else TUEVDataset

    print(f"\n--- Loading {args.dataset.upper()} train ---")
    t0 = time.time()
    train_ds = DSCls(
        data_dir=args.tuh_dir, split="train",
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        cache_dir=args.cache_dir,
        normalization=args.normalization,
    )
    print(f"--- Loading {args.dataset.upper()} eval ---")
    eval_ds = DSCls(
        data_dir=args.tuh_dir, split="eval",
        sample_rate=args.sample_rate,
        trial_duration_s=args.trial_duration_s,
        cache_dir=args.cache_dir,
        normalization=args.normalization,
    )
    print(f"Data loaded in {(time.time()-t0)/60:.1f} min")

    # FT path uses streaming Datasets (no np.stack, low RAM); frozen probe
    # path uses stacked arrays via dataset_to_xy (sklearn needs np.array).
    need_stacked_xy = args.mode in ("frozen", "both")
    train_stream, y_tr, rec_tr, pat_tr = dataset_to_streaming(train_ds, n_channels)
    test_stream,  y_te, rec_te, pat_te = dataset_to_streaming(eval_ds,  n_channels)
    if pat_tr is not None:
        print(f"[main] patient_ids available: train {len(set(pat_tr.tolist()))} unique "
              f"patients (matches LaBraM/CBraMod protocol)")
    else:
        print(f"[main] patient_ids NOT in cache (old cache); will fall back to "
              f"recording-disjoint val split. Rebuild cache to enable "
              f"patient-disjoint split matching LaBraM/CBraMod.")
    if need_stacked_xy:
        print(f"\n[main] --mode {args.mode}: also stacking train+eval to np.array "
              f"for frozen probe (RAM-heavy on TUAB)...")
        X_tr, _, _ = dataset_to_xy(train_ds, n_channels)
        X_te, _, _ = dataset_to_xy(eval_ds,  n_channels)
        print(f"Shapes: train {X_tr.shape}, eval {X_te.shape}")
    else:
        X_tr = X_te = None
        print(f"\nN trials: train {len(train_stream)}, eval {len(test_stream)} "
              f"(streaming mode, no np.stack)")
    print(f"Class counts (train): {np.bincount(y_tr, minlength=n_classes)}")
    print(f"Class counts (eval):  {np.bincount(y_te, minlength=n_classes)}")

    # Recording-level aggregation availability check
    aggregate_enabled = bool(args.aggregate)
    if aggregate_enabled:
        if rec_tr is None or rec_te is None:
            print("\n[aggregate] WARN: --aggregate requested but dataset cache "
                  "has no recording_ids (likely old cache). Falling back to "
                  "per-trial eval. Rebuild cache to enable aggregation.")
            aggregate_enabled = False
        else:
            n_rec_tr = len(np.unique(rec_tr))
            n_rec_te = len(np.unique(rec_te))
            print(f"\n[aggregate] Recording-level mode: "
                  f"train {n_rec_tr} recordings (from {len(y_tr)} trials), "
                  f"eval {n_rec_te} recordings (from {len(y_te)} trials)")
            print(f"[aggregate] Per-trial features will be mean-pooled within "
                  f"each recording before linear probe / FT.")

    results = {
        "checkpoint": args.checkpoint,
        "model_type": model_type_name,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "n_classes": n_classes,
        "n_train": int(len(y_tr)),
        "n_eval":  int(len(y_te)),
        "n_channels": int(n_channels),
        "ckpt_args": {k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
                          else str(v))
                      for k, v in ckpt_args.items()},
        "ft_config": {
            "protocol": args.ft_protocol,
            "base_lr": args.ft_base_lr,
            "weight_decay": args.ft_weight_decay,
            "layer_decay": args.ft_layer_decay,
            "warmup_epochs": args.ft_warmup_epochs,
            "patience": args.ft_patience,
            "batch_size": args.ft_batch_size,
            "max_epochs": args.max_epochs,
        },
    }

    # ── Frozen probe ──
    if args.mode in ("frozen", "both"):
        print(f"\n{'='*70}\n  Frozen probe ({args.n_reps} reps)\n{'='*70}")
        print("  [JEPA] Extracting features...")
        feat_tr = extract_features(model, X_tr, device)
        feat_te = extract_features(model, X_te, device)
        # Recording-level aggregation if enabled
        if aggregate_enabled:
            feat_tr_p, y_tr_p = aggregate_per_recording(feat_tr, y_tr, rec_tr)
            feat_te_p, y_te_p = aggregate_per_recording(feat_te, y_te, rec_te)
            print(f"  [aggregate] features: train {feat_tr.shape} → {feat_tr_p.shape}, "
                  f"eval {feat_te.shape} → {feat_te_p.shape}")
        else:
            feat_tr_p, y_tr_p = feat_tr, y_tr
            feat_te_p, y_te_p = feat_te, y_te
        results["jepa_frozen"] = run_frozen_probe(
            feat_tr_p, y_tr_p, feat_te_p, y_te_p, n_classes, args.dataset, args.n_reps)
        _print_metric_line("  JEPA frozen ", results["jepa_frozen"], args.dataset)

        if args.include_random_baseline:
            print("  [Random] Extracting features (untrained encoder)...")
            random_model = build_random_init(model_cls, n_channels, ckpt_args, device)
            r_tr = extract_features(random_model, X_tr, device)
            r_te = extract_features(random_model, X_te, device)
            del random_model; torch.cuda.empty_cache()
            if aggregate_enabled:
                r_tr_p, ry_tr_p = aggregate_per_recording(r_tr, y_tr, rec_tr)
                r_te_p, ry_te_p = aggregate_per_recording(r_te, y_te, rec_te)
            else:
                r_tr_p, ry_tr_p = r_tr, y_tr
                r_te_p, ry_te_p = r_te, y_te
            results["random_frozen"] = run_frozen_probe(
                r_tr_p, ry_tr_p, r_te_p, ry_te_p, n_classes, args.dataset, args.n_reps)
            _print_metric_line("  Rand frozen ", results["random_frozen"], args.dataset)

    # Save aggregation status in results for transparency
    results["aggregate_recording_level"] = aggregate_enabled

    # ── Fine-tune (streaming, low RAM) ──
    if args.mode in ("finetune", "both"):
        ft_kwargs = dict(
            ft_protocol=args.ft_protocol,
            ft_base_lr=args.ft_base_lr,
            ft_weight_decay=args.ft_weight_decay,
            ft_layer_decay=args.ft_layer_decay,
            ft_warmup_epochs=args.ft_warmup_epochs,
            ft_patience=args.ft_patience,
            ft_batch_size=args.ft_batch_size,
            ft_drop_path=args.ft_drop_path,
            ft_head_lr_mult=args.ft_head_lr_mult,
            train_rec_ids_np=rec_tr,   # Recording-disjoint fallback
            train_patient_ids_np=pat_tr,   # Patient-disjoint preferred (LaBraM/CBraMod)
            ft_monitor_test_every=args.ft_monitor_test_every,
            seed=args.seed,
        )
        print(f"\n{'='*70}\n  Fine-tune ({args.max_epochs} epochs, "
              f"streaming, protocol={args.ft_protocol})\n{'='*70}")
        print("  [JEPA] Fine-tuning...")
        results["jepa_finetune"] = run_finetune(
            model, train_stream, y_tr, test_stream, y_te,
            n_classes, args.dataset, device, args.max_epochs, **ft_kwargs)
        _print_metric_line_ft("  JEPA-FT ", results["jepa_finetune"], args.dataset)

        if args.include_random_baseline:
            print("  [Random] Fine-tuning from scratch...")
            random_model = build_random_init(model_cls, n_channels, ckpt_args, device)
            results["random_finetune"] = run_finetune(
                random_model, train_stream, y_tr, test_stream, y_te,
                n_classes, args.dataset, device, args.max_epochs, **ft_kwargs)
            _print_metric_line_ft("  Rand-FT ", results["random_finetune"], args.dataset)
            del random_model; torch.cuda.empty_cache()

    # ── Save (rank 0 only) ──
    if _is_main():
        out_path = args.output
        if out_path is None:
            ckpt_stem = Path(args.checkpoint).parent.name  # use model dir name
            out_path = f"/home/pxieaf/home2/eval/{ckpt_stem}_{args.dataset}.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n→ Saved: {out_path}")

    if _ddp_active():
        dist.destroy_process_group()


def _print_metric_line(prefix, agg, dataset):
    """Pretty-print frozen-probe aggregate dict."""
    if dataset == "tuab":
        ba = agg["balanced_accuracy"]
        roc = agg["roc_auc"]
        pr = agg["pr_auc"]
        print(f"{prefix}BA={ba['mean']:.3f}±{ba['std']:.3f}  "
              f"ROC-AUC={roc['mean']:.3f}±{roc['std']:.3f}  "
              f"PR-AUC={pr['mean']:.3f}±{pr['std']:.3f}")
    else:
        ba = agg["balanced_accuracy"]
        ck = agg["cohen_kappa"]
        f1 = agg["weighted_f1"]
        print(f"{prefix}BA={ba['mean']:.3f}±{ba['std']:.3f}  "
              f"κ={ck['mean']:.3f}±{ck['std']:.3f}  "
              f"wF1={f1['mean']:.3f}±{f1['std']:.3f}")


def _print_metric_line_ft(prefix, out, dataset):
    """Pretty-print fine-tune single-run dict."""
    if dataset == "tuab":
        print(f"{prefix}BA={out['balanced_accuracy']:.3f}  "
              f"ROC-AUC={out['roc_auc']:.3f}  PR-AUC={out['pr_auc']:.3f}  "
              f"(best_val_ba={out['best_val_ba']:.3f}, "
              f"epochs={out['epochs_trained']})")
    else:
        print(f"{prefix}BA={out['balanced_accuracy']:.3f}  "
              f"κ={out['cohen_kappa']:.3f}  wF1={out['weighted_f1']:.3f}  "
              f"(best_val_ba={out['best_val_ba']:.3f}, "
              f"epochs={out['epochs_trained']})")


if __name__ == "__main__":
    main()
