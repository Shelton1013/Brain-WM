"""
Adapter to evaluate EEG-JEPA in EEG-Bench framework.

Place this file in EEG-Bench/eeg_bench/models/bci/ (and/or models/clinical/)
Then register in benchmark_console.py.

Usage:
  # After installing EEG-Bench and placing this file:
  python benchmark_console.py --model eeg_jepa --task lr
  python benchmark_console.py --model eeg_jepa --task rf
  python benchmark_console.py --model eeg_jepa --task 4class
  python benchmark_console.py --model eeg_jepa --all

Setup:
  1. Copy this file to EEG-Bench/eeg_bench/models/bci/eeg_jepa_model.py
  2. Copy eeg_jepa.py to EEG-Bench/eeg_bench/models/bci/
  3. Add to benchmark_console.py:
       from eeg_bench.models.bci.eeg_jepa_model import EEGJEPAModel
       models["eeg_jepa"] = EEGJEPAModel
  4. Set JEPA_CHECKPOINT env var or edit CHECKPOINT_PATH below
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict

# Add Brain-WM to path so we can import eeg_jepa
BRAIN_WM_DIR = os.environ.get("BRAIN_WM_DIR", "/import/home/pxieaf/Brain-WM")
if BRAIN_WM_DIR not in sys.path:
    sys.path.insert(0, BRAIN_WM_DIR)

CHECKPOINT_PATH = os.environ.get(
    "JEPA_CHECKPOINT",
    "/home/share/data_makchen/peng/models/eeg_jepa/best_model.pt"
)


class EEGJEPAModel:
    """EEG-JEPA adapter for EEG-Bench's AbstractModel interface.

    Supports two modes:
      - frozen: frozen encoder + train linear probe (default)
      - finetune: unfreeze encoder + train end-to-end

    Set via EEG_JEPA_MODE env var: "frozen" or "finetune"
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = os.environ.get("EEG_JEPA_MODE", "frozen")
        self.model = None
        self.probe = None
        self.n_classes = None

        # Load JEPA checkpoint
        print(f"Loading EEG-JEPA checkpoint: {CHECKPOINT_PATH}")
        self.ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device,
                               weights_only=False)
        self.ckpt_args = self.ckpt.get("args", {})

        # Detect n_channels from checkpoint
        self.n_channels = 64  # default
        for key, val in self.ckpt["model_state_dict"].items():
            if "channel_embed" in key:
                self.n_channels = val.shape[0]
                break
        self.d_model = self.ckpt_args.get("d_model", 256)
        print(f"  n_channels={self.n_channels}, d_model={self.d_model}, mode={self.mode}")

    def _load_jepa(self):
        from eeg_jepa import EEGJEPA
        model = EEGJEPA(
            n_channels=self.n_channels,
            d_model=self.d_model,
            encoder_layers=self.ckpt_args.get("encoder_layers", 6),
        ).to(self.device)
        model.load_state_dict(self.ckpt["model_state_dict"])
        return model

    def _preprocess(self, X: List[np.ndarray], meta: List[Dict]) -> torch.Tensor:
        """Convert EEG-Bench format to our format.

        EEG-Bench: X is list of arrays, each [n_samples, n_channels, n_timepoints]
        Our model: expects [B, T, C] (time first, then channels)
        """
        all_trials = []
        for dataset_X in X:
            # dataset_X: [n_samples, n_channels, n_timepoints]
            for trial in dataset_X:
                # trial: [n_channels, n_timepoints] → [n_timepoints, n_channels]
                t = trial.T.astype(np.float32)

                # Resample to match our model's expected length if needed
                # Our model expects 1024 samples (4s × 256Hz)
                target_len = 1024
                if t.shape[0] != target_len:
                    # Simple resampling via interpolation
                    from scipy.signal import resample
                    t = resample(t, target_len, axis=0).astype(np.float32)

                # Channel mapping: if trial has more channels than model expects
                n_ch = t.shape[1]
                if n_ch > self.n_channels:
                    # Take first n_channels (or implement proper mapping)
                    t = t[:, :self.n_channels]
                elif n_ch < self.n_channels:
                    # Pad with zeros
                    pad = np.zeros((target_len, self.n_channels - n_ch), dtype=np.float32)
                    t = np.concatenate([t, pad], axis=1)

                # Z-score normalize
                mean = t.mean(axis=0, keepdims=True)
                std = t.std(axis=0, keepdims=True) + 1e-8
                t = (t - mean) / std

                all_trials.append(t)

        return torch.from_numpy(np.stack(all_trials))  # [N, T, C]

    def _extract_features(self, X_tensor: torch.Tensor) -> np.ndarray:
        """Extract frozen encoder features."""
        self.model.eval()
        all_features = []

        with torch.no_grad():
            loader = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=False)
            for (batch,) in loader:
                batch = batch.to(self.device)
                tokens = self.model._tokenize(batch)
                encoded = self.model._encode(tokens)
                pooled = encoded.mean(dim=1)  # [B, D]
                all_features.append(pooled.cpu().numpy())

        return np.concatenate(all_features)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        """Train linear probe (frozen) or fine-tune on labeled data."""
        # Concatenate labels
        all_labels = np.concatenate(y)
        self.n_classes = len(np.unique(all_labels))

        # Preprocess EEG
        X_tensor = self._preprocess(X, meta)
        y_tensor = torch.from_numpy(all_labels).long()

        # Load JEPA model
        self.model = self._load_jepa()

        if self.mode == "frozen":
            # Extract features, train sklearn linear classifier
            features = self._extract_features(X_tensor)

            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)

            self.classifier = LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs",
                multi_class="multinomial",
            )
            self.classifier.fit(features_scaled, all_labels)

        elif self.mode == "finetune":
            # End-to-end fine-tuning
            self.probe = nn.Sequential(
                nn.BatchNorm1d(self.d_model),
                nn.Linear(self.d_model, self.n_classes),
            ).to(self.device)

            params = [
                {"params": self.model.parameters(), "lr": 1e-4},
                {"params": self.probe.parameters(), "lr": 1e-3},
            ]
            optimizer = torch.optim.AdamW(params, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

            self.model.train()
            self.probe.train()
            for epoch in range(50):
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    tokens = self.model._tokenize(batch_x)
                    encoded = self.model._encode(tokens)
                    pooled = encoded.mean(dim=1)
                    logits = self.probe(pooled)
                    loss = criterion(logits, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        """Predict labels for test data."""
        X_tensor = self._preprocess(X, meta)

        if self.mode == "frozen":
            features = self._extract_features(X_tensor)
            features_scaled = self.scaler.transform(features)
            return self.classifier.predict(features_scaled)

        elif self.mode == "finetune":
            self.model.eval()
            self.probe.eval()
            all_preds = []

            with torch.no_grad():
                loader = DataLoader(TensorDataset(X_tensor), batch_size=64)
                for (batch,) in loader:
                    batch = batch.to(self.device)
                    tokens = self.model._tokenize(batch)
                    encoded = self.model._encode(tokens)
                    pooled = encoded.mean(dim=1)
                    logits = self.probe(pooled)
                    all_preds.append(logits.argmax(-1).cpu().numpy())

            return np.concatenate(all_preds)
