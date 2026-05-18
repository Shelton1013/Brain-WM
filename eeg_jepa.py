"""
EEG-JEPA: Simplest possible JEPA baseline for EEG.

Direct adaptation of I-JEPA to 1D EEG time series.
No Mamba, no custom encoder, no adversarial, no region masking.
Just: tokenize → mask → ViT encode visible → ViT predict masked → L2 loss.

If this works on PhysioNet MI → the JEPA paradigm is valid for EEG.
If this fails → the problem is fundamental, not architectural.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Transformer blocks
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 2. EMA target encoder
# ============================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.decay = decay
        self.target = copy.deepcopy(model)
        for p in self.target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for online_p, target_p in zip(model.parameters(), self.target.parameters()):
            target_p.data.mul_(self.decay).add_(online_p.data, alpha=1 - self.decay)

    def set_decay(self, decay: float):
        self.decay = decay


# ============================================================
# 3. EEG-JEPA model
# ============================================================

class EEGJEPA(nn.Module):
    """JEPA for EEG with VICReg dimensional regularization.

    Tokenization: each 100ms window × all channels → one token.
    Masking: random 60% of timestep tokens.
    Encoder: standard ViT, only processes VISIBLE tokens.
    Predictor: lightweight ViT, predicts MASKED tokens from visible.
    Target: EMA encoder processes ALL tokens.
    Loss: L2 (masked positions) + VICReg (variance + covariance on encoder output).

    VICReg is needed for EEG (but not vision/fMRI) because EEG's low
    information density causes the encoder to collapse into a tiny subspace.
    """

    def __init__(
        self,
        n_channels: int = 64,
        state_samples: int = 26,       # samples per 100ms window
        d_model: int = 256,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        predictor_layers: int = 3,
        predictor_dim: int = 128,
        predictor_heads: int = 4,
        mask_ratio: float = 0.60,
        ema_decay: float = 0.996,
        var_weight: float = 5.0,       # VICReg variance weight
        cov_weight: float = 1.0,       # VICReg covariance weight
        n_subjects: int = 109,         # for compatibility with dataloader
    ):
        super().__init__()
        self.state_samples = state_samples
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.var_weight = var_weight
        self.cov_weight = cov_weight

        # --- Tokenizer: linear projection of raw EEG window ---
        token_dim = state_samples * n_channels  # 26 * 64 = 1664
        self.patch_proj = nn.Linear(token_dim, d_model)
        self.patch_norm = nn.LayerNorm(d_model)

        # --- Positional embedding ---
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        # --- Encoder: standard ViT (processes visible tokens only) ---
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, encoder_heads)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- Predictor: lightweight ViT ---
        self.pred_proj = nn.Linear(d_model, predictor_dim)  # narrow down
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)
        self.pred_pos_embed = nn.Parameter(torch.randn(1, 256, predictor_dim) * 0.02)
        self.predictor = nn.ModuleList([
            TransformerBlock(predictor_dim, predictor_heads)
            for _ in range(predictor_layers)
        ])
        self.pred_norm = nn.LayerNorm(predictor_dim)
        self.pred_out = nn.Linear(predictor_dim, d_model)  # back to d_model

        # --- EMA (initialized lazily) ---
        self._ema_decay = ema_decay
        self._ema_initialized = False
        self.ema = None

    def _init_ema(self):
        if not self._ema_initialized:
            # EMA of tokenizer + encoder
            target_modules = nn.ModuleDict({
                "patch_proj": copy.deepcopy(self.patch_proj),
                "patch_norm": copy.deepcopy(self.patch_norm),
                "encoder": nn.ModuleList([copy.deepcopy(b) for b in self.encoder]),
                "encoder_norm": copy.deepcopy(self.encoder_norm),
            })
            self.ema = EMA(target_modules, self._ema_decay)
            self._ema_initialized = True

    def _update_ema(self):
        if self.ema is not None:
            online = nn.ModuleDict({
                "patch_proj": self.patch_proj,
                "patch_norm": self.patch_norm,
                "encoder": nn.ModuleList(self.encoder),
                "encoder_norm": self.encoder_norm,
            })
            self.ema.update(online)

    def _tokenize(self, eeg: torch.Tensor) -> torch.Tensor:
        """[B, T, C] → [B, N, D] tokens."""
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S
        # [B, T, C] → [B, N, S, C] → [B, N, S*C]
        windows = eeg[:, :N*S, :].reshape(B, N, S, C).reshape(B, N, S * C)
        tokens = self.patch_norm(self.patch_proj(windows))
        # Add position
        tokens = tokens + self.pos_embed[:, :N, :]
        return tokens

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run ViT encoder on tokens."""
        x = tokens
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    @torch.no_grad()
    def _encode_target(self, eeg: torch.Tensor) -> torch.Tensor:
        """EMA target: tokenize + encode ALL positions."""
        self._init_ema()
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S
        windows = eeg[:, :N*S, :].reshape(B, N, S, C).reshape(B, N, S * C)

        tgt = self.ema.target
        tokens = tgt["patch_norm"](tgt["patch_proj"](windows))
        tokens = tokens + self.pos_embed[:, :N, :].detach()

        x = tokens
        for block in tgt["encoder"]:
            x = block(x)
        return tgt["encoder_norm"](x)  # [B, N, D]

    def forward(self, eeg: torch.Tensor, return_predictions: bool = True) -> dict:
        """
        Args:
            eeg: [B, T, C] raw EEG
        Returns:
            dict with predictions, targets, mask info
        """
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        # --- Tokenize all positions ---
        all_tokens = self._tokenize(eeg)  # [B, N, D]

        if not return_predictions:
            encoded = self._encode(all_tokens)
            return {"brain_states": encoded}

        # --- Generate random mask ---
        n_mask = int(N * self.mask_ratio)
        n_vis = N - n_mask

        # Per-sample random permutation for masking
        noise = torch.rand(B, N, device=eeg.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        # Visible and masked indices
        ids_vis = ids_shuffle[:, :n_vis]    # [B, n_vis]
        ids_mask = ids_shuffle[:, n_vis:]   # [B, n_mask]

        # --- Encoder: process ONLY visible tokens ---
        vis_tokens = torch.gather(
            all_tokens, 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, self.d_model)
        )  # [B, n_vis, D]

        vis_encoded = self._encode(vis_tokens)  # [B, n_vis, D]

        # --- Predictor: visible context + mask tokens → predict masked ---
        # Project visible to predictor dim
        vis_pred = self.pred_proj(vis_encoded)  # [B, n_vis, pred_dim]

        # Add position embedding for visible tokens
        vis_pos = torch.gather(
            self.pred_pos_embed[:, :N, :].expand(B, -1, -1), 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, self.pred_proj.out_features)
        )
        vis_pred = vis_pred + vis_pos

        # Create mask tokens with position embedding for masked positions
        mask_tokens = self.mask_token.expand(B, n_mask, -1)  # [B, n_mask, pred_dim]
        mask_pos = torch.gather(
            self.pred_pos_embed[:, :N, :].expand(B, -1, -1), 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, self.pred_proj.out_features)
        )
        mask_tokens = mask_tokens + mask_pos

        # Concatenate: [visible, masked] → predictor
        pred_input = torch.cat([vis_pred, mask_tokens], dim=1)  # [B, N, pred_dim]
        for block in self.predictor:
            pred_input = block(pred_input)
        pred_output = self.pred_norm(pred_input)

        # Extract predictions for masked positions only (last n_mask tokens)
        masked_pred = self.pred_out(pred_output[:, n_vis:, :])  # [B, n_mask, D]

        # --- EMA target: encode ALL positions (full context) ---
        with torch.no_grad():
            target_encoded = self._encode_target(eeg)  # [B, N, D]

        # Extract targets at masked positions
        masked_target = torch.gather(
            target_encoded, 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, self.d_model)
        )  # [B, n_mask, D]

        return {
            "predictions": masked_pred,
            "targets": masked_target.detach(),
            "n_vis": n_vis,
            "n_mask": n_mask,
            # For compatibility with train.py
            "brain_states": vis_encoded,
            "subj_logits": None,
        }

    def compute_loss(self, outputs: dict, subject_ids: torch.Tensor = None) -> dict:
        """L2 prediction + VICReg variance/covariance on encoder output.

        EEG's low information density causes JEPA to collapse into a
        low-dimensional subspace (4/256 dims). VICReg forces all dimensions
        to be active and decorrelated — the key difference from vision JEPA.
        """
        pred = outputs["predictions"]    # [B, n_mask, D]
        target = outputs["targets"]      # [B, n_mask, D]
        encoded = outputs["brain_states"]  # [B, n_vis, D] encoder output

        # --- Prediction loss (L2 on masked positions) ---
        pred_loss = F.mse_loss(pred, target)

        # --- VICReg on encoder output (prevent dimensional collapse) ---
        B, N_vis, D = encoded.shape
        x = encoded.reshape(-1, D)  # [B*N_vis, D]

        # Variance: per-dim std must be >= 1
        std = x.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()

        # Covariance: off-diagonal of cov matrix → 0 (decorrelate dims)
        x_centered = x - x.mean(dim=0, keepdim=True)
        cov = (x_centered.T @ x_centered) / max(x.shape[0] - 1, 1)
        cov_loss = cov.fill_diagonal_(0).pow(2).sum() / D

        total = pred_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

        return {
            "total": total,
            "pred": pred_loss,
            "var": var_loss,
            "cov": cov_loss,
            "adv": torch.tensor(0.0, device=pred.device),
        }

    def update_ema(self):
        self._update_ema()

    def set_training_progress(self, progress: float):
        """Update EMA decay schedule."""
        if self.ema is not None:
            decay = 0.996 + (0.9999 - 0.996) * progress
            self.ema.set_decay(decay)

    def initialize_electrodes(self, electrode_names: list[str]):
        """Compatibility stub — JEPA baseline doesn't use electrode mapping."""
        pass
