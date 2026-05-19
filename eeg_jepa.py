"""
EEG-JEPA v3: LeJEPA-style single encoder + StopGrad + SIGReg.

Key changes from v2:
  - NO EMA target encoder (following Laya/LeJEPA).
    Single shared encoder processes ALL tokens. StopGrad decouples
    target representations from predictor gradient path.
  - SIGReg (isotropic Gaussian regularization) replaces VICReg.
    Forces representations to be diverse and approximately isotropic.
  - Channel-aware tokenizer and block masking retained.

Architecture (Laya-style):
  1. Tokenize ALL positions → encode ALL with shared encoder
  2. Context = encoder output at visible positions (gradients flow)
  3. Target = encoder output at masked positions (stop gradient)
  4. Predictor: context + mask tokens → predict target
  5. Loss = MSE(pred, stopgrad(target)) + λ·SIGReg(all representations)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Transformer block
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
# 2. Channel-aware tokenizer
# ============================================================

class ChannelAwareTokenizer(nn.Module):
    """Per-channel temporal encoding + spatial attention aggregation."""

    def __init__(self, n_channels: int, state_samples: int, d_model: int,
                 d_channel: int = 32):
        super().__init__()
        self.state_samples = state_samples
        self.n_channels = n_channels
        self.d_channel = d_channel

        self.temporal_encoder = nn.Sequential(
            nn.Linear(state_samples, d_channel * 2),
            nn.GELU(),
            nn.Linear(d_channel * 2, d_channel),
            nn.LayerNorm(d_channel),
        )

        self.channel_embed = nn.Parameter(
            torch.randn(n_channels, d_channel) * 0.02
        )

        self.spatial_query = nn.Parameter(torch.randn(1, 1, d_channel) * 0.02)
        self.spatial_attn = nn.MultiheadAttention(
            d_channel, num_heads=4, batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(d_channel)

        self.out_proj = nn.Linear(d_channel, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        windows = eeg[:, :N*S, :].reshape(B, N, S, C).permute(0, 1, 3, 2)
        flat = windows.reshape(B * N * C, S)
        ch_features = self.temporal_encoder(flat).reshape(B * N, C, self.d_channel)
        ch_features = ch_features + self.channel_embed.unsqueeze(0)

        query = self.spatial_query.expand(B * N, -1, -1)
        pooled, _ = self.spatial_attn(query, ch_features, ch_features)
        pooled = self.spatial_norm(pooled.squeeze(1))

        tokens = self.out_norm(self.out_proj(pooled.reshape(B, N, -1)))
        return tokens


# ============================================================
# 3. EEG-JEPA model (LeJEPA-style, no EMA)
# ============================================================

class EEGJEPA(nn.Module):
    """EEG-JEPA v3: Single encoder + StopGrad + SIGReg.

    No EMA target encoder. Single shared encoder processes ALL tokens.
    StopGrad on masked positions prevents trivial solutions.
    SIGReg prevents dimensional collapse (essential for EEG).
    """

    def __init__(
        self,
        n_channels: int = 64,
        state_samples: int = 26,
        d_model: int = 256,
        d_channel: int = 32,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        predictor_layers: int = 3,
        predictor_dim: int = 128,
        predictor_heads: int = 4,
        mask_ratio: float = 0.60,
        mask_block_size: int = 5,
        sigreg_weight: float = 0.05,   # Laya uses 0.05
        n_subjects: int = 109,
    ):
        super().__init__()
        self.state_samples = state_samples
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.sigreg_weight = sigreg_weight

        # --- Channel-aware tokenizer ---
        self.tokenizer = ChannelAwareTokenizer(
            n_channels, state_samples, d_model, d_channel,
        )

        # --- Temporal position embedding ---
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        # --- Encoder: standard ViT (processes ALL tokens) ---
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, encoder_heads)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- Predictor: lightweight ViT ---
        self.pred_proj = nn.Linear(d_model, predictor_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)
        self.pred_pos_embed = nn.Parameter(torch.randn(1, 256, predictor_dim) * 0.02)
        self.predictor = nn.ModuleList([
            TransformerBlock(predictor_dim, predictor_heads)
            for _ in range(predictor_layers)
        ])
        self.pred_norm = nn.LayerNorm(predictor_dim)
        self.pred_out = nn.Linear(predictor_dim, d_model)

        # No EMA — single encoder, StopGrad on targets

    # ---- Core methods ----

    def _tokenize(self, eeg: torch.Tensor) -> torch.Tensor:
        """[B, T, C] → [B, N, D] tokens with position embedding."""
        tokens = self.tokenizer(eeg)
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N, :]
        return tokens

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def _generate_block_mask(self, B: int, N: int, device: torch.device):
        """Block masking: contiguous temporal blocks of ~500ms."""
        n_mask = int(N * self.mask_ratio)
        n_vis = N - n_mask
        block_size = self.mask_block_size

        all_ids_vis = []
        all_ids_mask = []

        for b in range(B):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            n_masked = 0
            attempts = 0
            while n_masked < n_mask and attempts < 100:
                start = torch.randint(0, N, (1,)).item()
                length = torch.randint(
                    max(1, block_size - 2), block_size + 3, (1,),
                ).item()
                end = min(start + length, N)
                mask[start:end] = True
                n_masked = mask.sum().item()
                attempts += 1

            if n_masked > n_mask:
                masked_positions = mask.nonzero(as_tuple=True)[0]
                excess = n_masked - n_mask
                to_unmask = masked_positions[torch.randperm(len(masked_positions))[:excess]]
                mask[to_unmask] = False
            elif n_masked < n_mask:
                unmasked = (~mask).nonzero(as_tuple=True)[0]
                deficit = n_mask - n_masked
                to_mask = unmasked[torch.randperm(len(unmasked))[:deficit]]
                mask[to_mask] = True

            all_ids_vis.append((~mask).nonzero(as_tuple=True)[0])
            all_ids_mask.append(mask.nonzero(as_tuple=True)[0])

        ids_vis = torch.stack(all_ids_vis)
        ids_mask = torch.stack(all_ids_mask)
        return ids_vis, ids_mask, n_vis, n_mask

    def forward(self, eeg: torch.Tensor, return_predictions: bool = True) -> dict:
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        # --- Tokenize ALL positions ---
        all_tokens = self._tokenize(eeg)  # [B, N, D]

        # --- Encode ALL positions with shared encoder ---
        all_encoded = self._encode(all_tokens)  # [B, N, D]

        if not return_predictions:
            return {"brain_states": all_encoded}

        # --- Block masking ---
        ids_vis, ids_mask, n_vis, n_mask = self._generate_block_mask(
            B, N, eeg.device,
        )

        # --- Split: context (gradients) vs target (stop gradient) ---
        context_encoded = torch.gather(
            all_encoded, 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, self.d_model),
        )  # [B, n_vis, D] — gradients flow through

        target_encoded = torch.gather(
            all_encoded, 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, self.d_model),
        ).detach()  # [B, n_mask, D] — STOP GRADIENT

        # --- Predictor: context + mask tokens → predict targets ---
        vis_pred = self.pred_proj(context_encoded)
        pred_dim = self.pred_proj.out_features

        vis_pos = torch.gather(
            self.pred_pos_embed[:, :N, :].expand(B, -1, -1), 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, pred_dim),
        )
        vis_pred = vis_pred + vis_pos

        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        mask_pos = torch.gather(
            self.pred_pos_embed[:, :N, :].expand(B, -1, -1), 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, pred_dim),
        )
        mask_tokens = mask_tokens + mask_pos

        pred_input = torch.cat([vis_pred, mask_tokens], dim=1)
        for block in self.predictor:
            pred_input = block(pred_input)
        pred_output = self.pred_norm(pred_input)
        masked_pred = self.pred_out(pred_output[:, n_vis:, :])

        return {
            "predictions": masked_pred,       # [B, n_mask, D]
            "targets": target_encoded,         # [B, n_mask, D] (already detached)
            "all_encoded": all_encoded,        # [B, N, D] for SIGReg
            "n_vis": n_vis,
            "n_mask": n_mask,
            "brain_states": all_encoded,       # for eval compatibility
            "subj_logits": None,
        }

    def compute_loss(self, outputs: dict, subject_ids: torch.Tensor = None) -> dict:
        """L2 prediction + SIGReg on encoder output.

        SIGReg encourages representations to be approximately isotropic
        Gaussian: all dimensions active, uncorrelated, unit variance.
        Essential for EEG where low information density causes collapse.
        """
        pred = outputs["predictions"]       # [B, n_mask, D]
        target = outputs["targets"]         # [B, n_mask, D] (already stop-grad)
        all_enc = outputs["all_encoded"]    # [B, N, D]

        # --- Prediction loss (L2) ---
        pred_loss = F.mse_loss(pred, target)

        # --- SIGReg: push representations toward isotropic Gaussian ---
        B, N, D = all_enc.shape
        x = all_enc.reshape(-1, D)  # [B*N, D]

        # Variance term: per-dim std → 1
        std = x.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()

        # Covariance term: off-diagonal → 0 (decorrelation)
        x_centered = x - x.mean(dim=0, keepdim=True)
        cov = (x_centered.T @ x_centered) / max(x.shape[0] - 1, 1)
        cov_loss = cov.fill_diagonal_(0).pow(2).sum() / D

        sigreg_loss = var_loss + cov_loss
        total = pred_loss + self.sigreg_weight * sigreg_loss

        return {
            "total": total,
            "pred": pred_loss,
            "var": var_loss,
            "cov": cov_loss,
            "adv": torch.tensor(0.0, device=pred.device),
        }

    def update_ema(self):
        """No-op: no EMA in LeJEPA-style architecture."""
        pass

    def set_training_progress(self, progress: float):
        """No-op: no EMA decay to schedule."""
        pass

    def initialize_electrodes(self, electrode_names: list[str]):
        pass
