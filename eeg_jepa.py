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

from regularizers import distribution_reg


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

class DynamicChannelMixer(nn.Module):
    """Dynamic Channel Mixer with multi-query cross-attention (Laya-style).

    Unlike single-query aggregation (which compresses all spatial info into
    one vector), this uses N_q learned queries that each attend to different
    channel combinations. Query Specialization Loss ensures the queries
    don't collapse to identical patterns.

    This preserves fine-grained spatial information critical for MI:
    e.g., one query focuses on C3/C4 (motor cortex), another on Fz/Cz
    (midline), another on temporal channels, etc.
    """

    def __init__(self, n_channels: int, state_samples: int, d_model: int,
                 d_channel: int = 32, n_queries: int = 16):
        super().__init__()
        self.state_samples = state_samples
        self.n_channels = n_channels
        self.d_channel = d_channel
        self.n_queries = n_queries

        # Per-channel temporal encoder (shared across channels)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(state_samples, d_channel * 2),
            nn.GELU(),
            nn.Linear(d_channel * 2, d_channel),
            nn.LayerNorm(d_channel),
        )

        # Learnable channel embedding (electrode spatial identity)
        self.channel_embed = nn.Parameter(
            torch.randn(n_channels, d_channel) * 0.02
        )

        # Multi-query cross-attention: N_q queries attend over channels
        self.spatial_queries = nn.Parameter(
            torch.randn(n_queries, d_channel) * 0.02
        )
        self.spatial_attn = nn.MultiheadAttention(
            d_channel, num_heads=4, batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(d_channel)

        # Project concatenated query outputs to d_model
        self.out_proj = nn.Linear(n_queries * d_channel, d_model)
        self.out_norm = nn.LayerNorm(d_model)

        # Store attention weights for query specialization loss
        self._last_attn_weights = None

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """[B, T, C] → [B, N, D] with multi-query spatial aggregation."""
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        # Per-channel temporal encoding: [B*N*C, S] → [B*N, C, d_ch]
        windows = eeg[:, :N*S, :].reshape(B, N, S, C).permute(0, 1, 3, 2)
        flat = windows.reshape(B * N * C, S)
        ch_features = self.temporal_encoder(flat).reshape(B * N, C, self.d_channel)

        # Add channel embedding
        ch_features = ch_features + self.channel_embed.unsqueeze(0)

        # Multi-query cross-attention: [B*N, N_q, d_ch]
        queries = self.spatial_queries.unsqueeze(0).expand(B * N, -1, -1)
        pooled, attn_weights = self.spatial_attn(
            queries, ch_features, ch_features,
        )  # pooled: [B*N, N_q, d_ch], attn_weights: [B*N, N_q, C]
        pooled = self.spatial_norm(pooled)

        # Store attention weights for query specialization loss
        self._last_attn_weights = attn_weights

        # Concat all queries: [B*N, N_q * d_ch] → project to d_model
        pooled_flat = pooled.reshape(B * N, -1)  # [B*N, N_q * d_ch]
        tokens = self.out_norm(self.out_proj(pooled_flat).reshape(B, N, -1))
        return tokens  # [B, N, D]

    def get_query_specialization_loss(self) -> torch.Tensor:
        """Penalize pairwise similarity between query affinity vectors.

        Each query's affinity vector W_i is its average attention weight
        over channels. If two queries attend to the same channels,
        their affinity vectors will be similar → penalize this.

        Returns scalar loss (0 if no attention weights cached).
        """
        if self._last_attn_weights is None:
            return torch.tensor(0.0)

        # attn_weights: [B*N, N_q, C]
        # Average over batch*time to get per-query channel affinity
        W = self._last_attn_weights.mean(dim=0)  # [N_q, C]
        W = F.normalize(W, dim=-1)  # normalize each query's affinity

        # Pairwise similarity matrix
        similarity = W @ W.T  # [N_q, N_q]

        # Penalize off-diagonal entries (queries should be different)
        loss = similarity.fill_diagonal_(0).pow(2).sum() / self.n_queries
        return loss


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
        n_queries: int = 16,           # Dynamic Channel Mixer queries
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        predictor_layers: int = 3,
        predictor_dim: int = 128,
        predictor_heads: int = 4,
        mask_ratio: float = 0.60,
        mask_block_size: int = 5,
        sigreg_weight: float = 0.05,
        query_spec_weight: float = 0.1,  # Query Specialization Loss weight
        n_subjects: int = 109,
        reg_type: str = "sigreg",        # "sigreg" (CF test) | "vicreg" (var+cov)
    ):
        super().__init__()
        self.state_samples = state_samples
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.sigreg_weight = sigreg_weight
        self.query_spec_weight = query_spec_weight
        self.reg_type = reg_type

        # --- Dynamic Channel Mixer (Laya-style multi-query) ---
        self.tokenizer = DynamicChannelMixer(
            n_channels, state_samples, d_model, d_channel, n_queries,
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
        """L2 prediction + distribution regularization on encoder output.

        The regularizer (true SIGReg, or VICReg ablation) encourages
        representations to be approximately isotropic Gaussian: all
        dimensions active, uncorrelated, unit variance. Essential for EEG
        where low information density causes collapse.
        """
        pred = outputs["predictions"]       # [B, n_mask, D]
        target = outputs["targets"]         # [B, n_mask, D] (already stop-grad)
        all_enc = outputs["all_encoded"]    # [B, N, D]

        # --- Prediction loss (L2) ---
        pred_loss = F.mse_loss(pred, target)

        # --- Distribution regularization (true SIGReg, or VICReg ablation) ---
        x = all_enc.reshape(-1, all_enc.shape[-1])  # [B*N, D]
        reg, reg_info = distribution_reg(x, self.reg_type)

        # --- Query Specialization Loss (force diverse channel queries) ---
        query_loss = self.tokenizer.get_query_specialization_loss()

        total = (pred_loss
                 + self.sigreg_weight * reg
                 + self.query_spec_weight * query_loss)

        return {
            "total": total,
            "pred": pred_loss,
            **reg_info,
            "qspec": query_loss,
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
