"""
EEG-JEPA: JEPA for EEG with channel-aware tokenization and VICReg.

v2 changes from v1:
  - Channel-aware tokenizer: per-channel temporal encoding + learnable
    channel embedding + spatial attention aggregation. Preserves which
    electrode sees what (critical for motor imagery: C3/C4 channels).
  - VICReg (variance + covariance) prevents dimensional collapse.
  - Block masking: contiguous 500ms-1s blocks instead of random positions.
"""

import copy
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
    """Per-channel temporal encoding + spatial attention aggregation.

    Unlike flat linear projection (which treats all channels as one blob),
    this processes each channel independently, adds learnable electrode
    identity, and uses cross-attention to aggregate channels into one
    spatially-informed token per timestep.

    This preserves the spatial structure critical for motor imagery:
    the model can learn that C3/C4 channels matter most.
    """

    def __init__(self, n_channels: int, state_samples: int, d_model: int,
                 d_channel: int = 32):
        super().__init__()
        self.state_samples = state_samples
        self.n_channels = n_channels
        self.d_channel = d_channel

        # Per-channel temporal encoder (shared weights across channels)
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

        # Spatial cross-attention: aggregate C channels → 1 token
        self.spatial_query = nn.Parameter(torch.randn(1, 1, d_channel) * 0.02)
        self.spatial_attn = nn.MultiheadAttention(
            d_channel, num_heads=4, batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(d_channel)

        # Project to d_model
        self.out_proj = nn.Linear(d_channel, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """[B, T, C] → [B, N, D] one spatially-aware token per timestep."""
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        # [B, T, C] → [B, N, S, C] → [B, N, C, S]
        windows = eeg[:, :N*S, :].reshape(B, N, S, C).permute(0, 1, 3, 2)

        # Per-channel temporal encoding: [B*N*C, S] → [B*N*C, d_ch]
        flat = windows.reshape(B * N * C, S)
        ch_features = self.temporal_encoder(flat)  # [B*N*C, d_ch]
        ch_features = ch_features.reshape(B * N, C, self.d_channel)

        # Add channel embedding: each electrode gets a unique identity
        ch_features = ch_features + self.channel_embed.unsqueeze(0)  # [B*N, C, d_ch]

        # Spatial cross-attention: query attends over all channels
        query = self.spatial_query.expand(B * N, -1, -1)  # [B*N, 1, d_ch]
        pooled, _ = self.spatial_attn(query, ch_features, ch_features)
        pooled = self.spatial_norm(pooled.squeeze(1))  # [B*N, d_ch]

        # Project to d_model
        tokens = self.out_norm(self.out_proj(pooled.reshape(B, N, -1)))
        return tokens  # [B, N, D]


# ============================================================
# 3. EMA
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
# 4. EEG-JEPA model
# ============================================================

class EEGJEPA(nn.Module):
    """JEPA for EEG with channel-aware tokenization and VICReg.

    Tokenization: per-channel temporal encoding + spatial cross-attention.
    Masking: block masking (contiguous temporal blocks, like Laya).
    Encoder: standard ViT, only processes VISIBLE tokens.
    Predictor: lightweight ViT, predicts MASKED tokens from visible.
    Target: EMA encoder processes ALL tokens.
    Loss: L2 (masked positions) + VICReg (variance + covariance).
    """

    def __init__(
        self,
        n_channels: int = 64,
        state_samples: int = 26,
        d_model: int = 256,
        d_channel: int = 32,           # per-channel feature dim
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        predictor_layers: int = 3,
        predictor_dim: int = 128,
        predictor_heads: int = 4,
        mask_ratio: float = 0.60,
        mask_block_size: int = 5,      # ~500ms contiguous blocks
        ema_decay: float = 0.996,
        var_weight: float = 5.0,
        cov_weight: float = 1.0,
        n_subjects: int = 109,
    ):
        super().__init__()
        self.state_samples = state_samples
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.var_weight = var_weight
        self.cov_weight = cov_weight

        # --- Channel-aware tokenizer ---
        self.tokenizer = ChannelAwareTokenizer(
            n_channels, state_samples, d_model, d_channel,
        )

        # --- Temporal position embedding ---
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        # --- Encoder: standard ViT ---
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

        # --- EMA (lazy init) ---
        self._ema_decay = ema_decay
        self._ema_initialized = False
        self.ema = None

    # ---- EMA management ----

    def _init_ema(self):
        if not self._ema_initialized:
            target_modules = nn.ModuleDict({
                "tokenizer": copy.deepcopy(self.tokenizer),
                "encoder": nn.ModuleList([copy.deepcopy(b) for b in self.encoder]),
                "encoder_norm": copy.deepcopy(self.encoder_norm),
            })
            self.ema = EMA(target_modules, self._ema_decay)
            self._ema_initialized = True

    def _update_ema(self):
        if self.ema is not None:
            online = nn.ModuleDict({
                "tokenizer": self.tokenizer,
                "encoder": nn.ModuleList(self.encoder),
                "encoder_norm": self.encoder_norm,
            })
            self.ema.update(online)

    # ---- Core methods ----

    def _tokenize(self, eeg: torch.Tensor) -> torch.Tensor:
        """[B, T, C] → [B, N, D] tokens with position embedding."""
        tokens = self.tokenizer(eeg)  # [B, N, D]
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N, :]
        return tokens

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    @torch.no_grad()
    def _encode_target(self, eeg: torch.Tensor) -> torch.Tensor:
        """EMA target: tokenize + encode ALL positions."""
        self._init_ema()
        tgt = self.ema.target
        tokens = tgt["tokenizer"](eeg)
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N, :].detach()
        x = tokens
        for block in tgt["encoder"]:
            x = block(x)
        return tgt["encoder_norm"](x)

    def _generate_block_mask(self, B: int, N: int, device: torch.device):
        """Generate block masking: contiguous temporal blocks.

        Instead of random individual positions, masks contiguous blocks
        of ~500ms (5 timesteps). Forces model to bridge temporal gaps
        rather than interpolating from adjacent positions.
        """
        n_mask = int(N * self.mask_ratio)
        n_vis = N - n_mask
        block_size = self.mask_block_size

        # Generate mask per sample
        all_ids_vis = []
        all_ids_mask = []

        for b in range(B):
            mask = torch.zeros(N, dtype=torch.bool, device=device)

            # Randomly place blocks until we reach desired mask ratio
            n_masked = 0
            attempts = 0
            while n_masked < n_mask and attempts < 100:
                # Random block start
                start = torch.randint(0, N, (1,)).item()
                # Random block length (variable around block_size)
                length = torch.randint(
                    max(1, block_size - 2),
                    block_size + 3,
                    (1,),
                ).item()
                end = min(start + length, N)
                mask[start:end] = True
                n_masked = mask.sum().item()
                attempts += 1

            # Ensure exactly n_mask positions are masked
            if n_masked > n_mask:
                # Remove excess
                masked_positions = mask.nonzero(as_tuple=True)[0]
                excess = n_masked - n_mask
                to_unmask = masked_positions[torch.randperm(len(masked_positions))[:excess]]
                mask[to_unmask] = False
            elif n_masked < n_mask:
                # Add more
                unmasked = (~mask).nonzero(as_tuple=True)[0]
                deficit = n_mask - n_masked
                to_mask = unmasked[torch.randperm(len(unmasked))[:deficit]]
                mask[to_mask] = True

            vis_idx = (~mask).nonzero(as_tuple=True)[0]
            mask_idx = mask.nonzero(as_tuple=True)[0]
            all_ids_vis.append(vis_idx)
            all_ids_mask.append(mask_idx)

        ids_vis = torch.stack(all_ids_vis)    # [B, n_vis]
        ids_mask = torch.stack(all_ids_mask)  # [B, n_mask]
        return ids_vis, ids_mask, n_vis, n_mask

    def forward(self, eeg: torch.Tensor, return_predictions: bool = True) -> dict:
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        # --- Tokenize all positions ---
        all_tokens = self._tokenize(eeg)  # [B, N, D]

        if not return_predictions:
            encoded = self._encode(all_tokens)
            return {"brain_states": encoded}

        # --- Block masking ---
        ids_vis, ids_mask, n_vis, n_mask = self._generate_block_mask(
            B, N, eeg.device,
        )

        # --- Encoder: process ONLY visible tokens ---
        vis_tokens = torch.gather(
            all_tokens, 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        vis_encoded = self._encode(vis_tokens)

        # --- Predictor ---
        vis_pred = self.pred_proj(vis_encoded)
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

        # --- EMA target ---
        with torch.no_grad():
            target_encoded = self._encode_target(eeg)

        masked_target = torch.gather(
            target_encoded, 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, self.d_model),
        )

        return {
            "predictions": masked_pred,
            "targets": masked_target.detach(),
            "n_vis": n_vis,
            "n_mask": n_mask,
            "brain_states": vis_encoded,
            "subj_logits": None,
        }

    def compute_loss(self, outputs: dict, subject_ids: torch.Tensor = None) -> dict:
        """L2 prediction + VICReg on encoder output."""
        pred = outputs["predictions"]
        target = outputs["targets"]
        encoded = outputs["brain_states"]

        # --- Prediction loss ---
        pred_loss = F.mse_loss(pred, target)

        # --- VICReg (prevent dimensional collapse) ---
        B, N_vis, D = encoded.shape
        x = encoded.reshape(-1, D)

        std = x.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()

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
        if self.ema is not None:
            decay = 0.996 + (0.9999 - 0.996) * progress
            self.ema.set_decay(decay)

    def initialize_electrodes(self, electrode_names: list[str]):
        pass
