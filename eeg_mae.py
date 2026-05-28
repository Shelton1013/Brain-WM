"""
EEG-MAE: Masked Autoencoder baseline for EEG.

Pretraining objective: reconstruct raw EEG patches at masked positions.
This is the "reconstruction" baseline for ablation against JEPA (latent prediction).

Differences from EEG-JEPA:
  - JEPA: predict latent representations at masked positions
  - MAE: reconstruct raw EEG signal at masked positions
  - Same tokenizer, encoder, masking strategy
  - Decoder replaces predictor (projects back to raw patch dim)

Reference: Laya ablation (Table 3) shows JEPA > MAE by 7.4% on clinical tasks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_jepa import TransformerBlock, DynamicChannelMixer
from regularizers import distribution_reg


class EEGMAE(nn.Module):
    """Masked Autoencoder for EEG.

    Same architecture as EEG-JEPA but reconstructs raw EEG patches
    instead of predicting latent representations.

    Encoder sees only visible tokens → decoder reconstructs masked patches.
    """

    def __init__(
        self,
        n_channels: int = 64,
        state_samples: int = 26,
        d_model: int = 256,
        d_channel: int = 32,
        n_queries: int = 16,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        decoder_layers: int = 3,
        decoder_dim: int = 128,
        decoder_heads: int = 4,
        mask_ratio: float = 0.60,
        mask_block_size: int = 5,
        sigreg_weight: float = 0.05,
        query_spec_weight: float = 0.1,
        n_subjects: int = 109,
        reg_type: str = "sigreg",        # "sigreg" (CF test) | "vicreg" (var+cov)
    ):
        super().__init__()
        self.state_samples = state_samples
        self.n_channels = n_channels
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.sigreg_weight = sigreg_weight
        self.query_spec_weight = query_spec_weight
        self.reg_type = reg_type

        # Raw patch dimension (for reconstruction target)
        self.patch_dim = state_samples * n_channels  # 26 * 64 = 1664

        # --- Tokenizer (same as JEPA) ---
        self.tokenizer = DynamicChannelMixer(
            n_channels, state_samples, d_model, d_channel, n_queries,
        )

        # --- Positional embedding ---
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        # --- Encoder: processes VISIBLE tokens only (like MAE) ---
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, encoder_heads)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- Decoder: reconstructs masked patches ---
        self.decoder_proj = nn.Linear(d_model, decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, 256, decoder_dim) * 0.02)
        self.decoder = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads)
            for _ in range(decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Project back to raw patch dimension
        self.reconstruction_head = nn.Linear(decoder_dim, self.patch_dim)

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

    def _get_raw_patches(self, eeg: torch.Tensor) -> torch.Tensor:
        """Extract raw EEG patches as reconstruction targets.

        [B, T, C] → [B, N, S*C] where each patch is a flattened
        100ms × all_channels window.
        """
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S
        # [B, T, C] → [B, N, S, C] → [B, N, S*C]
        patches = eeg[:, :N*S, :].reshape(B, N, S, C).reshape(B, N, S * C)
        return patches

    def _generate_block_mask(self, B: int, N: int, device: torch.device):
        """Block masking (same as JEPA)."""
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

        # Tokenize all positions
        all_tokens = self._tokenize(eeg)  # [B, N, D]

        # For eval: encode all tokens
        if not return_predictions:
            encoded = self._encode(all_tokens)
            return {"brain_states": encoded}

        # Block masking
        ids_vis, ids_mask, n_vis, n_mask = self._generate_block_mask(
            B, N, eeg.device,
        )

        # Encoder: ONLY visible tokens (standard MAE)
        vis_tokens = torch.gather(
            all_tokens, 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        vis_encoded = self._encode(vis_tokens)  # [B, n_vis, D]

        # Decoder: visible encoded + mask tokens → reconstruct masked
        vis_dec = self.decoder_proj(vis_encoded)
        dec_dim = self.decoder_proj.out_features

        vis_pos = torch.gather(
            self.decoder_pos_embed[:, :N, :].expand(B, -1, -1), 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, dec_dim),
        )
        vis_dec = vis_dec + vis_pos

        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        mask_pos = torch.gather(
            self.decoder_pos_embed[:, :N, :].expand(B, -1, -1), 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, dec_dim),
        )
        mask_tokens = mask_tokens + mask_pos

        dec_input = torch.cat([vis_dec, mask_tokens], dim=1)
        for block in self.decoder:
            dec_input = block(dec_input)
        dec_output = self.decoder_norm(dec_input)

        # Reconstruct masked patches → raw EEG
        reconstructed = self.reconstruction_head(dec_output[:, n_vis:, :])  # [B, n_mask, S*C]

        # Get raw patch targets at masked positions
        raw_patches = self._get_raw_patches(eeg)  # [B, N, S*C]
        target_patches = torch.gather(
            raw_patches, 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, self.patch_dim),
        )  # [B, n_mask, S*C]

        return {
            "predictions": reconstructed,       # [B, n_mask, S*C]
            "targets": target_patches,           # [B, n_mask, S*C]
            "all_encoded": vis_encoded,          # [B, n_vis, D] for SIGReg
            "n_vis": n_vis,
            "n_mask": n_mask,
            "brain_states": vis_encoded,
            "subj_logits": None,
        }

    def compute_loss(self, outputs: dict, subject_ids: torch.Tensor = None) -> dict:
        """MSE reconstruction loss + distribution regularization."""
        pred = outputs["predictions"]       # [B, n_mask, S*C]
        target = outputs["targets"]         # [B, n_mask, S*C]
        encoded = outputs["all_encoded"]    # [B, n_vis, D]

        # --- Reconstruction loss (MSE on raw EEG patches) ---
        pred_loss = F.mse_loss(pred, target)

        # --- Distribution regularization (true SIGReg, or VICReg ablation) ---
        x = encoded.reshape(-1, encoded.shape[-1])
        reg, reg_info = distribution_reg(x, self.reg_type)

        # Query specialization loss
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
        pass

    def set_training_progress(self, progress: float):
        pass

    def initialize_electrodes(self, electrode_names: list[str]):
        pass
