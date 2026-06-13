"""
EEG-LeJEPA + Cross-Frequency: isolate the cross-frequency latent prediction.

This variant adds ONLY the cross-frequency objective on top of a spectral
tokenizer — no region masking. Its purpose is to measure cross-frequency
latent prediction (the paper's claimed novelty) cleanly:

  - vs base LeJEPA: effect of (spectral tokenizer + cross-frequency)
  - vs THIS SAME model run with --freq_mask_weight 0: effect of cross-frequency
    ALONE (identical architecture, only the auxiliary loss is toggled off).
    This is the matched control — no confound from differing tokenizers.

Cross-frequency prediction here follows LeJEPA: the target is NOT detached
(no StopGrad), and there is no transformer predictor. SIGReg on the encoded
representation is what prevents collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_jepa import TransformerBlock
from eeg_lejepa_full import SpectralTokenizer, CrossFrequencyPredictor
from regularizers import distribution_reg


class EEGLeJEPACrossFreq(nn.Module):
    """Base LeJEPA + spectral tokenizer + cross-frequency latent prediction."""

    def __init__(
        self,
        n_channels: int = 64,
        state_samples: int = 26,
        d_model: int = 256,
        d_channel: int = 32,
        n_queries: int = 16,
        n_bands: int = 5,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        mask_ratio: float = 0.60,
        mask_block_size: int = 5,
        freq_mask_weight: float = 1.0,   # 0.0 = matched no-CF control
        sigreg_lambda: float = 0.05,
        query_spec_weight: float = 0.1,
        n_subjects: int = 109,
        reg_type: str = "sigreg",
        cf_band_conditioned: bool = True,  # ★ predictor told which band to predict
        cf_preserve_spatial: bool = True,  # ★ keep per-channel band features
        cf_d_band: int = None,             # ★ per-band latent dim (None = legacy 8)
    ):
        super().__init__()
        self.state_samples = state_samples
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.sigreg_lambda = sigreg_lambda
        self.query_spec_weight = query_spec_weight
        self.freq_mask_weight = freq_mask_weight
        self.reg_type = reg_type
        self.cf_preserve_spatial = cf_preserve_spatial

        # Spectral tokenizer (preserves per-band representations for CF prediction)
        self.tokenizer = SpectralTokenizer(
            n_channels, state_samples, d_model, d_channel, n_queries, n_bands,
            d_band=cf_d_band,
        )

        # Cross-frequency predictor (the novelty being isolated)
        self.freq_predictor = CrossFrequencyPredictor(
            n_bands=n_bands, d_band=self.tokenizer.d_band,
            band_conditioned=cf_band_conditioned,
            preserve_spatial=cf_preserve_spatial,
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, encoder_heads)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Lightweight MLP prediction head (same as base LeJEPA, no transformer predictor)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _encode(self, tokens):
        x = tokens
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def _tokenize(self, eeg):
        """For eval: tokens without band info."""
        tokens = self.tokenizer(eeg, return_band_tokens=False)
        N = tokens.shape[1]
        return tokens + self.pos_embed[:, :N, :]

    def _generate_block_mask(self, B, N, device):
        n_mask = int(N * self.mask_ratio)
        n_vis = N - n_mask
        bs = self.mask_block_size
        all_vis, all_mask = [], []
        for b in range(B):
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            attempts = 0
            while mask.sum() < n_mask and attempts < 100:
                start = torch.randint(0, N, (1,)).item()
                length = torch.randint(max(1, bs-2), bs+3, (1,)).item()
                mask[start:min(start+length, N)] = True
                attempts += 1
            if mask.sum() > n_mask:
                pos = mask.nonzero(as_tuple=True)[0]
                mask[pos[torch.randperm(len(pos))[:mask.sum()-n_mask]]] = False
            elif mask.sum() < n_mask:
                unm = (~mask).nonzero(as_tuple=True)[0]
                mask[unm[torch.randperm(len(unm))[:n_mask-mask.sum()]]] = True
            all_vis.append((~mask).nonzero(as_tuple=True)[0])
            all_mask.append(mask.nonzero(as_tuple=True)[0])
        return torch.stack(all_vis), torch.stack(all_mask), n_vis, n_mask

    def forward(self, eeg, return_predictions=True):
        B, T, C = eeg.shape
        N = T // self.state_samples

        # Spectral tokenization (with band-level features during training)
        if self.training and return_predictions:
            tokens, band_tokens = self.tokenizer(
                eeg, return_band_tokens=True,
                band_tokens_spatial=self.cf_preserve_spatial,
            )
        else:
            tokens = self.tokenizer(eeg, return_band_tokens=False)
            band_tokens = None

        tokens = tokens + self.pos_embed[:, :N, :]

        # Cross-frequency masking (before encoder, no StopGrad)
        freq_loss = torch.tensor(0.0, device=eeg.device)
        if band_tokens is not None:
            freq_loss, _ = self.freq_predictor.mask_and_predict(band_tokens)

        all_encoded = self._encode(tokens)

        if not return_predictions:
            return {"brain_states": all_encoded}

        # Temporal block masking (after encoder)
        ids_vis, ids_mask, n_vis, n_mask = self._generate_block_mask(B, N, eeg.device)

        vis_encoded = torch.gather(all_encoded, 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, self.d_model))
        mask_encoded = torch.gather(all_encoded, 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, self.d_model))

        vis_context = vis_encoded.mean(dim=1, keepdim=True).expand(-1, n_mask, -1)
        predictions = self.pred_head(vis_context)

        return {
            "predictions": predictions,
            "targets": mask_encoded,
            "all_encoded": all_encoded,
            "freq_loss": freq_loss,
            "n_vis": n_vis, "n_mask": n_mask,
            "brain_states": all_encoded,
            "subj_logits": None,
        }

    def compute_loss(self, outputs, subject_ids=None):
        pred = outputs["predictions"]
        target = outputs["targets"]
        all_enc = outputs["all_encoded"]

        # Temporal prediction loss
        pred_loss = F.mse_loss(pred, target)

        # Cross-frequency loss (the isolated novelty)
        freq_loss = outputs.get("freq_loss", torch.tensor(0.0, device=pred.device))

        # Distribution regularization (true SIGReg, or VICReg ablation)
        x = all_enc.reshape(-1, all_enc.shape[-1])
        reg, reg_info = distribution_reg(x, self.reg_type)

        # Query specialization
        query_loss = self.tokenizer.get_query_specialization_loss()

        total = ((1 - self.sigreg_lambda) * pred_loss
                 + self.sigreg_lambda * reg
                 + self.freq_mask_weight * freq_loss
                 + self.query_spec_weight * query_loss)

        return {
            "total": total, "pred": pred_loss,
            **reg_info,
            "freq": freq_loss,
            "qspec": query_loss,
            "adv": torch.tensor(0.0, device=pred.device),
        }

    def update_ema(self): pass
    def set_training_progress(self, p): pass
    def initialize_electrodes(self, e): pass
