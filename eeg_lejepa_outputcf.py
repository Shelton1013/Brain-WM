"""
EEG-LeJEPA + Output-Space Cross-Frequency Prediction (Plan B).

Avoid SpectralTokenizer's bottleneck entirely: keep the proven
DynamicChannelMixer main path unchanged, and apply cross-frequency
prediction on the encoder's OUTPUT representations.

Architecture:
  EEG ─→ DynamicChannelMixer ─→ encoder ─→ encoded [B, N, D]
                                               │
                                               ├─→ Main JEPA loss
                                               │   (temporal block mask)
                                               │
                                               └─→ band_head (Linear D → 5 × d_band_view)
                                                       │
                                                       └─→ band_views [B, N, 5, d_band_view]
                                                              │
                                                              └─→ CrossFreqPredictor
                                                                    (mask 1-2 band views,
                                                                     predict from visible)
                                                                       │
                                                                       └─→ CF loss
                                                                             │
                                            (CF gradient backprops through band_head → encoder
                                             encoder learns to organize its output along
                                             implicit frequency dimensions)

vs lejepa_crossfreq (SpectralTokenizer):
  - No 26→8 or 40→32 compression
  - No "frequency-decomposition before encoder" cost
  - CF still trains the encoder (via band_head gradient)
  - Tradeoff: "bands" are learned virtual views, not physical δ/θ/α/β/γ

vs lejepa_multistream (Plan A):
  - Encoder sequence stays at N tokens (not N + N×n_bands)
  - Training cost ≈ base LeJEPA + small band_head + small CF predictor
  - Tradeoff: CF representation is at the END of encoder, not throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_jepa import TransformerBlock, DynamicChannelMixer
from regularizers import distribution_reg


class EEGLeJEPAOutputCF(nn.Module):
    """Base LeJEPA + cross-frequency prediction on encoder output.

    Main path identical to base LeJEPA (DynamicChannelMixer + encoder + MLP
    head). CF path adds a small band_head that projects each encoder-output
    token into n_bands learned "band views"; CF predictor masks 1-2 views
    and predicts them from the others. CF gradient flows back through
    band_head into the encoder.
    """

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
        freq_mask_weight: float = 1.0,
        sigreg_lambda: float = 0.05,
        query_spec_weight: float = 0.1,
        n_subjects: int = 109,
        reg_type: str = "sigreg",
        cf_band_conditioned: bool = True,
        # cf_d_band reused as "d_band_view" — the dimension of each band view
        # projection. Default 64 (4× smaller than d_model=256). Larger =
        # more expressive band views but more compute.
        cf_d_band: int = 64,
        # Present for CLI compatibility; unused (no SpectralTokenizer here):
        cf_preserve_spatial: bool = True,
        # Max number of temporal tokens for pos_embed. Default 256 supports
        # 4s @ 256Hz with state_samples=26 (N=39). Increase to ~1024 for
        # 60s+ chunks. Old checkpoints (256) still load if you do not
        # increase trial_duration_s.
        max_seq_len: int = 256,
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
        self.n_bands = n_bands
        self.cf_band_conditioned = cf_band_conditioned
        self.d_band_view = cf_d_band if cf_d_band is not None else 64
        self.max_seq_len = max_seq_len

        # ─── Main path: identical to base LeJEPA ──────────────────────
        self.tokenizer = DynamicChannelMixer(
            n_channels, state_samples, d_model, d_channel, n_queries,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, encoder_heads)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ─── CF apparatus on encoder output (NEW) ─────────────────────
        # Project encoder output to per-band views
        self.band_head = nn.Linear(d_model, n_bands * self.d_band_view)
        # Per-band identity for distinguishing views
        self.band_embed_view = nn.Parameter(
            torch.randn(n_bands, self.d_band_view) * 0.02,
        )
        # Predictor MLP for cross-band prediction
        in_dim = self.d_band_view * 2 if cf_band_conditioned else self.d_band_view
        self.cf_predictor = nn.Sequential(
            nn.Linear(in_dim, self.d_band_view * 2),
            nn.GELU(),
            nn.Linear(self.d_band_view * 2, self.d_band_view),
        )
        # Per-band mask tokens (used when band_conditioned=True)
        self.cf_band_mask_tokens = nn.Parameter(
            torch.randn(n_bands, self.d_band_view) * 0.02,
        )

    def _tokenize(self, eeg):
        tokens = self.tokenizer(eeg)
        N = tokens.shape[1]
        return tokens + self.pos_embed[:, :N, :]

    def _encode(self, tokens):
        x = tokens
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

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

        all_tokens = self._tokenize(eeg)
        all_encoded = self._encode(all_tokens)

        if not return_predictions:
            return {"brain_states": all_encoded}

        # ── Main JEPA loss: temporal block mask + predict ─────────────
        ids_vis, ids_mask, n_vis, n_mask = self._generate_block_mask(B, N, eeg.device)

        vis_encoded = torch.gather(
            all_encoded, 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        mask_encoded = torch.gather(
            all_encoded, 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, self.d_model),
        )

        vis_context = vis_encoded.mean(dim=1, keepdim=True).expand(-1, n_mask, -1)
        predictions = self.pred_head(vis_context)

        # ── CF loss: project encoded → band views → mask + predict ────
        freq_loss = self._compute_cf_loss_on_output(all_encoded)

        return {
            "predictions": predictions,
            "targets": mask_encoded,
            "all_encoded": all_encoded,
            "freq_loss": freq_loss,
            "n_vis": n_vis, "n_mask": n_mask,
            "brain_states": all_encoded,
            "subj_logits": None,
        }

    def _compute_cf_loss_on_output(self, all_encoded: torch.Tensor) -> torch.Tensor:
        """Project encoded tokens to per-band views, mask & predict.

        all_encoded: [B, N, D] (encoder output).
        Returns: scalar CF loss.
        """
        if not self.training:
            return torch.tensor(0.0, device=all_encoded.device)

        B, N, D = all_encoded.shape

        # Project to n_bands × d_band_view
        band_proj = self.band_head(all_encoded)              # [B, N, n_bands*d_v]
        band_views = band_proj.reshape(B, N, self.n_bands, self.d_band_view)
        # Add per-band identity to distinguish views
        band_views = band_views + self.band_embed_view[None, None, :, :]

        n_mask = torch.randint(1, min(3, self.n_bands), (1,)).item()
        perm = torch.randperm(self.n_bands, device=all_encoded.device)
        masked_bands = perm[:n_mask]
        visible_bands = perm[n_mask:]
        if len(visible_bands) == 0:
            return torch.tensor(0.0, device=all_encoded.device)

        visible = band_views[:, :, visible_bands, :]         # [B, N, n_vis, d_v]
        context = visible.mean(dim=2)                         # [B, N, d_v]

        total_loss = torch.tensor(0.0, device=all_encoded.device)
        for band_idx in masked_bands:
            if self.cf_band_conditioned:
                band_id = self.cf_band_mask_tokens[band_idx]
                band_id_exp = band_id.view(1, 1, -1).expand(B, N, -1)
                pred_input = torch.cat([context, band_id_exp], dim=-1)
            else:
                pred_input = context

            predicted = self.cf_predictor(pred_input)         # [B, N, d_v]
            target = band_views[:, :, band_idx, :]            # [B, N, d_v]
            # No StopGrad — LeJEPA-consistent
            total_loss = total_loss + F.mse_loss(predicted, target)

        return total_loss / n_mask

    def compute_loss(self, outputs, subject_ids=None):
        pred = outputs["predictions"]
        target = outputs["targets"]
        all_enc = outputs["all_encoded"]
        freq_loss = outputs.get(
            "freq_loss", torch.tensor(0.0, device=pred.device),
        )

        pred_loss = F.mse_loss(pred, target)

        x = all_enc.reshape(-1, all_enc.shape[-1])
        reg, reg_info = distribution_reg(x, self.reg_type)

        try:
            query_loss = self.tokenizer.get_query_specialization_loss()
        except Exception:
            query_loss = torch.tensor(0.0, device=pred.device)

        total = ((1 - self.sigreg_lambda) * pred_loss
                 + self.sigreg_lambda * reg
                 + self.freq_mask_weight * freq_loss
                 + self.query_spec_weight * query_loss)

        return {
            "total": total,
            "pred": pred_loss,
            **reg_info,
            "freq": freq_loss,
            "qspec": query_loss,
            "adv": torch.tensor(0.0, device=pred.device),
        }

    def update_ema(self): pass
    def set_training_progress(self, p): pass
    def initialize_electrodes(self, e): pass
