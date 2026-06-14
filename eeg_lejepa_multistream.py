"""
EEG-LeJEPA Multi-Stream JEPA — avoid SpectralTokenizer's compression cost
by running encoder on BOTH main temporal tokens AND per-band tokens jointly.

Key idea:
  - Main tokenizer: proven DynamicChannelMixer (no frequency-decomp cost)
  - Band tokenizer: parallel LearnableFilterBank + shared per-band tokenizer
    that produces N × n_bands tokens, each of FULL d_model dimension
    (no d_band=8 / d_band=32 bottleneck)
  - Both streams concatenated in sequence dimension, processed by the
    SAME encoder
  - Main JEPA loss masks temporal positions
  - CF loss masks band positions (per time step)
  - Encoder receives gradients from both losses via shared weights

Compared to SpectralTokenizer-based crossfreq variants:
  - No 26 → 8 compression
  - No 40 → 32 compression
  - Channel mixing happens INSIDE each band (preserved per-band spatial structure)
  - CF directly trains encoder via shared encoder gradients
  - Cost: encoder sequence length is N + N×n_bands (6× longer with n_bands=5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_jepa import TransformerBlock, DynamicChannelMixer
from eeg_lejepa_full import LearnableFilterBank
from regularizers import distribution_reg


# ============================================================
# Band tokenizer: produces [B, N, n_bands, D] per-band-per-time tokens
# ============================================================

class BandTokenizer(nn.Module):
    """Per-band temporal tokens, one full d_model vector per (time, band).

    Pipeline (for each frequency band, in parallel):
      1. Bandpass filter (shared LearnableFilterBank)
      2. Per-channel temporal MLP encode (shared across bands)
      3. Channel mixer attention (shared queries across bands, run per band)
      4. Project to d_model
      5. Add per-band identity embedding

    Output shape: [B, N, n_bands, d_model] — each (time, band) is a full token.
    No compression bottleneck (each band gets full d_model dim).
    """

    def __init__(self, n_channels: int, state_samples: int, d_model: int,
                 d_channel: int = 32, n_queries: int = 16,
                 n_bands: int = 5, sample_rate: int = 256):
        super().__init__()
        self.state_samples = state_samples
        self.n_bands = n_bands
        self.n_queries = n_queries
        self.d_channel = d_channel

        # Shared filter bank (5 learnable bandpass filters)
        self.filter_bank = LearnableFilterBank(n_bands, 17, sample_rate)

        # Per-channel temporal encoder (shared across all bands)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(state_samples, d_channel * 2),
            nn.GELU(),
            nn.Linear(d_channel * 2, d_channel),
            nn.LayerNorm(d_channel),
        )

        # Channel embedding (per electrode, shared across bands)
        self.channel_embed = nn.Parameter(torch.randn(n_channels, d_channel) * 0.02)

        # Multi-query cross-attention (shared queries, run per band)
        self.spatial_queries = nn.Parameter(torch.randn(n_queries, d_channel) * 0.02)
        self.spatial_attn = nn.MultiheadAttention(
            d_channel, num_heads=4, batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(d_channel)

        # Project pooled queries → d_model
        self.out_proj = nn.Linear(n_queries * d_channel, d_model)
        self.out_norm = nn.LayerNorm(d_model)

        # Per-band identity embedding (added after token formation)
        self.band_embed = nn.Parameter(torch.randn(n_bands, d_model) * 0.02)

        self._last_attn_weights = None

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """[B, T, C] → [B, N, n_bands, D]"""
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        # Window: [B, T, C] → [B, N, S, C] → [B, N, C, S]
        windows = eeg[:, :N*S, :].reshape(B, N, S, C).permute(0, 1, 3, 2)
        flat = windows.reshape(B * N * C, S)  # [B*N*C, S]

        # Filter bank: [B*N*C, S] → [B*N*C, n_bands, S]
        filtered = self.filter_bank(flat)

        # Per-band temporal encoding (shared encoder)
        # Reshape to apply encoder per (B,N,C,band) → encode each 26-sample window
        band_flat = filtered.reshape(B * N * C * self.n_bands, S)
        ch_features = self.temporal_encoder(band_flat)  # [B*N*C*n_bands, d_channel]
        ch_features = ch_features.reshape(B * N, C, self.n_bands, self.d_channel)

        # Add channel embedding (broadcast across bands)
        ch_features = ch_features + self.channel_embed[None, :, None, :]

        # Channel mixer PER BAND: for each (B*N, band), mix C channels
        # [B*N, C, n_bands, d_channel] → [B*N, n_bands, C, d_channel]
        ch_features = ch_features.permute(0, 2, 1, 3).contiguous()
        # → [B*N*n_bands, C, d_channel]
        ch_features = ch_features.reshape(B * N * self.n_bands, C, self.d_channel)

        # Multi-query cross-attention
        queries = self.spatial_queries.unsqueeze(0).expand(B * N * self.n_bands, -1, -1)
        pooled, attn_weights = self.spatial_attn(queries, ch_features, ch_features)
        pooled = self.spatial_norm(pooled)
        self._last_attn_weights = attn_weights

        # Project to d_model
        pooled_flat = pooled.reshape(B * N * self.n_bands, -1)
        tokens = self.out_norm(self.out_proj(pooled_flat))
        # [B*N*n_bands, D] → [B, N, n_bands, D]
        tokens = tokens.reshape(B, N, self.n_bands, -1)

        # Add band identity
        tokens = tokens + self.band_embed[None, None, :, :]

        return tokens

    def get_query_specialization_loss(self) -> torch.Tensor:
        if self._last_attn_weights is None:
            return torch.tensor(0.0)
        W = F.normalize(self._last_attn_weights.mean(dim=0), dim=-1)
        sim = W @ W.T
        return sim.fill_diagonal_(0).pow(2).sum() / self.n_queries


# ============================================================
# Main model: Multi-Stream JEPA
# ============================================================

class EEGLeJEPAMultiStream(nn.Module):
    """JEPA with parallel main + per-band streams, processed by shared encoder.

    Architecture flow:
      EEG ─┬─ DynamicChannelMixer ──→ main_tokens [B, N, D]
           │                              │
           └─ BandTokenizer ──→ band_tokens [B, N, n_bands, D]
                                      │
      Concatenate in sequence dim: [main; band_flat] of length N + N*n_bands
                                      ↓
      Single Transformer Encoder (shared) → encoded [B, N + N*n_bands, D]
                                      ↓
        ┌── main_encoded [B, N, D] ───→ Main JEPA loss (temporal block mask)
        └── band_encoded [B, N, n_bands, D] ──→ CF loss (band mask + predict)
                                      ↓
                              VICReg/SIGReg on main_encoded (anti-collapse)
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
        # The following are present for CLI compatibility with crossfreq variants
        # but have no effect in this architecture (band tokens are inherently
        # full-D and per-channel-then-mixed):
        cf_preserve_spatial: bool = True,
        cf_d_band: int = None,
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

        # Main tokenizer: proven DynamicChannelMixer (no frequency decomp)
        self.main_tokenizer = DynamicChannelMixer(
            n_channels, state_samples, d_model, d_channel, n_queries,
        )

        # Band tokenizer: parallel filter-bank + per-band encoder
        self.band_tokenizer = BandTokenizer(
            n_channels, state_samples, d_model, d_channel, n_queries,
            n_bands, sample_rate=256,
        )

        # Position embeddings (separate for main vs band streams)
        self.pos_embed_main = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        self.pos_embed_band = nn.Parameter(torch.randn(1, 256 * n_bands, d_model) * 0.02)

        # Shared encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, encoder_heads)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Main JEPA prediction head (lightweight MLP)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Cross-frequency predictor on D-dim band-encoded features
        # band-conditioned: concat context + per-band identity for prediction
        in_dim = d_model * 2 if cf_band_conditioned else d_model
        self.cf_predictor = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        # Per-band mask tokens (used when band_conditioned=True)
        self.cf_band_mask_tokens = nn.Parameter(torch.randn(n_bands, d_model) * 0.02)

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def _tokenize(self, eeg: torch.Tensor) -> torch.Tensor:
        """For eval: produce main tokens only (no band stream)."""
        main_tokens = self.main_tokenizer(eeg)
        N = main_tokens.shape[1]
        return main_tokens + self.pos_embed_main[:, :N, :]

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

        # Main tokens (always computed)
        main_tokens = self.main_tokenizer(eeg)  # [B, N, D]
        main_tokens = main_tokens + self.pos_embed_main[:, :N, :]

        # Eval mode: skip band stream entirely for efficiency
        if not (return_predictions and self.training):
            main_encoded = self._encode(main_tokens)
            return {"brain_states": main_encoded}

        # Training: compute band tokens and concatenate
        band_tokens = self.band_tokenizer(eeg)  # [B, N, n_bands, D]
        N_band = N * self.n_bands
        band_tokens_flat = band_tokens.reshape(B, N_band, self.d_model)
        band_tokens_flat = band_tokens_flat + self.pos_embed_band[:, :N_band, :]

        # Concatenate streams: [main tokens; band tokens]
        combined = torch.cat([main_tokens, band_tokens_flat], dim=1)  # [B, N + N_band, D]

        # Shared encoder
        encoded = self._encode(combined)
        main_encoded = encoded[:, :N]                                  # [B, N, D]
        band_encoded = encoded[:, N:].reshape(B, N, self.n_bands, self.d_model)

        # ── Main JEPA loss: temporal block masking on main_encoded ───
        ids_vis, ids_mask, n_vis, n_mask = self._generate_block_mask(B, N, eeg.device)

        vis_main = torch.gather(
            main_encoded, 1,
            ids_vis.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        mask_main = torch.gather(
            main_encoded, 1,
            ids_mask.unsqueeze(-1).expand(-1, -1, self.d_model),
        )

        # Simple prediction: mean of visible → MLP head → predict masked
        vis_context = vis_main.mean(dim=1, keepdim=True).expand(-1, n_mask, -1)
        predictions = self.pred_head(vis_context)

        # ── Cross-frequency loss: band masking on band_encoded ────────
        freq_loss = self._compute_cf_loss(band_encoded)

        return {
            "predictions": predictions,
            "targets": mask_main,
            "all_encoded": main_encoded,   # for VICReg/SIGReg
            "freq_loss": freq_loss,
            "n_vis": n_vis, "n_mask": n_mask,
            "brain_states": main_encoded,
            "subj_logits": None,
        }

    def _compute_cf_loss(self, band_encoded: torch.Tensor) -> torch.Tensor:
        """Mask 1-2 bands, predict from visible bands' mean (per time position).

        band_encoded: [B, N, n_bands, D]
        """
        B, N, n_bands, D = band_encoded.shape

        n_mask = torch.randint(1, min(3, n_bands), (1,)).item()
        perm = torch.randperm(n_bands, device=band_encoded.device)
        masked_bands = perm[:n_mask]
        visible_bands = perm[n_mask:]
        if len(visible_bands) == 0:
            return torch.tensor(0.0, device=band_encoded.device)

        # Context: mean of visible bands (per (B, N) position)
        visible = band_encoded[:, :, visible_bands, :]  # [B, N, n_vis, D]
        context = visible.mean(dim=2)                    # [B, N, D]

        total_loss = torch.tensor(0.0, device=band_encoded.device)
        for band_idx in masked_bands:
            if self.cf_band_conditioned:
                band_id = self.cf_band_mask_tokens[band_idx]    # [D]
                band_id_exp = band_id.view(1, 1, -1).expand(B, N, -1)
                pred_input = torch.cat([context, band_id_exp], dim=-1)
            else:
                pred_input = context

            predicted = self.cf_predictor(pred_input)            # [B, N, D]
            target = band_encoded[:, :, band_idx, :]             # [B, N, D]
            # No StopGrad on target — LeJEPA-consistent
            total_loss = total_loss + F.mse_loss(predicted, target)

        return total_loss / n_mask

    def compute_loss(self, outputs, subject_ids=None):
        pred = outputs["predictions"]
        target = outputs["targets"]
        all_enc = outputs["all_encoded"]
        freq_loss = outputs.get(
            "freq_loss", torch.tensor(0.0, device=pred.device),
        )

        # Main JEPA prediction loss (no StopGrad)
        pred_loss = F.mse_loss(pred, target)

        # Anti-collapse regularization on main encoded path
        x = all_enc.reshape(-1, all_enc.shape[-1])
        reg, reg_info = distribution_reg(x, self.reg_type)

        # Query specialization (from main tokenizer, lightweight)
        try:
            query_loss = self.main_tokenizer.get_query_specialization_loss()
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
