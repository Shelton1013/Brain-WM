"""
EEG-LeJEPA + Spectral: LeJEPA with frequency-aware tokenizer.

Replaces the raw MLP temporal encoder with LearnableFilterBank:
  raw EEG window → 5 frequency bands (delta/theta/alpha/beta/gamma)
  → per-band encoding → concat → channel mixer

This explicitly models EEG's frequency structure, which is the
foundation of all EEG analysis (ERD/ERS, spectral power, coherence).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_jepa import TransformerBlock, DynamicChannelMixer


# ============================================================
# Learnable Filter Bank (from BrainWM, standalone version)
# ============================================================

class LearnableFilterBank(nn.Module):
    """Learnable 1D conv filters initialized from classical EEG bands.

    5 bands: delta(0.5-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-100)
    Filters are initialized as bandpass but learnable during training.
    """

    def __init__(self, n_filters: int = 5, kernel_size: int = 17, sample_rate: int = 256):
        super().__init__()
        self.n_filters = n_filters
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.filters = nn.Conv1d(1, n_filters, kernel_size,
                                  padding=kernel_size // 2, bias=False)
        self._init_bandpass(sample_rate, kernel_size)

    def _init_bandpass(self, fs, ks):
        bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 100.0)]
        t = torch.arange(ks, dtype=torch.float32) - ks // 2
        with torch.no_grad():
            for i, (lo, hi) in enumerate(bands):
                if i >= self.n_filters:
                    break
                lo_n = lo / (fs / 2)
                hi_n = min(hi / (fs / 2), 0.99)
                bp = hi_n * torch.sinc(hi_n * t) - lo_n * torch.sinc(lo_n * t)
                bp = bp * torch.hamming_window(ks, periodic=False)
                bp = bp / (bp.abs().sum() + 1e-8)
                self.filters.weight.data[i, 0, :] = bp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] → [B, n_filters, T]"""
        return self.filters(x.unsqueeze(1))


# ============================================================
# Spectral Channel Mixer
# ============================================================

class SpectralChannelMixer(nn.Module):
    """Frequency-aware tokenizer: FilterBank → per-band encoding → channel mixer.

    Unlike DynamicChannelMixer which encodes raw time windows,
    this first decomposes into frequency bands, then encodes
    each band separately, preserving spectral structure.

    Pipeline:
      [B, T, C] → [B, N, C, n_bands, S] → FilterBank per channel
      → per-band temporal encoding → concat bands
      → channel embedding → multi-query cross-attention → token
    """

    def __init__(self, n_channels: int, state_samples: int, d_model: int,
                 d_channel: int = 32, n_queries: int = 16,
                 n_bands: int = 5, sample_rate: int = 256):
        super().__init__()
        self.state_samples = state_samples
        self.n_channels = n_channels
        self.d_channel = d_channel
        self.n_queries = n_queries
        self.n_bands = n_bands

        # Learnable filter bank (shared across channels)
        self.filter_bank = LearnableFilterBank(n_bands, kernel_size=17,
                                                sample_rate=sample_rate)

        # Per-band temporal encoder: [state_samples] → [d_band]
        d_band = d_channel // n_bands  # split d_channel across bands
        self.d_band = max(d_band, 4)
        self.band_encoder = nn.Sequential(
            nn.Linear(state_samples, self.d_band * 2),
            nn.GELU(),
            nn.Linear(self.d_band * 2, self.d_band),
            nn.LayerNorm(self.d_band),
        )

        # Band embedding (which frequency band)
        self.band_embed = nn.Parameter(torch.randn(n_bands, self.d_band) * 0.02)

        # Project concat bands to d_channel
        self.band_proj = nn.Linear(n_bands * self.d_band, d_channel)
        self.band_norm = nn.LayerNorm(d_channel)

        # Channel embedding
        self.channel_embed = nn.Parameter(torch.randn(n_channels, d_channel) * 0.02)

        # Multi-query cross-attention (same as DynamicChannelMixer)
        self.spatial_queries = nn.Parameter(torch.randn(n_queries, d_channel) * 0.02)
        self.spatial_attn = nn.MultiheadAttention(d_channel, num_heads=4,
                                                    batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_channel)

        # Output projection
        self.out_proj = nn.Linear(n_queries * d_channel, d_model)
        self.out_norm = nn.LayerNorm(d_model)

        self._last_attn_weights = None

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """[B, T, C] → [B, N, D]"""
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        # Window: [B, N, S, C] → [B, N, C, S]
        windows = eeg[:, :N*S, :].reshape(B, N, S, C).permute(0, 1, 3, 2)

        # Apply filter bank per channel: [B*N*C, S] → [B*N*C, n_bands, S]
        flat = windows.reshape(B * N * C, S)
        filtered = self.filter_bank(flat)  # [B*N*C, n_bands, S]

        # Per-band encoding: [B*N*C, n_bands, S] → [B*N*C, n_bands, d_band]
        filtered = filtered.transpose(1, 2).reshape(B * N * C * self.n_bands, S)
        band_features = self.band_encoder(filtered)  # [B*N*C*n_bands, d_band]
        band_features = band_features.reshape(B * N * C, self.n_bands, self.d_band)

        # Add band embedding
        band_features = band_features + self.band_embed.unsqueeze(0)

        # Concat bands: [B*N*C, n_bands * d_band] → [B*N*C, d_channel]
        band_concat = band_features.reshape(B * N * C, -1)
        ch_features = self.band_norm(self.band_proj(band_concat))
        ch_features = ch_features.reshape(B * N, C, self.d_channel)

        # Add channel embedding
        ch_features = ch_features + self.channel_embed.unsqueeze(0)

        # Multi-query cross-attention
        queries = self.spatial_queries.unsqueeze(0).expand(B * N, -1, -1)
        pooled, attn_weights = self.spatial_attn(queries, ch_features, ch_features)
        pooled = self.spatial_norm(pooled)
        self._last_attn_weights = attn_weights

        # Project to d_model
        pooled_flat = pooled.reshape(B * N, -1)
        tokens = self.out_norm(self.out_proj(pooled_flat).reshape(B, N, -1))
        return tokens

    def get_query_specialization_loss(self) -> torch.Tensor:
        if self._last_attn_weights is None:
            return torch.tensor(0.0)
        W = self._last_attn_weights.mean(dim=0)
        W = F.normalize(W, dim=-1)
        similarity = W @ W.T
        return similarity.fill_diagonal_(0).pow(2).sum() / self.n_queries


# ============================================================
# EEG-LeJEPA + Spectral
# ============================================================

class EEGLeJEPASpectral(nn.Module):
    """LeJEPA with frequency-aware tokenizer."""

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
        sigreg_lambda: float = 0.05,
        query_spec_weight: float = 0.1,
        n_subjects: int = 109,
    ):
        super().__init__()
        self.state_samples = state_samples
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.sigreg_lambda = sigreg_lambda
        self.query_spec_weight = query_spec_weight

        # Spectral tokenizer (frequency-aware)
        self.tokenizer = SpectralChannelMixer(
            n_channels, state_samples, d_model, d_channel, n_queries, n_bands,
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

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
            "n_vis": n_vis, "n_mask": n_mask,
            "brain_states": all_encoded,
            "subj_logits": None,
        }

    def compute_loss(self, outputs, subject_ids=None):
        pred = outputs["predictions"]
        target = outputs["targets"]
        all_enc = outputs["all_encoded"]

        pred_loss = F.mse_loss(pred, target)

        B, N, D = all_enc.shape
        x = all_enc.reshape(-1, D)
        var_loss = F.relu(1.0 - x.std(dim=0)).mean()
        x_c = x - x.mean(dim=0, keepdim=True)
        cov = (x_c.T @ x_c) / max(x.shape[0]-1, 1)
        cov_loss = cov.fill_diagonal_(0).pow(2).sum() / D

        query_loss = self.tokenizer.get_query_specialization_loss()

        total = ((1 - self.sigreg_lambda) * pred_loss
                 + self.sigreg_lambda * (var_loss + cov_loss)
                 + self.query_spec_weight * query_loss)

        return {"total": total, "pred": pred_loss, "var": var_loss,
                "cov": cov_loss, "qspec": query_loss,
                "adv": torch.tensor(0.0, device=pred.device)}

    def update_ema(self): pass
    def set_training_progress(self, p): pass
    def initialize_electrodes(self, e): pass
