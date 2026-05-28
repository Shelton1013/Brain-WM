"""
EEG-LeJEPA Full: Tri-dimensional masking in LeJEPA framework.

Three orthogonal masking dimensions:
  1. Temporal block masking (which time steps) — same as Laya
  2. Region masking (which brain regions) — spatial prior
  3. Cross-frequency latent prediction (which frequency bands) — OUR NOVELTY

Cross-frequency latent prediction: mask one frequency band's latent
representation and predict it from the other bands. This is ONLY
meaningful in a latent prediction framework (JEPA), not MAE:
  - MAE: reconstruct raw signal → frequency info helps reconstruction
  - JEPA: predict latent freq representation → learns inter-band relationships
    (e.g., alpha suppression ↔ beta enhancement in motor imagery)

No predictor, no StopGrad, only SIGReg — following LeJEPA theory.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_jepa import TransformerBlock
from regularizers import distribution_reg


# ============================================================
# 1. Learnable Filter Bank
# ============================================================

class LearnableFilterBank(nn.Module):
    """5-band learnable bandpass filters: delta/theta/alpha/beta/gamma."""

    def __init__(self, n_filters=5, kernel_size=17, sample_rate=256):
        super().__init__()
        self.n_filters = n_filters
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.filters = nn.Conv1d(1, n_filters, kernel_size,
                                  padding=kernel_size // 2, bias=False)
        bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 100.0)]
        t = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        with torch.no_grad():
            for i, (lo, hi) in enumerate(bands[:n_filters]):
                lo_n, hi_n = lo / (sample_rate / 2), min(hi / (sample_rate / 2), 0.99)
                bp = hi_n * torch.sinc(hi_n * t) - lo_n * torch.sinc(lo_n * t)
                bp = bp * torch.hamming_window(kernel_size, periodic=False)
                self.filters.weight.data[i, 0, :] = bp / (bp.abs().sum() + 1e-8)

    def forward(self, x):
        """[B, T] → [B, n_filters, T]"""
        return self.filters(x.unsqueeze(1))


# ============================================================
# 2. Spectral Tokenizer with band-level outputs
# ============================================================

class SpectralTokenizer(nn.Module):
    """Frequency-aware tokenizer that preserves per-band representations.

    Output: tokens [B, N, D] AND band_tokens [B, N, n_bands, d_band]
    The band_tokens are used for cross-frequency masking/prediction.
    """

    def __init__(self, n_channels, state_samples, d_model,
                 d_channel=32, n_queries=16, n_bands=5, sample_rate=256):
        super().__init__()
        self.state_samples = state_samples
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.n_queries = n_queries
        self.d_channel = d_channel

        self.filter_bank = LearnableFilterBank(n_bands, 17, sample_rate)

        # Per-band encoder
        self.d_band = max(d_channel // n_bands, 8)
        self.band_encoder = nn.Sequential(
            nn.Linear(state_samples, self.d_band * 2),
            nn.GELU(),
            nn.Linear(self.d_band * 2, self.d_band),
            nn.LayerNorm(self.d_band),
        )
        self.band_embed = nn.Parameter(torch.randn(n_bands, self.d_band) * 0.02)

        # Project bands to channel dim
        self.band_proj = nn.Linear(n_bands * self.d_band, d_channel)
        self.band_norm = nn.LayerNorm(d_channel)

        # Channel embedding + mixer
        self.channel_embed = nn.Parameter(torch.randn(n_channels, d_channel) * 0.02)
        self.spatial_queries = nn.Parameter(torch.randn(n_queries, d_channel) * 0.02)
        self.spatial_attn = nn.MultiheadAttention(d_channel, 4, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_channel)

        self.out_proj = nn.Linear(n_queries * d_channel, d_model)
        self.out_norm = nn.LayerNorm(d_model)

        self._last_attn_weights = None

    def forward(self, eeg, return_band_tokens=False):
        """
        Returns:
            tokens: [B, N, D]
            band_tokens (optional): [B, N, n_bands, d_band] per-band representations
        """
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        # Window + filter
        windows = eeg[:, :N*S, :].reshape(B, N, S, C).permute(0, 1, 3, 2)  # [B,N,C,S]
        flat = windows.reshape(B * N * C, S)
        filtered = self.filter_bank(flat)  # [B*N*C, n_bands, S]

        # Per-band encoding
        band_flat = filtered.permute(0, 2, 1).reshape(B * N * C * self.n_bands, S)
        band_features = self.band_encoder(band_flat)  # [B*N*C*n_bands, d_band]
        band_features = band_features.reshape(B * N * C, self.n_bands, self.d_band)
        band_features = band_features + self.band_embed.unsqueeze(0)

        # Save per-band per-channel features before aggregation
        # [B*N, C, n_bands, d_band]
        band_per_channel = band_features.reshape(B * N, C, self.n_bands, self.d_band)

        # Aggregate bands → channel feature
        band_concat = band_features.reshape(B * N * C, -1)
        ch_features = self.band_norm(self.band_proj(band_concat))
        ch_features = ch_features.reshape(B * N, C, self.d_channel)
        ch_features = ch_features + self.channel_embed.unsqueeze(0)

        # Channel mixer
        queries = self.spatial_queries.unsqueeze(0).expand(B * N, -1, -1)
        pooled, attn_weights = self.spatial_attn(queries, ch_features, ch_features)
        pooled = self.spatial_norm(pooled)
        self._last_attn_weights = attn_weights

        tokens = self.out_norm(self.out_proj(pooled.reshape(B * N, -1)).reshape(B, N, -1))

        if return_band_tokens:
            # Aggregate band tokens across channels (mean over C)
            # [B*N, C, n_bands, d_band] → [B, N, n_bands, d_band]
            band_tokens = band_per_channel.mean(dim=1).reshape(B, N, self.n_bands, self.d_band)
            return tokens, band_tokens

        return tokens

    def get_query_specialization_loss(self):
        if self._last_attn_weights is None:
            return torch.tensor(0.0)
        W = F.normalize(self._last_attn_weights.mean(dim=0), dim=-1)
        sim = W @ W.T
        return sim.fill_diagonal_(0).pow(2).sum() / self.n_queries


# ============================================================
# 3. Cross-Frequency Predictor
# ============================================================

class CrossFrequencyPredictor(nn.Module):
    """Predict masked frequency band from unmasked bands in latent space.

    This is the core novelty: inter-band latent prediction.
    Only meaningful in JEPA (latent prediction), not MAE (reconstruction).

    Neuroscience motivation:
      - alpha suppression ↔ beta enhancement (motor imagery)
      - theta-gamma coupling (memory)
      - delta-beta coupling (cognitive control)
    Learning these relationships in latent space = better representations.
    """

    def __init__(self, n_bands=5, d_band=8):
        super().__init__()
        self.n_bands = n_bands
        self.d_band = d_band

        # Learnable mask token per band
        self.band_mask_tokens = nn.Parameter(torch.randn(n_bands, d_band) * 0.02)

        # Cross-band predictor: predict masked band from unmasked bands
        self.predictor = nn.Sequential(
            nn.Linear(d_band, d_band * 2),
            nn.GELU(),
            nn.Linear(d_band * 2, d_band),
        )

    def mask_and_predict(self, band_tokens, mask_prob=0.3):
        """
        Args:
            band_tokens: [B, N, n_bands, d_band]
            mask_prob: probability of masking each band
        Returns:
            loss: cross-frequency prediction loss
            n_masked: number of bands masked (for logging)
        """
        if not self.training:
            return torch.tensor(0.0, device=band_tokens.device), 0

        B, N, n_bands, d_band = band_tokens.shape

        # Randomly select 1-2 bands to mask
        n_mask = torch.randint(1, min(3, n_bands), (1,)).item()
        perm = torch.randperm(n_bands, device=band_tokens.device)
        masked_bands = perm[:n_mask]
        visible_bands = perm[n_mask:]

        if len(visible_bands) == 0:
            return torch.tensor(0.0, device=band_tokens.device), 0

        # Context: mean of visible bands
        visible = band_tokens[:, :, visible_bands, :]  # [B, N, n_vis, d]
        context = visible.mean(dim=2)  # [B, N, d]

        # Predict each masked band
        predicted = self.predictor(context)  # [B, N, d]

        # Target: original masked band representations
        total_loss = torch.tensor(0.0, device=band_tokens.device)
        for band_idx in masked_bands:
            target = band_tokens[:, :, band_idx, :].detach()  # [B, N, d]
            total_loss = total_loss + F.mse_loss(predicted, target)

        return total_loss / n_mask, n_mask


# ============================================================
# 4. Region Masker (simplified)
# ============================================================

class RegionMasker(nn.Module):
    """Mask entire brain regions in token space."""

    ELECTRODE_REGIONS = {
        "Fp1": 0, "Fp2": 0, "F3": 0, "F4": 0, "F7": 0, "F8": 0, "Fz": 0,
        "C3": 1, "C4": 1, "Cz": 1, "FC1": 1, "FC2": 1, "FC3": 1, "FC4": 1,
        "FC5": 1, "FC6": 1, "CP1": 1, "CP2": 1, "CP3": 1, "CP4": 1, "CP5": 1, "CP6": 1,
        "P3": 2, "P4": 2, "Pz": 2, "P7": 2, "P8": 2, "PO3": 2, "PO4": 2,
        "T3": 3, "T4": 3, "T5": 3, "T6": 3, "T7": 3, "T8": 3,
        "O1": 4, "O2": 4, "Oz": 4,
    }

    def __init__(self, n_regions=5, d_model=256, mask_prob=0.4):
        super().__init__()
        self.n_regions = n_regions
        self.d_per_region = d_model // n_regions
        self.mask_prob = mask_prob
        self.mask_tokens = nn.Parameter(
            torch.randn(n_regions, self.d_per_region) * 0.02
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.d_per_region, self.d_per_region * 2),
            nn.GELU(),
            nn.Linear(self.d_per_region * 2, self.d_per_region),
        )

    def forward(self, tokens):
        """Apply region masking and compute prediction loss."""
        if not self.training or torch.rand(1).item() > self.mask_prob:
            return tokens, torch.tensor(0.0, device=tokens.device)

        B, N, D = tokens.shape
        d = self.d_per_region
        R = self.n_regions

        n_mask = torch.randint(1, 3, (1,)).item()
        perm = torch.randperm(R, device=tokens.device)
        masked_idx = perm[:n_mask]
        unmasked_idx = perm[n_mask:]

        # Replace masked regions
        region_mask = torch.zeros(D, dtype=torch.bool, device=tokens.device)
        token_values = torch.zeros(D, device=tokens.device)
        for r in masked_idx:
            s, e = r.item() * d, min((r.item() + 1) * d, D)
            region_mask[s:e] = True
            token_values[s:e] = self.mask_tokens[r, :e-s]

        masked_tokens = torch.where(
            region_mask.unsqueeze(0).unsqueeze(0), token_values.unsqueeze(0).unsqueeze(0), tokens)

        # Predict masked from unmasked
        if len(unmasked_idx) > 0:
            unmasked_feats = [tokens[:, :, r.item()*d:min((r.item()+1)*d, D)] for r in unmasked_idx]
            context = torch.stack(unmasked_feats, dim=0).mean(dim=0)
            predicted = self.predictor(context)
            loss = torch.tensor(0.0, device=tokens.device)
            for r in masked_idx:
                s, e = r.item() * d, min((r.item() + 1) * d, D)
                target = tokens[:, :, s:e].detach()
                pred_slice = predicted[:, :, :e-s]
                loss = loss + F.mse_loss(pred_slice, target)
            loss = loss / n_mask
        else:
            loss = torch.tensor(0.0, device=tokens.device)

        return masked_tokens, loss


# ============================================================
# 5. Full Model: EEG-LeJEPA with Tri-dimensional Masking
# ============================================================

class EEGLeJEPAFull(nn.Module):
    """EEG-LeJEPA with tri-dimensional masking.

    Masking dimensions:
      1. Temporal: block masking of time steps (LeJEPA standard)
      2. Spatial: brain region masking (neuroscience prior)
      3. Spectral: cross-frequency latent prediction (OUR NOVELTY)

    All within the LeJEPA framework (no predictor, no StopGrad, only SIGReg).
    """

    def __init__(
        self,
        n_channels: int = 64,
        state_samples: int = 26,
        d_model: int = 256,
        d_channel: int = 32,
        n_queries: int = 16,
        n_bands: int = 5,
        n_regions: int = 5,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        mask_ratio: float = 0.60,
        mask_block_size: int = 5,
        region_mask_prob: float = 0.4,
        freq_mask_weight: float = 1.0,
        region_mask_weight: float = 1.0,
        sigreg_lambda: float = 0.05,
        query_spec_weight: float = 0.1,
        n_subjects: int = 109,
        reg_type: str = "sigreg",        # "sigreg" (true LeJEPA) | "vicreg" (ablation)
    ):
        super().__init__()
        self.state_samples = state_samples
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.sigreg_lambda = sigreg_lambda
        self.query_spec_weight = query_spec_weight
        self.freq_mask_weight = freq_mask_weight
        self.region_mask_weight = region_mask_weight
        self.reg_type = reg_type

        # Spectral tokenizer (preserves per-band representations)
        self.tokenizer = SpectralTokenizer(
            n_channels, state_samples, d_model, d_channel, n_queries, n_bands,
        )

        # Cross-frequency predictor (our novelty)
        self.freq_predictor = CrossFrequencyPredictor(
            n_bands=n_bands, d_band=self.tokenizer.d_band,
        )

        # Region masker
        self.region_masker = RegionMasker(
            n_regions=n_regions, d_model=d_model, mask_prob=region_mask_prob,
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        # Encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, encoder_heads)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Temporal prediction head (lightweight MLP)
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
        """For eval: just return tokens without band info."""
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

        # === Spectral tokenization (get both tokens and band-level features) ===
        if self.training and return_predictions:
            tokens, band_tokens = self.tokenizer(eeg, return_band_tokens=True)
        else:
            tokens = self.tokenizer(eeg, return_band_tokens=False)
            band_tokens = None

        # Add position embedding
        tokens = tokens + self.pos_embed[:, :N, :]

        # === Dimension 3: Cross-frequency masking (before encoder) ===
        freq_loss = torch.tensor(0.0, device=eeg.device)
        if band_tokens is not None:
            freq_loss, _ = self.freq_predictor.mask_and_predict(band_tokens)

        # === Dimension 2: Region masking (before encoder) ===
        region_loss = torch.tensor(0.0, device=eeg.device)
        if self.training and return_predictions:
            tokens, region_loss = self.region_masker(tokens)

        # === Encoder ===
        all_encoded = self._encode(tokens)

        if not return_predictions:
            return {"brain_states": all_encoded}

        # === Dimension 1: Temporal block masking (after encoder) ===
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
            "region_loss": region_loss,
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

        # Cross-frequency loss (our novelty)
        freq_loss = outputs.get("freq_loss", torch.tensor(0.0, device=pred.device))

        # Region masking loss
        region_loss = outputs.get("region_loss", torch.tensor(0.0, device=pred.device))

        # Distribution regularization (true SIGReg, or VICReg ablation)
        x = all_enc.reshape(-1, all_enc.shape[-1])
        reg, reg_info = distribution_reg(x, self.reg_type)

        # Query specialization
        query_loss = self.tokenizer.get_query_specialization_loss()

        total = ((1 - self.sigreg_lambda) * pred_loss
                 + self.sigreg_lambda * reg
                 + self.freq_mask_weight * freq_loss
                 + self.region_mask_weight * region_loss
                 + self.query_spec_weight * query_loss)

        return {
            "total": total, "pred": pred_loss,
            **reg_info,
            "freq": freq_loss, "rmask": region_loss,
            "qspec": query_loss,
            "adv": torch.tensor(0.0, device=pred.device),
        }

    def update_ema(self): pass
    def set_training_progress(self, p): pass
    def initialize_electrodes(self, e): pass
