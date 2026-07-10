"""EEG-LeJEPA v3 — frequency-native pretraining.

Motivation (see design doc / memory frozen-probe-consolidated):
v2's continuous-MSE targets (latent JEPA + raw-waveform MAE) are easy to satisfy
without learning the discriminative *spectral* structure that separates EEG
classes — which is why v2 barely beats a random encoder. v3 makes frequency an
architectural prior:

  1. Learnable FILTERBANK tokenizer: each patch is decomposed into N frequency
     bands by a learnable Conv1d bank (physiologically initialised δ/θ/α/β/γ),
     so the encoder input already carries band structure.
  2. Real CROSS-FREQUENCY masked prediction: random bands are masked at the
     tokenizer (removed from the encoder input); the model predicts the masked
     band's log-power from the *other* bands + spatio-temporal context. No
     leakage (target band excluded), a hard spectral target, genuine cross-band
     coupling — this is the primary objective.
  3. Cross-band JEPA (latent consistency): predict the full-band encoding from
     the band-masked encoding (dense latent MSE). Secondary.
  4. SIGReg anti-collapse (unchanged).
  MAE (raw-waveform reconstruction) is DROPPED — it conflicted with JEPA and
  reconstructed noise.

Downstream interface (`_tokenize` / `_encode` / `d_model`) matches v2 so the
existing eval code works unchanged.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_lejepa_v2 import CrissCrossBlock          # reuse criss-cross encoder
from regularizers import distribution_reg

# Physiological band centres (Hz) used to initialise the filterbank.
_BAND_CENTERS_HZ = [2.0, 6.0, 10.0, 21.0, 38.0]    # δ θ α β γ


class FilterbankTokenizer(nn.Module):
    """Raw EEG → per-band patch embeddings + per-band log-power targets.

    Input : eeg [B, T, C]
    Output:
        band_emb   [B, C, Tp, N, d_band]   per-band patch embeddings
        band_logpow[B, C, Tp, N]           log(1+power) target per band/patch
        token      [B, C, Tp, D]           combined (all-band) encoder token (no pos)
    """

    def __init__(self, patch_len: int, d_model: int, n_bands: int = 5,
                 d_band: int = 64, filt_kernel: int = 65, sample_rate: int = 256):
        super().__init__()
        self.patch_len = patch_len
        self.n_bands = n_bands
        self.d_band = d_band
        # depthwise-over-band conv: 1 -> N filters, applied per channel
        self.filters = nn.Conv1d(1, n_bands, kernel_size=filt_kernel,
                                 padding=filt_kernel // 2, bias=False)
        self._init_bandpass(filt_kernel, sample_rate)
        self.band_proj = nn.Linear(patch_len, d_band)       # per-band patch -> d_band
        self.combine = nn.Linear(n_bands * d_band, d_model)  # bands -> token
        self.norm = nn.LayerNorm(d_model)

    def _init_bandpass(self, K: int, fs: int):
        """Init each filter as a Hann-windowed sinusoid at its band centre so the
        bank starts frequency-selective (still fully learnable)."""
        n = torch.arange(K) - K // 2
        win = torch.hann_window(K, periodic=False)
        with torch.no_grad():
            for b in range(self.n_bands):
                fc = _BAND_CENTERS_HZ[b % len(_BAND_CENTERS_HZ)]
                kernel = win * torch.cos(2 * math.pi * fc * n / fs)
                kernel = kernel - kernel.mean()               # zero-DC
                kernel = kernel / (kernel.norm() + 1e-6)
                self.filters.weight[b, 0].copy_(kernel)

    def forward(self, eeg: torch.Tensor):
        B, T, C = eeg.shape
        Tp = T // self.patch_len
        T_use = Tp * self.patch_len
        x = eeg[:, :T_use, :].permute(0, 2, 1).reshape(B * C, 1, T_use)  # [B*C,1,T]
        bands = self.filters(x)                                # [B*C, N, T_use]
        bands = bands.reshape(B, C, self.n_bands, Tp, self.patch_len)   # [B,C,N,Tp,P]
        # per-band log-power target (from the real band-filtered patch)
        logpow = torch.log1p((bands ** 2).mean(dim=-1))        # [B,C,N,Tp]
        band_logpow = logpow.permute(0, 1, 3, 2).contiguous()  # [B,C,Tp,N]
        # per-band patch embedding
        band_emb = self.band_proj(bands)                       # [B,C,N,Tp,d_band]
        band_emb = band_emb.permute(0, 1, 3, 2, 4).contiguous()  # [B,C,Tp,N,d_band]
        token = self._combine(band_emb)                        # [B,C,Tp,D]
        return band_emb, band_logpow, token

    def _combine(self, band_emb: torch.Tensor) -> torch.Tensor:
        B, C, Tp, N, d = band_emb.shape
        tok = self.combine(band_emb.reshape(B, C, Tp, N * d))
        return self.norm(tok)


class EEGLeJEPA_v3(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        encoder_layers: int = 12,
        n_heads: int = 8,
        patch_len: int = 200,
        max_time_patches: int = 64,
        max_channels: int = 32,
        n_bands: int = 5,
        d_band: int = 64,
        filt_kernel: int = 65,
        sample_rate: int = 256,
        band_mask_ratio: float = 0.30,
        jepa_weight: float = 0.3,
        cf_weight: float = 1.0,
        sigreg_lambda: float = 0.05,
        reg_type: str = "sigreg",
        **_ignored,          # tolerate extra ckpt args
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.n_bands = n_bands
        self.d_band = d_band
        self.band_mask_ratio = band_mask_ratio
        self.jepa_weight = jepa_weight
        self.cf_weight = cf_weight
        self.sigreg_lambda = sigreg_lambda
        self.reg_type = reg_type

        self.tokenizer = FilterbankTokenizer(
            patch_len, d_model, n_bands, d_band, filt_kernel, sample_rate)

        self.pos_time = nn.Parameter(torch.zeros(1, 1, max_time_patches, d_model))
        self.pos_channel = nn.Parameter(torch.zeros(1, max_channels, 1, d_model))
        nn.init.normal_(self.pos_time, std=0.02)
        nn.init.normal_(self.pos_channel, std=0.02)

        self.encoder_blocks = nn.ModuleList(
            [CrissCrossBlock(d_model, n_heads) for _ in range(encoder_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)

        # learnable per-band mask token (replaces a masked band's embedding)
        self.cf_mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, d_band))
        nn.init.normal_(self.cf_mask_token, std=0.02)

        # heads
        self.jepa_predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.cf_head = nn.Linear(d_model, n_bands)   # predict per-band log-power

        self._cached_enc_4d = None

    # ── encoder plumbing ────────────────────────────────────────────
    def _add_pos(self, tok: torch.Tensor) -> torch.Tensor:
        B, C, Tp, D = tok.shape
        C = min(C, self.pos_channel.shape[1]); Tp = min(Tp, self.pos_time.shape[2])
        tok = tok[:, :C, :Tp, :]
        return tok + self.pos_time[:, :, :Tp, :] + self.pos_channel[:, :C, :, :]

    def _encode_4d(self, tok: torch.Tensor) -> torch.Tensor:
        x = tok
        for blk in self.encoder_blocks:
            x = blk(x)
        return self.encoder_norm(x)

    # ── SSL forward ─────────────────────────────────────────────────
    def forward(self, eeg: torch.Tensor, subject_ids=None):
        band_emb, band_logpow, _ = self.tokenizer(eeg)     # [B,C,Tp,N,d], [B,C,Tp,N]
        B, C, Tp, N, d = band_emb.shape
        Cc = min(C, self.pos_channel.shape[1]); Tpp = min(Tp, self.pos_time.shape[2])
        band_emb = band_emb[:, :Cc, :Tpp]
        band_logpow = band_logpow[:, :Cc, :Tpp]
        B, C, Tp, N, d = band_emb.shape

        # band mask: [B,C,Tp,N] True = masked (predict it)
        bmask = torch.rand(B, C, Tp, N, device=eeg.device) < self.band_mask_ratio
        # guarantee ≥1 visible band per position: force band 0 visible where all masked
        all_masked = bmask.all(dim=-1)                        # [B,C,Tp]
        bmask[..., 0] = bmask[..., 0] & ~all_masked

        mask_tok = self.cf_mask_token.expand(B, C, Tp, N, d)
        band_ctx = torch.where(bmask.unsqueeze(-1), mask_tok, band_emb)
        tok_ctx = self.tokenizer._combine(band_ctx)         # [B,C,Tp,D]
        enc_ctx = self._encode_4d(self._add_pos(tok_ctx))    # [B,C,Tp,D]

        # ─ Cross-frequency spectral prediction (primary) ─
        cf_pred = self.cf_head(enc_ctx)                      # [B,C,Tp,N]
        if bmask.any():
            loss_cf = F.mse_loss(cf_pred[bmask], band_logpow[bmask])
        else:
            loss_cf = enc_ctx.sum() * 0.0

        # ─ Cross-band JEPA (dense latent consistency) ─
        with torch.no_grad():
            tok_all = self.tokenizer._combine(band_emb)
            enc_tgt = self._encode_4d(self._add_pos(tok_all))
        loss_jepa = F.mse_loss(self.jepa_predictor(enc_ctx), enc_tgt.detach())

        # ─ SIGReg anti-collapse ─
        reg, _ = distribution_reg(enc_ctx.reshape(-1, self.d_model), self.reg_type)

        total = (self.cf_weight * loss_cf
                 + self.jepa_weight * loss_jepa
                 + self.sigreg_lambda * reg)
        return {
            "total": total,
            "cf": loss_cf.detach(),
            "jepa": loss_jepa.detach(),
            "sig": reg.detach() if isinstance(reg, torch.Tensor) else torch.tensor(reg),
            "mae": torch.tensor(0.0, device=eeg.device),
            "pajr": torch.tensor(0.0, device=eeg.device),
        }

    # ── downstream interface (matches v2) ───────────────────────────
    def _tokenize(self, eeg: torch.Tensor) -> torch.Tensor:
        band_emb, _, _ = self.tokenizer(eeg)
        tok = self.tokenizer._combine(band_emb)
        enc = self._encode_4d(self._add_pos(tok))
        self._cached_enc_4d = enc
        B, C, Tp, D = enc.shape
        return enc.reshape(B, C * Tp, D)

    def _encode(self, tokens_flat: torch.Tensor) -> torch.Tensor:
        if self._cached_enc_4d is not None:
            enc = self._cached_enc_4d
            self._cached_enc_4d = None
            B, C, Tp, D = enc.shape
            return enc.reshape(B, C * Tp, D)
        return tokens_flat


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
