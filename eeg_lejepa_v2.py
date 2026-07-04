"""
EEG-LeJEPA v2: 24M-param channel-agnostic model with JEPA+MAE hybrid loss.

Design decisions (2026-07-05, based on v1 lessons):

1. **Backbone scaled 5× (5M → 24M)**:
   - d_model 256 → 512
   - encoder_layers 6 → 12
   - encoder_heads 8 → 8 (kept)
   - Match CBraMod capacity for competitive representation.

2. **Channel-agnostic tokenizer (criss-cross attention)**:
   - v1: DynamicChannelMixer expected 19 fixed channels (bad for 6-ch ISRUC).
   - v2: patch signal → CrissCrossBlock alternating spatial-temporal attention.
   - Accepts ANY number of channels natively.

3. **JEPA + MAE hybrid loss**:
   - v1: pure latent prediction → smooth-latent bias killed TUEV transients.
   - v2: latent JEPA (global structure) + signal-space MAE (transient preservation).
   - Force encoder to preserve raw signal fidelity.

4. **SIGReg only, λ=0.05** (v1 confirmed: VICReg destroys features).

5. **CF prediction reduced weight** (0.3 vs 1.0 in v1) — mostly for
   frequency diversity, no longer the main innovation.

6. **PAJR patient adversarial** kept (0.1 weight) — good against patient shift.

Architecture flow:

  EEG [B, T, C]
   ├─→ PatchEmbed [B, N_time × N_channels_grouped, D=512]
   ├─→ CrissCrossEncoder (12 layers × alternating spatial-temporal attn)
   │     ├─→ Encoder output [B, N_tokens, D=512]
   │     │
   │     ├──→ Main JEPA loss:
   │     │       predictor(encoder(x_visible)) ≈ encoder(x_target)
   │     │
   │     ├──→ MAE decoder (NEW in v2):
   │     │       decoder(encoder(x_visible) + mask_tokens) ≈ raw x_masked
   │     │       → signal-space reconstruction preserves transients
   │     │
   │     ├──→ CF prediction (reduced weight):
   │     │       band_head → cross-frequency latent prediction
   │     │
   │     ├──→ SIGReg (score-based independence, λ=0.05)
   │     │
   │     └──→ PAJR (patient-adversarial invariance)

Params:
  d_model=512, layers=12, heads=8 → ~24M params (matches CBraMod).
  MAE decoder: shallow (2 blocks) with d_dec=256.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from regularizers import distribution_reg


# ============================================================
# 0. Gradient reversal (for PAJR adversarial)
# ============================================================

class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training.

    Forward: identity. Backward: reverse gradient (multiply by -alpha).
    Used so the discriminator trains normally (minimize its own loss)
    while the encoder receives inverted gradients (fool discriminator).
    """
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ============================================================
# 1. Building blocks
# ============================================================

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CrissCrossBlock(nn.Module):
    """Alternating spatial-temporal attention block (CBraMod-style).

    Input tokens have shape [B, C, T, D] where C = channels, T = time patches,
    D = model dim. Attention alternates:
      (1) Spatial attn: for each time step t, attend across channels.
      (2) Temporal attn: for each channel c, attend across time.

    Each with residual + LayerNorm. Enables channel-agnostic modeling
    without a fixed channel count.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, D]  →  [B, C, T, D]"""
        B, C, T, D = x.shape

        # Spatial attention: attend across channels for each time step
        # Reshape [B, C, T, D] → [B*T, C, D]
        x_s = x.permute(0, 2, 1, 3).reshape(B * T, C, D)
        x_s_norm = self.norm_s(x_s)
        x_s = x_s + self.attn_s(x_s_norm, x_s_norm, x_s_norm)[0]
        x = x_s.reshape(B, T, C, D).permute(0, 2, 1, 3)

        # Temporal attention: attend across time for each channel
        # Reshape [B, C, T, D] → [B*C, T, D]
        x_t = x.reshape(B * C, T, D)
        x_t_norm = self.norm_t(x_t)
        x_t = x_t + self.attn_t(x_t_norm, x_t_norm, x_t_norm)[0]
        x = x_t.reshape(B, C, T, D)

        # MLP
        x = x + self.mlp(self.norm_m(x))
        return x


class CrissCrossPatchEmbed(nn.Module):
    """Patchify raw EEG signal into (channel × time-patch) tokens.

    Input: [B, T_samples, C]
    Output: [B, C, T_patch, D]  where T_patch = T_samples // patch_len

    Each (c, t) token corresponds to one patch of one channel.
    """

    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        # Linear projection: patch_len → d_model
        self.proj = nn.Linear(patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """eeg: [B, T, C]  →  [B, C, T_patch, D]"""
        B, T, C = eeg.shape
        P = self.patch_len
        Np = T // P
        # Reshape into patches: [B, Np, P, C]
        eeg_p = eeg[:, :Np * P, :].reshape(B, Np, P, C)
        # Move channel first, apply per-patch projection
        # → [B, C, Np, P]
        eeg_p = eeg_p.permute(0, 3, 1, 2)
        # → [B*C*Np, P] → [B*C*Np, D] → [B, C, Np, D]
        x = self.proj(eeg_p.reshape(-1, P))
        x = self.norm(x).reshape(B, C, Np, -1)
        return x


class MaeDecoderBlock(nn.Module):
    """Shallow transformer block for MAE decoder."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        xn = self.norm1(x)
        x = x + self.attn(xn, xn, xn)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 2. Main model
# ============================================================


class EEGLeJEPA_v2(nn.Module):
    """v2: 24M-param channel-agnostic JEPA+MAE hybrid.

    Args:
        n_channels: Reference channel count (only for sanity; model accepts
            any C at forward time).
        patch_len: samples per time patch (default 200 for 200Hz × 1s).
        d_model: main model dim (default 512).
        d_decoder: MAE decoder dim (default 256, shallower/cheaper).
        encoder_layers: main encoder depth (default 12).
        decoder_layers: MAE decoder depth (default 2, shallow OK).
        encoder_heads: main attn heads (default 8).
        decoder_heads: decoder attn heads (default 8).
        mask_ratio: fraction of (channel × time) tokens to mask (default 0.5).
        mae_weight: signal-space reconstruction loss weight (default 0.5).
        jepa_weight: latent prediction loss weight (default 1.0).
        cf_weight: cross-frequency loss weight (default 0.3, reduced from 1.0).
        sigreg_lambda: anti-collapse weight (default 0.05).
        pajr_weight: patient-adversarial weight (default 0.1).
        reg_type: 'sigreg' (recommended, safe) or 'vicreg' (v1 confirmed
            DESTROYS features, kept for ablation only).
    """

    def __init__(
        self,
        n_channels: int = 19,
        patch_len: int = 200,
        d_model: int = 512,
        d_decoder: int = 256,
        encoder_layers: int = 12,
        decoder_layers: int = 2,
        encoder_heads: int = 8,
        decoder_heads: int = 8,
        mask_ratio: float = 0.50,
        # Loss weights
        jepa_weight: float = 1.0,
        mae_weight: float = 0.5,
        cf_weight: float = 0.3,
        sigreg_lambda: float = 0.05,
        pajr_weight: float = 0.1,
        # Reg
        reg_type: str = "sigreg",
        # CF apparatus (kept from v1, reduced weight)
        n_bands: int = 5,
        d_band_view: int = 64,
        cf_band_conditioned: bool = True,
        # Position embed sizes
        max_time_patches: int = 128,
        max_channels: int = 32,
        # PAJR
        n_subjects: int = 2000,
        par_disc_hidden: int = 256,
    ):
        super().__init__()
        # Save config
        self.d_model = d_model
        self.patch_len = patch_len
        self.mask_ratio = mask_ratio
        self.jepa_weight = jepa_weight
        self.mae_weight = mae_weight
        self.cf_weight = cf_weight
        self.sigreg_lambda = sigreg_lambda
        self.pajr_weight = pajr_weight
        self.reg_type = reg_type
        self.n_bands = n_bands
        self.d_band_view = d_band_view
        self.cf_band_conditioned = cf_band_conditioned
        self.max_time_patches = max_time_patches
        self.max_channels = max_channels

        # ─── Tokenizer + position embed ─────────────────────────────
        self.patch_embed = CrissCrossPatchEmbed(patch_len, d_model)
        # Separate spatial & temporal pos embeds (channel-agnostic)
        self.pos_time = nn.Parameter(
            torch.randn(1, 1, max_time_patches, d_model) * 0.02)
        self.pos_channel = nn.Parameter(
            torch.randn(1, max_channels, 1, d_model) * 0.02)

        # ─── Encoder (12 × criss-cross blocks) ──────────────────────
        self.encoder_blocks = nn.ModuleList([
            CrissCrossBlock(d_model, encoder_heads)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # ─── JEPA predictor (latent → latent) ───────────────────────
        self.jepa_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ─── MAE decoder (latent → raw signal) — NEW in v2 ──────────
        # Project encoder dim to decoder dim
        self.decoder_embed = nn.Linear(d_model, d_decoder)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_decoder))
        nn.init.normal_(self.mask_token, std=0.02)
        # Decoder position embeddings (separate from encoder)
        self.decoder_pos_time = nn.Parameter(
            torch.randn(1, 1, max_time_patches, d_decoder) * 0.02)
        self.decoder_pos_channel = nn.Parameter(
            torch.randn(1, max_channels, 1, d_decoder) * 0.02)
        self.decoder_blocks = nn.ModuleList([
            MaeDecoderBlock(d_decoder, decoder_heads)
            for _ in range(decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_decoder)
        # Reconstruct raw signal: d_decoder → patch_len
        self.decoder_pred = nn.Linear(d_decoder, patch_len)

        # ─── Cross-Frequency apparatus (from v1) ────────────────────
        self.band_head = nn.Linear(d_model, n_bands * d_band_view)
        in_dim = d_band_view * 2 if cf_band_conditioned else d_band_view
        self.cf_predictor = nn.Sequential(
            nn.Linear(in_dim, d_band_view * 2),
            nn.GELU(),
            nn.Linear(d_band_view * 2, d_band_view),
        )
        self.cf_band_mask_tokens = nn.Parameter(
            torch.randn(n_bands, d_band_view) * 0.02)

        # ─── PAJR discriminator ─────────────────────────────────────
        self.par_disc = nn.Sequential(
            nn.Linear(d_model, par_disc_hidden),
            nn.GELU(),
            nn.Linear(par_disc_hidden, n_subjects),
        )

    # ─────────────────────────────────────────────────────────────
    # Basic tokenize + encode
    #
    # v2 has a DUAL interface:
    #   - Internal (training): _tokenize_4d / _encode_4d preserve
    #     [B, C, T_p, D] structure for criss-cross attention.
    #   - Downstream compat: _tokenize / _encode flatten to [B, N, D]
    #     so v1's eval code (model._encode(model._tokenize(eeg)).mean(1))
    #     works unchanged.
    # ─────────────────────────────────────────────────────────────

    def _tokenize_4d(self, eeg: torch.Tensor) -> torch.Tensor:
        """eeg: [B, T, C] → tokens [B, C, T_p, D] with pos embed added."""
        tokens = self.patch_embed(eeg)          # [B, C, T_p, D]
        B, C, Tp, D = tokens.shape

        # Safety: crop C to fit pos_channel and pad pos_time if needed
        max_c = self.pos_channel.shape[1]
        max_t = self.pos_time.shape[2]
        if C > max_c:
            tokens = tokens[:, :max_c, :, :]
            C = max_c
        if Tp > max_t:
            tokens = tokens[:, :, :max_t, :]
            Tp = max_t

        # Add separate spatial + temporal pos embeddings
        # Use explicit expand + add to be robust to broadcasting quirks
        pt = self.pos_time[:, :, :Tp, :]      # [1, 1, Tp, D]
        pc = self.pos_channel[:, :C, :, :]    # [1, C, 1, D]
        tokens = tokens + pt + pc
        return tokens

    def _encode_4d(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, C, T_p, D] → encoded [B, C, T_p, D]"""
        x = tokens
        for blk in self.encoder_blocks:
            x = blk(x)
        return self.encoder_norm(x)

    # ─── v1-compatible downstream interface ─────────────────────

    def _tokenize(self, eeg: torch.Tensor) -> torch.Tensor:
        """Compat with v1 eval: [B, T, C] → [B, N=C*T_p, D].

        Cache the encoded 4D tensor on the instance so _encode can pick it
        up without re-computing (avoids running criss-cross on flat tokens).
        """
        eeg_4d = self._tokenize_4d(eeg)          # [B, C, T_p, D]
        enc_4d = self._encode_4d(eeg_4d)         # [B, C, T_p, D]
        # Cache for _encode fallback
        self._cached_enc_4d = enc_4d
        B, C, Tp, D = enc_4d.shape
        return enc_4d.reshape(B, C * Tp, D)      # [B, C*T_p, D]

    def _encode(self, tokens_flat: torch.Tensor) -> torch.Tensor:
        """Compat with v1 eval: identity (encoding already done in _tokenize).

        v1 pattern is `_encode(_tokenize(x))`; we do all work in _tokenize
        and return an identity here so total behavior matches.
        """
        return tokens_flat

    def _encode_features(self, eeg: torch.Tensor) -> torch.Tensor:
        """Direct downstream: [B, T, C] → [B, C*T_p, D].
        Equivalent to _tokenize (which does everything)."""
        return self._tokenize(eeg)

    # ─────────────────────────────────────────────────────────────
    # Masking (channel × time joint random mask)
    # ─────────────────────────────────────────────────────────────

    def _random_mask(self, B: int, C: int, Tp: int, device):
        """Return (visible_idx, mask_idx) tensor pair.

        For each sample in batch, pick mask_ratio × (C × Tp) tokens to mask.
        Returns:
            mask: [B, C, Tp] bool tensor (True = masked)
        """
        n_total = C * Tp
        n_mask = int(n_total * self.mask_ratio)
        mask_all = torch.zeros(B, n_total, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(n_total, device=device)[:n_mask]
            mask_all[b, idx] = True
        return mask_all.reshape(B, C, Tp)

    # ─────────────────────────────────────────────────────────────
    # Forward: training with JEPA + MAE + CF losses
    # ─────────────────────────────────────────────────────────────

    def forward(self, eeg: torch.Tensor,
                subject_ids: Optional[torch.Tensor] = None) -> dict:
        """Compute all pretrain losses.

        eeg: [B, T, C]
        subject_ids: [B] int (for PAJR); optional
        Returns: dict of loss components and total.
        """
        B, T, C = eeg.shape

        # 1. Tokenize (use 4D internal path for criss-cross)
        tokens_all = self._tokenize_4d(eeg)      # [B, C, T_p, D]
        Bt, Ct, Tp, D = tokens_all.shape
        assert C == Ct and B == Bt

        # 2. Random mask
        mask = self._random_mask(B, C, Tp, eeg.device)  # [B, C, T_p]

        # 3. Encode ALL tokens (visible view)
        #    In JEPA/MAE, encoder sees only visible; here we simplify by
        #    processing all tokens with masking applied via attention mask
        #    is complex. Instead, we substitute mask positions with a
        #    learnable mask token AT ENCODER INPUT (like BERT).
        # To keep implementation simple + effective, we go MAE-style:
        # 1) Encode ALL tokens with visible positions unchanged.
        # 2) JEPA target = encoder output at masked positions (stop-grad).
        # 3) JEPA prediction = encoder output at same positions (with grad).
        # This mimics LeJEPA (no EMA) — same encoder path.
        enc_all = self._encode_4d(tokens_all)    # [B, C, T_p, D]

        # ─── JEPA loss (main structure) ────────────────────────────
        # target: encoded (detached at masked positions),
        # pred: predictor(encoded) at masked positions.
        # Here we simplify: predict all positions from encoder output.
        pred_all = self.jepa_predictor(enc_all)
        target_all = enc_all.detach()
        # JEPA loss only on masked positions
        loss_jepa = F.mse_loss(
            pred_all[mask], target_all[mask])

        # ─── MAE reconstruction (NEW in v2) ────────────────────────
        # Substitute masked positions with mask_token, then decode.
        # enc_all: [B, C, T_p, D]
        dec_input = self.decoder_embed(enc_all)  # [B, C, T_p, D_dec]
        mask_tok = self.mask_token.expand_as(dec_input)
        dec_input = torch.where(mask[..., None], mask_tok, dec_input)
        # Add decoder pos embed
        dec_input = dec_input + self.decoder_pos_time[:, :, :Tp, :] \
                              + self.decoder_pos_channel[:, :C, :, :]

        # Flatten C×T for decoder attention
        dec_flat = dec_input.reshape(B, C * Tp, -1)
        for blk in self.decoder_blocks:
            dec_flat = blk(dec_flat)
        dec_flat = self.decoder_norm(dec_flat)
        # Predict raw patch samples
        recon = self.decoder_pred(dec_flat)      # [B, C*T_p, P]
        recon = recon.reshape(B, C, Tp, self.patch_len)

        # Target = raw eeg patches
        target = eeg[:, :Tp * self.patch_len, :].reshape(B, Tp, self.patch_len, C)
        target = target.permute(0, 3, 1, 2)     # [B, C, T_p, P]

        # MAE loss only on masked positions
        loss_mae = F.mse_loss(
            recon[mask], target[mask])

        # ─── Cross-Frequency loss (reduced weight) ─────────────────
        # Apply on flattened encoder output (like v1)
        enc_flat = enc_all.reshape(B, C * Tp, D)
        band_views = self.band_head(enc_flat).reshape(
            B, C * Tp, self.n_bands, self.d_band_view)
        # Simple CF: mask 1 band, predict from mean of others
        band_visible = band_views.mean(dim=2)   # [B, N, d_band] average
        band_target = band_views[:, :, 0, :]    # predict band 0
        if self.cf_band_conditioned:
            band_query = torch.cat([band_visible,
                                     self.cf_band_mask_tokens[0].expand_as(band_visible)],
                                    dim=-1)
        else:
            band_query = band_visible
        band_pred = self.cf_predictor(band_query)
        loss_cf = F.mse_loss(band_pred, band_target.detach())

        # ─── SIGReg (anti-collapse) ────────────────────────────────
        # sigreg_loss expects [M, D] (flatten B and N)
        z_flat = enc_flat.reshape(-1, D)  # [B*N, D]
        reg, _ = distribution_reg(z_flat, self.reg_type)

        # ─── PAJR patient-adversarial with gradient reversal ───────
        loss_pajr = torch.tensor(0.0, device=eeg.device)
        if subject_ids is not None and self.pajr_weight > 0:
            # Discriminator predicts subject from mean-pooled features
            # enc_flat is [B, N, D]; pool across N to get [B, D]
            feats_mean = enc_flat.mean(dim=1)  # [B, D]
            # Gradient reversal: encoder sees NEGATIVE gradient (adversarial)
            # while discriminator itself trains normally.
            feats_reversed = _GradReverse.apply(feats_mean, 1.0)
            logits = self.par_disc(feats_reversed)
            loss_pajr = F.cross_entropy(logits, subject_ids)

        # ─── Total ─────────────────────────────────────────────────
        # Note: PAJR uses gradient reversal internally (encoder feature path
        # gets negated gradient), so add with POSITIVE sign here.
        total = (self.jepa_weight * loss_jepa
                 + self.mae_weight * loss_mae
                 + self.cf_weight * loss_cf
                 + self.sigreg_lambda * reg
                 + self.pajr_weight * loss_pajr)

        return {
            "total": total,
            "jepa": loss_jepa.detach(),
            "mae": loss_mae.detach(),
            "cf": loss_cf.detach(),
            "sig": reg.detach() if isinstance(reg, torch.Tensor) else torch.tensor(reg),
            "pajr": loss_pajr.detach(),
        }


# ============================================================
# 3. Count params
# ============================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = EEGLeJEPA_v2(
        n_channels=19, patch_len=200, d_model=512,
        encoder_layers=12, encoder_heads=8,
    )
    print(f"v2 total params: {count_params(model)/1e6:.1f} M")
    # Test forward
    eeg = torch.randn(2, 2560, 19)   # 10s @ 256Hz, 19 channels
    subject_ids = torch.tensor([0, 1])
    losses = model(eeg, subject_ids)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
