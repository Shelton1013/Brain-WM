"""SpecBraM model and historical EEG-LeJEPA v3 ablations.

The current paper uses tokenizer_type='linear', mask_axis='time',
target_type='fft_bandpower', fft_bands='standard', target_log='log',
jepa_weight=0 and sigreg_lambda=0. See configs/specbram_main.json.
Targets are classical-band powers computed by a patch-local FFT, with log(P+1e-3).

Constructor defaults retain the earlier filterbank/frequency-mask/log1p model
for backwards compatibility. They are not the current paper configuration.
Always use v3_kwargs_from_args when loading an existing checkpoint.
The downstream `_tokenize` / `_encode` interface matches the older encoders.
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
# Exact clinical band edges (Hz) for the FFT band-power target (ablation).
_BAND_EDGES_HZ = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
# Measured -3 dB passbands of the fixed 65-tap target filters at 256 Hz (the
# "narrow" bands the default `bandpower` target actually measures). Used by the
# FFT target to separate band definition from estimator.
_NARROW_EDGES_HZ = [(2.9, 6.3), (3.7, 8.5), (7.2, 12.8), (18.1, 23.9), (35.1, 40.9)]
# "nodelta": standard theta..gamma only -- equivalent to computing the standard-band
# target on a 4-Hz high-passed signal (tests whether the slow/ocular band hurts vigilance)
FFT_BANDS = {"standard": _BAND_EDGES_HZ, "narrow": _NARROW_EDGES_HZ,
             "nodelta": _BAND_EDGES_HZ[1:]}

# Pretext targets (ablation — isolates the prediction target on a fixed backbone):
#   bandpower     : legacy log1p power from the fixed Hann filterbank
#   fft_bandpower : log band-power in exact δθαβγ bands via per-patch FFT
#   raw_recon     : reconstruct the raw masked waveform (CBraMod-style MAE)
#   band_recon    : reconstruct the fixed-filterbank band WAVEFORMS of the patch
TARGET_TYPES = ("bandpower", "fft_bandpower", "raw_recon", "band_recon")


def v3_kwargs_from_args(a: dict) -> dict:
    """Constructor kwargs for EEGLeJEPA_v3 from a checkpoint's saved args.
    Defaults preserve legacy checkpoints; saved fields select the current paper model."""
    return dict(
        d_model=a.get("d_model", 512),
        encoder_layers=a.get("encoder_layers", 12), n_heads=8,
        patch_len=a.get("patch_len", 200),
        max_time_patches=a.get("max_time_patches", 64),
        max_channels=a.get("max_channels", 32),
        n_bands=a.get("n_bands", 5),
        d_band=a.get("cf_d_band", 64),
        filt_kernel=a.get("filt_kernel", 65),
        sample_rate=256,
        band_mask_ratio=a.get("band_mask_ratio", 0.30),
        jepa_weight=a.get("jepa_weight", 0.3),
        cf_weight=a.get("cf_weight", 1.0),
        sigreg_lambda=a.get("sigreg_lambda", 0.05),
        reg_type=a.get("reg_type", "sigreg"),
        mask_axis=a.get("mask_axis", "frequency"),
        target_type=a.get("target_type", "bandpower"),
        patch_local_filter=bool(a.get("patch_local_filter", False)),
        tokenizer_type=a.get("tokenizer_type", "filterbank"),
        fft_bands=a.get("fft_bands", "standard"),
        fft_taper=a.get("fft_taper", "rect"),
        target_log=a.get("target_log", "log1p"),
    )


class FilterbankTokenizer(nn.Module):
    """Raw EEG → per-band patch embeddings + per-band log-power targets.

    Input : eeg [B, T, C]
    Output:
        band_emb   [B, C, Tp, N, d_band]   per-band patch embeddings
        band_logpow[B, C, Tp, N]           log(1+power) target per band/patch
        token      [B, C, Tp, D]           combined (all-band) encoder token (no pos)
    """

    def __init__(self, patch_len: int, d_model: int, n_bands: int = 5,
                 d_band: int = 64, filt_kernel: int = 65, sample_rate: int = 256,
                 learnable_target: bool = False, patch_local: bool = False):
        super().__init__()
        self.patch_len = patch_len
        # ablation (leakage control): filter each patch in isolation (zero-pad
        # at patch edges) so a masked patch's samples cannot reach the band
        # signals of its visible neighbours through the 65-tap kernel.
        self.patch_local = patch_local
        self.n_bands = n_bands
        self.d_band = d_band
        # LEARNABLE bank — used only for the encoder INPUT (tokenization).
        self.filters = nn.Conv1d(1, n_bands, kernel_size=filt_kernel,
                                 padding=filt_kernel // 2, bias=False)
        self._init_bandpass(self.filters.weight.data, filt_kernel, sample_rate)
        # FIXED bank — used only for the CF TARGET (non-learnable buffer). Keeps
        # the spectral target stable so the model can't collapse it to a trivial
        # constant by degenerating the learnable filters.
        tgt = torch.zeros(n_bands, 1, filt_kernel)
        self._init_bandpass(tgt, filt_kernel, sample_rate)
        self.register_buffer("target_filters", tgt)
        self.tgt_pad = filt_kernel // 2

        self.learnable_target = learnable_target   # ablation: collapse-prone target
        self.band_proj = nn.Linear(patch_len, d_band)       # per-band patch -> d_band
        self.combine = nn.Linear(n_bands * d_band, d_model)  # bands -> token
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _init_bandpass(weight: torch.Tensor, K: int, fs: int):
        """Fill `weight` [N,1,K] with Hann-windowed sinusoids at band centres."""
        n = torch.arange(K) - K // 2
        win = torch.hann_window(K, periodic=False)
        with torch.no_grad():
            for b in range(weight.shape[0]):
                fc = _BAND_CENTERS_HZ[b % len(_BAND_CENTERS_HZ)]
                kernel = win * torch.cos(2 * math.pi * fc * n / fs)
                kernel = kernel - kernel.mean()               # zero-DC
                kernel = kernel / (kernel.norm() + 1e-6)
                weight[b, 0].copy_(kernel)

    def _filter(self, x: torch.Tensor, weight: torch.Tensor, B: int, C: int,
                Tp: int) -> torch.Tensor:
        """x [B*C,1,T_use] -> band signals [B,C,N,Tp,P] (full-signal or per-patch)."""
        P, N = self.patch_len, self.n_bands
        if self.patch_local:
            xp = x.reshape(B * C * Tp, 1, P)
            y = F.conv1d(xp, weight, padding=weight.shape[-1] // 2)   # [BCTp,N,P]
            return y.reshape(B, C, Tp, N, P).permute(0, 1, 3, 2, 4)
        y = F.conv1d(x, weight, padding=weight.shape[-1] // 2)
        return y.reshape(B, C, N, Tp, P)

    def forward(self, eeg: torch.Tensor):
        B, T, C = eeg.shape
        Tp = T // self.patch_len
        T_use = Tp * self.patch_len
        x = eeg[:, :T_use, :].permute(0, 2, 1).reshape(B * C, 1, T_use)  # [B*C,1,T]

        # ── INPUT: learnable bank ──
        bands = self._filter(x, self.filters.weight, B, C, Tp)

        # ── CF TARGET ──
        if self.learnable_target:
            # ABLATION: target from the LEARNABLE bank, WITH gradient. The model
            # can then game it by collapsing the filters so the target -> 0.
            logpow = torch.log1p((bands ** 2).mean(dim=-1))
            band_logpow = logpow.permute(0, 1, 3, 2).contiguous()
        else:
            # DEFAULT: fixed bank, no grad -> stable, ungameable.
            with torch.no_grad():
                tbands = self._filter(x, self.target_filters, B, C, Tp)
                band_logpow = torch.log1p((tbands ** 2).mean(dim=-1)) \
                    .permute(0, 1, 3, 2).contiguous()

        band_emb = self.band_proj(bands)                       # [B,C,N,Tp,d_band]
        band_emb = band_emb.permute(0, 1, 3, 2, 4).contiguous()  # [B,C,Tp,N,d_band]
        token = self._combine(band_emb)                        # [B,C,Tp,D]
        return band_emb, band_logpow, token

    @torch.no_grad()
    def fixed_band_signals(self, eeg: torch.Tensor) -> torch.Tensor:
        """Fixed-filterbank band waveforms per patch: [B,C,Tp,N,P]."""
        B, T, C = eeg.shape
        Tp = T // self.patch_len
        x = eeg[:, :Tp * self.patch_len, :].permute(0, 2, 1).reshape(B * C, 1, -1)
        return self._filter(x, self.target_filters, B, C, Tp).permute(0, 1, 3, 2, 4)

    def _combine(self, band_emb: torch.Tensor) -> torch.Tensor:
        B, C, Tp, N, d = band_emb.shape
        tok = self.combine(band_emb.reshape(B, C, Tp, N * d))
        return self.norm(tok)


class LinearPatchTokenizer(FilterbankTokenizer):
    """Main-model plain linear patch embedding, without a learnable input filterbank.
    at the input. The raw patch is projected straight to N*d_band features
    (split into N chunks so masking / _combine / the encoder are unchanged).
    The fixed reference filterbank is kept ONLY to compute the band-power
    target, which never depends on the input tokenizer."""

    def __init__(self, patch_len: int, d_model: int, n_bands: int = 5,
                 d_band: int = 64, filt_kernel: int = 65, sample_rate: int = 256,
                 patch_local: bool = False, **_):
        super().__init__(patch_len, d_model, n_bands, d_band, filt_kernel,
                         sample_rate, learnable_target=False, patch_local=patch_local)
        del self.filters, self.band_proj            # no learnable filterbank
        self.patch_proj = nn.Linear(patch_len, n_bands * d_band)

    def forward(self, eeg: torch.Tensor):
        B, T, C = eeg.shape
        P, N = self.patch_len, self.n_bands
        Tp = T // P
        x = eeg[:, :Tp * P, :].permute(0, 2, 1).reshape(B * C, 1, Tp * P)
        with torch.no_grad():
            tbands = self._filter(x, self.target_filters, B, C, Tp)
            band_logpow = torch.log1p((tbands ** 2).mean(dim=-1)) \
                .permute(0, 1, 3, 2).contiguous()
        band_emb = self.patch_proj(x.reshape(B, C, Tp, P)) \
            .reshape(B, C, Tp, N, self.d_band)
        return band_emb, band_logpow, self._combine(band_emb)


TOKENIZERS = {"filterbank": FilterbankTokenizer, "linear": LinearPatchTokenizer}


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
        cf_learnable_target: bool = False,   # ablation: collapse-prone CF target
        mask_axis: str = "frequency",        # ablation: "frequency" (mask bands,
                                             # cross-freq) vs "time" (mask whole
                                             # time-patches, cross-time). Only the
                                             # MASK AXIS differs — isolates the
                                             # frequency-vs-time-masking claim.
        target_type: str = "bandpower",      # ablation: pretext target (TARGET_TYPES)
        patch_local_filter: bool = False,    # ablation: per-patch filtering (no leak)
        tokenizer_type: str = "filterbank",  # ablation: "linear" = no spectral input bias
        fft_bands: str = "standard",         # fft_bandpower: "standard" | "narrow" edges
        fft_taper: str = "rect",             # fft_bandpower: "rect" | "multitaper" (DPSS)
        target_log: str = "log1p",           # fft_bandpower: "log1p" | "log" (log(P+floor))
        **_ignored,          # tolerate extra ckpt args
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.n_bands = n_bands
        self.d_band = d_band
        self.band_mask_ratio = band_mask_ratio
        assert mask_axis in ("frequency", "time")
        self.mask_axis = mask_axis
        assert target_type in TARGET_TYPES, target_type
        if target_type in ("raw_recon", "band_recon"):
            assert mask_axis == "time", "waveform targets need whole-patch masking"
        self.target_type = target_type
        assert fft_bands in FFT_BANDS, fft_bands
        self.fft_edges = FFT_BANDS[fft_bands]
        assert fft_taper in ("rect", "multitaper") and target_log in ("log1p", "log")
        self.fft_taper, self.target_log = fft_taper, target_log
        self.mt_nw, self.mt_k = 2.0, 3          # time-bandwidth 2 -> +-2.6 Hz at 0.78 s
        self.log_floor = 1e-3                   # robust-scaled units: well below band powers
        self._dpss = None
        self.sample_rate = sample_rate
        self.jepa_weight = jepa_weight
        self.cf_weight = cf_weight
        self.sigreg_lambda = sigreg_lambda
        self.reg_type = reg_type

        assert tokenizer_type in TOKENIZERS, tokenizer_type
        self.tokenizer = TOKENIZERS[tokenizer_type](
            patch_len, d_model, n_bands, d_band, filt_kernel, sample_rate,
            learnable_target=cf_learnable_target, patch_local=patch_local_filter)

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
        # predict per-band log-power (FFT targets may use fewer bands than the tokenizer)
        n_tgt = len(self.fft_edges) if target_type == "fft_bandpower" else n_bands
        self.cf_head = nn.Linear(d_model, n_tgt)
        # waveform-reconstruction heads exist only for those ablations, so the
        # paper model's state_dict is unchanged
        if target_type == "raw_recon":
            self.recon_head = nn.Linear(d_model, patch_len)
        elif target_type == "band_recon":
            self.recon_head = nn.Linear(d_model, n_bands * patch_len)

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

        # mask: [B,C,Tp,N] True = masked (predict it). Two axes (ablation):
        if self.mask_axis == "time":
            # mask WHOLE time-patches (all bands) -> predict their band-power
            # from OTHER time-patches = cross-TIME. Guarantee ≥1 visible patch.
            tmask = torch.rand(B, C, Tp, device=eeg.device) < self.band_mask_ratio
            all_t_masked = tmask.all(dim=-1)                  # [B,C]
            tmask[..., 0] = tmask[..., 0] & ~all_t_masked
            bmask = tmask.unsqueeze(-1).expand(B, C, Tp, N).clone()
        else:
            # mask random BANDS per position -> predict from OTHER bands = cross-FREQ
            bmask = torch.rand(B, C, Tp, N, device=eeg.device) < self.band_mask_ratio
            all_masked = bmask.all(dim=-1)                    # [B,C,Tp]
            bmask[..., 0] = bmask[..., 0] & ~all_masked

        mask_tok = self.cf_mask_token.expand(B, C, Tp, N, d)
        band_ctx = torch.where(bmask.unsqueeze(-1), mask_tok, band_emb)
        tok_ctx = self.tokenizer._combine(band_ctx)         # [B,C,Tp,D]
        enc_ctx = self._encode_4d(self._add_pos(tok_ctx))    # [B,C,Tp,D]

        # ─ Pretext loss on masked positions (primary) ─
        if self.target_type in ("bandpower", "fft_bandpower"):
            if self.target_type == "fft_bandpower":
                band_logpow = self._fft_band_logpow(eeg)[:, :C, :Tp]
            cf_pred = self.cf_head(enc_ctx)                  # [B,C,Tp,N]
            if bmask.any() and cf_pred.shape[-1] != N:
                # fewer target bands than tokenizer bands (fft_bands=nodelta): time
                # masking hides whole patches, so mask by patch
                pm = bmask.any(-1)
                loss_cf = F.mse_loss(cf_pred[pm], band_logpow[pm])
            elif bmask.any():
                loss_cf = F.mse_loss(cf_pred[bmask], band_logpow[bmask])
            else:
                loss_cf = enc_ctx.sum() * 0.0
        else:
            tmask = bmask[..., 0]                            # [B,C,Tp] whole patches
            P = self.patch_len
            if self.target_type == "raw_recon":
                tgt = eeg[:, :Tp * P, :C].permute(0, 2, 1).reshape(B, C, Tp, P)
            else:
                tgt = self.tokenizer.fixed_band_signals(eeg)[:, :C, :Tp] \
                    .reshape(B, C, Tp, N * P)
            pred = self.recon_head(enc_ctx)
            if tmask.any():
                loss_cf = F.mse_loss(pred[tmask], tgt[tmask])
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

    @torch.no_grad()
    def _fft_band_logpow(self, eeg: torch.Tensor) -> torch.Tensor:
        """Classical-band variance via patch-local FFT, then log(P+floor) or legacy log1p:
        [B,C,Tp,N]. Patch-local by construction (no cross-patch leakage)."""
        B, T, C = eeg.shape
        P = self.patch_len
        Tp = T // P
        xp = eeg[:, :Tp * P, :].permute(0, 2, 1).reshape(B, C, Tp, P).float()
        if self.fft_taper == "multitaper":
            # K DPSS tapers (unit energy): averaging the K eigenspectra cuts the
            # variance of each bin by ~1/K at the cost of +-W Hz smoothing
            if getattr(self, "_dpss", None) is None or self._dpss.device != eeg.device:
                from scipy.signal.windows import dpss
                w = torch.tensor(dpss(P, self.mt_nw, self.mt_k).copy(), dtype=torch.float32)
                self._dpss = (w / w.norm(dim=-1, keepdim=True)).to(eeg.device)   # [K,P]
            spec = (torch.fft.rfft(xp.unsqueeze(-2) * self._dpss, dim=-1).abs() ** 2).mean(-2) * P
        else:
            spec = torch.fft.rfft(xp, dim=-1).abs() ** 2      # [B,C,Tp,P//2+1]
        freqs = torch.fft.rfftfreq(P, d=1.0 / self.sample_rate).to(eeg.device)
        out = []
        for lo, hi in self.fft_edges:
            sel = (freqs >= lo) & (freqs < hi)
            # one-sided Parseval: band variance = 2 * sum|X_k|^2 / P^2
            out.append(2.0 * spec[..., sel].sum(-1) / (P * P))
        bp = torch.stack(out, dim=-1)
        if self.target_log == "log":                           # true log, small floor
            return torch.log(bp + self.log_floor)
        return torch.log1p(bp)

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
