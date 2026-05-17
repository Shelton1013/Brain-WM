"""
BrainWM v3: Causal Predictive Coding Foundation Model for EEG

v3 changes from v2:
  - Region masking (Brain-JEPA Cross-ROI inspired): predict masked brain
    regions from unmasked ones via cross-attention
  - InfoNCE contrastive prediction loss with within-sequence negatives
  - VICReg (variance + covariance) regularization to prevent collapse
  - Content-only EMA targets (no temporal position embedding) to prevent
    position-matching shortcut in contrastive loss
  - EEG-specific augmentations: electrode pop, powerline noise, slow drift
  - Wider prediction horizons: k=[1, 3, 5] (100ms, 300ms, 500ms ahead)
  - Delayed adversarial training: GRL starts at 30% training progress
  - EMA decay caps at 0.9999 (never fully freezes target encoder)

v2 features retained:
  - 100ms state resolution (40 states/trial)
  - Channel-independent encoding with temporal attention
  - Regional cross-attention aggregation (5 brain regions)
  - Causal Mamba world model (forward-only prediction)
  - Subject adversarial training with gradient reversal
  - Scheduled sampling for rollout robustness
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BrainWMConfig


# ============================================================
# 1. Data Augmentation (v3: + EEG-specific perturbations)
# ============================================================

class EEGAugmentation(nn.Module):
    """EEG data augmentation with domain-specific perturbations.

    Generic augmentations (v2):
      - Time shift, amplitude scaling, Gaussian noise, channel dropout

    EEG-specific augmentations (v3, EchoJEPA-inspired domain adaptation):
      - Electrode pop: sudden spike on a single channel
      - Powerline interference: 50/60Hz sinusoidal noise
      - Slow drift: low-frequency baseline wander
    """

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.time_shift = config.aug_time_shift_samples
        self.amp_lo, self.amp_hi = config.aug_amplitude_scale
        self.noise_std = config.aug_gaussian_noise_std
        self.chan_drop_p = config.aug_channel_dropout_p
        self.pop_prob = config.aug_electrode_pop_prob
        self.powerline_prob = config.aug_powerline_prob
        self.drift_prob = config.aug_slow_drift_prob
        self.sample_rate = config.sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        B, T, C = x.shape

        # 1. Random time shift (+-25ms)
        if self.time_shift > 0:
            shift = torch.randint(-self.time_shift, self.time_shift + 1, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=1)

        # 2. Random amplitude scaling per channel
        scale = torch.empty(1, 1, C, device=x.device).uniform_(self.amp_lo, self.amp_hi)
        x = x * scale

        # 3. Gaussian noise
        if self.noise_std > 0:
            ch_std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
            noise = torch.randn_like(x) * ch_std * self.noise_std
            x = x + noise

        # 4. Channel dropout
        if self.chan_drop_p > 0:
            mask = torch.bernoulli(torch.full((1, 1, C), 1 - self.chan_drop_p, device=x.device))
            x = x * mask

        # 5. Electrode pop artifact: brief high-amplitude spike on 1 channel
        if torch.rand(1).item() < self.pop_prob:
            ch = torch.randint(0, C, (1,)).item()
            t_center = torch.randint(0, T, (1,)).item()
            width = torch.randint(3, 10, (1,)).item()
            t_start = max(0, t_center - width)
            t_end = min(T, t_center + width)
            amplitude = x[:, :, ch].std() * torch.empty(1, device=x.device).uniform_(3.0, 8.0)
            t_range = torch.arange(t_start, t_end, device=x.device, dtype=x.dtype)
            spike = amplitude * torch.exp(-0.5 * ((t_range - t_center) / max(width / 3, 1)) ** 2)
            x[:, t_start:t_end, ch] = x[:, t_start:t_end, ch] + spike.unsqueeze(0)

        # 6. Powerline interference: 50/60Hz sinusoidal on all channels
        if torch.rand(1).item() < self.powerline_prob:
            t = torch.arange(T, device=x.device, dtype=x.dtype) / self.sample_rate
            freq = 50.0 if torch.rand(1).item() < 0.5 else 60.0
            phase = torch.rand(1, device=x.device) * 2 * math.pi
            amplitude = x.std() * torch.empty(1, device=x.device).uniform_(0.05, 0.2)
            interference = amplitude * torch.sin(2 * math.pi * freq * t + phase)
            x = x + interference.unsqueeze(0).unsqueeze(-1)

        # 7. Slow baseline drift on random channels
        if torch.rand(1).item() < self.drift_prob:
            n_ch = torch.randint(1, max(2, C // 8), (1,)).item()
            channels = torch.randperm(C, device=x.device)[:n_ch]
            t = torch.arange(T, device=x.device, dtype=x.dtype) / T
            freq = torch.empty(1, device=x.device).uniform_(0.1, 0.5)
            drift = x.std() * 0.5 * torch.sin(2 * math.pi * freq * t)
            x[:, :, channels] = x[:, :, channels] + drift.unsqueeze(0).unsqueeze(-1)

        return x


# ============================================================
# 2. Learnable Filter Bank
# ============================================================

class LearnableFilterBank(nn.Module):
    """Learnable 1D conv filters initialized from classical EEG frequency bands."""

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.n_filters = config.n_freq_filters
        kernel_size = config.filter_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.filters = nn.Conv1d(
            in_channels=1, out_channels=self.n_filters,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
        )
        self._initialize_bandpass(config.sample_rate, kernel_size)

    def _initialize_bandpass(self, fs: int, kernel_size: int):
        bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 100.0)]
        t = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        with torch.no_grad():
            for i, (low, high) in enumerate(bands):
                if i >= self.n_filters:
                    break
                low_n = low / (fs / 2)
                high_n = min(high / (fs / 2), 0.99)
                h_low = low_n * torch.sinc(low_n * t)
                h_high = high_n * torch.sinc(high_n * t)
                bp = h_high - h_low
                bp = bp * torch.hamming_window(kernel_size, periodic=False)
                bp = bp / (bp.abs().sum() + 1e-8)
                self.filters.weight.data[i, 0, :] = bp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] -> [B, n_filters, T]"""
        return self.filters(x.unsqueeze(1))


# ============================================================
# 3. Channel Encoder (with Temporal Attention)
# ============================================================

class TemporalAttentionBlock(nn.Module):
    """Single self-attention layer for temporal token sequences."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class ChannelEncoder(nn.Module):
    """Encode a single channel's 100ms window into a latent vector.

    Pipeline: FilterBank -> reshape to tokens -> Temporal Attention -> pool -> project
    """

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.state_samples = config.state_samples
        self.n_filters = config.n_freq_filters
        self.attn_dim = config.channel_attn_dim

        self.filter_bank = LearnableFilterBank(config)
        self.token_proj = nn.Linear(1, config.channel_attn_dim)

        n_tokens = config.n_freq_filters * config.state_samples
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, config.channel_attn_dim) * 0.02)

        self.attn_layers = nn.ModuleList([
            TemporalAttentionBlock(config.channel_attn_dim, config.channel_attn_heads)
            for _ in range(config.channel_attn_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.channel_attn_dim),
            nn.Linear(config.channel_attn_dim, config.encoder_hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, state_samples] -> [B, d]"""
        freq = self.filter_bank(x)
        B = freq.shape[0]
        tokens = freq.reshape(B, -1, 1)
        tokens = self.token_proj(tokens) + self.pos_embed
        for layer in self.attn_layers:
            tokens = layer(tokens)
        pooled = tokens.mean(dim=1)
        return self.output_proj(pooled)


# ============================================================
# 4. Regional Attention Aggregation
# ============================================================

class RegionalAttentionAggregation(nn.Module):
    """Aggregate channel latents into region latents via cross-attention."""

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.n_regions = config.n_regions
        self.d = config.encoder_hidden_dim
        self.electrode_region_map = config.get_electrode_region_map()

        self.region_queries = nn.Parameter(torch.randn(config.n_regions, config.encoder_hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            config.encoder_hidden_dim, num_heads=4, batch_first=True,
        )
        self.norm = nn.LayerNorm(config.encoder_hidden_dim)

    def initialize_for_electrodes(self, electrode_names: list[str]):
        self.electrode_names = electrode_names
        self.n_electrodes = len(electrode_names)
        self.region_to_indices = {r: [] for r in range(self.n_regions)}
        for idx, name in enumerate(electrode_names):
            region = self.electrode_region_map.get(name, None)
            if region is not None:
                self.region_to_indices[region].append(idx)

    def forward(self, channel_latents: torch.Tensor) -> torch.Tensor:
        """[B, C, d] -> [B, R, d]"""
        B, C, d = channel_latents.shape
        region_outputs = []

        for r in range(self.n_regions):
            indices = self.region_to_indices[r]
            if len(indices) == 0:
                region_outputs.append(torch.zeros(B, 1, d, device=channel_latents.device))
                continue
            kv = channel_latents[:, indices, :]
            q = self.region_queries[r:r+1].unsqueeze(0).expand(B, -1, -1)
            out, _ = self.cross_attn(q, kv, kv)
            region_outputs.append(out)

        regions = torch.cat(region_outputs, dim=1)
        return self.norm(regions)


# ============================================================
# 5. Brain State Composer
# ============================================================

class BrainStateComposer(nn.Module):
    """Compose channel-level EEG into brain state sequences.

    Pipeline:
      1. Batched channel encoding (all channels x windows in one pass)
      2. Regional cross-attention aggregation per time step
      3. Concatenate R regions -> brain state [D]
      4. Optionally add temporal position embedding
    """

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.n_regions = config.n_regions
        self.state_samples = config.state_samples
        self.d = config.encoder_hidden_dim

        self.channel_encoder = ChannelEncoder(config)
        self.regional_agg = RegionalAttentionAggregation(config)

        self.region_embedding = nn.Parameter(
            torch.randn(config.n_regions, config.encoder_hidden_dim) * 0.02
        )
        self.temporal_embedding = nn.Embedding(256, config.brain_state_dim)

    def initialize_for_electrodes(self, electrode_names: list[str]):
        self.regional_agg.initialize_for_electrodes(electrode_names)
        self.n_channels = len(electrode_names)

    def forward(self, eeg: torch.Tensor, return_content_only: bool = False) -> torch.Tensor:
        """
        Args:
            eeg: [B, T, C] raw EEG (after augmentation)
            return_content_only: if True, return states WITHOUT temporal position
                embedding (used by EMA target encoder for contrastive targets)
        Returns:
            [B, N, D] brain state sequence (N=40 for 4s trial, D=640)
        """
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S

        # ---- Batched channel encoding ----
        eeg_windowed = eeg[:, :N * S, :].reshape(B, N, S, C)
        eeg_flat = eeg_windowed.permute(0, 1, 3, 2).reshape(B * N * C, S)
        z_flat = self.channel_encoder(eeg_flat)
        z_all = z_flat.reshape(B, N, C, self.d)

        # ---- Batched regional aggregation ----
        z_bn = z_all.reshape(B * N, C, self.d)
        region_latents = self.regional_agg(z_bn)
        region_latents = region_latents.reshape(B, N, self.n_regions, self.d)

        # Add region embedding
        region_latents = region_latents + self.region_embedding.unsqueeze(0).unsqueeze(0)

        # Concat regions: [B, N, R*d] = [B, N, D]
        content_states = region_latents.reshape(B, N, -1)

        if return_content_only:
            return content_states

        # Add temporal position embedding (only for online encoder path)
        pos = torch.arange(N, device=content_states.device)
        states = content_states + self.temporal_embedding(pos).unsqueeze(0)
        return states


# ============================================================
# 5b. Region Mask Predictor (Brain-JEPA Cross-ROI inspired)
# ============================================================

class RegionMaskPredictor(nn.Module):
    """Predict masked brain region latents from unmasked regions.

    Inspired by Brain-JEPA's Cross-ROI masking: given latents from
    visible brain regions, predict representations of masked regions
    via cross-attention. Forces the model to learn inter-region
    functional connectivity.
    """

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        d = config.encoder_hidden_dim
        self.d = d

        self.mask_tokens = nn.Parameter(torch.randn(config.n_regions, d) * 0.02)

        self.cross_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(
        self,
        unmasked_regions: torch.Tensor,
        masked_indices: torch.Tensor,
        region_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            unmasked_regions: [BN, R_unmasked, d] latents of visible regions
            masked_indices: [R_masked] indices of masked regions
            region_embedding: [R, d] region embeddings from BrainStateComposer
        Returns:
            [BN, R_masked, d] predicted latents for masked regions
        """
        BN = unmasked_regions.shape[0]

        q = self.mask_tokens[masked_indices] + region_embedding[masked_indices]
        q = q.unsqueeze(0).expand(BN, -1, -1)

        out, _ = self.cross_attn(q, unmasked_regions, unmasked_regions)
        out = self.norm1(out + q)
        out = out + self.ffn(self.norm2(out))
        return out


# ============================================================
# 6. Subject Adversary (cross-subject generalization)
# ============================================================

class SubjectAdversary(nn.Module):
    """Subject classifier with gradient reversal via register_hook."""

    def __init__(self, input_dim: int, n_subjects: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_subjects),
        )

    def forward(self, brain_states: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        if brain_states.dim() == 3:
            x = brain_states.mean(dim=1)
        else:
            x = brain_states

        if x.requires_grad and alpha > 0:
            x.register_hook(lambda grad: -alpha * grad)

        return self.classifier(x)


# ============================================================
# 7. Causal Mamba Block
# ============================================================

class MambaBlock(nn.Module):
    """Causal selective state space block."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).expand(self.d_inner, -1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, L, D = x.shape

        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        x_path = x_path.transpose(1, 2)
        x_path = self.conv1d(x_path)[:, :, :L]
        x_path = x_path.transpose(1, 2)
        x_path = F.silu(x_path)

        y = self._selective_scan(x_path)

        z = F.silu(z)
        output = self.out_proj(y * z)
        return output + residual

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_ssm = self.x_proj(x)
        B_param = x_ssm[:, :, :self.d_state]
        C_param = x_ssm[:, :, self.d_state:self.d_state * 2]
        delta = F.softplus(x_ssm[:, :, -1:].expand(-1, -1, D))
        A = -torch.exp(self.A_log)

        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt = delta[:, t, :]
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt.unsqueeze(-1) * B_param[:, t, :].unsqueeze(1)
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y = (h * C_param[:, t, :].unsqueeze(1)).sum(-1) + self.D * x[:, t, :]
            outputs.append(y)
        return torch.stack(outputs, dim=1)


class CausalMambaWorldModel(nn.Module):
    """Causal Mamba world model for brain state dynamics."""

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.brain_state_dim, config.mamba_d_model)
        self.layers = nn.ModuleList([
            MambaBlock(config.mamba_d_model, config.mamba_d_state,
                       config.mamba_d_conv, config.mamba_expand)
            for _ in range(config.mamba_n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.mamba_d_model)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """[B, N, D] -> [B, N, d_model]  (causal)"""
        x = self.input_proj(states)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ============================================================
# 8. Prediction Head (with horizon + position embedding)
# ============================================================

class PredictionHead(nn.Module):
    """Shared MLP with horizon and position embeddings."""

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        d = config.mamba_d_model

        self.horizon_embedding = nn.Embedding(
            max(config.prediction_horizons) + 1, d
        )
        self.position_embedding = nn.Embedding(256, d)

        self.mlp = nn.Sequential(
            nn.Linear(d, config.predictor_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.predictor_hidden_dim),
            nn.Linear(config.predictor_hidden_dim, config.brain_state_dim),
        )

    def forward(self, hidden: torch.Tensor, k: int, start_pos: int = 0) -> torch.Tensor:
        B, M, d = hidden.shape
        h_emb = self.horizon_embedding(torch.tensor(k, device=hidden.device))
        x = hidden + h_emb.unsqueeze(0).unsqueeze(0)

        target_positions = torch.arange(start_pos + k, start_pos + k + M, device=hidden.device)
        target_positions = target_positions.clamp(max=255)
        p_emb = self.position_embedding(target_positions)
        x = x + p_emb.unsqueeze(0)

        return self.mlp(x)


# ============================================================
# 9. EMA Target Encoder
# ============================================================

class EMAEncoder:
    """EMA copy of state composer for prediction targets.

    Always returns content-only states (no temporal position embedding)
    so contrastive loss cannot be solved by position matching.
    """

    def __init__(self, online_encoder: nn.Module, decay_start: float = 0.996, decay_end: float = 1.0):
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.current_decay = decay_start
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def set_decay(self, progress: float):
        self.current_decay = self.decay_start + (self.decay_end - self.decay_start) * progress

    @torch.no_grad()
    def update(self, online_encoder: nn.Module):
        m = self.current_decay
        for online_p, target_p in zip(online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(m).add_(online_p.data, alpha=1 - m)

    def __call__(self, *args, **kwargs):
        self.target_encoder.eval()
        return self.target_encoder(*args, return_content_only=True, **kwargs)


# ============================================================
# 10. Full BrainWM v3 Model
# ============================================================

class BrainWM(nn.Module):
    """Brain World Model v3.

    Pipeline:
        Raw EEG -> Augmentation -> ChannelEncoder (per channel, per 100ms)
        -> RegionalAttentionAggregation -> BrainStateComposition
        -> [Region Masking] -> CausalMambaWorldModel -> PredictionHead
        + RegionMaskPredictor (Cross-ROI prediction)
        + SubjectAdversary (gradient reversal)

    Training objectives:
        1. Within-sequence InfoNCE (temporal prediction, k=1,3,5)
        2. Region masking prediction (spatial, Brain-JEPA inspired)
        3. VICReg (variance + covariance regularization)
        4. Subject adversarial (cross-subject invariance)
    """

    def __init__(self, config: BrainWMConfig, n_subjects: int = 109):
        super().__init__()
        self.config = config

        self.augmentation = EEGAugmentation(config)
        self.brain_state_composer = BrainStateComposer(config)
        self.world_model = CausalMambaWorldModel(config)
        self.prediction_head = PredictionHead(config)

        # Region mask predictor (Brain-JEPA Cross-ROI inspired)
        self.region_mask_predictor = RegionMaskPredictor(config)
        self.region_mask_prob = 0.5     # probability of applying region masking
        self.region_mask_lambda = 1.0   # region prediction loss weight

        # Subject adversary for cross-subject invariance
        self.subject_adversary = SubjectAdversary(config.brain_state_dim, n_subjects)
        self.adv_alpha = 0.0        # GRL strength, ramped up during training
        self.adv_lambda = 0.1       # adversarial loss weight
        self.var_lambda = 5.0       # VICReg variance weight
        self.cov_lambda = 1.0       # VICReg covariance weight
        self.nce_temperature = 0.1  # InfoNCE temperature

        self.prediction_horizons = config.prediction_horizons
        self.horizon_weights = config.horizon_weights
        self.smooth_l1_beta = config.smooth_l1_beta

        # EMA (initialized on first forward)
        self._ema_initialized = False
        self.ema_encoder = None

        # Scheduled sampling
        self.ss_prob = config.scheduled_sampling_start

    def initialize_electrodes(self, electrode_names: list[str]):
        self.brain_state_composer.initialize_for_electrodes(electrode_names)

    def set_training_progress(self, progress: float):
        """Update EMA decay, scheduled sampling, and adversary strength."""
        if self.ema_encoder is not None:
            self.ema_encoder.set_decay(progress)
        self.ss_prob = (
            self.config.scheduled_sampling_start
            + (self.config.scheduled_sampling_end - self.config.scheduled_sampling_start) * progress
        )
        # Delayed adversarial: 0 until 30%, ramp to 1.0 at 80%
        if progress < 0.3:
            self.adv_alpha = 0.0
        else:
            self.adv_alpha = min((progress - 0.3) / 0.5, 1.0)

    def _init_ema(self):
        if not self._ema_initialized:
            self.ema_encoder = EMAEncoder(
                self.brain_state_composer,
                self.config.ema_decay_start,
                self.config.ema_decay_end,
            )
            self._ema_initialized = True

    @torch.no_grad()
    def _get_target_states(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return content-only targets (no temporal position embedding).

        EMA encoder's __call__ passes return_content_only=True to
        BrainStateComposer, so returned states have no position info.
        """
        self._init_ema()
        return self.ema_encoder(eeg)

    def _apply_scheduled_sampling(
        self, brain_states: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Replace some real states with model predictions during training."""
        if not self.training or self.ss_prob <= 0:
            return brain_states

        B, N, D = brain_states.shape
        states = brain_states.clone()

        for t in range(1, N):
            if torch.rand(1).item() < self.ss_prob:
                with torch.no_grad():
                    pred = self.prediction_head(hidden_states[:, t-1:t, :], k=1, start_pos=t-1)
                states[:, t, :] = pred.squeeze(1).detach()
        return states

    def _apply_region_masking(self, brain_states: torch.Tensor, content_states: torch.Tensor):
        """Mask 1-2 brain regions and predict them from the rest.

        Args:
            brain_states: [B, N, D] full states with position embedding
            content_states: [B, N, D] content-only states (no position)
        Returns:
            dict with masked_states, predicted/original regions, or None
        """
        if not self.training or torch.rand(1).item() > self.region_mask_prob:
            return None

        B, N, D = brain_states.shape
        d = self.config.encoder_hidden_dim  # 128
        R = self.config.n_regions           # 5

        # Randomly mask 1-2 regions
        n_mask = torch.randint(1, 3, (1,)).item()
        perm = torch.randperm(R, device=brain_states.device)
        masked_idx = perm[:n_mask]
        unmasked_idx = perm[n_mask:]

        # Replace masked region slices with mask tokens in full states
        masked_states = brain_states.clone()
        for r in masked_idx:
            start, end = r.item() * d, (r.item() + 1) * d
            mask_token = self.region_mask_predictor.mask_tokens[r]
            masked_states[:, :, start:end] = mask_token.unsqueeze(0).unsqueeze(0)

        # Extract unmasked region latents from CONTENT states (no position)
        unmasked_regions = torch.stack(
            [content_states[:, :, r.item()*d:(r.item()+1)*d] for r in unmasked_idx],
            dim=2,
        )  # [B, N, R_unmasked, d]

        # Extract original masked region latents from content states
        original_regions = torch.stack(
            [content_states[:, :, r.item()*d:(r.item()+1)*d] for r in masked_idx],
            dim=2,
        )  # [B, N, n_mask, d]

        # Predict masked regions from unmasked ones
        unmasked_flat = unmasked_regions.reshape(B * N, -1, d)
        region_emb = self.brain_state_composer.region_embedding.detach()
        predicted = self.region_mask_predictor(
            unmasked_flat, masked_idx, region_emb,
        )  # [B*N, n_mask, d]
        predicted = predicted.reshape(B, N, n_mask, d)

        return {
            "masked_states": masked_states,
            "masked_indices": masked_idx,
            "predicted_regions": predicted,
            "original_regions": original_regions,
        }

    def forward(self, eeg: torch.Tensor, return_predictions: bool = True) -> dict:
        """
        Args:
            eeg: [B, T, C] raw EEG
        Returns:
            dict with brain_states, hidden_states, predictions, targets
        """
        # 1. Augmentation (training only)
        eeg = self.augmentation(eeg)

        # 2. Encode to brain states: [B, N, D]
        brain_states = self.brain_state_composer(eeg)  # with position embedding

        # 2b. Get content-only states for region masking (subtract position)
        N = brain_states.shape[1]
        pos = torch.arange(N, device=brain_states.device)
        pos_emb = self.brain_state_composer.temporal_embedding(pos).unsqueeze(0)
        content_states = brain_states - pos_emb

        # 3. Region masking (training only)
        result = {"brain_states": brain_states}
        wm_input = brain_states

        if self.training and return_predictions:
            region_mask_info = self._apply_region_masking(brain_states, content_states)
            if region_mask_info is not None:
                wm_input = region_mask_info["masked_states"]
                result["region_mask_info"] = region_mask_info

        # 4. World model (causal): [B, N, D] -> [B, N, d_model]
        hidden_states = self.world_model(wm_input)

        # 5. Scheduled sampling (training only)
        if self.training and self.ss_prob > 0:
            brain_states_ss = self._apply_scheduled_sampling(brain_states, hidden_states)
            if not torch.equal(brain_states_ss, brain_states):
                hidden_states = self.world_model(brain_states_ss)
                brain_states = brain_states_ss

        result["hidden_states"] = hidden_states

        # 6. Subject adversary (MUST be inside forward() for DDP)
        subj_logits = self.subject_adversary(brain_states, alpha=self.adv_alpha)
        result["subj_logits"] = subj_logits

        if return_predictions:
            # 7. EMA targets (content-only, no position embedding)
            with torch.no_grad():
                target_states = self._get_target_states(eeg)

            # 8. Multi-horizon predictions
            predictions = {}
            for k in self.prediction_horizons:
                pred = self.prediction_head(hidden_states[:, :-k, :], k=k)
                predictions[k] = pred

            result["predictions"] = predictions
            result["targets"] = target_states.detach()

        return result

    def compute_loss(self, outputs: dict, subject_ids: torch.Tensor = None) -> dict:
        """Within-sequence InfoNCE + region masking + VICReg + adversarial loss."""
        predictions = outputs["predictions"]
        targets = outputs["targets"]
        subj_logits = outputs["subj_logits"]
        brain_states = outputs["brain_states"]

        pred_losses = {}
        total_loss = torch.tensor(0.0, device=targets.device)

        # --- Prediction loss (within-sequence InfoNCE) ---
        for i, k in enumerate(self.prediction_horizons):
            pred = predictions[k]                    # [B, M, D]
            target = targets[:, k:, :].detach()      # [B, M, D]
            B_cur, M, D = pred.shape

            pred_norm = F.normalize(pred, dim=-1)
            target_norm = F.normalize(target, dim=-1)

            # Per-sample similarity: [B, M, M]
            logits = torch.bmm(pred_norm, target_norm.transpose(1, 2)) / self.nce_temperature
            labels = torch.arange(M, device=logits.device).unsqueeze(0).expand(B_cur, -1)
            loss_k = F.cross_entropy(logits.reshape(-1, M), labels.reshape(-1))

            pred_losses[f"pred_k{k}"] = loss_k
            total_loss = total_loss + self.horizon_weights[i] * loss_k

        # --- VICReg variance + covariance regularization ---
        B, N, D = brain_states.shape
        states_flat = brain_states.reshape(-1, D)

        std_per_dim = states_flat.std(dim=0)
        var_loss = F.relu(1.0 - std_per_dim).mean()

        states_centered = states_flat - states_flat.mean(dim=0, keepdim=True)
        cov = (states_centered.T @ states_centered) / max(states_flat.shape[0] - 1, 1)
        cov_loss = (cov.fill_diagonal_(0).pow(2).sum() / D)

        vicreg_loss = self.var_lambda * var_loss + self.cov_lambda * cov_loss
        total_loss = total_loss + vicreg_loss

        # --- Region masking loss (Brain-JEPA Cross-ROI) ---
        region_mask_loss = torch.tensor(0.0, device=targets.device)
        if "region_mask_info" in outputs:
            info = outputs["region_mask_info"]
            pred_r = info["predicted_regions"]      # [B, N, n_mask, d]
            masked_idx = info["masked_indices"]

            d = self.config.encoder_hidden_dim
            target_regions = torch.stack(
                [targets[:, :, r.item()*d:(r.item()+1)*d] for r in masked_idx],
                dim=2,
            )

            pred_flat = F.normalize(pred_r.reshape(-1, d), dim=-1)
            target_flat = F.normalize(target_regions.detach().reshape(-1, d), dim=-1)
            region_mask_loss = 2.0 - 2.0 * (pred_flat * target_flat).sum(-1).mean()

        total_loss = total_loss + self.region_mask_lambda * region_mask_loss

        # --- Subject adversarial loss ---
        if subject_ids is not None:
            adv_loss = F.cross_entropy(subj_logits, subject_ids)
        else:
            adv_loss = subj_logits.sum() * 0.0
        total_loss = total_loss + self.adv_lambda * adv_loss

        return {
            "total": total_loss, "adv": adv_loss,
            "var": var_loss, "cov": cov_loss,
            "rmask": region_mask_loss,
            **pred_losses,
        }

    def update_ema(self):
        if self.ema_encoder is not None:
            self.ema_encoder.update(self.brain_state_composer)

    # ---- Inference methods ----

    @torch.no_grad()
    def predict_future(self, eeg: torch.Tensor, n_future_steps: int = 10) -> torch.Tensor:
        """Autoregressive rollout."""
        self.eval()
        brain_states = self.brain_state_composer(eeg)
        generated = []
        current_states = brain_states

        for step in range(n_future_steps):
            hidden = self.world_model(current_states)
            pos = current_states.shape[1] - 1
            next_state = self.prediction_head(hidden[:, -1:, :], k=1, start_pos=pos)
            generated.append(next_state)
            current_states = torch.cat([current_states, next_state], dim=1)

        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def compute_prediction_error(self, eeg: torch.Tensor) -> torch.Tensor:
        """Per-timestep prediction error (100ms resolution).

        Returns:
            [B, N-1] prediction error at each 100ms step
        """
        self.eval()
        outputs = self.forward(eeg, return_predictions=True)
        pred = outputs["predictions"][1]
        target = outputs["targets"][:, 1:, :]

        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        pe = 1.0 - (pred_norm * target_norm).sum(dim=-1)
        return pe  # [B, N-1]

    @torch.no_grad()
    def compute_regional_prediction_error(self, eeg: torch.Tensor) -> dict:
        """Per-region prediction error for spatial analysis (MMN/P300)."""
        self.eval()
        outputs = self.forward(eeg, return_predictions=True)
        pred = outputs["predictions"][1]
        target = outputs["targets"][:, 1:, :]

        d = self.config.encoder_hidden_dim
        region_errors = {}
        for r, name in enumerate(self.config.region_names):
            start, end = r * d, (r + 1) * d
            pred_r = F.normalize(pred[:, :, start:end], dim=-1)
            tgt_r = F.normalize(target[:, :, start:end], dim=-1)
            region_errors[name] = 1.0 - (pred_r * tgt_r).sum(dim=-1)

        return region_errors
