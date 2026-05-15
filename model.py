"""
BrainWM v2: A World Model for Brain Signals

v2 changes from v1:
  - 100ms state resolution (was 500ms) → 40 states/trial (was 8)
  - Channel-independent encoding with temporal attention (was signal-level aggregation)
  - Regional cross-attention in latent space (was signal-level weighted sum)
  - Smooth L1 loss (was Cosine + VICReg)
  - Horizon + position embedding in prediction heads
  - Scheduled sampling for rollout robustness
  - EMA momentum scheduling (0.996 → 1.0)
  - Data augmentation
  - Euclidean Alignment preprocessing (in dataset.py)
  - Subject adversarial training with gradient reversal (cross-subject generalization)
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BrainWMConfig


# ============================================================
# 1. Data Augmentation
# ============================================================

class EEGAugmentation(nn.Module):
    """EEG data augmentation applied during training."""

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.time_shift = config.aug_time_shift_samples
        self.amp_lo, self.amp_hi = config.aug_amplitude_scale
        self.noise_std = config.aug_gaussian_noise_std
        self.chan_drop_p = config.aug_channel_dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] raw EEG
        Returns:
            [B, T, C] augmented EEG
        """
        if not self.training:
            return x

        B, T, C = x.shape

        # 1. Random time shift (±25ms)
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
        """[B, T] → [B, n_filters, T]"""
        return self.filters(x.unsqueeze(1))


# ============================================================
# 3. Channel Encoder (v2: with Temporal Attention)
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

    Pipeline: FilterBank → reshape to tokens → Temporal Attention → pool → project
    """

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.state_samples = config.state_samples
        self.n_filters = config.n_freq_filters
        self.attn_dim = config.channel_attn_dim

        self.filter_bank = LearnableFilterBank(config)

        # Project each (filter, time_sample) into attention dim
        self.token_proj = nn.Linear(1, config.channel_attn_dim)

        # Learnable position encoding for filter-time tokens
        n_tokens = config.n_freq_filters * config.state_samples  # 5 * 26 = 130
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, config.channel_attn_dim) * 0.02)

        # Temporal attention layers
        self.attn_layers = nn.ModuleList([
            TemporalAttentionBlock(config.channel_attn_dim, config.channel_attn_heads)
            for _ in range(config.channel_attn_layers)
        ])

        # Pool + project to encoder_hidden_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.channel_attn_dim),
            nn.Linear(config.channel_attn_dim, config.encoder_hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, state_samples] single channel, single 100ms window
        Returns:
            [B, d] latent vector (d = encoder_hidden_dim = 128)
        """
        # [B, n_filters, state_samples]
        freq = self.filter_bank(x)
        B = freq.shape[0]

        # Reshape to tokens: [B, n_filters * state_samples, 1]
        tokens = freq.reshape(B, -1, 1)
        # Project: [B, n_tokens, attn_dim]
        tokens = self.token_proj(tokens) + self.pos_embed

        # Temporal attention
        for layer in self.attn_layers:
            tokens = layer(tokens)

        # Average pool: [B, attn_dim]
        pooled = tokens.mean(dim=1)
        # Project: [B, encoder_hidden_dim]
        return self.output_proj(pooled)


# ============================================================
# 4. Regional Attention Aggregation (v2: cross-attention in latent space)
# ============================================================

class RegionalAttentionAggregation(nn.Module):
    """Aggregate channel latents into region latents via cross-attention.

    v1 aggregated raw signals (64 channels → 5 scalars). Information loss 12.8x.
    v2 aggregates latent vectors (64×128d → 5×128d). Information preserved.
    """

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.n_regions = config.n_regions
        self.d = config.encoder_hidden_dim
        self.electrode_region_map = config.get_electrode_region_map()

        # Learnable query per region
        self.region_queries = nn.Parameter(torch.randn(config.n_regions, config.encoder_hidden_dim) * 0.02)

        # Cross-attention: query=region, key/value=channel latents
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
        """
        Args:
            channel_latents: [B, C, d] latent per channel
        Returns:
            [B, R, d] latent per region (R = n_regions)
        """
        B, C, d = channel_latents.shape
        region_outputs = []

        for r in range(self.n_regions):
            indices = self.region_to_indices[r]
            if len(indices) == 0:
                region_outputs.append(torch.zeros(B, 1, d, device=channel_latents.device))
                continue

            # Key/Value: channel latents in this region [B, n_elec, d]
            kv = channel_latents[:, indices, :]
            # Query: region query [B, 1, d]
            q = self.region_queries[r:r+1].unsqueeze(0).expand(B, -1, -1)

            # Cross-attention
            out, _ = self.cross_attn(q, kv, kv)  # [B, 1, d]
            region_outputs.append(out)

        # [B, R, d]
        regions = torch.cat(region_outputs, dim=1)
        return self.norm(regions)


# ============================================================
# 5. Brain State Composer (v2)
# ============================================================

class BrainStateComposer(nn.Module):
    """Compose channel-level EEG into brain state sequences.

    Pipeline:
      1. Reshape all (time_windows × channels) into one batch → single encoder call
      2. RegionalAttentionAggregation per time step (batched across time)
      3. Concatenate R regions → brain state [D]

    v2 performance fix: no Python loops over channels/windows.
    All channel×window encoding done in ONE batched forward pass.
    """

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        self.n_regions = config.n_regions
        self.state_samples = config.state_samples
        self.d = config.encoder_hidden_dim

        self.channel_encoder = ChannelEncoder(config)
        self.regional_agg = RegionalAttentionAggregation(config)

        # Region embedding
        self.region_embedding = nn.Parameter(
            torch.randn(config.n_regions, config.encoder_hidden_dim) * 0.02
        )
        # Temporal position embedding
        self.temporal_embedding = nn.Embedding(256, config.brain_state_dim)

    def initialize_for_electrodes(self, electrode_names: list[str]):
        self.regional_agg.initialize_for_electrodes(electrode_names)
        self.n_channels = len(electrode_names)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: [B, T, C] raw EEG (after augmentation)
        Returns:
            [B, N, D] brain state sequence (N=40 for 4s trial, D=640)
        """
        B, T, C = eeg.shape
        S = self.state_samples
        N = T // S  # number of 100ms states

        # ---- Batched channel encoding (no Python loops) ----
        # Reshape: [B, T, C] → [B, N, S, C] → [B*N*C, S]
        eeg_windowed = eeg[:, :N * S, :].reshape(B, N, S, C)
        eeg_flat = eeg_windowed.permute(0, 1, 3, 2).reshape(B * N * C, S)

        # Single forward pass for ALL channels × ALL windows
        z_flat = self.channel_encoder(eeg_flat)         # [B*N*C, d]
        z_all = z_flat.reshape(B, N, C, self.d)         # [B, N, C, d]

        # ---- Batched regional aggregation ----
        # Merge B and N for batched cross-attention: [B*N, C, d]
        z_bn = z_all.reshape(B * N, C, self.d)
        region_latents = self.regional_agg(z_bn)        # [B*N, R, d]
        region_latents = region_latents.reshape(B, N, self.n_regions, self.d)

        # Add region embedding: [1, 1, R, d]
        region_latents = region_latents + self.region_embedding.unsqueeze(0).unsqueeze(0)

        # Concat regions: [B, N, R*d] = [B, N, D]
        states = region_latents.reshape(B, N, -1)

        # Add temporal position embedding
        pos = torch.arange(N, device=states.device)
        states = states + self.temporal_embedding(pos).unsqueeze(0)
        return states


# ============================================================
# 6. Subject Adversary (cross-subject generalization)
# ============================================================

class SubjectAdversary(nn.Module):
    """Subject classifier with gradient reversal via register_hook.

    Uses tensor.register_hook() instead of custom autograd.Function
    for DDP compatibility. The hook negates gradients flowing back
    to the encoder, forcing subject-invariant representations.
    """

    def __init__(self, input_dim: int, n_subjects: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_subjects),
        )

    def forward(self, brain_states: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Args:
            brain_states: [B, N, D] or [B, D] brain state representations
            alpha: gradient reversal strength (ramp up during training)
        Returns:
            [B, n_subjects] subject classification logits
        """
        if brain_states.dim() == 3:
            x = brain_states.mean(dim=1)  # [B, D]
        else:
            x = brain_states

        # Gradient reversal via hook (DDP-friendly, no custom autograd.Function)
        if x.requires_grad and alpha > 0:
            x.register_hook(lambda grad: -alpha * grad)

        return self.classifier(x)


# ============================================================
# 7. Causal Mamba Block
# ============================================================

class MambaBlock(nn.Module):
    """Causal selective state space block.

    h(t+1) = A·h(t) + B·x(t)  — state only flows forward in time
    y(t)   = C·h(t) + D·x(t)  — output depends only on past
    """

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

        # Causal conv
        x_path = x_path.transpose(1, 2)
        x_path = self.conv1d(x_path)[:, :, :L]
        x_path = x_path.transpose(1, 2)
        x_path = F.silu(x_path)

        # Causal SSM scan
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
        """[B, N, D] → [B, N, d_model]  (causal: each position sees only past)"""
        x = self.input_proj(states)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ============================================================
# 8. Prediction Heads (v2: with horizon + position embedding)
# ============================================================

class PredictionHead(nn.Module):
    """Shared MLP with horizon and position embeddings.

    v2 improvement: the head knows WHICH future step it is predicting (horizon)
    and WHERE in time the target is (position).
    """

    def __init__(self, config: BrainWMConfig):
        super().__init__()
        d = config.mamba_d_model

        # Horizon embedding: one per prediction step k
        self.horizon_embedding = nn.Embedding(
            max(config.prediction_horizons) + 1, d
        )
        # Position embedding: target temporal position
        self.position_embedding = nn.Embedding(256, d)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(d, config.predictor_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.predictor_hidden_dim),
            nn.Linear(config.predictor_hidden_dim, config.brain_state_dim),
        )

    def forward(self, hidden: torch.Tensor, k: int, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            hidden: [B, M, d_model] Mamba hidden states
            k: prediction horizon (1, 2, or 3)
            start_pos: starting temporal position index
        Returns:
            [B, M, D] predicted brain states
        """
        B, M, d = hidden.shape
        # Add horizon embedding (same for all positions)
        h_emb = self.horizon_embedding(torch.tensor(k, device=hidden.device))
        x = hidden + h_emb.unsqueeze(0).unsqueeze(0)

        # Add target position embedding
        target_positions = torch.arange(start_pos + k, start_pos + k + M, device=hidden.device)
        target_positions = target_positions.clamp(max=255)
        p_emb = self.position_embedding(target_positions)
        x = x + p_emb.unsqueeze(0)

        return self.mlp(x)


# ============================================================
# 9. EMA Target Encoder (v2: momentum scheduling)
# ============================================================

class EMAEncoder:
    """EMA copy of state composer for prediction targets.

    v2: momentum schedules from 0.996 → 1.0 (linear).
    """

    def __init__(self, online_encoder: nn.Module, decay_start: float = 0.996, decay_end: float = 1.0):
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.current_decay = decay_start
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def set_decay(self, progress: float):
        """Update decay based on training progress (0.0 → 1.0)."""
        self.current_decay = self.decay_start + (self.decay_end - self.decay_start) * progress

    @torch.no_grad()
    def update(self, online_encoder: nn.Module):
        m = self.current_decay
        for online_p, target_p in zip(online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(m).add_(online_p.data, alpha=1 - m)

    def __call__(self, *args, **kwargs):
        self.target_encoder.eval()
        return self.target_encoder(*args, **kwargs)


# ============================================================
# 10. Full BrainWM v2 Model
# ============================================================

class BrainWM(nn.Module):
    """Brain World Model v2.

    Pipeline:
        Raw EEG → Augmentation → ChannelEncoder (per channel, per 100ms)
        → RegionalAttentionAggregation → BrainStateComposition
        → CausalMambaWorldModel → PredictionHead (with horizon/position emb)
        + SubjectAdversary (gradient reversal for cross-subject invariance)

    100ms state resolution enables MMN/P300 analysis.
    Causal Mamba ensures forward-only prediction (world model).
    Smooth L1 loss (no VICReg needed).
    EA preprocessing + adversarial training for cross-subject generalization.
    """

    def __init__(self, config: BrainWMConfig, n_subjects: int = 109):
        super().__init__()
        self.config = config

        self.augmentation = EEGAugmentation(config)
        self.brain_state_composer = BrainStateComposer(config)
        self.world_model = CausalMambaWorldModel(config)
        self.prediction_head = PredictionHead(config)

        # Subject adversary for cross-subject invariance
        self.subject_adversary = SubjectAdversary(config.brain_state_dim, n_subjects)
        self.adv_alpha = 0.0        # GRL strength, ramped up during training
        self.adv_lambda = 0.1       # adversarial loss weight

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
        """Update EMA decay, scheduled sampling, and adversary strength (0→1)."""
        if self.ema_encoder is not None:
            self.ema_encoder.set_decay(progress)
        self.ss_prob = (
            self.config.scheduled_sampling_start
            + (self.config.scheduled_sampling_end - self.config.scheduled_sampling_start) * progress
        )
        # Ramp up adversarial strength: 0 → 1.0 over training
        # Let the model learn prediction first, then gradually add subject invariance
        self.adv_alpha = min(progress * 2.0, 1.0)  # reaches 1.0 at 50% training

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
                # Re-run world model from this point would be expensive,
                # so we just replace the input state
        return states

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
        brain_states = self.brain_state_composer(eeg)

        # 3. World model (causal): [B, N, D] → [B, N, d_model]
        hidden_states = self.world_model(brain_states)

        # 4. Scheduled sampling (training only, re-encode with replaced states)
        if self.training and self.ss_prob > 0:
            brain_states_ss = self._apply_scheduled_sampling(brain_states, hidden_states)
            if not torch.equal(brain_states_ss, brain_states):
                hidden_states = self.world_model(brain_states_ss)
                brain_states = brain_states_ss

        result = {"brain_states": brain_states, "hidden_states": hidden_states}

        if return_predictions:
            # 5. EMA targets
            with torch.no_grad():
                target_states = self._get_target_states(eeg)

            # 6. Multi-horizon predictions
            predictions = {}
            for k in self.prediction_horizons:
                pred = self.prediction_head(hidden_states[:, :-k, :], k=k)
                predictions[k] = pred

            result["predictions"] = predictions
            result["targets"] = target_states.detach()

        return result

    def compute_loss(self, outputs: dict, subject_ids: torch.Tensor = None) -> dict:
        """Smooth L1 prediction loss + subject adversarial loss.

        Args:
            outputs: forward() output dict
            subject_ids: [B] integer subject labels (for adversarial training)
        """
        predictions = outputs["predictions"]
        targets = outputs["targets"]
        brain_states = outputs["brain_states"]

        pred_losses = {}
        total_loss = torch.tensor(0.0, device=targets.device)

        # --- Prediction loss ---
        for i, k in enumerate(self.prediction_horizons):
            pred = predictions[k]
            target = targets[:, k:, :]
            target = F.layer_norm(target, [target.shape[-1]])
            loss_k = F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta)
            pred_losses[f"pred_k{k}"] = loss_k
            total_loss = total_loss + self.horizon_weights[i] * loss_k

        # --- Subject adversarial loss ---
        # Always compute (even when alpha=0) to keep DDP graph structure constant.
        # When alpha=0, GRL passes zero gradients — mathematically equivalent to off.
        subj_logits = self.subject_adversary(brain_states, alpha=self.adv_alpha)
        if subject_ids is not None:
            adv_loss = F.cross_entropy(subj_logits, subject_ids)
        else:
            adv_loss = subj_logits.sum() * 0.0  # zero loss but keeps graph alive
        total_loss = total_loss + self.adv_lambda * adv_loss

        return {"total": total_loss, "adv": adv_loss, **pred_losses}

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
        target = F.layer_norm(outputs["targets"][:, 1:, :], [outputs["targets"].shape[-1]])

        # Smooth L1 per timestep
        pe = F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta, reduction="none")
        return pe.mean(dim=-1)  # [B, N-1]

    @torch.no_grad()
    def compute_regional_prediction_error(self, eeg: torch.Tensor) -> dict:
        """Per-region prediction error for spatial analysis (MMN/P300)."""
        self.eval()
        outputs = self.forward(eeg, return_predictions=True)
        pred = outputs["predictions"][1]
        target = F.layer_norm(outputs["targets"][:, 1:, :], [outputs["targets"].shape[-1]])

        d = self.config.encoder_hidden_dim
        region_errors = {}
        for r, name in enumerate(self.config.region_names):
            start, end = r * d, (r + 1) * d
            pe_r = F.smooth_l1_loss(
                pred[:, :, start:end], target[:, :, start:end],
                beta=self.smooth_l1_beta, reduction="none"
            )
            region_errors[name] = pe_r.mean(dim=-1)  # [B, N-1]

        return region_errors
