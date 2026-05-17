"""BrainWM v2 configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BrainWMConfig:
    # ---------- EEG input ----------
    sample_rate: int = 256          # Hz, all data resampled to this
    state_duration_ms: int = 100    # v2: 100ms per state (v1 was 500ms)
    state_samples: int = 26         # sample_rate * state_duration_ms / 1000
    trial_duration_s: int = 4       # seconds per trial
    n_states_per_trial: int = 40    # v2: 40 states (v1 was 8)

    # ---------- Brain regions ----------
    n_regions: int = 5
    region_names: List[str] = field(default_factory=lambda: [
        "frontal", "central", "parietal", "temporal", "occipital"
    ])

    # ---------- Channel encoder ----------
    n_freq_filters: int = 5         # learnable filter bank bands
    filter_kernel_size: int = 17    # v2: shorter for 100ms window (~65ms)
    channel_attn_dim: int = 64      # temporal attention hidden dim
    channel_attn_heads: int = 2     # temporal attention heads
    channel_attn_layers: int = 2    # temporal attention layers
    encoder_hidden_dim: int = 128   # latent dim per region

    # ---------- Regional attention ----------
    region_query_dim: int = 128     # cross-attention query dim

    # ---------- Brain state ----------
    brain_state_dim: int = 640      # n_regions * encoder_hidden_dim

    # ---------- World model (Causal Mamba) ----------
    mamba_n_layers: int = 6
    mamba_d_model: int = 512
    mamba_d_state: int = 64
    mamba_expand: int = 2
    mamba_d_conv: int = 4

    # ---------- Prediction ----------
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 2, 3])
    predictor_hidden_dim: int = 512
    horizon_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])

    # ---------- EMA target encoder ----------
    ema_decay_start: float = 0.996
    ema_decay_end: float = 0.9999

    # ---------- Loss ----------
    smooth_l1_beta: float = 1.0     # v2: Smooth L1 instead of Cosine+VICReg

    # ---------- Data augmentation ----------
    aug_time_shift_samples: int = 6     # ±25ms at 256Hz
    aug_amplitude_scale: tuple = (0.8, 1.2)
    aug_gaussian_noise_std: float = 0.1  # relative to channel std
    aug_channel_dropout_p: float = 0.1

    # ---------- Scheduled sampling ----------
    scheduled_sampling_start: float = 0.0
    scheduled_sampling_end: float = 0.3

    # ---------- Training ----------
    batch_size: int = 64
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    n_epochs: int = 50
    warmup_epochs: int = 5
    gradient_clip: float = 3.0
    num_workers: int = 4

    # ---------- 10-20 electrode -> region mapping ----------
    @staticmethod
    def get_electrode_region_map() -> dict:
        """Standard 10-20/10-10 electrode to brain region assignment."""
        return {
            # Frontal (region 0)
            "Fp1": 0, "Fp2": 0, "F3": 0, "F4": 0,
            "F7": 0, "F8": 0, "Fz": 0, "Fpz": 0,
            "AF3": 0, "AF4": 0, "AF7": 0, "AF8": 0,
            "F1": 0, "F2": 0, "F5": 0, "F6": 0,
            # Central (region 1)
            "C3": 1, "C4": 1, "Cz": 1,
            "C1": 1, "C2": 1, "C5": 1, "C6": 1,
            "FC1": 1, "FC2": 1, "FC3": 1, "FC4": 1,
            "FC5": 1, "FC6": 1, "FCz": 1,
            "CP1": 1, "CP2": 1, "CP3": 1, "CP4": 1,
            "CP5": 1, "CP6": 1, "CPz": 1,
            # Parietal (region 2)
            "P3": 2, "P4": 2, "Pz": 2,
            "P1": 2, "P2": 2, "P5": 2, "P6": 2,
            "P7": 2, "P8": 2, "POz": 2,
            "PO3": 2, "PO4": 2, "PO7": 2, "PO8": 2,
            # Temporal (region 3)
            "T3": 3, "T4": 3, "T5": 3, "T6": 3,
            "T7": 3, "T8": 3,
            "FT7": 3, "FT8": 3, "FT9": 3, "FT10": 3,
            "TP7": 3, "TP8": 3, "TP9": 3, "TP10": 3,
            # Occipital (region 4)
            "O1": 4, "O2": 4, "Oz": 4,
            "PO9": 4, "PO10": 4, "Iz": 4,
        }
