# BrainWM v3: JEPA-Inspired Upgrade Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix deepcopy crash, add Brain-JEPA-inspired region masking, EEG-specific augmentation, and wider prediction horizons to make BrainWM a competitive EEG foundation model.

**Architecture:** BrainStateComposer gains a `return_content_only` flag (fixes deepcopy + gives position-free targets). New RegionMaskPredictor cross-attends over unmasked regions to predict masked ones (Brain-JEPA Cross-ROI analog). EEGAugmentation gets domain-specific perturbations (EchoJEPA-inspired). Prediction horizons widen to [1, 3, 5] for harder temporal prediction.

**Tech Stack:** PyTorch, DDP, Mamba SSM (custom impl)

---

## File Map

| File | Changes |
|---|---|
| `config.py` | Add region masking params, wider horizons, new augmentation params |
| `model.py:30-73` | EEGAugmentation: add electrode pop, powerline noise, slow drift |
| `model.py:265-336` | BrainStateComposer: add `return_content_only` param, remove `_content_states` |
| `model.py:531-557` | EMAEncoder: pass `return_content_only=True` |
| `model.py` (new ~340) | New RegionMaskPredictor class |
| `model.py:564-697` | BrainWM: add region masking in forward(), fix _get_target_states() |
| `model.py:698-774` | compute_loss: add region masking loss term |
| `train.py:97,130-142` | Log region masking loss (`rmask`) |

---

### Task 1: Fix deepcopy crash + content-only targets

**Files:**
- Modify: `model.py:265-336` (BrainStateComposer)
- Modify: `model.py:531-557` (EMAEncoder)
- Modify: `model.py:634-644` (_get_target_states)

The deepcopy crash happens because `_content_states` is a non-leaf tensor stored on the module. Fix: remove `_content_states`, add `return_content_only` parameter instead.

- [ ] **Step 1: Modify BrainStateComposer.forward()**

Replace lines 297-336 in `model.py`:

```python
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
    N = T // S  # number of 100ms states

    # ---- Batched channel encoding (no Python loops) ----
    eeg_windowed = eeg[:, :N * S, :].reshape(B, N, S, C)
    eeg_flat = eeg_windowed.permute(0, 1, 3, 2).reshape(B * N * C, S)
    z_flat = self.channel_encoder(eeg_flat)         # [B*N*C, d]
    z_all = z_flat.reshape(B, N, C, self.d)         # [B, N, C, d]

    # ---- Batched regional aggregation ----
    z_bn = z_all.reshape(B * N, C, self.d)
    region_latents = self.regional_agg(z_bn)        # [B*N, R, d]
    region_latents = region_latents.reshape(B, N, self.n_regions, self.d)

    # Add region embedding: [1, 1, R, d]
    region_latents = region_latents + self.region_embedding.unsqueeze(0).unsqueeze(0)

    # Concat regions: [B, N, R*d] = [B, N, D]
    content_states = region_latents.reshape(B, N, -1)

    if return_content_only:
        return content_states

    # Add temporal position embedding (only for online encoder path)
    pos = torch.arange(N, device=content_states.device)
    states = content_states + self.temporal_embedding(pos).unsqueeze(0)
    return states
```

- [ ] **Step 2: Modify EMAEncoder.__call__()**

Replace lines 555-557 in `model.py`:

```python
def __call__(self, *args, **kwargs):
    self.target_encoder.eval()
    return self.target_encoder(*args, return_content_only=True, **kwargs)
```

- [ ] **Step 3: Simplify _get_target_states()**

Replace lines 634-644 in `model.py`:

```python
@torch.no_grad()
def _get_target_states(self, eeg: torch.Tensor) -> torch.Tensor:
    """Return content-only targets (no temporal position embedding).

    EMA encoder's __call__ passes return_content_only=True to
    BrainStateComposer, so returned states have no position info.
    This prevents InfoNCE from being solved by position matching.
    """
    self._init_ema()
    return self.ema_encoder(eeg)
```

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('model.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

---

### Task 2: Add RegionMaskPredictor module

**Files:**
- Modify: `model.py` (insert new class after BrainStateComposer, around line 340)

- [ ] **Step 1: Add RegionMaskPredictor class**

Insert after the BrainStateComposer class (before SubjectAdversary):

```python
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
        d = config.encoder_hidden_dim  # 128
        self.d = d

        # Learnable mask token per region
        self.mask_tokens = nn.Parameter(torch.randn(config.n_regions, d) * 0.02)

        # Cross-attention: masked queries attend to unmasked regions
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

        # Queries: mask tokens + region embeddings for masked regions
        q = self.mask_tokens[masked_indices] + region_embedding[masked_indices]
        q = q.unsqueeze(0).expand(BN, -1, -1)  # [BN, R_masked, d]

        # Cross-attention
        out, _ = self.cross_attn(q, unmasked_regions, unmasked_regions)
        out = self.norm1(out + q)
        out = out + self.ffn(self.norm2(out))
        return out
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('model.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

---

### Task 3: Add region masking to BrainWM forward pass

**Files:**
- Modify: `model.py` BrainWM.__init__() and BrainWM.forward()

- [ ] **Step 1: Update BrainWM.__init__()**

Add after `self.prediction_head = PredictionHead(config)`:

```python
# Region mask predictor (Brain-JEPA Cross-ROI inspired)
self.region_mask_predictor = RegionMaskPredictor(config)
self.region_mask_prob = 0.5     # probability of applying region masking
self.region_mask_lambda = 1.0   # region prediction loss weight
```

- [ ] **Step 2: Add _apply_region_masking method to BrainWM**

Insert before `forward()`:

```python
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
        "predicted_regions": predicted,
        "original_regions": original_regions,
    }
```

- [ ] **Step 3: Update BrainWM.forward() to use region masking**

The forward method needs to:
1. Get content_states alongside brain_states
2. Apply region masking before world model
3. Pass region_mask_info through outputs

Replace the forward method body (lines 665-708):

```python
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

    # 3. Region masking (training only) — mask before world model
    result = {"brain_states": brain_states}
    wm_input = brain_states

    if self.training and return_predictions:
        region_mask_info = self._apply_region_masking(brain_states, content_states)
        if region_mask_info is not None:
            wm_input = region_mask_info["masked_states"]
            result["region_mask_info"] = region_mask_info

    # 4. World model (causal): [B, N, D] → [B, N, d_model]
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
```

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('model.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

---

### Task 4: Add region masking loss to compute_loss

**Files:**
- Modify: `model.py` compute_loss()

- [ ] **Step 1: Add region masking loss after VICReg block**

Insert after the VICReg loss block (before adversarial loss):

```python
# --- Region masking loss (Brain-JEPA Cross-ROI) ---
region_mask_loss = torch.tensor(0.0, device=targets.device)
if "region_mask_info" in outputs:
    info = outputs["region_mask_info"]
    pred_r = info["predicted_regions"]      # [B, N, n_mask, d]
    orig_r = info["original_regions"]       # [B, N, n_mask, d]

    # Get target regions from EMA targets
    d = self.config.encoder_hidden_dim
    masked_idx = info.get("masked_indices", None)
    if masked_idx is not None:
        target_regions = torch.stack(
            [targets[:, :, r.item()*d:(r.item()+1)*d] for r in masked_idx],
            dim=2,
        )  # [B, N, n_mask, d]
    else:
        target_regions = orig_r

    # Cosine similarity loss
    pred_flat = F.normalize(pred_r.reshape(-1, d), dim=-1)
    target_flat = F.normalize(target_regions.detach().reshape(-1, d), dim=-1)
    region_mask_loss = 2.0 - 2.0 * (pred_flat * target_flat).sum(-1).mean()

total_loss = total_loss + self.region_mask_lambda * region_mask_loss
```

- [ ] **Step 2: Update return dict**

Change the return statement to include `rmask`:

```python
return {
    "total": total_loss, "adv": adv_loss,
    "var": var_loss, "cov": cov_loss,
    "rmask": region_mask_loss,
    **pred_losses,
}
```

- [ ] **Step 3: Store masked_indices in _apply_region_masking return dict**

In `_apply_region_masking`, add to the returned dict:

```python
return {
    "masked_states": masked_states,
    "masked_indices": masked_idx,       # <-- add this
    "predicted_regions": predicted,
    "original_regions": original_regions,
}
```

(This was already included in the Task 3 code — verify it's there.)

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('model.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

---

### Task 5: EEG-specific augmentations

**Files:**
- Modify: `model.py:30-73` (EEGAugmentation)
- Modify: `config.py` (new augmentation params)

- [ ] **Step 1: Add config parameters**

Add to `config.py` after `aug_channel_dropout_p`:

```python
aug_electrode_pop_prob: float = 0.15     # prob of spike artifact per trial
aug_powerline_prob: float = 0.2          # prob of 50Hz interference
aug_slow_drift_prob: float = 0.15        # prob of slow baseline drift
```

- [ ] **Step 2: Update EEGAugmentation**

Replace the EEGAugmentation class in `model.py`:

```python
class EEGAugmentation(nn.Module):
    """EEG data augmentation with domain-specific perturbations.

    Generic augmentations (v2):
      - Time shift, amplitude scaling, Gaussian noise, channel dropout

    EEG-specific augmentations (v3, EchoJEPA-inspired domain adaptation):
      - Electrode pop: sudden spike on a single channel
      - Powerline interference: 50Hz sinusoidal noise
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
            spike = amplitude * torch.exp(-0.5 * ((torch.arange(t_start, t_end, device=x.device).float() - t_center) / max(width / 3, 1)) ** 2)
            x[:, t_start:t_end, ch] = x[:, t_start:t_end, ch] + spike.unsqueeze(0)

        # 6. Powerline interference: 50Hz sinusoidal on all channels
        if torch.rand(1).item() < self.powerline_prob:
            t = torch.arange(T, device=x.device, dtype=x.dtype) / self.sample_rate
            freq = 50.0 if torch.rand(1).item() < 0.5 else 60.0
            phase = torch.rand(1, device=x.device) * 2 * math.pi
            amplitude = x.std() * torch.empty(1, device=x.device).uniform_(0.05, 0.2)
            interference = amplitude * torch.sin(2 * math.pi * freq * t + phase)
            x = x + interference.unsqueeze(0).unsqueeze(-1)

        # 7. Slow baseline drift
        if torch.rand(1).item() < self.drift_prob:
            n_ch = torch.randint(1, max(2, C // 8), (1,)).item()
            channels = torch.randperm(C)[:n_ch]
            t = torch.arange(T, device=x.device, dtype=x.dtype) / T
            freq = torch.empty(1, device=x.device).uniform_(0.1, 0.5)
            drift = x.std() * 0.5 * torch.sin(2 * math.pi * freq * t)
            x[:, :, channels] = x[:, :, channels] + drift.unsqueeze(0).unsqueeze(-1)

        return x
```

- [ ] **Step 3: Verify syntax of both files**

Run: `python -c "import ast; ast.parse(open('model.py', encoding='utf-8').read()); ast.parse(open('config.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

---

### Task 6: Widen prediction horizons + update config

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Update prediction_horizons and horizon_weights**

```python
prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 5])
horizon_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
```

This changes k1/k2/k3 → k1/k3/k5 in the logs. k5 = 500ms ahead, much harder than k3 = 300ms.

- [ ] **Step 2: Update train.py logging to match new horizon names**

In `train.py`, change the print format:

```python
print_main(
    f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
    f"loss={losses['total']:.4f} "
    + " ".join(f"k{k}={losses.get(f'pred_k{k}', 0):.4f}" for k in config.prediction_horizons)
    + f" adv={losses.get('adv', 0):.4f} "
    f"var={losses.get('var', 0):.4f} "
    f"cov={losses.get('cov', 0):.4f} "
    f"rmask={losses.get('rmask', 0):.4f} "
    f"lr={lr:.2e} ema={ema_m:.4f} \u03b1={raw_model.adv_alpha:.2f}"
)
```

- [ ] **Step 3: Update total_losses accumulator in train_one_epoch**

```python
total_losses = {"total": 0, "adv": 0, "var": 0, "cov": 0, "rmask": 0}
for k in config.prediction_horizons:
    total_losses[f"pred_k{k}"] = 0
```

- [ ] **Step 4: Verify syntax of both files**

Run: `python -c "import ast; ast.parse(open('model.py', encoding='utf-8').read()); ast.parse(open('train.py', encoding='utf-8').read()); ast.parse(open('config.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

---

### Task 7: Update docstring and clean up stale comments

**Files:**
- Modify: `model.py:1-15` (module docstring)

- [ ] **Step 1: Update module docstring**

```python
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
```

---

## Expected Training Behavior After All Changes

```
Epoch 1: k1~3.6 k3~3.6 k5~3.6 (random, log(35)≈3.56)
          var~0.5→0.1 (decreasing = good)
          cov~10→1 (decreasing = good)
          rmask~1.5 (cosine loss, should decrease)
          adv~4.7 (random, alpha=0)

Epoch 15: k1<k3<k5 (near predictions easier than far)
           alpha starts ramping up

Epoch 50: k1~1.5 k3~2.5 k5~3.0 (prediction learned but not trivial)
           var≈0, cov≈0 (representations well-spread)
           rmask~0.5 (cross-region prediction improving)
           adv~4.7 (adversary still confused = subject-invariant)
```
