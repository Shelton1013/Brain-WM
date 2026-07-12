# EEG-LeJEPA v3 — Model Usage Guide

How to load a pretrained checkpoint and use the encoder to extract features for
your own downstream EEG task (frozen probe or fine-tuning).

---

## 1. Quick facts

| Property | Value |
|---|---|
| **Sampling rate** | **256 Hz** (resample your data to this) |
| **Channels** | **19**, 10–20 montage, **fixed order** (see below) |
| **Window length** | **10 s → 2560 samples** (input time dim) |
| **Input shape** | `[B, T, C] = [batch, 2560, 19]`, `float32` |
| **Normalization** | per-recording **robust z-score** (median / IQR), per channel |
| **Token output** | `[B, C·Tp, D] = [B, 228, 512]` (19 ch × 12 time-patches × 512) |
| **Pooled feature** | `[B, 512]` (mean over tokens) — use this for downstream |
| **d_model** | 512 · **patch_len** 200 · **encoder** 12 layers criss-cross |

> The window may be longer than 10 s (up to `max_time_patches·patch_len = 64·200
> = 12800 samples = 50 s`) and channels fewer than 19, but the model was
> **pretrained on 19-ch / 10-s** — match that for best results.

---

## 2. Channel order (must match exactly)

```python
CANONICAL_19 = ['Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz',
                'T3','T4','T5','T6','P3','P4','Pz','O1','O2']
```

The last axis of the input must be these 19 channels **in this order**. If your
recording uses the newer names: `T3=T7, T4=T8, T5=P7, T6=P8`. Missing channels
should be zero-filled at the correct index (the model tolerates it, but expect
some degradation).

---

## 3. Required files

The checkpoint needs the model class definitions to load. Copy these from the
repo (no other repo code required for inference):

```
eeg_lejepa_v3.py      # EEGLeJEPA_v3 (the model)
eeg_lejepa_v2.py      # provides CrissCrossBlock, MLP (imported by v3)
regularizers.py       # provides distribution_reg (imported by v3)
```

Dependencies: `torch`, `numpy`, (`mne` only for reading EDF/preprocessing).

---

## 4. Load the model

```python
import torch
from eeg_lejepa_v3 import EEGLeJEPA_v3

def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = EEGLeJEPA_v3(
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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

model = load_model("checkpoint_ep15.pt")
```

---

## 5. Preprocess your EEG

Resample → pick the 19 channels in order → convert to µV → per-recording robust
scale → cut into 10-s windows.

```python
import numpy as np, mne

CANONICAL_19 = ['Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz',
                'T3','T4','T5','T6','P3','P4','Pz','O1','O2']

def preprocess(raw):
    """raw: an mne.io.Raw. Returns [n_windows, 2560, 19] float32."""
    raw = raw.copy().resample(256)
    # pick + reorder to canonical 19 (rename/zero-fill as needed for your data)
    raw.pick_channels(CANONICAL_19, ordered=True)
    sig = raw.get_data().T.astype(np.float32) * 1e6          # [T, 19] in µV

    # per-recording robust z-score, per channel
    med = np.median(sig, axis=0, keepdims=True)
    q75 = np.percentile(sig, 75, axis=0, keepdims=True)
    q25 = np.percentile(sig, 25, axis=0, keepdims=True)
    sig = (sig - med) / ((q75 - q25) / 1.349 + 1e-6)

    # non-overlapping 10-s windows
    W = 2560
    n = sig.shape[0] // W
    return sig[:n * W].reshape(n, W, 19)                     # [n, 2560, 19]
```

> Match the pretraining preprocessing: 0.5–45 Hz band-pass is baked into the
> pretrain data (v3-filter checkpoints). You do **not** need to re-filter for a
> frozen probe, but keeping your downstream band-pass ≈ 0.5–45 Hz reduces
> train/test mismatch.

---

## 6. Extract features (frozen probe)

```python
@torch.no_grad()
def extract_features(model, x, device="cuda", batch=64):
    """x: [N, 2560, 19] np.float32  ->  [N, 512] features."""
    feats = []
    x = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(x), batch):
        b = x[i:i+batch].to(device)
        tok = model._tokenize(b)             # [b, 228, 512] token features
        f = model._encode(tok).mean(1)       # [b, 512]  per-window feature
        feats.append(f.cpu())
    return torch.cat(feats).numpy()

# frozen linear probe with sklearn
from sklearn.linear_model import LogisticRegression
Ftr = extract_features(model, X_train)       # [N_tr, 512]
Fte = extract_features(model, X_test)         # [N_te, 512]
clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Ftr, y_train)
acc = clf.score(Fte, y_test)
```

`model._encode(model._tokenize(x))` returns `[B, 228, 512]` token embeddings;
`.mean(1)` pools to one `512-d` vector per window. Use the pooled vector as the
sample representation.

---

## 7. Fine-tune the whole model

Attach a head and train end-to-end (unfreeze the encoder):

```python
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder                     # the loaded EEGLeJEPA_v3
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, n_classes))
    def forward(self, x):                          # x: [B, 2560, 19]
        feats = self.encoder._encode(self.encoder._tokenize(x)).mean(1)  # [B,512]
        return self.head(feats)                    # [B, n_classes]

clf = Classifier(model, n_classes=5).to("cuda")
opt = torch.optim.AdamW(clf.parameters(), lr=1e-4, weight_decay=5e-2)
# standard training loop; CrossEntropyLoss ...
```

> Tip: for small downstream datasets, a **frozen probe** or **LP-FT** (freeze the
> encoder for a few warm-up epochs, then unfreeze at a low LR, e.g. 1e-5) often
> beats plain full fine-tuning, which can distort the pretrained features.

---

## 8. Shape reference

| Call | Shape | Meaning |
|---|---|---|
| input `x` | `[B, 2560, 19]` | 10 s, 256 Hz, 19 ch, robust-normalized |
| `model._tokenize(x)` | `[B, 228, 512]` | token embeddings (19·12 tokens) |
| `model._encode(tok)` | `[B, 228, 512]` | encoded tokens |
| `.mean(1)` | `[B, 512]` | **per-window feature (use this)** |

`228 = 19 channels × 12 time-patches` (`12 = 2560 / patch_len 200`).

---

## 9. Notes & gotchas

- **Input orientation is `[B, time, channel]`**, not `[B, channel, time]`.
- **Do not skip normalization** — the model expects robust-scaled input; raw µV
  will give poor features.
- The model is **channel-order sensitive** (positional channel embedding). Keep
  the canonical 19 order.
- `model.eval()` before inference (disables any stochastic behavior).
- `_tokenize` caches the encoded tensor internally so `_encode` reuses it — call
  them as a pair `model._encode(model._tokenize(x))` per batch.
- Checkpoints from different runs (no-filter / filter) share this identical
  interface; only the pretraining data preprocessing differs.
