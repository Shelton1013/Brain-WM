"""Smoke test for EEG-LeJEPA v3 (filterbank + cross-frequency). Run before pilot.

    python smoke_v3.py
Checks: forward runs, cf/jepa losses are non-trivial (>0), backward works,
downstream _tokenize/_encode produce [B, C*Tp, D], param count sane.
"""
import torch
from eeg_lejepa_v3 import EEGLeJEPA_v3, count_params

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
B, T, C = 4, 2560, 19          # 10 s @ 256 Hz, 19 ch

m = EEGLeJEPA_v3(d_model=512, encoder_layers=12, n_bands=5, d_band=64,
                 patch_len=200, max_time_patches=64, max_channels=32).to(dev)
print(f"params: {count_params(m)/1e6:.1f}M")

x = torch.randn(B, T, C, device=dev)

# ── SSL forward + backward ──
out = m(x)
print("loss dict:", {k: float(v) for k, v in out.items()})
assert out["cf"].item() > 1e-6,   "CF loss is ~0 — spectral target trivial?"
assert out["jepa"].item() > 1e-6, "JEPA loss is ~0 — latent task trivial?"
out["total"].backward()
gsum = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
print(f"grad flows: total|grad|={gsum:.1f}  (filterbank grad="
      f"{m.tokenizer.filters.weight.grad.abs().sum().item():.3f})")
assert gsum > 0 and m.tokenizer.filters.weight.grad is not None

# ── downstream interface ──
m.eval()
with torch.no_grad():
    tok = m._tokenize(x)          # [B, C*Tp, D]
    feat = m._encode(tok).mean(1) # [B, D]
Tp = T // 200
print(f"_tokenize -> {tuple(tok.shape)} (expect [{B},{C*Tp},512])")
print(f"pooled feat -> {tuple(feat.shape)} (expect [{B},512])")
assert tok.shape == (B, C * Tp, 512) and feat.shape == (B, 512)

# ── filterbank frequency response sanity ──
w = m.tokenizer.filters.weight.detach().cpu()   # [N,1,K]
fft = torch.fft.rfft(w[:, 0], n=256).abs()
peak_hz = fft.argmax(dim=-1).tolist()            # index ~ Hz (fs=256, n=256)
print(f"filterbank peak freqs (Hz, init δθαβγ≈2/6/10/21/38): {peak_hz}")

print("\n✅ v3 smoke test PASSED")
