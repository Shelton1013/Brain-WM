"""Noise robustness of frozen features (ISRUC / HMC, seq2seq sleep staging).

Tests the method's motivation directly: a waveform-reconstruction target must
spend capacity modelling nuisance signal, a band-power target is built to
ignore it. Protocol: train the sleep sequence head on CLEAN frozen features
(same head/protocol as the frozen probe), then evaluate on the clean test set
and on test sets corrupted with controlled artifacts added to the model input:

  emg   : Gaussian noise band-passed 20-100 Hz (muscle artifact)
  line  : 50 Hz mains sinusoid, random phase per channel
  drift : 0.05-0.5 Hz band-limited noise (electrode / sweat drift)
  white : broadband Gaussian noise

Amplitudes are relative to each channel's std in the clean input (the input is
robust-scaled, so ~1). Zero-padded (absent) channels stay zero. Noise is seeded
and identical across checkpoints, so models are compared on the same corruption.

    python eval_noise_robustness.py --dataset isruc --checkpoint .../checkpoint_ep16.pt \
        --data_dir /home/pxieaf/home2/datasets/isruc/subgroupI_official --output out.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score

from eval_tuh_clinical import load_pretrained
from eval_sleep_seq2seq import SeqHead, compute_metrics, precompute_seq_features, _encode_epochs
from eval_label_efficiency_seq2seq import load_seqs

FS = 256
NOISES = {"emg": [0.5, 1.0, 2.0], "line": [0.5, 1.0, 2.0],
          "drift": [1.0, 2.0, 4.0], "white": [0.5, 1.0, 2.0]}
# --weak: broadband noise at >=1x signal std drives every model to chance, so
# EMG / white are also swept at sub-signal amplitudes
NOISES_WEAK = {"emg": [0.1, 0.2, 0.3], "white": [0.1, 0.2, 0.3]}
_SOS = {"emg": butter(4, [20, 100], btype="band", fs=FS, output="sos"),
        "drift": butter(2, [0.05, 0.5], btype="band", fs=FS, output="sos")}


def corrupt(x, kind, level, rng):
    """x [B, S, T, C] float32 numpy -> corrupted copy."""
    B, S, T, C = x.shape
    std = x.std(axis=2, keepdims=True)                     # per epoch & channel
    present = std > 1e-6                                   # zero-padded channels stay 0
    if kind == "line":
        t = np.arange(T) / FS
        phase = rng.uniform(0, 2 * np.pi, size=(B, S, 1, C))
        n = np.sin(2 * np.pi * 50.0 * t[None, None, :, None] + phase) * np.sqrt(2)
    else:
        n = rng.standard_normal((B, S, T, C)).astype(np.float32)
        if kind in _SOS:
            n = sosfiltfilt(_SOS[kind], n, axis=2)
        n = n / (n.std(axis=2, keepdims=True) + 1e-8)      # unit-std noise
    return (x + level * std * n * present).astype(np.float32)


def features(model, X, device, kind=None, level=0.0, seed=0, batch_size=16):
    if kind is None:
        return precompute_seq_features(model, X, device, batch_size)
    rng = np.random.default_rng(seed)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = corrupt(X[i:i + batch_size], kind, level, rng)
            out.append(_encode_epochs(model, torch.from_numpy(xb).to(device)).cpu())
    return torch.cat(out)


def train_head(Ftr, ytr, Fva, yva, n_classes, d_model, device, seed, max_epochs=50,
               batch_size=256):
    """Same recipe as eval_sleep_seq2seq.run_head_only, but returns the head."""
    torch.manual_seed(seed)
    head = SeqHead(d_model, n_classes).to(device)
    tr = DataLoader(TensorDataset(Ftr, torch.from_numpy(ytr)), batch_size=batch_size,
                    shuffle=True, drop_last=True)
    cw = torch.bincount(torch.from_numpy(ytr).reshape(-1).long(), minlength=n_classes).float()
    cw = 1.0 / cw.clamp(min=1); cw = (cw / cw.sum() * n_classes).to(device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    opt = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * len(tr),
                                                       eta_min=1e-6)
    best, best_state, no_imp = 0.0, None, 0
    for _ in range(max_epochs):
        head.train()
        for bf, by in tr:
            loss = crit(head(bf.to(device)).reshape(-1, n_classes), by.to(device).reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        val_ba = balanced_accuracy_score(yva.reshape(-1), predict(head, Fva, device))
        if val_ba > best:
            best, no_imp = val_ba, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= 15:
                break
    head.load_state_dict(best_state)
    return head


@torch.no_grad()
def predict(head, F, device, batch_size=256):
    head.eval()
    return torch.cat([head(F[i:i + batch_size].to(device)).argmax(-1).cpu().reshape(-1)
                      for i in range(0, len(F), batch_size)]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["isruc", "hmc"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust")
    p.add_argument("--n_reps", type=int, default=3)
    p.add_argument("--weak", action="store_true", help="use NOISES_WEAK levels")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, *_ = load_pretrained(a.checkpoint, device)
    Xtr, ytr, Xva, yva, Xte, yte, n_classes = load_seqs(a.dataset, a)
    t0 = time.time()
    Ftr, Fva = features(model, Xtr, device), features(model, Xva, device)
    tests = {("clean", 0.0): features(model, Xte, device)}
    for k, (kind, levels) in enumerate((NOISES_WEAK if a.weak else NOISES).items()):
        for j, lv in enumerate(levels):
            tests[(kind, lv)] = features(model, Xte, device, kind, lv, seed=1000 * k + j)
    print(f"features ({(time.time() - t0) / 60:.1f} min)", flush=True)

    per = {key: [] for key in tests}
    for r in range(a.n_reps):
        head = train_head(Ftr, ytr, Fva, yva, n_classes, model.d_model, device, seed=r)
        for key, Fte in tests.items():
            per[key].append(compute_metrics(yte.reshape(-1), predict(head, Fte, device)))

    res = {"checkpoint": a.checkpoint, "dataset": a.dataset, "results": {}}
    clean = np.mean([m["balanced_accuracy"] for m in per[("clean", 0.0)]])
    for (kind, lv), ms in per.items():
        ba = [m["balanced_accuracy"] for m in ms]
        res["results"][f"{kind}@{lv}"] = {
            "balanced_accuracy": [float(np.mean(ba)), float(np.std(ba))],
            "cohen_kappa": float(np.mean([m["cohen_kappa"] for m in ms])),
            "drop_vs_clean": float(np.mean(ba) - clean)}
        print(f"  {kind:>5}@{lv:<4} BA {np.mean(ba):.4f}  drop {np.mean(ba) - clean:+.4f}",
              flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.output, "w"), indent=2)
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
