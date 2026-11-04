"""SEED-VIG vigilance estimation (PERCLOS regression): linear probe + fine-tuning.

Metrics as CBraMod: Pearson correlation, R2, RMSE on the test sessions.
  lp : frozen encoder, mean-pooled features, ridge regression; alpha chosen on
       the validation sessions (R2), refit on train, test scored once.
       Deterministic, so several checkpoints share one data load.
  ft : full fine-tuning (LaBraM-style layer decay, as eval_faced.py), MSE loss,
       epoch selected on validation R2 (as CBraMod's train_for_regression).

    python eval_seedvig.py --mode lp --checkpoints main=.../checkpoint_ep16.pt \
        --random_init --output lp.json
    python eval_seedvig.py --mode ft --checkpoints main=.../checkpoint_ep16.pt \
        --seed 0 --output ft_s0.json
"""
import argparse
import copy
import json
import os
import random as _pyrandom
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dataset_seedvig import cbramod_split, channel_slot_weights, load_xy
from eval_mumtaz import extract_features
from eval_tuh_clinical import (
    load_pretrained, build_random_init,
    _build_labram_optimizer, _build_cosine_warmup_scheduler,
)

# wide grid: with 512-d features from 13k highly correlated 8-s windows the
# validation optimum sat at the old upper edge (1e5) for our encoder
ALPHAS = [10.0 ** e for e in range(-2, 9)]


def metrics(y, pred):
    return {"pearson": float(np.corrcoef(y, pred)[0, 1]),
            "r2": float(r2_score(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5)}


def map_channels(model):
    """Pretrained 19 slot embeddings -> 17 SEED-VIG leads (see dataset_seedvig)."""
    W = torch.from_numpy(channel_slot_weights()).to(model.pos_channel.device)
    with torch.no_grad():
        src = model.pos_channel[0, :W.shape[1], 0, :].clone()          # [19, D]
        model.pos_channel[0, :W.shape[0], 0, :] = W @ src
    return model


def get_data(root, cache_dir, raw=False, bandpass=(0.5, 45.0)):
    """Ours: 256 Hz, band-passed (default 0.5-45 Hz), robust-scaled. raw=True:
    CBraMod/LaBraM input (200 Hz raw µV, no filter; the encoders divide by 100)."""
    lo, hi = bandpass
    cache = Path(cache_dir) / ("seedvig_numsplit_200hz_raw_uv.npz" if raw
                               else f"seedvig_numsplit_256hz_bp{lo:g}-{hi:g}_robust.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: (d[f"X_{k}"], d[f"y_{k}"]) for k in ("train", "val", "test")}, d["split"].item()
    split = cbramod_split(root)
    kw = dict(sample_rate=200, bandpass=None, robust=False) if raw else {"bandpass": bandpass}
    data = {k: load_xy(root, split[k], **kw)[:2] for k in ("train", "val", "test")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")   # parallel jobs
    np.savez(tmp, split=np.array(split, dtype=object),
             **{f"{a}_{k}": v for k, (X, y) in data.items() for a, v in (("X", X), ("y", y))})
    tmp.replace(cache)
    return data, split


def baseline_features(name, X, device, bs=64):
    """Mean-pooled features of an official foundation model (eval_linear_probe_v2
    adapters), all 17 SEED-VIG leads at their own positions."""
    from dataset_seedvig import VIG_CHANNELS
    from eval_linear_probe_v2 import cbramod_encoder, labram_encoder
    enc = {"cbramod": cbramod_encoder, "labram": labram_encoder}[name](device)[0]
    chans = list(range(X.shape[2]))
    kw = {"names": VIG_CHANNELS} if name == "labram" else {}
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(enc(torch.from_numpy(X[i:i + bs]).to(device), chans, **kw).float().cpu())
    return torch.cat(out).numpy()


def linear_probe(model, data, device, feats=None):
    F = feats or {k: extract_features(model, X, device) for k, (X, _) in data.items()}
    sc = StandardScaler().fit(F["train"])
    F = {k: sc.transform(v) for k, v in F.items()}
    y = {k: v[1] for k, v in data.items()}
    best = max(ALPHAS, key=lambda a: r2_score(
        y["val"], Ridge(alpha=a).fit(F["train"], y["train"]).predict(F["val"])))
    reg = Ridge(alpha=best).fit(F["train"], y["train"])
    out = {"test": metrics(y["test"], reg.predict(F["test"])),
           "val": metrics(y["val"], reg.predict(F["val"])), "alpha": best}
    print(f"    LP alpha={best:g} test {out['test']}", flush=True)
    return out


@torch.no_grad()
def _predict(model, head, X, device, bs=128):
    model.eval(); head.eval()
    out = [head(model._encode(model._tokenize(torch.from_numpy(X[i:i + bs]).to(device)))
                .mean(1)).squeeze(-1).float().cpu() for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def finetune(base_model, data, device, max_epochs=50, batch_size=64, base_lr=5e-4,
             weight_decay=0.05, layer_decay=0.65, warmup_epochs=5):
    model = copy.deepcopy(base_model)
    head = nn.Sequential(nn.BatchNorm1d(model.d_model), nn.Linear(model.d_model, 1)).to(device)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = data["train"], data["val"], data["test"]
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = _build_labram_optimizer(model, head, base_lr=base_lr, weight_decay=weight_decay,
                                        layer_decay=layer_decay, head_lr_mult=1.0)
    scheduler = _build_cosine_warmup_scheduler(optimizer, steps_per_epoch=len(loader),
                                               warmup_epochs=warmup_epochs,
                                               total_epochs=max_epochs)
    crit = nn.MSELoss()
    best, best_state = -np.inf, None
    for ep in range(max_epochs):
        model.train(); head.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = crit(head(model._encode(model._tokenize(bx)).mean(1)).squeeze(-1), by)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 3.0)
            optimizer.step(); scheduler.step()
        v = metrics(y_val, _predict(model, head, X_val, device))
        if v["r2"] > best:
            best = v["r2"]
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
        print(f"      ep{ep + 1:03d} val r2={v['r2']:.4f} pearson={v['pearson']:.4f} "
              f"best_r2={best:.4f}", flush=True)
    model.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    m = metrics(y_te, _predict(model, head, X_te, device))
    m["best_val_r2"] = float(best)
    print(f"    FT test {m}", flush=True)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", default=[], help="name=path ...")
    p.add_argument("--baselines", nargs="*", default=[],
                   choices=["cbramod", "labram", "reve_base", "reve_large", "csbrain"],
                   help="lp only: official foundation-model weights")
    p.add_argument("--root", default="/home/pxieaf/home2/datasets/SEED-VIG")
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--mode", choices=["lp", "ft"], required=True)
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--bandpass", type=float, nargs=2, default=[0.5, 45.0],
                   help="input band for our encoder; e.g. 2 45 removes most blink/EOG energy")
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    _pyrandom.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, split = get_data(a.root, a.cache_dir, bandpass=tuple(a.bandpass))
    print("SEED-VIG", {k: v[0].shape for k, v in data.items()}, flush=True)

    res = {"mode": a.mode, "seed": a.seed, "bandpass": a.bandpass, "split": split,
           "results": {}}
    out = Path(a.output)
    for spec in a.checkpoints:
        name, ck = spec.split("=", 1)
        model, cls, _, n_ch, ck_args = load_pretrained(ck, device)
        runs = [(name, map_channels(model))]
        if a.random_init and not res["results"].get("random_init"):
            torch.manual_seed(0)
            runs.append(("random_init", build_random_init(cls, n_ch, ck_args, device)))
        for nm, m in runs:
            print(f"  [{nm}]", flush=True)
            res["results"][nm] = (linear_probe(m, data, device) if a.mode == "lp" else
                                  finetune(m, data, device, a.max_epochs, a.batch_size))
            res["results"][nm]["checkpoint"] = ck if nm != "random_init" else None
            json.dump(res, open(out, "w"), indent=2)          # incremental
        del model; torch.cuda.empty_cache()
    if a.baselines:
        assert a.mode == "lp"
        raw, _ = get_data(a.root, a.cache_dir, raw=True)
        for b in a.baselines:
            print(f"  [{b}]", flush=True)
            if b.startswith("reve"):                 # two frozen readouts, one forward pass
                from dataset_seedvig import VIG_CHANNELS, N_PER_SESSION
                from eval_linear_probe_v2 import reve_features
                per = {k: reve_features(b.split("_")[1], X, VIG_CHANNELS,
                                        np.arange(len(X)) // N_PER_SESSION, device)
                       for k, (X, _) in raw.items()}
                for pl in ("mean", "attn"):
                    res["results"][f"{b}_{pl}"] = linear_probe(
                        None, raw, device, feats={k: per[k][pl] for k in per})
                json.dump(res, open(out, "w"), indent=2)
                continue
            if b == "csbrain":
                from dataset_seedvig import VIG_CHANNELS
                from eval_linear_probe_v2 import csbrain_encoder
                enc = csbrain_encoder(device, "seedvig")[0]
                with torch.no_grad():
                    feats = {k: np.concatenate([enc(torch.from_numpy(X[i:i + 64]).to(device), list(range(X.shape[2])),
                                                    names=VIG_CHANNELS).float().cpu().numpy()
                                                for i in range(0, len(X), 64)]) for k, (X, _) in raw.items()}
            else:
                feats = {k: baseline_features(b, X, device) for k, (X, _) in raw.items()}
            res["results"][b] = linear_probe(None, raw, device, feats=feats)
            json.dump(res, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
