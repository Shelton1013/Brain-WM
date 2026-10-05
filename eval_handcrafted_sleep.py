"""Strong direct-feature baselines for sleep staging (ISRUC / HMC).

Answers "does self-supervised pretraining add anything over classical spectral
features?" with the strongest reasonable feature pipeline, under the SAME
split, labels and downstream sequence head as our frozen probe.

Feature sets (per EEG channel that is present; zero-padded channels skipped):
  basic : log absolute + relative power in 5 bands (the earlier baseline)
  rich  : 1-Hz log PSD 0.5-45 Hz, relative power in delta/theta/alpha/
          sigma(spindle)/beta/gamma, Hjorth activity/mobility/complexity,
          spectral entropy, 95% spectral edge, line length, kurtosis
Heads (features z-scored on train):
  seqhead  : the SAME 1-layer sequence Transformer as our frozen probe (matched)
  strong   : Linear->256 + 2-layer sequence Transformer (more capacity)
  gbdt     : gradient-boosted trees on each epoch +/-2 neighbours (classical)
Extras:
  --label_fracs : label-efficiency curve with the matched head
  --concat_ckpt : our frozen encoder features concatenated with rich features

    python eval_handcrafted_sleep.py --dataset isruc --data_dir ... \
        --trial_duration_s 10 --concat_ckpt .../checkpoint_ep16.pt --output out.json
"""
import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from eval_sleep_seq2seq import SeqHead, SEQ_LEN, compute_metrics

BANDS5 = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BANDS6 = [(0.5, 4), (4, 8), (8, 11), (11, 16), (16, 30), (30, 45)]


def datasets(a):
    kw = dict(sample_rate=a.sample_rate, trial_duration_s=a.trial_duration_s,
              normalization=a.normalization,
              cache_dir=a.cache_dir if a.trial_duration_s <= 10 else None)
    if a.dataset == "isruc":
        from dataset_isruc import ISRUCDataset, CBRAMOD_ISRUC_SPLITS as SP, N_CLASSES
        mk = lambda s: ISRUCDataset(data_dir=a.data_dir, subjects=s, **kw)
    else:
        from dataset_hmc import HMCDataset, make_hmc_split, N_CLASSES
        SP = make_hmc_split(a.data_dir)
        mk = lambda s: HMCDataset(a.data_dir, s, **kw)
    return {k: mk(SP[k]) for k in ("train", "val", "test")}, N_CLASSES


def epoch_features(x, fs, kind, chans):
    """x [T, 19] one epoch -> feature vector over the present channels."""
    x = x[:, chans].T.astype(np.float64)                         # [c, T]
    f, p = welch(x, fs=fs, nperseg=2 * fs, axis=-1)
    tot = p[:, (f >= 0.5) & (f < 45)].sum(-1, keepdims=True) + 1e-12
    if kind == "basic":
        ab = np.stack([p[:, (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS5], -1)
        return np.concatenate([np.log(ab + 1e-12), ab / tot], -1).ravel()
    edges = np.arange(0.5, 45.5, 1.0)
    psd1 = np.stack([p[:, (f >= lo) & (f < lo + 1)].mean(-1) for lo in edges[:-1]], -1)
    rel6 = np.stack([p[:, (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS6], -1) / tot
    d1, d2 = np.diff(x, axis=-1), np.diff(x, n=2, axis=-1)
    act = x.var(-1)
    mob = np.sqrt(d1.var(-1) / (act + 1e-12))
    comp = np.sqrt(d2.var(-1) / (d1.var(-1) + 1e-12)) / (mob + 1e-12)
    band = (f >= 0.5) & (f < 45)
    pn = p[:, band] / tot
    ent = -(pn * np.log(pn + 1e-12)).sum(-1)
    sef = f[band][np.argmax(np.cumsum(pn, -1) >= 0.95, axis=-1)]
    ll = np.abs(d1).mean(-1)
    ku = kurtosis(x, axis=-1)
    extra = np.stack([np.log(act + 1e-12), mob, comp, ent, sef, np.log(ll + 1e-12), ku], -1)
    return np.concatenate([np.log(psd1 + 1e-12), rel6, extra], -1).ravel()


def to_sequences(ds, feats):
    """Same grouping as eval_sleep_seq2seq.build_sequences, on features."""
    by = OrderedDict()
    for i, s in enumerate(ds.subject_ids):
        by.setdefault(s, []).append(i)
    F, Y = [], []
    for idx in by.values():
        for k in range(0, len(idx) - SEQ_LEN + 1, SEQ_LEN):
            ch = idx[k:k + SEQ_LEN]
            F.append(feats[ch]); Y.append([ds.labels[j] for j in ch])
    return np.stack(F).astype(np.float32), np.asarray(Y, np.int64)


class StrongHead(nn.Module):
    def __init__(self, d_in, n_classes, d=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(0.1))
        layer = nn.TransformerEncoderLayer(d, 8, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.cls = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x):
        return self.cls(self.enc(self.proj(x)))


def train_head(make, Ftr, ytr, Fva, yva, Fte, yte, n_classes, device, seed, epochs=50):
    """run_head_only's recipe (AdamW 5e-4, cosine, label smoothing, class
    weights, best val BA, patience 15) for any head."""
    torch.manual_seed(seed); np.random.seed(seed)
    head = make().to(device)
    tr = DataLoader(TensorDataset(torch.from_numpy(Ftr), torch.from_numpy(ytr)),
                    batch_size=256, shuffle=True, drop_last=len(Ftr) > 256)
    cw = torch.bincount(torch.from_numpy(ytr).reshape(-1), minlength=n_classes).float()
    cw = 1.0 / cw.clamp(min=1); cw = (cw / cw.sum() * n_classes).to(device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    opt = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=5e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr), eta_min=1e-6)
    pred = lambda F: torch.cat([head(torch.from_numpy(F[i:i + 256]).to(device)).argmax(-1).cpu()
                                for i in range(0, len(F), 256)]).numpy().reshape(-1)
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        head.train()
        for bx, by in tr:
            loss = crit(head(bx.to(device)).reshape(-1, n_classes), by.to(device).reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        head.eval()
        with torch.no_grad():
            va = balanced_accuracy_score(yva.reshape(-1), pred(Fva))
        if va > best:
            best, bad = va, 0
            state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    head.load_state_dict(state); head.eval()
    with torch.no_grad():
        return compute_metrics(yte.reshape(-1), pred(Fte))


def gbdt(Ftr, ytr, Fte, yte, ctx=2):
    """Per-epoch boosted trees with +/-ctx neighbour features (within sequence)."""
    def flat(F):
        pad = np.pad(F, ((0, 0), (ctx, ctx), (0, 0)), mode="edge")
        return np.concatenate([pad[:, k:k + F.shape[1]] for k in range(2 * ctx + 1)], -1) \
            .reshape(-1, F.shape[-1] * (2 * ctx + 1))
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         class_weight="balanced", random_state=0)
    clf.fit(flat(Ftr), ytr.reshape(-1))
    return compute_metrics(yte.reshape(-1), clf.predict(flat(Fte)))


def agg(ms):
    return {k: [float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))]
            for k in ("balanced_accuracy", "cohen_kappa", "weighted_f1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["isruc", "hmc"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    ap.add_argument("--sample_rate", type=int, default=256)
    ap.add_argument("--trial_duration_s", type=int, default=10)
    ap.add_argument("--normalization", default="per_recording_robust")
    ap.add_argument("--n_reps", type=int, default=3)
    ap.add_argument("--label_fracs", type=float, nargs="*", default=[0.10, 0.25, 0.50])
    ap.add_argument("--concat_ckpt", default=None)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = a.sample_rate

    t0 = time.time()
    dss, n_classes = datasets(a)
    first = np.stack(dss["train"].trials[:2000])
    chans = [c for c in range(first.shape[-1]) if np.abs(first[..., c]).max() > 0]
    print(f"present channels {chans}; loaded in {(time.time() - t0) / 60:.1f} min", flush=True)

    feats = {}
    for kind in ("basic", "rich"):
        seq = {}
        for split, ds in dss.items():
            E = np.stack([epoch_features(x, fs, kind, chans) for x in ds.trials])
            # flat epochs give NaN kurtosis / inf logs: neural heads collapse on
            # NaN (trees tolerate it), so sanitise before z-scoring
            E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
            seq[split] = to_sequences(ds, E)
        mu = seq["train"][0].reshape(-1, seq["train"][0].shape[-1]).mean(0)
        sd = seq["train"][0].reshape(-1, seq["train"][0].shape[-1]).std(0) + 1e-6
        feats[kind] = {k: ((F - mu) / sd, Y) for k, (F, Y) in seq.items()}
        print(f"{kind}: dim {mu.shape[0]}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    if a.concat_ckpt:
        from eval_tuh_clinical import load_pretrained
        model, *_ = load_pretrained(a.concat_ckpt, dev)
        seq = {}
        with torch.no_grad():
            for split, ds in dss.items():
                out = []
                for i in range(0, len(ds.trials), 256):
                    b = torch.from_numpy(np.stack(ds.trials[i:i + 256])).to(dev)
                    out.append(model._encode(model._tokenize(b)).mean(1).cpu().numpy())
                seq[split] = to_sequences(ds, np.concatenate(out))
        mu = seq["train"][0].reshape(-1, seq["train"][0].shape[-1]).mean(0)
        sd = seq["train"][0].reshape(-1, seq["train"][0].shape[-1]).std(0) + 1e-6
        ours = {k: ((F - mu) / sd, Y) for k, (F, Y) in seq.items()}
        feats["ours_frozen"] = ours
        feats["ours+rich"] = {k: (np.concatenate([ours[k][0], feats["rich"][k][0]], -1),
                                  ours[k][1]) for k in ours}
        del model; torch.cuda.empty_cache()

    def pad8(F):                      # SeqHead uses 8 attention heads
        r = (-F.shape[-1]) % 8
        return np.pad(F, ((0, 0), (0, 0), (0, r))) if r else F

    res = {"dataset": a.dataset, "trial_duration_s": a.trial_duration_s,
           "channels": chans, "results": {}}
    for name, S in feats.items():
        (Ftr, ytr), (Fva, yva), (Fte, yte) = S["train"], S["val"], S["test"]
        r = {"dim": int(Ftr.shape[-1])}
        P = [pad8(F) for F in (Ftr, Fva, Fte)]
        r["seqhead"] = agg([train_head(lambda: SeqHead(P[0].shape[-1], n_classes),
                                       P[0], ytr, P[1], yva, P[2], yte, n_classes, dev, s)
                            for s in range(a.n_reps)])
        if name in ("basic", "rich", "ours_frozen"):
            r["strong"] = agg([train_head(lambda: StrongHead(Ftr.shape[-1], n_classes),
                                          Ftr, ytr, Fva, yva, Fte, yte, n_classes, dev, s)
                               for s in range(a.n_reps)])
            r["gbdt"] = gbdt(Ftr, ytr, Fte, yte)
        if name in ("rich", "ours_frozen"):
            r["label_eff"] = {}
            for fr in a.label_fracs:
                ms = []
                for rep in range(5):
                    idx = np.random.RandomState(100 + rep).choice(
                        len(Ftr), max(4, int(round(len(Ftr) * fr))), replace=False)
                    ms.append(train_head(lambda: SeqHead(P[0].shape[-1], n_classes),
                                         P[0][idx], ytr[idx], P[1], yva, P[2], yte,
                                         n_classes, dev, rep))
                r["label_eff"][f"{fr:.2f}"] = agg(ms)
        res["results"][name] = r
        print(f"{name:12s} dim {r['dim']:4d} | seqhead BA {r['seqhead']['balanced_accuracy'][0]:.4f}"
              + (f" | strong {r['strong']['balanced_accuracy'][0]:.4f} | gbdt "
                 f"{r['gbdt']['balanced_accuracy']:.4f}" if "strong" in r else ""), flush=True)
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.output, "w"), indent=2)
    Path(str(a.output) + ".done").touch()


if __name__ == "__main__":
    main()
