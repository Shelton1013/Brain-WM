"""Strict linear probe v2 (ISRUC / HMC) for SpecBraM variants AND published EEG
foundation models, under one protocol — the primary representation metric.

Every 30-s sleep epoch is cut into three 10-s windows; each window is encoded by
the frozen model and mean-pooled over its tokens; the epoch feature is the mean
of the three windows ("avg3"), or the middle window only ("center10", our
earlier protocol). Features are z-scored on train; a multinomial logistic
regression (class-balanced) is fitted with C chosen on validation, then scored
once on test. The same C is reused for a label-efficiency curve (fractions of the
training epochs, 5 random subsets each). Deterministic.

Each model gets its own documented input contract:
  specbram : 256 Hz, per-recording robust scaling, canonical 19-ch layout
  cbramod  : 200 Hz, raw µV / 100, the recording's EEG channels, 1-s patches
  labram   : 200 Hz, raw µV / 100, channel ids from LaBraM's standard_1020 list

    python eval_linear_probe_v2.py --dataset isruc --model specbram \
        --checkpoint .../checkpoint_ep16.pt --output out.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

from dataset_mumtaz import OUR_19CH_ORDER

BASE = Path("/home/pxieaf/home2/baselines")
C_GRID = (0.01, 0.1, 1.0)
FRACS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


def load_split(a, sample_rate, normalization):
    """{split: (trials [N, 30*sr, 19], labels [N])} — every scored 30-s epoch."""
    kw = dict(sample_rate=sample_rate, trial_duration_s=30,
              normalization=normalization, cache_dir=None)
    if a.dataset == "isruc":
        from dataset_isruc import ISRUCDataset, CBRAMOD_ISRUC_SPLITS as SP
        mk = lambda s: ISRUCDataset(data_dir=a.data_dir, subjects=s, **kw)
    else:
        from dataset_hmc import HMCDataset, make_hmc_split
        SP = make_hmc_split(a.data_dir)
        mk = lambda s: HMCDataset(a.data_dir, s, **kw)
    out = {}
    for k in ("train", "val", "test"):
        ds = mk(SP[k])
        out[k] = (ds.trials, np.asarray(ds.labels, np.int64))
    return out


def windows(trial, sr, mode):
    """[30*sr, 19] -> list of [10*sr, 19] windows."""
    w = 10 * sr
    return [trial[w:2 * w]] if mode == "center10" else [trial[i * w:(i + 1) * w] for i in range(3)]


# ----------------------------- model adapters -----------------------------

def specbram_encoder(ckpt, device, random_init=False):
    from eval_tuh_clinical import load_pretrained, build_random_init
    model, cls, _, n_ch, args = load_pretrained(ckpt, device)
    if random_init:
        torch.manual_seed(0)
        model = build_random_init(cls, n_ch, args, device)
    model.eval()

    def enc(x):                                   # x [B, 2560, 19]
        return model._encode(model._tokenize(x)).mean(1)
    return enc, 256, "per_recording_robust", None


def cbramod_encoder(device):
    sys.path.insert(0, str(BASE / "CBraMod"))
    from models.cbramod import CBraMod
    m = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                seq_len=30, n_layer=12, nhead=8)
    m.load_state_dict(torch.load(BASE / "CBraMod/pretrained_weights/pretrained_weights.pth",
                                 map_location="cpu"))
    m.proj_out = torch.nn.Identity()
    m.to(device).eval()

    def enc(x, chans):                            # x [B, 2000, C] raw µV
        x = x[:, :, chans].permute(0, 2, 1) / 100.0
        B, C, T = x.shape
        return m(x.reshape(B, C, T // 200, 200)).mean(dim=(1, 2))
    return enc, 200, "none", "cbramod"


def labram_encoder(device):
    sys.path.insert(0, str(BASE / "LaBraM"))
    import modeling_finetune  # noqa: F401  (registers timm models)
    from timm.models import create_model
    import utils as lutils
    m = create_model("labram_base_patch200_200", pretrained=False, num_classes=0,
                     drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0,
                     use_mean_pooling=True, init_scale=0.001, use_rel_pos_bias=False,
                     use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)
    # official checkpoint stores numpy scalars in its metadata: allow-list just
    # those types instead of disabling weights_only
    import argparse as _ap
    safe = [np.core.multiarray.scalar, np.dtype, type(np.dtype("float64")), _ap.Namespace]
    with torch.serialization.safe_globals(safe):
        ck = torch.load(BASE / "LaBraM/checkpoints/labram-base.pth", map_location="cpu")
    sd = ck.get("model", ck)
    sd = {k[8:]: v for k, v in sd.items() if k.startswith("student.")} or sd
    # with mean pooling LaBraM applies fc_norm instead of its pretrained final
    # norm: carry the pretrained final-norm parameters over so features match
    for p in ("weight", "bias"):
        if f"norm.{p}" in sd and f"fc_norm.{p}" not in sd:
            sd[f"fc_norm.{p}"] = sd.pop(f"norm.{p}")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"LaBraM load: missing {missing} | unexpected {unexpected}")
    m.to(device).eval()
    std = lutils.standard_1020

    def enc(x, chans, names=None):               # names: other montages (SEED-VIG)
        names = names or [OUR_19CH_ORDER[c].upper() for c in chans]
        ids = torch.tensor([0] + [std.index(n) + 1 for n in names], device=x.device)
        x = x[:, :, chans].permute(0, 2, 1) / 100.0
        B, C, T = x.shape
        return m.forward_features(x.reshape(B, C, T // 200, 200), input_chans=ids)
    return enc, 200, "none", "labram"


def biot_encoder(device, ckpt_name="EEG-six-datasets-18-channels.ckpt"):
    """BIOT, 18-channel release. Its last two channel tokens are C3-A2 (16) and
    C4-A1 (17) — the SHHS sleep montage — so the recording's C3/C4 are fed
    there; the other ISRUC/HMC leads have no counterpart among BIOT's bipolar
    channel tokens. Per-channel 95th-percentile scaling, 200 Hz, as BIOT's code."""
    sys.path.insert(0, str(BASE / "BIOT"))
    from model.biot import BIOTEncoder
    m = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=18, n_fft=200,
                    hop_length=100)
    m.load_state_dict(torch.load(BASE / "BIOT/pretrained-models" / ckpt_name,
                                 map_location="cpu"))
    m.to(device).eval()
    c3, c4 = OUR_19CH_ORDER.index("C3"), OUR_19CH_ORDER.index("C4")

    def enc(x, chans):                            # x [B, 2000, 19] raw µV
        x = x[:, :, [c3, c4]].permute(0, 2, 1)
        x = x / (torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True) + 1e-8)
        return m(x, n_channel_offset=16)
    return enc, 200, "none", "biot"


# ------------------------------- probing ----------------------------------

@torch.no_grad()
def featurize(enc, kind, trials, sr, mode, chans, device, bs=128):
    feats = []
    for i in range(0, len(trials), bs):
        batch = trials[i:i + bs]
        per_win = []
        for wi in range(1 if mode == "center10" else 3):
            x = torch.from_numpy(np.stack([windows(t, sr, mode)[wi] for t in batch])
                                 .astype(np.float32)).to(device)
            per_win.append((enc(x) if kind is None else enc(x, chans)).float().cpu())
        feats.append(torch.stack(per_win).mean(0))
    return torch.cat(feats).numpy()


def fit(Ftr, ytr, C):
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ftr, ytr)


def score(y, p):
    return {"balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "cohen_kappa": float(cohen_kappa_score(y, p)),
            "weighted_f1": float(f1_score(y, p, average="weighted"))}


def probe_modes(enc, kind, data, sr, chans, dev, modes):
    out = {}
    for mode in modes:
        F = {k: featurize(enc, kind, tr, sr, mode, chans, dev) for k, (tr, _) in data.items()}
        y = {k: data[k][1] for k in data}
        sc = StandardScaler().fit(F["train"])
        F = {k: sc.transform(v) for k, v in F.items()}
        best = max(((balanced_accuracy_score(y["val"], fit(F["train"], y["train"], C)
                                             .predict(F["val"])), C) for C in C_GRID))
        C = best[1]
        full = score(y["test"], fit(F["train"], y["train"], C).predict(F["test"]))
        curve = {}
        for fr in FRACS:
            reps = []
            for r in range(1 if fr >= 1.0 else 5):
                idx = np.random.RandomState(100 + r).choice(
                    len(y["train"]), max(20, int(round(len(y["train"]) * fr))), replace=False)
                if len(np.unique(y["train"][idx])) < 2:
                    continue
                reps.append(score(y["test"], fit(F["train"][idx], y["train"][idx], C)
                                  .predict(F["test"])))
            curve[f"{fr:.2f}"] = {k: [float(np.mean([m[k] for m in reps])),
                                      float(np.std([m[k] for m in reps]))] for k in reps[0]}
        out[mode] = {"C": C, "val_ba": float(best[0]), "test": full,
                     "label_eff": curve, "dim": int(F["train"].shape[1])}
        print(f"    [{mode}] BA {full['balanced_accuracy']:.4f} k {full['cohen_kappa']:.4f} "
              f"(C={C}) | LE " + " ".join(f"{k}:{v['balanced_accuracy'][0]:.3f}"
                                          for k, v in curve.items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["isruc", "hmc"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", choices=["specbram", "cbramod", "labram", "biot", "biot_sleep"],
                    required=True)
    ap.add_argument("--checkpoints", nargs="*", default=[], metavar="NAME=PATH",
                    help="specbram only: several checkpoints probed on one data load")
    ap.add_argument("--random_init", action="store_true",
                    help="specbram only: also probe the randomly initialised architecture")
    ap.add_argument("--modes", nargs="+", default=["avg3", "center10"])
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    if a.model == "specbram":
        todo = [c.split("=", 1) for c in a.checkpoints]
        if a.random_init:
            todo.append(("random_init", todo[0][1]))
        sr, norm = 256, "per_recording_robust"
    else:
        todo = [(a.model, None)]
        sr, norm = 200, "none"
    data = load_split(a, sr, norm)
    first = np.stack(data["train"][0][:500])
    chans = [c for c in range(first.shape[-1]) if np.abs(first[..., c]).max() > 0]
    print(f"{a.dataset}: {sr} Hz, norm={norm}, channels "
          f"{[OUR_19CH_ORDER[c] for c in chans]}; data {(time.time() - t0) / 60:.1f} min",
          flush=True)

    out = Path(a.output)
    res = json.load(open(out)) if out.exists() else {"dataset": a.dataset, "results": {}}
    for name, path in todo:
        if name in res["results"]:
            print(f"  skip {name} (done)"); continue
        t = time.time()
        if a.model == "cbramod":
            enc, _, _, kind = cbramod_encoder(dev)
        elif a.model == "biot":
            enc, _, _, kind = biot_encoder(dev)
        elif a.model == "biot_sleep":
            enc, _, _, kind = biot_encoder(dev, "EEG-SHHS+PREST-18-channels.ckpt")
        elif a.model == "labram":
            enc, _, _, kind = labram_encoder(dev)
        else:
            enc, _, _, kind = specbram_encoder(path, dev, random_init=(name == "random_init"))
        print(f"  {name}", flush=True)
        res["results"][name] = probe_modes(enc, kind, data, sr, chans, dev, a.modes)
        res["results"][name]["checkpoint"] = path
        print(f"  {name} done in {(time.time() - t) / 60:.1f} min", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=2)
        del enc; torch.cuda.empty_cache()
    Path(str(out) + ".done").touch()


if __name__ == "__main__":
    main()
