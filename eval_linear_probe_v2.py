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
import os
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

BASE = Path(os.environ.get("SPECBRAM_BASELINES", "/home/pxieaf/home2/baselines"))
C_GRID = (0.01, 0.1, 1.0)
FRACS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
# readout of the frozen tokens: "mean" = mean over all (channel, time) tokens (main
# protocol); "chan" = mean over time within each present channel, channels concatenated
# (keeps spatial information; still frozen + logistic regression)
READOUT = {"mode": "mean"}
PRED = {"dir": None, "name": None, "ds": None}   # --pred_dir: save test predictions
SUBJ = {}                                         # split -> subject id of every epoch


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
        trials = ds.trials
        if normalization == "none" and getattr(a, "reve_norm", False):
            trials = reve_normalize(trials, ds.subject_ids)
        if normalization == "none" and getattr(a, "rn_scale", None) is not None:
            trials = robust_rescale(trials, ds.subject_ids, a.rn_scale)
        out[k] = (trials, np.asarray(ds.labels, np.int64))
        SUBJ[k] = np.asarray(ds.subject_ids)
    return out


def raw_scale(trials):
    """dataset-wide typical amplitude: median over channels of IQR/1.349 (µV)."""
    x = np.concatenate([np.asarray(t, np.float32) for t in trials[:2000]], 0)
    iqr = (np.percentile(x, 75, 0) - np.percentile(x, 25, 0)) / 1.349
    return float(np.median(iqr[iqr > 0]))


def robust_rescale(trials, rec_ids, scale):
    """Our input normalization (per-recording, per-channel median/IQR) for a baseline
    that expects µV: rescaled to the dataset-wide typical amplitude, so only the
    recording-to-recording differences are removed."""
    trials = [np.asarray(t, np.float32) for t in trials]
    rec_ids = np.asarray(rec_ids)
    out = [None] * len(trials)
    for r in np.unique(rec_ids):
        idx = np.where(rec_ids == r)[0]
        x = np.concatenate([trials[i] for i in idx], 0)
        med = np.median(x, 0)
        iqr = (np.percentile(x, 75, 0) - np.percentile(x, 25, 0)) / 1.349
        iqr[iqr == 0] = 1.0
        for i in idx:
            out[i] = ((trials[i] - med) / iqr * scale).astype(np.float32)
    return out


def reve_normalize(trials, rec_ids, clip=15.0):
    """REVE's input convention: z-score each recording (statistics over the whole
    recording, per channel), then clip at +-15 SD. Channels that are absent (all
    zero) stay zero."""
    trials = [np.asarray(t, np.float32) for t in trials]
    rec_ids = np.asarray(rec_ids)
    out = [None] * len(trials)
    for r in np.unique(rec_ids):
        idx = np.where(rec_ids == r)[0]
        x = np.concatenate([trials[i] for i in idx], 0)          # [n*T, 19]
        mu, sd = x.mean(0), x.std(0)
        sd[sd == 0] = 1.0
        for i in idx:
            out[i] = np.clip((trials[i] - mu) / sd, -clip, clip).astype(np.float32)
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
        h = model._encode(model._tokenize(x))     # [B, 19*Tp, D], channel-major
        if READOUT["mode"] == "chan":
            B, N, D = h.shape
            return h.reshape(B, 19, N // 19, D).mean(2)[:, enc.chans].flatten(1)
        return h.mean(1)
    enc.chans = list(range(19))
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
        h = m(x.reshape(B, C, T // 200, 200))                     # [B, C, P, D]
        return h.mean(2).flatten(1) if READOUT["mode"] == "chan" else h.mean(dim=(1, 2))
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
        if READOUT["mode"] == "chan":
            h = m.forward_features(x.reshape(B, C, T // 200, 200), input_chans=ids,
                                   return_patch_tokens=True)       # [B, C*A, D], channel-major
            return h.reshape(B, C, -1, h.shape[-1]).mean(2).flatten(1)
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


CSB_DIR = BASE / "CSBrain"
# CSBrain assigns every channel to a brain region (0 frontal, 1 parietal, 2 temporal,
# 3 occipital, 4 central) and orders channels within a region; we use the table of
# CSBrain's own wrapper for each dataset (models/model_for_<task>.py)
CSB_FILE = {"isruc": "isruc", "hmc": "hmc", "ma": "stress", "mumtaz": "mumtaz", "faced": "faced",
            "physio": "physio", "tuab": "tuab", "tuev": "tuev", "seedvig": "seedvig"}


def _csb_topology(fname):
    import re
    txt = (CSB_DIR / "models" / f"model_for_{fname}.py").read_text()
    out = {}
    for reg, body in re.findall(r"(\d)\s*:\s*\[([^\]]*)\]", txt):
        for pos, nm in enumerate(re.findall(r"['\"]([^'\"]+)['\"]", body)):
            out.setdefault(nm.upper(), (int(reg), pos))
    return out


def csbrain_layout(names, task):
    """-> kept channel indices, brain_regions (in kept order), sorted_indices."""
    topo = _csb_topology(CSB_FILE.get(task, "mumtaz"))
    merged = {}
    for f in (CSB_DIR / "models").glob("model_for_*.py"):
        for k, v in _csb_topology(f.stem[len("model_for_"):]).items():
            merged.setdefault(k, v)
    keep, regions, order = [], [], []
    for i, n in enumerate(names):
        key = str(n).upper().split("-")[0]          # bipolar TCP: region of the first electrode
        rp = topo.get(key) or merged.get(key)
        if rp is None:
            continue
        keep.append(i); regions.append(rp[0]); order.append(rp[1])
    sorted_idx = sorted(range(len(keep)), key=lambda j: (regions[j], order[j]))
    return keep, regions, sorted_idx


def csbrain_encoder(device, task):
    """CSBrain (official release, models/CSBrain.py + pth/CSBrain.pth): CBraMod's input
    convention (200 Hz, uV/100, 1-s patches); the region layout depends on the channel
    set, so the backbone is built per channel set (same weights)."""
    # CBraMod's repo also has a top-level package named `models`: import CSBrain's in
    # isolation and restore whatever `models` was loaded before
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "models" or k.startswith("models.")}
    sys.path.insert(0, str(CSB_DIR))
    try:
        from models.CSBrain import CSBrain
    finally:
        sys.path.remove(str(CSB_DIR))
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)
    sd = torch.load(CSB_DIR / "gdrive/pth/CSBrain.pth", map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    cache = {}

    def build(names):
        keep, regions, sidx = csbrain_layout(names, task)
        m = CSBrain(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30,
                    n_layer=12, nhead=8, brain_regions=regions, sorted_indices=sidx)
        ms = m.state_dict()
        miss = [k for k in ms if k not in sd or sd[k].shape != ms[k].shape]
        ms.update({k: v for k, v in sd.items() if k in ms and v.shape == ms[k].shape})
        m.load_state_dict(ms)
        m.proj_out = torch.nn.Identity()
        print(f"    CSBrain: {len(keep)}/{len(names)} channels, regions {sorted(set(regions))}, "
              f"unmatched weights {miss}", flush=True)
        return keep, m.to(device).eval()

    def enc(x, chans, names=None):               # x [B, T, C] raw uV, 200 Hz
        names = names or [OUR_19CH_ORDER[c].upper() for c in chans]
        key = tuple(names)
        if key not in cache:
            cache[key] = build(names)
        keep, m = cache[key]
        x = x[:, :, [chans[i] for i in keep]].permute(0, 2, 1) / 100.0
        B, C, T = x.shape
        h = m(x.reshape(B, C, T // 200, 200))                     # [B, C, P, D] (sorted)
        return h.mean(2).flatten(1) if READOUT["mode"] == "chan" else h.mean(dim=(1, 2))
    return enc, 200, "none", "csbrain"


REVE_DIR = BASE / "reve"


def reve_encoder(device, size="base", pool="mean"):
    """REVE (official HF release, brain-bzh/reve-{base,large}). Input: 200 Hz,
    per-recording z-score clipped at +-15 SD (REVE paper, Sec. 3.1.1), electrode
    positions from REVE's own position bank. Patch 200 samples, overlap 20.
    pool: "mean" = our strict protocol (mean over channel x time tokens);
          "attn" = REVE's own pretrained attention pooling (frozen query)."""
    import importlib.util
    import types
    from safetensors.torch import load_file
    pkg = types.ModuleType("reve_pkg"); pkg.__path__ = [str(REVE_DIR / f"reve-{size}")]
    sys.modules["reve_pkg"] = pkg
    for mod in ("configuration_reve", "modeling_reve"):
        spec = importlib.util.spec_from_file_location(f"reve_pkg.{mod}", REVE_DIR / f"reve-{size}" / f"{mod}.py")
        m_ = importlib.util.module_from_spec(spec); sys.modules[f"reve_pkg.{mod}"] = m_
        spec.loader.exec_module(m_)
    cfg_mod, mdl_mod = sys.modules["reve_pkg.configuration_reve"], sys.modules["reve_pkg.modeling_reve"]
    cfg = json.load(open(REVE_DIR / f"reve-{size}" / "config.json"))
    conf = cfg_mod.ReveConfig(**{k: v for k, v in cfg.items()
                                 if k not in ("architectures", "auto_map", "model_type",
                                              "transformers_version", "dtype")})
    # transformers 4.17's post_init() calls _init_weights, which REVE does not define;
    # the weights are overwritten by the checkpoint right below anyway
    mdl_mod.Reve._init_weights = lambda self, module: None
    model = mdl_mod.Reve(conf)
    sd = load_file(str(REVE_DIR / f"reve-{size}" / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"REVE-{size} load: missing {missing} | unexpected {unexpected}", flush=True)
    model.to(device).eval()
    bank = json.load(open(REVE_DIR / "reve-positions" / "positions.json"))
    bank = {k.upper(): v for k, v in bank.items()}

    def enc(x, chans, names=None):               # x [B, T, 19] z-scored, 200 Hz
        names = names or [OUR_19CH_ORDER[c].upper() for c in chans]
        pos = torch.tensor([bank[n] for n in names], dtype=torch.float32, device=x.device)
        x = x[:, :, chans].permute(0, 2, 1).contiguous()
        h = model(x, pos.unsqueeze(0).expand(x.shape[0], -1, -1))    # [B, C, P, E]
        if READOUT["mode"] == "chan":
            return h.mean(2).flatten(1)
        return model.attention_pooling(h) if pool == "attn" else h.mean(dim=(1, 2))
    enc.model = model
    return enc, 200, "none", "reve"


@torch.no_grad()
def reve_features(size, X, names, rec, device, bs=64, clip=15.0):
    """Both frozen REVE readouts for windows X [N, T, C] (µV, 200 Hz; array or
    memmap) with channel names and recording ids: per-recording, per-channel
    z-score clipped at +-15 SD (REVE's convention), channels outside REVE's
    position bank dropped. -> {"mean": [N, E], "attn": [N, E]}."""
    model = reve_encoder(device, size, "mean")[0].model
    bank = json.load(open(REVE_DIR / "reve-positions" / "positions.json"))
    bank = {k.upper(): v for k, v in bank.items()}
    names = [str(n).upper() for n in names]
    keep = [i for i, n in enumerate(names) if n in bank]
    print(f"    REVE-{size}: {len(keep)}/{len(names)} channels in its position bank", flush=True)
    pos = torch.tensor([bank[names[i]] for i in keep], dtype=torch.float32, device=device)
    rec = np.asarray(rec)
    out = {"mean": np.zeros((len(X), model.embed_dim), np.float32),
           "attn": np.zeros((len(X), model.embed_dim), np.float32)}
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        xr = np.asarray(X[idx], np.float32)[:, :, keep]              # [n, T, c]
        mu = xr.mean((0, 1), keepdims=True); sd = xr.std((0, 1), keepdims=True)
        sd[sd == 0] = 1.0
        xr = np.clip((xr - mu) / sd, -clip, clip)
        for j in range(0, len(idx), bs):
            x = torch.from_numpy(np.ascontiguousarray(xr[j:j + bs].transpose(0, 2, 1))).to(device)
            h = model(x, pos.unsqueeze(0).expand(x.shape[0], -1, -1))          # [B, c, P, E]
            out["mean"][idx[j:j + bs]] = h.mean(dim=(1, 2)).float().cpu().numpy()
            out["attn"][idx[j:j + bs]] = model.attention_pooling(h).float().cpu().numpy()
    return out


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
        clf_full = fit(F["train"], y["train"], C)
        full = score(y["test"], clf_full.predict(F["test"]))
        if PRED["dir"]:
            Path(PRED["dir"]).mkdir(parents=True, exist_ok=True)
            np.savez(Path(PRED["dir"]) / f"{PRED['ds']}__{PRED['name']}__{mode}.npz", y=y["test"],
                     pred=clf_full.predict(F["test"]), subj=SUBJ.get("test"), val_ba=float(best[0]),
                     **({"feat": F["test"].astype(np.float16)} if os.environ.get("SAVE_FEAT") else {}))
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
    ap.add_argument("--model", choices=["specbram", "cbramod", "labram", "biot", "biot_sleep",
                                        "reve_base", "reve_large", "cbramod_rn", "labram_rn", "csbrain"],
                    required=True)
    ap.add_argument("--checkpoints", nargs="*", default=[], metavar="NAME=PATH",
                    help="specbram only: several checkpoints probed on one data load")
    ap.add_argument("--random_init", action="store_true",
                    help="specbram only: also probe the randomly initialised architecture")
    ap.add_argument("--modes", nargs="+", default=["avg3", "center10"])
    ap.add_argument("--readout", choices=["mean", "chan"], default="mean")
    ap.add_argument("--pred_dir", default=None, help="save test predictions per model")
    ap.add_argument("--no_label_eff", action="store_true", help="full-label probe only")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    READOUT["mode"] = a.readout
    if a.no_label_eff:
        global FRACS
        FRACS = (1.00,)

    if a.model == "specbram":
        todo = [c.split("=", 1) for c in a.checkpoints]
        if a.random_init:
            todo.append(("random_init", todo[0][1]))
        sr, norm = 256, "per_recording_robust"
    elif a.model.endswith("_rn"):
        # input-convention check: the baseline on our per-recording normalization
        todo = [(a.model, None)]
        sr, norm = 200, "none"
        a.rn_scale = raw_scale(load_split(a, sr, "none")["train"][0])
        print(f"  rescale to dataset median amplitude {a.rn_scale:.2f} uV", flush=True)
    elif a.model.startswith("reve"):
        # two frozen, label-free readouts: mean pooling (ours) and REVE's own
        # pretrained attention pooling
        todo = ([(f"{a.model}_chan", None)] if a.readout == "chan"
                else [(f"{a.model}_mean", None), (f"{a.model}_attn", None)])
        sr, norm = 200, "none"
        a.reve_norm = True
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
        if a.model in ("cbramod", "cbramod_rn"):
            enc, _, _, kind = cbramod_encoder(dev)
        elif a.model == "biot":
            enc, _, _, kind = biot_encoder(dev)
        elif a.model == "biot_sleep":
            enc, _, _, kind = biot_encoder(dev, "EEG-SHHS+PREST-18-channels.ckpt")
        elif a.model in ("labram", "labram_rn"):
            enc, _, _, kind = labram_encoder(dev)
        elif a.model == "csbrain":
            enc, _, _, kind = csbrain_encoder(dev, a.dataset)
        elif a.model.startswith("reve"):
            enc, _, _, kind = reve_encoder(dev, a.model.split("_")[1],
                                           "mean" if a.readout == "chan" else name.rsplit("_", 1)[1])
        else:
            enc, _, _, kind = specbram_encoder(path, dev, random_init=(name == "random_init"))
        PRED.update(dir=a.pred_dir, name=name, ds=a.dataset)
        if kind is None:
            enc.chans = chans                          # our models: present leads only
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
