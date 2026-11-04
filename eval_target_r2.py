"""How much of its pretraining target does each model predict? (efficiency analysis)

For a band-power and a raw-waveform-reconstruction checkpoint, mask held-out windows exactly as
in pretraining and measure, on the masked patches, the fraction of target variance explained
R^2 = 1 - MSE / Var(target). For band power also compute a zero-floor iid-model
reference, 1 - psi'(m_b) / Var(y_b). This is not an empirical Bayes ceiling.

Held-out data: windows of the TUAB evaluation split in the pretraining input convention
(19 leads, 256 Hz, per-recording robust scaling).

    python eval_target_r2.py --out ~/home2/eval_results/target_r2.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import polygamma

from eeg_lejepa_v3 import EEGLeJEPA_v3, v3_kwargs_from_args

M = Path("/home/pxieaf/home2/model")
DATA = Path("/home/pxieaf/home2/datasets/lp_tasks/tuab")
RUNS = {"bp": {42: "abl_linear_fftstdlog_nosig4g_s42_sub2100",
               43: "abl_linear_fftstdlog_nosig_s43_sub2100",
               44: "abl_linear_fftstdlog_nosig_s44_sub2100"},
        "raw": {42: "abl_linear_rawrecon_nosig_s42_sub2100",
                43: "abl_linear_rawrecon_nosig_s43_sub2100",
                44: "abl_linear_rawrecon_nosig_s44_sub2100"}}


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = EEGLeJEPA_v3(**v3_kwargs_from_args(ck.get("args", {})))
    m.load_state_dict(ck["model_state_dict"])
    return m.to(dev).eval()


@torch.no_grad()
def masked_pred(m, eeg):
    """Mirror of EEGLeJEPA_v3.forward up to the pretext prediction: (pred, target) on masked
    patches, [n, k]."""
    band_emb, _, _ = m.tokenizer(eeg)
    B, C, Tp, N, d = band_emb.shape
    Cc = min(C, m.pos_channel.shape[1]); Tpp = min(Tp, m.pos_time.shape[2])
    band_emb = band_emb[:, :Cc, :Tpp]
    B, C, Tp, N, d = band_emb.shape
    if m.mask_axis == "time":
        tmask = torch.rand(B, C, Tp, device=eeg.device) < m.band_mask_ratio
        all_t = tmask.all(dim=-1)
        tmask[..., 0] = tmask[..., 0] & ~all_t
        bmask = tmask.unsqueeze(-1).expand(B, C, Tp, N).clone()
    else:
        bmask = torch.rand(B, C, Tp, N, device=eeg.device) < m.band_mask_ratio
        all_m = bmask.all(dim=-1)
        bmask[..., 0] = bmask[..., 0] & ~all_m
    band_ctx = torch.where(bmask.unsqueeze(-1), m.cf_mask_token.expand(B, C, Tp, N, d), band_emb)
    enc = m._encode_4d(m._add_pos(m.tokenizer._combine(band_ctx)))
    pm = bmask.any(-1)                                         # masked patches [B,C,Tp]
    if m.target_type == "fft_bandpower":
        tgt = m._fft_band_logpow(eeg)[:, :C, :Tp]
        pred = m.cf_head(enc)
    else:
        P = m.patch_len
        tgt = eeg[:, :Tp * P, :C].permute(0, 2, 1).reshape(B, C, Tp, P)
        pred = m.recon_head(enc)
    return pred[pm].float(), tgt[pm].float()


def evaluate(m, X, dev, bs=64):
    sse = s1 = s2 = 0.0
    n = 0
    for i in range(0, len(X), bs):
        torch.manual_seed(i)                                   # identical masks for every model
        eeg = torch.as_tensor(np.asarray(X[i:i + bs], dtype=np.float32), device=dev)
        p, y = masked_pred(m, eeg)
        sse = sse + ((p - y) ** 2).sum(0).double().cpu()
        s1 = s1 + y.sum(0).double().cpu()
        s2 = s2 + (y ** 2).sum(0).double().cpu()
        n += y.shape[0]
    mean, var = s1 / n, s2 / n - (s1 / n) ** 2
    return {"n_patches": n, "mse": (sse / n).tolist(), "var": var.tolist(), "mean": mean.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_windows", type=int, default=4000)
    ap.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    ap.add_argument("--model_dir", type=Path, default=M)
    ap.add_argument("--data_dir", type=Path, default=DATA)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    meta = np.load(a.data_dir / "meta.npz")
    X = np.load(a.data_dir / "ours256.npy", mmap_mode="r")
    idx = np.flatnonzero(meta["split"] == "test")
    idx = np.sort(np.random.default_rng(0).choice(idx, a.n_windows, replace=False))
    Xs = np.asarray(X[idx], dtype=np.float32)
    if Xs.shape[1] == 19:                                      # [N,C,T] -> [N,T,C]
        Xs = Xs.transpose(0, 2, 1)
    print("held-out windows", Xs.shape, flush=True)

    res = {}
    for kind, runs in RUNS.items():
        for seed, run in runs.items():
            eps = a.epochs if seed == 42 else [16]
            for e in eps:
                f = a.model_dir / run / f"checkpoint_ep{e}.pt"
                if not f.exists():
                    print("missing", f, flush=True)
                    continue
                m = load(f, dev)
                r = evaluate(m, Xs, dev)
                mse, var = np.array(r["mse"]), np.array(r["var"])
                if kind == "bp":
                    r["r2_band"] = (1 - mse / var).tolist()
                    r["r2"] = float(np.mean(1 - mse / var))
                    edges = getattr(m, "fft_edges", None)
                    df = m.sample_rate / m.patch_len
                    k = np.arange(m.patch_len // 2 + 1) * df
                    mb = [int(((k >= lo) & (k < hi)).sum()) for lo, hi in edges]
                    noise = [float(polygamma(1, x)) for x in mb]
                    r["m_b"], r["noise"] = mb, noise
                    r["ceiling_band"] = (1 - np.array(noise) / var).tolist()  # legacy key: ideal reference only
                else:
                    tot = float((var + (np.array(r["mean"]) - np.mean(r["mean"])) ** 2).sum())
                    r["r2"] = float(1 - mse.sum() / tot)               # pooled over samples
                    for key in ("mse", "var", "mean"):
                        r[key] = float(np.mean(r[key]))
                res[f"{kind}_s{seed}_ep{e}"] = r
                print(kind, seed, e, "R2 =", round(r["r2"], 4),
                      [round(x, 3) for x in r.get("r2_band", [])], flush=True)
                del m
                torch.cuda.empty_cache()
                json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
