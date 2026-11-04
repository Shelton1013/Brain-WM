"""Handcrafted spectral features under the SAME strict linear probe as the
foundation-model probes beyond sleep (eval_linear_probe_tasks.py / _tuab.py):
same samples, splits, standardization, C grid and metrics; only the feature
extractor differs. Features are computed on OUR input contract of each window
(0.5-45 Hz, 256 Hz, per-recording robust scaling, 19 leads), over the whole window
(as the encoders mean-pool the whole window).

  basic : log absolute + relative power in the 5 standard bands
  rich  : 1-Hz log PSD 0.5-45 Hz, relative power in 6 bands, Hjorth parameters,
          spectral entropy, 95% spectral edge, line length, kurtosis

    python eval_handcrafted_tasks.py --task ma --output lp_ma_hc.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from eval_handcrafted_sleep import epoch_features
from eval_linear_probe_tasks import our_input, probe

ROOT = Path("/home/pxieaf/home2/datasets/lp_tasks")
SR = 256
CH = list(range(19))
_X = None


def _init(path):
    global _X
    _X = np.load(path, mmap_mode="r") if isinstance(path, str) else path


def _feat(args):
    i0, i1, kind = args
    return np.stack([epoch_features(np.asarray(_X[i], dtype=np.float64), SR, kind, CH)
                     for i in range(i0, i1)])


def features(X, kind, workers, path=None):
    n = len(X)
    chunks = [(i, min(i + 512, n), kind) for i in range(0, n, 512)]
    with Pool(workers, initializer=_init, initargs=(path if path else X,)) as pool:
        return np.concatenate(pool.map(_feat, chunks))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["ma", "mumtaz", "physio", "faced", "tuab", "tuev"])
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.task in ("tuab", "tuev"):
        meta = np.load(ROOT / a.task / "meta.npz")
        y_all, sp = meta["y"], meta["split"]
        path = str(ROOT / a.task / "ours256.npy")
        X = np.load(path, mmap_mode="r")                        # [N, T, 19] float16
    else:
        d = np.load(ROOT / f"{a.task}.npz")
        y_all, sp = d["y"], d["split"]
        X, path = our_input(d["X"], [str(c) for c in d["ch"]], d["rec"]), None
    parts = {k: sp == k for k in ("train", "val", "test")}
    y = {k: y_all[m] for k, m in parts.items()}
    n_cls = int(y_all.max()) + 1
    print(f"{a.task}: {X.shape}, classes {n_cls}", flush=True)

    out = Path(a.output)
    res = json.load(open(out)) if out.exists() else {"task": a.task, "results": {}}
    for kind in ("basic", "rich"):
        name = f"hc_{kind}"
        if name in res["results"]:
            continue
        F = np.nan_to_num(features(X, kind, a.workers, path))
        print(f"  [{name}] {F.shape}", flush=True)
        res["results"][name] = probe({k: F[m] for k, m in parts.items()}, y, n_cls)
        json.dump(res, open(out, "w"), indent=2)
    json.dump(res, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
