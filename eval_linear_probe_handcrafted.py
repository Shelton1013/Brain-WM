"""Handcrafted spectral features under the SAME strict linear probe as
eval_linear_probe_v2.py (ISRUC / HMC): same epochs, splits, standardization,
class-balanced logistic regression with C chosen on validation, same five
label-efficiency subsets. Only the feature extractor differs.

Feature sets (eval_handcrafted_sleep.epoch_features, present channels only):
  basic : log absolute + relative power in the 5 standard bands
  rich  : 1-Hz log PSD 0.5-45 Hz, relative power in 6 bands, Hjorth parameters,
          spectral entropy, 95% spectral edge, line length, kurtosis
Modes:
  avg3    : features of the three 10-s windows averaged (exactly how our encoder
            is probed)
  epoch30 : features computed on the whole 30-s epoch

    python eval_linear_probe_handcrafted.py --dataset isruc --data_dir ... --output out.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from eval_handcrafted_sleep import epoch_features
from eval_linear_probe_v2 import C_GRID, FRACS, fit, load_split, score

SR = 256
_CH = None


def _feat(args):
    trial, kind, mode = args
    if mode == "epoch30":
        return epoch_features(trial, SR, kind, _CH)
    w = 10 * SR
    return np.mean([epoch_features(trial[i * w:(i + 1) * w], SR, kind, _CH) for i in range(3)], 0)


def _init(ch):
    global _CH
    _CH = ch


def probe(F, y):
    sc = StandardScaler().fit(F["train"])
    F = {k: np.nan_to_num(sc.transform(v)) for k, v in F.items()}
    best = max(((balanced_accuracy_score(y["val"], fit(F["train"], y["train"], C).predict(F["val"])), C)
                for C in C_GRID))
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
            reps.append(score(y["test"], fit(F["train"][idx], y["train"][idx], C).predict(F["test"])))
        curve[f"{fr:.2f}"] = {k: [float(np.mean([m[k] for m in reps])), float(np.std([m[k] for m in reps]))]
                              for k in reps[0]}
    print(f"    BA {full['balanced_accuracy']:.4f} (C={C}) | LE "
          + " ".join(f"{k}:{v['balanced_accuracy'][0]:.3f}" for k, v in curve.items()), flush=True)
    return {"C": C, "val_ba": float(best[0]), "test": full, "label_eff": curve,
            "dim": int(F["train"].shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["isruc", "hmc"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    data = load_split(a, SR, "per_recording_robust")
    first = np.stack(data["train"][0][:500])
    chans = [c for c in range(first.shape[-1]) if np.abs(first[..., c]).max() > 0]
    y = {k: data[k][1] for k in data}
    res = {"dataset": a.dataset, "results": {}}
    with Pool(a.workers, initializer=_init, initargs=(chans,)) as pool:
        for kind in ("basic", "rich"):
            key = f"hc_{kind}"
            res["results"][key] = {}
            for mode in ("avg3", "epoch30"):
                print(f"  [{key} / {mode}]", flush=True)
                F = {k: np.stack(pool.map(_feat, [(t, kind, mode) for t in tr], chunksize=64))
                     for k, (tr, _) in data.items()}
                res["results"][key][mode] = probe(F, y)
                Path(a.output).parent.mkdir(parents=True, exist_ok=True)
                json.dump(res, open(a.output, "w"), indent=2)
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
