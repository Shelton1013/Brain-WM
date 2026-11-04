"""Label-free evaluation of frozen sleep features: k-means clustering.

Same frozen features as the strict linear probe (eval_linear_probe_v2.py: mean-pooled
tokens, a 30-s epoch = mean of its three 10-s windows, each model in its own input
convention). No label is used to fit anything: features of the evaluated split are
standardized, reduced to a fixed 64 PCA dimensions (removes the effect of different
feature widths), and clustered with k-means, k = number of sleep stages (5), five
seeds. Agreement with the expert stages: NMI and ARI (mean over seeds).
REVE: the readout (mean pooling or its pretrained attention pooling) is chosen on the
validation split by NMI; the test split is scored once.

    python eval_cluster_sleep.py --dataset isruc --data_dir ... --model specbram \
        --checkpoints m_s42=.../checkpoint_ep16.pt --output cl_isruc.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

import eval_linear_probe_v2 as lp

N_PCA, SEEDS = 64, (0, 1, 2, 3, 4)


def cluster(F, y):
    Z = StandardScaler().fit_transform(F)
    pca = PCA(n_components=min(N_PCA, Z.shape[1]), random_state=0).fit(Z)
    Z = pca.transform(Z)
    k = len(np.unique(y))
    nmi, ari = [], []
    for s in SEEDS:
        c = KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Z)
        nmi.append(normalized_mutual_info_score(y, c))
        ari.append(adjusted_rand_score(y, c))
    return {"nmi": [float(np.mean(nmi)), float(np.std(nmi))],
            "ari": [float(np.mean(ari)), float(np.std(ari))],
            "pca_var": float(pca.explained_variance_ratio_.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["isruc", "hmc"], required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", required=True,
                    choices=["specbram", "cbramod", "labram", "biot", "reve_base", "reve_large", "csbrain"])
    ap.add_argument("--checkpoints", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    if a.model == "specbram":
        todo, sr, norm = [c.split("=", 1) for c in a.checkpoints], 256, "per_recording_robust"
    elif a.model.startswith("reve"):
        todo, sr, norm = [(f"{a.model}_mean", None), (f"{a.model}_attn", None)], 200, "none"
        a.reve_norm = True
    else:
        todo, sr, norm = [(a.model, None)], 200, "none"
    data = lp.load_split(a, sr, norm)
    data = {k: data[k] for k in ("val", "test")}
    first = np.stack(data["val"][0][:500])
    chans = [c for c in range(first.shape[-1]) if np.abs(first[..., c]).max() > 0]
    print(f"{a.dataset}: data {(time.time() - t0) / 60:.1f} min, channels {len(chans)}", flush=True)

    out = Path(a.output)
    res = json.load(open(out)) if out.exists() else {"dataset": a.dataset, "results": {}}
    for name, path in todo:
        if name in res["results"]:
            continue
        if a.model == "specbram":
            enc, _, _, kind = lp.specbram_encoder(path, dev)
        elif a.model.startswith("reve"):
            enc, _, _, kind = lp.reve_encoder(dev, a.model.split("_")[1], name.rsplit("_", 1)[1])
        elif a.model == "csbrain":
            enc, _, _, kind = lp.csbrain_encoder(dev, a.dataset)
        else:
            enc, _, _, kind = {"cbramod": lp.cbramod_encoder, "labram": lp.labram_encoder,
                               "biot": lp.biot_encoder}[a.model](dev)
        r = {}
        for split, (trials, y) in data.items():
            F = lp.featurize(enc, kind, trials, sr, "avg3", chans, dev)
            r[split] = cluster(F, y)
            print(f"  {name} [{split}] NMI {r[split]['nmi'][0]:.3f} ARI {r[split]['ari'][0]:.3f}", flush=True)
        r["checkpoint"] = path
        res["results"][name] = r
        json.dump(res, open(out, "w"), indent=2)
        del enc; torch.cuda.empty_cache()
    Path(str(out) + ".done").touch()


if __name__ == "__main__":
    main()
