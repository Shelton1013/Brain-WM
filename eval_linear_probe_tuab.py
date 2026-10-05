"""Strict linear probe on TUAB (abnormal vs normal) — CBraMod's samples / split
(build_tuab_lp.py), same protocol as eval_linear_probe_tasks.py.

  specbram : ours256 windows (0.5-45 Hz, 256 Hz, per-recording robust scaling, 19 leads)
  cbramod  : CBraMod's 16 TCP bipolar channels (mono200 differences) / 100
  labram   : the 19 monopolar leads by name / 100

    python eval_linear_probe_tuab.py --checkpoints linbp_s42=... --random_init \
        --baselines cbramod labram --output lp_tuab.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_linear_probe_tasks import probe

D = Path("/home/pxieaf/home2/datasets/lp_tasks/tuab")


def batches(arr, bs):
    for i in range(0, len(arr), bs):
        yield np.asarray(arr[i:i + bs], dtype=np.float32)


@torch.no_grad()
def feats_ours(model, ours, dev, bs=256):
    model.eval()
    return np.concatenate([model._encode(model._tokenize(torch.from_numpy(x).to(dev))).mean(1)
                           .float().cpu().numpy() for x in batches(ours, bs)])


@torch.no_grad()
def feats_baseline(name, mono, ch, tcp, dev, bs=128):
    from eval_linear_probe_v2 import cbramod_encoder, labram_encoder
    enc = (cbramod_encoder if name == "cbramod" else labram_encoder)(dev)[0]
    out = []
    for x in batches(mono, bs):                                   # [B, 19, 2000] µV
        if name == "cbramod":
            x = x[:, tcp[:, 0]] - x[:, tcp[:, 1]]                 # [B, 16, 2000]
            f = enc(torch.from_numpy(x.transpose(0, 2, 1).copy()).to(dev), list(range(16)))
        else:
            f = enc(torch.from_numpy(x.transpose(0, 2, 1).copy()).to(dev), list(range(19)),
                    names=list(ch))
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--baselines", nargs="*", default=[], choices=["cbramod", "labram"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = np.load(D / "meta.npz")
    y_all, sp = meta["y"], meta["split"]
    ch, tcp = [str(c) for c in meta["mono_ch"]], meta["tcp"]
    parts = {k: sp == k for k in ("train", "val", "test")}
    y = {k: y_all[m] for k, m in parts.items()}
    print("TUAB", {k: int(m.sum()) for k, m in parts.items()}, flush=True)

    out = Path(a.output)
    res = json.load(open(out)) if out.exists() else {"task": "tuab", "results": {}}
    save = lambda: json.dump(res, open(out, "w"), indent=2)
    todo = [c.split("=", 1) for c in a.checkpoints]
    if todo:
        ours = np.load(D / "ours256.npy", mmap_mode="r")
        from eval_tuh_clinical import load_pretrained, build_random_init
        for name, ck in todo + ([("random_init", todo[0][1])] if a.random_init else []):
            if name in res["results"]:
                continue
            model, cls, _, n_ch, args = load_pretrained(ck, dev)
            if name == "random_init":
                torch.manual_seed(0)
                model = build_random_init(cls, n_ch, args, dev)
            print(f"  [{name}]", flush=True)
            F = feats_ours(model, ours, dev)
            res["results"][name] = probe({k: F[m] for k, m in parts.items()}, y, 2)
            res["results"][name]["checkpoint"] = None if name == "random_init" else ck
            save(); del model; torch.cuda.empty_cache()
    if a.baselines:
        mono = np.load(D / "mono200.npy", mmap_mode="r")
        for b in a.baselines:
            if b in res["results"]:
                continue
            print(f"  [{b}]", flush=True)
            F = feats_baseline(b, mono, ch, tcp, dev)
            res["results"][b] = probe({k: F[m] for k, m in parts.items()}, y, 2)
            save()
    save()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
