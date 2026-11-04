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
import os
from pathlib import Path

import numpy as np
import torch

from eval_linear_probe_tasks import probe, PRED_CTX, subject_of

ROOT = Path(os.environ.get("SPECBRAM_TASK_DATA", "/home/pxieaf/home2/datasets/lp_tasks"))


class _TView:
    """[N, C, T] memmap seen as [N, T, C] without loading it (rows on demand)."""
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return np.asarray(self.arr[idx]).transpose(0, 2, 1)


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
    p.add_argument("--task", default="tuab", choices=["tuab", "tuev"])
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--baselines", nargs="*", default=[],
                   choices=["cbramod", "labram", "reve_base", "reve_large", "csbrain"])
    p.add_argument("--output", required=True)
    p.add_argument("--pred_dir", default=None, help="save test predictions per model")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = ROOT / a.task
    meta = np.load(D / "meta.npz")
    y_all, sp = meta["y"], meta["split"]
    n_cls = int(y_all.max()) + 1                  # TUAB 2 (BA/AUC-PR/AUROC), TUEV 6 (BA/kappa/F1)
    ch, tcp = [str(c) for c in meta["mono_ch"]], meta["tcp"]
    parts = {k: sp == k for k in ("train", "val", "test")}
    y = {k: y_all[m] for k, m in parts.items()}
    subj_test = subject_of(a.task, meta["rec"][parts["test"]])

    def arm(name):
        if a.pred_dir:
            PRED_CTX["path"] = str(Path(a.pred_dir) / f"{a.task}__{name}.npz")
            PRED_CTX["subj"] = subj_test
    print(a.task.upper(), {k: int(m.sum()) for k, m in parts.items()}, f"classes={n_cls}", flush=True)

    out = Path(a.output)
    res = json.load(open(out)) if out.exists() else {"task": a.task, "results": {}}
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
            arm(name)
            res["results"][name] = probe({k: F[m] for k, m in parts.items()}, y, n_cls)
            res["results"][name]["checkpoint"] = None if name == "random_init" else ck
            save(); del model; torch.cuda.empty_cache()
    if a.baselines:
        mono = np.load(D / "mono200.npy", mmap_mode="r")
        for b in a.baselines:
            if b.startswith("reve"):                 # two frozen readouts, one forward pass
                if all(f"{b}_{pl}" in res["results"] for pl in ("mean", "attn")):
                    continue
                from eval_linear_probe_v2 import reve_features
                print(f"  [{b}]", flush=True)
                FF = reve_features(b.split("_")[1], _TView(mono), ch, meta["rec"], dev)
                for pl, F in FF.items():
                    arm(f"{b}_{pl}")
                    res["results"][f"{b}_{pl}"] = probe({k: F[m] for k, m in parts.items()}, y, n_cls)
                save()
                continue
            if b in res["results"]:
                continue
            print(f"  [{b}]", flush=True)
            if b == "csbrain":
                from eval_linear_probe_v2 import csbrain_encoder
                enc = csbrain_encoder(dev, a.task)[0]
                names = [f"{ch[i]}-{ch[j]}" for i, j in tcp]
                feats = []
                with torch.no_grad():
                    for x in batches(mono, 128):                    # [B, 19, T] uV
                        x = x[:, tcp[:, 0]] - x[:, tcp[:, 1]]
                        feats.append(enc(torch.from_numpy(x.transpose(0, 2, 1).copy()).to(dev),
                                       list(range(16)), names=names).float().cpu().numpy())
                F = np.concatenate(feats)
            else:
                F = feats_baseline(b, mono, ch, tcp, dev)
            arm(b)
            res["results"][b] = probe({k: F[m] for k, m in parts.items()}, y, n_cls)
            save()
    save()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
