"""Label-efficiency for seq2seq sleep (ISRUC / HMC).

Same idea as eval_label_efficiency.py but for the seq2seq protocol: encode every
20-epoch sequence through the FROZEN encoder ONCE, then train the 1-layer seq
transformer + head on a subsampled fraction of TRAIN sequences (full val for
early stop, full test for eval). Pretrained vs random-init on identical splits.

Claim to look for: at 1-10% of training sequences, pretrained >> random (gap =
label efficiency), same as the Mumtaz per-sample curve.

    python eval_label_efficiency_seq2seq.py --dataset isruc \\
        --checkpoint .../checkpoint_ep8.pt \\
        --data_dir /home/pxieaf/home2/datasets/isruc/subgroupI_official \\
        --output le_isruc.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_tuh_clinical import load_pretrained, build_random_init
from eval_sleep_seq2seq import (
    build_sequences, precompute_seq_features, run_head_only, SEQ_LEN,
)

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def load_seqs(dataset, args):
    dsk = dict(sample_rate=args.sample_rate, trial_duration_s=args.trial_duration_s,
               normalization=args.normalization, cache_dir=args.cache_dir)
    if dataset == "isruc":
        from dataset_isruc import ISRUCDataset, CBRAMOD_ISRUC_SPLITS as SP, N_CLASSES
        mk = lambda subs: ISRUCDataset(data_dir=args.data_dir, subjects=subs, **dsk)
    else:
        from dataset_hmc import HMCDataset, make_hmc_split, N_CLASSES
        SP = make_hmc_split(args.data_dir)
        mk = lambda subs: HMCDataset(args.data_dir, subs, **dsk)
    Xtr, ytr = build_sequences(mk(SP["train"]))
    Xva, yva = build_sequences(mk(SP["val"]))
    Xte, yte = build_sequences(mk(SP["test"]))
    return Xtr, ytr, Xva, yva, Xte, yte, N_CLASSES


def curve(Ftr, ytr, Fva, yva, Fte, yte, n_classes, d_model, device, n_reps, max_epochs):
    res = {}
    N = Ftr.shape[0]
    for frac in FRACTIONS:
        reps = []
        for r in range(n_reps if frac < 1.0 else 1):
            rng = np.random.RandomState(100 + r)
            n = max(4, int(round(N * frac)))
            idx = rng.choice(N, min(n, N), replace=False)
            m = run_head_only(Ftr[idx], ytr[idx], Fva, yva, Fte, yte,
                              n_classes, d_model, device, max_epochs)
            reps.append(m)
        agg = {}
        for k in ("balanced_accuracy", "cohen_kappa", "weighted_f1"):
            v = [x[k] for x in reps]
            agg[k] = (float(np.mean(v)), float(np.std(v)))
        agg["n_train_seq"] = int(max(4, round(N * frac)))
        res[f"{frac:.2f}"] = agg
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["isruc", "hmc"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust")
    p.add_argument("--n_reps", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    print(f"\n{'='*72}\n  Label efficiency (seq2seq) — {args.dataset}\n{'='*72}")
    model, model_cls, mtype, n_ch, ckpt_args = load_pretrained(args.checkpoint, device)
    Xtr, ytr, Xva, yva, Xte, yte, n_classes = load_seqs(args.dataset, args)
    print(f"  seqs: train {Xtr.shape[0]}  val {Xva.shape[0]}  test {Xte.shape[0]}")

    t = time.time()
    Ftr = precompute_seq_features(model, Xtr, device, args.batch_size)
    Fva = precompute_seq_features(model, Xva, device, args.batch_size)
    Fte = precompute_seq_features(model, Xte, device, args.batch_size)
    print(f"  pretrained features encoded ({(time.time()-t)/60:.1f} min)")
    rm = build_random_init(model_cls, n_ch, ckpt_args, device)
    Rtr = precompute_seq_features(rm, Xtr, device, args.batch_size)
    Rva = precompute_seq_features(rm, Xva, device, args.batch_size)
    Rte = precompute_seq_features(rm, Xte, device, args.batch_size)
    print(f"  random features encoded")

    d = model.d_model
    pre = curve(Ftr, ytr, Fva, yva, Fte, yte, n_classes, d, device, args.n_reps, args.max_epochs)
    rnd = curve(Rtr, ytr, Rva, yva, Rte, yte, n_classes, d, device, args.n_reps, args.max_epochs)

    print(f"\n  {'frac':>6} {'n_seq':>6} {'pretrained BA':>16} {'random BA':>16} {'gap':>7}")
    for f in [f"{x:.2f}" for x in FRACTIONS]:
        pb = pre[f]["balanced_accuracy"]; rb = rnd[f]["balanced_accuracy"]
        print(f"  {f:>6} {pre[f]['n_train_seq']:>6} {pb[0]:>8.3f}±{pb[1]:.3f}   "
              f"{rb[0]:>8.3f}±{rb[1]:.3f}   {pb[0]-rb[0]:>+.3f}")

    out = {"checkpoint": args.checkpoint, "dataset": args.dataset,
           "model_type": mtype, "fractions": FRACTIONS,
           "pretrained": pre, "random": rnd}
    path = args.output or f"/home/pxieaf/home2/eval_results/le_{args.dataset}_seq2seq.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n→ Saved: {path}")


if __name__ == "__main__":
    main()
