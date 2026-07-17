"""Handcrafted spectral baseline for seq2seq sleep staging (ISRUC / HMC).

The seq2seq analog of eval_handcrafted_spectral.py. Sleep staging is the hardest,
most meaningful place to run this control: the textbook features for sleep scoring
ARE band powers (AASM rules key off delta/theta/alpha/sigma), so band power + the
same sequence transformer is a genuinely strong baseline — not a straw man.

Fairness = reuse the EXACT frozen-probe machinery from eval_sleep_seq2seq.py:
same splits (CBraMod/CSBrain), same build_sequences, same 1-layer SeqHead over
20-epoch sequences, same run_head_only training + metrics + n_reps. We replace
ONLY the per-epoch representation: pretrained encoder features -> Welch band power.

Three-way comparison (all on identical splits/head):
    Random-frozen  <  handcrafted band power  <  JEPA-frozen ?
JEPA-frozen and Random-frozen come from the existing `run_evals.sh isruc_frozen /
hmc_frozen` runs. Beating handcrafted band power HERE is the strongest evidence
that pretraining captures cross-band/-channel/-time structure the textbook
spectral features miss.

Usage:
    python eval_handcrafted_seq2seq.py --dataset isruc \\
        --data_dir /home/pxieaf/home2/datasets/isruc/subgroupI_official \\
        --cache_dir /home/pxieaf/home2/dataset_cache --n_reps 3 --output out.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_sleep_seq2seq import build_sequences, run_head_only, SEQ_LEN
from eval_handcrafted_spectral import bandpower_features, BANDS


def seq_bandpower(X_seq, fs):
    """X_seq [N, seq, T, C] -> handcrafted per-epoch features [N, seq, F]."""
    N, S, T, C = X_seq.shape
    flat = X_seq.reshape(N * S, T, C)
    feat = bandpower_features(flat, fs)              # [N*S, F]
    return feat.reshape(N, S, -1)                    # [N, seq, F]


def standardize_pad(Ftr, Fva, Fte, multiple=8):
    """Per-dim z-score on TRAIN epoch stats, then zero-pad feature dim up to a
    multiple of `multiple` so it fits the 8-head sequence transformer."""
    d = Ftr.shape[-1]
    flat_tr = Ftr.reshape(-1, d)
    mu = flat_tr.mean(0, keepdims=True)
    sd = flat_tr.std(0, keepdims=True) + 1e-6

    def _norm(F):
        return (F - mu) / sd

    Ftr, Fva, Fte = _norm(Ftr), _norm(Fva), _norm(Fte)
    pad = (-d) % multiple
    if pad:
        def _pad(F):
            z = np.zeros((*F.shape[:-1], pad), dtype=F.dtype)
            return np.concatenate([F, z], axis=-1)
        Ftr, Fva, Fte = _pad(Ftr), _pad(Fva), _pad(Fte)
    return Ftr, Fva, Fte, d + pad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["isruc", "hmc"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust")
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--n_reps", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    dsk = dict(sample_rate=args.sample_rate, trial_duration_s=args.trial_duration_s,
               normalization=args.normalization, cache_dir=args.cache_dir)
    if args.dataset == "isruc":
        from dataset_isruc import ISRUCDataset, CBRAMOD_ISRUC_SPLITS as SP, N_CLASSES
        mk = lambda subs: ISRUCDataset(data_dir=args.data_dir, subjects=subs, **dsk)
        ref = "CBraMod 0.7865 / CSBrain 0.7925 (BA)"
    else:
        from dataset_hmc import HMCDataset, make_hmc_split, N_CLASSES
        SP = make_hmc_split(args.data_dir)
        mk = lambda subs: HMCDataset(args.data_dir, subs, **dsk)
        ref = "CSBrain 0.7345 (BA)"

    print(f"\n{'='*72}")
    print(f"  Handcrafted spectral baseline — {args.dataset.upper()} seq2seq "
          f"(seq_len={SEQ_LEN})")
    print(f"  Per-epoch feature: log-abs + relative band power "
          f"({2*len(BANDS)}×n_ch), same SeqHead as frozen probe")
    print(f"  Reference: {ref}")
    print(f"{'='*72}")

    t0 = time.time()
    tr_ds, va_ds, te_ds = mk(SP["train"]), mk(SP["val"]), mk(SP["test"])
    Xtr, ytr = build_sequences(tr_ds)
    Xva, yva = build_sequences(va_ds)
    Xte, yte = build_sequences(te_ds)
    print(f"  Sequences: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape} "
          f"({(time.time()-t0)/60:.1f} min)")

    Ftr = seq_bandpower(Xtr, args.sample_rate)
    Fva = seq_bandpower(Xva, args.sample_rate)
    Fte = seq_bandpower(Xte, args.sample_rate)
    Ftr, Fva, Fte, d_model = standardize_pad(Ftr, Fva, Fte, multiple=8)
    print(f"  Handcrafted per-epoch dim (padded): {d_model}")

    Ftr_t = torch.from_numpy(Ftr.astype(np.float32))
    Fva_t = torch.from_numpy(Fva.astype(np.float32))
    Fte_t = torch.from_numpy(Fte.astype(np.float32))

    reps = []
    for rep in range(args.n_reps):
        torch.manual_seed(42 + rep); np.random.seed(42 + rep)
        m = run_head_only(Ftr_t, ytr, Fva_t, yva, Fte_t, yte, N_CLASSES,
                          d_model, device, args.max_epochs)
        print(f"  Rep {rep+1}/{args.n_reps}: BA={m['balanced_accuracy']:.4f} "
              f"κ={m['cohen_kappa']:.4f} wF1={m['weighted_f1']:.4f}")
        reps.append(m)

    agg = {}
    for k in reps[0]:
        vals = [m[k] for m in reps if isinstance(m.get(k), (int, float))]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    agg["_per_rep"] = reps

    print(f"\n  Handcrafted-spectral seq2seq ({args.n_reps} reps):")
    for k in ("balanced_accuracy", "cohen_kappa", "weighted_f1"):
        print(f"    {k:20s} {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

    results = {
        "dataset": args.dataset,
        "feature": "handcrafted_bandpower_logabs+rel",
        "feature_dim": int(d_model),
        "seq_len": SEQ_LEN, "n_classes": N_CLASSES,
        "handcrafted_frozen": agg,
        "split": {k: list(v) for k, v in SP.items()},
    }
    out = Path(args.output) if args.output else \
        Path(f"/home/pxieaf/home2/eval_results/handcrafted/handcrafted_{args.dataset}_seq2seq.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n→ Saved: {out}")


if __name__ == "__main__":
    main()
