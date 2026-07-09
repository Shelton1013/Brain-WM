"""Sequence-to-sequence sleep staging eval (ISRUC / HMC) — CBraMod/CSBrain
comparable protocol.

Both SOTA papers (CBraMod, CSBrain) treat sleep staging as seq2seq: group 30-s
epochs into sequences of 20, run each epoch through the (pretrained) SAMPLE
ENCODER, then a shared 1-layer Transformer SEQUENCE ENCODER predicts one stage
per epoch — exploiting sleep-stage transition structure. Single-epoch numbers
are NOT comparable to their tables; this script is.

Per-epoch data comes from the existing ISRUCDataset / HMCDataset (same channel
handling, middle-10s window, per-recording robust norm — our own preprocessing,
which must match our pretrain). We only add the sequence grouping + sequence
Transformer on top.

Splits (subject-disjoint, matching CBraMod/CSBrain):
  ISRUC-I : 1-80 train / 81-90 val / 91-100 test
  HMC     : first 100 / next 25 / rest  (by sorted subject id)

Usage:
    python eval_sleep_seq2seq.py --dataset isruc \\
        --checkpoint .../checkpoint_ep8.pt \\
        --data_dir /home/pxieaf/home2/datasets/isruc/subgroupI_official \\
        --cache_dir /home/pxieaf/home2/dataset_cache \\
        --n_reps 3 --include_random_baseline
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, f1_score,
)

from eval_tuh_clinical import load_pretrained, build_random_init

SEQ_LEN = 20


def compute_metrics(y_true, y_pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "weighted_f1":       float(f1_score(y_true, y_pred, average="weighted")),
    }


def build_sequences(ds, seq_len=SEQ_LEN):
    """Group a per-epoch dataset into non-overlapping seq_len sequences within
    each subject (temporal order preserved). Returns X [N,seq,T,C], y [N,seq]."""
    from collections import OrderedDict
    by_subj = OrderedDict()
    for i, s in enumerate(ds.subject_ids):
        by_subj.setdefault(s, []).append(i)
    Xs, ys = [], []
    for _, idxs in by_subj.items():
        for k in range(0, len(idxs) - seq_len + 1, seq_len):
            chunk = idxs[k:k + seq_len]
            Xs.append(np.stack([ds.trials[j] for j in chunk]))
            ys.append(np.array([ds.labels[j] for j in chunk], dtype=np.int64))
    if not Xs:
        raise ValueError("no full sequences built (recording shorter than seq_len?)")
    return np.stack(Xs).astype(np.float32), np.stack(ys)


class SeqHead(nn.Module):
    """1-layer Transformer sequence encoder + per-position classifier."""
    def __init__(self, d_model, n_classes, nhead=8):
        super().__init__()
        self.seq_enc = nn.TransformerEncoderLayer(
            d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, feats):          # feats: [B, seq, d]
        h = self.seq_enc(feats)
        return self.cls(self.norm(h))  # [B, seq, n_classes]


def _encode_epochs(model, x_seq):
    """x_seq: [B, seq, T, C] → per-epoch pooled features [B, seq, d]."""
    B, S, T, C = x_seq.shape
    flat = x_seq.reshape(B * S, T, C)
    feats = model._encode(model._tokenize(flat)).mean(1)   # [B*S, d]
    return feats.reshape(B, S, -1)


def run_finetune_seq(base_model, Xtr, ytr, Xva, yva, Xte, yte, n_classes,
                     device, max_epochs=50, batch_size=16):
    model = copy.deepcopy(base_model)
    head = SeqHead(model.d_model, n_classes).to(device)

    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=batch_size, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=batch_size, shuffle=False, num_workers=2,
                    pin_memory=True, persistent_workers=True)

    cw = torch.bincount(torch.from_numpy(ytr).reshape(-1).long(), minlength=n_classes).float()
    cw = (1.0 / cw.clamp(min=1)); cw = (cw / cw.sum() * n_classes).to(device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    steps = max(1, len(tr))
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=1e-4, weight_decay=5e-2, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs * steps, eta_min=1e-6)

    best_val, best_state, patience, no_imp, ep = 0.0, None, 10, 0, 0
    for ep in range(max_epochs):
        model.train(); head.train()
        t0 = time.time()
        for bx, by in tr:
            bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
            logits = head(_encode_epochs(model, bx))      # [B,seq,C]
            loss = crit(logits.reshape(-1, n_classes), by.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0)
            opt.step(); sched.step()

        model.eval(); head.eval()
        vp, vl = [], []
        with torch.no_grad():
            for bx, by in va:
                bx = bx.to(device)
                p = head(_encode_epochs(model, bx)).argmax(-1).cpu()
                vp.append(p.reshape(-1)); vl.append(by.reshape(-1))
        val_ba = balanced_accuracy_score(torch.cat(vl).numpy(), torch.cat(vp).numpy())
        imp = val_ba > best_val
        if imp:
            best_val = val_ba
            best_state = {"m": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                          "h": {k: v.cpu().clone() for k, v in head.state_dict().items()}}
            no_imp = 0
        else:
            no_imp += 1
        print(f"      ep{ep+1:03d}{'*' if imp else ' '} val_ba={val_ba:.4f} "
              f"best={best_val:.4f} no_improve={no_imp}/{patience} "
              f"{time.time()-t0:.0f}s", flush=True)
        if no_imp >= patience:
            print(f"      early stop ep{ep+1} (best_val={best_val:.4f})"); break

    if best_state:
        model.load_state_dict(best_state["m"]); model.to(device)
        head.load_state_dict(best_state["h"]);  head.to(device)
    model.eval(); head.eval()
    preds = []
    Xte_t = torch.from_numpy(Xte)
    with torch.no_grad():
        for i in range(0, len(Xte_t), batch_size):
            bx = Xte_t[i:i+batch_size].to(device)
            preds.append(head(_encode_epochs(model, bx)).argmax(-1).cpu().numpy().reshape(-1))
    preds = np.concatenate(preds)
    m = compute_metrics(yte.reshape(-1), preds)
    m["best_val_ba"] = float(best_val); m["epochs"] = int(ep + 1)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["isruc", "hmc"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache_dir", default="/home/pxieaf/home2/dataset_cache")
    p.add_argument("--sample_rate", type=int, default=256)
    p.add_argument("--trial_duration_s", type=int, default=10)
    p.add_argument("--normalization", default="per_recording_robust")
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_reps", type=int, default=3)
    p.add_argument("--include_random_baseline", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device if args.device != "auto" else "cpu")

    # Build per-epoch datasets for the 3 splits, then group into sequences.
    dsk = dict(sample_rate=args.sample_rate, trial_duration_s=args.trial_duration_s,
               normalization=args.normalization, cache_dir=args.cache_dir)
    if args.dataset == "isruc":
        from dataset_isruc import ISRUCDataset, CBRAMOD_ISRUC_SPLITS as SP, N_CLASSES
        mk = lambda subs: ISRUCDataset(data_dir=args.data_dir, subjects=subs, **dsk)
        tr_ds, va_ds, te_ds = mk(SP["train"]), mk(SP["val"]), mk(SP["test"])
        ref = "CBraMod 0.7865 / CSBrain 0.7925 (BA)"
    else:
        from dataset_hmc import HMCDataset, make_hmc_split, N_CLASSES
        SP = make_hmc_split(args.data_dir)
        mk = lambda subs: HMCDataset(args.data_dir, subs, **dsk)
        tr_ds, va_ds, te_ds = mk(SP["train"]), mk(SP["val"]), mk(SP["test"])
        ref = "CSBrain 0.7345 (BA)"

    print(f"\n{'='*72}\n  {args.dataset.upper()} seq2seq sleep staging (seq_len={SEQ_LEN})\n{'='*72}")
    print(f"  Reference: {ref}")
    model, model_cls, mtype, n_ch, ckpt_args = load_pretrained(args.checkpoint, device)

    t0 = time.time()
    Xtr, ytr = build_sequences(tr_ds)
    Xva, yva = build_sequences(va_ds)
    Xte, yte = build_sequences(te_ds)
    print(f"Sequences: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape} "
          f"({(time.time()-t0)/60:.1f} min)")

    results = {"checkpoint": args.checkpoint, "dataset": args.dataset,
               "model_type": mtype, "seq_len": SEQ_LEN, "n_classes": N_CLASSES,
               "split": {k: list(v) for k, v in SP.items()}}

    def _agg(reps):
        agg = {}
        for k in reps[0]:
            vals = [m[k] for m in reps if isinstance(m.get(k), (int, float))]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        agg["_per_rep"] = reps
        return agg

    print(f"\n  JEPA seq2seq FT ({args.n_reps} reps × {args.max_epochs} ep)")
    jr = []
    for rep in range(args.n_reps):
        torch.manual_seed(42 + rep); np.random.seed(42 + rep)
        print(f"  Rep {rep+1}/{args.n_reps}")
        m = run_finetune_seq(model, Xtr, ytr, Xva, yva, Xte, yte, N_CLASSES,
                             device, args.max_epochs, args.batch_size)
        print(f"    BA={m['balanced_accuracy']:.4f} κ={m['cohen_kappa']:.4f} "
              f"wF1={m['weighted_f1']:.4f}")
        jr.append(m)
    results["jepa_finetune"] = _agg(jr)

    if args.include_random_baseline:
        print(f"\n  Random-init seq2seq FT ({args.n_reps} reps)")
        rr = []
        for rep in range(args.n_reps):
            torch.manual_seed(42 + rep); np.random.seed(42 + rep)
            print(f"  Rep {rep+1}/{args.n_reps}")
            rm = build_random_init(model_cls, n_ch, ckpt_args, device)
            m = run_finetune_seq(rm, Xtr, ytr, Xva, yva, Xte, yte, N_CLASSES,
                                 device, args.max_epochs, args.batch_size)
            print(f"    BA={m['balanced_accuracy']:.4f} κ={m['cohen_kappa']:.4f} "
                  f"wF1={m['weighted_f1']:.4f}")
            rr.append(m)
        results["random_finetune"] = _agg(rr)

    print(f"\n{'='*72}\n  SUMMARY ({args.dataset.upper()} seq2seq)\n{'='*72}")
    print(f"  {'Model':22s} {'BA':>9s} {'κ':>9s} {'wF1':>9s}")
    if "random_finetune" in results:
        r = results["random_finetune"]
        print(f"  {'Random (Ours)':22s} {r['balanced_accuracy']['mean']:>9.4f} "
              f"{r['cohen_kappa']['mean']:>9.4f} {r['weighted_f1']['mean']:>9.4f}")
    j = results["jepa_finetune"]
    print(f"  {'JEPA (Ours)':22s} {j['balanced_accuracy']['mean']:>9.4f} "
          f"{j['cohen_kappa']['mean']:>9.4f} {j['weighted_f1']['mean']:>9.4f}")

    out = Path(args.output) if args.output else Path(args.checkpoint).parent / f"{args.dataset}_seq2seq.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n→ Saved: {out}")


if __name__ == "__main__":
    main()
