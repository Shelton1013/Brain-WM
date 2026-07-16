"""
train_v2.py — Dedicated pretrain script for EEG-LeJEPA v2.

v2 model:
  - 24M params (d_model 512, 12 layers)
  - Channel-agnostic criss-cross attention tokenizer
  - JEPA + MAE hybrid loss + reduced CF + SIGReg + PAJR

Usage (8-GPU DDP):
  torchrun --standalone --nproc_per_node=8 train_v2.py \\
      --output_dir /home/pxieaf/home2/model/lejepa_v2 \\
      --tueg_dir /home/pxieaf/home2/tuh/tuh_eeg/v2.0.1/edf \\
      --data_cache_dir /home/pxieaf/home2/dataset_cache \\
      --epochs 15 --batch_size 4 --lr 3e-4 \\
      --d_model 512 --encoder_layers 12 \\
      --patch_len 200 --mask_ratio 0.5 \\
      --jepa_weight 1.0 --mae_weight 0.5 --cf_weight 0.3 \\
      --reg_type sigreg --sigreg_lambda 0.05

Reuses:
  - Dataset loader from train_jepa (multi-dataset TUEG loading)
  - Training loop (DDP + AdamW + linear-warmup + cosine)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from pprint import pprint as _pprint

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Reuse dataset code from dataset_multi (same as train_jepa)
from dataset_multi import MultiDatasetEEG, _sid_before_underscore

from eeg_lejepa_v2 import EEGLeJEPA_v2, count_params
from eeg_lejepa_v3 import EEGLeJEPA_v3


# ============================================================
# DDP helpers
# ============================================================

def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return get_rank() == 0


def setup_ddp():
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False


def pprint(*args, **kw):
    if is_main():
        print(*args, **kw)


# ============================================================
# LR scheduler: linear warmup + cosine
# ============================================================

def build_warmup_cosine_schedule(optimizer, warmup_steps: int,
                                   total_steps: int, min_lr_ratio: float = 1e-3):
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch,
                    grad_clip: float = 3.0, log_every: int = 50):
    model.train()
    total, n_batches = 0.0, 0
    losses_sum = {"jepa": 0.0, "mae": 0.0, "cf": 0.0, "sig": 0.0, "pajr": 0.0}
    t0 = time.time()

    for step, batch in enumerate(loader):
        # dataset returns (eeg, subject_id, ...)
        if isinstance(batch, (tuple, list)):
            eeg = batch[0].to(device, non_blocking=True)
            subject_ids = batch[1].long().to(device, non_blocking=True) \
                if len(batch) > 1 else None
        else:
            eeg = batch.to(device, non_blocking=True)
            subject_ids = None

        loss_dict = model(eeg, subject_ids)
        loss = loss_dict["total"]

        # NaN/Inf guard (safety net when SKIP_FINITE_CHECK bypasses the cache
        # scan, or on a rare EA-ill-conditioned trial): drop this batch.
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total += loss.item()
        n_batches += 1
        for k in losses_sum:
            v = loss_dict.get(k, 0.0)
            losses_sum[k] += v.item() if hasattr(v, "item") else float(v)

        if step % log_every == 0 and is_main():
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            avg_loss = total / max(1, n_batches)
            comp_str = " ".join(f"{k}={losses_sum[k]/max(1,n_batches):.4f}"
                                for k in losses_sum)
            print(f"ep{epoch}  step {step:4d}  loss={avg_loss:.4f}  "
                  f"lr={lr:.2e}  {comp_str}  ({elapsed:.1f}s)", flush=True)

    return total / max(1, n_batches), {k: v / max(1, n_batches)
                                          for k, v in losses_sum.items()}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pretrain EEG-LeJEPA v2 (24M param + criss-cross + JEPA+MAE)")

    # Data
    parser.add_argument("--data_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets/physionet")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_subjects", type=int, default=109,
                        help="physionet MI subjects (for PAJR discriminator "
                             "size). 0 = skip physionet entirely (TUEG-only).")
    parser.add_argument("--multi_dataset", action="store_true")
    parser.add_argument("--moabb_datasets", type=str, nargs="*",
                        default=["Cho2017", "Lee2019_MI"])
    parser.add_argument("--moabb_min_channels", type=int, default=19)
    parser.add_argument("--moabb_min_subjects", type=int, default=10)
    parser.add_argument("--hbn_dir", type=str, default=None)
    parser.add_argument("--tueg_dir", type=str, default=None)
    parser.add_argument("--tueg_max_files", type=int, default=None)
    parser.add_argument("--tueg_exclude_dirs", type=str, nargs="*", default=[])
    parser.add_argument("--download_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets")
    parser.add_argument("--data_cache_dir", type=str,
                        default="/home/pxieaf/home2/dataset_cache")

    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_decoder", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--patch_len", type=int, default=200,
                        help="samples per patch (200 for 200Hz × 1s)")
    parser.add_argument("--max_time_patches", type=int, default=64,
                        help="pos_embed size for time dim")
    parser.add_argument("--max_channels", type=int, default=32,
                        help="pos_embed size for channel dim")
    parser.add_argument("--mask_ratio", type=float, default=0.5)

    # Loss weights
    parser.add_argument("--jepa_weight", type=float, default=1.0)
    parser.add_argument("--mae_weight", type=float, default=0.5,
                        help="signal-space reconstruction weight (NEW in v2)")
    parser.add_argument("--cf_weight", type=float, default=0.3,
                        help="cross-frequency weight (reduced from v1's 1.0)")
    parser.add_argument("--sigreg_lambda", type=float, default=0.05,
                        help="anti-collapse weight; use 0.0 for no-reg baseline")
    parser.add_argument("--pajr_weight", type=float, default=0.1)
    parser.add_argument("--arch", type=str, default="v2", choices=["v2", "v3"],
                        help="v3 = frequency-native (filterbank tokenizer + real "
                             "cross-frequency spectral prediction, no MAE)")
    parser.add_argument("--drop_short_recording_min", type=float, default=0.0)
    parser.add_argument("--trim_start_end_sec", type=int, default=0)
    parser.add_argument("--notch_freq", type=float, default=0.0)
    parser.add_argument("--reject_abs_uv", type=float, default=0.0)
    parser.add_argument("--n_bands", type=int, default=5)
    parser.add_argument("--band_mask_ratio", type=float, default=0.30)
    parser.add_argument("--filt_kernel", type=int, default=65)
    parser.add_argument("--cf_learnable_target", action="store_true",
                        help="ablation: use the LEARNABLE filterbank as the CF "
                             "target (collapse-prone; demonstrates why a fixed "
                             "target is needed)")
    parser.add_argument("--reg_type", type=str, default="sigreg",
                        choices=["sigreg", "vicreg"],
                        help="sigreg is SAFE (v1 confirmed); vicreg DESTROYS features")
    parser.add_argument("--cf_d_band", type=int, default=64)
    parser.add_argument("--cf_band_conditioned", type=int, default=1,
                        choices=[0, 1])

    # Training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="per-GPU batch. 24M @ d_model 512 uses ~15-20GB")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # Data-loading options (from train_jepa)
    parser.add_argument("--normalization", type=str, default="per_recording_robust",
                        choices=["per_trial_zscore", "per_recording_robust"])
    parser.add_argument("--trial_duration_s", type=int, default=10)

    args = parser.parse_args()

    # ─── DDP init ─────────────────────────────────────────────────
    device, distributed = setup_ddp()
    world_size = get_world_size()
    torch.manual_seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    pprint(f"[DDP] rank={get_rank()} world_size={world_size} device={device}")

    # ─── Dataset ──────────────────────────────────────────────────
    pprint("Building dataset (matches clean_v1 config for cache-hit)...")
    sources = []
    if args.n_subjects > 0:
        sources.append({"type": "physionet", "n_subjects": args.n_subjects})
    if args.multi_dataset:
        for name in (args.moabb_datasets or []):
            sources.append({"type": "moabb", "name": name})

    # TUEG exclude set (must match prebuild_tueg_cache for cache-hit)
    tueg_exclude_ids = set()
    if args.tueg_dir and args.tueg_exclude_dirs:
        for d in args.tueg_exclude_dirs:
            d = Path(d)
            if not d.is_dir():
                continue
            for edf in d.rglob("*.edf"):
                pid = _sid_before_underscore(edf)
                if pid:
                    tueg_exclude_ids.add(pid)
        pprint(f"  TUEG exclude: {len(tueg_exclude_ids)} patient IDs")

    if args.tueg_dir:
        src_dict = {"type": "tueg", "path": args.tueg_dir,
                    "max_files": args.tueg_max_files}
        if tueg_exclude_ids:
            src_dict["exclude_patient_ids"] = tueg_exclude_ids
        sources.append(src_dict)

    dataset = MultiDatasetEEG(
        sources=sources,
        physionet_data_dir=args.data_dir,
        download_dir=args.download_dir,
        cache_dir=args.data_cache_dir if args.data_cache_dir else None,
        trial_duration_s=args.trial_duration_s,
        normalization=args.normalization,
        # CBraMod-style artifact filters — must match prebuild for cache-hit.
        drop_short_recording_min=args.drop_short_recording_min,
        trim_start_end_sec=args.trim_start_end_sec,
        notch_freq=args.notch_freq,
        reject_abs_uv=args.reject_abs_uv,
    )
    pprint(f"Dataset: {len(dataset)} trials, {dataset.n_subjects} subjects, "
           f"{len(dataset.electrode_names)} channels")

    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) \
        if distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=args.num_workers > 0,
    )
    pprint(f"Effective batch size: {args.batch_size} × {world_size} = "
           f"{args.batch_size * world_size}")

    # ─── Model ────────────────────────────────────────────────────
    model_kwargs = dict(
        n_channels=19,   # reference only; criss-cross accepts any C
        patch_len=args.patch_len,
        d_model=args.d_model,
        d_decoder=args.d_decoder,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_heads=8,
        decoder_heads=8,
        mask_ratio=args.mask_ratio,
        n_subjects=int(dataset.n_subjects),
        reg_type=args.reg_type,
        jepa_weight=args.jepa_weight,
        mae_weight=args.mae_weight,
        cf_weight=args.cf_weight,
        sigreg_lambda=args.sigreg_lambda,
        pajr_weight=args.pajr_weight,
        n_bands=5,
        d_band_view=args.cf_d_band,
        cf_band_conditioned=bool(args.cf_band_conditioned),
        max_time_patches=args.max_time_patches,
        max_channels=args.max_channels,
    )
    if args.arch == "v3":
        model = EEGLeJEPA_v3(
            d_model=args.d_model, encoder_layers=args.encoder_layers, n_heads=8,
            patch_len=args.patch_len, max_time_patches=args.max_time_patches,
            max_channels=args.max_channels, n_bands=args.n_bands,
            d_band=args.cf_d_band, filt_kernel=args.filt_kernel,
            sample_rate=256, band_mask_ratio=args.band_mask_ratio,
            jepa_weight=args.jepa_weight, cf_weight=args.cf_weight,
            sigreg_lambda=args.sigreg_lambda, reg_type=args.reg_type,
            cf_learnable_target=args.cf_learnable_target,
        ).to(device)
    else:
        model = EEGLeJEPA_v2(**model_kwargs).to(device)
    raw_model = model
    if distributed:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                    find_unused_parameters=True)
    n_params = count_params(raw_model)
    pprint(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ─── Optimizer + scheduler ────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    betas=(0.9, 0.95))
    total_steps = len(loader) * args.epochs
    warmup_steps = len(loader) * args.warmup_epochs
    scheduler = build_warmup_cosine_schedule(optimizer, warmup_steps, total_steps)

    # ─── Save args ────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "args.json", "w") as f:
            args_dict = vars(args)
            args_dict["effective_batch_size"] = args.batch_size * world_size
            args_dict["n_params"] = n_params
            args_dict["n_dataset"] = len(dataset)
            args_dict["n_subjects_actual"] = int(dataset.n_subjects)
            json.dump(args_dict, f, indent=2)

    # ─── Train ────────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        avg_loss, comp = train_one_epoch(
            model, loader, optimizer, scheduler, device, epoch,
            grad_clip=args.grad_clip,
        )

        if is_main():
            comp_str = " ".join(f"{k}={comp[k]:.4f}" for k in comp)
            pprint(f"[Epoch {epoch}/{args.epochs}] avg_loss={avg_loss:.4f}  {comp_str}")

            # Save checkpoint
            args_to_save = vars(args).copy()
            args_to_save["model"] = "lejepa_v3" if args.arch == "v3" else "lejepa_v2"
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "args": args_to_save,
            }, output_dir / f"checkpoint_ep{epoch}.pt")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "loss": avg_loss,
                    "args": args_to_save,
                }, output_dir / "best_model.pt")
                # ⚠ best_model.pt = LOWEST PRETRAIN LOSS (MAE-dominated). For
                # this SSL, lower loss = more-trained ≠ better downstream —
                # over-training collapses transfer. DO NOT pick best_model.pt
                # for downstream; sweep checkpoint_ep*.pt and select the epoch
                # by a downstream proxy (early epochs usually win).
                pprint(f"  → best_model.pt saved (min-loss {avg_loss:.4f}); "
                       f"NOTE: pick downstream ckpt from checkpoint_ep*.pt, "
                       f"NOT best_model.pt")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
