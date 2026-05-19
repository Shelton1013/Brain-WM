"""
Train EEG-JEPA baseline.

Usage:
  # Single GPU
  python train_jepa.py --n_subjects 10 --epochs 10

  # Multi-GPU
  CUDA_VISIBLE_DEVICES=3,7 torchrun --nproc_per_node=2 train_jepa.py --n_subjects 109 --epochs 50 --batch_size 8
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from eeg_jepa import EEGJEPA
from dataset import PhysioNetMIDataset
from dataset_multi import MultiDatasetEEG


# ============================================================
# Distributed helpers (same as train.py)
# ============================================================

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main():
    return get_rank() == 0

def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), False

def cleanup_dist():
    if is_dist():
        dist.destroy_process_group()

def pprint(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


# ============================================================
# LR scheduler
# ============================================================

def cosine_schedule(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / optimizer.defaults["lr"],
                   0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Train / Validate
# ============================================================

def train_epoch(model, raw_model, loader, optimizer, scheduler, device,
                epoch, global_step, total_steps):
    model.train()
    total_loss = 0.0
    n = 0

    for batch_idx, (eeg, _subject_ids) in enumerate(loader):
        eeg = eeg.to(device)

        progress = global_step / max(total_steps, 1)
        raw_model.set_training_progress(progress)

        outputs = model(eeg, return_predictions=True)
        losses = raw_model.compute_loss(outputs)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()

        raw_model.update_ema()

        total_loss += losses["total"].item()
        n += 1
        global_step += 1

        if batch_idx % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            ema_d = raw_model.ema.decay if hasattr(raw_model, 'ema') and raw_model.ema else 0
            pprint(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={losses['total']:.6f} "
                f"pred={losses['pred']:.6f} "
                f"var={losses.get('var', 0):.4f} "
                f"cov={losses.get('cov', 0):.4f} "
                f"lr={lr:.2e} ema={ema_d:.4f}"
            )

    return total_loss / max(n, 1), global_step


@torch.no_grad()
def validate(model, raw_model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for eeg, _ in loader:
        eeg = eeg.to(device)
        outputs = model(eeg, return_predictions=True)
        losses = raw_model.compute_loss(outputs)
        total_loss += losses["total"].item()
        n += 1

    if is_dist():
        t = torch.tensor(total_loss, device=device)
        dist.all_reduce(t)
        total_loss = t.item()
        nt = torch.tensor(n, device=device)
        dist.all_reduce(nt)
        n = nt.item()

    return total_loss / max(n, 1)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train EEG-JEPA baseline")
    parser.add_argument("--data_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets/physionet")
    parser.add_argument("--output_dir", type=str,
                        default="/home/share/data_makchen/peng/models/eeg_jepa")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_subjects", type=int, default=109)
    parser.add_argument("--multi_dataset", action="store_true",
                        help="Use multi-dataset pretraining (PhysioNet + MOABB)")
    parser.add_argument("--moabb_datasets", type=str, nargs="*",
                        default=["Cho2017", "Lee2019_MI", "BNCI2014001",
                                 "Shin2017A", "Weibo2014"],
                        help="MOABB datasets to include")
    parser.add_argument("--edf_dir", type=str, default=None,
                        help="Path to directory of .edf files (e.g., TUH corpus)")
    parser.add_argument("--edf_max_files", type=int, default=None)
    parser.add_argument("--download_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets",
                        help="Where MOABB auto-downloads data")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--mask_ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device, distributed = setup_dist()
    world_size = get_world_size()
    pprint(f"Device: {device}, World size: {world_size}")

    torch.manual_seed(args.seed + get_rank())

    # Dataset
    pprint("Loading dataset...")
    if args.multi_dataset:
        sources = [{"type": "physionet", "n_subjects": args.n_subjects}]
        for name in (args.moabb_datasets or []):
            sources.append({"type": "moabb", "name": name})
        if args.edf_dir:
            sources.append({"type": "edf_dir", "path": args.edf_dir,
                            "max_files": args.edf_max_files})
        dataset = MultiDatasetEEG(
            sources=sources,
            physionet_data_dir=args.data_dir,
            download_dir=args.download_dir,
        )
    else:
        dataset = PhysioNetMIDataset(
            subjects=list(range(1, args.n_subjects + 1)),
            sample_rate=256,
            trial_duration_s=4,
            data_dir=args.data_dir,
        )
    n_channels = len(dataset.electrode_names)
    pprint(f"Channels: {n_channels}, Trials: {len(dataset)}")

    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=4, pin_memory=True,
    )

    # Model
    pprint(f"Building EEG-JEPA (d={args.d_model}, layers={args.encoder_layers}, "
           f"mask={args.mask_ratio:.0%})...")
    model = EEGJEPA(
        n_channels=n_channels,
        state_samples=26,
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        encoder_heads=8,
        predictor_layers=3,
        predictor_dim=128,
        predictor_heads=4,
        mask_ratio=args.mask_ratio,
        n_subjects=args.n_subjects,
    ).to(device)

    raw_model = model
    if distributed:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                    find_unused_parameters=True)

    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    pprint(f"Parameters: {n_params:,}")

    # Optimizer
    scaled_lr = args.lr * world_size
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=scaled_lr,
                                  weight_decay=0.05, betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 5
    scheduler = cosine_schedule(optimizer, warmup_steps, total_steps)

    # Output
    output_dir = Path(args.output_dir)
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Training
    pprint(f"\nTraining {args.epochs} epochs on {world_size} GPU(s)...")
    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if distributed:
            train_sampler.set_epoch(epoch)

        train_loss, global_step = train_epoch(
            model, raw_model, train_loader, optimizer, scheduler,
            device, epoch, global_step, total_steps,
        )
        val_loss = validate(model, raw_model, val_loader, device)
        elapsed = time.time() - t0

        pprint(f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) | "
               f"Train={train_loss:.6f} Val={val_loss:.6f}")

        if is_main() and val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "val_loss": best_val,
                "args": vars(args),
            }, output_dir / "best_model.pt")
            pprint(f"  → Best model saved (val={best_val:.6f})")

    pprint(f"\nDone. Best val loss: {best_val:.6f}")
    cleanup_dist()


if __name__ == "__main__":
    main()
