"""BrainWM v2 training script with DDP (DistributedDataParallel) multi-GPU support.

Usage:
  # Single GPU
  CUDA_VISIBLE_DEVICES=7 python train.py --n_subjects 10 --epochs 10

  # Multi-GPU (2 cards)
  CUDA_VISIBLE_DEVICES=7,8 torchrun --nproc_per_node=2 train.py --n_subjects 109 --epochs 50

  # Multi-GPU (4 cards)
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py --n_subjects 109 --epochs 50
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from config import BrainWMConfig
from model import BrainWM
from dataset import EEGDataset, PhysioNetMIDataset


# ============================================================
# Distributed helpers
# ============================================================

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main_process():
    return get_rank() == 0

def setup_distributed():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), True
    else:
        # Single GPU fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, False

def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()

def print_main(*args, **kwargs):
    """Only print on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


# ============================================================
# LR scheduler
# ============================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Train / Validate
# ============================================================

def train_one_epoch(model, raw_model, dataloader, optimizer, scheduler, config,
                    device, epoch, global_step, total_steps):
    """
    Args:
        model: DDP-wrapped model (or raw model in single-GPU mode)
        raw_model: unwrapped model (for EMA update and progress setting)
    """
    model.train()
    total_losses = {"total": 0, "adv": 0, "rmask": 0}
    for k in config.prediction_horizons:
        total_losses[f"pred_k{k}"] = 0
    n_batches = 0

    for batch_idx, (eeg, subject_ids) in enumerate(dataloader):
        eeg = eeg.to(device)
        subject_ids = subject_ids.to(device)

        # Update training progress (on raw model, not DDP wrapper)
        progress = global_step / max(total_steps, 1)
        raw_model.set_training_progress(progress)

        # Forward
        outputs = model(eeg, return_predictions=True)
        losses = raw_model.compute_loss(outputs, subject_ids=subject_ids)

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()

        # Update EMA (on raw model)
        raw_model.update_ema()

        for k in total_losses:
            if k in losses:
                total_losses[k] += losses[k].item()
        n_batches += 1
        global_step += 1

        if batch_idx % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            ema_m = raw_model.ema_encoder.current_decay if raw_model.ema_encoder else 0
            pred_str = " ".join(
                f"k{k}={losses.get(f'pred_k{k}', 0):.4f}" for k in config.prediction_horizons
            )
            print_main(
                f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"loss={losses['total']:.4f} {pred_str} "
                f"rmask={losses.get('rmask', 0):.4f} "
                f"adv={losses.get('adv', 0):.4f} "
                f"lr={lr:.2e} ema={ema_m:.4f} \u03b1={raw_model.adv_alpha:.2f}"
            )

    avg = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    return avg, global_step


@torch.no_grad()
def validate(model, raw_model, dataloader, device):
    model.eval()
    total_losses = {"total": 0}
    n_batches = 0

    for eeg, subject_ids in dataloader:
        eeg = eeg.to(device)
        subject_ids = subject_ids.to(device)
        outputs = model(eeg, return_predictions=True)
        losses = raw_model.compute_loss(outputs, subject_ids=subject_ids)
        total_losses["total"] += losses["total"].item()
        n_batches += 1

    # Average across all GPUs
    if is_dist():
        for k in total_losses:
            t = torch.tensor(total_losses[k], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_losses[k] = t.item()
        n_t = torch.tensor(n_batches, device=device)
        dist.all_reduce(n_t, op=dist.ReduceOp.SUM)
        n_batches = n_t.item()

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train BrainWM v2 (DDP)")
    parser.add_argument("--data_dir", type=str, default="/home/share/data_makchen/peng/datasets/physionet")
    parser.add_argument("--dataset", type=str, default="physionet",
                        choices=["physionet", "preprocessed"])
    parser.add_argument("--output_dir", type=str, default="/home/share/data_makchen/peng/models/brainwm")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size PER GPU. Effective batch = this × n_gpus")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_subjects", type=int, default=109)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup distributed
    device, distributed = setup_distributed()
    world_size = get_world_size()
    rank = get_rank()

    print_main(f"Device: {device}, World size: {world_size}, Distributed: {distributed}")

    torch.manual_seed(args.seed + rank)  # Different seed per rank for data diversity
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    config = BrainWMConfig()
    config.n_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr

    # Dataset (all ranks load the same dataset, sampler handles splitting)
    print_main("Loading dataset...")
    if args.dataset == "physionet":
        dataset = PhysioNetMIDataset(
            subjects=list(range(1, args.n_subjects + 1)),
            sample_rate=config.sample_rate,
            trial_duration_s=config.trial_duration_s,
            data_dir=args.data_dir,
        )
        electrode_names = dataset.electrode_names
    else:
        dataset = EEGDataset(
            data_dir=args.data_dir,
            sample_rate=config.sample_rate,
            trial_duration_s=config.trial_duration_s,
        )
        electrode_names = dataset.electrode_names

    if electrode_names is None:
        raise ValueError("Could not determine electrode names")

    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),  # Same split across ranks
    )
    print_main(f"Train: {n_train}, Val: {n_val}")
    print_main(f"Effective batch size: {args.batch_size} × {world_size} = {args.batch_size * world_size}")

    # Samplers
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Model
    n_subjects = dataset.n_subjects
    print_main(f"Building BrainWM v2 (n_subjects={n_subjects})...")
    model = BrainWM(config, n_subjects=n_subjects)
    model.initialize_electrodes(electrode_names)
    model = model.to(device)

    # Keep a reference to the raw model (before DDP wrap)
    raw_model = model

    if distributed:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                    find_unused_parameters=True)

    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print_main(f"Trainable parameters: {n_params:,}")
    print_main(f"State resolution: {config.state_duration_ms}ms → {config.n_states_per_trial} states/trial")

    # Optimizer (scale LR by world_size for linear scaling rule)
    scaled_lr = config.learning_rate * world_size
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=scaled_lr,
        weight_decay=config.weight_decay, betas=(0.9, 0.98),
    )
    total_steps = len(train_loader) * config.n_epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr=config.min_learning_rate,
    )

    # Output directory (only rank 0 saves)
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            cfg_dict = {k: str(v) for k, v in vars(config).items()}
            cfg_dict["world_size"] = world_size
            cfg_dict["effective_batch_size"] = args.batch_size * world_size
            json.dump(cfg_dict, f, indent=2)

    # Training loop
    print_main(f"\nTraining for {config.n_epochs} epochs on {world_size} GPU(s)...")
    best_val_loss = float("inf")
    history = []
    global_step = 0

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()

        # Set epoch for distributed sampler (ensures different shuffle per epoch)
        if distributed:
            train_sampler.set_epoch(epoch)

        train_losses, global_step = train_one_epoch(
            model, raw_model, train_loader, optimizer, scheduler, config,
            device, epoch, global_step, total_steps,
        )
        val_losses = validate(model, raw_model, val_loader, device)
        elapsed = time.time() - t0

        print_main(
            f"Epoch {epoch}/{config.n_epochs} ({elapsed:.1f}s) | "
            f"Train={train_losses['total']:.4f} Val={val_losses['total']:.4f}"
        )

        # Only rank 0 saves
        if is_main_process():
            history.append({"epoch": epoch, "train": train_losses, "val": val_losses})

            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "electrode_names": electrode_names,
                    "config": {k: str(v) for k, v in vars(config).items()},
                }, output_dir / "best_model.pt")
                print_main(f"  → Best model saved (val={best_val_loss:.4f})")

            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "electrode_names": electrode_names,
                }, output_dir / f"checkpoint_ep{epoch}.pt")

    # Save history
    if is_main_process():
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print_main(f"\nDone. Best val loss: {best_val_loss:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
