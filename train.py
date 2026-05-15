"""BrainWM v2 training script."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from config import BrainWMConfig
from model import BrainWM
from dataset import EEGDataset, PhysioNetMIDataset


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler, config, device, epoch,
                    total_epochs, global_step, total_steps):
    model.train()
    total_losses = {"total": 0, "adv": 0}
    for k in config.prediction_horizons:
        total_losses[f"pred_k{k}"] = 0
    n_batches = 0

    for batch_idx, (eeg, subject_ids) in enumerate(dataloader):
        eeg = eeg.to(device)
        subject_ids = subject_ids.to(device)

        # Update training progress for EMA decay + scheduled sampling + adversary ramp
        progress = global_step / max(total_steps, 1)
        model.set_training_progress(progress)

        # Forward
        outputs = model(eeg, return_predictions=True)
        losses = model.compute_loss(outputs, subject_ids=subject_ids)

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()

        # Update EMA
        model.update_ema()

        for k in total_losses:
            if k in losses:
                total_losses[k] += losses[k].item()
        n_batches += 1
        global_step += 1

        if batch_idx % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            ema_m = model.ema_encoder.current_decay if model.ema_encoder else 0
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"loss={losses['total']:.4f} "
                f"k1={losses.get('pred_k1', 0):.4f} "
                f"k2={losses.get('pred_k2', 0):.4f} "
                f"k3={losses.get('pred_k3', 0):.4f} "
                f"adv={losses.get('adv', 0):.4f} "
                f"lr={lr:.2e} ema={ema_m:.4f} α={model.adv_alpha:.2f}"
            )

    avg = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    return avg, global_step


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_losses = {"total": 0}
    n_batches = 0

    for eeg, subject_ids in dataloader:
        eeg = eeg.to(device)
        subject_ids = subject_ids.to(device)
        outputs = model(eeg, return_predictions=True)
        losses = model.compute_loss(outputs, subject_ids=subject_ids)
        total_losses["total"] += losses["total"].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


def main():
    parser = argparse.ArgumentParser(description="Train BrainWM v2")
    parser.add_argument("--data_dir", type=str, default="./data/physionet")
    parser.add_argument("--dataset", type=str, default="physionet",
                        choices=["physionet", "preprocessed"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_subjects", type=int, default=109)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    config = BrainWMConfig()
    config.n_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr

    # Dataset
    print("Loading dataset...")
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
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    # Model
    n_subjects = dataset.n_subjects
    print(f"Building BrainWM v2 (n_subjects={n_subjects})...")
    model = BrainWM(config, n_subjects=n_subjects)
    model.initialize_electrodes(electrode_names)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"State resolution: {config.state_duration_ms}ms → {config.n_states_per_trial} states/trial")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, betas=(0.9, 0.98),
    )
    total_steps = len(train_loader) * config.n_epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr=config.min_learning_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(config).items()}, f, indent=2)

    # Training
    print(f"\nTraining for {config.n_epochs} epochs...")
    best_val_loss = float("inf")
    history = []
    global_step = 0

    for epoch in range(1, config.n_epochs + 1):
        t0 = time.time()
        train_losses, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device,
            epoch, config.n_epochs, global_step, total_steps,
        )
        val_losses = validate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{config.n_epochs} ({elapsed:.1f}s) | "
            f"Train={train_losses['total']:.4f} Val={val_losses['total']:.4f}"
        )

        history.append({"epoch": epoch, "train": train_losses, "val": val_losses})

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "electrode_names": electrode_names,
                "config": {k: str(v) for k, v in vars(config).items()},
            }, output_dir / "best_model.pt")
            print(f"  → Best model saved (val={best_val_loss:.4f})")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "electrode_names": electrode_names,
            }, output_dir / f"checkpoint_ep{epoch}.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
