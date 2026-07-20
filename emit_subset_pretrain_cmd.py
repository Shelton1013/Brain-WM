"""Reconstruct the exact pretraining command from a checkpoint's saved args,
overriding output_dir + injecting --pretrain_n_subjects. So a data-efficiency
subset run matches the main model in EVERY hyperparameter except data amount.

    python emit_subset_pretrain_cmd.py \
        --checkpoint /home/pxieaf/home2/model/main_time_r30_nojepa/checkpoint_ep4.pt \
        --n_subjects 2100 \
        --output_dir /home/pxieaf/home2/model/main_time_r30_nojepa_sub2100 \
        --nproc 8
"""
import argparse
import torch

# Keys added at save time (not CLI flags) or overridden here — never emit.
# epochs/warmup_epochs handled explicitly (scaled), so exclude from generic loop.
BLACKLIST = {
    "n_dataset", "n_subjects_actual", "local_rank", "rank", "world_size",
    "output_dir", "pretrain_n_subjects", "resume", "resume_from",
    "epochs", "warmup_epochs",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_subjects", type=int, required=True,
                   help="how many subjects to keep (~2100 ≈ 2500 h)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--epoch_mode", choices=["matched_steps", "same_epochs"],
                   default="matched_steps",
                   help="matched_steps: scale epochs ×(full/subset) so the subset "
                        "sees the SAME number of gradient updates (isolates DATA "
                        "amount — the fair data-efficiency test). same_epochs: keep "
                        "main's epoch count (less data AND less compute; risks "
                        "undertraining, confounds the claim).")
    a = p.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = ck["args"]
    d = vars(args) if not isinstance(args, dict) else args

    full_subj = int(d.get("n_subjects_actual") or 0)
    main_epochs = int(d.get("epochs") or 15)
    main_warmup = int(d.get("warmup_epochs") or 1)
    scale = max(1, round(full_subj / a.n_subjects)) if full_subj else 1
    if a.epoch_mode == "matched_steps":
        new_epochs = main_epochs * scale
        new_warmup = main_warmup * scale
    else:
        new_epochs, new_warmup = main_epochs, main_warmup

    parts = [f"torchrun --nproc_per_node={a.nproc} train_v2.py"]
    for k in sorted(d):
        if k in BLACKLIST:
            continue
        v = d[k]
        if v is None:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)               # store_true
        elif isinstance(v, (list, tuple)):
            if len(v):
                parts.append(flag + " " + " ".join(str(x) for x in v))
        else:
            parts.append(f"{flag} {v}")
    parts.append(f"--epochs {new_epochs}")
    parts.append(f"--warmup_epochs {new_warmup}")
    parts.append(f"--output_dir {a.output_dir}")
    parts.append(f"--pretrain_n_subjects {a.n_subjects}")

    dur = d.get("trial_duration_s", 10)
    full_h = (d.get("n_dataset") or 0) * dur / 3600
    sub_h = a.n_subjects * (d.get("n_dataset") or 0) / max(full_subj, 1) * dur / 3600
    print("\n# ---- exact subset-pretrain command (matches main model except data) ----")
    print(" \\\n  ".join(parts))
    print()
    print(f"# main model : {full_subj:,} subjects / {d.get('n_dataset'):,} trials (~{full_h:,.0f} h), "
          f"epochs={main_epochs} warmup={main_warmup}")
    print(f"# subset run : {a.n_subjects:,} subjects (~{sub_h:,.0f} h expected), "
          f"scale=×{scale}, epochs={new_epochs} warmup={new_warmup}  [{a.epoch_mode}]")
    print(f"# → subset sees ~{'SAME' if a.epoch_mode=='matched_steps' else '1/%d'%scale} "
          f"gradient updates as main; compare FINAL checkpoints.")


if __name__ == "__main__":
    main()
