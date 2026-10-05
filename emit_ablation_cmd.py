"""Emit the exact pretraining command of a reference run with ablation overrides.

Reads the reference run's args.json (e.g. the 2,388 h primary model) so an
ablation differs from it ONLY in the overridden flags — same data subset, seed,
effective batch, LR schedule and step budget.

    python emit_ablation_cmd.py \
        --ref_args /home/pxieaf/home2/model/sub2100_2500h_matched/args.json \
        --output_dir /home/pxieaf/home2/model/abl_rawrecon_sub2100 \
        --nproc 4 --batch_size 128 --stop_after_epoch 16 \
        --set target_type=raw_recon
"""
import argparse
import json

# Saved-at-runtime keys that are not CLI flags, or that we set explicitly.
BLACKLIST = {
    "n_dataset", "n_subjects_actual", "local_rank", "rank", "world_size",
    "output_dir", "resume", "resume_from", "n_params", "effective_batch_size",
    "model", "batch_size", "stop_after_epoch",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_args", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=None,
                   help="per-GPU batch; default keeps the reference effective batch")
    p.add_argument("--master_port", type=int, default=29500)
    p.add_argument("--stop_after_epoch", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override flags; VALUE true/false for store_true flags")
    a = p.parse_args()

    d = json.load(open(a.ref_args))
    for kv in a.set:
        k, v = kv.split("=", 1)
        d[k] = {"true": True, "false": False}.get(v.lower(), v)

    eff = int(d["effective_batch_size"])
    bs = a.batch_size or eff // a.nproc
    assert bs * a.nproc == eff, f"{bs}x{a.nproc} != reference effective batch {eff}"

    parts = [f"torchrun --nproc_per_node={a.nproc} --master_port={a.master_port} "
             f"train_v2.py"]
    for k in sorted(d):
        v = d[k]
        if k in BLACKLIST or v is None:
            continue
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            if v:
                parts.append(f"--{k} " + " ".join(str(x) for x in v))
        else:
            parts.append(f"--{k} {v}")
    parts += [f"--batch_size {bs}", f"--output_dir {a.output_dir}"]
    if a.stop_after_epoch:
        parts.append(f"--stop_after_epoch {a.stop_after_epoch}")
    print(" ".join(parts))


if __name__ == "__main__":
    main()
