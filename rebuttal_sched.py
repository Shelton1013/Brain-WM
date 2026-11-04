"""Priority GPU scheduler for the AAAI rebuttal experiments.

    nohup python rebuttal_sched.py > ~/home2/logs_2/rebuttal/sched.out 2>&1 &

Jobs run in priority order on a private GPU pool (0-1 are left to other users).
A job starts when its dependencies are met and enough pool GPUs are free; a
ready job that does not fit blocks lower-priority jobs (no starvation of the
4-GPU pretraining runs), except jobs marked short (frozen probes, < ~1 h),
which may backfill. A job is skipped if its output already exists, and a job
whose process is still running from a previous scheduler instance is adopted
(its GPUs stay reserved), so the scheduler can be restarted at any time.
The GPU pool is re-read every minute from LOG/pool (e.g. "0 1 2 3 4 5 6 7 8 9").
"""
import os
import subprocess
import sys
import time
from pathlib import Path

POOL = [2, 3, 4, 5, 6, 7, 8, 9]            # default if LOG/pool is absent
REPO = Path(__file__).resolve().parent
M = Path("/home/pxieaf/home2/model")
LOG = Path("/home/pxieaf/home2/logs_2/rebuttal")
CACHE = "/home/pxieaf/home2/dataset_cache"
REF = M / "sub2100_2500h_matched/args.json"
FULL = M / "main_time_r30_nojepa"
DATA = {"isruc": "/home/pxieaf/home2/datasets/isruc/subgroupI_official",
        "hmc": "/home/pxieaf/home2/datasets/HMC"}
EP = 16
PY = sys.executable


class Job:
    def __init__(self, name, n_gpu, cmd, out, deps=(), short=False, key=None, light=False):
        self.name, self.n_gpu, self.cmd, self.out = name, n_gpu, cmd, Path(out)
        self.deps, self.short = list(deps), short
        self.light = light      # CPU-bound, <1 GB GPU: does not reserve its GPU
        self.proc, self.gpus, self.state = None, [], "wait"   # wait|run|done|fail
        self.pid = None                   # adopted process from a previous run
        # unique string identifying this job's process on the command line
        self.key = key or (f"--output_dir {self.out.parent}" if n_gpu > 1
                           else f"--output {self.out}")

    def poll(self):
        """None while running, else return code (adopted jobs: 0 = exited)."""
        if self.proc is not None:
            return self.proc.poll()
        return None if Path(f"/proc/{self.pid}").exists() else 0


def pretrain(name, port, overrides):
    out = M / name
    cmd = (f"{PY} emit_ablation_cmd.py --ref_args {REF} --output_dir {out} "
           f"--nproc 4 --batch_size 128 --master_port {port} "
           f"--stop_after_epoch {EP} --set {' '.join(overrides)}")
    # emit the torchrun command, then run it under THIS interpreter (a bare
    # `torchrun` on PATH can resolve to another env without mne)
    run = (f'CMD=$({cmd}) && echo "$CMD" && '
           f'eval "${{CMD/torchrun/{PY} -m torch.distributed.run}} --resume_from auto"')
    # --resume_from auto: continue from the latest checkpoint_ep*.pt if one exists
    # (e.g. filterbank raw recon s44 was stopped at ep14 on 10-01)
    return Job(f"pre_{name}", 4, run, out / f"checkpoint_ep{EP}.pt")


def sleep_eval(ckpt, ds, mode, deps=()):
    ckpt = Path(ckpt)
    out = ckpt.with_name(f"{ckpt.stem}_{ds}_{mode}.json")
    extra = "--freeze_encoder" if mode == "frozen" else "--batch_size 8"
    cmd = (f"{PY} eval_sleep_seq2seq.py --dataset {ds} --checkpoint {ckpt} "
           f"--data_dir {DATA[ds]} --cache_dir {CACHE} --n_reps 3 {extra} "
           f"--output {out}")
    return Job(f"eval_{ckpt.parent.name}_{ckpt.stem}_{ds}_{mode}", 1, cmd, out,
               deps=deps, short=(mode == "frozen"))


def le_eval(ckpt, ds, deps=()):
    """Frozen label-efficiency curve (1-100% of train sequences, 5 reps)."""
    ckpt = Path(ckpt)
    out = ckpt.with_name(f"{ckpt.stem}_le_{ds}.json")
    cmd = (f"{PY} eval_label_efficiency_seq2seq.py --dataset {ds} --checkpoint {ckpt} "
           f"--data_dir {DATA[ds]} --cache_dir {CACHE} --output {out}")
    return Job(f"le_{ckpt.parent.name}_{ckpt.stem}_{ds}", 1, cmd, out,
               deps=deps, short=True)


def le_for(ckpt, deps=()):
    return [le_eval(ckpt, ds, deps) for ds in ("isruc", "hmc")]


def ma_eval(ckpt, seed, deps=()):
    """Mental Arithmetic, CSBrain split, same protocol as run_evals.sh."""
    ckpt = Path(ckpt)
    out = ckpt.with_name(f"{ckpt.stem}_mentalarith_seed{seed}.json")
    cmd = (f"{PY} eval_mental_arithmetic.py --mode both --checkpoint {ckpt} "
           f"--data_dir /home/pxieaf/home2/datasets/mental_arithmetic/"
           f"eeg-during-mental-arithmetic-tasks-1.0.0 --cache_dir {CACHE} "
           f"--sample_rate 256 --trial_duration_s 5 --normalization per_recording_robust "
           f"--split_mode csbrain --frozen_reps 5 --ft_protocol labram "
           f"--ft_layer_decay 0.65 --ft_warmup_epochs 5 --max_epochs 50 "
           f"--ft_patience 50 --ft_select ba --seed {seed} --output {out}")
    return Job(f"ma_{ckpt.parent.name}_s{seed}", 1, cmd, out, deps=deps)


def mumtaz_eval(ckpt, seed, deps=()):
    """Mumtaz2016, CBraMod split, same protocol as run_evals.sh."""
    ckpt = Path(ckpt)
    out = ckpt.with_name(f"{ckpt.stem}_mumtaz_seed{seed}.json")
    cmd = (f"{PY} eval_mumtaz.py --mode both --checkpoint {ckpt} "
           f"--mumtaz_dir /home/pxieaf/home2/datasets/mumtaz2016_full "
           f"--cache_dir {CACHE} --sample_rate 256 --trial_duration_s 10 "
           f"--normalization per_recording_robust --split_mode cbramod "
           f"--frozen_reps 5 --ft_protocol onecycle --max_epochs 50 "
           f"--seed {seed} --output {out}")
    return Job(f"mumtaz_{ckpt.parent.name}_s{seed}", 1, cmd, out, deps=deps)


FACED_PROC = Path("/home/pxieaf/home2/datasets/FACED_proc")
FACED_LMDB = Path("/home/pxieaf/home2/datasets/FACED_cbramod_lmdb")
CBRAMOD = Path("/home/pxieaf/home2/baselines/CBraMod")
CBRAMOD_PY = "/home/pxieaf/home2/envs/cbramod/bin/python"

# CBraMod's own FACED LMDB, built from the same rebuilt arrays (CPU job)
faced_lmdb = Job("faced_cbramod_lmdb", 0,
                 f"rm -rf {FACED_LMDB} && {CBRAMOD_PY} build_faced_lmdb_cbramod.py "
                 f"--proc_dir {FACED_PROC} --out {FACED_LMDB} && touch {FACED_LMDB}/.done",
                 FACED_LMDB / ".done")


def faced_eval(ckpt, seed, mode="both", random=False, deps=()):
    ckpt = Path(ckpt)
    out = ckpt.with_name(f"{ckpt.stem}_faced_{mode}_seed{seed}.json")
    cmd = (f"{PY} eval_faced.py --checkpoint {ckpt} --proc_dir {FACED_PROC} "
           f"--mode {mode} --seed {seed} --output {out}"
           + (" --include_random_baseline" if random else ""))
    return Job(f"faced_{ckpt.parent.name}_{mode}_s{seed}", 1, cmd, out, deps=deps,
               short=(mode == "frozen"))


def cbramod_faced(seed):
    """Unmodified CBraMod finetune_main.py with its FACED defaults."""
    out_dir = M / "baseline_cbramod_faced" / f"seed{seed}"
    cmd = (f"mkdir -p {out_dir} && cd {CBRAMOD} && {CBRAMOD_PY} finetune_main.py "
           f"--downstream_dataset FACED --datasets_dir {FACED_LMDB} "
           f"--num_of_classes 9 --model_dir {out_dir} --cuda 0 --seed {seed} "
           f"--num_workers 4 && touch {out_dir}/.done")
    return Job(f"cbramod_faced_s{seed}", 1, cmd, out_dir / ".done",
               deps=[faced_lmdb.name])


def mumtaz_v2(ckpt, seed):
    """Mumtaz aligned protocol, fixed before running and applied to every model:
    0.5-45 Hz band-pass (= pretraining band), 5-s windows (= CBraMod/CSBrain),
    LaBraM-style FT identical to Mental Arithmetic."""
    ckpt = Path(ckpt)
    out = ckpt.with_name(f"{ckpt.stem}_mumtazv2_seed{seed}.json")
    cmd = (f"{PY} eval_mumtaz.py --mode both --checkpoint {ckpt} "
           f"--mumtaz_dir /home/pxieaf/home2/datasets/mumtaz2016_full "
           f"--cache_dir {CACHE} --sample_rate 256 --trial_duration_s 5 "
           f"--normalization per_recording_robust --split_mode cbramod "
           f"--bandpass 0.5 45 --frozen_reps 5 --ft_protocol labram "
           f"--ft_layer_decay 0.65 --ft_warmup_epochs 5 --max_epochs 50 "
           f"--ft_patience 50 --seed {seed} --output {out}")
    return Job(f"mumtazv2_{ckpt.parent.name}_s{seed}", 1, cmd, out)


HC_DIR = Path("/home/pxieaf/home2/eval_results/handcrafted")


def handcrafted(name, dataset, data_dir, extra):
    """Welch band-power features + the same LogReg probe (CPU)."""
    out = HC_DIR / f"handcrafted_{name}.json"
    cmd = (f"mkdir -p {HC_DIR} && {PY} eval_handcrafted_spectral.py --dataset {dataset} "
           f"--data_dir {data_dir} --cache_dir {CACHE} {extra} --output {out}")
    return Job(f"hc_{name}", 0, cmd, out, short=True)   # CPU: never blocks GPU jobs


def noise_eval(ckpt, ds, deps=(), weak=False):
    """Frozen-feature robustness to EMG / mains / drift / white noise."""
    ckpt = Path(ckpt)
    tag = "noiseweak" if weak else "noise"
    out = ckpt.with_name(f"{ckpt.stem}_{tag}_{ds}.json")
    cmd = (f"{PY} eval_noise_robustness.py --dataset {ds} --checkpoint {ckpt} "
           f"--data_dir {DATA[ds]} --cache_dir {CACHE} --output {out}"
           + (" --weak" if weak else ""))
    return Job(f"{tag}_{ckpt.parent.name}_{ckpt.stem}_{ds}", 1, cmd, out,
               deps=deps, short=True)


def fewlabel_ft(ckpt, ds, n_seq, random_only=False, deps=()):
    """End-to-end FT on a fixed (seed-42) subset of n_seq training sequences."""
    ckpt = Path(ckpt)
    tag = "randomFT" if random_only else "ft"
    out = ckpt.with_name(f"{ckpt.stem}_{ds}_{tag}_n{n_seq}.json")
    cmd = (f"{PY} eval_sleep_seq2seq.py --dataset {ds} --checkpoint {ckpt} "
           f"--data_dir {DATA[ds]} --cache_dir {CACHE} --n_reps 3 --batch_size 8 "
           f"--max_train_seqs {n_seq} --output {out}"
           + (" --random_only" if random_only else ""))
    return Job(f"few_{ckpt.parent.name}_{ckpt.stem}_{ds}_{tag}_n{n_seq}", 1, cmd, out,
               deps=deps)


def evals_for(pre_job):
    ck = pre_job.out
    return [sleep_eval(ck, ds, mode, deps=[pre_job.name])
            for mode in ("ft", "frozen") for ds in ("isruc", "hmc")]


pre_raw = pretrain("abl_rawrecon_sub2100", 29511, ["target_type=raw_recon"])
pre_fft = pretrain("abl_fftband_sub2100", 29512, ["target_type=fft_bandpower"])
pre_loc = pretrain("abl_patchlocal_sub2100", 29513, ["patch_local_filter=true"])
pre_bnd = pretrain("abl_bandrecon_sub2100", 29514, ["target_type=band_recon"])
# our target re-run under the ablations' exact launcher (4x128, current code):
# the primary was trained 8x64, and SIGReg is computed on the per-GPU batch
pre_ours = pretrain("abl_bandpower_rerun_sub2100", 29515, ["target_type=bandpower"])
# pretraining-seed replicates for the key comparison (ours vs raw waveform):
# same 2,100-subject subset (subset_seed 42), new init / data order
PRE_SEEDS = [pretrain(f"abl_{tag}_s{s}_sub2100", port,
                      [f"target_type={tt}", f"seed={s}", "subset_seed=42"])
             for (tag, tt), s, port in [(("bandpower", "bandpower"), 43, 29521),
                                        (("rawrecon", "raw_recon"), 43, 29522),
                                        (("bandpower", "bandpower"), 44, 29523),
                                        (("rawrecon", "raw_recon"), 44, 29524)]]
# option B (10-01): main model = linear tokenizer -> its pretraining seeds 43/44
# (band power + matched raw-waveform comparator); same subset, new init/order
PRE_LIN_SEEDS = [pretrain(f"abl_linear_{tag}_s{s}_sub2100", port,
                          ["tokenizer_type=linear", f"target_type={tt}", f"seed={s}",
                           "subset_seed=42"])
                 for (tag, tt), s, port in [(("bandpower", "bandpower"), 43, 29541),
                                            (("rawrecon", "raw_recon"), 43, 29542),
                                            (("bandpower", "bandpower"), 44, 29543),
                                            (("rawrecon", "raw_recon"), 44, 29544)]]
# 2x2 factorial {filterbank, linear} tokenizer x {band power, raw waveform}:
# the linear-tokenizer cells (seed 42, same subset/launcher as every ablation)
PRE_LINEAR = [pretrain(f"abl_linear_{tag}_sub2100", port,
                       ["tokenizer_type=linear", f"target_type={tt}"])
              for tag, tt, port in [("bandpower", "bandpower", 29531),
                                    ("rawrecon", "raw_recon", 29532)]]

PRIMARY = M / "sub2100_2500h_matched/checkpoint_ep16.pt"

# --- paper-direction experiments: robustness, label efficiency (FT), compute ---
_OURS = M / "abl_bandpower_rerun_sub2100"            # matched launcher (4x128)
_RAW = M / "abl_rawrecon_sub2100"
_CKPTS = [(PRIMARY, []), (_OURS / "checkpoint_ep16.pt", []),
          (_RAW / "checkpoint_ep16.pt", []),
          (M / "abl_bandrecon_sub2100/checkpoint_ep16.pt", []),
          *[(p.out, [p.name]) for p in PRE_LINEAR]]
_FEW = {"isruc": (178, 356), "hmc": (226, 451)}       # 5% / 10% of train sequences
DIRECTION_JOBS = [
    # (1) noise robustness of frozen features, every target incl. the 2x2 cells
    *[noise_eval(ck, ds, deps) for ck, deps in _CKPTS for ds in ("isruc", "hmc")],
    *[noise_eval(ck, ds, deps, weak=True) for ck, deps in _CKPTS for ds in ("isruc", "hmc")],
    # (2) compute efficiency: frozen probe along pretraining (ours vs raw waveform)
    *[sleep_eval(d / f"checkpoint_ep{e}.pt", ds, "frozen")
      for e in (2, 4, 8) for d in (_OURS, _RAW) for ds in ("isruc", "hmc")],
    # (3) few-label end-to-end FT: ours vs raw waveform (+ 2x2 cells) + from scratch
    *[fewlabel_ft(ck, ds, n, deps=deps) for ds, ns in _FEW.items() for n in ns
      for ck, deps in [(_OURS / "checkpoint_ep16.pt", []), (_RAW / "checkpoint_ep16.pt", []),
                       *[(p.out, [p.name]) for p in PRE_LINEAR]]],
    *[fewlabel_ft(_OURS / "checkpoint_ep16.pt", ds, n, random_only=True)
      for ds, ns in _FEW.items() for n in ns],
    # (2b) compute efficiency under FT at an early checkpoint
    *[sleep_eval(d / "checkpoint_ep4.pt", ds, "ft") for d in (_OURS, _RAW)
      for ds in ("isruc", "hmc")],
]
FACED_JOBS = [
    faced_lmdb,
    # ours: FT x3 seeds (seed 0 also runs the random-init/from-scratch baseline)
    faced_eval(PRIMARY, 0, random=True), faced_eval(PRIMARY, 1), faced_eval(PRIMARY, 2),
    # CBraMod baseline on the identical rebuilt data, its own code + defaults
    *[cbramod_faced(s) for s in (3407, 3408, 3409)],
    # target study on FACED (frozen: seconds-to-minutes; FT for the waveform target)
    *[faced_eval(M / d / "checkpoint_ep16.pt", 0, mode="frozen")
      for d in ("abl_rawrecon_sub2100", "abl_bandrecon_sub2100",
                "abl_fftband_sub2100", "abl_patchlocal_sub2100")],
    *[faced_eval(M / "abl_rawrecon_sub2100/checkpoint_ep16.pt", s) for s in (0, 1, 2)],
]

MA_DIR = "/home/pxieaf/home2/datasets/mental_arithmetic/eeg-during-mental-arithmetic-tasks-1.0.0"
MUMTAZ_DIR = "/home/pxieaf/home2/datasets/mumtaz2016_full"
CLINICAL_JOBS = [
    # handcrafted band power: MA (CSBrain split, 5 s) and Mumtaz (old + aligned protocol)
    handcrafted("mentalarith_csbrain_5s", "mental_arith", MA_DIR, "--trial_duration_s 5"),
    handcrafted("mumtaz_cbramod_10s_raw", "mumtaz_cbramod", MUMTAZ_DIR, "--trial_duration_s 10"),
    handcrafted("mumtaz_cbramod_5s_bp", "mumtaz_cbramod", MUMTAZ_DIR,
                "--trial_duration_s 5 --bandpass 0.5 45"),
    # Mumtaz aligned protocol: ours + both reconstruction targets, 5 seeds each
    *[mumtaz_v2(M / d / "checkpoint_ep16.pt", s) for s in range(5)
      for d in ("sub2100_2500h_matched", "abl_rawrecon_sub2100", "abl_bandrecon_sub2100")],
]

_LP = {"ours_orig_ep16": PRIMARY,
       **{f"ours_ep{e}": M / f"abl_bandpower_rerun_sub2100/checkpoint_ep{e}.pt"
          for e in (2, 4, 8, 16)},
       **{f"raw_ep{e}": M / f"abl_rawrecon_sub2100/checkpoint_ep{e}.pt"
          for e in (2, 4, 8, 16)},
       "bandrecon_ep16": M / "abl_bandrecon_sub2100/checkpoint_ep16.pt",
       "fft_ep16": M / "abl_fftband_sub2100/checkpoint_ep16.pt",
       "patchlocal_ep16": M / "abl_patchlocal_sub2100/checkpoint_ep16.pt",
       "linear_bandpower_ep16": M / "abl_linear_bandpower_sub2100/checkpoint_ep16.pt"}
LP_DIR = Path("/home/pxieaf/home2/eval_results/linear_probe")


def linear_probe(ds):
    """Strict linear probe (per-epoch LogReg), many checkpoints, one data load."""
    out = LP_DIR / f"lp_{ds}.json"                 # written incrementally...
    done = LP_DIR / f"lp_{ds}.done"                 # ...so completion = marker
    ck = " ".join(f"{k}={v}" for k, v in _LP.items())
    cmd = (f"mkdir -p {LP_DIR} && {PY} eval_linear_probe_sleep.py --dataset {ds} "
           f"--data_dir {DATA[ds]} --cache_dir {CACHE} --checkpoints {ck} "
           f"--random_init --output {out} && touch {done}")
    return Job(f"linprobe_{ds}", 1, cmd, done, short=True, key=f"--output {out}")


HC_SLEEP_DIR = Path("/home/pxieaf/home2/eval_results/handcrafted_sleep")


def handcrafted_sleep(ds, win):
    """Strong direct-feature baselines (basic/rich x seqhead/strong/gbdt, label
    efficiency; 10-s run also concatenates our frozen features)."""
    out = HC_SLEEP_DIR / f"hc_{ds}_{win}s_v2.json"      # v2: NaN-sanitised features
    # cap threads: 4 of these at once with default OpenMP (48 threads each)
    # oversubscribe the CPU and stall sklearn's gradient boosting
    cmd = (f"mkdir -p {HC_SLEEP_DIR} && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "
           f"{PY} eval_handcrafted_sleep.py --dataset {ds} "
           f"--data_dir {DATA[ds]} --cache_dir {CACHE} --trial_duration_s {win} "
           f"--output {out}"
           + (f" --concat_ckpt {M}/abl_bandpower_rerun_sub2100/checkpoint_ep16.pt"
              if win == 10 else ""))
    return Job(f"hcsleep_{ds}_{win}s", 1, cmd, Path(str(out) + ".done"), short=True,
               key=f"--output {out}", light=True)


def linear_probe_2x2(ds):
    """Strict linear probe for the last 2x2 cell (separate file: the first
    probe run may still be writing lp_<ds>.json)."""
    out = LP_DIR / f"lp_{ds}_2x2.json"
    ck = (f"linear_rawrecon_ep16={M}/abl_linear_rawrecon_sub2100/checkpoint_ep16.pt "
          f"linear_bandpower_ep16={M}/abl_linear_bandpower_sub2100/checkpoint_ep16.pt")
    cmd = (f"mkdir -p {LP_DIR} && {PY} eval_linear_probe_sleep.py --dataset {ds} "
           f"--data_dir {DATA[ds]} --cache_dir {CACHE} --checkpoints {ck} "
           f"--output {out} && touch {out}.done")
    return Job(f"linprobe2x2_{ds}", 1, cmd, Path(f"{out}.done"), short=True,
               key=f"--output {out}", deps=[PRE_LINEAR[1].name])


LP2_DIR = Path("/home/pxieaf/home2/eval_results/linear_probe_v2")
_CK = lambda d, e=16: f"{M}/{d}/checkpoint_ep{e}.pt"
LP2_GROUPS = {
    "A": {"ours_ep16": _CK("abl_bandpower_rerun_sub2100"), "raw_ep16": _CK("abl_rawrecon_sub2100"),
          "linear_bandpower_ep16": _CK("abl_linear_bandpower_sub2100"),
          "linear_rawrecon_ep16": _CK("abl_linear_rawrecon_sub2100")},
    "B": {"ours_orig_ep16": str(PRIMARY), "bandrecon_ep16": _CK("abl_bandrecon_sub2100"),
          "fft_ep16": _CK("abl_fftband_sub2100"), "patchlocal_ep16": _CK("abl_patchlocal_sub2100"),
          "freqmask_ep16": _CK("abl_freqmask_sub2100")},
    "C": {f"{t}_ep{e}": _CK(d, e) for e in (4, 8)
          for t, d in (("ours", "abl_bandpower_rerun_sub2100"), ("raw", "abl_rawrecon_sub2100"))},
    "S43": {"ours_s43_ep16": _CK("abl_bandpower_s43_sub2100"),
            "raw_s43_ep16": _CK("abl_rawrecon_s43_sub2100")},
    "S44": {"ours_s44_ep16": _CK("abl_bandpower_s44_sub2100")},
    # linear tokenizer = main model from 10-01: its own pretraining seeds
    "L43": {"linbp_s43_ep16": _CK("abl_linear_bandpower_s43_sub2100"),
            "linraw_s43_ep16": _CK("abl_linear_rawrecon_s43_sub2100")},
    "L44": {"linbp_s44_ep16": _CK("abl_linear_bandpower_s44_sub2100"),
            "linraw_s44_ep16": _CK("abl_linear_rawrecon_s44_sub2100")},
}


def lp2(ds, model, group=None):
    """Linear probe v2 (avg3 + center10, label efficiency). Primary metric."""
    tag = group or model
    out = LP2_DIR / f"lp2_{ds}_{tag}.json"
    extra = ""
    deps = []
    if model == "specbram":
        extra = " --checkpoints " + " ".join(f"{k}={v}" for k, v in LP2_GROUPS[group].items())
        extra += " --random_init" if group == "A" else ""
        deps = {"S43": [p.name for p in PRE_SEEDS[:2]],
                "S44": [PRE_SEEDS[2].name],
                "L43": [p.name for p in PRE_LIN_SEEDS[:2]],
                "L44": [p.name for p in PRE_LIN_SEEDS[2:]]}.get(group, [])
    cmd = (f"mkdir -p {LP2_DIR} && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "
           f"{PY} eval_linear_probe_v2.py --dataset {ds} --data_dir {DATA[ds]} "
           f"--model {model}{extra} --output {out}")
    return Job(f"lp2_{ds}_{tag}", 1, cmd, Path(f"{out}.done"), deps=deps, short=True,
               key=f"--output {out}", light=True)


JOBS = [                                   # priority order
    *[lp2(ds, "specbram", g) for g in ("A",) for ds in ("isruc", "hmc")],
    *[lp2(ds, m) for m in ("cbramod", "labram", "biot", "biot_sleep")
      for ds in ("isruc", "hmc")],
    *[lp2(ds, "specbram", g) for g in ("B", "C", "S43", "S44") for ds in ("isruc", "hmc")],
    *PRE_LIN_SEEDS,
    *[lp2(ds, "specbram", g) for g in ("L43", "L44") for ds in ("isruc", "hmc")],
    linear_probe("isruc"), linear_probe("hmc"),
    linear_probe_2x2("isruc"), linear_probe_2x2("hmc"),
    *[handcrafted_sleep(ds, w) for w in (10, 30) for ds in ("isruc", "hmc")],
    # ep4/ep8 label-efficiency for every other ablation target too
    *[le_eval(M / d / f"checkpoint_ep{e}.pt", ds, deps=deps) for e in (4, 8)
      for d, deps in [("abl_bandrecon_sub2100", []), ("abl_fftband_sub2100", []),
                      ("abl_patchlocal_sub2100", []), ("abl_freqmask_sub2100", []),
                      ("abl_linear_bandpower_sub2100", []),
                      ("abl_linear_rawrecon_sub2100", [PRE_LINEAR[1].name])]
      for ds in ("isruc", "hmc")],
    # label-efficiency curve at earlier checkpoints too (no epoch selection:
    # every checkpoint is reported), ours vs raw waveform
    *[le_eval(M / d / f"checkpoint_ep{e}.pt", ds) for e in (4, 8)
      for d in ("abl_bandpower_rerun_sub2100", "abl_rawrecon_sub2100")
      for ds in ("isruc", "hmc")],
    # TOP PRIORITY: the matched-backbone target ablation on ISRUC/HMC
    pre_bnd, *evals_for(pre_bnd),
    pre_ours, *evals_for(pre_ours), *le_for(pre_ours.out, [pre_ours.name]),
    # 2x2 linear-tokenizer pair FIRST (user priority)
    *[j for p in PRE_LINEAR for j in (p, *evals_for(p), *le_for(p.out, [p.name]))],
    # paper-direction experiments (before pretraining seeds):
    *DIRECTION_JOBS,
    # pretraining-seed replicates: seed 43 pair -> seed 44 pair
    *[j for p in PRE_SEEDS for j in (p, *evals_for(p), *le_for(p.out, [p.name]))],
    *FACED_JOBS,
    *CLINICAL_JOBS,
    # early read-out: frozen probe of matched intermediate epochs (ep8)
    *[sleep_eval(M / d / "checkpoint_ep8.pt", ds, "frozen")
      for d in ("abl_rawrecon_sub2100", "abl_fftband_sub2100",
                "sub2100_2500h_matched") for ds in ("isruc", "hmc")],
    pre_raw, pre_fft,
    *evals_for(pre_raw), *evals_for(pre_fft),
    # target study: frozen label-efficiency curve per pretext target
    *le_for(pre_raw.out, [pre_raw.name]), *le_for(pre_fft.out, [pre_fft.name]),
    *le_for(pre_loc.out, [pre_loc.name]), *le_for(pre_bnd.out, [pre_bnd.name]),
    *le_for(M / "abl_freqmask_sub2100/checkpoint_ep16.pt"),
    *[sleep_eval(M / ck, ds, "frozen") for ds in ("isruc", "hmc")
      for ck in ("abl_maeonly_sub2100/checkpoint_ep14.pt",
                 "abl_freqmask_sub2100/checkpoint_ep16.pt")],
    sleep_eval(FULL / "checkpoint_ep5.pt", "isruc", "ft"),
    sleep_eval(FULL / "checkpoint_ep5.pt", "hmc", "ft"),
    pre_loc, *evals_for(pre_loc),
    sleep_eval(FULL / "checkpoint_ep5.pt", "isruc", "frozen"),
    sleep_eval(FULL / "checkpoint_ep5.pt", "hmc", "frozen"),
    # target study beyond sleep: reconstruction targets on MA / Mumtaz
    # (primary already has checkpoint_ep16 MA seeds 0-4 and Mumtaz seeds 0-2)
    *[ma_eval(p.out, s, [p.name]) for s in range(5) for p in (pre_raw, pre_bnd)],
    *[mumtaz_eval(p.out, s, [p.name]) for s in range(3) for p in (pre_raw, pre_bnd)],
    sleep_eval(FULL / "checkpoint_ep3.pt", "isruc", "ft"),
    sleep_eval(FULL / "checkpoint_ep3.pt", "hmc", "ft"),
]


# noise robustness deferred to follow-up work (user decision 09-29): drop the
# not-yet-finished noise jobs so GPUs go to the target ablation / label efficiency
JOBS = [j for j in JOBS if not j.name.startswith(("noise_", "noiseweak_"))]
# (10-02: filterbank raw recon s44 re-added below)
# 09-30 user decision: linear probe is the primary metric; every fine-tuning
# style job (full FT, few-label FT, Mumtaz/MA/FACED evals, CBraMod FACED FT)
# drops to the bottom of the queue. Pretraining + frozen/LP jobs keep order.
_is_ft = lambda n: (n.endswith("_ft") or "_ft_n" in n or "randomFT" in n
                    or n.startswith(("mumtazv2_", "ma_", "mumtaz_", "faced_",
                                     "cbramod_faced")))
JOBS = [j for j in JOBS if not _is_ft(j.name)] + [j for j in JOBS if _is_ft(j.name)]
def ft30(ckpt, ds, rep):
    """Full FT on the whole 30-s epoch (3 x 10-s windows averaged), one rep per
    job (seed 42+rep) so the three reps run in parallel; merged at read time."""
    ckpt = Path(ckpt)
    out = ckpt.with_name(f"{ckpt.stem}_{ds}_ft30s_rep{rep}.json")
    cmd = (f"{PY} eval_sleep_seq2seq.py --dataset {ds} --checkpoint {ckpt} "
           f"--data_dir {DATA[ds]} --cache_dir {CACHE} --n_reps 1 --rep_start {rep} "
           # micro_batch 2: 30-s epochs OOM at 8 seqs x 20 x 3 windows (46.8 GB);
           # gradient accumulation keeps the effective batch at 8
           f"--batch_size 8 --micro_batch 2 --trial_duration_s 30 --output {out}")
    # short=True: may use idle GPUs while a 4-GPU pretraining waits for its slot
    return Job(f"ft30_{ckpt.parent.name}_{ds}_r{rep}", 1, cmd, out, short=True)


# 30-s FT runs ahead of every other fine-tuning job (after pretraining / probes)
# user 10-01: one seed only for now (rep 0 = seed 42); reps 1-2 can be added later
_FT30 = [ft30(M / d / "checkpoint_ep16.pt", ds, r) for r in range(1) for ds in ("isruc", "hmc")
         # user 10-01: the linear tokenizer is the model going forward (matched
         # comparator = linear tokenizer + raw-waveform reconstruction)
         for d in ("abl_linear_bandpower_sub2100", "abl_linear_rawrecon_sub2100")]
# user 10-02: 30-s FT on pretraining seeds 43/44 too (rep 0 each) -> 3-seed error bars
_FT30 += [ft30(M / f"abl_linear_{t}_s{s}_sub2100" / "checkpoint_ep16.pt", ds, 0)
          for ds in ("hmc", "isruc") for s in (43, 44) for t in ("bandpower", "rawrecon")]
_k = next((i for i, j in enumerate(JOBS) if _is_ft(j.name)), len(JOBS))
JOBS = JOBS[:_k] + _FT30 + JOBS[_k:]

# user 10-02: 3 pretraining seeds for every remaining ablation cell.
# filterbank raw recon s44 first (completes the filterbank pair), then each
# other target at seeds 43/44. Frequency masking s42 was trained 8x64 with the
# old launcher -> re-run under this launcher (4x128) so its 3 seeds match.
_SEED_TARGETS = [("rawrecon", "raw", ["target_type=raw_recon"]),
                 ("fftband", "fft", ["target_type=fft_bandpower"]),
                 ("patchlocal", "patchlocal", ["patch_local_filter=true"]),
                 ("bandrecon", "bandrecon", ["target_type=band_recon"]),
                 ("freqmask", "freqmask", ["mask_axis=frequency"])]
_runs = [("rawrecon", "raw", ["target_type=raw_recon"], 44)]
_runs += [(t, k, o, s) for s in (43, 44) for t, k, o in _SEED_TARGETS[1:]]
_runs += [("freqmask", "freqmask_rerun", ["mask_axis=frequency"], 42)]
PRE_ABL_SEEDS = []
for i, (tag, key, ov, s) in enumerate(_runs):
    name = (f"abl_{tag}_rerun_sub2100" if key.endswith("_rerun")
            else f"abl_{tag}_s{s}_sub2100")
    p = pretrain(name, 29551 + i, ov + [f"seed={s}", "subset_seed=42"])
    lpkey = f"{key}_ep16" if key.endswith("_rerun") else f"{key}_s{s}_ep16"
    LP2_GROUPS[name] = {lpkey: _CK(name)}
    PRE_ABL_SEEDS.append(p)
_abl_lp = [j for p in PRE_ABL_SEEDS for ds in ("isruc", "hmc")
           for j in [lp2(ds, "specbram", p.name[4:])]]
for j in _abl_lp:                                   # lp2 deps only cover L/S groups
    j.deps = ["pre_" + j.name.split("_", 2)[2]]
# 10-s FT (secondary metric) at the very bottom
_abl_ft = [sleep_eval(p.out, ds, "ft", deps=[p.name])
           for p in PRE_ABL_SEEDS for ds in ("isruc", "hmc")]
for j in _abl_ft:      # GPUs 8/9 are idle while two 4-GPU pretrains run: let FT use them
    j.short = True
_k = next((i for i, j in enumerate(JOBS) if _is_ft(j.name)), len(JOBS))
JOBS = JOBS[:_k] + PRE_ABL_SEEDS + _abl_lp + JOBS[_k:] + _abl_ft
# user 10-05: SEED-VIG vigilance regression (CBraMod paper split). One LP job
# (deterministic ridge, every checkpoint + random init + CBraMod/LaBraM on one
# data load) and FT for the linear-tokenizer pair x 3 pretraining seeds.
VIG_DIR = Path("/home/pxieaf/home2/eval_results/seedvig")
_VIG_CK = {f"{k}_s{s}": _CK(d if s == 42 else
                            d.replace("_rerun", "").replace("_sub2100", f"_s{s}_sub2100"))
           for s in (42, 43, 44)
           for k, d in (("linbp", "abl_linear_bandpower_sub2100"),
                        ("linraw", "abl_linear_rawrecon_sub2100"),
                        ("fbbp", "abl_bandpower_rerun_sub2100"),
                        ("fbraw", "abl_rawrecon_sub2100"))}
_vig_lp_out = VIG_DIR / "lp_all.json"
_VIG = [Job("seedvig_lp", 1,
            f"mkdir -p {VIG_DIR} && {PY} eval_seedvig.py --mode lp --random_init "
            f"--checkpoints {' '.join(f'{k}={v}' for k, v in _VIG_CK.items())} "
            f"--baselines cbramod labram --output {_vig_lp_out} && touch {_vig_lp_out}.done",
            Path(f"{_vig_lp_out}.done"), short=True, key=f"--output {_vig_lp_out}",
            light=True)]   # frozen inference, few GB: may share a GPU with an FT job
for k in ("linbp", "linraw"):
    for s in (42, 43, 44):
        o = VIG_DIR / f"ft_{k}_s{s}.json"
        _VIG.append(Job(f"seedvig_ft_{k}_s{s}", 1,
                        f"mkdir -p {VIG_DIR} && {PY} eval_seedvig.py --mode ft --seed 0 "
                        f"--checkpoints {k}_s{s}={_VIG_CK[f'{k}_s{s}']} --output {o}",
                        o, short=True))
# 10-05 explanation check: is raw recon's SEED-VIG edge carried by blink / EOG
# energy (< ~2-4 Hz)? Same LP with the input high-passed at 2 or 4 Hz.
for lo in (2, 4):
    o = VIG_DIR / f"lp_all_hp{lo}.json"
    _VIG.append(Job(f"seedvig_lp_hp{lo}", 1,
                    f"mkdir -p {VIG_DIR} && {PY} eval_seedvig.py --mode lp --random_init "
                    f"--bandpass {lo} 45 "
                    f"--checkpoints {' '.join(f'{k}={v}' for k, v in _VIG_CK.items())} "
                    f"--output {o} && touch {o}.done",
                    Path(f"{o}.done"), short=True, key=f"--output {o}", light=True))
# 10-05: FT at the validation-selected 4 Hz high-pass (linear tok, both targets)
for k in ("linbp", "linraw"):
    for s in (42, 43, 44):
        o = VIG_DIR / f"ft_hp4_{k}_s{s}.json"
        _VIG.append(Job(f"seedvig_ft_hp4_{k}_s{s}", 1,
                        f"mkdir -p {VIG_DIR} && {PY} eval_seedvig.py --mode ft --seed 0 "
                        f"--bandpass 4 45 --checkpoints {k}_s{s}={_VIG_CK[f'{k}_s{s}']} "
                        f"--output {o}", o, short=True))
# 10-05: strict linear probe on CBraMod's other downstream tasks (our earlier
# "weak on other tasks" verdicts were FT-based). CBraMod's own preprocessing ->
# npz (CPU), then every checkpoint + random init + CBraMod/LaBraM on one load.
TASK_NPZ = Path("/home/pxieaf/home2/datasets/lp_tasks")
LPT_DIR = Path("/home/pxieaf/home2/eval_results/linear_probe_tasks")
for t in ("ma", "mumtaz", "physio", "faced"):
    b = Job(f"npz_{t}", 0, f"mkdir -p {TASK_NPZ} && OMP_NUM_THREADS=8 {PY} build_task_npz.py "
            f"--task {t} --out_dir {TASK_NPZ}", TASK_NPZ / f"{t}.npz", short=True,
            key=f"--task {t} --out_dir")
    o = LPT_DIR / f"lp_{t}.json"
    _VIG += [b, Job(f"lptask_{t}", 1,
                    f"mkdir -p {LPT_DIR} && OMP_NUM_THREADS=8 {PY} eval_linear_probe_tasks.py "
                    f"--task {t} --random_init --baselines cbramod labram "
                    f"--checkpoints {' '.join(f'{k}={v}' for k, v in _VIG_CK.items())} "
                    f"--output {o} && touch {o}.done",
                    Path(f"{o}.done"), deps=[b.name], short=True,
                    key=f"--task {t} --random_init", light=True)]
# 10-06 (CVPR B+C): linear tokenizer + per-patch FFT band-power target with
#   B: standard δθαβγ edges   C: the measured passbands of the default filters.
# main (filter, narrow) vs C isolates the estimator; C vs B the band definition.
PRE_FFTB = []
for s in (42, 43, 44):                            # seed 42 of both first
    for i, (tag, bands) in enumerate((("fftstd", "standard"), ("fftnarrow", "narrow"))):
        name = f"abl_linear_{tag}_s{s}_sub2100"
        p = pretrain(name, 29561 + 2 * (s - 42) + i,
                     ["tokenizer_type=linear", "target_type=fft_bandpower",
                      f"fft_bands={bands}", f"seed={s}", "subset_seed=42"])
        PRE_FFTB.append(p)
        LP2_GROUPS[name] = {f"lin{tag}_s{s}_ep16": _CK(name)}
# 10-06 (theory-aligned standard-band targets): B used a rectangular FFT and
# log1p of tiny band variances. D1 = standard bands + true log;
# D2 = standard bands + 3-taper DPSS multitaper + true log.
for s in (42, 43, 44):
    for i, (tag, extra) in enumerate((("fftstdlog", ["target_log=log"]),
                                      ("fftstdmt", ["fft_taper=multitaper", "target_log=log"]))):
        name = f"abl_linear_{tag}_s{s}_sub2100"
        p = pretrain(name, 29571 + 2 * (s - 42) + i,
                     ["tokenizer_type=linear", "target_type=fft_bandpower", "fft_bands=standard",
                      *extra, f"seed={s}", "subset_seed=42"])
        PRE_FFTB.append(p)
        LP2_GROUPS[name] = {f"lin{tag}_s{s}_ep16": _CK(name)}
# 10-06 SIGReg necessity (the target is fixed, so collapse is impossible without it):
#   F1 = main target (filter, narrow, log1p) with SIGReg 0 -- 2 GPUs x 256 (same
#        effective batch; SIGReg is off, so the per-GPU batch does not matter)
#   F2 = D2 (standard bands, multitaper, log) with SIGReg 0 -- fully theory-aligned
def pretrain2(name, port, overrides):
    out = M / name
    cmd = (f"{PY} emit_ablation_cmd.py --ref_args {REF} --output_dir {out} "
           f"--nproc 2 --batch_size 256 --master_port {port} "
           f"--stop_after_epoch {EP} --set {' '.join(overrides)}")
    run = (f'CMD=$({cmd}) && echo "$CMD" && '
           f'eval "${{CMD/torchrun/{PY} -m torch.distributed.run}} --resume_from auto"')
    return Job(f"pre_{name}", 2, run, out / f"checkpoint_ep{EP}.pt")


_F = [pretrain2("abl_linear_bandpower_nosig_s42_sub2100", 29581,
                ["tokenizer_type=linear", "target_type=bandpower", "sigreg_lambda=0",
                 "seed=42", "subset_seed=42"]),
      pretrain("abl_linear_fftstdmt_nosig_s42_sub2100", 29582,
               ["tokenizer_type=linear", "target_type=fft_bandpower", "fft_bands=standard",
                "fft_taper=multitaper", "target_log=log", "sigreg_lambda=0",
                "seed=42", "subset_seed=42"])]
# 10-07: F1/F2 showed SIGReg is unnecessary (and hurts the standard-band target)
#   G = D1 (standard bands, rect FFT, true log) with SIGReg 0, 2 GPUs x 256
_F.append(pretrain2("abl_linear_fftstdlog_nosig_s42_sub2100", 29583,
                    ["tokenizer_type=linear", "target_type=fft_bandpower", "fft_bands=standard",
                     "target_log=log", "sigreg_lambda=0", "seed=42", "subset_seed=42"]))
# 10-08: D1 held over 3 seeds -> G (D1 + SIGReg 0) seeds 43/44 on 4 GPUs
for s in (43, 44):
    _F.insert(0, pretrain(f"abl_linear_fftstdlog_nosig_s{s}_sub2100", 29584 + s - 43,
                          ["tokenizer_type=linear", "target_type=fft_bandpower",
                           "fft_bands=standard", "target_log=log", "sigreg_lambda=0",
                           f"seed={s}", "subset_seed=42"]))
for p in _F:
    LP2_GROUPS[p.name[4:]] = {p.name[4:].replace("abl_linear_", "lin").replace("_sub2100", "_ep16"): _CK(p.name[4:])}

# user 10-06: standard-band theory alignment is the top priority -> D1/D2 seed 42
# run before everything else (B/C seed 44 resume afterwards via --resume_from auto)
_first = [p for p in PRE_FFTB if p.name.endswith(("fftstdmt_s42_sub2100", "fftstdlog_s42_sub2100"))]
# 10-07: D1 (standard bands, rect FFT, true log) beat the main target on seed 42
# -> its seeds 43/44 go next, ahead of everything else still queued
_d1 = [p for p in PRE_FFTB if p.name.endswith(("fftstdlog_s43_sub2100", "fftstdlog_s44_sub2100"))]
PRE_FFTB = _first + _d1 + _F + [p for p in PRE_FFTB if p not in _first and p not in _d1]
_VIG += PRE_FFTB
for p in PRE_FFTB:
    for ds in ("isruc", "hmc"):
        j = lp2(ds, "specbram", p.name[4:])
        j.deps = [p.name]
        _VIG.append(j)

# 10-05: TUAB linear probe (CBraMod preprocessing / split; ~409k 10-s windows).
TUAB_LP = TASK_NPZ / "tuab"
_tb = Job("npz_tuab", 0, f"OMP_NUM_THREADS=2 {PY} build_tuab_lp.py --out_dir {TUAB_LP} "
          f"--workers 24", TUAB_LP / "DONE", short=True, key="build_tuab_lp.py")
_VIG.append(_tb)
for grp, ckeys, extra in (("lin", [k for k in _VIG_CK if k.startswith("lin")], "--random_init"),
                          ("fb", [k for k in _VIG_CK if k.startswith("fb")], ""),
                          ("base", [], "--baselines cbramod labram")):
    o = LPT_DIR / f"lp_tuab_{grp}.json"
    ck = (" --checkpoints " + " ".join(f"{k}={_VIG_CK[k]}" for k in ckeys)) if ckeys else ""
    _VIG.append(Job(f"lptuab_{grp}", 1,
                    f"mkdir -p {LPT_DIR} && OMP_NUM_THREADS=8 {PY} eval_linear_probe_tuab.py"
                    f"{ck} {extra} --output {o} && touch {o}.done",
                    Path(f"{o}.done"), deps=[_tb.name], short=True, key=f"--output {o}"))

# 10-09: new main model D1 (linear tok, standard bands, rect FFT, true log, SIGReg 0.05)
# -> every downstream evaluation again, 3 pretraining seeds
D1_CK = {f"d1_s{s}": _CK(f"abl_linear_fftstdlog_s{s}_sub2100") for s in (42, 43, 44)}
_ck = " ".join(f"{k}={v}" for k, v in D1_CK.items())
_D1 = []
for t in ("ma", "mumtaz", "physio", "faced"):
    o = LPT_DIR / f"lp_{t}_d1.json"
    _D1.append(Job(f"lptask_{t}_d1", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_tasks.py "
                   f"--task {t} --checkpoints {_ck} --output {o} && touch {o}.done",
                   Path(f"{o}.done"), short=True, key=f"--output {o}", light=True))
o = LPT_DIR / "lp_tuab_d1.json"
_D1.append(Job("lptuab_d1", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_tuab.py "
               f"--checkpoints {_ck} --output {o} && touch {o}.done",
               Path(f"{o}.done"), short=True, key=f"--output {o}"))
for bp in ((0.5, 45), (4, 45)):
    o = VIG_DIR / f"lp_d1_bp{bp[0]:g}.json"
    _D1.append(Job(f"seedvig_lp_d1_bp{bp[0]:g}", 1,
                   f"mkdir -p {VIG_DIR} && {PY} eval_seedvig.py --mode lp --bandpass {bp[0]} {bp[1]} "
                   f"--checkpoints {_ck} --output {o} && touch {o}.done",
                   Path(f"{o}.done"), short=True, key=f"--output {o}", light=True))
# SEED-VIG fine-tuning of the main model (0.5-45 Hz, as the other targets)
for s in (42, 43, 44):
    o = VIG_DIR / f"ft_d1_s{s}.json"
    _D1.append(Job(f"seedvig_ft_d1_s{s}", 1,
                   f"mkdir -p {VIG_DIR} && {PY} eval_seedvig.py --mode ft --seed 0 "
                   f"--checkpoints d1_s{s}={D1_CK[f'd1_s{s}']} --output {o}", o, short=True))
# fine-tuning is secondary: 30-s sleep FT (one run per pretraining seed) last
_D1 += [ft30(M / f"abl_linear_fftstdlog_s{s}_sub2100" / "checkpoint_ep16.pt", ds, 0)
        for s in (42, 43, 44) for ds in ("hmc", "isruc")]
_VIG += _D1

# 10-09: TUEV (6-class events) linear probe, CBraMod preprocessing / split
TUEV_LP = TASK_NPZ / "tuev"
_te = Job("npz_tuev", 0, f"OMP_NUM_THREADS=2 {PY} build_tuev_lp.py --out_dir {TUEV_LP} "
          f"--workers 24", TUEV_LP / "DONE", short=True, key="build_tuev_lp.py")
_VIG.append(_te)
for grp, ck, extra in (("d1", _ck, ""),
                       ("lin", " ".join(f"{k}={_VIG_CK[k]}" for k in _VIG_CK if k.startswith("lin")),
                        "--random_init"),
                       ("base", "", "--baselines cbramod labram")):
    o = LPT_DIR / f"lp_tuev_{grp}.json"
    _VIG.append(Job(f"lptuev_{grp}", 1,
                    f"mkdir -p {LPT_DIR} && OMP_NUM_THREADS=8 {PY} eval_linear_probe_tuab.py --task tuev"
                    f"{' --checkpoints ' + ck if ck else ''} {extra} --output {o} && touch {o}.done",
                    Path(f"{o}.done"), deps=[_te.name], short=True, key=f"--output {o}"))
# 10-09 (free GPUs): (1) vigilance hypothesis -- D1 without the delta band target
#   (= D1 target on a 4-Hz high-passed signal; input and tokenizer unchanged);
#   (2) D1 target with the filterbank tokenizer -> completes tokenizer x target for the
#   main target (vs fbraw, 3 seeds); (3) linear probe along pretraining for D1 seed 42.
_N = []
# 10-09 evening: the paper drops SIGReg (fixed target -> no collapse to prevent), so every
# new run uses sigreg_lambda=0; the SIGReg versions of these two runs were stopped at ep3.
_NOSIG = ["sigreg_lambda=0"]
_NEW = [("nd_s42", "abl_linear_fftnodelta_nosig_s42_sub2100", 29601,
         ["tokenizer_type=linear", "target_type=fft_bandpower", "fft_bands=nodelta",
          "target_log=log", "seed=42", "subset_seed=42", *_NOSIG]),
        ("fbd1_s42", "abl_fb_fftstdlog_nosig_s42_sub2100", 29602,
         ["tokenizer_type=filterbank", "target_type=fft_bandpower", "fft_bands=standard",
          "target_log=log", "seed=42", "subset_seed=42", *_NOSIG]),
        # main model without SIGReg: seeds 43/44 exist (4x128); seed 42 was run on 2x256
        # -> re-pretrain it on 4x128 so all three seeds share one launcher
        ("g_s42", "abl_linear_fftstdlog_nosig4g_s42_sub2100", 29603,
         ["tokenizer_type=linear", "target_type=fft_bandpower", "fft_bands=standard",
          "target_log=log", "seed=42", "subset_seed=42", *_NOSIG])]
for s in (43, 44):
    _NEW.append((f"fbd1_s{s}", f"abl_fb_fftstdlog_nosig_s{s}_sub2100", 29604 + s - 43,
                 ["tokenizer_type=filterbank", "target_type=fft_bandpower", "fft_bands=standard",
                  "target_log=log", f"seed={s}", "subset_seed=42", *_NOSIG]))
_EXIST = {f"g_s{s}": f"abl_linear_fftstdlog_nosig_s{s}_sub2100" for s in (43, 44)}


def _downstream(key, name, deps, sleep_lp=True, ft=False):
    out = []
    ck = f"{key}={_CK(name)}"
    if sleep_lp:
        LP2_GROUPS[name] = {f"{key}_ep16": _CK(name)}
        for ds in ("isruc", "hmc"):
            j = lp2(ds, "specbram", name)
            j.deps = list(deps)
            out.append(j)
    for t in ("ma", "mumtaz", "physio", "faced"):
        o = LPT_DIR / f"lp_{t}_{key}.json"
        out.append(Job(f"lptask_{t}_{key}", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_tasks.py "
                       f"--task {t} --checkpoints {ck} --output {o} && touch {o}.done",
                       Path(f"{o}.done"), deps=deps, short=True, key=f"--output {o}", light=True))
    for task in ("tuab", "tuev"):
        o = LPT_DIR / f"lp_{task}_{key}.json"
        out.append(Job(f"lp{task}_{key}", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_tuab.py "
                       f"--task {task} --checkpoints {ck} --output {o} && touch {o}.done",
                       Path(f"{o}.done"), deps=deps, short=True, key=f"--output {o}"))
    for bp in ((0.5, 45), (4, 45)):
        o = VIG_DIR / f"lp_{key}_bp{bp[0]:g}.json"
        out.append(Job(f"seedvig_lp_{key}_bp{bp[0]:g}", 1,
                       f"mkdir -p {VIG_DIR} && {PY} eval_seedvig.py --mode lp --bandpass {bp[0]} {bp[1]} "
                       f"--checkpoints {ck} --output {o} && touch {o}.done",
                       Path(f"{o}.done"), deps=deps, short=True, key=f"--output {o}", light=True))
    if ft:
        o = VIG_DIR / f"ft_{key}.json"
        out.append(Job(f"seedvig_ft_{key}", 1, f"mkdir -p {VIG_DIR} && {PY} eval_seedvig.py --mode ft "
                       f"--seed 0 --checkpoints {ck} --output {o}", o, deps=deps, short=True))
        for ds in ("hmc", "isruc"):
            j = ft30(M / name / "checkpoint_ep16.pt", ds, 0)
            j.deps = list(deps)
            out.append(j)
    return out


_PRE, _EV = [], []
for key, name, port, ov in _NEW:
    p = pretrain(name, port, ov)
    _PRE.append(p)
    _EV += _downstream(key, name, [p.name], ft=key.startswith("g_"))
for key, name in _EXIST.items():                   # already pretrained: evaluate now
    _EV += _downstream(key, name, [], sleep_lp=False, ft=True)
# 10-10: every comparison run was trained with SIGReg 0.05 -> re-run the controls without it
# so that they match the (no-SIGReg) main model. Priority: (1) linear raw recon, the main
# comparison, with the full downstream set; (2) multitaper s43/44; (3) filterbank raw recon;
# (4) band-waveform recon -- (2)-(4) only need the sleep linear probe (target-study table).
_CTRL = [("lr", "abl_linear_rawrecon_nosig_s{s}_sub2100", ["tokenizer_type=linear", "target_type=raw_recon"],
          (42, 43, 44), True),
         ("mt", "abl_linear_fftstdmt_nosig_s{s}_sub2100",
          ["tokenizer_type=linear", "target_type=fft_bandpower", "fft_bands=standard",
           "fft_taper=multitaper", "target_log=log"], (43, 44), False),
         ("fr", "abl_fb_rawrecon_nosig_s{s}_sub2100", ["tokenizer_type=filterbank", "target_type=raw_recon"],
          (42, 43, 44), False),
         ("br", "abl_fb_bandrecon_nosig_s{s}_sub2100", ["tokenizer_type=filterbank", "target_type=band_recon"],
          (42, 43, 44), False)]
_port = 29611
for tag, pat, ov, seeds, full in _CTRL:
    for s in seeds:
        name = pat.format(s=s)
        p = pretrain(name, _port, ov + [f"seed={s}", "subset_seed=42", *_NOSIG])
        _port += 1
        _PRE.append(p)
        if full:
            _EV += _downstream(f"{tag}_s{s}", name, [p.name], ft=True)
        else:
            LP2_GROUPS[name] = {f"{tag}_s{s}_ep16": _CK(name)}
            for ds in ("isruc", "hmc"):
                j = lp2(ds, "specbram", name)
                j.deps = [p.name]
                _EV.append(j)
# 10-12: scale-matched control -- our architecture at CBraMod's parameter count
# (d_model 144 -> ~4.2M parameters vs CBraMod ~4.0M), band power vs raw recon,
# no SIGReg, seed 42; frozen probes only. Placed ahead of the remaining queue.
_SMALL_PRE, _SMALL_EV = [], []
for key, name, tgt in (("sbp_s42", "abl_small144_fftstdlog_nosig_s42_sub2100",
                        ["target_type=fft_bandpower", "fft_bands=standard", "target_log=log"]),
                       ("srr_s42", "abl_small144_rawrecon_nosig_s42_sub2100", ["target_type=raw_recon"])):
    p = pretrain(name, _port, ["tokenizer_type=linear", "d_model=144", *tgt, "seed=42",
                                "subset_seed=42", *_NOSIG])
    _port += 1
    _SMALL_PRE.append(p)
    _SMALL_EV += _downstream(key, name, [p.name], ft=False)
# 10-12 (cont.): more seeds for the scale-matched pair and the no-delta target
for s in (43, 44):
    for key, name, ov in ((f"sbp_s{s}", f"abl_small144_fftstdlog_nosig_s{s}_sub2100",
                           ["d_model=144", "target_type=fft_bandpower", "fft_bands=standard",
                            "target_log=log"]),
                          (f"srr_s{s}", f"abl_small144_rawrecon_nosig_s{s}_sub2100",
                           ["d_model=144", "target_type=raw_recon"])):
        # (no-delta seeds 43/44 dropped 10-13: SEED-VIG removed from the paper)
        p = pretrain(name, _port, ["tokenizer_type=linear", *ov, f"seed={s}", "subset_seed=42", *_NOSIG])
        _port += 1
        _SMALL_PRE.append(p)
        _SMALL_EV += _downstream(key, name, [p.name], ft=False)
_N = _SMALL_PRE + _PRE + _SMALL_EV + _EV
LP2_GROUPS["d1curve"] = {f"d1_s42_ep{e}": _CK("abl_linear_fftstdlog_s42_sub2100", e)
                         for e in (1, 2, 4, 8, 12)}
LP2_GROUPS["d1curve"].update({f"linraw_s42_ep{e}": _CK("abl_linear_rawrecon_sub2100", e)
                              for e in (1, 2, 4, 8, 12)})
_N += [lp2(ds, "specbram", "d1curve") for ds in ("isruc", "hmc")]
# handcrafted spectral features beyond sleep, same probe (CPU only)
for t in ("ma", "mumtaz", "physio", "faced", "tuev", "tuab"):
    o = LPT_DIR / f"lp_{t}_hc.json"
    _N.append(Job(f"lphc_{t}", 0, f"OMP_NUM_THREADS=1 {PY} eval_handcrafted_tasks.py --task {t} "
                  f"--workers 8 --output {o} && touch {o}.done",
                  Path(f"{o}.done"), short=True, key=f"--output {o}"))
_VIG += _N
# 10-12: REVE measured under our strict probe (no more quoted linear-probe numbers)
_REVE = [lp2(ds, m) for m in ("reve_base", "reve_large") for ds in ("isruc", "hmc")]
_RB = "--baselines reve_base reve_large"
for t in ("ma", "mumtaz", "physio", "faced"):
    o = LPT_DIR / f"lp_{t}_reve.json"
    _REVE.append(Job(f"lptask_{t}_reve", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_tasks.py --task {t} "
                     f"{_RB} --output {o} && touch {o}.done", Path(f"{o}.done"), short=True,
                     key=f"--output {o}", light=True))
for task in ("tuev", "tuab"):
    o = LPT_DIR / f"lp_{task}_reve.json"
    _REVE.append(Job(f"lp{task}_reve", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_tuab.py --task {task} "
                     f"{_RB} --output {o} && touch {o}.done", Path(f"{o}.done"), short=True,
                     key=f"--output {o}"))
o = VIG_DIR / "lp_reve.json"
_REVE.append(Job("seedvig_lp_reve", 1, f"mkdir -p {VIG_DIR} && {PY} eval_seedvig.py --mode lp {_RB} "
                 f"--output {o} && touch {o}.done", Path(f"{o}.done"), short=True,
                 key=f"--output {o}", light=True))
# 10-12 (cont.): (a) linear probe along pretraining for the no-SIGReg main model and
# its matched raw recon (seed 42); (b) input-convention check for CBraMod/LaBraM;
# (c) label-free k-means clustering of frozen sleep features for every model
LP2_GROUPS["mcurve"] = {f"m_s42_ep{e}": _CK("abl_linear_fftstdlog_nosig4g_s42_sub2100", e)
                        for e in (1, 2, 4, 8, 12)}
LP2_GROUPS["mcurve"].update({f"lr_s42_ep{e}": _CK("abl_linear_rawrecon_nosig_s42_sub2100", e)
                             for e in (1, 2, 4, 8, 12)})
_REVE += [lp2(ds, "specbram", "mcurve") for ds in ("isruc", "hmc")]
_REVE += [lp2(ds, m) for m in ("cbramod_rn", "labram_rn") for ds in ("isruc", "hmc")]
CL_DIR = Path("/home/pxieaf/home2/eval_results/cluster")
_main_ck = {f"m_s{s}": _CK(d) for s, d in ((42, "abl_linear_fftstdlog_nosig4g_s42_sub2100"),
                                          (43, "abl_linear_fftstdlog_nosig_s43_sub2100"),
                                          (44, "abl_linear_fftstdlog_nosig_s44_sub2100"))}
_main_ck.update({f"lr_s{s}": _CK(f"abl_linear_rawrecon_nosig_s{s}_sub2100") for s in (42, 43, 44)})
for ds in ("isruc", "hmc"):
    for m in ("specbram", "cbramod", "labram", "biot", "reve_base", "reve_large"):
        o = CL_DIR / f"cl_{ds}_{m}.json"
        ck = (" --checkpoints " + " ".join(f"{k}={v}" for k, v in _main_ck.items())) if m == "specbram" else ""
        _REVE.append(Job(f"cluster_{ds}_{m}", 1, f"mkdir -p {CL_DIR} && OMP_NUM_THREADS=8 {PY} "
                         f"eval_cluster_sleep.py --dataset {ds} --data_dir {DATA[ds]} --model {m}{ck} "
                         f"--output {o}", Path(f"{o}.done"), short=True, key=f"--output {o}", light=True))
# 10-12 (cont.): stronger frozen readout -- per-channel token means, concatenated
# (spatial information kept; still logistic regression), full labels only
for ds in ("isruc", "hmc"):
    for m in ("specbram", "cbramod", "labram", "reve_base", "reve_large"):
        o = LP2_DIR / f"lp2chan_{ds}_{m}.json"
        ck = (" --checkpoints " + " ".join(f"{k}={v}" for k, v in _main_ck.items())) if m == "specbram" else ""
        _REVE.append(Job(f"lp2chan_{ds}_{m}", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_v2.py "
                         f"--dataset {ds} --data_dir {DATA[ds]} --model {m}{ck} --readout chan "
                         f"--modes avg3 --no_label_eff --output {o}", Path(f"{o}.done"), short=True,
                         key=f"--output {o}", light=True))
# 10-13: CSBrain (8.9M parameters, official weights) under every probe; replaces REVE-Large
_CSB = [lp2(ds, "csbrain") for ds in ("isruc", "hmc")]
for ds in ("isruc", "hmc"):
    o = LP2_DIR / f"lp2chan_{ds}_csbrain.json"
    _CSB.append(Job(f"lp2chan_{ds}_csbrain", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_v2.py --dataset {ds} "
                    f"--data_dir {DATA[ds]} --model csbrain --readout chan --modes avg3 --no_label_eff "
                    f"--output {o}", Path(f"{o}.done"), short=True, key=f"--output {o}", light=True))
    o = CL_DIR / f"cl_{ds}_csbrain.json"
    _CSB.append(Job(f"cluster_{ds}_csbrain", 1, f"mkdir -p {CL_DIR} && OMP_NUM_THREADS=8 {PY} eval_cluster_sleep.py "
                    f"--dataset {ds} --data_dir {DATA[ds]} --model csbrain --output {o}", Path(f"{o}.done"),
                    short=True, key=f"--output {o}", light=True))
for t in ("ma", "mumtaz", "physio", "faced"):
    o = LPT_DIR / f"lp_{t}_csbrain.json"
    _CSB.append(Job(f"lptask_{t}_csbrain", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_tasks.py --task {t} "
                    f"--baselines csbrain --output {o} && touch {o}.done", Path(f"{o}.done"), short=True,
                    key=f"--output {o}", light=True))
for task in ("tuev", "tuab"):
    o = LPT_DIR / f"lp_{task}_csbrain.json"
    _CSB.append(Job(f"lp{task}_csbrain", 1, f"OMP_NUM_THREADS=8 {PY} eval_linear_probe_tuab.py --task {task} "
                    f"--baselines csbrain --output {o} && touch {o}.done", Path(f"{o}.done"), short=True,
                    key=f"--output {o}"))
_REVE = _CSB + _REVE

# 10-14: re-run the overview probes once with --pred_dir (test predictions + subject ids)
# for a subject-level bootstrap; scores are deterministic and must match the earlier runs
BOOT = Path("/home/pxieaf/home2/eval_results/bootstrap")
_mck = " ".join(f"{k}={v}" for k, v in _main_ck.items() if k.startswith("m_"))
_BOOT = []
for ds in ("isruc", "hmc"):
    for m in ("specbram", "cbramod", "labram", "biot", "csbrain", "reve_base"):
        o = BOOT / f"lp2_{ds}_{m}.json"
        ck = f" --checkpoints {_mck}" if m == "specbram" else ""
        _BOOT.append(Job(f"boot_{ds}_{m}", 1, f"mkdir -p {BOOT} && OMP_NUM_THREADS=8 {PY} eval_linear_probe_v2.py "
                         f"--dataset {ds} --data_dir {DATA[ds]} --model {m}{ck} --modes avg3 --no_label_eff "
                         f"--pred_dir {BOOT}/preds --output {o}", Path(f"{o}.done"), short=True,
                         key=f"--output {o}", light=True))
_BL = "--baselines cbramod labram csbrain reve_base"
for t in ("ma", "mumtaz", "physio", "faced"):
    o = BOOT / f"lp_{t}.json"
    _BOOT.append(Job(f"boot_{t}", 1, f"mkdir -p {BOOT} && OMP_NUM_THREADS=8 {PY} eval_linear_probe_tasks.py --task {t} "
                     f"--checkpoints {_mck} {_BL} --pred_dir {BOOT}/preds --output {o} && touch {o}.done",
                     Path(f"{o}.done"), short=True, key=f"--output {o}", light=True))
for task in ("tuev", "tuab"):
    o = BOOT / f"lp_{task}.json"
    _BOOT.append(Job(f"boot_{task}", 1, f"mkdir -p {BOOT} && OMP_NUM_THREADS=8 {PY} eval_linear_probe_tuab.py "
                     f"--task {task} --checkpoints {_mck} {_BL} --pred_dir {BOOT}/preds --output {o} && touch {o}.done",
                     Path(f"{o}.done"), short=True, key=f"--output {o}"))
_REVE = _BOOT + _REVE

# 10-13: full fine-tuning on the non-sleep tasks (appendix only): main model and matched
# raw recon (no SIGReg), three pretraining seeds each; lowest priority
FTT_DIR = Path("/home/pxieaf/home2/eval_results/finetune_tasks")
_FTT = []
for t in ("ma", "mumtaz", "physio", "faced", "tuev", "tuab"):
    for nm, ck in _main_ck.items():
        o = FTT_DIR / f"ft_{t}_{nm}.json"
        _FTT.append(Job(f"fttask_{t}_{nm}", 1, f"mkdir -p {FTT_DIR} && {PY} eval_finetune_tasks.py "
                        f"--task {t} --checkpoint {ck} --name {nm} --output {o}", o, short=True,
                        key=f"--output {o}"))
JOBS = _REVE + _VIG + JOBS + _FTT
_seen = set()                              # drop repeats, keep first (highest priority)
JOBS = [j for j in JOBS if not (j.name in _seen or _seen.add(j.name))]


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M')}] {msg}"
    print(line, flush=True)
    with open(LOG / "queue.log", "a") as f:
        f.write(line + "\n")


def read_pool():
    try:
        return [int(g) for g in (LOG / "pool").read_text().split()]
    except (OSError, ValueError):
        return list(POOL)


def least_loaded(n, jobs):
    """Pool GPUs ordered by used memory (+8 GB per light job already there)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        used = {int(a): float(b) for a, b in (l.split(",") for l in out.strip().splitlines())}
    except Exception:
        used = {}
    for j in jobs:
        if j.state == "run" and j.light:
            for g in j.gpus:
                used[g] = used.get(g, 0) + 8000
    pool = read_pool()
    return sorted(pool, key=lambda g: used.get(g, 0))[:n]


def find_running(key):
    """(pid, gpus) of a live process whose command line contains key."""
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
            if key not in cmd or "rebuttal_sched" in cmd:
                continue
            env = (d / "environ").read_bytes().split(b"\x00")
        except OSError:
            continue
        for e in env:
            if e.startswith(b"CUDA_VISIBLE_DEVICES="):
                gpus = [int(g) for g in e.split(b"=", 1)[1].decode().split(",") if g]
                return int(d.name), gpus
    return None


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    by_name = {j.name: j for j in JOBS}
    for j in JOBS:
        if j.out.exists():
            j.state = "done"
            log(f"skip {j.name} (output exists)")
            continue
        found = find_running(j.key)
        if found:
            j.pid, j.gpus = found
            j.state = "run"
            log(f"ADOPT {j.name} (pid {j.pid}) on GPU {j.gpus}")
    log(f"===== scheduler start: {len(JOBS)} jobs, pool {read_pool()} =====")
    while any(j.state in ("wait", "run") for j in JOBS):
        for j in JOBS:                                   # reap finished jobs
            if j.state == "run" and (rc := j.poll()) is not None:
                ok = rc == 0 and j.out.exists()
                j.state = "done" if ok else "fail"
                log(f"{'DONE' if ok else 'FAIL'} {j.name} (rc={rc}) "
                    f"-> freed GPU {j.gpus}")
        busy = {g for j in JOBS if j.state == "run" and not j.light for g in j.gpus}
        free = [g for g in read_pool() if g not in busy]
        blocked = False
        for j in JOBS:                                   # start ready jobs
            if j.state != "wait":
                continue
            dep_states = [by_name[d].state for d in j.deps]
            if "fail" in dep_states:
                j.state = "fail"
                log(f"FAIL {j.name} (dependency failed)")
                continue
            if any(s != "done" for s in dep_states):
                continue
            if blocked and not j.short:
                continue
            if j.light and len(free) < j.n_gpu:
                # light jobs barely touch the GPU: share the pool GPU(s) with the
                # least memory in use, counting light jobs already placed there
                j.gpus = least_loaded(j.n_gpu, JOBS)
            elif len(free) < j.n_gpu:
                blocked = blocked or not j.short
                continue
            else:
                j.gpus, free = free[:j.n_gpu], free[j.n_gpu:]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, j.gpus)),
                       PYTHONUNBUFFERED="1")
            lf = open(LOG / f"{j.name}.log", "a")
            j.proc = subprocess.Popen(["bash", "-c", j.cmd], stdout=lf,
                                      stderr=subprocess.STDOUT, env=env)
            j.state = "run"
            log(f"START {j.name} on GPU {j.gpus}")
        time.sleep(60)
    log("===== scheduler finished: " + ", ".join(
        f"{j.name}={j.state}" for j in JOBS) + " =====")


if __name__ == "__main__":
    main()
