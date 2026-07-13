#!/usr/bin/env bash
# ==========================================================================
# EEG-LeJEPA v3 downstream eval suite.
# Usage: set CKPT (+ GPU) below, then run the blocks you want (copy-paste or
#        `bash run_evals.sh <task>`). Each block writes JSON next to the ckpt.
#
# Conventions:
#   Mumtaz / Siena : tiny (63 / 14 subj) -> 5 seeds (42-46), --mode both (frozen+FT)
#   ISRUC  / HMC   : large, low-variance  -> 3 reps in one run (frozen and FT separate)
#   Paths are fixed to the server layout. Swap CKPT only.
# ==========================================================================
set -u

# ---- EDIT THESE ----------------------------------------------------------
CKPT=${CKPT:-/home/pxieaf/home2/model/lejepa_v3_filter_full_e15/checkpoint_ep2.pt}
GPU=${GPU:-8}                     # 10 GPUs (0-9); training uses 0-7, so 8 or 9 are free for eval
# --------------------------------------------------------------------------
OUT=$(dirname "$CKPT"); LOG=/home/pxieaf/home2/logs_2
COMMON="--sample_rate 256 --trial_duration_s 10 --normalization per_recording_robust"
CACHE=/home/pxieaf/home2/dataset_cache
run(){ echo -e "\n### $1"; shift; CUDA_VISIBLE_DEVICES=$GPU "$@"; }

task=${1:-help}
case "$task" in

# ---------------- MUMTAZ (depression, LogReg, 5 seed) --------------------
mumtaz)
  for S in 42 43 44 45 46; do
    run "mumtaz seed $S" python eval_mumtaz.py --mode both --checkpoint "$CKPT" \
      --mumtaz_dir /home/pxieaf/home2/datasets/mumtaz2016 --cache_dir $CACHE $COMMON \
      --frozen_reps 5 --ft_protocol onecycle --max_epochs 50 --n_reps 3 \
      --include_random_baseline --seed $S \
      --output "$OUT/mumtaz_seed${S}.json"
  done 2>&1 | tee $LOG/eval_mumtaz.log ;;

# ---------------- SIENA (seizure, LogReg, 5 seed, imbalanced) ------------
siena)
  for S in 42 43 44 45 46; do
    run "siena seed $S" python eval_siena.py --mode both --checkpoint "$CKPT" \
      --siena_dir /home/pxieaf/home2/datasets/Siena/1.0.0 --cache_dir $CACHE $COMMON \
      --frozen_reps 5 --negative_per_positive 0 --include_random_baseline --seed $S \
      --output "$OUT/siena_seed${S}.json"
  done 2>&1 | tee $LOG/eval_siena.log ;;

# ---------------- ISRUC (sleep, seq2seq, 3 reps) -------------------------
isruc_frozen)
  run "isruc frozen" python eval_sleep_seq2seq.py --dataset isruc --checkpoint "$CKPT" \
    --data_dir /home/pxieaf/home2/datasets/isruc/subgroupI_official --cache_dir $CACHE \
    --freeze_encoder --n_reps 3 --include_random_baseline \
    --output "$OUT/isruc_frozen.json" 2>&1 | tee $LOG/eval_isruc_frozen.log ;;
isruc_ft)
  run "isruc ft" python eval_sleep_seq2seq.py --dataset isruc --checkpoint "$CKPT" \
    --data_dir /home/pxieaf/home2/datasets/isruc/subgroupI_official --cache_dir $CACHE \
    --n_reps 3 --include_random_baseline --batch_size 8 \
    --output "$OUT/isruc_ft.json" 2>&1 | tee $LOG/eval_isruc_ft.log ;;

# ---------------- HMC (sleep, seq2seq, 3 reps) --------------------------
hmc_frozen)
  run "hmc frozen" python eval_sleep_seq2seq.py --dataset hmc --checkpoint "$CKPT" \
    --data_dir /home/pxieaf/home2/datasets/HMC --cache_dir $CACHE \
    --freeze_encoder --n_reps 3 --include_random_baseline \
    --output "$OUT/hmc_frozen.json" 2>&1 | tee $LOG/eval_hmc_frozen.log ;;
hmc_ft)
  run "hmc ft" python eval_sleep_seq2seq.py --dataset hmc --checkpoint "$CKPT" \
    --data_dir /home/pxieaf/home2/datasets/HMC --cache_dir $CACHE \
    --n_reps 3 --include_random_baseline --batch_size 8 \
    --output "$OUT/hmc_ft.json" 2>&1 | tee $LOG/eval_hmc_ft.log ;;

# ---------------- TUAB (abnormal, optional, TUH-derived) ----------------
tuab)
  run "tuab" python eval_tuh_clinical.py --checkpoint "$CKPT" \
    --data_dir /home/pxieaf/home2/tuh/tuh_eeg_abnormal/v3.0.1/edf \
    --cache_dir $CACHE $COMMON --kind tuab --n_reps 3 --include_random_baseline \
    --output "$OUT/tuab.json" 2>&1 | tee $LOG/eval_tuab.log ;;

help|*)
  cat <<EOF
Set CKPT and GPU at the top (or export them), then:
  bash run_evals.sh mumtaz        # 5-seed frozen+FT
  bash run_evals.sh siena         # 5-seed frozen+FT
  bash run_evals.sh isruc_frozen  # 3-rep
  bash run_evals.sh isruc_ft      # 3-rep
  bash run_evals.sh hmc_frozen
  bash run_evals.sh hmc_ft
  bash run_evals.sh tuab          # optional
Current CKPT=$CKPT  GPU=$GPU
EOF
;;
esac
