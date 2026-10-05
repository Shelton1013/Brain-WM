#!/usr/bin/env bash
# Guard for the patch_local leakage-control pretraining (run detached).
#   1. wait until the job on GPU 0 (full-data control ep5 HMC FT) has written
#      its result;
#   2. give rebuttal_sched.py 10 min to start pre_abl_patchlocal_sub2100; if it
#      has not, launch it here on 4 free GPUs and restart the scheduler so it
#      ADOPTS the run (no duplicate launch);
#   3. watch the first 30 min and report whether training really started.
# Everything is logged to $LOG/watch_patchlocal.log.
set -u
M=/home/pxieaf/home2/model
LOG=/home/pxieaf/home2/logs_2/rebuttal
OUT=$M/abl_patchlocal_sub2100
PLOG=$LOG/pre_abl_patchlocal_sub2100.log
GATE=$M/main_time_r30_nojepa/checkpoint_ep5_hmc_ft.json
WLOG=$LOG/watch_patchlocal.log
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate brainwm
PY=$(which python)

log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> "$WLOG"; }
running(){ ps -eo args | grep -v grep | grep -q -- "--output_dir $OUT\b"; }
free_gpus(){  # GPUs with <1 GB used, from the pool
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v pool=" $(cat $LOG/pool 2>/dev/null) " \
        '$2 < 1000 && index(pool, " "$1" ") {print $1}'
}

log "watcher start; waiting for $GATE"
until [ -f "$GATE" ]; do sleep 60; done
log "GPU-0 job finished ($(basename $GATE) written)"

for i in $(seq 1 10); do running && break; sleep 60; done
if running; then
  log "scheduler started patch_local"
else
  G=$(free_gpus | head -4 | paste -sd,)
  if [ "$(echo "$G" | tr ',' '\n' | grep -c .)" -lt 4 ]; then
    log "NOT STARTED and only [$G] free - leaving it to the scheduler"; exit 1
  fi
  log "scheduler did not start it within 10 min; launching on GPU $G"
  CMD=$($PY emit_ablation_cmd.py --ref_args $M/sub2100_2500h_matched/args.json \
        --output_dir $OUT --nproc 4 --batch_size 128 --master_port 29513 \
        --stop_after_epoch 16 --set patch_local_filter=true)
  CMD=${CMD/torchrun/$PY -m torch.distributed.run}
  CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 setsid nohup $CMD >> "$PLOG" 2>&1 < /dev/null &
  sleep 30
  SP=$(ps -eo pid,args | awk '$2 ~ /python/ && $3 == "rebuttal_sched.py" {print $1}')
  [ -n "$SP" ] && kill $SP && sleep 3
  (setsid nohup $PY rebuttal_sched.py >> $LOG/sched.out 2>&1 < /dev/null &)
  log "scheduler restarted (old pid ${SP:-none}) to adopt the run"
fi

START_LINES=$(wc -l < "$PLOG" 2>/dev/null || echo 0)
for i in $(seq 1 30); do
  sleep 60
  NEW=$(tail -n +$((START_LINES + 1)) "$PLOG" 2>/dev/null)
  if echo "$NEW" | grep -q "Traceback\|ChildFailedError"; then
    log "FAILED: $(echo "$NEW" | grep -m1 -E "Error" | cut -c1-200)"; exit 1
  fi
  if echo "$NEW" | grep -qE "lr=|\[Epoch"; then
    log "OK: training steps running - $(echo "$NEW" | grep -E "lr=|\[Epoch" | tail -1 | cut -c1-160)"
    exit 0
  fi
done
running && log "still loading data after 30 min (process alive)" \
        || log "FAILED: process gone, no training output"
