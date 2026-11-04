#!/bin/bash
# Test features of SpecBraM (main checkpoint), CBraMod and REVE-Base on all eight datasets,
# for the t-SNE case analysis (SAVE_FEAT=1 stores them next to the predictions).
cd ~/Brain-WM
PY=~/anaconda3/envs/brainwm/bin/python
OUT=~/home2/eval_results/feats
LOG=~/home2/logs_2/rebuttal
CK="m_s42=/home/pxieaf/home2/model/abl_linear_fftstdlog_nosig4g_s42_sub2100/checkpoint_ep16.pt"
declare -A DATA=([isruc]=/home/pxieaf/home2/datasets/isruc/subgroupI_official [hmc]=/home/pxieaf/home2/datasets/HMC)
mkdir -p $OUT/preds
g=0
GPUS=(0 1 2 3 4 5 6 7 9)
run() {   # run <name> <cmd...>
  CUDA_VISIBLE_DEVICES=${GPUS[$((g % 9))]} SAVE_FEAT=1 OMP_NUM_THREADS=8 setsid nohup "$@" \
    > $LOG/feat_$NAME.log 2>&1 < /dev/null &
  g=$((g + 1))
}
for ds in isruc hmc; do
  NAME=${ds}_specbram; run $PY eval_linear_probe_v2.py --dataset $ds --data_dir ${DATA[$ds]} --model specbram \
      --checkpoints $CK --modes avg3 --no_label_eff --pred_dir $OUT/preds --output $OUT/lp2_${ds}_specbram.json
  for m in cbramod reve_base; do
    NAME=${ds}_$m; run $PY eval_linear_probe_v2.py --dataset $ds --data_dir ${DATA[$ds]} --model $m \
        --modes avg3 --no_label_eff --pred_dir $OUT/preds --output $OUT/lp2_${ds}_$m.json
  done
done
for t in ma mumtaz physio faced; do
  NAME=$t; run $PY eval_linear_probe_tasks.py --task $t --checkpoints $CK --baselines cbramod reve_base \
      --pred_dir $OUT/preds --output $OUT/lp_$t.json
done
for t in tuev tuab; do
  NAME=$t; run $PY eval_linear_probe_tuab.py --task $t --checkpoints $CK --baselines cbramod reve_base \
      --pred_dir $OUT/preds --output $OUT/lp_$t.json
done
echo "launched $g jobs"
