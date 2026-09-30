#!/usr/bin/env bash
# R-1 Step 2: re-run the W-1 250 ms global ResNet-SE+CD arm twice more, changing
# nothing. Exactly the flags that produced results/win250_cnn_global (W-1
# jobs/jobs_window_cnn.txt line 5) -- the full explicit set, not the abbreviated one
# in the plan (they are equal: batch/lr/patience/val-frac all match defaults).
set -u
cd "$(dirname "$0")/.."
PY="C:/dev/emg-stride-thesis/.venv/Scripts/python.exe"
LOG=logs/r1_step2.log
NPZ=data/windows/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=data/features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
COMMON="--npz $NPZ --meta $META --arch resnet_se --augmentation chandrop --aug-chandrop-p 0.2 --norm-mode global --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --resume"

echo "==== R-1 STEP 2 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"
for R in r2 r3; do
  echo "=========== [$R] $(date -u +%FT%TZ) ===========" | tee -a "$LOG"
  "$PY" -u src/run_cnn_arch_loso.py $COMMON --out "results/repro_250global_$R" >> "$LOG" 2>&1
  rc=$?
  echo "[$R] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[$R] FAILED, stopping" | tee -a "$LOG"; exit $rc; }
done
echo "==== R-1 STEP 2 RUNS DONE $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# pairwise reproducibility between the three identical runs
for PAIR in "results/win250_cnn_global results/repro_250global_r2" \
            "results/win250_cnn_global results/repro_250global_r3" \
            "results/repro_250global_r2 results/repro_250global_r3"; do
  set -- $PAIR
  echo "----- pair: $1  vs  $2 -----" | tee -a "$LOG"
  "$PY" -u src/check_cnn_reproducibility.py --a "$1" --a-col f1_macro --a-label "$1" \
        --b "$2" --b-col f1_macro --b-label "$2" >> "$LOG" 2>&1
done
echo "==== R-1 STEP 2 END $(date -u +%FT%TZ) rc=0 ====" | tee -a "$LOG"
