#!/usr/bin/env bash
# W-5: separating depth from capacity. Four 40-fold runs on the SE-free, skip-free
# resnet_nores backbone, plus two 3-epoch smoke tests. See
# EXPERIMENT_PLAN_DEPTH_VS_CAPACITY.md. Every flag other than --arch/--widths/
# --blocks-per-stage/--augmentation matches the W-3 runs exactly.
#   SHALLOW-MATCHED : bps=1, widths 48,96,192  -> 522,196 params  (-2.47% vs BASE 535,396)
#   NARROW-MATCHED  : bps=2, widths 21,42,84   -> 231,697 params  (-0.75% vs bps=1/(32,64,128) target 233,444)
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/w5_depth_capacity.log
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--norm-mode per_subject --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --resume"

run () {  # $1 name  $2 widths  $3 bps  $4 augflags
  local name=$1 widths=$2 bps=$3 augf=$4
  echo "--- [$name] $(date -u +%FT%TZ)  widths=$widths bps=$bps  $augf ---" | tee -a "$LOG"
  "$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_nores \
    --widths "$widths" --blocks-per-stage "$bps" $augf $BASE \
    --out "results_w5_$name" >> "$LOG" 2>&1
  local rc=$?; echo "[$name] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[$name] FAILED, stopping" | tee -a "$LOG"; exit $rc; }
}

echo "==== W-5 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# --- §2 gate 3: two smoke tests, 1 fold, 3 epochs each ---
for sm in "shallow 48,96,192 1" "narrow 21,42,84 2"; do
  set -- $sm; nm=$1; w=$2; b=$3
  rm -rf "_w5_smoke_$nm"
  echo "--- [smoke $nm] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_nores \
    --widths "$w" --blocks-per-stage "$b" --augmentation none --norm-mode per_subject \
    --epochs 3 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --resume \
    --heldout 1 --out "_w5_smoke_$nm" >> "$LOG" 2>&1
  rc=$?; echo "[smoke $nm] exit $rc" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[smoke $nm] FAILED, stopping" | tee -a "$LOG"; exit $rc; }
  [ -s "_w5_smoke_$nm/cnn_arch_subjectwise.csv" ] || { echo "[smoke $nm] no CSV, stopping" | tee -a "$LOG"; exit 1; }
done

# --- the four 40-fold runs ---
run shallow_noaug    "48,96,192" 1 "--augmentation none"
run shallow_chandrop "48,96,192" 1 "--augmentation chandrop --aug-chandrop-p 0.2"
run narrow_noaug     "21,42,84"  2 "--augmentation none"
run narrow_chandrop  "21,42,84"  2 "--augmentation chandrop --aug-chandrop-p 0.2"

echo "--- [analysis] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u w5_depth_capacity_stats.py 2>&1 | tee -a "$LOG"
echo "==== W-5 END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
