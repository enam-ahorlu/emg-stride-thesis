#!/usr/bin/env bash
# W-4: the gain-jitter control. Requires the augment_batch change of
# EXPERIMENT_PLAN_GAIN_JITTER.md §2 and its §2.2 inertness assertion FIRST.
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/w4_gainjitter.log
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --norm-mode per_subject --resume"

echo "==== W-4 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

rm -rf _w4_smoke
echo "--- [smoke] gainjitter heldout 1 $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet \
  --augmentation gainjitter --aug-gain-sd 0.4 $BASE --heldout 1 --out _w4_smoke >> "$LOG" 2>&1
rc=$?; echo "[smoke] exit $rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[smoke] FAILED, stopping" | tee -a "$LOG"; exit $rc; }
grep -m1 "\[aug\]" "$LOG" | tee -a "$LOG"

echo "--- [1] gainjitter sd=0.4, 40 folds $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet \
  --augmentation gainjitter --aug-gain-sd 0.4 $BASE --out results_w4_gainjitter >> "$LOG" 2>&1
rc=$?; echo "[1] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[1] FAILED, stopping" | tee -a "$LOG"; exit $rc; }

echo "--- [2] analysis $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u w4_gainjitter_stats.py 2>&1 | tee -a "$LOG"
echo "==== W-4 END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
