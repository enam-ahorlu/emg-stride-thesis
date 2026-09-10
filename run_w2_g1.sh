#!/usr/bin/env bash
# W-2: (1) fresh plain-ResNet baseline under current code, then (2) Stage G1.
# Both interaction differences then use current-code baselines (era-internal).
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/w2_g1.log
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --norm-mode per_subject --resume"

echo "==== W-2 G1 CHAIN START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# --- 0. smoke: one fold of the exact G1 command, checks the RNG-neutrality assert ---
rm -rf _w2_g1_smoke
echo "--- [smoke] G1 heldout 1 $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet --augmentation chandrop \
  --aug-chandrop-p 0.2 $BASE --heldout 1 --out _w2_g1_smoke --instrument _w2_g1_smoke/instr >> "$LOG" 2>&1
rc=$?; echo "[smoke] exit $rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[smoke] FAILED, stopping" | tee -a "$LOG"; exit $rc; }

# --- 1. fresh plain-ResNet baseline, current code, no augmentation ---
echo "--- [1] plain ResNet no-aug repro $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet --augmentation none \
  $BASE --out results_cd_resnet_noaug_repro >> "$LOG" 2>&1
rc=$?; echo "[1] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[1] FAILED, stopping" | tee -a "$LOG"; exit $rc; }

# --- 2. Stage G1: plain ResNet + channel dropout p=0.2, instrumented ---
echo "--- [2] G1 ResNet+CD $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet --augmentation chandrop \
  --aug-chandrop-p 0.2 $BASE --out results_cd_resnet_nose_chandrop \
  --instrument results_cd_resnet_nose_chandrop/instr >> "$LOG" 2>&1
rc=$?; echo "[2] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[2] FAILED, stopping" | tee -a "$LOG"; exit $rc; }

echo "==== W-2 G1 CHAIN END $(date -u +%FT%TZ) rc=0 ====" | tee -a "$LOG"
