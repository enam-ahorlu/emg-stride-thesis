#!/usr/bin/env bash
# W-3: skip-connection ablation. Two 40-fold runs, both SE-free, matching the
# G1 pair flag for flag except --arch. See EXPERIMENT_PLAN_RESIDUAL_ABLATION.md.
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/w3_nores.log
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --norm-mode per_subject --resume"

echo "==== W-3 CHAIN START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# --- 0. smoke: one fold, confirms resnet_nores builds and trains at all ---
rm -rf _w3_smoke
echo "--- [smoke] nores heldout 1 $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_nores --augmentation none \
  $BASE --heldout 1 --out _w3_smoke >> "$LOG" 2>&1
rc=$?; echo "[smoke] exit $rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[smoke] FAILED, stopping" | tee -a "$LOG"; exit $rc; }

# --- 1. no-skip baseline, no augmentation. THE TRAINABILITY CHECK. ---
echo "--- [1] nores no-aug $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_nores --augmentation none \
  $BASE --out results_w3_nores_noaug >> "$LOG" 2>&1
rc=$?; echo "[1] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[1] FAILED, stopping" | tee -a "$LOG"; exit $rc; }

# Stop here and read §4 of the plan before launching [2] if the baseline looks low.
"$PY" -c "import pandas as pd; m=pd.read_csv('results_w3_nores_noaug/cnn_arch_summary.csv')['f1_macro_mean'][0]; print(f'[gate] nores no-aug mean {m:.4f}; plain resnet 0.7600; delta {(m-0.76)*100:+.2f} pp')" | tee -a "$LOG"

# --- 2. no-skip + channel dropout p=0.2 ---
echo "--- [2] nores + CD $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_nores --augmentation chandrop \
  --aug-chandrop-p 0.2 $BASE --out results_w3_nores_chandrop >> "$LOG" 2>&1
rc=$?; echo "[2] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[2] FAILED, stopping" | tee -a "$LOG"; exit $rc; }

echo "==== W-3 CHAIN END $(date -u +%FT%TZ) rc=0 ====" | tee -a "$LOG"
