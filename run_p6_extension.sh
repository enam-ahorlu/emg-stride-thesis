#!/usr/bin/env bash
# P-6 optional extension (EXPERIMENT_PLAN_CD_PARITY.md section 5B.3). Gain jitter
# at sd = 0.30 and sd = 0.50, matching the multiplicative SD of channel dropout
# at p = 0.1 and p = 0.5 (sqrt(p(1-p)) = 0.30 and 0.50). With the existing
# four-rate sweep this gives two curves over one shared variance axis.
#
# RUN ONLY IF p6_ladder_stats.py reported the core arms coherent.
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/p6_extension.log
mkdir -p _run_logs
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --norm-mode per_subject --resume"

echo "==== P6 EXTENSION START $(date -u +%FT%TZ) ====" | tee -a "$LOG"
for SD in 0.30 0.50; do
  OUT="results_p6_gainjitter_resnet_se_sd${SD}"
  echo "--- [gainjitter sd=$SD] -> $OUT  $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$PY" -u run_cnn_arch_loso.py --npz "$NPZ" --meta "$META" --arch resnet_se \
    --augmentation gainjitter --aug-gain-sd "$SD" $BASE --out "$OUT" >> "$LOG" 2>&1
  rc=$?; echo "[sd=$SD] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[sd=$SD] FAILED, stopping" | tee -a "$LOG"; exit $rc; }
  "$PY" -c "import pandas as pd;m=pd.read_csv('$OUT/cnn_arch_summary.csv');print('[sd=$SD] mean F1',round(float(m['f1_macro_mean'][0]),4))" | tee -a "$LOG"
done
echo "==== P6 EXTENSION END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
