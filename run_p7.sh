#!/usr/bin/env bash
# P-7 (EXPERIMENT_PLAN_P7_DIVERGENCE.md). One 40-fold run, no code changes.
# mpchandrop at requested SD 0.50 -> p' = 0.2, mean 1.0, Bernoulli form, 20% zeroed + rescale.
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/p7.log
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
echo "==== P7 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz "$NPZ" --meta "$META" --arch resnet_se --norm-mode per_subject \
  --augmentation mpchandrop --aug-gain-sd 0.50 \
  --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --resume \
  --out results_p7_mpchandrop_resnet_se_sd0.50 >> "$LOG" 2>&1
rc=$?; echo "[p7] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[p7] FAILED" | tee -a "$LOG"; exit $rc; }
"$PY" -c "import pandas as pd;m=pd.read_csv('results_p7_mpchandrop_resnet_se_sd0.50/cnn_arch_summary.csv');print('[p7] mean F1',round(float(m['f1_macro_mean'][0]),4),'n',int(m['n'][0]))" | tee -a "$LOG"
echo "==== P7 END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
