#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/s1_classical.log
mkdir -p _run_logs
F=features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_ext.npz
M=features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_meta.csv
echo "==== S1 CLASSICAL START $(date -u +%FT%TZ) ====" | tee -a "$LOG"
for NORM in per_subject global; do
  OUT="results_aonly_$( [ "$NORM" = per_subject ] && echo persubj || echo global )"
  echo "--- [$NORM -> $OUT] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$PY" -u train_classical_loso.py --features "$F" --meta "$M" --out "$OUT" \
    --models SVM,RF --norm-mode "$NORM" --inner-splits 5 --cv-scheme loso \
    --n-jobs 1 --rf-n-jobs 4 --seed 42 --save-preds --flush-preds --resume >> "$LOG" 2>&1
  echo "[$NORM] exit $? $(date -u +%FT%TZ)" | tee -a "$LOG"
done
echo "==== S1 CLASSICAL END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
