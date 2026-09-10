#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/b8_rest.log
mkdir -p _run_logs
echo "==== B8 REST START $(date -u +%FT%TZ) ====" | tee -a "$LOG"
FO=features_out
run () { # tag npz meta win models
  echo "--- [$1] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$PY" -u b8_movement_blocked_sd.py --features "$2" --meta "$3" --window-ms "$4" --models "$5" --tag "$1" --out results_b8_sd >> "$LOG" 2>&1
  echo "[$1] exit $? $(date -u +%FT%TZ)" | tee -a "$LOG"
}
run base_w250 $FO/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_base.npz $FO/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv 250 SVM,RF
run ext_w250  $FO/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz  $FO/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv 250 SVM,RF
run base_w150 $FO/windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_base.npz $FO/windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_meta.csv 150 SVM,RF
run ext_w150  $FO/windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_ext.npz  $FO/windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_meta.csv 150 SVM,RF
run freq72_w150 $FO/freq_windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_ext.npz $FO/freq_windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_meta.csv 150 SVM,RF,LDA
echo "==== B8 REST END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
