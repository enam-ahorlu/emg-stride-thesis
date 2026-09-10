#!/usr/bin/env bash
# P-5 and P-6 core runs (EXPERIMENT_PLAN_CD_PARITY.md sections 5A and 5B).
# Inertness (p5p6_inertness.py) and the P-6 multiplier gate (p6_multiplier_gate.py)
# have already PASSED. All arms: resnet_se, per-subject norm, 250 ms, seed 42,
# every other flag default. --resume so an interruption continues.
#
#   P-5   subset          -> results_p5_subset_resnet_se
#   P-6   gainjitter 0.40  -> results_p6_gainjitter_resnet_se_sd0.40
#   P-6   mpchandrop 0.40  -> results_p6_mpchandrop_resnet_se_sd0.40
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/p5_p6.log
mkdir -p _run_logs
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --norm-mode per_subject --resume"

echo "==== P5/P6 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

run () {
  local tag="$1" aug="$2" out="$3"; shift 3
  echo "--- [$tag] $aug -> $out  $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$PY" -u run_cnn_arch_loso.py --npz "$NPZ" --meta "$META" --arch resnet_se \
    --augmentation "$aug" $BASE --out "$out" "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "[$tag] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[$tag] FAILED, stopping" | tee -a "$LOG"; exit $rc; }
  "$PY" -c "import pandas as pd;m=pd.read_csv('$out/cnn_arch_summary.csv');print('[$tag] mean F1',round(float(m['f1_macro_mean'][0]),4),'n',int(m['n'][0]))" | tee -a "$LOG"
}

run P5-subset       subset     results_p5_subset_resnet_se
run P6-gainjitter04 gainjitter results_p6_gainjitter_resnet_se_sd0.40 --aug-gain-sd 0.4
run P6-mpchandrop04 mpchandrop results_p6_mpchandrop_resnet_se_sd0.40 --aug-gain-sd 0.4

echo "==== P5/P6 CORE END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
