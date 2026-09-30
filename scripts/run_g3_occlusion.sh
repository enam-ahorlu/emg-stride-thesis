#!/usr/bin/env bash
# W-2 Stage G3 (re-scoped): the matched no-augmentation instrumented baseline.
# The channel-dropout half already exists in results/cd_resnet_nose_chandrop/instr.
# See EXPERIMENT_PLAN_G3_OCCLUSION.md. Note --arch resnet, NOT resnet_se: G1 was
# SE-free and the baseline must match it.
set -u
cd "$(dirname "$0")/.."
PY="C:/dev/emg-stride-thesis/.venv/Scripts/python.exe"
LOG=logs/g3_occlusion.log
NPZ=data/windows/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=data/features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --norm-mode per_subject --resume"

echo "==== G3 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# --- 0. smoke: one fold, confirms the instrument path writes occlusion.csv ---
rm -rf results/smoke/g3_smoke
echo "--- [smoke] heldout 1 $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u src/run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet --augmentation none \
  $BASE --heldout 1 --out results/smoke/g3_smoke --instrument results/smoke/g3_smoke/instr >> "$LOG" 2>&1
rc=$?; echo "[smoke] exit $rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[smoke] FAILED, stopping" | tee -a "$LOG"; exit $rc; }
[ -s results/smoke/g3_smoke/instr/occlusion.csv ] || { echo "[smoke] no occlusion.csv, stopping" | tee -a "$LOG"; exit 1; }

# --- 1. the instrumented no-augmentation baseline ---
echo "--- [1] resnet no-aug, instrumented $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u src/run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet --augmentation none \
  $BASE --out results/g3_noaug_instr --instrument results/g3_noaug_instr/instr >> "$LOG" 2>&1
rc=$?; echo "[1] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[1] FAILED, stopping" | tee -a "$LOG"; exit $rc; }

# --- 2. reproduction gate: within 1.5 pp of the 1 September baseline ---
"$PY" -c "
import pandas as pd, sys
a = pd.read_csv('results/g3_noaug_instr/cnn_arch_summary.csv')['f1_macro_mean'][0]
b = pd.read_csv('results/cd_resnet_noaug_repro/cnn_arch_summary.csv')['f1_macro_mean'][0]
d = (a - b) * 100
print(f'[gate] instrumented baseline {a:.4f} vs results/cd_resnet_noaug_repro {b:.4f}: {d:+.2f} pp')
print('[gate] PASS' if abs(d) <= 1.5 else '[gate] FAIL, exceeds the 1.5 pp R-1 gate; stop and report')
sys.exit(0 if abs(d) <= 1.5 else 2)
" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ $rc -ne 0 ] && { echo "[gate] FAILED" | tee -a "$LOG"; exit $rc; }

# --- 3. analysis, no GPU ---
echo "--- [2] occlusion analysis $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u src/g3_occlusion_stats.py 2>&1 | tee -a "$LOG"

echo "==== G3 END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
