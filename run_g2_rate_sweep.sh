#!/usr/bin/env bash
# W-2 Stage G2, the dropout-rate sweep. Reactivated 2 September 2026.
# Pre-registration is EXPERIMENT_PLAN_CHANNEL_DROPOUT.md §4.1 and §4.3, unaltered.
# --arch resnet_se here, NOT resnet: the operating-point question concerns the
# model of record, which is the SE variant. See the plan status header.
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/g2_rate_sweep.log
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --norm-mode per_subject --resume"

echo "==== G2 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# p = 0.2 FIRST: it is the era-internal comparator, and if it fails the 1.5 pp
# reproduction gate against the published 0.8395 there is no point running the rest.
for P in 0.2 0.1 0.3 0.5; do
  OUT="results_cd_rate_p${P}"
  echo "--- [p=$P] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se \
    --augmentation chandrop --aug-chandrop-p $P $BASE \
    --out "$OUT" --instrument "$OUT/instr" >> "$LOG" 2>&1
  rc=$?; echo "[p=$P] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[p=$P] FAILED, stopping" | tee -a "$LOG"; exit $rc; }

  if [ "$P" = "0.2" ]; then
    "$PY" -c "
import pandas as pd, sys
m = pd.read_csv('results_cd_rate_p0.2/cnn_arch_summary.csv')['f1_macro_mean'][0]
d = (m - 0.8395) * 100
print(f'[gate] fresh p=0.2 {m:.4f} vs published 0.8395: {d:+.2f} pp')
print('[gate] PASS' if abs(d) <= 1.5 else '[gate] FAIL, exceeds the 1.5 pp R-1 gate')
sys.exit(0 if abs(d) <= 1.5 else 2)
" | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "[gate] FAILED, stopping before the sweep" | tee -a "$LOG"; exit $rc; }
  fi
done

echo "--- analysis $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u g2_rate_stats.py 2>&1 | tee -a "$LOG"
echo "==== G2 END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
