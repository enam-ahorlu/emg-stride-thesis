#!/usr/bin/env bash
# S-1 step 4: deep arm. resnet_se + channel dropout, per-subject norm, 40-fold LOSO,
# on the active-only STDUP windows. Confirms the H outcome (STDUP stays top) on the
# model-of-record family. GPU serializes; no memory guard (house convention for CNN);
# --resume for kill-safety. Exact command line is also dumped to the out folder's
# run_config.json by run_cnn_arch_loso.py.
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/s1_deeparm.log
mkdir -p _run_logs
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly.npz
META=windows_WAK_UPS_DNS_STDUP_v1_meta_w250_ov50_conf60_Aonly.csv
OUT=results_aonly_resnet_se_cd_persubj

echo "==== S1 DEEPARM START $(date -u +%FT%TZ) ====" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz "$NPZ" --meta "$META" --arch resnet_se \
  --augmentation chandrop --aug-chandrop-p 0.2 \
  --epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 \
  --norm-mode per_subject --resume --out "$OUT" \
  --save-proba "$OUT/proba" --model-tag RESNET_SE_AONLY >> "$LOG" 2>&1
rc=$?
echo "[S1 deeparm] exit $rc $(date -u +%FT%TZ)" | tee -a "$LOG"
if [ $rc -eq 0 ]; then
  "$PY" -c "import pandas as pd; m=pd.read_csv('$OUT/cnn_arch_summary.csv'); print('[S1 deeparm] mean F1', round(float(m['f1_macro_mean'][0]),4), 'n', int(m['n'][0]))" | tee -a "$LOG"
fi
echo "==== S1 DEEPARM END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
