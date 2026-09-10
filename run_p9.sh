#!/usr/bin/env bash
# P-9 (EXPERIMENT_PLAN_LOCUS.md section 3). Graded channel attenuation, transfer
# question only (per Enam's amendment: the per-subject coupling half is retired
# because P-8 showed it needs ~150 subjects). Two 40-fold runs on --arch resnet
# (SE-free), per-subject norm, 250 ms, seed 42, with the extended instrumentation
# (occlusion.csv + attenuation.csv). Their numbers live only with each other and
# must NOT be cross-compared with the Section 4.8.2 occlusion.csv (run-to-run SD).
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/p9.log
mkdir -p _run_logs
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --norm-mode per_subject --resume"

echo "==== P9 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# --- smoke: 1 fold, confirm attenuation.csv is written and alpha=0 == occlusion ---
rm -rf _p9_smoke
echo "--- [smoke] resnet none, heldout 1 $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet --augmentation none \
  $BASE --heldout 1 --out _p9_smoke --instrument _p9_smoke/instr >> "$LOG" 2>&1
rc=$?; echo "[smoke] exit $rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[smoke] FAILED" | tee -a "$LOG"; exit $rc; }
"$PY" -c "
import pandas as pd, sys
a = pd.read_csv('_p9_smoke/instr/attenuation.csv')
o = pd.read_csv('_p9_smoke/instr/occlusion.csv')
a0 = a[a.alpha==0.0].set_index('channel')['drop_pp'].round(9)
oo = o.set_index('channel')['drop_pp'].round(9)
ok = a0.reindex(oo.index).equals(oo)
print('[smoke gate] attenuation alpha=0 reproduces occlusion row-for-row:', ok)
print('  max abs diff:', float((a0.reindex(oo.index)-oo).abs().max()))
sys.exit(0 if ok else 2)
" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ $rc -ne 0 ] && { echo "[smoke gate] FAILED" | tee -a "$LOG"; exit $rc; }

run () {
  local name=$1 augf=$2
  local out="results_p9_atten_$name"
  echo "--- [$name] $augf $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet $augf \
    $BASE --out "$out" --instrument "$out/instr" >> "$LOG" 2>&1
  local rc=$?; echo "[$name] exit $rc at $(date -u +%FT%TZ)" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[$name] FAILED" | tee -a "$LOG"; exit $rc; }
  "$PY" -c "import pandas as pd;m=pd.read_csv('$out/cnn_arch_summary.csv');print('[$name] mean F1',round(float(m['f1_macro_mean'][0]),4),'n',int(m['n'][0]))" | tee -a "$LOG"
}

run noaug    "--augmentation none"
run chandrop "--augmentation chandrop --aug-chandrop-p 0.2"

echo "--- [analysis] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u p9_attenuation_stats.py 2>&1 | tee -a "$LOG"
echo "==== P9 END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
