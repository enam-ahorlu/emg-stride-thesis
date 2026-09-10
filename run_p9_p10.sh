#!/usr/bin/env bash
# P-9 (FOUR arms, both backbones, EXPERIMENT_PLAN_LOCUS.md section 3.4 amended)
# then P-10 (3 runs, resnet_se mean-preserving upward sweep, section 4).
# GPU serializes; no memory guard (house convention for CNN). All --resume.
#   A1 resnet     none      -> results_p9_atten_resnet_noaug        (reproduction of g3_noaug_instr)
#   A2 resnet     chandrop  -> results_p9_atten_resnet_chandrop     (reproduction of cd_resnet_nose_chandrop)
#   A3 resnet_se  none      -> results_p9_atten_resnet_se_noaug     (first measurement)
#   A4 resnet_se  chandrop  -> results_p9_atten_resnet_se_chandrop  (first measurement, model of record)
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/p9_p10.log
mkdir -p _run_logs
NPZ=windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
BASE="--epochs 40 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15 --seed 42 --norm-mode per_subject --resume"

echo "==== P9(4-arm)/P10 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# ---------- P-9 smoke + section 3.2 alpha=0 gate (resnet_se this time) ----------
rm -rf _p9_smoke
echo "--- [P9 smoke] resnet_se none heldout 1 $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se --augmentation none \
  $BASE --heldout 1 --out _p9_smoke --instrument _p9_smoke/instr >> "$LOG" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "[P9 smoke] FAILED $rc" | tee -a "$LOG"; exit $rc; }
"$PY" -c "
import pandas as pd, sys
a=pd.read_csv('_p9_smoke/instr/attenuation.csv'); o=pd.read_csv('_p9_smoke/instr/occlusion.csv')
a0=a[a.alpha==0.0].set_index('channel')['drop_pp']; oo=o.set_index('channel')['drop_pp']
ok=bool((a0.reindex(oo.index)-oo).abs().max() < 1e-9)
print('[P9 smoke gate] alpha=0 == occlusion (resnet_se):', ok)
sys.exit(0 if ok else 2)
" | tee -a "$LOG"
[ ${PIPESTATUS[0]} -ne 0 ] && { echo "[P9 smoke gate] FAILED" | tee -a "$LOG"; exit 2; }

p9run () {  # $1 tag  $2 arch  $3 augflags  $4 outdir
  echo "--- [P9 $1] $2 $3 $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch $2 $3 \
    $BASE --out "$4" --instrument "$4/instr" >> "$LOG" 2>&1
  local rc=$?; echo "[P9 $1] exit $rc $(date -u +%FT%TZ)" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[P9 $1] FAILED" | tee -a "$LOG"; exit $rc; }
  "$PY" -c "import pandas as pd;m=pd.read_csv('$4/cnn_arch_summary.csv');print('[P9 $1] mean F1',round(float(m['f1_macro_mean'][0]),4),'n',int(m['n'][0]))" | tee -a "$LOG"
}
p9run A1 resnet    "--augmentation none"                       results_p9_atten_resnet_noaug
p9run A2 resnet    "--augmentation chandrop --aug-chandrop-p 0.2" results_p9_atten_resnet_chandrop
p9run A3 resnet_se "--augmentation none"                       results_p9_atten_resnet_se_noaug
p9run A4 resnet_se "--augmentation chandrop --aug-chandrop-p 0.2" results_p9_atten_resnet_se_chandrop

echo "--- [P9 analysis] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u p9_attenuation_stats.py 2>&1 | tee -a "$LOG"

# ---------- P-10: multiplier gates then 3 runs ----------
echo "--- [P10 multiplier gates] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u p6_multiplier_gate.py 0.60 0.80 1.00 2>&1 | tee -a "$LOG"
[ ${PIPESTATUS[0]} -ne 0 ] && { echo "[P10 gate] FAILED" | tee -a "$LOG"; exit 2; }

p10run () {
  local sd=$1 out="results_p10_mpchandrop_resnet_se_sd$1"
  echo "--- [P10 sd=$sd] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
  "$PY" -u run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se \
    --augmentation mpchandrop --aug-gain-sd $sd $BASE --out "$out" >> "$LOG" 2>&1
  local rc=$?; echo "[P10 sd=$sd] exit $rc $(date -u +%FT%TZ)" | tee -a "$LOG"
  [ $rc -ne 0 ] && { echo "[P10 sd=$sd] FAILED" | tee -a "$LOG"; exit $rc; }
  "$PY" -c "import pandas as pd;m=pd.read_csv('$out/cnn_arch_summary.csv');print('[P10 sd=$sd] mean F1',round(float(m['f1_macro_mean'][0]),4),'n',int(m['n'][0]))" | tee -a "$LOG"
}
p10run 0.60
p10run 0.80
p10run 1.00
echo "--- [P10 analysis] $(date -u +%FT%TZ) ---" | tee -a "$LOG"
"$PY" -u p10_ceiling_stats.py 2>&1 | tee -a "$LOG"

echo "==== P9/P10 END $(date -u +%FT%TZ) ====" | tee -a "$LOG"
