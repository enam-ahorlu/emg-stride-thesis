#!/usr/bin/env bash
# EXPERIMENT_PLAN_FILTER.md Phase 2 -- 4 arms x 2 models, cheap refit
# (--reuse-params-dir, --save-proba, no grid search), one model at a time,
# guard-wrapped, per sept-2026-wave1 lessons. Stops immediately if arm A
# does not reproduce 0.7767/0.7732 within 0.003 -- everything else is a
# delta against arm A.
set -x
cd "C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code"
PY=".venv/Scripts/python.exe"
PYABS='C:\Users\enama\OneDrive\Desktop\Documents\MSc CS\FInal Project\06_Code\.venv\Scripts\python.exe'

"$PY" run_filter_phase2_setup.py || { echo "SETUP_FAILED"; exit 1; }

FEAT_DIR="features_out_filter"
declare -A STEM
STEM[A]="freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbFalse_ceFalse_features_ext"
STEM[B]="freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbTrue_ceFalse_features_ext"
STEM[C]="freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbFalse_ceTrue_features_ext"
STEM[D]="freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbTrue_ceTrue_features_ext"

fold_count () { local f="$1"; if [ -f "$f" ]; then echo $(($(wc -l < "$f") - 1)); else echo 0; fi; }
log () { echo "[$(date +%H:%M:%S)] $*"; }

GATE_LOG="results_filter_causal/phase2_gate.log"
: > "$GATE_LOG"

for ARM in A B C D; do
  OUTDIR="results_filter_causal/$ARM"
  mkdir -p "$OUTDIR/proba"
  STEMV="${STEM[$ARM]}"
  FEAT="$FEAT_DIR/${STEMV}.npz"
  META="$FEAT_DIR/${STEMV%_features_ext}_features_meta.csv"
  REUSE="results_filter_causal/reuse_params_$ARM"

  for MODEL in SVM RF; do
    CSV="$OUTDIR/${STEMV}__${MODEL}_nested_loso_subjectwise.csv"
    n=$(fold_count "$CSV")
    if [ "$n" -ge 40 ]; then
      log "arm $ARM / $MODEL already complete (40/40), skipping"
      continue
    fi
    log "arm $ARM / $MODEL: starting (checkpoint $n/40)"
    "$PY" run_with_memory_guard.py --max-mem-percent 92 --min-free-gb 1.2 --max-restarts 30 -- \
      "$PYABS" train_classical_loso.py \
        --features "$FEAT" --meta "$META" \
        --out "$OUTDIR" --models "$MODEL" --norm-mode per_subject \
        --cv-scheme loso --inner-splits 5 --n-jobs 1 --rf-n-jobs 1 --seed 42 \
        --reuse-params-dir "$REUSE" --save-proba --proba-out "$OUTDIR/proba" \
        --save-preds --flush-preds --resume
    rc=$?
    n2=$(fold_count "$CSV")
    log "arm $ARM / $MODEL: guard exit=$rc, checkpoint -> $n2/40"
    if [ "$n2" -lt 40 ]; then
      log "arm $ARM / $MODEL: FAILED to reach 40/40 -- stopping the whole Phase 2 chain"
      echo "PHASE2_FAILED arm=$ARM model=$MODEL" >> "$GATE_LOG"
      exit 1
    fi
  done

  if [ "$ARM" == "A" ]; then
    # Phase 2 GATE: arm A must reproduce 0.7767 (SVM) / 0.7732 (RF) within 0.003
    "$PY" -c "
import pandas as pd
svm = pd.read_csv('results_filter_causal/A/${STEM[A]}__SVM_nested_loso_summary.csv')['f1_macro_mean'].iloc[0]
rf  = pd.read_csv('results_filter_causal/A/${STEM[A]}__RF_nested_loso_summary.csv')['f1_macro_mean'].iloc[0]
ok = abs(svm - 0.7767) <= 0.003 and abs(rf - 0.7732) <= 0.003
print(f'arm A reproduction: SVM={svm:.4f} (pub 0.7767) RF={rf:.4f} (pub 0.7732)  {\"PASS\" if ok else \"FAIL\"}')
import sys
sys.exit(0 if ok else 1)
" | tee -a "$GATE_LOG"
    if [ "${PIPESTATUS[0]}" != "0" ]; then
      log "PHASE 2 GATE FAILED on arm A reproduction -- stopping before arms B/C/D are trusted."
      echo "PHASE2_GATE_FAILED" >> "$GATE_LOG"
      exit 1
    fi
  fi
done

log "PHASE2_ALL_COMPLETE"
echo "PHASE2_ALL_COMPLETE" >> "$GATE_LOG"
