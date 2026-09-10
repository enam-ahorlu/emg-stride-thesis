#!/usr/bin/env bash
# W-1 Stage 4 orchestration. Runs unattended; safe to lose the chat connection.
set -u
cd "$(dirname "$0")"
PY="C:/Users/enama/OneDrive/Desktop/Documents/MSc CS/FInal Project/06_Code/.venv/Scripts/python.exe"
LOG=_run_logs/stage4_ladder.log
echo "==== STAGE 4 START $(date -u +%FT%TZ) ====" | tee -a "$LOG"

# 0. syntax check the two patched modules
"$PY" -c "import ast; [ast.parse(open(f).read()) for f in ('analyze_between_subject_variance.py','run_alignment_ladder_loso.py')]; print('syntax OK')" 2>&1 | tee -a "$LOG" || { echo "SYNTAX FAIL - abort" | tee -a "$LOG"; exit 1; }

# 1. re-run the 250 ms ladder gate with NO env override and NO --no-gate.
#    The existing results_alignment_ladder_loso/ has all 5 rungs x 40 subjects,
#    so --resume skips compute and this just re-summarises + re-checks the gate.
#    Confirms the FEAT/META env-var patch changed nothing.
echo "---- [1] 250 ms re-gate (patch inertness) $(date -u +%FT%TZ) ----" | tee -a "$LOG"
unset LADDER_FEAT LADDER_META
"$PY" -u run_alignment_ladder_loso.py --out results_alignment_ladder_loso --rungs 3,0 \
      --inner-splits 5 --seed 42 --resume 2>&1 | tee -a "$LOG"
if ! grep -q "\[GATE rung 3\].*PASS" "$LOG"; then
    echo "==== 250 ms RE-GATE DID NOT PASS - STOPPING, not running 400 ms ====" | tee -a "$LOG"
    exit 2
fi
echo "---- [1] 250 ms re-gate PASSED ----" | tee -a "$LOG"

# 2. 400 ms ladder. LADDER_FEAT/LADDER_META point analyze_between_subject_variance
#    at the 400 ms Freq-72 set; --no-gate skips the hardcoded 250 ms gate values
#    but keeps the rung-4 vs rung-3 over-alignment falsification check.
echo "---- [2] 400 ms ladder $(date -u +%FT%TZ) ----" | tee -a "$LOG"
export LADDER_FEAT="features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_ext.npz"
export LADDER_META="features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_meta.csv"
"$PY" -u run_alignment_ladder_loso.py --out results_win400_ladder --rungs 3,0,1,2,4 \
      --inner-splits 5 --seed 42 --resume --no-gate 2>&1 | tee -a "$LOG"
rc=$?
echo "==== STAGE 4 END $(date -u +%FT%TZ) rc=$rc ====" | tee -a "$LOG"
exit $rc
