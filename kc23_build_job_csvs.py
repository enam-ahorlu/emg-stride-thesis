#!/usr/bin/env python3
"""
kc23_build_job_csvs.py
========================
RUN_ORDER_KC23.md Section 2/3 -- builds kc23_jobs_gpu.csv and
kc23_jobs_cpu.csv, one row per individual training/analysis run implied by
items 6 to 15 (including 11b/11c for KC-D6) and C-a to C-e, in the
dispatcher's order, with depends_on and gate_script filled.

This is a GENERATOR, not a hand-maintained CSV, because the arm tables in the
three EXPERIMENT_PLAN_KC23_*.md files are long and this makes the mapping
from plan table to job row explicit and auditable in code rather than in ~250
hand-typed CSV lines. Run it once to (re)write the two CSVs:

  python kc23_build_job_csvs.py

Every gate_script named below is the file name the corresponding
EXPERIMENT_PLAN_KC23_*.md section says the stage's stats live in (e.g.
kc23_c2_whitening_stats.py). NONE of these stats scripts exist yet as of this
Phase-1 session -- writing them is downstream work, gated on the runs they
analyse landing first, per the dispatcher's own phase order. kc23_queue.py
treats a missing gate_script as "not yet implemented" and continues rather
than crashing, and this is flagged explicitly in the run-back report, not
silently assumed away.

KC-D6 Stage 2 (item 11c) is NOT enumerated here: RUN_ORDER_KC23.md item 11c
says its jobs are "generated only after the D6 manipulation gate decides
which families pass" (Stage 1, item 11b). Guessing Stage 2's knob values
before Stage 1's gate result exists would defeat the point of staging it.
A single placeholder row records the dependency instead; a follow-up
generator (kc23_d6_stage2_job_gen.py, not yet written) appends the real rows
once 11b's per-family manipulation-gate letters are known.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Absolute, PRE-QUOTED path: kc23_queue.py launches jobs with shell=True
# (cmd.exe on Windows), which (a) does not resolve a bare relative
# ".venv/Scripts/python.exe" as a runnable command from cwd the way a POSIX
# shell or this repo's own interactive git-bash sessions do, and (b) word-
# splits an UNQUOTED path at the first space -- and this project's path
# ("...MSc CS\FInal Project\06_Code") has two. Both failure modes were
# confirmed empirically via the queue's own --heldout 1 smoke test before
# this fix. PY therefore already carries its own double quotes; every
# f-string below that embeds {PY} gets a correctly shell-quoted executable
# for free.
PY = f'"{ROOT / ".venv" / "Scripts" / "python.exe"}"'

NPZ_250 = "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"
META_250 = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
NPZ_400 = "windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR.npz"
META_400 = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_meta.csv"
FEAT_250_FREQ = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META_250_FEAT = META_250
FEAT_250_BASE = "features_out/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_base.npz"
FEAT_400_FREQ = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w400_ov50_conf60_AorR_features_ext.npz"
META_400_FEAT = META_400

D1_TIER_A_SEEDS = [42, 7, 123, 1001]
D1_TIER_B_SEEDS = [42, 7, 123]
D3_D4_SEEDS = [42, 7, 123]
D5_SEEDS = [42, 7, 123, 1001, 2026]

gpu_rows = []
cpu_rows = []

# ============================================================================
# Bridging checkpoint jobs: the Phase-0/1 code changes and their inertness
# proofs were already completed and verified in this session (KC23_VERIFICATIONS.md,
# results_kc23_d0_capture/, results_kc23_c1_nested_selection/, and the
# per-stage inertness runs under results_kc23_c{2,3,5}_*inertness* /
# results_kc23_s2_adapter_inertness). These jobs exist only so the dependency
# graph below is self-contained and resolvable by kc23_queue.py: each just
# confirms the Phase-1 artifact directory is present (near-instant, exits 0),
# so it shows "done" the first time the queue runs.
# ============================================================================
def _checkpoint(job_id, check_dir, expected_outputs):
    # expected_outputs (added 2026-09-25) names a specific file the Phase-1 proof left in check_dir, so that "done"
    # rests on that file existing, not on the directory being non-empty.
    cpu_rows.append({"job_id": job_id, "stage": "PHASE1-CHECKPOINT", "seed": "",
                     "command": f'{PY} -c "import sys,pathlib; d=pathlib.Path(r\'{check_dir}\'); '
                                f'sys.exit(0 if d.exists() and any(d.iterdir()) else 1)"',
                     "out_dir": check_dir, "depends_on": "", "gate_script": "",
                     "expected_outputs": expected_outputs})

_checkpoint("d0_smoke_gate", "results_kc23_d0_capture/after_cpu",
            "augbatch__run_cnn_arch_loso_augment_batch__none.npy|EXISTS")
_checkpoint("code_c2_inertness", "results_kc23_c2_inertness_check", "ladder_loso_3_SVM_subjectwise.csv|EXISTS")
_checkpoint("code_c3_inertness", "results_kc23_c3_after_persubj", "*SVM_nested_loso_subjectwise.csv|EXISTS")
_checkpoint("code_c5_inertness", "results_kc23_c5_inertness_check", "b8_base_w250_subjectwise.csv|EXISTS")


def gpu(job_id, stage, seed, command, out_dir, depends_on=(), gate_script="", expected_outputs=""):
    gpu_rows.append({"job_id": job_id, "stage": stage, "seed": str(seed) if seed is not None else "",
                     "command": command, "out_dir": out_dir,
                     "depends_on": ";".join(depends_on), "gate_script": gate_script,
                     "expected_outputs": expected_outputs})


def cpu(job_id, stage, seed, command, out_dir, depends_on=(), gate_script="", expected_outputs=""):
    cpu_rows.append({"job_id": job_id, "stage": stage, "seed": str(seed) if seed is not None else "",
                     "command": command, "out_dir": out_dir,
                     "depends_on": ";".join(depends_on), "gate_script": gate_script,
                     "expected_outputs": expected_outputs})


SIAT_N = 40    # SIAT-LLMD subject count
ENABL3S_N = 10  # ENABL3S subject count
EO_CNN_ARCH_SIAT = f"cnn_arch_subjectwise.csv|{SIAT_N}"
EO_CNN_ARCH_ENABL3S = f"cnn_arch_subjectwise.csv|{ENABL3S_N}"
EO_ADABN_SIAT = f"adabn_subjectwise.csv|{SIAT_N}"
EO_NESTED_LOSO_SIAT = f"*_nested_loso_subjectwise.csv|{SIAT_N}"
EO_NESTED_LOSO_ENABL3S = f"*_nested_loso_subjectwise.csv|{ENABL3S_N}"
EO_LDA_SIAT = f"lda_subjectwise.csv|{SIAT_N}"
EO_LDA_ENABL3S = f"lda_subjectwise.csv|{ENABL3S_N}"
EO_ADABN_SIAT = f"adabn_subjectwise.csv|{SIAT_N}"
EO_DEEP_CORAL_SIAT = f"deep_coral_subjectwise.csv|{SIAT_N}"
EO_ADV_SIAT = f"adv_subjectwise.csv|{SIAT_N}"
EO_SIMPLECNN_LOSO_SIAT = f"per_subject_metrics_cnn_loso.csv|{SIAT_N}"
EO_D1_ARM = {  # arm -> expected_outputs, used both at seed 42 and seeds 7/123/1001
    "r1": EO_CNN_ARCH_SIAT, "r2": EO_CNN_ARCH_SIAT, "r3": EO_CNN_ARCH_SIAT, "r4": EO_CNN_ARCH_SIAT,
    "r5": EO_CNN_ARCH_SIAT, "r6": EO_CNN_ARCH_SIAT, "r7": EO_CNN_ARCH_SIAT, "r12": EO_CNN_ARCH_SIAT,
    "r8": EO_SIMPLECNN_LOSO_SIAT, "r9": EO_SIMPLECNN_LOSO_SIAT,
    "r10": EO_ADABN_SIAT, "r11": EO_DEEP_CORAL_SIAT,
    "r13": EO_CNN_ARCH_SIAT, "r14": EO_CNN_ARCH_SIAT, "r15": EO_CNN_ARCH_SIAT,
    "r16": EO_CNN_ARCH_SIAT, "r17": EO_CNN_ARCH_SIAT,
}
EO_VERDICT_LETTER = "*_VERDICT.md|LETTER"


def instr(out_dir):
    return f"{out_dir}/instr"


# ============================================================================
# #6 (Phase 2, GPU): KC-D1 Tier A, seed 42 re-run (R1-R12) -- the reproduction gate
# ============================================================================
D1_REPRO_GATE = "kc23_d1_replicate_stats.py"  # both the reproduction gate (D1.4) and the 17-contrast/headline gates live in one script
d1_s42_ids = []

def _d1_arm(arm, out, cmd):
    jid = f"d1_{arm}_s42"
    gpu(jid, "D1", 42, cmd, out, depends_on=("d0_smoke_gate",), expected_outputs=EO_D1_ARM[arm])
    d1_s42_ids.append(jid)
    return jid

_d1_arm("r1", "results_kc23_d1_r1_s42",
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se --augmentation none '
       f'--instrument {instr("results_kc23_d1_r1_s42")} --seed 42 --out results_kc23_d1_r1_s42 --resume')
_d1_arm("r2", "results_kc23_d1_r2_s42",
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se --augmentation chandrop '
       f'--aug-chandrop-p 0.2 --save-proba results_kc23_d1_r2_s42/proba --model-tag RESNET_SE_CD '
       f'--instrument {instr("results_kc23_d1_r2_s42")} --seed 42 --out results_kc23_d1_r2_s42 --resume')
_d1_arm("r3", "results_kc23_d1_r3_s42",
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se --augmentation gainjitter '
       f'--aug-gain-sd 0.40 --instrument {instr("results_kc23_d1_r3_s42")} --seed 42 --out results_kc23_d1_r3_s42 --resume')
_d1_arm("r4", "results_kc23_d1_r4_s42",
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet --augmentation none '
       f'--instrument {instr("results_kc23_d1_r4_s42")} --seed 42 --out results_kc23_d1_r4_s42 --resume')
_d1_arm("r5", "results_kc23_d1_r5_s42",
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet --augmentation chandrop '
       f'--instrument {instr("results_kc23_d1_r5_s42")} --seed 42 --out results_kc23_d1_r5_s42 --resume')
_d1_arm("r6", "results_kc23_d1_r6_s42",
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_nores --augmentation none '
       f'--instrument {instr("results_kc23_d1_r6_s42")} --seed 42 --out results_kc23_d1_r6_s42 --resume')
_d1_arm("r7", "results_kc23_d1_r7_s42",
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_nores --augmentation chandrop '
       f'--instrument {instr("results_kc23_d1_r7_s42")} --seed 42 --out results_kc23_d1_r7_s42 --resume')
# R8/R9: SimpleEMGCNN, the exact published script (train_cnn_loso.py has no --instrument;
# these feed only C15, which needs no instrumentation).
_d1_arm("r8", "results_kc23_d1_r8_s42",
       f'{PY} train_cnn_loso.py --npz {NPZ_250} --meta {META_250} --norm-mode per_subject '
       f'--seed 42 --out results_kc23_d1_r8_s42 --resume')
_d1_arm("r9", "results_kc23_d1_r9_s42",
       f'{PY} train_cnn_loso.py --npz {NPZ_250} --meta {META_250} --norm-mode per_subject '
       f'--augment chandrop --aug-chandrop-p 0.2 --seed 42 --out results_kc23_d1_r9_s42 --resume')
_d1_arm("r10", "results_kc23_d1_r10_s42",
       f'{PY} run_adabn_cnn_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se --augmentation chandrop '
       f'--epochs 40 --seed 42 --out results_kc23_d1_r10_s42 --resume')
_d1_arm("r11", "results_kc23_d1_r11_s42",
       f'{PY} run_deep_coral_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
       f'--augmentation chandrop --coral-lambda 30 --batch 256 --seed 42 --out results_kc23_d1_r11_s42 --resume')
_d1_arm("r12", "results_kc23_d1_r12_s42",
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_400} --meta {META_400} --arch resnet_se --augmentation chandrop '
       f'--aug-chandrop-p 0.2 --instrument {instr("results_kc23_d1_r12_s42")} --seed 42 --out results_kc23_d1_r12_s42 --resume')
# gate on the last of the seed-42 arms only (R2, the slowest/most-depended-on)
gpu_rows[-1]["gate_script"] = ""  # R12 has no gate itself
# NOTE: the D1.4 reproduction gate does NOT attach to d1_r2_s42 itself -- that
# fires as soon as R2 finishes, pointed at R2's own private out_dir, before R1
# and R10(pre) exist and before any aggregate input csv does either. Found
# live 2026-09-24 (kc23_d1_replicate_stats.py --out results_kc23_d1_r2_s42
# silently no-ops: none of its four expected input csvs live there). The real
# reproduction-gate job is added below, once R1/R2/R10 for seed 42 are all in.

# ============================================================================
# #6b (Phase 2, CPU): KC-D1.4 reproduction gate, once R1/R2/R10 (seed 42) exist
# ============================================================================
D1_AGGREGATE = "kc23_d1_aggregate.py"
cpu("d1_reproduction_check", "D1-REPRO", None,
   f'{PY} {D1_AGGREGATE} --root . --out results_kc23_d1_repro_check --require repro',
   "results_kc23_d1_repro_check", depends_on=("d1_r1_s42", "d1_r2_s42", "d1_r10_s42"),
   gate_script=D1_REPRO_GATE, expected_outputs=EO_VERDICT_LETTER)

# ============================================================================
# #7 (Phase 2, GPU+CPU): KC-S1 scripted buffer, 3 realizations (base training)
# ============================================================================
# The S1 stats gate (kc23_s1_scripted_stats.py) reads all three seed directories, so it cannot be the gate_script of
# any single s1_base row without firing before its siblings exist. It is wired as its own pseudo-row, s1_gate, below
# (the same pattern as d1_reproduction_check and d6_*_check). A stale note here, from before run_scripted_supervised.py
# implemented every arm, said the gate could not be wired; that stopped being true on 2026-09-24 and the gate was
# then never wired, so its D-S outcome never reached the halt protocol (found 2026-09-25).
EO_S1_SUBJECTWISE = f"s1_subjectwise.csv|{40 * 3 * 9}"  # 40 subjects x {5,10,25} x 9 arms
for seed in [42, 7, 123]:
    gpu(f"s1_base_s{seed}", "S1", seed,
       f'{PY} run_scripted_supervised.py --seed {seed} --out results_kc23_s1_scripted_s{seed} --resume',
       f"results_kc23_s1_scripted_s{seed}", depends_on=d1_s42_ids, gate_script="",
       expected_outputs=EO_S1_SUBJECTWISE)
# --report-only writes the full verdict (letter, K curve, S-ft against L1, determinism check) and exits 0; the queue
# then runs the same script again as the gate, so a D-S (exit 20) still reaches the halt protocol. Decision D-6b
# (25 September 2026) accepted D-S for the current data, and results_kc23_s1_gate/S1_VERDICT.md already exists with
# its letter, so this row is skipped as complete on the current seeds and only re-fires if they are re-run.
cpu("s1_gate", "S1-GATE", None,
   f'{PY} kc23_s1_scripted_stats.py --out results_kc23_s1_gate --report-only',
   "results_kc23_s1_gate", depends_on=("s1_base_s42", "s1_base_s7", "s1_base_s123"),
   gate_script="kc23_s1_scripted_stats.py", expected_outputs="S1_VERDICT.md|LETTER")

# ============================================================================
# #8 (Phase 2, GPU): KC-D5 ENABL3S deep, 5 realizations (E1/E2/E3)
# ============================================================================
NPZ_ENABL3S = "features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60.npz"
META_ENABL3S = "features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_meta.csv"
for seed in D5_SEEDS:
    gpu(f"d5_e1_s{seed}", "D5", seed,
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_ENABL3S} --meta {META_ENABL3S} --arch resnet_se '
       f'--augmentation none --instrument {instr(f"results_kc23_d5_e1_s{seed}")} --seed {seed} '
       f'--out results_kc23_d5_e1_s{seed} --resume', f"results_kc23_d5_e1_s{seed}", depends_on=d1_s42_ids,
       expected_outputs=EO_CNN_ARCH_ENABL3S)
    gpu(f"d5_e2_s{seed}", "D5", seed,
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_ENABL3S} --meta {META_ENABL3S} --arch resnet_se '
       f'--augmentation chandrop --aug-chandrop-p 0.2 --save-proba results_kc23_d5_e2_s{seed}/proba '
       f'--model-tag RESNET_SE_CD --instrument {instr(f"results_kc23_d5_e2_s{seed}")} --seed {seed} '
       f'--out results_kc23_d5_e2_s{seed} --resume', f"results_kc23_d5_e2_s{seed}", depends_on=d1_s42_ids,
       expected_outputs=EO_CNN_ARCH_ENABL3S)
    gpu(f"d5_e3_s{seed}", "D5", seed,
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_ENABL3S} --meta {META_ENABL3S} --arch resnet_se '
       f'--augmentation gainjitter --aug-gain-sd 0.40 --instrument {instr(f"results_kc23_d5_e3_s{seed}")} '
       f'--seed {seed} --out results_kc23_d5_e3_s{seed} --resume', f"results_kc23_d5_e3_s{seed}",
       depends_on=d1_s42_ids, gate_script="", expected_outputs=EO_CNN_ARCH_ENABL3S)

# The D5 gate used to sit on d5_e3_s2026, whose out_dir holds only that one run. The stats script read two CSVs that
# nothing produced, wrote a header-only D5_VERDICT.md and exited 0 (found 2026-09-25). It now runs as its own row
# after kc23_d5_aggregate.py builds those inputs from all 15 runs (decision D-6c: directions from the thesis on SIAT).
d5_run_ids = tuple(f"d5_e{e}_s{s}" for s in D5_SEEDS for e in (1, 2, 3))
cpu("d5_stats", "D5-STATS", None,
   f'{PY} kc23_d5_aggregate.py --root . --out results_kc23_d5_stats',
   "results_kc23_d5_stats", depends_on=d5_run_ids, gate_script="kc23_d5_replication_stats.py",
   expected_outputs="D5_VERDICT.md|LETTER")

# ============================================================================
# #9 (Phase 3, GPU): KC-D1 Tier A, seeds 7/123/1001
# ============================================================================
d1_arms_no89 = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r10", "r11", "r12"]  # r8/r9 use train_cnn_loso.py
d1_all_seed_ids = list(d1_s42_ids)
for seed in [7, 123, 1001]:
    for arm in d1_arms_no89:
        base = {"r1": ("resnet_se", "none", NPZ_250, META_250), "r2": ("resnet_se", "chandrop", NPZ_250, META_250),
               "r3": ("resnet_se", "gainjitter", NPZ_250, META_250), "r4": ("resnet", "none", NPZ_250, META_250),
               "r5": ("resnet", "chandrop", NPZ_250, META_250), "r6": ("resnet_nores", "none", NPZ_250, META_250),
               "r7": ("resnet_nores", "chandrop", NPZ_250, META_250), "r12": ("resnet_se", "chandrop", NPZ_400, META_400)}
        if arm in base:
            archv, augv, npzv, metav = base[arm]
            out = f"results_kc23_d1_{arm}_s{seed}"
            extra = ""
            if augv == "gainjitter":
                extra = " --aug-gain-sd 0.40"
            elif augv == "chandrop":
                extra = " --aug-chandrop-p 0.2"
            proba = f" --save-proba {out}/proba --model-tag RESNET_SE_CD" if arm == "r2" else ""
            cmdline = (f'{PY} run_cnn_arch_loso.py --npz {npzv} --meta {metav} --arch {archv} '
                      f'--augmentation {augv}{extra}{proba} --instrument {instr(out)} --seed {seed} '
                      f'--out {out} --resume')
            gpu(f"d1_{arm}_s{seed}", "D1", seed, cmdline, out, depends_on=("d1_r2_s42",),
               expected_outputs=EO_D1_ARM[arm])
            d1_all_seed_ids.append(f"d1_{arm}_s{seed}")
        elif arm == "r10":
            out = f"results_kc23_d1_r10_s{seed}"
            gpu(f"d1_r10_s{seed}", "D1", seed,
               f'{PY} run_adabn_cnn_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
               f'--augmentation chandrop --epochs 40 --seed {seed} --out {out} --resume', out,
               depends_on=("d1_r2_s42",), expected_outputs=EO_D1_ARM["r10"])
            d1_all_seed_ids.append(f"d1_r10_s{seed}")
        elif arm == "r11":
            out = f"results_kc23_d1_r11_s{seed}"
            gpu(f"d1_r11_s{seed}", "D1", seed,
               f'{PY} run_deep_coral_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
               f'--augmentation chandrop --coral-lambda 30 --batch 256 --seed {seed} --out {out} --resume', out,
               depends_on=("d1_r2_s42",), expected_outputs=EO_D1_ARM["r11"])
            d1_all_seed_ids.append(f"d1_r11_s{seed}")
    # R8/R9 (SimpleEMGCNN) at this seed
    for arm, aug in [("r8", None), ("r9", "chandrop")]:
        out = f"results_kc23_d1_{arm}_s{seed}"
        augflag = " --augment chandrop --aug-chandrop-p 0.2" if aug else ""
        gpu(f"d1_{arm}_s{seed}", "D1", seed,
           f'{PY} train_cnn_loso.py --npz {NPZ_250} --meta {META_250} --norm-mode per_subject{augflag} '
           f'--seed {seed} --out {out} --resume', out, depends_on=("d1_r2_s42",),
           expected_outputs=EO_D1_ARM[arm])
        d1_all_seed_ids.append(f"d1_{arm}_s{seed}")
# NOTE: the D1.5-D1.7 contrast/headline gate does NOT attach to d1_r2_s1001 --
# that job's own depends_on is only ("d1_r2_s42",), so it can (and did, in
# scheduling order) finish long before the other 46 Tier-A/Tier-B runs it
# needs are done, pointed at its own private out_dir. The real aggregate gate
# job is added after Tier B (below), once all 63 D1+D1B runs are collected.

# ============================================================================
# #10 (Phase 3, CPU): KC-D2 reliance analysis (analysis only, on D1's instrumented runs)
# ============================================================================
# The stats script reads d2_persubject_sums.csv, which nothing produced (the row ran the stats script as its own command).
# kc23_d2_aggregate.py builds it from the instrumented D1 runs (occlusion, attenuation and permutation sums, no clipping);
# the gate then writes D2_VERDICT.md, and the queue checks it carries a letter (26 Sept 2026 conformance pass).
cpu("d2_reliance", "D2", None,
   f'{PY} kc23_d2_aggregate.py --root . --out results_kc23_d2_reliance',
   "results_kc23_d2_reliance", depends_on=tuple(d1_all_seed_ids), gate_script="kc23_d2_reliance_stats.py",
   expected_outputs=EO_VERDICT_LETTER)

# ============================================================================
# #11 (Phase 4, GPU): KC-D4 channel-axis invariance sweep
# ============================================================================
d4_ids = []
for seed in D3_D4_SEEDS:
    for sd in [0.40, 0.50, 0.60, 0.80, 1.00]:
        out = f"results_kc23_d4_mpchandrop_sd{sd:.2f}_s{seed}"
        jid = f"d4_mpchandrop_sd{sd:.2f}_s{seed}"
        gpu(jid, "D4", seed,
           f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
           f'--augmentation mpchandrop --aug-gain-sd {sd} --instrument {instr(out)} --seed {seed} '
           f'--out {out} --resume', out, depends_on=("d1_r2_s1001",), expected_outputs=EO_CNN_ARCH_SIAT)
        d4_ids.append(jid)
    for sd in [0.80, 1.00]:
        out = f"results_kc23_d4_gainjitter_sd{sd:.2f}_s{seed}"
        jid = f"d4_gainjitter_sd{sd:.2f}_s{seed}"
        gpu(jid, "D4", seed,
           f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
           f'--augmentation gainjitter --aug-gain-sd {sd} --instrument {instr(out)} --seed {seed} '
           f'--out {out} --resume', out, depends_on=("d1_r2_s1001",), expected_outputs=EO_CNN_ARCH_SIAT)
        d4_ids.append(jid)
# The D4 gate no longer sits on the last run (d4_gainjitter_sd1.00_s123): it fired before its siblings existed and read
# inputs nothing produced. d4_stats (below, after Tier B) aggregates every run and then gates.

# ============================================================================
# #11b (Phase 4, GPU): KC-D6 Stage 1 -- every family at seed 42
# ============================================================================
D6_GATE = "kc23_d6_stats.py"       # sanity, manipulation, outcome and mechanism gates all live in one script
D6_AGGREGATE = "kc23_d6_aggregate.py"
d6_stage1_ids = []
adv_lambdas = [0, 0.03, 0.1, 0.3, 1, 3, 10]
for lam in adv_lambdas:
    out = f"results_kc23_d6_adv_marginal_l{lam}_s42"
    jid = f"d6_adv_marginal_l{lam}_s42"
    gpu(jid, "D6", 42,
       f'{PY} run_adv_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
       f'--augmentation chandrop --adv-lambda {lam} --adv-mode marginal --epochs 40 --batch 256 '
       f'--instrument {instr(out)} --seed 42 --out {out} --resume', out,
       depends_on=("d0_smoke_gate", "d1_r2_s42"), expected_outputs=EO_ADV_SIAT)
    d6_stage1_ids.append(jid)
# The D6.4 sanity gate only needs the lambda_max=0 arm ("after Stage 1, ADV at lambda_max=0" --
# it does not wait for the rest of the family). kc23_d6_aggregate.py extracts d6_sanity.csv from
# that one run's adv_subjectwise.csv; the gate script dispatches on out_dir's own name.
cpu("d6_sanity_check", "D6-SANITY", None,
   f'{PY} {D6_AGGREGATE} --root . --out results_kc23_d6_sanity_check --require sanity',
   "results_kc23_d6_sanity_check", depends_on=("d6_adv_marginal_l0_s42",), gate_script=D6_GATE,
   expected_outputs=EO_VERDICT_LETTER)
sfc_weights = [0.1, 1, 10, 100, 1000]
for w in sfc_weights:
    out = f"results_kc23_d6_sfc_w{w}_s42"
    jid = f"d6_sfc_w{w}_s42"
    gpu(jid, "D6", 42,
       f'{PY} run_deep_coral_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
       f'--augmentation chandrop --coral-lambda {w} --coral-normalize l2 --batch 256 --seed 42 '
       f'--instrument {instr(out)} --out {out} --resume', out, depends_on=("d1_r2_s42",), expected_outputs=EO_DEEP_CORAL_SIAT)
    d6_stage1_ids.append(jid)
for lam in [0.1, 1, 10]:
    out = f"results_kc23_d6_advps_l{lam}_s42"
    jid = f"d6_advps_l{lam}_s42"
    gpu(jid, "D6", 42,
       f'{PY} run_adv_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
       f'--norm-mode per_subject --augmentation chandrop --adv-lambda {lam} --adv-mode marginal '
       f'--epochs 40 --batch 256 --seed 42 --instrument {instr(out)} --out {out} --resume', out, depends_on=("d1_r2_s42",),
       expected_outputs=EO_ADV_SIAT)
    d6_stage1_ids.append(jid)
# The D6.5 manipulation gate needs every knob of every family done (per family, "across the knob
# grid, 40 folds, seed 42 re-run") -- was wired to fire on the LAST stage-1 job's own out_dir
# (which only ever held that one job's own knob), same wrong-directory mistake as D1/S1. Now a
# separate pseudo-job depending on ALL of Stage 1, writing into its own shared aggregate dir.
cpu("d6_manipulation_check", "D6-MANIP", None,
   f'{PY} {D6_AGGREGATE} --root . --out results_kc23_d6_manipulation_check --require manipulation',
   "results_kc23_d6_manipulation_check", depends_on=tuple(d6_stage1_ids), gate_script=D6_GATE,
   expected_outputs=EO_VERDICT_LETTER)
# ADV-C (mechanism): lambda_max at the ADV collapse point -- unknown until the
# ADV family above has run and the collapse point is found. One placeholder,
# same staging principle as 11c below.
cpu("d6_advc_placeholder", "D6", 42,
   "# PLACEHOLDER: ADV-C's lambda_max grid is the ADV collapse point (Section 6.2), "
   "determined only after the ADV family (d6_adv_marginal_l*_s42) has run. Generate its "
   "real job rows with a follow-up script once the collapse point is known.",
   "results_kc23_d6_advc_s42", depends_on=tuple(j for j in d6_stage1_ids if j.startswith("d6_adv_marginal")))

# ============================================================================
# #11c (Phase 4, GPU): KC-D6 Stage 2 -- seeds 7, 123, passing families only
# ============================================================================
cpu("d6_stage2_placeholder", "D6-Stage2", None,
   "# PLACEHOLDER: 11c's jobs are generated ONLY after the D6 manipulation gate "
   "(11b, d6_manipulation_check, gate_script=" + D6_GATE + ") decides which families pass (G-PASS/G-WEAK). "
   "Per RUN_ORDER_KC23.md item 11c, guessing these before 11b's result exists is out of "
   "scope for this generator. Write kc23_d6_stage2_job_gen.py to read 11b's gate output and "
   "append the real seed-7/123 rows for passing families to kc23_jobs_gpu.csv.",
   "results_kc23_d6_stage2", depends_on=("d6_sfc_w1000_s42",))

# ============================================================================
# #12 (Phase 4, GPU): KC-D3 axis against magnitude
# ============================================================================
d3_ids = []
for seed in D3_D4_SEEDS:
    for arm, mode, mag in [("x1", "gaussian", 0.10), ("x2", "gaussian", 0.40),
                            ("x3", "chanoffset", 0.40), ("x4", "globalgain", 0.40)]:
        out = f"results_kc23_d3_{arm}_s{seed}"
        jid = f"d3_{arm}_s{seed}"
        sigma_flag = f"--aug-sigma {mag}" if mode == "gaussian" else f"--aug-gain-sd {mag}"
        gpu(jid, "D3", seed,
           f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
           f'--augmentation {mode} {sigma_flag} --instrument {instr(out)} --seed {seed} '
           f'--out {out} --resume', out, depends_on=("d1_r2_s1001",), expected_outputs=EO_CNN_ARCH_SIAT)
        d3_ids.append(jid)
# Likewise the D3 gate no longer sits on d3_x4_s123; d3_stats (below) aggregates every run and then gates.

# ============================================================================
# #13 (Phase 4, GPU): KC-D1 Tier B (R13-R17), 3 realizations
# ============================================================================
tierb_ids = []
for seed in D1_TIER_B_SEEDS:
    specs = [("r13", "chandrop", 0.1), ("r14", "gainjitter", 0.30), ("r15", "chandrop", 0.5),
            ("r16", "gainjitter", 0.50), ("r17", "chandrop", 0.3)]
    for arm, mode, val in specs:
        out = f"results_kc23_d1_{arm}_s{seed}"
        jid = f"d1_{arm}_s{seed}"
        flag = f"--aug-chandrop-p {val}" if mode == "chandrop" else f"--aug-gain-sd {val}"
        gpu(jid, "D1B", seed,
           f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
           f'--augmentation {mode} {flag} --instrument {instr(out)} --seed {seed} --out {out} --resume',
           out, depends_on=("d1_r2_s42",), expected_outputs=EO_D1_ARM[arm])
        tierb_ids.append(jid)

# ============================================================================
# #13b (Phase 4, CPU): KC-D1.5-D1.7 full aggregate gate, once all 63 D1+D1B
# realizations exist (48 Tier A + 15 Tier B)
# ============================================================================
# The soft vote and stacking of the seed-specific ResNet-SE+CD with the published SVM, RF and CNN (C13, C13b): the aggregator
# reads results_kc23_d1_ensemble_s<seed>/ensemble_s<seed>.csv, which nothing built before kc23_d1_ensemble.py.
d1_ensemble_ids = []
for seed in [42, 7, 123, 1001]:
    out = f"results_kc23_d1_ensemble_s{seed}"
    cpu(f"d1_ensemble_s{seed}", "D1-ENS", seed,
       f'{PY} kc23_d1_ensemble.py --root . --seed {seed} --out {out}', out, depends_on=(f"d1_r2_s{seed}",),
       expected_outputs=f"ensemble_s{seed}.csv|{SIAT_N}")
    d1_ensemble_ids.append(f"d1_ensemble_s{seed}")
cpu("d1_full_aggregate", "D1-AGG", None,
   f'{PY} {D1_AGGREGATE} --root . --out results_kc23_d1_stats --require full',
   "results_kc23_d1_stats", depends_on=tuple(d1_all_seed_ids + tierb_ids + d1_ensemble_ids),
   gate_script=D1_REPRO_GATE, expected_outputs=EO_VERDICT_LETTER)

# ============================================================================
# #12b / #11d (Phase 4, CPU): KC-D3 and KC-D4 stats -- aggregate every run, then gate
# ============================================================================
cpu("d3_stats", "D3-STATS", None,
   f'{PY} kc23_d3_aggregate.py --root . --out results_kc23_d3_stats', "results_kc23_d3_stats",
   depends_on=tuple(d3_ids) + tuple(f"d1_{a}_s{sd}" for a in ("r1", "r3") for sd in D3_D4_SEEDS),
   gate_script="kc23_d3_axis_stats.py", expected_outputs="D3_VERDICT.md|LETTER")
cpu("d4_stats", "D4-STATS", None,
   f'{PY} kc23_d4_aggregate.py --root . --out results_kc23_d4_stats', "results_kc23_d4_stats",
   depends_on=tuple(d4_ids) + tuple(tierb_ids) + tuple(f"d1_{a}_s{sd}" for a in ("r1", "r2", "r3") for sd in D3_D4_SEEDS),
   gate_script="kc23_d4_invariance_stats.py", expected_outputs="D4_VERDICT.md|LETTER")

# ============================================================================
# #14 (Phase 5, GPU+CPU): KC-S3 active-only benchmark
# ============================================================================
NPZ_AONLY = "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly.npz"
META_AONLY = "features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_meta.csv"
cpu("s3_inventory", "S3", None,
   f'{PY} kc23_s3_inventory.py --out results_kc23_s3_active_benchmark',
   "results_kc23_s3_active_benchmark", depends_on=("d1_r2_s42",),
   expected_outputs="active_only_inventory.csv|16")   # 8 model families x 2 normalizations
for arch in ["simple", "resnet_se"]:
    for norm in ["global", "per_subject"]:
        out = f"results_kc23_s3_{arch}_{norm}"
        gpu(f"s3_{arch}_{norm}", "S3", 42,
           f'{PY} run_cnn_arch_loso.py --npz {NPZ_AONLY} --meta {META_AONLY} --arch {arch} '
           f'--norm-mode {norm} --seed 42 --save-proba {out}/proba --model-tag S3_{arch.upper()} '
           f'--out {out} --resume', out,
           depends_on=("s3_inventory",),
           gate_script="",  # KC-S3 has no halting letters per the plan; kc23_s3_inventory.py is not a gate
           expected_outputs=EO_CNN_ARCH_SIAT)
# ResNet-SE+CD, global only: the per-subject cell already exists as results_aonly_resnet_se_cd_persubj (seed 42, chandrop 0.2,
# resnet_se, per_subject, same active-only windows, proba saved), and the plan says "run only the missing cells" (S3.2 item 2),
# so the earlier per-subject row is dropped (26 Sept 2026; it had not run). kc23_s3_benchmark.py reads the existing directory.
out = "results_kc23_s3_resnet_se_cd_global"
gpu("s3_resnet_se_cd_global", "S3", 42,
   f'{PY} run_cnn_arch_loso.py --npz {NPZ_AONLY} --meta {META_AONLY} --arch resnet_se '
   f'--augmentation chandrop --aug-chandrop-p 0.2 --model-tag RESNET_SE_CD --norm-mode global '
   f'--seed 42 --save-proba {out}/proba --out {out} --resume', out, depends_on=("s3_inventory",), gate_script="",
   expected_outputs=EO_CNN_ARCH_SIAT)
FEAT_AONLY_FREQ = "features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_ext.npz"
for norm in ["global", "per_subject"]:
    out = f"results_kc23_s3_lda_{norm}"
    # LDA must go through run_lda_loso.py -- the script that produced the published LDA
    # figures (68.7 per-subject, 62.8 global, Table 4.1), reproduction-proven exact on
    # subjects 1-3 (2026-09-24) -- never train_classical_loso.py, which has no LDA
    # implementation at all (--models LDA now raises there rather than silently no-op'ing).
    # Features fixed 2026-09-24: this and s3_svm_per_subject below originally used
    # FEAT_250_BASE/META_250_FEAT -- the SIAT-wide (rest-included) features, not the
    # active-only ones results_aonly_global's own run_config.json shows the published
    # SVM/RF reference actually used (the freq/ext Aonly features + META_AONLY).
    cpu(f"s3_lda_{norm}", "S3", None,
       f'{PY} run_lda_loso.py --features {FEAT_AONLY_FREQ} --meta {META_AONLY} '
       f'--norm-mode {norm} --save-preds --out {out} --resume', out,
       depends_on=("s3_inventory",), gate_script="", expected_outputs=EO_LDA_SIAT)
cpu("s3_svm_per_subject", "S3", None,
   f'{PY} train_classical_loso.py --features {FEAT_AONLY_FREQ} --meta {META_AONLY} '
   f'--models SVM --norm-mode per_subject --out results_kc23_s3_svm_per_subject --resume',
   "results_kc23_s3_svm_per_subject", depends_on=("s3_inventory",), gate_script="",
   expected_outputs=EO_NESTED_LOSO_SIAT)
# s3_svm_per_subject duplicates results_aonly_persubj (same features, flags and seed) and had already run; the benchmark reads
# the earlier directory and keeps this one only as a reproduction check.
# The benchmark table and S3_VERDICT.md (S3.3): one row per family x normalization with the DNS->WAK critical-error rate.
cpu("s3_benchmark", "S3", None,
   f'{PY} kc23_s3_benchmark.py --out results_kc23_s3_active_benchmark',
   "results_kc23_s3_active_benchmark",
   depends_on=("s3_inventory", "s3_simple_global", "s3_simple_per_subject", "s3_resnet_se_global", "s3_resnet_se_per_subject",
               "s3_resnet_se_cd_global", "s3_lda_global", "s3_lda_per_subject", "c3_ensemble"),
   expected_outputs="benchmark_active_only.csv|EXISTS")
# SVMX/HGB (per plan: "if KC-C3 lands P2 or P3") are correctly absent -- KC-C3's tuning outcome
# (kc23_c3_tuning_stats.py) hasn't landed yet, so whether they're needed at all is still unknown;
# no row is generated for them here, matching kc23_s3_inventory.py's own conditional.

# ============================================================================
# #15 (Phase 5, GPU+CPU): KC-S2 ENABL3S transitions -- gated on F0 feasibility
# ============================================================================
S2_F0_GATE = "kc23_s2_f0_feasibility.py"
# The queue always invokes gate_script as `python <gate_script> --out <out_dir>`
# (no --root), so the transition count has to be computed and cached as part of
# the job's OWN command; the post-job gate call then reads the cache. Found
# live 2026-09-24: without this chaining, the gate crashed looking for a
# s2_transition_counts.csv nothing had ever written (FileNotFoundError), and
# the crash was (separately) being swallowed as a silent pass -- see
# kc23_queue.py's fail-closed fix.
cpu("s2_f0_feasibility", "S2-F0", None,
   f'{PY} adapt_external_dataset.py --root 5362627 --out results_kc23_s2_adapter_circuitmeta '
   f'--tag ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_kc23s2 --with-circuit-meta --resume '
   f'&& {PY} kc23_s2_f0_feasibility.py --out results_kc23_s2_adapter_circuitmeta --root 5362627',
   "results_kc23_s2_adapter_circuitmeta", depends_on=(), gate_script=S2_F0_GATE,
   expected_outputs="S2_F0_VERDICT.md|LETTER")
# S2.3: per-window LOSO predictions on ENABL3S (locked SVM/ResNet-SE+CD/soft,
# transductive + causal-100 + causal-balanced25). Reuses the circuit-meta
# windows s2_f0_feasibility already built (row-aligned with the published
# Freq-72 features, confirmed 2026-09-24: same 45525 rows, same subject order).
cpu("s2_predictions", "S2-PRED", None,
   f'{PY} kc23_s2_predictions.py --root . --out results_kc23_s2_predictions',
   "results_kc23_s2_predictions", depends_on=("s2_f0_feasibility",),
   # every window is predicted exactly once (its own held-out fold) x 3 models = 45525 x 3
   expected_outputs=f"s2_predictions_causal100.csv|{45525 * 3}")
# The ground-truth transition table (S2.2), built from the raw per-sample Mode signal and validated against the
# published windows (0 disagreements). 952 = 238 transitions of each of the 4 retained types over the 10 subjects.
S2_CIRCUITMETA_META = ("results_kc23_s2_adapter_circuitmeta/"
                       "windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_kc23s2_meta.csv")
cpu("s2_transition_table", "S2-TABLE", None,
   f'{PY} kc23_s2_transition_table.py --root 5362627 --out results_kc23_s2_transition_table '
   f'--windows-meta {S2_CIRCUITMETA_META}',
   "results_kc23_s2_transition_table", depends_on=("s2_f0_feasibility",),
   expected_outputs="s2_transition_table.csv|952")
# S2.4, rewritten 2026-09-25: the analysis over ALL THREE conditions' predictions, per model, against the
# ground-truth transition table. The old version had no expected_outputs, so a 107-byte placeholder verdict from
# 09-23 made it "skipped(complete)" without it ever reading the predictions (and it would have pooled the three
# models and mis-scaled window times if it had). 9 rows = 3 conditions x 3 models.
cpu("s2_transitions", "S2", None,
   f'{PY} kc23_s2_transitions.py --out results_kc23_s2_transitions '
   f'--preds-dir results_kc23_s2_predictions '
   f'--table results_kc23_s2_transition_table/s2_transition_table.csv',
   "results_kc23_s2_transitions", depends_on=("s2_predictions", "s2_transition_table"), gate_script="",
   expected_outputs="s2_measures.csv|9")

# ============================================================================
# S2b: the 400ms ENABL3S window trade -- new adapter tag, SVM + ResNet-SE+CD
# ============================================================================
cpu("s2b_adapter_400", "S2b", None,
   f'{PY} adapt_external_dataset.py --root 5362627 --out results_kc23_s2b_adapter_400 '
   f'--tag ENABL3S_WAK_UPS_DNS_STDUP_w400_ov50_conf60_kc23s2b --window-ms 400 --with-circuit-meta --resume',
   "results_kc23_s2b_adapter_400", depends_on=(),
   expected_outputs="windows_ENABL3S_WAK_UPS_DNS_STDUP_w400_ov50_conf60_kc23s2b_meta.csv|28125")
EO_S2B_FEAT = "freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w400_ov50_conf60_kc23s2b_features_ext.npz"
cpu("s2b_extract_features", "S2b", None,
   f'{PY} extract_features.py '
   f'--npz results_kc23_s2b_adapter_400/windows_ENABL3S_WAK_UPS_DNS_STDUP_w400_ov50_conf60_kc23s2b.npz '
   f'--meta results_kc23_s2b_adapter_400/windows_ENABL3S_WAK_UPS_DNS_STDUP_w400_ov50_conf60_kc23s2b_meta.csv '
   f'--out-dir results_kc23_s2b_features --prefix freq --use raw --freq --fs 1000 --no-wavelet',
   "results_kc23_s2b_features", depends_on=("s2b_adapter_400",),
   expected_outputs="freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w400_ov50_conf60_kc23s2b_features_meta.csv|28125")
# The window trade needs per-window predictions with circuit and time to score decision delay (S2.4), so the two old rows
# (run_cnn_arch_loso --save-proba and train_classical_loso, which give LOSO F1 with no window times) are replaced by ONE row
# that runs the same locked pipeline as s2_predictions on the 400 ms windows (kc23_s2_predictions.py --window-ms 400), then
# kc23_s2b_window_trade.py scores 250 against 400 ms and regenerates S2_VERDICT.md with Reading 2 (26 Sept 2026 conformance).
cpu("s2b_predictions_400", "S2b", None,
   f'{PY} kc23_s2_predictions.py --root . --out results_kc23_s2b_predictions --window-ms 400',
   "results_kc23_s2b_predictions", depends_on=("s2b_extract_features",),
   expected_outputs=f"s2_predictions_causal100.csv|{28125 * 3}")
cpu("s2b_window_trade", "S2b", None,
   f'{PY} kc23_s2b_window_trade.py --out results_kc23_s2b_window_trade --preds-250 results_kc23_s2_predictions '
   f'--preds-400 results_kc23_s2b_predictions --table results_kc23_s2_transition_table/s2_transition_table.csv '
   f'--s2-out results_kc23_s2_transitions',
   "results_kc23_s2b_window_trade", depends_on=("s2b_predictions_400", "s2_predictions", "s2_transition_table"),
   expected_outputs="s2b_measures.csv|9")

# ============================================================================
# C-a (Phase 2-5, CPU): KC-C2 whitening, 250 and 400 ms
# ============================================================================
c2_ids = []
for tag, npzf, metaf, gate_env in [("w250", FEAT_250_FREQ, META_250_FEAT, None),
                                    ("w400", FEAT_400_FREQ, META_400_FEAT, "w400")]:
    out = f"results_kc23_c2_whitening_{tag}"
    jid = f"c2_ladder_{tag}"
    if gate_env:
        # kc23_queue.py launches with shell=True, which on Windows is cmd.exe
        # (not bash) -- "VAR=val cmd" is bash syntax and silently fails under
        # cmd.exe (it tries to run "VAR=val" as a program). Use cmd's own
        # `set "VAR=val" &&` chaining instead, portable to how the queue
        # actually spawns jobs.
        cmdline = (f'set "LADDER_FEAT={npzf}" && set "LADDER_META={metaf}" && '
                  f'{PY} run_alignment_ladder_loso.py --rungs 3,0,4,4b,4c,4d,4lw,4o --out {out} --no-gate --resume')
    else:
        cmdline = f'{PY} run_alignment_ladder_loso.py --rungs 3,0,4,4b,4c,4d,4lw,4o --out {out} --resume'
    # The C2 gate needs the geometry rows (the subject probe under 4o), so it moved from c2_ladder_w400 to c2_geometry_w400.
    cpu(jid, "C2", None, cmdline, out, depends_on=("code_c2_inertness",), gate_script="",
       expected_outputs=f"ladder_loso_3_SVM_subjectwise.csv|{SIAT_N}")
    c2_ids.append(jid)
    if tag == "w400":
        cpu("c2_geometry_w400", "C2", None,
           f'{PY} kc23_c6_geometry.py --out {out} --feat {npzf} --meta {metaf} --rungs 3,0,4b,4c,4d,4lw,4o --resume',
           out, depends_on=(jid,), gate_script="kc23_c2_whitening_stats.py", expected_outputs="ladder_geometry.csv|EXISTS")

# ============================================================================
# C-b (Phase 2-5, CPU): KC-C3 classical tuning parity
# ============================================================================
c3_ids = []
for model in ["SVM", "RF", "HGB", "KNN"]:
    for norm in ["per_subject", "global"]:
        out = f"results_kc23_c3_{model.lower()}_{norm}"
        jid = f"c3_{model.lower()}_{norm}"
        search = "grid" if model in ("SVM", "KNN") else "random"
        # The tuning row NEVER carries --save-proba: that flag is train_classical_loso.py's cheap refit path (no search,
        # best_params taken from --reuse-params-dir, default results_loso_freq_persubj). The first SVM-X run combined
        # them, so all 40 folds used the published C=1 / gamma='scale' and the "tuned" SVM was the published one
        # (mean F1 0.7767); found in the 26 September conformance pass, that directory is set aside (INVALID.md).
        cpu(jid, "C3", None,
           f'{PY} train_classical_loso.py --features {FEAT_250_FREQ} --meta {META_250_FEAT} '
           f'--models {model} --norm-mode {norm} --grid extended --search {search} --n-iter 30 '
           f'--n-jobs 1 --rf-n-jobs 4 --out {out} --resume', out, depends_on=("code_c3_inertness",),
           expected_outputs=EO_NESTED_LOSO_SIAT)
        c3_ids.append(jid)
        if norm == "per_subject" and model in ("SVM", "RF", "HGB"):
            # probabilities for the ensemble (C3 Endpoint 3) and the best-member soft vote: a refit with the SELECTED params
            rid = f"{jid}_proba"
            cpu(rid, "C3", None,
               f'{PY} train_classical_loso.py --features {FEAT_250_FREQ} --meta {META_250_FEAT} '
               f'--models {model} --norm-mode {norm} --save-proba --proba-out {out}/proba '
               f'--reuse-params-dir {out} --n-jobs 1 --rf-n-jobs 4 --out {out}_refit --resume', f"{out}_refit",
               depends_on=(jid,), expected_outputs=EO_NESTED_LOSO_SIAT)
            c3_ids.append(rid)
cpu("c3_ensemble", "C3", None,
   f'{PY} kc23_c3_merge_proba.py --new-svm-dir results_kc23_c3_svm_per_subject/proba '
   f'--published-dir results_ensemble_v2/proba_aug_chandrop --out results_kc23_c3_ensemble_proba '
   f'&& {PY} ensemble_v2_combine.py --proba-dir results_kc23_c3_ensemble_proba --out results_kc23_c3_ensemble',
   "results_kc23_c3_ensemble", depends_on=tuple(c3_ids), gate_script="kc23_c3_tuning_stats.py",
   expected_outputs=f"ensemble_v2_subjectwise.csv|{SIAT_N}")

# ============================================================================
# C-c (Phase 2-5, CPU): KC-C4 richer established feature sets
# ============================================================================
cpu("c4_extract", "C4", None,
   f'{PY} kc23_c4_extract_rich.py --out features_out --freq72 features_out/'
   f'freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz',
   "features_out", depends_on=("c3_ensemble",),
   # BOTH feature files: the command used to pass no --freq72, so Rich-126 was never built (fixed 2026-09-25)
   expected_outputs="kc23_rich126_features.npz|EXISTS")
for feat in ["tdpsd54", "rich126"]:
    for model in ["SVM", "LDA"]:
        for norm in ["per_subject", "global"]:
            out = f"results_kc23_c4_{feat}_{model.lower()}_{norm}"
            gate = "kc23_c4_feature_stats.py" if (feat, model, norm) == ("rich126", "LDA", "global") else ""
            if model == "LDA":
                # LDA must go through run_lda_loso.py, never train_classical_loso.py -- see the
                # KC-S3 LDA rows above for why (it has no LDA implementation at all).
                # load_features_npz is generic (first key, 2D array), confirmed compatible with
                # kc23_c4_extract_rich.py's own npz format (single "X" key) by inspection.
                cmd = (f'{PY} run_lda_loso.py --features features_out/kc23_{feat}_features.npz '
                      f'--meta {META_250_FEAT} --norm-mode {norm} --out {out} --resume')
                eo = EO_LDA_SIAT
            else:
                cmd = (f'{PY} train_classical_loso.py --features features_out/kc23_{feat}_features.npz '
                      f'--meta {META_250_FEAT} --models {model} --norm-mode {norm} --grid extended --search grid '
                      f'--n-jobs 1 --out {out} --resume')     # plan C4.3: SVM-X, the KC-C3 grid (was the default grid)
                eo = EO_NESTED_LOSO_SIAT
            cpu(f"c4_{feat}_{model.lower()}_{norm}", "C4", None, cmd, out,
               depends_on=("c4_extract",), gate_script=gate, expected_outputs=eo)

# ============================================================================
# C-d (Phase 2-5, CPU plus ~1h GPU): KC-C5 leak decomposition
# ============================================================================
c5_ids = []
for dataset, feat, metaf, tag, time_units in [
        ("siat", FEAT_250_BASE, META_250_FEAT, "kc23_c5_siat", "seconds"),
        ("enabl3s", "features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_ext.npz",
         "features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_meta.csv",
         "kc23_c5_enabl3s", "samples")]:
    root = f"results_kc23_c5_leak_{dataset}"
    tu = f"--time-units {time_units}"
    # Each scheme gets its OWN subdirectory of the shared dataset root, never
    # the root itself: is_complete() marks a job's out_dir "already done" the
    # moment ANY *subjectwise.csv exists there, with no notion of which job
    # wrote it, so 11 jobs sharing one directory meant only the first to
    # finish would ever actually run -- found live 2026-09-24 before it had
    # silently eaten 10 of these 11 jobs (the queue hadn't reached C5 yet).
    # kc23_c5_leak_stats.py's gate is given one of these leaf subdirectories
    # (whichever job triggers it) and searches its PARENT for every sibling.
    this_dataset_ids = []
    n_subj = SIAT_N if dataset == "siat" else ENABL3S_N
    def c5_job(jid, scheme_subdir, cmd_tail, depends=("code_c5_inertness",), gate=""):
        out = f"{root}/{scheme_subdir}"
        cpu(jid, "C5", None,
           f'{PY} b8_movement_blocked_sd.py --features {feat} --meta {metaf} --models SVM,RF,LDA '
           f'--window-ms 250 --tag {tag} {cmd_tail} {tu} --out {out} --resume',
           out, depends_on=depends, gate_script=gate, expected_outputs=f"b8_*_subjectwise.csv|{n_subj}")
        this_dataset_ids.append(jid)
        return jid
    # Plan C5.3: P50, P0, B-g and I-g are the POOLED cv-unit (the subject-inclusive construction of the published SD
    # figures); W-B1 is the ONE per_subject arm ("naming control: a true per-subject model"). Every arm used to be
    # per_subject, which made W-B1 identical to B-1 (conformance pass, 26 Sept 2026; no C5 data existed).
    c5_job(f"c5_{dataset}_p50", "p50", "--scheme pooled_random --cv-unit pooled")
    c5_job(f"c5_{dataset}_p0", "p0", "--scheme pooled_random_nonoverlap --cv-unit pooled")
    for g in [1, 2, 4, 8, 16]:
        c5_job(f"c5_{dataset}_b{g}", f"b{g}", f"--scheme blocked --guard-windows {g} --cv-unit pooled")
    for g in [1, 4, 16]:
        c5_job(f"c5_{dataset}_i{g}", f"i{g}",
              f"--scheme interleaved --n-chunks 20 --guard-windows {g} --cv-unit pooled")
    jid = c5_job(f"c5_{dataset}_wb1", "wb1", "--scheme blocked --guard-windows 1 --cv-unit per_subject")
    c5_ids.append(jid)
    # The verdict is its own light pseudo-row (the gate reads every sibling arm and writes results_kc23_c5_leak_<ds>/
    # C5_VERDICT.md, which the queue then checks for a LETTER line). It used to hang off the W-B1 row, whose own
    # expected output is a b8 csv, so a gate that wrote no letter was never noticed.
    cpu(f"c5_{dataset}_verdict", "C5", None,
       f'{PY} kc23_c5_leak_stats.py --out {root}', root, depends_on=tuple(this_dataset_ids),
       gate_script="kc23_c5_leak_stats.py", expected_outputs="C5_VERDICT.md|LETTER")
    c5_ids.append(f"c5_{dataset}_verdict")

# b8_cnn_sd.py gained --scheme/--guard-windows/--n-chunks in commit 7e1be9a (inertness-proven on
# CPU); P50/P0 don't depend on the plateau guard and are runnable now. B/I at "the plateau guard"
# (C5.3) are NOT runnable yet -- g* is only known after the SIAT c5_siat_wb1 gate (above) has run;
# kc23_c5_cnn_job_gen.py (written alongside this fix) reads that gate's C5_decomposition.csv and
# appends the real c5_simplecnn_sd_b<g*>/c5_simplecnn_sd_i<g*> rows once it exists.
# Each scheme gets its OWN out_dir: is_complete() marks a job's out_dir "already done" the
# moment ANY file matching *subjectwise.csv exists there, with no notion of WHICH job wrote
# it -- two jobs sharing a directory means the second is silently skipped the instant the
# first finishes. Same D1/S1 out_dir-collision mistake found earlier tonight, here between
# sibling jobs rather than sibling stages.
gpu("c5_simplecnn_sd_p50", "C5", 42,
   f'{PY} b8_cnn_sd.py --npz {NPZ_250} --meta {META_250} --scheme pooled_random '
   f'--out results_kc23_c5_simplecnn_siat_p50 --resume', "results_kc23_c5_simplecnn_siat_p50",
   depends_on=("code_c5_inertness",), expected_outputs=f"b8_cnn_*_subjectwise.csv|{SIAT_N}")
gpu("c5_simplecnn_sd_p0", "C5", 42,
   f'{PY} b8_cnn_sd.py --npz {NPZ_250} --meta {META_250} --scheme pooled_random_nonoverlap '
   f'--out results_kc23_c5_simplecnn_siat_p0 --resume', "results_kc23_c5_simplecnn_siat_p0",
   depends_on=("code_c5_inertness",), expected_outputs=f"b8_cnn_*_subjectwise.csv|{SIAT_N}")
cpu("c5_simplecnn_sd_stage2_placeholder", "C5-Stage2", None,
   "# PLACEHOLDER: B/I at the plateau guard (C5.3's SimpleEMGCNN row) needs SIAT's own plateau g*, "
   "known only after c5_siat_wb1's gate (kc23_c5_leak_stats.py) has run. Run kc23_c5_cnn_job_gen.py "
   "once results_kc23_c5_leak_siat/C5_decomposition.csv exists, to append the real "
   "c5_simplecnn_sd_b<g*>/c5_simplecnn_sd_i<g*> GPU rows (own out_dir: results_kc23_c5_simplecnn_siat).",
   "results_kc23_c5_simplecnn_siat", depends_on=("c5_siat_verdict",))

# ============================================================================
# C-e (Phase 2-5, CPU): KC-C6 alignment ladder on ENABL3S
# ============================================================================
cpu("c6_ladder_enabl3s", "C6", None,
   'set "LADDER_FEAT=features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_ext.npz" && '
   'set "LADDER_META=features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_meta.csv" && '
   f'{PY} run_alignment_ladder_loso.py --rungs 3,0,1,2,4,4lw,4o --out results_kc23_c6_ladder_enabl3s '
   '--no-gate --resume', "results_kc23_c6_ladder_enabl3s", depends_on=tuple(c2_ids), gate_script="",
   expected_outputs=f"ladder_loso_3_SVM_subjectwise.csv|{ENABL3S_N}")
# C6 geometry rows (MMD, W1, the three probes pooled and within-movement, silhouette) on ENABL3S through the C2 environment
# override, then the C6 gate (R1 needs the centering fraction of the probe drop).
C6_FEAT = "features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_ext.npz"
C6_META = "features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_meta.csv"
cpu("c6_geometry", "C6", None,
   f'{PY} kc23_c6_geometry.py --out results_kc23_c6_ladder_enabl3s --feat {C6_FEAT} --meta {C6_META} '
   f'--rungs 3,0,1,2,4,4lw,4o --resume', "results_kc23_c6_ladder_enabl3s", depends_on=("c6_ladder_enabl3s",),
   gate_script="kc23_c6_ladder_stats.py", expected_outputs="ladder_geometry.csv|EXISTS")


# LIGHT rows (2026-09-25): read-only aggregators, gates and verdict builders, single-threaded, seconds to a minute or
# two. kc23_queue.py runs them in their own slot beside the ONE heavy CPU job, so a gate or a verdict is not starved
# behind hours of ladder or tuning work (the CPU lane is booked for about 16 more hours with c2_ladder_w250 alone).
# They are never "CPU-heavy", they never use more than one process, and a light row never runs two at once.
LIGHT_JOB_IDS = {
    "d0_smoke_gate", "code_c2_inertness", "code_c3_inertness", "code_c5_inertness",   # instant checkpoint checks
    "d1_reproduction_check", "d1_full_aggregate", "d2_reliance", "d5_stats", "s1_gate",
    "c5_siat_verdict", "c5_enabl3s_verdict", "s2b_window_trade", "s3_benchmark", "d3_stats", "d4_stats",
    "d6_sanity_check", "d6_manipulation_check", "s3_inventory", "s2_transition_table", "s2_transitions",
}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "stage", "seed", "command", "out_dir",
                                          "depends_on", "gate_script", "expected_outputs", "light"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "light": "1" if r["job_id"] in LIGHT_JOB_IDS else ""})
    print(f"[write] {path}: {len(rows)} rows")


if __name__ == "__main__":
    write_csv(ROOT / "kc23_jobs_gpu.csv", gpu_rows)
    write_csv(ROOT / "kc23_jobs_cpu.csv", cpu_rows)
    gpu_h = sum(1 for r in gpu_rows if not r["command"].startswith("#")) * 0  # placeholder, hours estimated in report
    print(f"[total] {len(gpu_rows)} GPU rows, {len(cpu_rows)} CPU rows")
