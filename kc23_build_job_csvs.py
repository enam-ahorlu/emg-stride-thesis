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
def _checkpoint(job_id, check_dir):
    cpu_rows.append({"job_id": job_id, "stage": "PHASE1-CHECKPOINT", "seed": "",
                     "command": f'{PY} -c "import sys,pathlib; d=pathlib.Path(r\'{check_dir}\'); '
                                f'sys.exit(0 if d.exists() and any(d.iterdir()) else 1)"',
                     "out_dir": check_dir, "depends_on": "", "gate_script": ""})

_checkpoint("d0_smoke_gate", "results_kc23_d0_capture/after_cpu")
_checkpoint("code_c2_inertness", "results_kc23_c2_inertness_check")
_checkpoint("code_c3_inertness", "results_kc23_c3_after_persubj")
_checkpoint("code_c5_inertness", "results_kc23_c5_inertness_check")


def gpu(job_id, stage, seed, command, out_dir, depends_on=(), gate_script=""):
    gpu_rows.append({"job_id": job_id, "stage": stage, "seed": str(seed) if seed is not None else "",
                     "command": command, "out_dir": out_dir,
                     "depends_on": ";".join(depends_on), "gate_script": gate_script})


def cpu(job_id, stage, seed, command, out_dir, depends_on=(), gate_script=""):
    cpu_rows.append({"job_id": job_id, "stage": stage, "seed": str(seed) if seed is not None else "",
                     "command": command, "out_dir": out_dir,
                     "depends_on": ";".join(depends_on), "gate_script": gate_script})


def instr(out_dir):
    return f"{out_dir}/instr"


# ============================================================================
# #6 (Phase 2, GPU): KC-D1 Tier A, seed 42 re-run (R1-R12) -- the reproduction gate
# ============================================================================
D1_REPRO_GATE = "kc23_d1_reproduction_gate.py"  # not yet written; checks R2/R1/R10 vs published (D1.4)
d1_s42_ids = []

def _d1_arm(arm, out, cmd):
    jid = f"d1_{arm}_s42"
    gpu(jid, "D1", 42, cmd, out, depends_on=("d0_smoke_gate",))
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
gpu_rows[[r["job_id"] for r in gpu_rows].index("d1_r2_s42")]["gate_script"] = D1_REPRO_GATE

# ============================================================================
# #7 (Phase 2, GPU+CPU): KC-S1 scripted buffer, 3 realizations (base training)
# ============================================================================
S1_GATE = "kc23_s1_scripted_stats.py"
for seed in [42, 7, 123]:
    gpu(f"s1_base_s{seed}", "S1", seed,
       f'{PY} run_scripted_supervised.py --npz {NPZ_250} --meta {META_250} --seed {seed} '
       f'--out results_kc23_s1_scripted_s{seed} --resume',
       f"results_kc23_s1_scripted_s{seed}", depends_on=d1_s42_ids,
       gate_script=(S1_GATE if seed == 123 else ""))

# ============================================================================
# #8 (Phase 2, GPU): KC-D5 ENABL3S deep, 5 realizations (E1/E2/E3)
# ============================================================================
NPZ_ENABL3S = "features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60.npz"
META_ENABL3S = "features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_meta.csv"
for seed in D5_SEEDS:
    gpu(f"d5_e1_s{seed}", "D5", seed,
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_ENABL3S} --meta {META_ENABL3S} --arch resnet_se '
       f'--augmentation none --instrument {instr(f"results_kc23_d5_e1_s{seed}")} --seed {seed} '
       f'--out results_kc23_d5_e1_s{seed} --resume', f"results_kc23_d5_e1_s{seed}", depends_on=d1_s42_ids)
    gpu(f"d5_e2_s{seed}", "D5", seed,
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_ENABL3S} --meta {META_ENABL3S} --arch resnet_se '
       f'--augmentation chandrop --aug-chandrop-p 0.2 --save-proba results_kc23_d5_e2_s{seed}/proba '
       f'--model-tag RESNET_SE_CD --instrument {instr(f"results_kc23_d5_e2_s{seed}")} --seed {seed} '
       f'--out results_kc23_d5_e2_s{seed} --resume', f"results_kc23_d5_e2_s{seed}", depends_on=d1_s42_ids)
    gpu(f"d5_e3_s{seed}", "D5", seed,
       f'{PY} run_cnn_arch_loso.py --npz {NPZ_ENABL3S} --meta {META_ENABL3S} --arch resnet_se '
       f'--augmentation gainjitter --aug-gain-sd 0.40 --instrument {instr(f"results_kc23_d5_e3_s{seed}")} '
       f'--seed {seed} --out results_kc23_d5_e3_s{seed} --resume', f"results_kc23_d5_e3_s{seed}",
       depends_on=d1_s42_ids, gate_script=("kc23_d5_replication_stats.py" if seed == D5_SEEDS[-1] else ""))

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
            gpu(f"d1_{arm}_s{seed}", "D1", seed, cmdline, out, depends_on=("d1_r2_s42",))
            d1_all_seed_ids.append(f"d1_{arm}_s{seed}")
        elif arm == "r10":
            out = f"results_kc23_d1_r10_s{seed}"
            gpu(f"d1_r10_s{seed}", "D1", seed,
               f'{PY} run_adabn_cnn_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
               f'--augmentation chandrop --epochs 40 --seed {seed} --out {out} --resume', out,
               depends_on=("d1_r2_s42",))
            d1_all_seed_ids.append(f"d1_r10_s{seed}")
        elif arm == "r11":
            out = f"results_kc23_d1_r11_s{seed}"
            gpu(f"d1_r11_s{seed}", "D1", seed,
               f'{PY} run_deep_coral_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
               f'--augmentation chandrop --coral-lambda 30 --batch 256 --seed {seed} --out {out} --resume', out,
               depends_on=("d1_r2_s42",))
            d1_all_seed_ids.append(f"d1_r11_s{seed}")
    # R8/R9 (SimpleEMGCNN) at this seed
    for arm, aug in [("r8", None), ("r9", "chandrop")]:
        out = f"results_kc23_d1_{arm}_s{seed}"
        augflag = " --augment chandrop --aug-chandrop-p 0.2" if aug else ""
        gpu(f"d1_{arm}_s{seed}", "D1", seed,
           f'{PY} train_cnn_loso.py --npz {NPZ_250} --meta {META_250} --norm-mode per_subject{augflag} '
           f'--seed {seed} --out {out} --resume', out, depends_on=("d1_r2_s42",))
        d1_all_seed_ids.append(f"d1_{arm}_s{seed}")
gpu_rows[[r["job_id"] for r in gpu_rows].index(f"d1_r2_s1001")]["gate_script"] = "kc23_d1_replicate_stats.py"  # D1 headline gate, after the last seed

# ============================================================================
# #10 (Phase 3, CPU): KC-D2 reliance analysis (analysis only, on D1's instrumented runs)
# ============================================================================
cpu("d2_reliance", "D2", None,
   f'{PY} kc23_d2_reliance_analysis.py --out results_kc23_d2_reliance',
   "results_kc23_d2_reliance", depends_on=tuple(d1_all_seed_ids), gate_script="kc23_d2_reliance_stats.py")

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
           f'--out {out} --resume', out, depends_on=("d1_r2_s1001",))
        d4_ids.append(jid)
    for sd in [0.80, 1.00]:
        out = f"results_kc23_d4_gainjitter_sd{sd:.2f}_s{seed}"
        jid = f"d4_gainjitter_sd{sd:.2f}_s{seed}"
        gpu(jid, "D4", seed,
           f'{PY} run_cnn_arch_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
           f'--augmentation gainjitter --aug-gain-sd {sd} --instrument {instr(out)} --seed {seed} '
           f'--out {out} --resume', out, depends_on=("d1_r2_s1001",))
        d4_ids.append(jid)
gpu_rows[[r["job_id"] for r in gpu_rows].index(d4_ids[-1])]["gate_script"] = "kc23_d4_invariance_stats.py"

# ============================================================================
# #11b (Phase 4, GPU): KC-D6 Stage 1 -- every family at seed 42
# ============================================================================
D6_SANITY_GATE = "kc23_d6_sanity_gate.py"
D6_MANIP_GATE = "kc23_d6_manipulation_gate.py"
d6_stage1_ids = []
adv_lambdas = [0, 0.03, 0.1, 0.3, 1, 3, 10]
for lam in adv_lambdas:
    out = f"results_kc23_d6_adv_marginal_l{lam}_s42"
    jid = f"d6_adv_marginal_l{lam}_s42"
    gpu(jid, "D6", 42,
       f'{PY} run_adv_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
       f'--augmentation chandrop --adv-lambda {lam} --adv-mode marginal --epochs 40 --batch 256 '
       f'--seed 42 --out {out} --resume', out,
       depends_on=("d0_smoke_gate", "d1_r2_s42"),
       gate_script=(D6_SANITY_GATE if lam == 0 else ""))
    d6_stage1_ids.append(jid)
sfc_weights = [0.1, 1, 10, 100, 1000]
for w in sfc_weights:
    out = f"results_kc23_d6_sfc_w{w}_s42"
    jid = f"d6_sfc_w{w}_s42"
    gpu(jid, "D6", 42,
       f'{PY} run_deep_coral_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
       f'--augmentation chandrop --coral-lambda {w} --coral-normalize l2 --batch 256 --seed 42 '
       f'--out {out} --resume', out, depends_on=("d1_r2_s42",))
    d6_stage1_ids.append(jid)
for lam in [0.1, 1, 10]:
    out = f"results_kc23_d6_advps_l{lam}_s42"
    jid = f"d6_advps_l{lam}_s42"
    gpu(jid, "D6", 42,
       f'{PY} run_adv_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
       f'--norm-mode per_subject --augmentation chandrop --adv-lambda {lam} --adv-mode marginal '
       f'--epochs 40 --batch 256 --seed 42 --out {out} --resume', out, depends_on=("d1_r2_s42",))
    d6_stage1_ids.append(jid)
gpu_rows[[r["job_id"] for r in gpu_rows].index(d6_stage1_ids[-1])]["gate_script"] = D6_MANIP_GATE
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
   "(11b, gate_script=" + D6_MANIP_GATE + ") decides which families pass (G-PASS/G-WEAK). "
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
           f'--out {out} --resume', out, depends_on=("d1_r2_s1001",))
        d3_ids.append(jid)
gpu_rows[[r["job_id"] for r in gpu_rows].index(d3_ids[-1])]["gate_script"] = "kc23_d3_axis_stats.py"

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
           out, depends_on=("d1_r2_s42",))
        tierb_ids.append(jid)

# ============================================================================
# #14 (Phase 5, GPU+CPU): KC-S3 active-only benchmark
# ============================================================================
NPZ_AONLY = "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly.npz"
META_AONLY = "features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_meta.csv"
cpu("s3_inventory", "S3", None,
   f'{PY} kc23_s3_inventory.py --out results_kc23_s3_active_benchmark',
   "results_kc23_s3_active_benchmark", depends_on=("d1_r2_s42",))
for arch in ["simple", "resnet_se"]:
    for norm in ["global", "per_subject"]:
        out = f"results_kc23_s3_{arch}_{norm}"
        gpu(f"s3_{arch}_{norm}", "S3", 42,
           f'{PY} run_cnn_arch_loso.py --npz {NPZ_AONLY} --meta {META_AONLY} --arch {arch} '
           f'--norm-mode {norm} --seed 42 --out {out} --resume', out,
           depends_on=("s3_inventory",),
           gate_script=("" if (arch, norm) != ("resnet_se", "per_subject") else "kc23_s3_benchmark_stats.py"))

# ============================================================================
# #15 (Phase 5, GPU+CPU): KC-S2 ENABL3S transitions -- gated on F0 feasibility
# ============================================================================
S2_F0_GATE = "kc23_s2_f0_feasibility.py"
cpu("s2_f0_feasibility", "S2-F0", None,
   f'{PY} adapt_external_dataset.py --root 5362627 --out results_kc23_s2_adapter_circuitmeta '
   f'--tag ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_kc23s2 --with-circuit-meta --resume',
   "results_kc23_s2_adapter_circuitmeta", depends_on=(), gate_script=S2_F0_GATE)
cpu("s2_transitions", "S2", None,
   f'{PY} kc23_s2_transitions.py --out results_kc23_s2_transitions',
   "results_kc23_s2_transitions", depends_on=("s2_f0_feasibility",), gate_script="")

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
    cpu(jid, "C2", None, cmdline, out, depends_on=("code_c2_inertness",),
       gate_script=("kc23_c2_whitening_stats.py" if tag == "w400" else ""))
    c2_ids.append(jid)

# ============================================================================
# C-b (Phase 2-5, CPU): KC-C3 classical tuning parity
# ============================================================================
c3_ids = []
for model in ["SVM", "RF", "HGB", "KNN"]:
    for norm in ["per_subject", "global"]:
        out = f"results_kc23_c3_{model.lower()}_{norm}"
        jid = f"c3_{model.lower()}_{norm}"
        search = "grid" if model in ("SVM", "KNN") else "random"
        proba = f" --save-proba --proba-out {out}/proba" if (model == "SVM" and norm == "per_subject") else ""
        cpu(jid, "C3", None,
           f'{PY} train_classical_loso.py --features {FEAT_250_FREQ} --meta {META_250_FEAT} '
           f'--models {model} --norm-mode {norm} --grid extended --search {search} --n-iter 30{proba} '
           f'--n-jobs 1 --rf-n-jobs 4 --out {out} --resume', out, depends_on=("code_c3_inertness",))
        c3_ids.append(jid)
cpu("c3_ensemble", "C3", None,
   "# NEEDS A DATA-PREP STEP FIRST: ensemble_v2_combine.py's --proba-dir points at ONE "
   "directory holding every member's {MODEL}_sub{K:02d}.npz (its existing usage combines "
   "SVM+RF+CNN+RESNET_SE from one shared proba dir). The KC-C3 SVM-X proba "
   "(results_kc23_c3_svm_per_subject/proba/SVM_sub*.npz) and the published ResNet-SE+CD "
   "proba (results_cnn_aug_resnet_se_chandrop_proba/) live in different directories and "
   "must be merged (or symlinked) into one dir first -- not implemented here. Once merged: "
   f'{PY} ensemble_v2_combine.py --proba-dir <merged_dir> --out results_kc23_c3_ensemble',
   "results_kc23_c3_ensemble", depends_on=tuple(c3_ids), gate_script="kc23_c3_tuning_stats.py")

# ============================================================================
# C-c (Phase 2-5, CPU): KC-C4 richer established feature sets
# ============================================================================
cpu("c4_extract", "C4", None,
   f'{PY} kc23_c4_extract_rich.py --out features_out',
   "features_out", depends_on=("c3_ensemble",))
for feat in ["tdpsd54", "rich126"]:
    for model in ["SVM", "LDA"]:
        for norm in ["per_subject", "global"]:
            out = f"results_kc23_c4_{feat}_{model.lower()}_{norm}"
            cpu(f"c4_{feat}_{model.lower()}_{norm}", "C4", None,
               f'{PY} train_classical_loso.py --features features_out/kc23_{feat}_features.npz '
               f'--meta {META_250_FEAT} --models {model} --norm-mode {norm} --out {out} --resume',
               out, depends_on=("c4_extract",),
               gate_script=("kc23_c4_feature_stats.py" if (feat, model, norm) == ("rich126", "LDA", "global") else ""))

# ============================================================================
# C-d (Phase 2-5, CPU plus ~1h GPU): KC-C5 leak decomposition
# ============================================================================
c5_ids = []
for dataset, feat, metaf, tag, time_units in [
        ("siat", FEAT_250_BASE, META_250_FEAT, "kc23_c5_siat", "seconds"),
        ("enabl3s", "features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_ext.npz",
         "features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_meta.csv",
         "kc23_c5_enabl3s", "samples")]:
    out = f"results_kc23_c5_leak_{dataset}"
    tu = f"--time-units {time_units}"
    cpu(f"c5_{dataset}_p50", "C5", None,
       f'{PY} b8_movement_blocked_sd.py --features {feat} --meta {metaf} --models SVM,RF,LDA '
       f'--window-ms 250 --tag {tag} --scheme pooled_random --cv-unit per_subject {tu} --out {out} --resume',
       out, depends_on=("code_c5_inertness",))
    cpu(f"c5_{dataset}_p0", "C5", None,
       f'{PY} b8_movement_blocked_sd.py --features {feat} --meta {metaf} --models SVM,RF,LDA '
       f'--window-ms 250 --tag {tag} --scheme pooled_random_nonoverlap --cv-unit per_subject {tu} --out {out} --resume',
       out, depends_on=("code_c5_inertness",))
    for g in [1, 2, 4, 8, 16]:
        cpu(f"c5_{dataset}_b{g}", "C5", None,
           f'{PY} b8_movement_blocked_sd.py --features {feat} --meta {metaf} --models SVM,RF,LDA '
           f'--window-ms 250 --tag {tag} --scheme blocked --guard-windows {g} --cv-unit per_subject {tu} --out {out} --resume',
           out, depends_on=("code_c5_inertness",))
    for g in [1, 4, 16]:
        cpu(f"c5_{dataset}_i{g}", "C5", None,
           f'{PY} b8_movement_blocked_sd.py --features {feat} --meta {metaf} --models SVM,RF,LDA '
           f'--window-ms 250 --tag {tag} --scheme interleaved --n-chunks 20 --guard-windows {g} '
           f'--cv-unit per_subject {tu} --out {out} --resume', out, depends_on=("code_c5_inertness",))
    jid = f"c5_{dataset}_wb1"
    cpu(jid, "C5", None,
       f'{PY} b8_movement_blocked_sd.py --features {feat} --meta {metaf} --models SVM,RF,LDA '
       f'--window-ms 250 --tag {tag} --scheme blocked --guard-windows 1 --cv-unit per_subject {tu} --out {out} --resume',
       out, depends_on=("code_c5_inertness",),
       gate_script=("kc23_c5_leak_stats.py" if dataset == "enabl3s" else ""))
    c5_ids.append(jid)
gpu("c5_simplecnn_sd", "C5", 42,
   "# NOT YET RUNNABLE: the plan's C5.3 arms table needs b8_cnn_sd.py to support "
   "the same --scheme/--guard-windows options as b8_movement_blocked_sd.py, but C5.2's "
   "code-change instructions name only b8_movement_blocked_sd.py -- b8_cnn_sd.py was NOT "
   "extended in this Phase-1 session (flagged in the report-back as a plan gap, not "
   "silently resolved). Its current CLI (--npz/--meta/--use/--norm/--epochs/--window-ms/"
   "--splits/--out) has no scheme selection at all. Extend it the same way before this job runs.",
   "results_kc23_c5_leak_siat", depends_on=("code_c5_inertness",))

# ============================================================================
# C-e (Phase 2-5, CPU): KC-C6 alignment ladder on ENABL3S
# ============================================================================
cpu("c6_ladder_enabl3s", "C6", None,
   'set "LADDER_FEAT=features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_ext.npz" && '
   'set "LADDER_META=features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_meta.csv" && '
   f'{PY} run_alignment_ladder_loso.py --rungs 3,0,1,2,4,4lw,4o --out results_kc23_c6_ladder_enabl3s '
   '--no-gate --resume', "results_kc23_c6_ladder_enabl3s", depends_on=tuple(c2_ids),
   gate_script="kc23_c6_ladder_stats.py")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "stage", "seed", "command", "out_dir",
                                          "depends_on", "gate_script"])
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {path}: {len(rows)} rows")


if __name__ == "__main__":
    write_csv(ROOT / "kc23_jobs_gpu.csv", gpu_rows)
    write_csv(ROOT / "kc23_jobs_cpu.csv", cpu_rows)
    gpu_h = sum(1 for r in gpu_rows if not r["command"].startswith("#")) * 0  # placeholder, hours estimated in report
    print(f"[total] {len(gpu_rows)} GPU rows, {len(cpu_rows)} CPU rows")
