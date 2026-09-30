#!/usr/bin/env python3
"""
src/kc23_d6_advc_job_gen.py
=========================
docs/plans/EXPERIMENT_PLAN_KC23_DEEP.md 6.2 and 6.6: the ADV-C mechanism arm, the mechanism gate and the secondary contrasts. Written
26 September 2026, before D6 Stage 1 finishes; run by hand once ADV has been run at seeds 42, 7 and 123 (Stage 2), because the
ADV-C lambda_max is "the ADV collapse point: the smallest lambda_max whose realization-mean F1 is 2 pts or more below the ADV
peak, plus the next grid value up", which is unknown until then (same staging principle as src/kc23_d6_stage2_job_gen.py).

  1. Reads the ADV arms (kc23_d6_aggregate: config-checked, divergence retries resolved) and finds the collapse point
     (kc23_d6_aggregate.collapse_point: the smallest lambda_max ABOVE the peak that is 2 pts or more below it).
  2. Collapse found: appends the ADV-C GPU rows, src/run_adv_align_loso.py --adv-mode classcond --oracle-target-labels (the target
     uses its TRUE labels: oracle, diagnostic only, never deployable; every output row carries oracle=True), instrumented with the
     within-class subject probe, at the collapse lambda_max and the next value up, seeds 42, 7 and 123.
     No collapse: appends NO ADV-C row and says so; the mechanism check then reports "ADV-C not run" (plan 6.2), not silence.
  3. Appends two light pseudo-rows to the CPU csv: d6_mechanism_check (waits for the ADV-C rows; the gate writes the
     mechanism letter C-M1/C-M2/C-M3) and d6_secondary_check (ADV at its best lambda_max against R2, R10, R11; ADV-PS at each
     lambda_max against R2).
Idempotent: a job_id already present is never re-appended. Any unreadable input exits 2 and appends nothing.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import kc23_d6_aggregate as agg
from kc23_d6_stage2_job_gen import META_250, NPZ_250, PY, ROOT, STAGE1_DEPENDS, append_rows

SEEDS = agg.OUTCOME_SEEDS
EO = "adv_subjectwise.csv|40"


def advc_rows(knobs: list, seeds=SEEDS) -> list[dict]:
    rows = []
    for sd in seeds:
        for k in knobs:
            out = agg.MECH_DIR["advc"](k, sd)
            rows.append({
                "job_id": f"d6_advc_l{k}_s{sd}", "stage": "D6", "seed": str(sd),
                "command": (f'{PY} src/run_adv_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se --augmentation chandrop '
                            f'--adv-lambda {k} --adv-mode classcond --oracle-target-labels --epochs 40 --batch 256 '
                            f'--instrument {out}/instr --within-class-probe --seed {sd} --out {out} --resume'),
                "out_dir": out, "depends_on": ";".join(STAGE1_DEPENDS), "gate_script": "", "expected_outputs": EO, "light": ""})
    return rows


def light_rows(advc_ids: list[str]) -> list[dict]:
    mech_dep = advc_ids or ["d6_manipulation_check"]
    return [
        {"job_id": "d6_mechanism_check", "stage": "D6-MECH", "seed": "",
         "command": f'{PY} src/kc23_d6_aggregate.py --root . --out results/kc23_d6_mechanism_check --require mechanism',
         "out_dir": "results/kc23_d6_mechanism_check", "depends_on": ";".join(mech_dep), "gate_script": "src/kc23_d6_stats.py",
         "expected_outputs": "*_VERDICT.md|LETTER", "light": "1"},
        {"job_id": "d6_secondary_check", "stage": "D6-SECONDARY", "seed": "",
         "command": f'{PY} src/kc23_d6_aggregate.py --root . --out results/kc23_d6_secondary_check --require secondary',
         "out_dir": "results/kc23_d6_secondary_check", "depends_on": "d6_manipulation_check", "gate_script": "src/kc23_d6_stats.py",
         "expected_outputs": "*_VERDICT.md|LETTER", "light": "1"},
    ]


def plan(root: Path) -> tuple[dict, list[dict], list[dict]]:
    mean_f1, _ = agg.adv_mean_f1(root, SEEDS)
    cp = agg.collapse_point(mean_f1, agg.FAMILY_KNOBS["adv_marginal"])
    if cp["collapse_knob"] is None:
        return cp, [], light_rows([])
    knobs = [cp["collapse_knob"]] + ([cp["next_knob"]] if cp["next_knob"] is not None else [])
    rows = advc_rows(knobs)
    return cp, rows, light_rows([r["job_id"] for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--gpu-csv", default=str(ROOT / "jobs/kc23_jobs_gpu.csv"))
    ap.add_argument("--cpu-csv", default=str(ROOT / "jobs/kc23_jobs_cpu.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        cp, gpu_rows, cpu_rows = plan(Path(args.root))
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[d6-advc-gen] cannot plan (ADV is not complete at seeds {SEEDS}, or a diverged arm has no retry): {e}",
              file=sys.stderr)
        return 2
    print(f"[d6-advc-gen] ADV peak lambda_max {cp['peak_knob']} (F1 {cp['peak_f1']:.4f}); "
          + (f"collapse at {cp['collapse_knob']}, next value up {cp['next_knob']}: {len(gpu_rows)} ADV-C rows"
             if cp["collapse_knob"] is not None else
             "NO COLLAPSE (no larger lambda_max is 2 pts or more below the peak): ADV-C is not run (plan 6.2); the mechanism "
             "check will report that"))
    if args.dry_run:
        print("[d6-advc-gen] --dry-run: nothing written")
        return 0
    append_rows(Path(args.gpu_csv), gpu_rows)
    append_rows(Path(args.cpu_csv), cpu_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
