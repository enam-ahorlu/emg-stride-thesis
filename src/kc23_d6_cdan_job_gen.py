#!/usr/bin/env python3
"""
src/kc23_d6_cdan_job_gen.py
=========================
docs/plans/EXPERIMENT_PLAN_KC23_DEEP.md 6.2: "ADV-CDAN (optional, deployable): conditional adversary on the multilinear map of embedding and
predicted class probabilities (Long et al., 2018); no target labels. Same lambda_max as ADV-C. Run only if ADV-C lands C-M1."
Written 26 September 2026, before D6 Stage 1 finishes; run by hand once results/kc23_d6_mechanism_check/D6_VERDICT.md exists.

  ADV-C landed C-M1  -> appends the ADV-CDAN GPU rows (src/run_adv_align_loso.py --adv-mode cdan, NO oracle flag) at the same lambda_max
                        values as ADV-C, seeds 42, 7 and 123, and one light pseudo-row, d6_mechanism_cdan_check, that re-runs the
                        mechanism aggregator once they are done (it tabulates CDAN beside ADV and ADV-C, no letter)
  C-M2 or C-M3      -> nothing appended; says CDAN is only run after C-M1
  C-NOT-RUN         -> nothing appended; ADV has no collapse, so there is no ADV-C and no CDAN
  no verdict / no mechanism line -> exit 2, nothing appended
Idempotent.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

import kc23_d6_aggregate as agg
from kc23_d6_stage2_job_gen import META_250, NPZ_250, PY, ROOT, STAGE1_DEPENDS, append_rows

SEEDS = agg.OUTCOME_SEEDS
MECH_DIR = "results/kc23_d6_mechanism_check"


def mechanism_letter(check_dir: Path) -> str:
    v = check_dir / "D6_VERDICT.md"
    if not v.exists():
        raise FileNotFoundError(f"{v} missing: run the mechanism check first")
    m = re.search(r"\*\*mechanism:\s*([A-Za-z0-9-]+)", v.read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"{v} has no '**mechanism: <letter>**' line (the check produced no letter)")
    return m.group(1)


def cdan_rows(knobs: list, seeds=SEEDS) -> list[dict]:
    rows = []
    for sd in seeds:
        for k in knobs:
            out = agg.MECH_DIR["cdan"](k, sd)
            rows.append({
                "job_id": f"d6_cdan_l{k}_s{sd}", "stage": "D6", "seed": str(sd),
                "command": (f'{PY} src/run_adv_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se --augmentation chandrop '
                            f'--adv-lambda {k} --adv-mode cdan --epochs 40 --batch 256 --instrument {out}/instr '
                            f'--within-class-probe --seed {sd} --out {out} --resume'),
                "out_dir": out, "depends_on": ";".join(STAGE1_DEPENDS), "gate_script": "",
                "expected_outputs": "adv_subjectwise.csv|40", "light": ""})
    return rows


def check_row(cdan_ids: list[str]) -> dict:
    return {"job_id": "d6_mechanism_cdan_check", "stage": "D6-MECH", "seed": "",
            "command": f'{PY} src/kc23_d6_aggregate.py --root . --out results/kc23_d6_mechanism_cdan_check --require mechanism',
            "out_dir": "results/kc23_d6_mechanism_cdan_check", "depends_on": ";".join(cdan_ids), "gate_script": "src/kc23_d6_stats.py",
            "expected_outputs": "*_VERDICT.md|LETTER", "light": "1"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--gpu-csv", default=str(ROOT / "jobs/kc23_jobs_gpu.csv"))
    ap.add_argument("--cpu-csv", default=str(ROOT / "jobs/kc23_jobs_cpu.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    try:
        letter = mechanism_letter(root / MECH_DIR)
        if letter != "C-M1":
            why = ("ADV has no collapse, so there is no ADV-C and no CDAN" if letter == "C-NOT-RUN"
                   else f"ADV-C landed {letter}, not C-M1")
            print(f"[d6-cdan-gen] ADV-CDAN not run: {why} (plan 6.2: only if ADV-C lands C-M1)")
            return 0
        meta = pd.read_csv(root / MECH_DIR / "d6_mechanism_meta.csv").iloc[0]
        knobs = [agg._orig_knob("adv_marginal", meta["collapse_knob"])]
        if not pd.isna(meta["next_knob"]):
            knobs.append(agg._orig_knob("adv_marginal", meta["next_knob"]))
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[d6-cdan-gen] cannot decide: {e}", file=sys.stderr)
        return 2
    rows = cdan_rows(knobs)
    print(f"[d6-cdan-gen] ADV-C landed C-M1: {len(rows)} ADV-CDAN rows at lambda_max {knobs}")
    if args.dry_run:
        print("[d6-cdan-gen] --dry-run: nothing written")
        return 0
    append_rows(Path(args.gpu_csv), rows)
    append_rows(Path(args.cpu_csv), [check_row([r["job_id"] for r in rows])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
