#!/usr/bin/env python3
"""
src/kc23_s3_extra_job_gen.py
==========================
docs/plans/EXPERIMENT_PLAN_KC23_DEPLOYMENT.md S3.2 item 2: the classical families to run on the active-only windows are "LDA, SVM and RF
(plus SVM-X and HGB if KC-C3 lands P2 or P3)". Written 26 September 2026, before any C3 result exists; run by hand once
results/kc23_c3_ensemble/C3_VERDICT.md has a letter (same staging principle as src/kc23_d6_stage2_job_gen.py).

Reads the "**Outcomes: P?, N?, E?**" line of C3_VERDICT.md:
  P2 or P3            -> appends the four CPU rows SVM-X and HGB x {global, per_subject} on the active-only Freq features
                         (results/kc23_s3_svmx_<norm>, results/kc23_s3_hgb_<norm>) and adds them to the s3_benchmark row's
                         depends_on, so the benchmark waits for them
  P1                  -> nothing appended; says the cells are not required
  P-OUT               -> nothing appended; says the plan's condition (P2 or P3) does not cover it and the extra cells are NOT
                         required (Enam, 26 September). src/kc23_s3_benchmark.py states the same in S3_VERDICT.md
  no verdict / no letter line -> exit 2, nothing appended (the condition cannot be decided)

The SVM-X rows use the KC-C3 grid (--grid extended --search grid); every row passes --flush-preds so the DNS to WAK
critical-error rate can be computed from per-window predictions. Idempotent.
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = f'"{ROOT / ".venv" / "Scripts" / "python.exe"}"'
FEAT_AONLY_FREQ = "data/features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_ext.npz"
META_AONLY = "data/features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_meta.csv"
C3_VERDICT = "results/kc23_c3_ensemble/C3_VERDICT.md"
N_SUBJECTS = 40
NORMS = ["global", "per_subject"]


def c3_letters(verdict: Path) -> list[str]:
    if not verdict.exists():
        raise FileNotFoundError(f"{verdict} missing: the S3 extra cells depend on KC-C3's outcome")
    m = re.search(r"\*\*Outcomes:\s*([^*\n]+)\*\*", verdict.read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"{verdict} has no 'Outcomes:' line (KC-C3 produced no letter)")
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def decide(letters: list[str]) -> tuple[bool, str]:
    if {"P2", "P3"} & set(letters):
        return True, "required (KC-C3 landed " + ", ".join(letters) + ")"
    if "P-OUT" in letters:
        return False, ("not required: KC-C3 landed P-OUT, outside the pre-registered grid; the plan's condition is P2 or P3 "
                       "(Enam, 26 September)")
    return False, "not required: KC-C3 landed " + ", ".join(letters) + ", not P2 or P3"


def build_rows() -> list[dict]:
    rows = []
    for fam, model, extra in (("svmx", "SVM", "--grid extended --search grid"), ("hgb", "HGB", "--search random --n-iter 30")):
        for norm in NORMS:
            out = f"results/kc23_s3_{fam}_{norm}"
            rows.append({
                "job_id": f"s3_{fam}_{norm}", "stage": "S3", "seed": "",
                "command": (f'{PY} src/train_classical_loso.py --features {FEAT_AONLY_FREQ} --meta {META_AONLY} --models {model} '
                            f'--norm-mode {norm} {extra} --flush-preds --n-jobs 1 --rf-n-jobs 4 --out {out} --resume'),
                "out_dir": out, "depends_on": "s3_inventory", "gate_script": "",
                "expected_outputs": f"*_nested_loso_subjectwise.csv|{N_SUBJECTS}", "light": ""})
    return rows


def append_rows(cpu_csv: Path, rows: list[dict]) -> int:
    with open(cpu_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        existing = list(r)
    ids = {x["job_id"] for x in existing}
    new = [x for x in rows if x["job_id"] not in ids]
    new_ids = [x["job_id"] for x in rows]
    for x in existing:                                # the benchmark waits for the new cells (idempotent)
        if x["job_id"] == "s3_benchmark":
            deps = [d for d in x["depends_on"].split(";") if d]
            x["depends_on"] = ";".join(deps + [d for d in new_ids if d not in deps])
    with open(cpu_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for x in existing + new:
            w.writerow({k: x.get(k, "") for k in fieldnames})
    return len(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict", default=str(ROOT / C3_VERDICT))
    ap.add_argument("--cpu-csv", default=str(ROOT / "jobs/kc23_jobs_cpu.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        needed, why = decide(c3_letters(Path(args.verdict)))
    except (FileNotFoundError, ValueError) as e:
        print(f"[s3-extra-gen] cannot decide: {e}", file=sys.stderr)
        return 2
    print(f"[s3-extra-gen] SVM-X and HGB cells {why}")
    if not needed:
        return 0
    rows = build_rows()
    if args.dry_run:
        print(f"[s3-extra-gen] --dry-run: would append {len(rows)} rows")
        return 0
    print(f"[s3-extra-gen] appended {append_rows(Path(args.cpu_csv), rows)} new row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
