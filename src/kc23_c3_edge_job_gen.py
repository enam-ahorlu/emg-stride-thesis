#!/usr/bin/env python3
"""
src/kc23_c3_edge_job_gen.py
=========================
docs/plans/EXPERIMENT_PLAN_KC23_CLASSICAL.md KC-C3 edge rule: "if the selected SVM C or gamma sits on a grid edge in more than 10 of 40
folds, extend that axis by two steps once and rerun; report both runs." Written 26 September 2026, before any C3 result
exists; run by hand once results/kc23_c3_ensemble/C3_edge_hits.csv exists (src/kc23_c3_tuning_stats.py writes it and states
whether the rule triggered), the same staging principle as kc23_d6_stage2_job_gen.py.

For each normalization with a TRIGGERED axis it appends one CPU row, c3_svm_<norm>_edge, that reruns SVM-X with that axis
extended by two steps on the side(s) that hit the edge, once:
  C grid          base 0.01 0.03 0.1 0.3 1 3 10 30           +2 up: 100, 300      +2 down: 0.003, 0.001
  gamma multiple  base 0.01 0.1 0.3 1 3 10 (x the fitted scale)  +2 up: 30, 100    +2 down: 0.003, 0.001
The rerun writes results/kc23_c3_svm_<norm>_edge/ (src/train_classical_loso.py --svm-c-grid / --svm-gamma-mult-grid; the
svm_extended_gamma.csv sidecar records the grids), and src/kc23_c3_tuning_stats.py then reports BOTH runs and recomputes the letters
with the rerun in place of the base run. An axis whose extension was already applied is never extended again. Nothing is
appended when the rule did not trigger; a missing or malformed edge table exits 2 with nothing appended. Idempotent.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = f'"{ROOT / ".venv" / "Scripts" / "python.exe"}"'
FEAT_250_FREQ = "data/features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META_250_FEAT = "data/features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
EDGE_TABLE = "results/kc23_c3_ensemble/C3_edge_hits.csv"
BASE_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
BASE_GAMMA = [0.01, 0.1, 0.3, 1, 3, 10]
EXTEND = {"C": {"up": [100, 300], "down": [0.003, 0.001]},
          "gamma multiplier": {"up": [30, 100], "down": [0.003, 0.001]}}
COLUMNS = {"norm", "axis", "low_edge", "folds_at_low_edge", "high_edge", "folds_at_high_edge", "triggered",
           "extension_already_applied"}
MORE_THAN = 10
N_SUBJECTS = 40


def extended_grids(rows: pd.DataFrame) -> tuple[list[float], list[float]]:
    """rows: the edge-table rows of ONE normalization. Returns (C grid, gamma-multiplier grid)."""
    grids = {"C": list(BASE_C), "gamma multiplier": list(BASE_GAMMA)}
    for r in rows.itertuples():
        if not bool(r.triggered) or bool(r.extension_already_applied):
            continue
        g = grids[r.axis]
        if r.folds_at_high_edge > MORE_THAN:
            g += EXTEND[r.axis]["up"]
        if r.folds_at_low_edge > MORE_THAN:
            g += EXTEND[r.axis]["down"]
        grids[r.axis] = sorted(set(g))
    return grids["C"], grids["gamma multiplier"]


def _fmt(vals) -> str:
    return ",".join(f"{v:g}" for v in vals)


def build_rows(table: pd.DataFrame) -> list[dict]:
    rows = []
    for norm in ("per_subject", "global"):
        t = table[table["norm"] == norm]
        active = t[t["triggered"].astype(bool) & ~t["extension_already_applied"].astype(bool)]
        if active.empty:
            continue
        c_grid, g_grid = extended_grids(t)
        out = f"results/kc23_c3_svm_{norm}_edge"
        rows.append({
            "job_id": f"c3_svm_{norm}_edge", "stage": "C3", "seed": "",
            "command": (f'{PY} src/train_classical_loso.py --features {FEAT_250_FREQ} --meta {META_250_FEAT} --models SVM '
                        f'--norm-mode {norm} --grid extended --search grid --svm-c-grid {_fmt(c_grid)} '
                        f'--svm-gamma-mult-grid {_fmt(g_grid)} --n-jobs 1 --out {out} --resume'),
            "out_dir": out, "depends_on": f"c3_svm_{norm}", "gate_script": "",
            "expected_outputs": f"*_nested_loso_subjectwise.csv|{N_SUBJECTS}", "light": ""})
    return rows


def append_rows(cpu_csv: Path, rows: list[dict]) -> int:
    with open(cpu_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        existing = list(r)
    ids = {x["job_id"] for x in existing}
    new = [x for x in rows if x["job_id"] not in ids]
    if new:
        with open(cpu_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for x in existing + new:
                w.writerow({k: x.get(k, "") for k in fieldnames})
    return len(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge-table", default=str(ROOT / EDGE_TABLE))
    ap.add_argument("--cpu-csv", default=str(ROOT / "jobs/kc23_jobs_cpu.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    p = Path(args.edge_table)
    try:
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run the C3 gate first")
        table = pd.read_csv(p)
        if not COLUMNS <= set(table.columns) or table.empty:
            raise ValueError(f"{p.name} lacks {sorted(COLUMNS - set(table.columns))} or is empty")
    except (FileNotFoundError, ValueError) as e:
        print(f"[c3-edge-gen] cannot decide: {e}", file=sys.stderr)
        return 2
    rows = build_rows(table)
    if not rows:
        print("[c3-edge-gen] the edge rule did not trigger (or its extension was already applied): nothing to append")
        return 0
    for r in rows:
        print(f"[c3-edge-gen] {r['job_id']}: {r['command'].split('--svm-c-grid')[1].split('--n-jobs')[0].strip()}")
    if args.dry_run:
        print(f"[c3-edge-gen] --dry-run: would append {len(rows)} row(s)")
        return 0
    print(f"[c3-edge-gen] appended {append_rows(Path(args.cpu_csv), rows)} new row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
