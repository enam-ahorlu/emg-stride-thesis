#!/usr/bin/env python3
"""
src/kc23_c5_cnn_job_gen.py
=========================
docs/plans/EXPERIMENT_PLAN_KC23_CLASSICAL.md C5.3: "SimpleEMGCNN (already per-subject,
src/b8_cnn_sd.py): P50, P0, B at the plateau guard, and I at the plateau guard."
P50/P0 are already runnable (c5_simplecnn_sd_p50/p0, src/kc23_build_job_csvs.py);
B and I need SIAT's own plateau guard g*, which C5.4's plateau rule computes
deterministically from the classical (SVM/RF/LDA) results -- so this reads
that decision straight out of src/kc23_c5_leak_stats.py's own C5_decomposition.csv
(the g_star field) and appends the real src/b8_cnn_sd.py --scheme blocked/
interleaved --guard-windows <g_star> GPU rows, run by hand once that file
exists (never guessed at beforehand), same staging principle as
kc23_d6_stage2_job_gen.py.

If g_star is None (L3, no plateau by g=16), no CNN B/I rows are generated --
same C5.4 reading as for the classical arms: the decomposition itself is
untrustworthy at that point, not just incomplete.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = f'"{ROOT / ".venv" / "Scripts" / "python.exe"}"'
NPZ_250 = "data/windows/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"
META_250 = "data/features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"


def read_g_star(decomposition_csv: Path) -> int | None:
    df = pd.read_csv(decomposition_csv)
    if "g_star" not in df.columns or df.empty:
        return None
    g = df.iloc[0]["g_star"]
    if pd.isna(g):
        return None
    return int(g)


def build_rows(g_star: int) -> list[dict]:
    rows = []
    for scheme, tag in [("blocked", "b"), ("interleaved", "i")]:
        jid = f"c5_simplecnn_sd_{tag}{g_star}"
        out = f"results/kc23_c5_simplecnn_siat_{tag}{g_star}"
        extra = " --n-chunks 20" if scheme == "interleaved" else ""
        cmd = (f'{PY} src/b8_cnn_sd.py --npz {NPZ_250} --meta {META_250} --scheme {scheme} '
              f'--guard-windows {g_star}{extra} --out {out} --resume')
        rows.append({"job_id": jid, "stage": "C5", "seed": "42", "command": cmd, "out_dir": out,
                    "depends_on": "code_c5_inertness", "gate_script": "", "expected_outputs": ""})
    return rows


def append_rows(gpu_csv: Path, rows: list[dict]) -> int:
    existing_ids = set()
    fieldnames = ["job_id", "stage", "seed", "command", "out_dir", "depends_on", "gate_script",
                 "expected_outputs"]
    if gpu_csv.exists():
        with open(gpu_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            fieldnames = r.fieldnames or fieldnames
            for row in r:
                existing_ids.add(row["job_id"])
    new_rows = [r for r in rows if r["job_id"] not in existing_ids]
    skipped = len(rows) - len(new_rows)
    if skipped:
        print(f"[c5-cnn-gen] {skipped} row(s) already present, not re-appended (idempotent)")
    if new_rows:
        with open(gpu_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            for r in new_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[c5-cnn-gen] appended {len(new_rows)} new row(s) to {gpu_csv}")
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decomposition-csv", required=True,
                    help="results/kc23_c5_leak_siat/C5_decomposition.csv, from src/kc23_c5_leak_stats.py")
    ap.add_argument("--gpu-csv", default=str(ROOT / "jobs/kc23_jobs_gpu.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    g_star = read_g_star(Path(args.decomposition_csv))
    if g_star is None:
        print("[c5-cnn-gen] no plateau (g_star is None, C5.4 outcome L3) -- nothing to append")
        return 0
    rows = build_rows(g_star)
    if args.dry_run:
        print(f"[c5-cnn-gen] --dry-run: would append {len(rows)} row(s) at g*={g_star}, nothing written")
        return 0
    append_rows(Path(args.gpu_csv), rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
