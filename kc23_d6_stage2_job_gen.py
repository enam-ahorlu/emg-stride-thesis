#!/usr/bin/env python3
"""
kc23_d6_stage2_job_gen.py
============================
RUN_ORDER_KC23.md item 11c: KC-D6 Stage 2 (seeds 7 and 123), passing families
only. Written 2026-09-24, per the KC23 audit -- run by hand once 11b's
manipulation gate (kc23_d6_stats.py) has produced a real D6_gates.csv, never
guessed at beforehand.

Reads a D6_gates.csv (rows: item, letter, ...; item = "manipulation_<family>",
letter in {G-PASS, G-WEAK, G-FAIL}) and appends, for every family whose letter
is G-PASS or G-WEAK, the seed-7 and seed-123 GPU job rows repeating that
family's exact Stage-1 knob sweep -- same command template, --seed swapped,
depends_on unchanged. G-FAIL families get no Stage-2 rows. Idempotent: a job_id
already present in the target CSV is never re-appended.

Family knob tables (must match kc23_build_job_csvs.py's Stage-1 generation
exactly, since Stage 2 repeats the same sweep at new seeds):
  adv_marginal: run_adv_align_loso.py --adv-mode marginal, lambda in
                {0, 0.03, 0.1, 0.3, 1, 3, 10}
  sfc:          run_deep_coral_align_loso.py --coral-normalize l2, weight in
                {0.1, 1, 10, 100, 1000}
  advps:        run_adv_align_loso.py --norm-mode per_subject --adv-mode
                marginal, lambda in {0.1, 1, 10}
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = f'"{ROOT / ".venv" / "Scripts" / "python.exe"}"'
NPZ_250 = "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"
META_250 = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
STAGE1_DEPENDS = ("d0_smoke_gate", "d1_r2_s42")  # matches kc23_build_job_csvs.py's Stage-1 wiring

FAMILY_SPECS = {
    "adv_marginal": {
        "knobs": [0, 0.03, 0.1, 0.3, 1, 3, 10],
        "job_id": lambda lam, seed: f"d6_adv_marginal_l{lam}_s{seed}",
        "out_dir": lambda lam, seed: f"results_kc23_d6_adv_marginal_l{lam}_s{seed}",
        "command": lambda lam, seed, out, extra="": (
            f'{PY} run_adv_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
            f'--augmentation chandrop --adv-lambda {lam} --adv-mode marginal --epochs 40 --batch 256 '
            f'--instrument {out}/instr --within-class-probe{extra} --seed {seed} --out {out} --resume'),
    },
    "sfc": {
        "knobs": [0.1, 1, 10, 100, 1000],
        "job_id": lambda w, seed: f"d6_sfc_w{w}_s{seed}",
        "out_dir": lambda w, seed: f"results_kc23_d6_sfc_w{w}_s{seed}",
        "command": lambda w, seed, out, extra="": (
            f'{PY} run_deep_coral_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
            f'--augmentation chandrop --coral-lambda {w} --coral-normalize l2 --batch 256 --seed {seed} '
            f'--instrument {out}/instr{extra} --out {out} --resume'),
    },
    "advps": {
        "knobs": [0.1, 1, 10],
        "job_id": lambda lam, seed: f"d6_advps_l{lam}_s{seed}",
        "out_dir": lambda lam, seed: f"results_kc23_d6_advps_l{lam}_s{seed}",
        "command": lambda lam, seed, out, extra="": (
            f'{PY} run_adv_align_loso.py --npz {NPZ_250} --meta {META_250} --arch resnet_se '
            f'--norm-mode per_subject --augmentation chandrop --adv-lambda {lam} --adv-mode marginal '
            f'--epochs 40 --batch 256 --seed {seed} --instrument {out}/instr{extra} --out {out} --resume'),
    },
}
EO = {"adv_marginal": "adv_subjectwise.csv|40", "sfc": "deep_coral_subjectwise.csv|40", "advps": "adv_subjectwise.csv|40"}
STAGE2_SEEDS = [7, 123]


def passing_families(gates_csv: Path) -> dict[str, str]:
    """Returns {family: letter} for every family whose letter is G-PASS or
    G-WEAK. A family with no manipulation_ row, or with G-FAIL, is excluded."""
    df = pd.read_csv(gates_csv)
    out = {}
    for _, row in df.iterrows():
        item = str(row["item"])
        if not item.startswith("manipulation_"):
            continue
        family = item[len("manipulation_"):]
        letter = str(row["letter"])
        if letter in ("G-PASS", "G-WEAK"):
            out[family] = letter
    return out


def build_stage2_rows(families: dict[str, str]) -> list[dict]:
    rows = []
    for family, letter in families.items():
        spec = FAMILY_SPECS.get(family)
        if spec is None:
            print(f"[d6-stage2-gen] WARNING: unknown family {family!r} in D6_gates.csv "
                 f"(no knob table here) -- skipped, not guessed", file=sys.stderr)
            continue
        for seed in STAGE2_SEEDS:
            for knob in spec["knobs"]:
                jid = spec["job_id"](knob, seed)
                out = spec["out_dir"](knob, seed)
                cmd = spec["command"](knob, seed, out)
                rows.append({"job_id": jid, "stage": "D6", "seed": str(seed), "command": cmd,
                            "out_dir": out, "depends_on": ";".join(STAGE1_DEPENDS), "gate_script": "",
                            "expected_outputs": EO[family]})
        print(f"[d6-stage2-gen] {family} ({letter}): {len(spec['knobs'])} knobs x "
              f"{len(STAGE2_SEEDS)} seeds appended", flush=True)
    return rows


def outcome_row(rows: list[dict], gates_csv: str) -> dict:
    """The D6 outcome gate as a light pseudo-row: aggregates the seeds 42/7/123 arms of every passing family
    (kc23_d6_aggregate.py --require outcome --gates ...) and the gate writes D6_VERDICT.md, which the queue then checks
    for a letter. It waits for every Stage-2 row."""
    return {"job_id": "d6_outcome_check", "stage": "D6-OUTCOME", "seed": "",
            "command": f'{PY} kc23_d6_aggregate.py --root . --out results_kc23_d6_outcome_check --require outcome --gates {gates_csv}',
            "out_dir": "results_kc23_d6_outcome_check", "depends_on": ";".join(r["job_id"] for r in rows),
            "gate_script": "kc23_d6_stats.py", "expected_outputs": "*_VERDICT.md|LETTER", "light": "1"}


def append_rows(gpu_csv: Path, rows: list[dict]) -> int:
    existing_ids = set()
    fieldnames = ["job_id", "stage", "seed", "command", "out_dir", "depends_on", "gate_script", "expected_outputs", "light"]
    if gpu_csv.exists():
        with open(gpu_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing_ids.add(r["job_id"])
                fieldnames = list(r.keys())
    new_rows = [r for r in rows if r["job_id"] not in existing_ids]
    skipped = len(rows) - len(new_rows)
    if skipped:
        print(f"[d6-stage2-gen] {skipped} row(s) already present, not re-appended (idempotent)")
    if new_rows:
        with open(gpu_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            for r in new_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[d6-stage2-gen] appended {len(new_rows)} new row(s) to {gpu_csv}")
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gates-csv", required=True, help="D6_gates.csv from kc23_d6_stats.py's --out")
    ap.add_argument("--gpu-csv", default=str(ROOT / "kc23_jobs_gpu.csv"))
    ap.add_argument("--cpu-csv", default=str(ROOT / "kc23_jobs_cpu.csv"),
                    help="the light outcome pseudo-row goes here, to the light lane, not the GPU lane")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    families = passing_families(Path(args.gates_csv))
    if not families:
        print("[d6-stage2-gen] no family passed (G-PASS/G-WEAK) -- nothing to append")
        return 0
    rows = build_stage2_rows(families)
    outcome = outcome_row(rows, args.gates_csv)
    if args.dry_run:
        print(f"[d6-stage2-gen] --dry-run: would append {len(rows)} GPU row(s) and the outcome row, nothing written")
        return 0
    append_rows(Path(args.gpu_csv), rows)
    append_rows(Path(args.cpu_csv), [outcome])
    return 0


if __name__ == "__main__":
    sys.exit(main())
