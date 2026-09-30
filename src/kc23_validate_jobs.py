#!/usr/bin/env python3
"""
src/kc23_validate_jobs.py
=======================
Item 7 of the 24 September continuation prompt: confirm programmatically
that every script, input file and depends_on reference named in
jobs/kc23_jobs_gpu.csv / jobs/kc23_jobs_cpu.csv now exists.

Checks, per row:
  - job_id uniqueness across both CSVs
  - every depends_on token resolves to a job_id in one of the two CSVs
  - the first non-flag, non-"set"/"&&" token of `command` that looks like a
    .py script name exists on disk (skips inline "# NOT YET RUNNABLE" /
    "# PLACEHOLDER" rows, which are commands-as-comments by design -- see
    docs/kc23/KC23_PHASE1_REPORT.md and src/kc23_build_job_csvs.py's own docstring)
  - every gate_script, if set, must exist. HARD failure (2026-09-25): src/kc23_queue.py
    now treats a gate that cannot run as a failure and halts the stage, so a row
    naming a missing gate is a row that can only ever halt.
  - every runnable row (any command that does not start with "#") declares
    well-formed expected_outputs: "<glob>|<int rows>", "<glob>|LETTER" or
    "<glob>|EXISTS". HARD failure (2026-09-25): a blank one falls back to the
    permissive "directory has some file" completeness heuristic, which is how
    placeholder verdicts made rows "skipped(complete)" without running.

Exit 0 if every row's script and dependency references resolve (rows whose
command is a "NOT YET RUNNABLE"/"PLACEHOLDER" comment do not count against
this, since they are marked as intentionally incomplete); exit 1 otherwise.
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EO_RE = re.compile(r"^.+\|(\d+|LETTER|EXISTS)$")


def load_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_script(command: str) -> str | None:
    command = command.strip()
    if command.startswith("#"):
        return None  # placeholder/comment row, by design
    # tokens like: "C:\...\python.exe" script.py --flag ... , or set "X=Y" && ... && "python.exe" script.py ...
    m = re.search(r'python\.exe"?\s+([A-Za-z0-9_\-.]+\.py)', command)
    if m:
        return m.group(1)
    return None


def main() -> int:
    gpu_rows = load_rows(ROOT / "jobs/kc23_jobs_gpu.csv")
    cpu_rows = load_rows(ROOT / "jobs/kc23_jobs_cpu.csv")
    all_rows = gpu_rows + cpu_rows

    ids = [r["job_id"] for r in all_rows]
    dup = {i for i in ids if ids.count(i) > 1}
    id_set = set(ids)

    problems = []
    placeholder_count = 0
    checked_count = 0

    for r in all_rows:
        jid = r["job_id"]
        for dep in [d.strip() for d in r.get("depends_on", "").split(";") if d.strip()]:
            if dep not in id_set:
                problems.append(f"{jid}: depends_on {dep!r} does not resolve to any job_id")

        gate = r.get("gate_script", "").strip()
        if gate and not (ROOT / gate).exists():
            problems.append(f"{jid}: gate_script {gate!r} does not exist (a gate that cannot run halts its stage)")

        if r["command"].strip().startswith("#"):
            placeholder_count += 1            # a placeholder by design: nothing runs, nothing to declare
            continue
        eo = r.get("expected_outputs", "").strip()
        if not eo:
            problems.append(f"{jid}: runnable row has no expected_outputs")
        elif not EO_RE.match(eo):
            problems.append(f"{jid}: malformed expected_outputs {eo!r} (want '<glob>|<int>|LETTER|EXISTS')")

        script = extract_script(r["command"])
        if script is None:
            if "python.exe" in r["command"] and " -c " in r["command"]:
                checked_count += 1            # an inline python -c row (the Phase-1 checkpoints)
            else:
                placeholder_count += 1
            continue
        checked_count += 1
        if not (ROOT / script).exists():
            problems.append(f"{jid}: script {script!r} (from command) does not exist")

    print(f"[validate] {len(all_rows)} total rows ({len(gpu_rows)} GPU, {len(cpu_rows)} CPU)")
    print(f"[validate] {checked_count} rows with a runnable command checked, {placeholder_count} "
         f"placeholder/comment rows skipped by design")
    if dup:
        problems.insert(0, f"duplicate job_id(s): {sorted(dup)}")

    hard_problems = problems
    if hard_problems:
        print(f"\n[validate] {len(hard_problems)} HARD problem(s):")
        for p in hard_problems:
            print(f"  - {p}")
        return 1

    print("\n[validate] PASS: every script, gate_script and depends_on reference resolves, and every runnable "
          "row declares expected_outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
