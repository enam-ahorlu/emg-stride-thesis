#!/usr/bin/env python3
"""
kc23_queue.py
==============
RUN_ORDER_KC23.md Section 2 -- a small resumable runner for the kill-critic
programme's GPU and CPU job queues.

Reads kc23_jobs_gpu.csv and kc23_jobs_cpu.csv (columns: job_id, stage, seed,
command, out_dir, depends_on, gate_script). Runs one GPU job and at most one
CPU job at a time, detached (each job is a subprocess this runner does not
block on beyond polling), logging to _run_logs/kc23/<job_id>.log. Writes
KC23_STATUS.md after every state change: done, running and queued jobs, their
wall-clock, and the last gate outcome. Skips any job whose out_dir already
holds a complete summary (an is_complete() heuristic; see its docstring), and
never deletes anything.

Gate protocol (RUN_ORDER_KC23.md Section 2 and 1): after a job whose
gate_script is set finishes, the runner invokes
`python <gate_script> --out <job.out_dir>` and reads its exit code:
  0  -> continue
  10 -> report (the gate itself already wrote its verdict file / printed the
        letter); continue
  20 -> ESCALATE: halt every job in the SAME stage that has not started, and
        every job elsewhere whose depends_on names a job in this stage.
        Appends a line to KC23_HALT.md. Does not touch KC23_HALT.md's own
        prior content (append-only).
A gate_script that does not exist yet (most of the per-stage stats scripts
are written closer to when each stage actually runs, per the plan) is treated
as "not yet implemented": the runner prints a warning and continues (rc=0
behaviour), rather than crashing the queue. This is a real gap, not silently
papered over -- KC23_STATUS.md marks these rows "gate: NOT IMPLEMENTED".

Usage:
  python kc23_queue.py                      # run the full queue
  python kc23_queue.py --dry-run            # print the schedule, run nothing
  python kc23_queue.py --max-jobs 2         # smoke test: run at most 2 jobs total
  python kc23_queue.py --gpu-csv kc23_jobs_gpu_smoke.csv --cpu-csv kc23_jobs_cpu_smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "_run_logs" / "kc23"
STATUS_MD = ROOT / "KC23_STATUS.md"
HALT_MD = ROOT / "KC23_HALT.md"

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

COMPLETE_GLOBS = ["*summary*.csv", "*_VERDICT.md", "*outcome.json", "*_summary.json",
                  "*subjectwise.csv"]


class Job:
    def __init__(self, row: dict):
        self.job_id = row["job_id"].strip()
        self.stage = row["stage"].strip()
        self.seed = row.get("seed", "").strip()
        self.command = row["command"].strip()
        self.out_dir = row["out_dir"].strip()
        self.depends_on = [d.strip() for d in row.get("depends_on", "").split(";") if d.strip()]
        self.gate_script = row.get("gate_script", "").strip()
        self.status = "queued"     # queued, running, done, failed, skipped(halted), skipped(complete)
        self.wallclock = None
        self.gate_rc = None
        self.proc = None
        self.t0 = None
        self.log_path = None

    def as_row(self):
        return {"job_id": self.job_id, "stage": self.stage, "seed": self.seed,
                "command": self.command, "out_dir": self.out_dir,
                "depends_on": ";".join(self.depends_on), "gate_script": self.gate_script}


def load_jobs(csv_path: Path):
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [Job(row) for row in csv.DictReader(f)]


def is_complete(out_dir: str) -> bool:
    """Heuristic: out_dir exists and holds at least one file matching a
    'this stage produced real output' pattern. Deliberately permissive (a
    false 'incomplete' just re-runs a finished job, wasting time; a false
    'complete' silently skips real work), so gate scripts and downstream
    review are the actual correctness backstop, not this check."""
    d = Path(out_dir)
    if not d.exists():
        return False
    return any(list(d.glob(p)) for p in COMPLETE_GLOBS)


def run_gate(job: "Job") -> int:
    if not job.gate_script:
        return 0
    gate_path = ROOT / job.gate_script
    if not gate_path.exists():
        print(f"[gate] {job.gate_script} does not exist yet -- treating as "
              f"NOT IMPLEMENTED, continuing (rc=0). This stage's outcome "
              f"letter has NOT been computed.")
        job.gate_rc = "NOT_IMPLEMENTED"
        return 0
    try:
        r = subprocess.run([PYTHON, str(gate_path), "--out", job.out_dir],
                           cwd=ROOT, capture_output=True, text=True, timeout=3600)
        print(f"[gate] {job.gate_script} --out {job.out_dir} -> rc={r.returncode}")
        if r.stdout:
            print(r.stdout[-2000:])
        return r.returncode
    except Exception as e:
        print(f"[gate] ERROR running {job.gate_script}: {e}")
        return 20  # fail closed: an unrunnable gate halts dependents rather than silently passing


def append_halt(job: "Job"):
    HALT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(HALT_MD, "a", encoding="utf-8") as f:
        f.write(f"\n## ESCALATE: {job.job_id} (stage {job.stage})\n"
                f"- Time: {datetime.now().isoformat()}\n"
                f"- out_dir: {job.out_dir}\n"
                f"- gate_script: {job.gate_script}\n"
                f"- Every not-yet-started job in stage {job.stage}, and every job "
                f"elsewhere depending on one, is now skipped.\n")
    print(f"[HALT] wrote {HALT_MD} for stage {job.stage}")


def deps_satisfied(job: "Job", by_id: dict, halted_stages: set) -> bool:
    for d in job.depends_on:
        dep = by_id.get(d)
        if dep is None:
            print(f"[warn] {job.job_id} depends_on unknown job_id {d!r}; treating as unsatisfied")
            return False
        if dep.stage in halted_stages:
            return False
        if dep.status != "done":
            return False
    return True


def write_status(gpu_jobs, cpu_jobs, halted_stages, gpu_running, cpu_running):
    lines = ["# KC23 status\n\n", f"Updated: {datetime.now().isoformat()}\n\n"]
    if halted_stages:
        lines.append(f"**Halted stages:** {sorted(halted_stages)}\n\n")
    for name, jobs, running in [("GPU queue", gpu_jobs, gpu_running), ("CPU queue", cpu_jobs, cpu_running)]:
        lines.append(f"## {name}\n\n")
        lines.append("| job_id | stage | seed | status | wallclock_s | gate |\n|---|---|---|---|---|---|\n")
        for j in jobs:
            gate_disp = j.gate_rc if j.gate_rc is not None else ""
            lines.append(f"| {j.job_id} | {j.stage} | {j.seed} | {j.status} | "
                         f"{j.wallclock if j.wallclock is not None else ''} | {gate_disp} |\n")
        lines.append("\n")
    STATUS_MD.write_text("".join(lines), encoding="utf-8")


def launch(job: "Job") -> "Job":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job.job_id}.log"
    Path(job.out_dir).mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== START {datetime.now().isoformat()} ===\n{job.command}\n")
        lf.flush()
        # shell=True on Windows: a plain argv list (even with an absolute .exe
        # path) intermittently failed to resolve here (WinError 2) when the
        # cwd contains spaces; the shell handles quoting/PATH resolution the
        # same way a person typing the command at a prompt would.
        job.proc = subprocess.Popen(job.command, stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=ROOT, shell=True)
    job.t0 = time.time()
    job.log_path = log_path
    job.status = "running"
    print(f"[launch] {job.job_id} -> {log_path} (pid {job.proc.pid})")
    return job


def finish(job: "Job", halted_stages: set):
    ret = job.proc.returncode
    job.wallclock = round(time.time() - job.t0, 1)
    if ret == 0:
        job.status = "done"
        rc = run_gate(job)
        job.gate_rc = rc if job.gate_rc != "NOT_IMPLEMENTED" else job.gate_rc
        if rc == 20:
            halted_stages.add(job.stage)
            append_halt(job)
        elif rc == 10:
            print(f"[gate] {job.job_id}: report (rc=10), continuing")
    else:
        job.status = "failed"
        print(f"[fail] {job.job_id} exited {ret}, wallclock {job.wallclock}s -- see {job.log_path}")


def pick_next(jobs, by_id, halted_stages):
    for job in jobs:
        if job.status != "queued":
            continue
        if job.stage in halted_stages:
            job.status = "skipped(halted)"
            continue
        if is_complete(job.out_dir):
            job.status = "skipped(complete)"
            continue
        if deps_satisfied(job, by_id, halted_stages):
            return job
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-csv", default="kc23_jobs_gpu.csv")
    ap.add_argument("--cpu-csv", default="kc23_jobs_cpu.csv")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true", help="print the schedule, run nothing")
    ap.add_argument("--max-jobs", type=int, default=None, help="stop after this many jobs finish (smoke testing)")
    args = ap.parse_args()

    gpu_jobs = load_jobs(ROOT / args.gpu_csv)
    cpu_jobs = load_jobs(ROOT / args.cpu_csv)
    by_id = {j.job_id: j for j in gpu_jobs + cpu_jobs}
    dup = [jid for jid in by_id if (sum(1 for j in gpu_jobs + cpu_jobs if j.job_id == jid)) > 1]
    if dup:
        sys.exit(f"[kc23-queue] duplicate job_id(s) across the two CSVs: {sorted(set(dup))}")

    print(f"[kc23-queue] {len(gpu_jobs)} GPU jobs, {len(cpu_jobs)} CPU jobs")

    if args.dry_run:
        for kind, jobs in [("GPU", gpu_jobs), ("CPU", cpu_jobs)]:
            print(f"\n=== {kind} queue ===")
            for j in jobs:
                print(f"  {j.job_id:30s} stage={j.stage:10s} seed={j.seed:6s} "
                     f"deps={j.depends_on} gate={j.gate_script}")
        return

    halted_stages = set()
    gpu_running = None
    cpu_running = None
    n_done = 0
    write_status(gpu_jobs, cpu_jobs, halted_stages, gpu_running, cpu_running)

    while True:
        changed = False
        if gpu_running is not None and gpu_running.proc.poll() is not None:
            finish(gpu_running, halted_stages)
            gpu_running = None
            n_done += 1
            changed = True
        if cpu_running is not None and cpu_running.proc.poll() is not None:
            finish(cpu_running, halted_stages)
            cpu_running = None
            n_done += 1
            changed = True

        if gpu_running is None:
            nxt = pick_next(gpu_jobs, by_id, halted_stages)
            if nxt is not None:
                gpu_running = launch(nxt)
                changed = True
        if cpu_running is None:
            nxt = pick_next(cpu_jobs, by_id, halted_stages)
            if nxt is not None:
                cpu_running = launch(nxt)
                changed = True

        if changed:
            write_status(gpu_jobs, cpu_jobs, halted_stages, gpu_running, cpu_running)

        if args.max_jobs is not None and n_done >= args.max_jobs:
            print(f"[kc23-queue] --max-jobs {args.max_jobs} reached, stopping "
                 f"(jobs still running are left to finish on their own; re-run to continue the queue)")
            break

        if gpu_running is None and cpu_running is None:
            remaining = [j for j in gpu_jobs + cpu_jobs if j.status == "queued"]
            if not remaining:
                print("[kc23-queue] all jobs done or skipped")
                break
            runnable = [j for j in remaining
                       if j.stage not in halted_stages and deps_satisfied(j, by_id, halted_stages)]
            if not runnable:
                print(f"[kc23-queue] {len(remaining)} jobs remain queued but none are runnable "
                     f"(blocked by halted stages or unmet dependencies) -- stopping")
                for j in remaining:
                    j.status = "skipped(blocked)"
                write_status(gpu_jobs, cpu_jobs, halted_stages, gpu_running, cpu_running)
                break

        time.sleep(args.poll_interval)

    write_status(gpu_jobs, cpu_jobs, halted_stages, gpu_running, cpu_running)
    done = sum(1 for j in gpu_jobs + cpu_jobs if j.status == "done")
    failed = sum(1 for j in gpu_jobs + cpu_jobs if j.status == "failed")
    print(f"[kc23-queue] done={done} failed={failed} halted_stages={sorted(halted_stages)}")


if __name__ == "__main__":
    main()
