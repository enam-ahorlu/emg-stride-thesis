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
Any other exit code (a gate script crashing -- an uncaught exception, a
missing input file, a CLI contract mismatch -- exits with Python's default
code 1, not 10 or 20) is treated exactly like 20: fail closed. A gate whose
contract broke has told us nothing, and silently continuing past that is
indistinguishable, downstream, from the gate having actually passed -- which
is worse than halting on a result that turns out to be fine. Found live on
2026-09-24: kc23_s2_f0_feasibility.py crashed (rc=1) and was being treated as
a clean pass before this fix.

Restart safety: deps_satisfied() treats a dependency as satisfied when its
status is "done" OR "skipped(complete)" (a resume recognized its out_dir as
already finished, so it never actually ran in THIS process). Found live on
2026-09-24, the first time this queue was ever restarted mid-run: with only
"done" accepted, every job depending on an already-finished-before-restart
run (which resumes as "skipped(complete)", not "done") became permanently
unsatisfiable, and once nothing else was runnable the queue declared the
remaining ~170 jobs "skipped(blocked)" and exited -- a resume, which should
be a no-op, silently halted the whole remaining pipeline instead.
A gate_script that does not exist is a FAILURE (2026-09-25): run_gate returns
20, the stage halts, and KC23_HALT.md says why. It used to be treated as "not
yet implemented" (rc 0, continue), which let a stage pass without its gate ever
running; every gate script now exists and kc23_validate_jobs.py enforces that.

Unattended operation (2026-09-25, for the Task Scheduler runner):
  queue.lock         one runner at a time. A second start exits 3 while the first
                     is alive (pid + create-time + 'kc23_queue' in its cmdline);
                     a lock left by a dead runner is taken over.
  running.json       every launched job (pid, create time). A restarted runner
                     ADOPTS a job whose process is still alive, verified against
                     its create time, instead of relaunching it. A dead one is
                     re-queued, and its --resume plus expected_outputs finish it.
  queue_state.json   halted stages and failed jobs, persisted. Without it an
                     automatic restart would forget an ESCALATE and release the
                     held jobs, and would re-run a job that genuinely failed on
                     every restart (the supervision rule is: report a failure,
                     do not retry it). A human clears an entry with
                     --retry JOB_ID or --clear-halt STAGE.
Completeness on a restart is always decided by expected_outputs, never by a
directory existing.

Usage:
  python kc23_queue.py                      # run the full queue
  python kc23_queue.py --dry-run            # print the schedule, run nothing
  python kc23_queue.py --max-jobs 2         # smoke test: run at most 2 jobs total
  python kc23_queue.py --gpu-csv kc23_jobs_gpu_smoke.csv --cpu-csv kc23_jobs_cpu_smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
        # "<glob-pattern-for-the-summary/subjectwise-file>|<expected-row-count>",
        # e.g. "*subjectwise.csv|40". Empty = no check (aggregators, gates,
        # placeholders, and anything not yet audited for its real output
        # contract -- see kc23_validate_jobs.py's own report of which rows
        # still have this blank).
        self.expected_outputs = row.get("expected_outputs", "").strip()
        # "1" = a LIGHT row (2026-09-25): a read-only aggregator or gate, single-threaded, seconds to a minute. It runs
        # in its own slot beside the one heavy CPU job instead of queueing behind hours of ladder/tuning work, so
        # gates and verdicts are not starved. It never counts as "CPU-heavy" and never exceeds one process.
        self.light = row.get("light", "").strip().lower() in ("1", "true", "yes")
        self.status = "queued"     # queued, running, done, failed, skipped(halted), skipped(complete)
        self.wallclock = None
        self.gate_rc = None
        self.proc = None
        self.t0 = None
        self.log_path = None
        self.output_check_reason = None

    def as_row(self):
        return {"job_id": self.job_id, "stage": self.stage, "seed": self.seed,
                "command": self.command, "out_dir": self.out_dir,
                "depends_on": ";".join(self.depends_on), "gate_script": self.gate_script,
                "expected_outputs": self.expected_outputs, "light": "1" if self.light else ""}


def check_expected_outputs(job: "Job") -> tuple[bool, str]:
    """Closes the "exit 0 having written nothing, or the wrong subject count"
    bug class found repeatedly on 2026-09-24 (LDA silently no-op'ing inside
    train_classical_loso.py; a stub run_scripted_supervised.py; the KC-D1
    reproduction gate's own letter never landing in its verdict file; etc.)
    at the infrastructure level, independent of any one script's own care.
    Returns (ok, reason) -- empty job.expected_outputs means "not audited for
    this yet", not "nothing expected", so it is always ok (no false failures).

    job.expected_outputs is "<glob-pattern>|<spec>": spec is either an int
    row count (a per-subject CSV) or the literal "LETTER" (a gate/stats/
    aggregate row's verdict file, which must exist AND actually contain a
    printed outcome, not just a header -- the exact bug d1_reproduction_check
    hit: D1_VERDICT.md existed but its letter was silently dropped)."""
    if not job.expected_outputs:
        return True, ""
    if "|" not in job.expected_outputs:
        return False, f"malformed expected_outputs {job.expected_outputs!r} (want 'pattern|count-or-LETTER')"
    pattern, spec = job.expected_outputs.rsplit("|", 1)
    out_dir = Path(job.out_dir)
    matches = sorted(out_dir.glob(pattern)) if out_dir.exists() else []
    if not matches:
        return False, f"no file matching {pattern!r} in {out_dir} (declared output missing)"
    if spec == "EXISTS":   # a non-tabular output (an npz, an npy): every match must be a non-empty file
        bad = [m.name for m in matches if not m.is_file() or m.stat().st_size == 0]
        if bad:
            return False, f"declared output(s) empty or not a file: {bad}"
        return True, ""
    if spec == "LETTER":
        import re
        text = matches[0].read_text(encoding="utf-8", errors="replace")
        if not re.search(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*", text):
            return False, f"{matches[0]} has no recognizable '**...: LETTER**' outcome line (header only?)"
        return True, ""
    try:
        expected_n = int(spec)
    except ValueError:
        return False, f"malformed expected_outputs {job.expected_outputs!r}: {spec!r} is not an int or LETTER"
    import pandas as pd
    try:
        n = len(pd.read_csv(matches[0]))
    except Exception as e:
        return False, f"could not read {matches[0]}: {e}"
    if n != expected_n:
        return False, f"{matches[0]} has {n} rows, expected {expected_n}"
    return True, ""


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
        print(f"[gate] {job.gate_script} does not exist -- a gate that cannot run is a failure, not a pass "
              f"(rc=20, this stage's outcome has NOT been computed).")
        return 20
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


def append_halt(job: "Job", held_job_ids: list[str] | None = None, note: str | None = None):
    HALT_MD.parent.mkdir(parents=True, exist_ok=True)
    held = held_job_ids or []
    with open(HALT_MD, "a", encoding="utf-8") as f:
        f.write(f"\n## ESCALATE: {job.job_id} (stage {job.stage})\n"
                f"- Time: {datetime.now().isoformat()}\n"
                f"- out_dir: {job.out_dir}\n"
                f"- gate_script: {job.gate_script}\n"
                + (f"- Note: {note}\n" if note else "")
                + f"- Held dependent jobs ({len(held)}): {held}\n"
                f"- Every not-yet-started job in stage {job.stage}, and every job "
                f"elsewhere depending on one (directly or transitively), is now skipped.\n")
    print(f"[HALT] wrote {HALT_MD} for stage {job.stage}, held {len(held)} dependent job(s)")
    STATUS_NOTE_PATH = HALT_MD.parent / "KC23_STATUS.md"
    if STATUS_NOTE_PATH.exists():
        with open(STATUS_NOTE_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n**ESCALATE** {datetime.now().isoformat()}: {job.job_id} (stage {job.stage}) "
                   f"-- see KC23_HALT.md. Held: {held}\n")


def find_held_dependents(halted_stage: str, all_jobs: list, by_id: dict) -> list[str]:
    """Every QUEUED job whose own stage is halted_stage, plus every QUEUED job
    elsewhere that depends -- directly or transitively -- on ANY job of that
    stage (matching deps_satisfied's own rule: it blocks on `dep.stage in
    halted_stages` regardless of whether that specific dependency has
    already finished -- the triggering job itself is very often already
    "done" by the time its own gate fires, and a dependent naming exactly
    THAT job, not some other still-queued sibling, must still show up here).
    Used to make the KC23_HALT.md entry name exactly what a reader would
    otherwise have to work out from the CSVs by hand."""
    stage_job_ids = {j.job_id for j in all_jobs if j.stage == halted_stage}
    held = {j.job_id for j in all_jobs if j.stage == halted_stage and j.status == "queued"}
    changed = True
    while changed:
        changed = False
        for j in all_jobs:
            if j.job_id in held or j.status != "queued":
                continue
            if any(d in held or d in stage_job_ids for d in j.depends_on):
                held.add(j.job_id)
                changed = True
    return sorted(held)


DONE_STATUSES = {"done", "skipped(complete)"}  # both mean "the output this depends on already exists"


def deps_satisfied(job: "Job", by_id: dict, halted_stages: set) -> bool:
    for d in job.depends_on:
        dep = by_id.get(d)
        if dep is None:
            print(f"[warn] {job.job_id} depends_on unknown job_id {d!r}; treating as unsatisfied")
            return False
        if dep.stage in halted_stages:
            return False
        if dep.status not in DONE_STATUSES:
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
    record_running(job)
    print(f"[launch] {job.job_id} -> {log_path} (pid {job.proc.pid})")
    return job


def finish(job: "Job", halted_stages: set, all_jobs: list, by_id: dict):
    ret = job.proc.returncode
    job.wallclock = round(time.time() - job.t0, 1)
    if ret == 0:
        ok, reason = check_expected_outputs(job)
        if not ok:
            job.status = "failed"
            job.output_check_reason = reason
            print(f"[fail] {job.job_id} exited 0 but failed its expected_outputs check: {reason}")
            return
        job.status = "done"
        rc = run_gate(job)
        job.gate_rc = rc
        note = None
        if rc not in (0, 10, 20):
            note = (f"gate exited rc={rc}, not one of the 0/10/20 contract -- the gate script "
                    f"almost certainly crashed or was miswired, not a data-driven escalation. "
                    f"Fail closed rather than treat this as a silent pass.")
            print(f"[gate] {job.job_id}: {note}")
            rc = 20
        if rc == 20:
            halted_stages.add(job.stage)
            held = find_held_dependents(job.stage, all_jobs, by_id)
            append_halt(job, held, note=note)
        elif rc == 10:
            print(f"[gate] {job.job_id}: report (rc=10), continuing")
    else:
        job.status = "failed"
        print(f"[fail] {job.job_id} exited {ret}, wallclock {job.wallclock}s -- see {job.log_path}")


def already_done(job: "Job") -> bool:
    """The skip rule a restart uses to decide "already done, don't re-run".
    Fixed 2026-09-24: this used to be is_complete()'s bare "any file exists"
    heuristic unconditionally -- how d1_reproduction_check's gate got skipped
    on a restart (the directory had d1_reproduction_inputs.csv and a header-
    only D1_VERDICT.md from an earlier real run, so is_complete() said "done"
    without ever checking the gate had produced a real letter). When
    job.expected_outputs is set, it is now the ONLY check: has the declared
    output actually landed, with the right row count or a real letter. Jobs
    not yet audited (expected_outputs blank) still fall back to the old
    permissive heuristic, so they are not all newly marked incomplete."""
    if job.expected_outputs:
        ok, _ = check_expected_outputs(job)
        return ok
    return is_complete(job.out_dir)


def pick_next(jobs, by_id, halted_stages, light=None):
    """light=None: any job (the GPU lane); False: only heavy jobs; True: only light ones."""
    for job in jobs:
        if job.status != "queued":
            continue
        if light is not None and job.light != light:
            continue
        if job.stage in halted_stages:
            job.status = "skipped(halted)"
            continue
        if already_done(job):
            job.status = "skipped(complete)"
            continue
        if deps_satisfied(job, by_id, halted_stages):
            return job
    return None


class GhostProc:
    """Stands in for subprocess.Popen when a job's process was started by a
    PREVIOUS queue instance (an --adopt job): this process never called
    Popen() on it, so it has no real child handle to poll or reap, only the
    PID to watch. poll() returns None while the PID is still alive; once it
    disappears, returncode is optimistically 0 (there is no way to recover
    the real exit code of a process this instance did not start) -- finish()'s
    check_expected_outputs() is the actual arbiter of whether it succeeded,
    exactly as for any other job."""
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        if self.returncode is None:
            import psutil
            if not psutil.pid_exists(self.pid):
                self.returncode = 0
        return self.returncode


def parse_adopt(spec: str, gpu_jobs: list, cpu_jobs: list, by_id: dict) -> tuple:
    """spec: 'job_id=pid[,job_id=pid...]'. Returns (gpu_running, cpu_running),
    either or both None. Verifies each PID is actually alive (psutil) before
    adopting it -- a stale/wrong PID is reported and refused, not silently
    adopted, since silently believing a dead PID is still running would wedge
    that lane forever."""
    import psutil
    gpu_ids = {j.job_id for j in gpu_jobs}
    cpu_ids = {j.job_id for j in cpu_jobs}
    gpu_running, cpu_running = None, None
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            sys.exit(f"[kc23-queue] --adopt: malformed entry {item!r} (want job_id=pid)")
        jid, pid_str = item.split("=", 1)
        jid = jid.strip()
        try:
            pid = int(pid_str)
        except ValueError:
            sys.exit(f"[kc23-queue] --adopt: {item!r}: {pid_str!r} is not a PID")
        job = by_id.get(jid)
        if job is None:
            sys.exit(f"[kc23-queue] --adopt: unknown job_id {jid!r}")
        if not psutil.pid_exists(pid):
            sys.exit(f"[kc23-queue] --adopt: PID {pid} for {jid!r} is not running -- refusing to adopt "
                     f"a dead PID (it would wedge this lane forever). Check the PID and retry, or omit "
                     f"--adopt if the job has actually already finished.")
        job.status = "running"
        job.proc = GhostProc(pid)
        job.t0 = time.time()
        job.log_path = LOG_DIR / f"{jid}.log"
        print(f"[kc23-queue] adopted {jid} (PID {pid}), will not relaunch it")
        if jid in gpu_ids:
            gpu_running = job
        elif jid in cpu_ids:
            cpu_running = job
    return gpu_running, cpu_running


# ---------------------------------------------------------------------------
# Unattended operation (2026-09-25): lock file, persisted halted/failed state,
# and adoption of jobs still running from a previous runner. See the module
# docstring. All paths are derived from LOG_DIR at call time (tests redirect it).
# ---------------------------------------------------------------------------
def _state_path() -> Path:
    return LOG_DIR / "queue_state.json"


def _running_path() -> Path:
    return LOG_DIR / "running.json"


def _lock_path() -> Path:
    return LOG_DIR / "queue.lock"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"[kc23-queue] {path} is unreadable ({e}). Refusing to start rather than forget what it holds; "
                 f"fix or delete it deliberately.")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _create_time(pid: int):
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _pid_matches(pid, create_time) -> bool:
    """The PID is alive AND is the same process that was recorded (a recycled PID has a different create time)."""
    import psutil
    if not pid or not psutil.pid_exists(int(pid)):
        return False
    if create_time is None:
        return True
    ct = _create_time(int(pid))
    return ct is not None and abs(ct - float(create_time)) < 2.0


def load_state() -> dict:
    state = _read_json(_state_path(), {"halted_stages": [], "failed": {}})
    state.setdefault("halted_stages", [])
    state.setdefault("failed", {})
    return state


def save_state(halted_stages: set, jobs: list) -> None:
    prior = _read_json(_state_path(), {"failed": {}}).get("failed", {})
    failed = {j.job_id: prior.get(j.job_id) or {"reason": j.output_check_reason or "non-zero exit",
                                                "at": datetime.now().isoformat()}
              for j in jobs if j.status == "failed"}
    _write_json(_state_path(), {"halted_stages": sorted(halted_stages), "failed": failed})


def apply_state(state: dict, by_id: dict, halted_stages: set) -> None:
    """Re-impose what the previous runner recorded. A job recorded as failed stays failed (not re-queued) unless
    its declared outputs are now actually complete, in which case it is healed by the normal skip rule."""
    for stage in state["halted_stages"]:
        halted_stages.add(stage)
        print(f"[kc23-queue] stage {stage!r} is HALTED (persisted). Clear it with --clear-halt {stage} after a decision.")
    for jid, info in state["failed"].items():
        job = by_id.get(jid)
        if job is None:
            print(f"[kc23-queue] persisted failure for unknown job {jid!r} ignored")
            continue
        if already_done(job):
            print(f"[kc23-queue] {jid} was recorded as failed but its outputs are now complete; treating it as done")
            continue
        job.status = "failed"
        job.output_check_reason = info.get("reason")
        print(f"[kc23-queue] {jid} is FAILED (persisted: {info.get('reason')}). Not retried; use --retry {jid}.")


def record_running(job: "Job") -> None:
    data = _read_json(_running_path(), {})
    data[job.job_id] = {"pid": job.proc.pid, "create_time": _create_time(job.proc.pid),
                        "started": datetime.now().isoformat()}
    _write_json(_running_path(), data)


def clear_running(job_id: str) -> None:
    data = _read_json(_running_path(), {})
    if job_id in data:
        del data[job_id]
        _write_json(_running_path(), data)


def auto_adopt(by_id: dict, gpu_ids: set, cpu_ids: set, taken: set, light_ids: set = frozenset()):
    """Adopt every job recorded in running.json whose process is still alive (checked against its create time).
    Returns (gpu_running, cpu_running, light_running). Entries for dead processes are dropped: those jobs are simply
    queued again and their --resume plus expected_outputs finish them. Two live jobs in one lane cannot happen in
    normal operation; if it does, refuse to start rather than run two GPU (or two heavy CPU) jobs at once."""
    data = _read_json(_running_path(), {})
    gpu_running, cpu_running, light_running = None, None, None
    for jid, info in list(data.items()):
        job = by_id.get(jid)
        if job is None or not _pid_matches(info.get("pid"), info.get("create_time")):
            print(f"[kc23-queue] running.json: {jid} is no longer running (or unknown); it will be re-queued")
            del data[jid]
            continue
        if jid in taken:
            continue
        job.status = "running"
        job.proc = GhostProc(int(info["pid"]))
        try:
            job.t0 = datetime.fromisoformat(info["started"]).timestamp()
        except Exception:
            job.t0 = time.time()
        job.log_path = LOG_DIR / f"{jid}.log"
        lane = "gpu" if jid in gpu_ids else ("light" if (jid in light_ids or job.light) else "cpu")
        held = {"gpu": gpu_running, "cpu": cpu_running, "light": light_running}[lane]
        if held is not None:
            sys.exit(f"[kc23-queue] running.json holds two live jobs for the {lane} lane (one of them {jid}); "
                     f"refusing to start. Stop one deliberately.")
        if lane == "gpu":
            gpu_running = job
        elif lane == "light":
            light_running = job
        else:
            cpu_running = job
        print(f"[kc23-queue] adopted {jid} (PID {info['pid']}), still running from a previous runner; not relaunching")
    _write_json(_running_path(), data)
    return gpu_running, cpu_running, light_running


def keep_awake(on: bool) -> bool:
    """Ask Windows not to put the machine to sleep while this runner has work (SetThreadExecutionState with
    ES_SYSTEM_REQUIRED; the display may still turn off). It is tied to this thread, so it is released when the runner
    exits or crashes, and it holds under any power scheme. Closing the lid can still sleep a laptop. Returns True if
    Windows accepted the request; False elsewhere (not an error: the machine simply is not asked to stay awake)."""
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0)))
    except Exception:
        return False


def _runner_alive(info: dict) -> bool:
    import psutil
    pid = info.get("pid")
    if not pid or not _pid_matches(pid, info.get("create_time")):
        return False
    try:
        return "kc23_queue" in " ".join(psutil.Process(int(pid)).cmdline())
    except psutil.Error:
        return False


def acquire_lock(pid: int | None = None) -> tuple[bool, str]:
    """One runner at a time. O_EXCL makes two simultaneous starters safe; a lock left by a dead runner (or a
    recycled PID) is taken over."""
    pid = pid or os.getpid()
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": pid, "create_time": _create_time(pid), "started": datetime.now().isoformat()})
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                info = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
            if _runner_alive(info):
                return False, f"another runner is active (PID {info.get('pid')})"
            path.unlink(missing_ok=True)      # stale: its runner is gone
            continue
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        return True, ""
    return False, "could not take the lock"


def release_lock(pid: int | None = None) -> None:
    path = _lock_path()
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if info.get("pid") == (pid or os.getpid()):
        path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-csv", default="kc23_jobs_gpu.csv")
    ap.add_argument("--cpu-csv", default="kc23_jobs_cpu.csv")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true", help="print the schedule, run nothing")
    ap.add_argument("--max-jobs", type=int, default=None, help="stop after this many jobs finish (smoke testing)")
    ap.add_argument("--adopt", default=None,
                    help="'job_id=pid[,job_id=pid]': jobs already running from a PREVIOUS queue "
                         "instance (independent child processes, still alive) -- adopted instead of "
                         "relaunched. See the restart procedure in RUN_ORDER_KC23.md / KC23_STATUS.md.")
    ap.add_argument("--retry", default=None,
                    help="'job_id[,job_id]': forget a persisted failure so the job is queued again")
    ap.add_argument("--clear-halt", default=None,
                    help="'STAGE[,STAGE]': forget a persisted halt after Enam's decision on the escalation")
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

    ok, why = acquire_lock()
    if not ok:
        print(f"[kc23-queue] {why}; exiting without touching anything")
        sys.exit(3)
    try:
        (LOG_DIR / "queue.pid").write_text(str(os.getpid()), encoding="utf-8")
        print(f"[kc23-queue] runner PID {os.getpid()}; keep-awake {'on' if keep_awake(True) else 'unavailable'}")
        run_queue(args, gpu_jobs, cpu_jobs, by_id)
    finally:
        keep_awake(False)
        release_lock()


def run_queue(args, gpu_jobs, cpu_jobs, by_id):
    halted_stages = set()
    gpu_running = None
    cpu_running = None       # the ONE heavy CPU job
    light_running = None     # the ONE light job (a read-only aggregator or gate); never counts as CPU-heavy
    n_done = 0
    state = load_state()
    if args.retry:
        for jid in [x.strip() for x in args.retry.split(",") if x.strip()]:
            state["failed"].pop(jid, None)
            print(f"[kc23-queue] --retry: forgot the persisted failure of {jid}")
    if args.clear_halt:
        state["halted_stages"] = [x for x in state["halted_stages"]
                                  if x not in {y.strip() for y in args.clear_halt.split(",")}]
        print(f"[kc23-queue] --clear-halt {args.clear_halt}: halt forgotten")
    apply_state(state, by_id, halted_stages)
    if args.adopt:
        gpu_running, cpu_running = parse_adopt(args.adopt, gpu_jobs, cpu_jobs, by_id)
        if cpu_running is not None and cpu_running.light:      # an explicitly adopted light job goes to its own slot
            light_running, cpu_running = cpu_running, None
    taken = {j.job_id for j in (gpu_running, cpu_running, light_running) if j is not None}
    a_gpu, a_cpu, a_light = auto_adopt(by_id, {j.job_id for j in gpu_jobs}, {j.job_id for j in cpu_jobs}, taken,
                                       {j.job_id for j in cpu_jobs if j.light})
    gpu_running = gpu_running or a_gpu
    cpu_running = cpu_running or a_cpu
    light_running = light_running or a_light
    save_state(halted_stages, gpu_jobs + cpu_jobs)
    write_status(gpu_jobs, cpu_jobs, halted_stages, gpu_running, cpu_running)

    while True:
        changed = False
        all_jobs = gpu_jobs + cpu_jobs
        if gpu_running is not None and gpu_running.proc.poll() is not None:
            finish(gpu_running, halted_stages, all_jobs, by_id)
            clear_running(gpu_running.job_id)
            gpu_running = None
            n_done += 1
            changed = True
            save_state(halted_stages, all_jobs)
        if cpu_running is not None and cpu_running.proc.poll() is not None:
            finish(cpu_running, halted_stages, all_jobs, by_id)
            clear_running(cpu_running.job_id)
            cpu_running = None
            n_done += 1
            changed = True
            save_state(halted_stages, all_jobs)
        if light_running is not None and light_running.proc.poll() is not None:
            finish(light_running, halted_stages, all_jobs, by_id)
            clear_running(light_running.job_id)
            light_running = None
            n_done += 1
            changed = True
            save_state(halted_stages, all_jobs)

        if gpu_running is None:
            nxt = pick_next(gpu_jobs, by_id, halted_stages)
            if nxt is not None:
                gpu_running = launch(nxt)
                changed = True
        if cpu_running is None:
            nxt = pick_next(cpu_jobs, by_id, halted_stages, light=False)
            if nxt is not None:
                cpu_running = launch(nxt)
                changed = True
        if light_running is None:
            nxt = pick_next(cpu_jobs, by_id, halted_stages, light=True)
            if nxt is not None:
                light_running = launch(nxt)
                changed = True

        if changed:
            write_status(gpu_jobs, cpu_jobs, halted_stages, gpu_running, cpu_running)

        if args.max_jobs is not None and n_done >= args.max_jobs:
            print(f"[kc23-queue] --max-jobs {args.max_jobs} reached, stopping "
                 f"(jobs still running are left to finish on their own; re-run to continue the queue)")
            break

        if gpu_running is None and cpu_running is None and light_running is None:
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
