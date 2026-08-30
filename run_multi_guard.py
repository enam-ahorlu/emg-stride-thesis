# run_multi_guard.py
# ---------------------------------------------------------------------------
# Multi-process memory-guarded supervisor. Runs many independent commands
# (each with its own --resume checkpointing) concurrently, scaling the number
# of concurrent children to available system RAM. Avoids in-script loky/joblib
# parallelism (which deadlocks on repeated GridSearchCV under detached/nested
# processes on Windows); instead we parallelise at the OS-process level with
# each child using --n-jobs 1.
#
# On memory pressure it kills the NEWEST running child (least invested work)
# and requeues it; when RAM recovers it relaunches from the child's checkpoint.
# Crashed children (exit != 0) are also requeued. Exits when all succeed.
#
# Usage:
#   python run_multi_guard.py --jobs-file jobs.txt --max-mem-percent 90 \
#          --min-free-gb 1.5 --max-concurrent 10
# jobs.txt: one full command per line (blank / #-comment lines ignored).
#           Quote paths with spaces (shlex parsing).
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse, shlex, subprocess, sys, time
try:
    import psutil
except ImportError:
    sys.exit("needs psutil: pip install psutil")


def main():
    ap = argparse.ArgumentParser("Multi-process memory-guarded supervisor.")
    ap.add_argument("--jobs-file", required=True)
    ap.add_argument("--max-mem-percent", type=float, default=90.0,
                    help="SOFT gate: stop launching new jobs above this %% used.")
    ap.add_argument("--min-free-gb", type=float, default=1.5,
                    help="SOFT gate: stop launching new jobs below this free GB.")
    ap.add_argument("--kill-mem-percent", type=float, default=95.0,
                    help="HARD limit: only kill a running job (emergency) above this %% used.")
    ap.add_argument("--kill-min-free-gb", type=float, default=0.7,
                    help="HARD limit: only kill a running job (emergency) below this free GB.")
    ap.add_argument("--max-concurrent", type=int, default=10)
    ap.add_argument("--launch-headroom-gb", type=float, default=1.0,
                    help="Only start a new child if free RAM exceeds min-free + this.")
    ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--cooldown", type=float, default=8.0)
    ap.add_argument("--stagger", type=float, default=4.0)
    args = ap.parse_args()

    with open(args.jobs_file) as fh:
        cmds = [l.strip() for l in fh if l.strip() and not l.lstrip().startswith("#")]
    jobs = [{"id": i, "cmd": c, "proc": None, "state": "pending"} for i, c in enumerate(cmds)]
    total = len(jobs)
    tot_gb = psutil.virtual_memory().total / 1e9
    print(f"[mguard] {total} jobs; RAM {tot_gb:.1f}GB; cap {args.max_concurrent} concurrent; "
          f"kill-newest if used>{args.max_mem_percent:.0f}% or free<{args.min_free_gb:.1f}GB", flush=True)

    def launch(j):
        j["proc"] = subprocess.Popen(shlex.split(j["cmd"]))
        j["state"] = "running"
        j["t"] = time.time()
        print(f"[mguard] launch job{j['id']}: ...{j['cmd'][-70:]}", flush=True)

    while any(j["state"] != "done" for j in jobs):
        # reap
        for j in jobs:
            if j["state"] == "running" and j["proc"].poll() is not None:
                rc = j["proc"].returncode
                if rc == 0:
                    j["state"] = "done"
                    print(f"[mguard] job{j['id']} DONE", flush=True)
                else:
                    j["state"] = "pending"
                    print(f"[mguard] job{j['id']} exit {rc}; will resume", flush=True)
        vm = psutil.virtual_memory()
        hard_over = vm.percent >= args.kill_mem_percent or (vm.available / 1e9) <= args.kill_min_free_gb
        soft_over = vm.percent >= args.max_mem_percent or (vm.available / 1e9) <= args.min_free_gb
        running = [j for j in jobs if j["state"] == "running"]
        # Kill only in a true emergency (hard limit) — a running SVM fold is minutes
        # of work, so we never kill on mere soft pressure; we just stop launching more.
        if hard_over and running:
            victim = max(running, key=lambda j: j["t"])  # newest
            print(f"[mguard] EMERGENCY MEM {vm.percent:.0f}% used / {vm.available/1e9:.1f}GB free "
                  f"-> kill newest job{victim['id']} and requeue", flush=True)
            try: victim["proc"].kill()
            except Exception: pass
            victim["state"] = "pending"
            time.sleep(args.cooldown)
            continue
        if not soft_over:
            pend = [j for j in jobs if j["state"] == "pending"]
            if pend and len(running) < args.max_concurrent and \
               (vm.available / 1e9) > (args.min_free_gb + args.launch_headroom_gb):
                launch(pend[0])
                time.sleep(args.stagger)
                continue
        ndone = sum(j["state"] == "done" for j in jobs)
        if int(time.time()) % 30 < args.poll:
            print(f"[mguard] progress {ndone}/{total} done, {len(running)} running, "
                  f"mem {vm.percent:.0f}%/{vm.available/1e9:.1f}GB free", flush=True)
        time.sleep(args.poll)
    print(f"[mguard] ALL {total} JOBS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
