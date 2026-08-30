# run_with_memory_guard.py
# ---------------------------------------------------------------------------
# Memory-guarded supervisor for the long LOSO experiments.
#
# Automates exactly what was previously done by hand: launch an experiment,
# watch system memory, and if it gets dangerously high, kill the run and
# relaunch it with --resume so it continues from the last checkpointed subject
# instead of starting over. It also relaunches automatically if the child
# crashes (e.g. an out-of-memory BrokenProcessPool). Loops until the child
# finishes successfully.
#
# Requires: pip install psutil
#
# Usage (put the real command after a literal `--`):
#   python run_with_memory_guard.py --max-mem-percent 88 -- ^
#       python run_streaming_norm_loso.py ^
#       --features features_out/freq_..._features_ext.npz ^
#       --meta     features_out/freq_..._features_meta.csv ^
#       --configs transductive,calib25,calib50,calib100,running --models SVM,RF --resume
#
# (^ is the PowerShell line-continuation; or just put it all on one line.)
# The wrapped script MUST support --resume (the experiment scripts do). If you
# forget it, the guard adds it for you.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
import time
import subprocess

try:
    import psutil
except ImportError:
    sys.exit("This supervisor needs psutil:  pip install psutil")


def proc_tree_rss_gb(pid):
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
        return sum(pr.memory_info().rss for pr in procs if pr.is_running()) / 1e9
    except psutil.NoSuchProcess:
        return 0.0


def kill_tree(pid):
    try:
        p = psutil.Process(pid)
        kids = p.children(recursive=True)
        for c in kids:
            try: c.terminate()
            except psutil.NoSuchProcess: pass
        try: p.terminate()
        except psutil.NoSuchProcess: pass
        gone, alive = psutil.wait_procs([p] + kids, timeout=10)
        for pr in alive:
            try: pr.kill()
            except psutil.NoSuchProcess: pass
    except psutil.NoSuchProcess:
        pass


def main():
    ap = argparse.ArgumentParser("Memory-guarded auto-restart supervisor.")
    ap.add_argument("--max-mem-percent", type=float, default=90.0,
                    help="Kill+restart the child if system RAM usage exceeds this %% (default 90).")
    ap.add_argument("--min-free-gb", type=float, default=1.0,
                    help="Kill+restart if available RAM drops below this many GB (default 1.0).")
    ap.add_argument("--poll", type=float, default=3.0, help="Seconds between memory checks.")
    ap.add_argument("--cooldown", type=float, default=8.0,
                    help="Seconds to wait after a kill so the OS reclaims memory before restarting.")
    ap.add_argument("--max-restarts", type=int, default=1000)
    ap.add_argument("--max-child-gb", type=float, default=0.0,
                    help="Also kill+restart if the child (process tree) itself grows past this "
                         "many GB of RSS, regardless of system-wide free memory (0=disabled). "
                         "Needed because Windows memory compression can keep system-wide free "
                         "memory looking fine while a single process's own commit balloons "
                         "unboundedly -- the system-wide checks alone can miss that entirely.")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- <command to run>  (e.g. -- python run_streaming_norm_loso.py ... --resume)")
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.exit("Nothing to run. Put the command after a literal `--`.")
    if "--resume" not in cmd:
        print("[guard] WARNING: '--resume' not in the command; adding it so restarts continue "
              "instead of starting from scratch.")
        cmd = cmd + ["--resume"]

    total_gb = psutil.virtual_memory().total / 1e9
    print(f"[guard] system RAM: {total_gb:.1f} GB total. "
          f"Will restart child if usage > {args.max_mem_percent:.0f}% or free < {args.min_free_gb:.1f} GB.")

    restarts = 0
    while restarts <= args.max_restarts:
        print(f"\n[guard] launching (attempt {restarts + 1}): {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        killed = False
        ret = None
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            vm = psutil.virtual_memory()
            child_gb = proc_tree_rss_gb(proc.pid) if args.max_child_gb > 0 else 0.0
            sys_high = vm.percent >= args.max_mem_percent or (vm.available / 1e9) <= args.min_free_gb
            child_high = args.max_child_gb > 0 and child_gb >= args.max_child_gb
            if sys_high or child_high:
                if not args.max_child_gb:
                    child_gb = proc_tree_rss_gb(proc.pid)
                reason = "system-wide" if sys_high else "child process itself"
                print(f"[guard] MEMORY HIGH ({reason}): {vm.percent:.0f}% used, {vm.available/1e9:.1f} GB free "
                      f"(child ~{child_gb:.1f} GB). Killing and will resume from last checkpoint.")
                kill_tree(proc.pid)
                killed = True
                break
            time.sleep(args.poll)

        if not killed and ret == 0:
            print("[guard] child finished successfully. All done.")
            return 0
        if not killed and ret != 0:
            print(f"[guard] child exited with code {ret} (likely crash/OOM). Relaunching with --resume.")
        restarts += 1
        print(f"[guard] cooldown {args.cooldown:.0f}s for memory to free, then restart #{restarts} ...")
        time.sleep(args.cooldown)

    print("[guard] reached --max-restarts; stopping. Re-run to continue from the checkpoints.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
