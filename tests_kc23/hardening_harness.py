"""Stand-in for the queue runner, used by test_kc23_queue_hardening.py. It uses the queue's real spawn_detached (and, with
--daemon, the queue's real redirect_output_and_ignore_console_signals) to start one long-lived 'job', records the pids and its
console window handle in a JSON file, then idles like a runner would. --old-flags reproduces the pre-hardening launch (the job
shares the runner's console and process group) as the control."""
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_queue as q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--old-flags", action="store_true")
    ap.add_argument("--seconds", type=int, default=90)
    args = ap.parse_args()
    work = Path(args.workdir)
    if args.daemon:
        q.LOG_DIR = work
        q.redirect_output_and_ignore_console_signals()
    if args.old_flags:
        q.DETACH_FLAGS = 0
    job_cmd = f'"{sys.executable}" -c "import time; time.sleep({args.seconds})"'
    with open(work / "job.log", "a") as lf:
        proc = q.spawn_detached(job_cmd, lf, str(work))
    hwnd = ctypes.windll.kernel32.GetConsoleWindow() if os.name == "nt" else 0
    Path(args.state).write_text(json.dumps({"runner_pid": os.getpid(), "shell_pid": proc.pid, "console_hwnd": int(hwnd),
                                            "stdout_is_none": sys.stdout is None}), encoding="utf-8")
    print("runner surrogate alive", flush=True)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(0.5)


if __name__ == "__main__":
    main()
