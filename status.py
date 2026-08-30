"""Clean status snapshot for the streaming supervisor. Run: python status.py [glob]"""
import glob, os, sys, time, psutil

SCRIPT = "run_streaming" + "_norm_loso.py"   # split so this file never self-matches
GUARD = "run_multi" + "_guard.py"
pat = sys.argv[1] if len(sys.argv) > 1 else "results_loso_freq_streaming/*_subjectwise.csv"

me = os.getpid()
running = []
for p in psutil.process_iter(["pid", "cmdline"]):
    if p.info["pid"] == me:
        continue
    cl = " ".join(p.info["cmdline"] or [])
    if SCRIPT in cl:
        running.append(p)
# measure aggregate CPU
for p in running:
    try: p.cpu_percent(None)
    except Exception: pass
time.sleep(2)
cpu = 0.0
for p in running:
    try: cpu += p.cpu_percent(None)
    except Exception: pass

files = sorted(glob.glob(pat))
total = 0
per = []
for f in files:
    try:
        with open(f) as fh:
            n = sum(1 for _ in fh) - 1
    except Exception:
        n = -1
    total += max(n, 0)
    per.append(os.path.basename(f).replace("streaming_", "").replace("_subjectwise.csv", "") + f"={n}")

guard_alive = any(GUARD in " ".join(p.info["cmdline"] or [])
                  for p in psutil.process_iter(["cmdline"]))
kills = done = 0
try:
    L = open("logs/mguard_streaming.log").read()
    kills = L.count("kill newest"); done = L.count(" DONE"); alldone = "ALL" in L and "JOBS DONE" in L
except Exception:
    alldone = False
m = psutil.virtual_memory()
print(f"mem={m.percent:.0f}% free={m.available/1e9:.1f}GB | child_procs={len(running)} cpu={cpu:.0f}% "
      f"| guard_alive={guard_alive} jobs_done={done}/10 kills={kills} alldone={alldone}")
print(f"total_subjects={total} | " + " ".join(per))
