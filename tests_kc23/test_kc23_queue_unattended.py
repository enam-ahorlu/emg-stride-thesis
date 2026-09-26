"""kc23_queue.py unattended-operation features (25 September 2026): the EXISTS
output spec, the lock file, running.json adoption, and persisted halted/failed
state. Real subprocesses are used for the process-identity checks (a PID and its
create time), and an in-process run of run_queue() for the restart behaviour:
a failed job is not retried by a restarted runner, and a halted stage stays
halted. Everything is redirected into tmp_path; nothing touches the live queue."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_queue as q

PY = sys.executable


@pytest.fixture(autouse=True)
def redirect(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    monkeypatch.setattr(q, "HALT_MD", tmp_path / "KC23_HALT.md")
    monkeypatch.setattr(q, "STATUS_MD", tmp_path / "KC23_STATUS.md")
    monkeypatch.setattr(q, "LOG_DIR", tmp_path / "_run_logs")
    monkeypatch.setattr(q, "PYTHON", PY)
    monkeypatch.chdir(tmp_path)


def _job(job_id, command="", stage="S", out_dir=None, depends="", gate="", expected=""):
    return q.Job({"job_id": job_id, "stage": stage, "seed": "", "command": command,
                  "out_dir": out_dir or f"out_{job_id}", "depends_on": depends, "gate_script": gate,
                  "expected_outputs": expected})


@pytest.fixture
def sleeper():
    """A real process whose command line contains 'kc23_queue', like a runner."""
    p = subprocess.Popen([PY, "-c", "import time  # kc23_queue\ntime.sleep(120)"])
    time.sleep(0.5)
    yield p
    p.kill()
    p.wait()


# ------------------------------------------------------------------ EXISTS spec
def test_exists_spec(tmp_path):
    j = _job("a", out_dir="o", expected="f.npz|EXISTS")
    assert not q.check_expected_outputs(j)[0]                       # missing
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "f.npz").write_bytes(b"")
    assert not q.check_expected_outputs(j)[0]                       # present but empty
    (tmp_path / "o" / "f.npz").write_bytes(b"data")
    assert q.check_expected_outputs(j)[0]


# ------------------------------------------------------------------ lock file
def test_lock_second_runner_refused_while_first_alive(sleeper):
    ok, _ = q.acquire_lock(pid=sleeper.pid)
    assert ok
    ok2, why = q.acquire_lock(pid=12345678)
    assert not ok2 and str(sleeper.pid) in why


def test_lock_taken_over_when_holder_is_dead():
    dead = subprocess.Popen([PY, "-c", "pass  # kc23_queue"])
    dead.wait()
    ok, _ = q.acquire_lock(pid=dead.pid)             # records a lock for a process that no longer exists
    assert ok
    ok2, _ = q.acquire_lock(pid=99999999)
    assert ok2


def test_lock_not_honoured_for_a_recycled_pid(sleeper):
    ok, _ = q.acquire_lock(pid=sleeper.pid)
    assert ok
    lock = q._lock_path()
    info = q.json.loads(lock.read_text())
    info["create_time"] = info["create_time"] - 1000.0        # same PID, but a different process started it
    lock.write_text(q.json.dumps(info))
    ok2, _ = q.acquire_lock(pid=99999999)
    assert ok2


def test_lock_not_honoured_when_the_process_is_not_a_runner():
    other = subprocess.Popen([PY, "-c", "import time\ntime.sleep(60)"])   # alive, but its cmdline has no kc23_queue
    time.sleep(0.5)
    try:
        assert q.acquire_lock(pid=other.pid)[0]
        assert q.acquire_lock(pid=99999999)[0]
    finally:
        other.kill()


def test_release_only_removes_own_lock(sleeper):
    q.acquire_lock(pid=sleeper.pid)
    q.release_lock(pid=1)                              # someone else's release does nothing
    assert q._lock_path().exists()
    q.release_lock(pid=sleeper.pid)
    assert not q._lock_path().exists()


# ------------------------------------------------------------------ running.json adoption
class _P:
    def __init__(self, pid):
        self.pid = pid


def test_auto_adopt_live_job_and_not_relaunched(sleeper):
    j = _job("gpu1")
    j.proc = _P(sleeper.pid)
    q.record_running(j)
    fresh = _job("gpu1")
    gpu, cpu, light = q.auto_adopt({"gpu1": fresh}, {"gpu1"}, set(), set())
    assert gpu is fresh and cpu is None and fresh.status == "running"
    assert fresh.proc.poll() is None                    # alive
    sleeper.kill(); sleeper.wait()
    assert fresh.proc.poll() == 0                       # gone -> finish() will judge it by expected_outputs


def test_auto_adopt_ignores_a_recycled_pid(sleeper):
    j = _job("gpu1")
    j.proc = _P(sleeper.pid)
    q.record_running(j)
    data = q._read_json(q._running_path(), {})
    data["gpu1"]["create_time"] -= 1000.0
    q._write_json(q._running_path(), data)
    fresh = _job("gpu1")
    gpu, cpu, light = q.auto_adopt({"gpu1": fresh}, {"gpu1"}, set(), set())
    assert gpu is None and fresh.status == "queued"
    assert q._read_json(q._running_path(), {}) == {}


def test_auto_adopt_dead_pid_is_requeued_not_adopted():
    dead = subprocess.Popen([PY, "-c", "pass"])
    dead.wait()
    j = _job("cpu1")
    j.proc = _P(dead.pid)
    q.record_running(j)
    fresh = _job("cpu1")
    gpu, cpu, light = q.auto_adopt({"cpu1": fresh}, set(), {"cpu1"}, set())
    assert gpu is None and cpu is None and fresh.status == "queued"


def test_auto_adopt_refuses_two_live_jobs_in_one_lane(sleeper):
    other = subprocess.Popen([PY, "-c", "import time\ntime.sleep(60)"])
    time.sleep(0.3)
    try:
        for jid, proc in (("g1", sleeper), ("g2", other)):
            j = _job(jid); j.proc = _P(proc.pid); q.record_running(j)
        with pytest.raises(SystemExit):
            q.auto_adopt({"g1": _job("g1"), "g2": _job("g2")}, {"g1", "g2"}, set(), set())
    finally:
        other.kill()


# ------------------------------------------------------------------ persisted state
def test_state_roundtrip_and_failed_job_not_requeued():
    a, b = _job("a"), _job("b")
    a.status = "failed"; a.output_check_reason = "boom"
    q.save_state({"S"}, [a, b])
    st = q.load_state()
    assert st["halted_stages"] == ["S"] and "a" in st["failed"] and "b" not in st["failed"]
    a2, b2 = _job("a"), _job("b")
    halted = set()
    q.apply_state(st, {"a": a2, "b": b2}, halted)
    assert halted == {"S"} and a2.status == "failed" and b2.status == "queued"


def test_failed_job_whose_outputs_are_now_complete_is_healed(tmp_path):
    a = _job("a", out_dir="o", expected="r.csv|EXISTS")
    a.status = "failed"
    q.save_state(set(), [a])
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "r.csv").write_text("x")
    a2 = _job("a", out_dir="o", expected="r.csv|EXISTS")
    q.apply_state(q.load_state(), {"a": a2}, set())
    assert a2.status == "queued"                        # the normal skip rule then marks it complete


def test_unreadable_state_file_refuses_to_start():
    q._state_path().parent.mkdir(parents=True, exist_ok=True)
    q._state_path().write_text("{not json")
    with pytest.raises(SystemExit):
        q.load_state()


# ------------------------------------------------------------------ restart behaviour, end to end
def _args(**kw):
    d = dict(poll_interval=0.05, max_jobs=None, adopt=None, retry=None, clear_halt=None)
    d.update(kw)
    return argparse.Namespace(**d)


def _cmd(code):
    return f'"{PY}" -c "{code}"'


def _run(jobs, **kw):
    by_id = {j.job_id: j for j in jobs}
    q.run_queue(_args(**kw), jobs, [], by_id)


def test_failed_job_is_not_retried_by_a_restarted_runner_until_asked(tmp_path):
    def mk():
        a = _job("a", command=_cmd("open('runs_a','a').write('x'); import sys; sys.exit(3)"))
        b = _job("b", command=_cmd("open('runs_b','a').write('x')"), depends="a")
        return a, b
    a, b = mk()
    _run([a, b])
    assert a.status == "failed" and (tmp_path / "runs_a").read_text() == "x"
    assert not (tmp_path / "runs_b").exists()
    a2, b2 = mk()                                       # a fresh runner process, same CSVs
    _run([a2, b2])
    assert a2.status == "failed" and (tmp_path / "runs_a").read_text() == "x"      # NOT run again
    assert not (tmp_path / "runs_b").exists()
    a3, b3 = mk()
    _run([a3, b3], retry="a")                           # a human asked for it
    assert (tmp_path / "runs_a").read_text() == "xx"


def test_halted_stage_stays_halted_across_a_runner_restart(tmp_path):
    (tmp_path / "gate20.py").write_text("import sys; sys.exit(20)")
    def mk():
        g = _job("g", stage="STG", command=_cmd("open('og/done.txt','w').write('x')"), out_dir="og",
                 gate="gate20.py", expected="done.txt|EXISTS")
        h = _job("h", stage="STG", command=_cmd("open('ran_h','w').write('x')"), depends="g")
        return g, h
    g, h = mk()
    _run([g, h])
    assert g.gate_rc == 20 and not (tmp_path / "ran_h").exists()
    assert q.load_state()["halted_stages"] == ["STG"]
    g2, h2 = mk()
    _run([g2, h2])                                       # a restarted runner: g is complete, the halt is remembered
    assert g2.status in ("skipped(halted)", "skipped(complete)") and h2.status in ("skipped(halted)", "skipped(blocked)")
    assert not (tmp_path / "ran_h").exists()
    g3, h3 = mk()
    _run([g3, h3], clear_halt="STG")                    # Enam decided to continue
    assert (tmp_path / "ran_h").exists() and q.load_state()["halted_stages"] == []


def test_second_runner_process_exits_without_running_anything(tmp_path, sleeper, monkeypatch):
    q.acquire_lock(pid=sleeper.pid)
    (tmp_path / "g.csv").write_text(
        "job_id,stage,seed,command,out_dir,depends_on,gate_script,expected_outputs\n"
        f"x,S,,\"{PY} -c pass\",ox,,,\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["kc23_queue.py", "--gpu-csv", str(tmp_path / "g.csv"),
                                      "--cpu-csv", str(tmp_path / "none.csv")])
    with pytest.raises(SystemExit) as e:
        q.main()
    assert e.value.code == 3


# ------------------------------------------------------------------ light lane
def _light(job_id, command, **kw):
    j = _job(job_id, command=command, **kw)
    j.light = True
    return j


def test_light_row_runs_beside_a_running_heavy_cpu_job(tmp_path):
    """The reason the lane exists: a gate must not wait behind hours of heavy CPU work."""
    heavy = _job("heavy", command=_cmd("import time; time.sleep(4); open('heavy_done','w').write('x')"),
                 expected="heavy_done|EXISTS", out_dir=".")
    lite = _light("lite", _cmd("open('lite_done','w').write('x')"), out_dir=".", expected="lite_done|EXISTS")
    by_id = {"heavy": heavy, "lite": lite}
    q.run_queue(_args(), [], [heavy, lite], by_id)
    assert heavy.status == "done" and lite.status == "done"
    assert (tmp_path / "lite_done").stat().st_mtime < (tmp_path / "heavy_done").stat().st_mtime   # lite finished first


def test_only_one_light_job_at_a_time_and_only_one_heavy(tmp_path):
    a = _light("a", _cmd("import time; time.sleep(1.5); open('a_end','w').write('x')"), out_dir=".", expected="a_end|EXISTS")
    b = _light("b", _cmd("open('b_start','w').write('x')"), out_dir=".", expected="b_start|EXISTS")
    q.run_queue(_args(), [], [a, b], {"a": a, "b": b})
    assert a.status == "done" and b.status == "done"
    assert (tmp_path / "a_end").stat().st_mtime <= (tmp_path / "b_start").stat().st_mtime         # b waited for a


def test_pick_next_filters_by_lane():
    h = _job("h"); l = _light("l", "")
    assert q.pick_next([h, l], {"h": h, "l": l}, set(), light=True) is l
    assert q.pick_next([h, l], {"h": h, "l": l}, set(), light=False) is h
    assert q.pick_next([h, l], {"h": h, "l": l}, set()) is h            # no filter: first in order (the GPU lane)


def test_auto_adopt_puts_a_live_light_job_in_the_light_slot(sleeper):
    j = _light("lite", "")
    j.proc = _P(sleeper.pid)
    q.record_running(j)
    fresh = _light("lite", "")
    gpu, cpu, light = q.auto_adopt({"lite": fresh}, set(), {"lite"}, set(), {"lite"})
    assert light is fresh and cpu is None and gpu is None


def test_light_column_parsed_from_csv_row():
    assert q.Job({"job_id": "x", "stage": "S", "command": "c", "out_dir": "o", "light": "1"}).light
    assert not q.Job({"job_id": "x", "stage": "S", "command": "c", "out_dir": "o"}).light


def test_keep_awake_never_raises_and_round_trips():
    on = q.keep_awake(True)
    off = q.keep_awake(False)
    assert isinstance(on, bool) and isinstance(off, bool)      # True on Windows; a plain False elsewhere, never an error


def test_complete_rows_are_resolved_for_every_lane_at_startup_so_light_rows_are_not_blocked(tmp_path):
    """Found on the first hand-off: while the GPU lane was busy, already-finished GPU rows stayed 'queued' (a lane only
    resolves rows as its own scan reaches them), so a light row depending on one could not start."""
    (tmp_path / "g2").mkdir()
    (tmp_path / "g2" / "r.csv").write_text("x")
    busy = _job("g1", command=_cmd("import time; time.sleep(4); open('g1_end','w').write('x')"), out_dir=".",
                expected="g1_end|EXISTS")
    done = _job("g2", out_dir="g2", expected="r.csv|EXISTS")                  # complete by its outputs, second in the lane
    lite = _light("lite", _cmd("open('lite_ran','w').write('x')"), out_dir=".", expected="lite_ran|EXISTS",
                  depends="g2")
    by_id = {"g1": busy, "g2": done, "lite": lite}
    q.run_queue(_args(), [busy, done], [lite], by_id)
    assert lite.status == "done" and busy.status == "done"
    assert (tmp_path / "lite_ran").stat().st_mtime < (tmp_path / "g1_end").stat().st_mtime    # did not wait for g1


def test_mark_complete_respects_halted_stages(tmp_path):
    (tmp_path / "o").mkdir(); (tmp_path / "o" / "r.csv").write_text("x")
    a = _job("a", stage="H", out_dir="o", expected="r.csv|EXISTS")
    b = _job("b", stage="OK", out_dir="o", expected="r.csv|EXISTS")
    assert q.mark_complete([a, b], {"H"}) == 1
    assert a.status == "queued" and b.status == "skipped(complete)"
