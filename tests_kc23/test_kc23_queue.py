"""kc23_queue.py fail-closed gate handling. Found live on 2026-09-24:
kc23_s2_f0_feasibility.py crashed (uncaught exception, rc=1) and
kc23_queue.finish() only special-cased rc 10/20, so the crash was treated
exactly like a clean pass and the downstream job ran unchecked. These tests
run against HALT_MD/STATUS_MD redirected into tmp_path so they never touch
the real KC23_HALT.md / KC23_STATUS.md -- important while the real queue may
be running live in the same checkout.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_queue as q


def make_job(job_id="j1", stage="S", gate_script=""):
    return q.Job({"job_id": job_id, "stage": stage, "seed": "", "command": "",
                 "out_dir": f"results_{job_id}", "depends_on": "", "gate_script": gate_script})


@pytest.fixture(autouse=True)
def redirect_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "HALT_MD", tmp_path / "KC23_HALT.md")
    monkeypatch.setattr(q, "STATUS_MD", tmp_path / "KC23_STATUS.md")
    monkeypatch.setattr(q, "LOG_DIR", tmp_path / "_run_logs")
    yield


class FakeProc:
    def __init__(self, returncode):
        self.returncode = returncode


def test_finish_rc0_continues(monkeypatch):
    job = make_job(gate_script="some_gate.py")
    job.proc = FakeProc(0)
    job.t0 = 0
    monkeypatch.setattr(q, "run_gate", lambda j: 0)
    halted = set()
    q.finish(job, halted, [job], {job.job_id: job})
    assert job.status == "done"
    assert job.gate_rc == 0
    assert halted == set()


def test_finish_rc10_reports_and_continues(monkeypatch):
    job = make_job(gate_script="some_gate.py")
    job.proc = FakeProc(0)
    job.t0 = 0
    monkeypatch.setattr(q, "run_gate", lambda j: 10)
    halted = set()
    q.finish(job, halted, [job], {job.job_id: job})
    assert job.gate_rc == 10
    assert halted == set()
    assert not q.HALT_MD.exists()


def test_finish_rc20_halts_and_writes_halt_md(monkeypatch):
    job = make_job(stage="STAGEX", gate_script="some_gate.py")
    job.proc = FakeProc(0)
    job.t0 = 0
    monkeypatch.setattr(q, "run_gate", lambda j: 20)
    halted = set()
    q.finish(job, halted, [job], {job.job_id: job})
    assert job.gate_rc == 20
    assert halted == {"STAGEX"}
    assert q.HALT_MD.exists()
    assert "ESCALATE" in q.HALT_MD.read_text()


@pytest.mark.parametrize("crash_rc", [1, 2, 255])
def test_finish_unexpected_rc_fails_closed(monkeypatch, crash_rc):
    """The core regression test: any gate exit code outside {0, 10, 20} must
    be treated as an escalate (halt the stage, write KC23_HALT.md), not
    silently swallowed as a pass."""
    job = make_job(stage="STAGEY", gate_script="some_gate.py")
    job.proc = FakeProc(0)
    job.t0 = 0
    monkeypatch.setattr(q, "run_gate", lambda j: crash_rc)
    halted = set()
    q.finish(job, halted, [job], {job.job_id: job})
    assert job.gate_rc == crash_rc, "KC23_STATUS.md should show the real crash code, not a fake 20"
    assert halted == {"STAGEY"}, "but control flow must still halt the stage as if it were a 20"
    assert q.HALT_MD.exists()
    text = q.HALT_MD.read_text()
    assert "ESCALATE" in text
    assert f"rc={crash_rc}" in text, "the halt entry should record the actual unexpected exit code"


def test_finish_not_implemented_gate_does_not_fail_closed(monkeypatch):
    """A gate script that simply doesn't exist yet is a known, documented gap
    (job.gate_rc == 'NOT_IMPLEMENTED') -- distinct from a crash, and must NOT
    trip the new fail-closed path."""
    job = make_job(stage="STAGEZ", gate_script="does_not_exist.py")
    job.proc = FakeProc(0)
    job.t0 = 0
    halted = set()
    q.finish(job, halted, [job], {job.job_id: job})
    assert job.gate_rc == "NOT_IMPLEMENTED"
    assert halted == set()
    assert not q.HALT_MD.exists()


def test_run_gate_end_to_end_crash_script_returns_nonzero(tmp_path):
    """Real subprocess, real script: tests_kc23/_synthetic_crash_gate.py
    (an uncaught exception) must surface as a nonzero, non-10/20 rc through
    the actual run_gate() subprocess path, not just in a mocked unit test."""
    job = make_job(gate_script="tests_kc23/_synthetic_crash_gate.py")
    rc = q.run_gate(job)
    assert rc not in (0, 10, 20)


def test_run_gate_end_to_end_escalate_script_returns_20():
    job = make_job(gate_script="tests_kc23/_synthetic_escalate_gate.py")
    rc = q.run_gate(job)
    assert rc == 20
