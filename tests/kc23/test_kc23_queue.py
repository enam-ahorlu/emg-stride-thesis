"""src/kc23_queue.py fail-closed gate handling. Found live on 2026-09-24:
src/kc23_s2_f0_feasibility.py crashed (uncaught exception, rc=1) and
kc23_queue.finish() only special-cased rc 10/20, so the crash was treated
exactly like a clean pass and the downstream job ran unchecked. These tests
run against HALT_MD/STATUS_MD redirected into tmp_path so they never touch
the real docs/kc23/KC23_HALT.md / docs/kc23/KC23_STATUS.md -- important while the real queue may
be running live in the same checkout.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
import kc23_queue as q


def make_job(job_id="j1", stage="S", gate_script=""):
    return q.Job({"job_id": job_id, "stage": stage, "seed": "", "command": "",
                 "out_dir": f"results/{job_id}", "depends_on": "", "gate_script": gate_script})


@pytest.fixture(autouse=True)
def redirect_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "HALT_MD", tmp_path / "docs/kc23/KC23_HALT.md")
    monkeypatch.setattr(q, "STATUS_MD", tmp_path / "docs/kc23/KC23_STATUS.md")
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
    be treated as an escalate (halt the stage, write docs/kc23/KC23_HALT.md), not
    silently swallowed as a pass."""
    job = make_job(stage="STAGEY", gate_script="some_gate.py")
    job.proc = FakeProc(0)
    job.t0 = 0
    monkeypatch.setattr(q, "run_gate", lambda j: crash_rc)
    halted = set()
    q.finish(job, halted, [job], {job.job_id: job})
    assert job.gate_rc == crash_rc, "docs/kc23/KC23_STATUS.md should show the real crash code, not a fake 20"
    assert halted == {"STAGEY"}, "but control flow must still halt the stage as if it were a 20"
    assert q.HALT_MD.exists()
    text = q.HALT_MD.read_text()
    assert "ESCALATE" in text
    assert f"rc={crash_rc}" in text, "the halt entry should record the actual unexpected exit code"


def test_finish_missing_gate_script_halts_the_stage(monkeypatch):
    """Changed 2026-09-25. A gate script that does not exist used to be treated
    as 'not yet implemented' (rc 0, continue), which let a stage pass with its
    gate never having run. It is now a failure like any other unrunnable gate:
    the stage halts and docs/kc23/KC23_HALT.md records it."""
    job = make_job(stage="STAGEZ", gate_script="does_not_exist.py")
    job.proc = FakeProc(0)
    job.t0 = 0
    halted = set()
    q.finish(job, halted, [job], {job.job_id: job})
    assert job.gate_rc == 20
    assert halted == {"STAGEZ"}
    assert q.HALT_MD.exists()


def test_run_gate_end_to_end_crash_script_returns_nonzero(tmp_path):
    """Real subprocess, real script: tests/kc23/_synthetic_crash_gate.py
    (an uncaught exception) must surface as a nonzero, non-10/20 rc through
    the actual run_gate() subprocess path, not just in a mocked unit test."""
    job = make_job(gate_script="tests/kc23/_synthetic_crash_gate.py")
    rc = q.run_gate(job)
    assert rc not in (0, 10, 20)


def test_run_gate_end_to_end_escalate_script_returns_20():
    job = make_job(gate_script="tests/kc23/_synthetic_escalate_gate.py")
    rc = q.run_gate(job)
    assert rc == 20


# ---- expected_outputs (2026-09-24): closes the "exit 0 having written
# nothing, or the wrong subject count" bug class at the infrastructure level.

def test_check_expected_outputs_empty_is_always_ok(tmp_path):
    job = make_job()
    job.expected_outputs = ""
    ok, reason = q.check_expected_outputs(job)
    assert ok


def test_check_expected_outputs_missing_file_fails(tmp_path):
    job = make_job()
    job.out_dir = str(tmp_path / "nope")
    job.expected_outputs = "*subjectwise.csv|40"
    ok, reason = q.check_expected_outputs(job)
    assert not ok
    assert "missing" in reason or "no file matching" in reason


def test_check_expected_outputs_wrong_count_fails(tmp_path):
    import pandas as pd
    d = tmp_path / "out"; d.mkdir(parents=True)
    pd.DataFrame({"subject": range(1, 11)}).to_csv(d / "x_subjectwise.csv", index=False)  # 10, not 40
    job = make_job()
    job.out_dir = str(d)
    job.expected_outputs = "*subjectwise.csv|40"
    ok, reason = q.check_expected_outputs(job)
    assert not ok
    assert "10" in reason and "40" in reason


def test_check_expected_outputs_correct_count_passes(tmp_path):
    import pandas as pd
    d = tmp_path / "out"; d.mkdir(parents=True)
    pd.DataFrame({"subject": range(1, 41)}).to_csv(d / "x_subjectwise.csv", index=False)
    job = make_job()
    job.out_dir = str(d)
    job.expected_outputs = "*subjectwise.csv|40"
    ok, reason = q.check_expected_outputs(job)
    assert ok, reason


def test_check_expected_outputs_malformed_fails(tmp_path):
    job = make_job()
    job.expected_outputs = "*subjectwise.csv"  # no |count
    ok, reason = q.check_expected_outputs(job)
    assert not ok


def test_check_expected_outputs_letter_mode_header_only_fails(tmp_path):
    """The exact d1_reproduction_check bug: a verdict file exists but has
    only its header, no letter -- must fail, not silently pass."""
    d = tmp_path / "out"; d.mkdir(parents=True)
    (d / "X_VERDICT.md").write_text("# KC-D1 verdict\n\n", encoding="utf-8")
    job = make_job()
    job.out_dir = str(d)
    job.expected_outputs = "*_VERDICT.md|LETTER"
    ok, reason = q.check_expected_outputs(job)
    assert not ok


def test_check_expected_outputs_letter_mode_real_letter_passes(tmp_path):
    d = tmp_path / "out"; d.mkdir(parents=True)
    (d / "X_VERDICT.md").write_text("# KC-D1 verdict\n\n- **reproduction: PASS**\n", encoding="utf-8")
    job = make_job()
    job.out_dir = str(d)
    job.expected_outputs = "*_VERDICT.md|LETTER"
    ok, reason = q.check_expected_outputs(job)
    assert ok, reason


def test_already_done_uses_expected_outputs_when_set(tmp_path):
    d = tmp_path / "out"; d.mkdir(parents=True)
    (d / "x_VERDICT.md").write_text("# verdict\n\n", encoding="utf-8")  # header only -- not really done
    job = make_job()
    job.out_dir = str(d)
    job.expected_outputs = "*_VERDICT.md|LETTER"
    assert not q.already_done(job), "a header-only verdict must not count as done, even though is_complete() would say yes"


def test_already_done_falls_back_to_heuristic_when_blank(tmp_path):
    d = tmp_path / "out"; d.mkdir(parents=True)
    (d / "x_VERDICT.md").write_text("# verdict\n\n", encoding="utf-8")
    job = make_job()
    job.out_dir = str(d)
    job.expected_outputs = ""
    assert q.already_done(job)  # unaudited rows keep the old permissive behavior


# ---- --adopt / GhostProc (2026-09-24): restart without killing an in-flight
# job whose child process is independent of the queue's own PID.

def test_ghostproc_poll_none_while_pid_alive(monkeypatch):
    import sys as _sys
    fake_psutil = __import__("types").SimpleNamespace(pid_exists=lambda p: True)
    monkeypatch.setitem(_sys.modules, "psutil", fake_psutil)   # restored after the test (it used to leak)
    gp = q.GhostProc(pid=99999)
    assert gp.poll() is None
    assert gp.returncode is None


def test_ghostproc_poll_returns_0_once_pid_gone(monkeypatch):
    import sys as _sys
    fake_psutil = __import__("types").SimpleNamespace(pid_exists=lambda p: False)
    monkeypatch.setitem(_sys.modules, "psutil", fake_psutil)   # restored after the test (it used to leak)
    gp = q.GhostProc(pid=99999)
    assert gp.poll() == 0
    assert gp.returncode == 0


def test_parse_adopt_refuses_dead_pid(monkeypatch):
    import sys as _sys
    fake_psutil = __import__("types").SimpleNamespace(pid_exists=lambda p: False)
    monkeypatch.setitem(_sys.modules, "psutil", fake_psutil)   # restored after the test (it used to leak)
    job = make_job("real_job")
    with pytest.raises(SystemExit):
        q.parse_adopt("real_job=99999", [job], [], {job.job_id: job})


def test_parse_adopt_refuses_unknown_job_id(monkeypatch):
    import sys as _sys
    fake_psutil = __import__("types").SimpleNamespace(pid_exists=lambda p: True)
    monkeypatch.setitem(_sys.modules, "psutil", fake_psutil)   # restored after the test (it used to leak)
    with pytest.raises(SystemExit):
        q.parse_adopt("nope=123", [], [], {})


def test_parse_adopt_marks_running_and_routes_gpu_vs_cpu(monkeypatch):
    import sys as _sys
    fake_psutil = __import__("types").SimpleNamespace(pid_exists=lambda p: True)
    monkeypatch.setitem(_sys.modules, "psutil", fake_psutil)   # restored after the test (it used to leak)
    gpu_job = make_job("gpu_job")
    cpu_job = make_job("cpu_job")
    by_id = {gpu_job.job_id: gpu_job, cpu_job.job_id: cpu_job}
    gr, cr = q.parse_adopt("gpu_job=111,cpu_job=222", [gpu_job], [cpu_job], by_id)
    assert gr is gpu_job and cr is cpu_job
    assert gpu_job.status == "running" and gpu_job.proc.pid == 111
    assert cpu_job.status == "running" and cpu_job.proc.pid == 222


def test_finish_fails_closed_when_outputs_wrong_even_though_process_exited_0(monkeypatch, tmp_path):
    """The user's exact wording: 'FAILED ... whatever its exit code' -- a
    process that exits 0 but wrote the wrong subject count must still be
    marked failed, and the gate must never run on it."""
    import pandas as pd
    d = tmp_path / "out"; d.mkdir(parents=True)
    pd.DataFrame({"subject": range(1, 11)}).to_csv(d / "x_subjectwise.csv", index=False)  # 10, not 40
    job = make_job(gate_script="some_gate.py")
    job.out_dir = str(d)
    job.expected_outputs = "*subjectwise.csv|40"
    job.proc = FakeProc(0)
    job.t0 = 0
    gate_was_called = []
    monkeypatch.setattr(q, "run_gate", lambda j: gate_was_called.append(1) or 0)
    halted = set()
    q.finish(job, halted, [job], {job.job_id: job})
    assert job.status == "failed"
    assert job.output_check_reason is not None
    assert gate_was_called == [], "the gate must not run when the output check itself fails"


def test_deps_satisfied_accepts_skipped_complete():
    """Regression test for the queue-wide halt found live 2026-09-24, the
    first time the queue was ever restarted mid-run: a dependency recognized
    by resume as already finished gets status 'skipped(complete)', not
    'done' -- deps_satisfied() must accept either, or every downstream job
    of an already-finished-before-restart run becomes permanently
    unsatisfiable and the whole remaining queue falsely looks deadlocked."""
    upstream = make_job("upstream", stage="D1")
    upstream.status = "skipped(complete)"
    downstream = make_job("downstream", stage="S1")
    downstream.depends_on = ["upstream"]
    by_id = {"upstream": upstream, "downstream": downstream}
    assert q.deps_satisfied(downstream, by_id, set())


def test_deps_satisfied_rejects_queued_or_failed():
    for bad_status in ("queued", "failed", "running"):
        upstream = make_job("upstream", stage="D1")
        upstream.status = bad_status
        downstream = make_job("downstream", stage="S1")
        downstream.depends_on = ["upstream"]
        by_id = {"upstream": upstream, "downstream": downstream}
        assert not q.deps_satisfied(downstream, by_id, set()), bad_status


def test_pick_next_does_not_falsely_deadlock_after_a_resume(monkeypatch, tmp_path):
    """End-to-end version of the same bug: with a mix of 'done' and
    'skipped(complete)' upstream jobs (as happens after any restart),
    main()'s own deadlock check must still find the newly-runnable job."""
    j1 = make_job("j1", stage="A")
    j1.status = "skipped(complete)"
    j2 = make_job("j2", stage="A")
    j2.status = "done"
    j3 = make_job("j3", stage="B")
    j3.depends_on = ["j1", "j2"]
    by_id = {j.job_id: j for j in (j1, j2, j3)}
    remaining = [j for j in (j1, j2, j3) if j.status == "queued"]
    assert remaining == [j3]
    runnable = [j for j in remaining if j.stage not in set() and q.deps_satisfied(j, by_id, set())]
    assert runnable == [j3], "j3 must be runnable once both its deps are done/skipped(complete)"


def test_status_notes_survive_status_regeneration(tmp_path, monkeypatch):
    import kc23_queue as q
    monkeypatch.setattr(q, "STATUS_MD", tmp_path / "docs/kc23/KC23_STATUS.md")
    monkeypatch.setattr(q, "STATUS_NOTES_MD", tmp_path / "docs/kc23/KC23_STATUS_NOTES.md")
    q.write_status([], [], set(), None, None)
    assert "Notes" not in (tmp_path / "docs/kc23/KC23_STATUS.md").read_text()
    (tmp_path / "docs/kc23/KC23_STATUS_NOTES.md").write_text("Commit abc1234: conformance.\n", encoding="utf-8")
    q.write_status([], [], set(), None, None)
    q.write_status([], [], set(), None, None)          # regenerated twice: the note is there once, not lost and not doubled
    text = (tmp_path / "docs/kc23/KC23_STATUS.md").read_text()
    assert text.count("Commit abc1234: conformance.") == 1 and "preserved across regenerations" in text
