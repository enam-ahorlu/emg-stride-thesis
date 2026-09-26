"""The aggregate-then-gate rows (d5_stats, s1_gate, the D1 and D6 checks): the command builds the inputs and the GATE
writes the verdict, so a '<glob>|LETTER' expectation is checked after the gate, against a verdict newer than this
run. Found 2026-09-26: d5_stats was failed for not yet having a verdict (checked before its gate ran), and D1's check
had only ever passed because an earlier run had left a verdict behind."""
import os
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


class _Done:
    returncode = 0


def _row(gate_body: str, name="gate.py", stage="STG"):
    (Path(q.ROOT) / name).write_text(gate_body)
    (Path(q.ROOT) / "out").mkdir(exist_ok=True)
    j = q.Job({"job_id": "agg", "stage": stage, "seed": "", "command": "", "out_dir": "out", "depends_on": "",
               "gate_script": name, "expected_outputs": "X_VERDICT.md|LETTER"})
    j.proc = _Done()
    j.t0 = time.time() - 5
    return j


WRITES_LETTER = "import sys\nopen('out/X_VERDICT.md','w').write('# v\\n\\n- **thing: E-R**\\n')\nsys.exit(0)\n"
WRITES_NOTHING = "import sys\nsys.exit(0)\n"
ESCALATES = "import sys\nopen('out/X_VERDICT.md','w').write('# v\\n\\n**Outcome: D-S**\\n')\nsys.exit(20)\n"
FAILS_CLOSED = "import sys\nopen('out/X_VERDICT.md','w').write('# v\\n\\nNO OUTCOME COMPUTED. input missing\\n')\nsys.exit(20)\n"


def test_verdict_written_by_the_gate_is_accepted():
    j = _row(WRITES_LETTER)
    halted = set()
    q.finish(j, halted, [j], {"agg": j})
    assert j.status == "done" and j.gate_rc == 0 and halted == set()


def test_gate_exiting_zero_without_a_verdict_is_a_failure_and_halts():
    j = _row(WRITES_NOTHING)
    halted = set()
    q.finish(j, halted, [j], {"agg": j})
    assert j.status == "failed" and halted == {"STG"}
    assert "no file matching" in j.output_check_reason


def test_stale_verdict_from_an_earlier_run_does_not_satisfy_the_check():
    j = _row(WRITES_NOTHING)
    v = Path("out/X_VERDICT.md")
    v.write_text("# v\n\n- **thing: E-R**\n")
    old = time.time() - 3600
    os.utime(v, (old, old))
    halted = set()
    q.finish(j, halted, [j], {"agg": j})
    assert j.status == "failed" and halted == {"STG"} and "predates" in j.output_check_reason


def test_escalation_with_a_real_letter_is_done_and_halts_but_is_not_a_failure():
    j = _row(ESCALATES)
    halted = set()
    q.finish(j, halted, [j], {"agg": j})
    assert j.status == "done" and j.gate_rc == 20 and halted == {"STG"}
    assert "D-S" in q.HALT_MD.read_text() or q.HALT_MD.exists()


def test_a_gate_that_fails_closed_writes_no_letter_and_the_row_is_failed_and_halted():
    j = _row(FAILS_CLOSED)
    halted = set()
    q.finish(j, halted, [j], {"agg": j})
    assert j.status == "failed" and halted == {"STG"}


def test_a_row_without_a_gate_still_checks_its_outputs_before_anything_else(tmp_path):
    j = q.Job({"job_id": "plain", "stage": "S", "seed": "", "command": "", "out_dir": "out2", "depends_on": "",
               "gate_script": "", "expected_outputs": "*_VERDICT.md|LETTER"})
    j.proc = _Done(); j.t0 = time.time()
    (tmp_path / "out2").mkdir()
    q.finish(j, set(), [j], {"plain": j})
    assert j.status == "failed"                          # no verdict, no gate to write one: unchanged behaviour


def test_check_expected_outputs_freshness_parameter(tmp_path):
    j = q.Job({"job_id": "x", "stage": "S", "seed": "", "command": "", "out_dir": "o", "depends_on": "",
               "gate_script": "", "expected_outputs": "V.md|LETTER"})
    (tmp_path / "o").mkdir()
    f = tmp_path / "o" / "V.md"
    f.write_text("- **a: B**\n")
    assert q.check_expected_outputs(j)[0]                # no freshness requirement: the skip rule on a restart
    assert not q.check_expected_outputs(j, not_older_than=time.time() + 100)[0]
    assert q.check_expected_outputs(j, not_older_than=time.time() - 100)[0]
