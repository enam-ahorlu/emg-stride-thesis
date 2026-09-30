"""src/kc23_c3_edge_job_gen.py: the edge-rule rerun rows (extend a triggered axis by two steps, once)."""
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
import kc23_c3_edge_job_gen as g

REPO = Path(__file__).resolve().parents[2]
FIELDS = ["job_id", "stage", "seed", "command", "out_dir", "depends_on", "gate_script", "expected_outputs", "light"]


def _table(rows):
    cols = ["norm", "axis", "low_edge", "folds_at_low_edge", "high_edge", "folds_at_high_edge", "triggered",
            "extension_already_applied"]
    return pd.DataFrame(rows, columns=cols)


def _row(norm, axis, low, high, applied=False):
    trig = (low > 10 or high > 10) and not applied
    lo_e, hi_e = (0.01, 30) if axis == "C" else (0.01, 10)
    return [norm, axis, lo_e, low, hi_e, high, trig, applied]


def _csvs(tmp: Path, table):
    t = tmp / "edge.csv"
    table.to_csv(t, index=False)
    c = tmp / "cpu.csv"
    with open(c, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow({"job_id": "c3_svm_per_subject", "stage": "C3", "command": "x", "out_dir": "o", "depends_on": ""})
    return t, c


def _run(*argv):
    return subprocess.run([sys.executable, str(REPO / "src/kc23_c3_edge_job_gen.py"), *argv], capture_output=True, text=True)


def test_extension_is_two_steps_on_the_side_that_hit_the_edge():
    t = _table([_row("per_subject", "C", 0, 14), _row("per_subject", "gamma multiplier", 0, 3)])
    c, gm = g.extended_grids(t)
    assert c == [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300] and gm == g.BASE_GAMMA
    t = _table([_row("per_subject", "C", 12, 0), _row("per_subject", "gamma multiplier", 0, 15)])
    c, gm = g.extended_grids(t)
    assert c == [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30] and gm == [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]


def test_both_edges_of_one_axis_extend_both_ways():
    c, _ = g.extended_grids(_table([_row("global", "C", 11, 11), _row("global", "gamma multiplier", 0, 0)]))
    assert c[0] == 0.001 and c[-1] == 300 and len(c) == 12


def test_exactly_10_folds_does_not_trigger_the_rule():
    assert g.build_rows(_table([_row("per_subject", "C", 0, 10), _row("per_subject", "gamma multiplier", 10, 0)])) == []


def test_an_extension_already_applied_is_never_extended_again():
    t = _table([_row("per_subject", "C", 0, 20, applied=True), _row("per_subject", "gamma multiplier", 0, 0)])
    assert g.build_rows(t) == []


def test_rows_appended_for_the_triggered_normalizations_only_with_the_two_new_flags(tmp_path):
    t, c = _csvs(tmp_path, _table([_row("per_subject", "C", 0, 22), _row("per_subject", "gamma multiplier", 0, 0),
                                   _row("global", "C", 0, 2), _row("global", "gamma multiplier", 0, 0)]))
    r = _run("--edge-table", str(t), "--cpu-csv", str(c))
    assert r.returncode == 0 and "appended 1" in r.stdout
    rows = {x["job_id"]: x for x in csv.DictReader(open(c, newline="", encoding="utf-8"))}
    assert set(rows) == {"c3_svm_per_subject", "c3_svm_per_subject_edge"}
    e = rows["c3_svm_per_subject_edge"]
    assert "--svm-c-grid 0.01,0.03,0.1,0.3,1,3,10,30,100,300" in e["command"]
    assert "--svm-gamma-mult-grid 0.01,0.1,0.3,1,3,10" in e["command"] and "--norm-mode per_subject" in e["command"]
    assert e["out_dir"] == "results/kc23_c3_svm_per_subject_edge" and e["depends_on"] == "c3_svm_per_subject"
    assert e["expected_outputs"] == "*_nested_loso_subjectwise.csv|40"


def test_it_is_idempotent(tmp_path):
    t, c = _csvs(tmp_path, _table([_row("per_subject", "C", 0, 22), _row("per_subject", "gamma multiplier", 0, 0)]))
    _run("--edge-table", str(t), "--cpu-csv", str(c))
    assert "appended 0" in _run("--edge-table", str(t), "--cpu-csv", str(c)).stdout
    assert len(list(csv.DictReader(open(c, newline="", encoding="utf-8")))) == 2


def test_no_trigger_appends_nothing(tmp_path):
    t, c = _csvs(tmp_path, _table([_row("per_subject", "C", 1, 2), _row("per_subject", "gamma multiplier", 0, 0)]))
    before = c.read_text()
    r = _run("--edge-table", str(t), "--cpu-csv", str(c))
    assert r.returncode == 0 and "did not trigger" in r.stdout and c.read_text() == before


def test_a_missing_or_malformed_edge_table_cannot_decide(tmp_path):
    _, c = _csvs(tmp_path, _table([]))
    before = c.read_text()
    assert _run("--edge-table", str(tmp_path / "none.csv"), "--cpu-csv", str(c)).returncode == 2
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    assert _run("--edge-table", str(bad), "--cpu-csv", str(c)).returncode == 2 and c.read_text() == before


def test_the_grid_flags_are_the_real_svm_runner_flags():
    import numpy as np
    import train_classical_loso as t
    X = np.random.default_rng(0).normal(0, 1, (120, 8))
    grid, scale, mults = t.svm_extended_grid(X, [], t._csv_floats("0.01,0.03,0.1,0.3,1,3,10,30,100,300"),
                                             t._csv_floats("0.01,0.1,0.3,1,3,10,30,100"))
    assert len(grid["clf__C"]) == 10 and mults[-1] == 100
