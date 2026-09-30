"""src/kc23_s3_extra_job_gen.py: the SVM-X and HGB active-only cells, only if KC-C3 lands P2 or P3."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
import kc23_s3_extra_job_gen as g

REPO = Path(__file__).resolve().parents[2]
FIELDS = ["job_id", "stage", "seed", "command", "out_dir", "depends_on", "gate_script", "expected_outputs", "light"]


def _cpu_csv(p: Path):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow({"job_id": "s3_inventory", "stage": "S3", "command": "x", "out_dir": "o", "depends_on": ""})
        w.writerow({"job_id": "s3_benchmark", "stage": "S3", "command": "y", "out_dir": "o2", "depends_on": "s3_inventory;c3_ensemble",
                    "expected_outputs": "benchmark_active_only.csv|EXISTS"})


def _verdict(p: Path, line: str):
    p.write_text(f"# KC-C3 verdict\n\n{line}\n", encoding="utf-8")


def _run(*argv):
    return subprocess.run([sys.executable, str(REPO / "src/kc23_s3_extra_job_gen.py"), *argv], capture_output=True, text=True)


@pytest.mark.parametrize("line,needed", [("**Outcomes: P2, N1, E1**", True), ("**Outcomes: P3, N2, E1**", True),
                                         ("**Outcomes: P1, N1, E1**", False), ("**Outcomes: P-OUT, N1, E1**", False)])
def test_the_condition_is_p2_or_p3_only(tmp_path, line, needed):
    v = tmp_path / "v.md"
    _verdict(v, line)
    assert g.decide(g.c3_letters(v))[0] is needed


def test_p_out_is_stated_as_not_covered(tmp_path):
    v = tmp_path / "v.md"
    _verdict(v, "**Outcomes: P-OUT, N1, E1**")
    assert "P-OUT" in g.decide(g.c3_letters(v))[1] and "P2 or P3" in g.decide(g.c3_letters(v))[1]


def test_p2_appends_four_instrumented_rows_and_makes_the_benchmark_wait_for_them(tmp_path):
    v, csvp = tmp_path / "v.md", tmp_path / "cpu.csv"
    _verdict(v, "**Outcomes: P2, N1, E1**")
    _cpu_csv(csvp)
    r = _run("--verdict", str(v), "--cpu-csv", str(csvp))
    assert r.returncode == 0 and "appended 4" in r.stdout
    rows = {x["job_id"]: x for x in csv.DictReader(open(csvp, newline="", encoding="utf-8"))}
    new = ["s3_svmx_global", "s3_svmx_per_subject", "s3_hgb_global", "s3_hgb_per_subject"]
    assert all(n in rows for n in new)
    for n in new:
        assert "--flush-preds" in rows[n]["command"] and rows[n]["expected_outputs"] == "*_nested_loso_subjectwise.csv|40"
        assert "Aonly" in rows[n]["command"]
    assert "--grid extended --search grid" in rows["s3_svmx_per_subject"]["command"]
    assert "--models HGB" in rows["s3_hgb_global"]["command"]
    deps = rows["s3_benchmark"]["depends_on"].split(";")
    assert set(new) <= set(deps) and deps[:2] == ["s3_inventory", "c3_ensemble"]


def test_it_is_idempotent(tmp_path):
    v, csvp = tmp_path / "v.md", tmp_path / "cpu.csv"
    _verdict(v, "**Outcomes: P3, N1, E1**")
    _cpu_csv(csvp)
    _run("--verdict", str(v), "--cpu-csv", str(csvp))
    r = _run("--verdict", str(v), "--cpu-csv", str(csvp))
    assert "appended 0" in r.stdout
    rows = list(csv.DictReader(open(csvp, newline="", encoding="utf-8")))
    assert len(rows) == 6 and len({x["job_id"] for x in rows}) == 6
    deps = next(x for x in rows if x["job_id"] == "s3_benchmark")["depends_on"].split(";")
    assert len(deps) == len(set(deps))


@pytest.mark.parametrize("line", ["**Outcomes: P1, N1, E1**", "**Outcomes: P-OUT, N1, E1**"])
def test_nothing_is_appended_when_the_cells_are_not_required(tmp_path, line):
    v, csvp = tmp_path / "v.md", tmp_path / "cpu.csv"
    _verdict(v, line)
    _cpu_csv(csvp)
    before = csvp.read_text()
    r = _run("--verdict", str(v), "--cpu-csv", str(csvp))
    assert r.returncode == 0 and "not required" in r.stdout and csvp.read_text() == before


def test_a_missing_or_letterless_verdict_cannot_decide_and_appends_nothing(tmp_path):
    csvp = tmp_path / "cpu.csv"
    _cpu_csv(csvp)
    before = csvp.read_text()
    assert _run("--verdict", str(tmp_path / "none.md"), "--cpu-csv", str(csvp)).returncode == 2
    v = tmp_path / "v.md"
    v.write_text("# KC-C3 verdict\n\nNO OUTCOME COMPUTED. something missing\n", encoding="utf-8")
    assert _run("--verdict", str(v), "--cpu-csv", str(csvp)).returncode == 2
    assert csvp.read_text() == before
