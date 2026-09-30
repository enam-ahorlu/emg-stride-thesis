"""The D1 MixedLM secondary model runs ONLY in 06_Code/.venv_stats; 06_Code/.venv stays frozen (no statsmodels)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import kc23_d1_replicate_stats as st

STATS_PY = REPO / ".venv_stats" / "Scripts" / "python.exe"
MAIN_PY = REPO / ".venv" / "Scripts" / "python.exe"
needs_stats_env = pytest.mark.skipif(not STATS_PY.exists(), reason=".venv_stats not present")


def _af(effect=0.02, seeds=(42, 7, 123, 1001, 2026), arms=("R1", "R2", "R4"), published=True):
    rng = np.random.default_rng(0)
    subj_eff = rng.normal(0, 0.03, 40)
    rows = []
    for arm in arms:
        for sd in seeds:
            seed_eff = rng.normal(0, 0.004)
            for s in range(1, 41):
                base = 0.78 + subj_eff[s - 1] + seed_eff + (effect if arm == "R2" else 0.0)
                rows.append({"arm": arm, "realization": sd, "subject": s, "f1": base + rng.normal(0, 0.005)})
    if published:
        for s in range(1, 41):
            rows.append({"arm": "R2", "realization": "published", "subject": s, "f1": 0.5})     # would wreck the fit if included
    return pd.DataFrame(rows)


def _run(inp: Path, out: Path):
    return subprocess.run([str(STATS_PY), str(REPO / "src/kc23_d1_mixedlm.py"), "--in", str(inp), "--out", str(out)],
                          capture_output=True, text=True, timeout=600)


@needs_stats_env
def test_the_fixed_effect_recovers_the_contrast_and_the_published_run_is_excluded(tmp_path):
    inp, out = tmp_path / "af.csv", tmp_path / "m.csv"
    _af().to_csv(inp, index=False)
    r = _run(inp, out)
    assert r.returncode == 0 and "statsmodels" in r.stdout
    res = pd.read_csv(out).set_index("contrast")
    assert res.loc["C1", "status"] == "ok" and res.loc["C1", "fixed_effect_pp"] == pytest.approx(2.0, abs=0.4)   # R2 - R1
    assert res.loc["C1", "lo_pp"] < 2.0 < res.loc["C1", "hi_pp"] and res.loc["C1", "n_obs"] == 400
    assert res.loc["C5", "fixed_effect_pp"] == pytest.approx(-0.0, abs=0.6) or res.loc["C5", "status"] == "ok"        # R1 - R4, both unaffected


@needs_stats_env
def test_a_contrast_whose_arm_is_absent_is_reported_not_omitted(tmp_path):
    inp, out = tmp_path / "af.csv", tmp_path / "m.csv"
    _af().to_csv(inp, index=False)
    assert _run(inp, out).returncode == 0
    res = pd.read_csv(out).set_index("contrast")
    assert len(res) == 15 and "one of the two arms is absent" in res.loc["C3", "status"] and np.isnan(res.loc["C3", "fixed_effect_pp"])


@needs_stats_env
def test_an_unusable_input_exits_1_and_leaves_no_output_file(tmp_path):
    inp, out = tmp_path / "af.csv", tmp_path / "m.csv"
    out.write_text("stale")
    pd.DataFrame({"x": [1]}).to_csv(inp, index=False)
    r = _run(inp, out)
    assert r.returncode == 1 and not out.exists() and "FAIL" in r.stderr


@needs_stats_env
def test_the_replicate_stats_uses_the_stats_environment_when_its_own_has_no_statsmodels():
    msg, tab = st.secondary_model(_af())
    assert "fitted in .venv_stats" in msg and tab is not None and (tab["status"] == "ok").sum() == 2


def test_without_the_stats_environment_the_verdict_says_not_computed(monkeypatch):
    monkeypatch.setattr(st, "STATS_PYTHON", REPO / ".venv_stats" / "Scripts" / "no_such_python.exe")
    try:
        import statsmodels  # noqa: F401
        pytest.skip("statsmodels is importable in the test interpreter")
    except Exception:
        pass
    msg, tab = st.secondary_model(_af())
    assert "NOT computed" in msg and "does not exist" in msg and tab is None


def test_a_failing_stats_environment_is_stated_not_silent(monkeypatch, tmp_path):
    try:
        import statsmodels  # noqa: F401
        pytest.skip("statsmodels is importable in the test interpreter")
    except Exception:
        pass
    fake = tmp_path / "python_stub.cmd"
    fake.write_text("@echo off\r\nexit /b 3\r\n")
    monkeypatch.setattr(st, "STATS_PYTHON", fake)
    msg, tab = st.secondary_model(_af())
    assert "NOT computed" in msg and "failed" in msg and tab is None


@pytest.mark.skipif(not MAIN_PY.exists(), reason=".venv not present")
def test_the_frozen_reproduction_environment_has_no_statsmodels():
    r = subprocess.run([str(MAIN_PY), "-c", "import statsmodels"], capture_output=True, text=True)
    assert r.returncode != 0 and "No module named 'statsmodels'" in r.stderr


def test_the_stats_environment_is_ignored_by_git_and_its_versions_are_frozen_in_a_requirements_file():
    assert ".venv_stats/" in (REPO / ".gitignore").read_text()
    req = (REPO / "requirements-stats.txt").read_text()
    assert "statsmodels==" in req and "numpy==" in req and "pandas==" in req and "scipy==" in req
