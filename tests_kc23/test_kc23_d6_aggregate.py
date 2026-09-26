"""Tests for kc23_d6_aggregate.py against the REAL output layout.

Uses the actual smoke-test fixture directories already on disk
(results_kc23_d6_smoke_marginal, _classcond, _cdan; results_kc23_d05_smoke_l2)
as the source of real column schemas, copied into the directory-naming
convention kc23_build_job_csvs.py actually uses
(results_kc23_d6_adv_marginal_l<knob>_s42, etc.) so the full discovery path
is exercised against genuine file content, not hand-typed synthetic rows.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_d6_aggregate import build_sanity, build_manipulation, run, FAMILY_KNOBS, FAMILY_OUT_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_MARGINAL = REPO_ROOT / "results_kc23_d6_smoke_marginal"
SMOKE_CLASSCOND = REPO_ROOT / "results_kc23_d6_smoke_classcond"
SMOKE_CDAN = REPO_ROOT / "results_kc23_d6_smoke_cdan"
SMOKE_SFC = REPO_ROOT / "results_kc23_d05_smoke_l2"

pytestmark = pytest.mark.skipif(not SMOKE_MARGINAL.exists(), reason="smoke fixtures not present in this checkout")


def _place_knob(root: Path, family: str, knob, source_dir: Path):
    dest = root / FAMILY_OUT_DIR[family](knob)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("adv_subjectwise.csv", "alignment_subjectwise.csv", "deep_coral_subjectwise.csv"):
        src = source_dir / name
        if src.exists():
            shutil.copy(src, dest / name)


def test_real_marginal_fixture_has_expected_columns():
    df = pd.read_csv(SMOKE_MARGINAL / "alignment_subjectwise.csv")
    assert {"subject", "domain_probe_bacc", "class_probe_tgt_bacc"}.issubset(df.columns)
    adv = pd.read_csv(SMOKE_MARGINAL / "adv_subjectwise.csv")
    assert {"subject", "adv_lambda", "f1_macro"}.issubset(adv.columns)


def _place_lambda0(root: Path):
    """The real smoke fixture was run at a NON-zero lambda; make the lambda 0 arm explicit, as the real job writes."""
    _place_knob(root, "adv_marginal", 0, SMOKE_MARGINAL)
    f = root / FAMILY_OUT_DIR["adv_marginal"](0) / "adv_subjectwise.csv"
    df = pd.read_csv(f)
    df["adv_lambda"] = 0
    df.to_csv(f, index=False)


def test_build_sanity_from_real_marginal_fixture(tmp_path):
    _place_lambda0(tmp_path)
    sanity = build_sanity(tmp_path)
    assert sanity is not None
    assert list(sanity.columns) == ["subject", "f1"]
    assert len(sanity) == 1  # the smoke fixture has one subject


def test_build_sanity_has_no_fallback_to_nonzero_lambda_rows(tmp_path):
    # The old code used ALL rows when no adv_lambda == 0 row existed, scoring the sanity gate on another lambda.
    _place_knob(tmp_path, "adv_marginal", 0, SMOKE_MARGINAL)
    assert pd.read_csv(SMOKE_MARGINAL / "adv_subjectwise.csv")["adv_lambda"].ne(0).all()
    assert build_sanity(tmp_path) is None


def test_build_sanity_missing_lambda0_returns_none(tmp_path):
    assert build_sanity(tmp_path) is None


def test_build_manipulation_incomplete_family_returns_none(tmp_path):
    # only one of adv_marginal's 7 knobs present
    _place_knob(tmp_path, "adv_marginal", 0, SMOKE_MARGINAL)
    assert build_manipulation(tmp_path, "adv_marginal") is None


def test_build_manipulation_complete_family_from_real_fixtures(tmp_path):
    # reuse the same real fixture content at every knob position -- this
    # tests the DISCOVERY/PARSING path against genuine columns, not that the
    # numbers differ per knob (they won't, since it's the same source file).
    for knob in FAMILY_KNOBS["adv_marginal"]:
        _place_knob(tmp_path, "adv_marginal", knob, SMOKE_MARGINAL)
    manip = build_manipulation(tmp_path, "adv_marginal")
    assert manip is not None
    assert set(manip.columns) == {"realization", "knob", "domain_probe", "subject_probe"}
    assert manip["knob"].nunique() == len(FAMILY_KNOBS["adv_marginal"])


def test_build_manipulation_sfc_uses_alignment_file_too(tmp_path):
    if not SMOKE_SFC.exists():
        pytest.skip("SFC smoke fixture not present")
    for knob in FAMILY_KNOBS["sfc"]:
        _place_knob(tmp_path, "sfc", knob, SMOKE_SFC)
    manip = build_manipulation(tmp_path, "sfc")
    assert manip is not None
    assert manip["knob"].nunique() == len(FAMILY_KNOBS["sfc"])


def test_run_end_to_end_writes_only_what_is_ready(tmp_path):
    root = tmp_path / "root"
    out = tmp_path / "out"
    _place_lambda0(root)
    rc = run(out, root)
    assert rc == 0
    assert (out / "d6_sanity.csv").exists()
    assert not (out / "d6_manipulation_adv_marginal.csv").exists()  # only 1/7 knobs present


def test_run_require_sanity_writes_only_sanity(tmp_path):
    root, out = tmp_path / "root", tmp_path / "out"
    _place_lambda0(root)
    assert run(out, root, require="sanity") == 0
    assert [p.name for p in out.glob("*.csv")] == ["d6_sanity.csv"]


def test_run_require_manipulation_fails_when_a_family_is_incomplete_and_leaves_nothing(tmp_path):
    root, out = tmp_path / "root", tmp_path / "out"
    for knob in FAMILY_KNOBS["adv_marginal"]:
        _place_knob(root, "adv_marginal", knob, SMOKE_MARGINAL)      # this family complete; sfc and advps absent
    assert run(out, root, require="manipulation") == 1
    assert not list(out.glob("*.csv"))


def test_run_require_sanity_fails_with_nothing_ready(tmp_path):
    assert run(tmp_path / "out", tmp_path / "root", require="sanity") == 1


def test_run_default_mode_fails_when_nothing_ready_instead_of_exit_zero(tmp_path):
    assert run(tmp_path / "out", tmp_path / "root") == 1
