"""Per-input fail-closed tests (fail-open sweep, level 2, 25 September 2026):
remove ONE required input at a time from a complete synthetic fixture and
require a non-zero exit and a verdict with no outcome line. Covers the scripts
whose inputs are plain files and that had no such test: C4 and the S3
inventory. (S1, D1, D5, D6, S2, C3 and C5 carry their own in their test files.)
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_c4_feature_stats as c4
import kc23_s3_inventory as s3

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")


def _svm_dir(root: Path, name: str, mean: float):
    d = root / name
    d.mkdir(parents=True)
    f1 = np.full(40, mean) + np.linspace(-0.01, 0.01, 40)
    pd.DataFrame({"subject": range(1, 41), "f1_macro": f1}).to_csv(
        d / "features__SVM_nested_loso_subjectwise.csv", index=False)


def _c4_root(root: Path):
    _svm_dir(root, c4.FREQ72_DIR, 0.777)
    for feat in c4.FEATURE_SETS:
        _svm_dir(root, f"results_kc23_c4_{feat}_svm_per_subject", 0.78)
        _svm_dir(root, f"results_kc23_c4_{feat}_svm_global", 0.74)


def test_c4_complete_fixture_computes_a_letter(tmp_path, monkeypatch):
    monkeypatch.setattr(c4, "ROOT", tmp_path)
    _c4_root(tmp_path)
    out = tmp_path / "results_kc23_c4_rich126_lda_global"
    assert c4.run(out) == 0
    assert "**Outcome: F-A**" in (out / "C4_VERDICT.md").read_text()


@pytest.mark.parametrize("victim", [c4.FREQ72_DIR] + [f"results_kc23_c4_{f}_svm_{n}" for f in c4.FEATURE_SETS
                                                       for n in ("per_subject", "global")])
def test_c4_each_input_missing_fails_closed(tmp_path, monkeypatch, victim):
    monkeypatch.setattr(c4, "ROOT", tmp_path)
    _c4_root(tmp_path)
    shutil.rmtree(tmp_path / victim)
    out = tmp_path / "results_kc23_c4_rich126_lda_global"
    assert c4.run(out) == 20
    assert not LETTER_RE.search((out / "C4_VERDICT.md").read_text())


def test_c4_has_no_published_number_fallback_for_freq72(tmp_path, monkeypatch):
    # the old code used np.full(40, 0.777) when results_loso_freq_persubj was absent
    monkeypatch.setattr(c4, "ROOT", tmp_path)
    _c4_root(tmp_path)
    shutil.rmtree(tmp_path / c4.FREQ72_DIR)
    assert c4.run(tmp_path / "out") == 20


def test_c4_incomplete_subject_count_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(c4, "ROOT", tmp_path)
    _c4_root(tmp_path)
    p = next((tmp_path / "results_kc23_c4_tdpsd54_svm_global").glob("*.csv"))
    pd.read_csv(p).iloc[:30].to_csv(p, index=False)
    assert c4.run(tmp_path / "out") == 20


def test_c4_stale_verdict_is_replaced_when_an_input_disappears(tmp_path, monkeypatch):
    monkeypatch.setattr(c4, "ROOT", tmp_path)
    _c4_root(tmp_path)
    out = tmp_path / "out"
    assert c4.run(out) == 0
    shutil.rmtree(tmp_path / "results_kc23_c4_rich126_svm_global")
    assert c4.run(out) == 20
    assert not LETTER_RE.search((out / "C4_VERDICT.md").read_text())
    assert not (out / "C4_tests.csv").exists()


def test_s3_inventory_missing_manifest_exits_nonzero_and_writes_nothing(tmp_path, monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("run manifest missing")
    monkeypatch.setattr(s3, "build_inventory", missing)
    out = tmp_path / "out"
    assert s3.run(out) == 1
    assert not (out / "active_only_inventory.csv").exists()


def test_s3_build_inventory_raises_on_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        s3.build_inventory(tmp_path / "nope.csv")


def test_s3_build_inventory_raises_on_manifest_without_dir_column(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame({"x": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        s3.build_inventory(p)
