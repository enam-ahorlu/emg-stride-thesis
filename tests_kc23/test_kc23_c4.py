"""kc23_c4_feature_stats.py: F-A/F-B/F-C, F-OUT for the cases the plan's table does not cover, and run()'s real-directory
inputs (SVM-X comparator is SVM-X on Freq-72; LDA reading beside it; every input required)."""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c4_feature_stats import classify_set, classify_fa, mark_norm_gain, run, READINGS, FEATURE_SETS

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
RNG = np.random.default_rng(2)
N = 40


def _f1(base, noise=0.01):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def _results(freq72, persubj_means, global_mean=0.72):
    res = {}
    for feat, m in persubj_means.items():
        res[feat] = classify_set(_f1(m, 0.005), _f1(global_mean, 0.005), freq72)
    return mark_norm_gain(res)


def test_fa_extends():
    freq72 = _f1(0.777, noise=0.005)
    assert classify_fa(_results(freq72, {"tdpsd54": 0.780, "rich126": 0.780})) == "F-A"


def test_fb_moderate_gain():
    freq72 = _f1(0.777, noise=0.005)
    assert classify_fa(_results(freq72, {"tdpsd54": 0.790, "rich126": 0.780})) == "F-B"


def test_fc_large_gain():
    freq72 = _f1(0.777, noise=0.005)
    assert classify_fa(_results(freq72, {"tdpsd54": 0.810, "rich126": 0.780})) == "F-C"


def test_f_out_when_a_set_loses_more_than_a_point():
    freq72 = _f1(0.777, noise=0.005)
    assert classify_fa(_results(freq72, {"tdpsd54": 0.740, "rich126": 0.780})) == "F-OUT"


def test_f_out_when_within_a_point_but_the_normalization_gain_is_absent():
    freq72 = _f1(0.777, noise=0.005)
    res = _results(freq72, {"tdpsd54": 0.780, "rich126": 0.780}, global_mean=0.779)   # per-subject barely above global
    res["rich126"]["norm_gain_present"] = False
    assert classify_fa(res) == "F-OUT"


def test_the_norm_gain_needs_to_be_positive_and_significant():
    freq72 = _f1(0.777, noise=0.005)
    res = _results(freq72, {"tdpsd54": 0.780, "rich126": 0.780}, global_mean=0.82)    # per-subject WORSE than global
    assert not any(r["norm_gain_present"] for r in res.values())


def _write(d: Path, mean, rng, lda=False):
    d.mkdir(parents=True, exist_ok=True)
    f1 = np.clip(mean + rng.normal(0, 0.005, N), 0.05, 0.99)
    name = "lda_subjectwise.csv" if lda else "x__SVM_nested_loso_subjectwise.csv"
    pd.DataFrame({"subject": range(1, N + 1), "f1_macro": f1}).to_csv(d / name, index=False)


def _tree(root: Path, svm_new=0.78, lda_new=0.78, seed=0):
    rng = np.random.default_rng(seed)
    for reading, (comp, pattern) in READINGS.items():
        lda = reading == "LDA"
        _write(root / comp, 0.777 if not lda else 0.70, rng, lda)
        for feat in FEATURE_SETS:
            m = lda_new if lda else svm_new
            _write(root / pattern.format(feat=feat, norm="per_subject"), m if not lda else 0.70 + (m - 0.78), rng, lda)
            _write(root / pattern.format(feat=feat, norm="global"), 0.65, rng, lda)
    return root


def test_run_end_to_end_and_the_comparator_is_svmx_on_freq72_not_the_published_svm(tmp_path):
    assert READINGS["SVM-X"][0] == "results_kc23_c3_svm_per_subject"
    _tree(tmp_path)
    out = tmp_path / "o"
    assert run(out, root=tmp_path) == 0
    v = (out / "C4_VERDICT.md").read_text()
    assert "**Outcome: F-A**" in v and "LDA reading" in v and LETTER_RE.search(v)
    assert set(pd.read_csv(out / "C4_tests.csv")["reading"]) == {"SVM-X", "LDA"}


def test_run_escalates_when_the_lda_reading_alone_escalates(tmp_path):
    _tree(tmp_path, svm_new=0.78, lda_new=0.84)          # LDA sets gain 6 pt over Freq-72 LDA
    out = tmp_path / "o"
    assert run(out, root=tmp_path) == 20
    assert "**Outcome: F-A**" in (out / "C4_VERDICT.md").read_text()      # the letter is the SVM-X reading


@pytest.mark.parametrize("victim", ["results_kc23_c3_svm_per_subject", "results_lda_persubj",
                                    "results_kc23_c4_rich126_lda_global", "results_kc23_c4_tdpsd54_svm_per_subject"])
def test_run_fails_closed_on_any_missing_input(tmp_path, victim):
    _tree(tmp_path)
    shutil.rmtree(tmp_path / victim)
    out = tmp_path / "o"
    assert run(out, root=tmp_path) == 20
    assert not LETTER_RE.search((out / "C4_VERDICT.md").read_text()) and not (out / "C4_tests.csv").exists()


def test_run_exit_10_on_f_out(tmp_path):
    _tree(tmp_path, svm_new=0.72)
    out = tmp_path / "o"
    assert run(out, root=tmp_path) == 10
    assert "**Outcome: F-OUT**" in (out / "C4_VERDICT.md").read_text()


def test_a_stale_verdict_is_replaced_when_an_input_disappears(tmp_path):
    _tree(tmp_path)
    out = tmp_path / "o"
    assert run(out, root=tmp_path) == 0
    shutil.rmtree(tmp_path / "results_kc23_c4_rich126_svm_global")
    assert run(out, root=tmp_path) == 20
    assert not LETTER_RE.search((out / "C4_VERDICT.md").read_text()) and not (out / "C4_tests.csv").exists()


def test_there_is_no_published_number_fallback_and_an_incomplete_subject_count_fails(tmp_path):
    _tree(tmp_path)
    p = next((tmp_path / "results_kc23_c4_tdpsd54_svm_global").glob("*.csv"))
    pd.read_csv(p).iloc[:30].to_csv(p, index=False)
    assert run(tmp_path / "o", root=tmp_path) == 20
