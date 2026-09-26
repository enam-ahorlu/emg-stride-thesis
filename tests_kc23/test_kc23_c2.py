"""Synthetic tests for kc23_c2_whitening_stats.py, hitting every letter of
both C2.4 outcome grids (Endpoint 1: W1/W2/W3; Endpoint 2: M1/M2/M3)."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c2_whitening_stats import classify_w, classify_m

RNG = np.random.default_rng(0)
N = 40


def _f1(base, noise=0.02):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def test_w1_large_penalty():
    f1_r3 = _f1(0.80)
    f1_4lw = _f1(0.70)  # 10pp penalty
    letter, t = classify_w(f1_r3, f1_4lw)
    assert letter == "W1", t


def test_w2_moderate_penalty():
    f1_r3 = _f1(0.80, noise=0.005)
    f1_4lw = _f1(0.775, noise=0.005)  # ~2.5pp penalty, tight noise -> significant
    letter, t = classify_w(f1_r3, f1_4lw)
    assert letter == "W2", t


def test_w3_negligible_penalty():
    f1_r3 = _f1(0.80)
    f1_4lw = _f1(0.799)  # ~0.1pp penalty
    letter, t = classify_w(f1_r3, f1_4lw)
    assert letter == "W3", t


def test_w3_not_significant():
    f1_r3 = _f1(0.80, noise=0.15)
    f1_4lw = _f1(0.79, noise=0.15)  # noisy, likely not significant even if >1pp on average
    letter, t = classify_w(f1_r3, f1_4lw)
    assert letter == "W3" or t["p_raw"] >= 0.05


def test_m1_mechanism_supported():
    f1_r3 = _f1(0.75, noise=0.01)
    f1_4o = _f1(0.745, noise=0.01)   # within 1pt of rung3
    f1_4lw = _f1(0.70, noise=0.01)   # 4o - 4lw >= 2pt (0.745-0.70=4.5pt)
    letter, r3, lw = classify_m(f1_r3, f1_4o, f1_4lw)
    assert letter == "M1", (r3, lw)


def test_m2_mechanism_unsupported():
    f1_r3 = _f1(0.75)
    f1_4o = _f1(0.70)
    f1_4lw = _f1(0.699)  # 4o falls about as far as 4lw
    letter, r3, lw = classify_m(f1_r3, f1_4o, f1_4lw)
    assert letter == "M2", (r3, lw)


def test_m3_partial():
    f1_r3 = _f1(0.75, noise=0.005)
    f1_4o = _f1(0.735, noise=0.005)   # 1.5pt below rung3 (not >= -1)
    f1_4lw = _f1(0.72, noise=0.005)   # 1.5pt below 4o (not < 1pt gap)
    letter, r3, lw = classify_m(f1_r3, f1_4o, f1_4lw)
    assert letter == "M3", (r3, lw)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


# ---------------------------------------------------------------------------------------------------------------
# 26 September 2026: Endpoint 1 for EVERY deployable variant, the subject probe under 4o, fail-closed inputs.
import re
import pandas as pd
import pytest
from kc23_c2_whitening_stats import run as c2_run, VARIANTS

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")


def _write_ladder(d, gaps=None, geometry=False, subjects=40):
    d.mkdir(parents=True, exist_ok=True)
    gaps = gaps or {"4b": 0.06, "4c": 0.07, "4d": 0.08, "4lw": 0.09, "4o": 0.01}
    base = np.linspace(0.70, 0.82, subjects)
    rung = {"3": base, **{k: base - g + 0.0003 * np.arange(subjects) % 3 * 0.001 for k, g in gaps.items()}}
    for r, v in rung.items():
        pd.DataFrame({"subject": range(1, subjects + 1), "f1_macro": v}).to_csv(d / f"ladder_loso_{r}_SVM_subjectwise.csv", index=False)
    if geometry:
        pd.DataFrame([{"rung": r, "subject_probe_linear": pr, "chance_floor": 0.025}
                      for r, pr in (("0", 0.78), ("3", 0.02), ("4o", 0.03))]).to_csv(d / "ladder_geometry.csv", index=False)


def test_c2_reports_every_deployable_variant_not_just_4lw(tmp_path):
    _write_ladder(tmp_path / "results_kc23_c2_whitening_w250")
    out = tmp_path / "results_kc23_c2_whitening_w250"
    assert c2_run(out) in (0, 10)
    v = (out / "C2_VERDICT.md").read_text()
    for var in VARIANTS:
        assert f"| {var} |" in v
    pen = pd.read_csv(out / "C2_variant_penalties.csv").set_index("variant")
    assert set(pen.index) == set(VARIANTS) and pen.loc["4lw", "delta_pp"] == pytest.approx(9.0, abs=0.3)
    assert "**Endpoint 1 outcome: W1**" in v                       # a 9 pt penalty on 4lw


def test_c2_letter_still_conditions_on_4lw_only(tmp_path):
    # 4b has a huge penalty but 4lw a tiny one: the headline letter must follow 4lw (W3, escalate)
    out = tmp_path / "results_kc23_c2_whitening_w250"
    _write_ladder(out, gaps={"4b": 0.12, "4c": 0.12, "4d": 0.12, "4lw": 0.001, "4o": 0.0})
    assert c2_run(out) == 20
    assert "**Endpoint 1 outcome: W3**" in (out / "C2_VERDICT.md").read_text()


def test_c2_subject_probe_under_4o_is_reported_when_geometry_exists_and_said_absent_otherwise(tmp_path):
    a = tmp_path / "results_kc23_c2_whitening_w250"; _write_ladder(a, geometry=True); c2_run(a)
    assert "Subject probe under 4o (linear, class-pooled): 0.030" in (a / "C2_VERDICT.md").read_text()
    b = tmp_path / "results_kc23_c2_whitening_w400"; _write_ladder(b); c2_run(b)
    assert "Subject probe under 4o: NOT reported" in (b / "C2_VERDICT.md").read_text()


@pytest.mark.parametrize("missing", ["3", "4b", "4c", "4d", "4lw", "4o"])
def test_c2_each_ladder_file_missing_fails_closed_with_no_letter(tmp_path, missing):
    out = tmp_path / "results_kc23_c2_whitening_w250"
    _write_ladder(out)
    (out / f"ladder_loso_{missing}_SVM_subjectwise.csv").unlink()
    assert c2_run(out) == 20 and not LETTER_RE.search((out / "C2_VERDICT.md").read_text())


def test_c2_incomplete_subject_count_fails_closed(tmp_path):
    out = tmp_path / "results_kc23_c2_whitening_w250"
    _write_ladder(out)
    p = out / "ladder_loso_4c_SVM_subjectwise.csv"
    pd.read_csv(p).iloc[:30].to_csv(p, index=False)
    assert c2_run(out) == 20


def test_c2_stale_verdict_is_replaced_when_an_input_disappears(tmp_path):
    out = tmp_path / "results_kc23_c2_whitening_w250"
    _write_ladder(out)
    assert c2_run(out) in (0, 10)
    (out / "ladder_loso_4d_SVM_subjectwise.csv").unlink()
    assert c2_run(out) == 20 and not LETTER_RE.search((out / "C2_VERDICT.md").read_text())
    assert not (out / "C2_tests.csv").exists()
