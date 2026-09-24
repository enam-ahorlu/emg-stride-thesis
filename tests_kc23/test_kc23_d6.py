"""Synthetic tests for kc23_d6_stats.py: sanity PASS/FAIL, G-PASS/G-WEAK/G-FAIL,
X1-X4, C-M1-3, plus run()'s directory-name-driven dispatch and fail-closed
behavior (fixed 2026-09-24: a missing expected file used to mean "nothing to
check" (exit 0); now it's exit 20)."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_d6_stats import classify_sanity, classify_manipulation, classify_outcome, classify_mechanism, run


def test_sanity_pass():
    letter, d = classify_sanity(0.825)  # within 1.5pt of 0.830
    assert letter == "PASS", d


def test_sanity_fail():
    letter, d = classify_sanity(0.80)  # 3pt off
    assert letter == "FAIL", d


def test_gpass():
    domain = np.array([0.95, 0.85, 0.75, 0.60])  # falls 35pt
    subj = np.array([[0.9, 0.7, 0.5, 0.3], [0.92, 0.72, 0.5, 0.28], [0.88, 0.68, 0.48, 0.26]])
    letter, d = classify_manipulation(domain, subj)
    assert letter == "G-PASS", d


def test_gweak_partial_fall():
    domain = np.array([0.95, 0.92, 0.90, 0.88])  # falls only 7pt
    subj = np.array([[0.9, 0.7, 0.5, 0.3], [0.92, 0.72, 0.5, 0.28], [0.88, 0.68, 0.48, 0.26]])
    letter, d = classify_manipulation(domain, subj)
    assert letter == "G-WEAK", d


def test_gfail():
    domain = np.array([0.95, 0.945, 0.94, 0.94])  # falls <2pt
    subj = np.array([[0.9, 0.89, 0.89, 0.88]] * 3)  # flat, no trend
    letter, d = classify_manipulation(domain, subj)
    assert letter == "G-FAIL", d


def test_x1_third_axis_measured():
    invariance = np.array([0.9, 0.7, 0.5, 0.3, 0.2])  # falls (rises invariance) monotonically
    f1 = np.array([0.70, 0.76, 0.74, 0.68, 0.60])       # interior peak, falls >=2pt to last
    class_metric = np.array([0.05, 0.07, 0.065, 0.05, 0.03])  # tracks F1
    inv_trend = np.tile(invariance, (3, 1)) + np.random.default_rng(0).normal(0, 0.01, (3, 5))
    letter, d = classify_outcome(invariance, f1, class_metric, inv_trend)
    assert letter == "X1", d


def test_x2_no_falling_limb():
    invariance = np.array([0.9, 0.7, 0.5, 0.3, 0.2])
    f1 = np.array([0.70, 0.72, 0.74, 0.76, 0.78])  # rising throughout, no peak-then-fall
    class_metric = np.array([0.05, 0.06, 0.07, 0.08, 0.09])
    inv_trend = np.tile(invariance, (3, 1))
    letter, d = classify_outcome(invariance, f1, class_metric, inv_trend)
    assert letter == "X2", d


def test_x3_falling_limb_only():
    invariance = np.array([0.9, 0.7, 0.5, 0.3, 0.2])
    f1 = np.array([0.78, 0.74, 0.70, 0.66, 0.62])  # falls from the first step, no rising limb
    class_metric = np.array([0.09, 0.07, 0.05, 0.03, 0.01])
    inv_trend = np.tile(invariance, (3, 1))
    letter, d = classify_outcome(invariance, f1, class_metric, inv_trend)
    assert letter == "X3", d


def test_x4_no_invariance_movement():
    invariance = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # flat, no trend
    f1 = np.array([0.70, 0.76, 0.74, 0.68, 0.60])
    class_metric = np.array([0.05, 0.07, 0.065, 0.05, 0.03])
    inv_trend = np.tile(invariance, (3, 1)) + np.random.default_rng(1).normal(0, 0.001, (3, 5))
    letter, d = classify_outcome(invariance, f1, class_metric, inv_trend)
    assert letter == "X4", d


def test_cm1_mechanism_supported():
    letter, d = classify_mechanism(advc_at_collapse=0.74, adv_at_collapse=0.70, adv_peak=0.75,
                                   advc_invariance=0.6, adv_invariance=0.5)
    assert letter == "C-M1", d


def test_cm2_collapses_like_adv():
    letter, d = classify_mechanism(advc_at_collapse=0.70, adv_at_collapse=0.702, adv_peak=0.75,
                                   advc_invariance=0.4, adv_invariance=0.5)
    assert letter == "C-M2", d


def test_cm3_partial():
    letter, d = classify_mechanism(advc_at_collapse=0.71, adv_at_collapse=0.70, adv_peak=0.75,
                                   advc_invariance=0.4, adv_invariance=0.5)
    assert letter == "C-M3", d


def test_run_sanity_dir_fails_closed_when_file_missing(tmp_path):
    out = tmp_path / "results_kc23_d6_sanity_check"
    rc = run(out)
    assert rc == 20
    assert "FAIL" in (out / "D6_VERDICT.md").read_text()


def test_run_manipulation_dir_fails_closed_on_missing_family(tmp_path):
    out = tmp_path / "results_kc23_d6_manipulation_check"
    out.mkdir(parents=True)
    # only 2 of the 3 expected families present
    for fam in ("adv_marginal", "sfc"):
        pd.DataFrame([{"realization": 1, "knob": 0, "domain_probe": 0.9, "subject_probe": 0.9}]
                    ).to_csv(out / f"d6_manipulation_{fam}.csv", index=False)
    rc = run(out)
    assert rc == 20


def test_run_sanity_dir_passes_on_real_pass_value(tmp_path):
    out = tmp_path / "results_kc23_d6_sanity_check"
    out.mkdir(parents=True)
    pd.DataFrame([{"subject": s, "f1": 0.83} for s in range(1, 41)]).to_csv(out / "d6_sanity.csv", index=False)
    rc = run(out)
    assert rc == 0
    assert "PASS" in (out / "D6_VERDICT.md").read_text()


def test_run_sanity_dir_escalates_on_drift(tmp_path):
    out = tmp_path / "results_kc23_d6_sanity_check"
    out.mkdir(parents=True)
    pd.DataFrame([{"subject": s, "f1": 0.60} for s in range(1, 41)]).to_csv(out / "d6_sanity.csv", index=False)
    rc = run(out)
    assert rc == 20
    assert "FAIL" in (out / "D6_VERDICT.md").read_text()


def test_run_manipulation_dir_gpass_end_to_end(tmp_path):
    out = tmp_path / "results_kc23_d6_manipulation_check"
    out.mkdir(parents=True)
    knobs = [0, 1, 2, 3]
    for fam in ("adv_marginal", "sfc", "advps"):
        rows = []
        for real in range(1, 41):
            for i, k in enumerate(knobs):
                # domain probe falls by 20pt across the grid; subject probe falls monotonically too
                rows.append({"realization": real, "knob": k,
                            "domain_probe": 0.95 - 0.20 * (i / (len(knobs) - 1)),
                            "subject_probe": 0.90 - 0.30 * (i / (len(knobs) - 1))})
        pd.DataFrame(rows).to_csv(out / f"d6_manipulation_{fam}.csv", index=False)
    rc = run(out)
    assert rc == 0
    gates = pd.read_csv(out / "D6_gates.csv").set_index("item")["letter"]
    assert gates["manipulation_adv_marginal"] == "G-PASS"


def test_run_manipulation_dir_gfail_end_to_end(tmp_path):
    out = tmp_path / "results_kc23_d6_manipulation_check"
    out.mkdir(parents=True)
    knobs = [0, 1, 2, 3]
    for fam in ("adv_marginal", "sfc", "advps"):
        rows = []
        for real in range(1, 41):
            for k in knobs:
                # domain probe barely moves (<2pt) -- G-FAIL regardless of the subject probe
                rows.append({"realization": real, "knob": k, "domain_probe": 0.90 - 0.005 * k,
                            "subject_probe": 0.90 - 0.30 * (k / max(knobs))})
        pd.DataFrame(rows).to_csv(out / f"d6_manipulation_{fam}.csv", index=False)
    rc = run(out)
    assert rc in (0, 10)
    gates = pd.read_csv(out / "D6_gates.csv").set_index("item")["letter"]
    assert gates["manipulation_adv_marginal"] == "G-FAIL"


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
