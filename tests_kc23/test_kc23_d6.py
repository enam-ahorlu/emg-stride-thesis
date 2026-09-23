"""Synthetic tests for kc23_d6_stats.py: sanity PASS/FAIL, G-PASS/G-WEAK/G-FAIL,
X1-X4, C-M1-3."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_d6_stats import classify_sanity, classify_manipulation, classify_outcome, classify_mechanism


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


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
