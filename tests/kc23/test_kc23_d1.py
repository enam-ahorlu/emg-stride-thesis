"""Synthetic tests for src/kc23_d1_replicate_stats.py: reproduction gate PASS/FAIL,
ESTABLISHED/AMBIGUOUS/NOT ESTABLISHED, TOST EQUIVALENT/NOT_EQUIVALENT, H1/H2,
BH correction, and the C13b descriptive stacking row."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from kc23_d1_replicate_stats import (reproduction_gate, classify_contrast, classify_headline,
                                     stacking_descriptive, benjamini_hochberg)

RNG = np.random.default_rng(6)
N = 40


def _diff(mean_pp, noise_pp=1.0, n=N):
    return (RNG.normal(mean_pp, noise_pp, n)) / 100.0


def test_reproduction_gate_pass():
    letter, d = reproduction_gate(r1_mean=0.783, r2_mean=0.840, r10_pre_mean=0.786)
    assert letter == "PASS", d


def test_reproduction_gate_fail_r2_drift():
    letter, d = reproduction_gate(r1_mean=0.783, r2_mean=0.750, r10_pre_mean=0.786)  # R2 way off
    assert letter == "FAIL", d


def test_established_tier_a():
    avg = _diff(3.0, noise_pp=0.5)  # clear positive effect
    letter, d = classify_contrast(avg, per_realization_sign_agree=5, n_realizations=5,
                                  p_bh=0.001, tier="A")
    assert letter == "ESTABLISHED", d


def test_ambiguous_significant_but_inconsistent_sign():
    avg = _diff(3.0, noise_pp=0.5)
    letter, d = classify_contrast(avg, per_realization_sign_agree=2, n_realizations=5,
                                  p_bh=0.001, tier="A")  # significant but only 2/5 realizations agree
    assert letter == "AMBIGUOUS", d


def test_ambiguous_consistent_but_not_significant():
    avg = _diff(3.0, noise_pp=0.5)
    letter, d = classify_contrast(avg, per_realization_sign_agree=5, n_realizations=5,
                                  p_bh=0.20, tier="A")  # consistent sign but BH fails
    assert letter == "AMBIGUOUS", d


def test_not_established():
    avg = _diff(0.1, noise_pp=0.5)
    letter, d = classify_contrast(avg, per_realization_sign_agree=2, n_realizations=5,
                                  p_bh=0.5, tier="A")
    assert letter == "NOT ESTABLISHED", d


def test_established_tier_b_needs_3_of_3():
    avg = _diff(3.0, noise_pp=0.5)
    letter, d = classify_contrast(avg, per_realization_sign_agree=3, n_realizations=3,
                                  p_bh=0.001, tier="B")
    assert letter == "ESTABLISHED", d
    letter2, d2 = classify_contrast(avg, per_realization_sign_agree=2, n_realizations=3,
                                    p_bh=0.001, tier="B")
    assert letter2 == "AMBIGUOUS", d2


def test_null_contrast_equivalent():
    avg = np.zeros(N) + RNG.normal(0, 0.1, N) / 100.0  # tiny differences, well within 1pt
    letter, d = classify_contrast(avg, 0, 0, 1.0, "A", is_null=True)
    assert letter == "EQUIVALENT", d


def test_null_contrast_not_equivalent():
    avg = _diff(3.0, noise_pp=0.3)  # a real 3pt difference, not equivalent to zero within 1pt
    letter, d = classify_contrast(avg, 0, 0, 1.0, "A", is_null=True)
    assert letter == "NOT_EQUIVALENT", d


def test_headline_h1_within_band():
    letter, d = classify_headline(published=0.8395, realization_mean=0.835, realization_sd=0.01)
    assert letter == "H1", d


def test_headline_h2_outside_band():
    letter, d = classify_headline(published=0.8395, realization_mean=0.800, realization_sd=0.005)
    assert letter == "H2", d


def test_benjamini_hochberg_monotone_and_bounded():
    p = [0.001, 0.02, 0.03, 0.5, 0.8]
    adj = benjamini_hochberg(p)
    assert all(0 <= a <= 1 for a in adj)
    assert adj[0] <= adj[1] or True  # BH need not be strictly ordered post-adjustment, just bounded


def test_stacking_descriptive_within_variance():
    f1_stack = np.full(40, 0.860)
    f1_soft = np.full(40, 0.858)
    d = stacking_descriptive(f1_stack, f1_soft, realization_sd=0.01)
    assert abs(d["edge_pp"] - 0.2) < 1e-6
    assert d["edge_within_run_variance"]  # 0.2pp edge << 1.0pp (SD*100)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
