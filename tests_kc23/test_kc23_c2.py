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
