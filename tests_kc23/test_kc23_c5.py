"""Synthetic tests for kc23_c5_leak_stats.py: find_plateau and L1/L2/L3."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c5_leak_stats import find_plateau, classify_l

RNG = np.random.default_rng(3)
N = 40


def _f1(base, noise=0.005):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def test_plateau_found_early():
    b = {1: _f1(0.80), 2: _f1(0.799), 4: _f1(0.798), 8: _f1(0.75), 16: _f1(0.70)}
    g_star, detail = find_plateau(b)
    assert g_star == 1, detail


def test_no_plateau():
    b = {1: _f1(0.90), 2: _f1(0.85), 4: _f1(0.80), 8: _f1(0.75), 16: _f1(0.70)}
    g_star, detail = find_plateau(b)
    assert g_star is None, detail


def test_l1_overlap_dominant():
    p50 = _f1(0.92)
    p0 = _f1(0.80)     # big overlap drop
    i_g = _f1(0.79)
    b_g = _f1(0.78)    # total = p50-b_g = 14pt, overlap=12pt (>=60%)
    letters, detail = classify_l(p50, p0, i_g, b_g, g_star=1, published_blocked_sd=None)
    assert "L1" in letters, (letters, detail)


def test_l2_drift_dominant():
    p50 = _f1(0.90)
    p0 = _f1(0.89)     # small overlap drop (1pt)
    i_g = _f1(0.88)    # small autocorr drop (1pt)
    b_g = _f1(0.80)    # big drift drop (8pt); total=10pt, drift=8pt (>=40%)
    letters, detail = classify_l(p50, p0, i_g, b_g, g_star=1, published_blocked_sd=None)
    assert "L2" in letters, (letters, detail)


def test_l3_no_plateau():
    letters, detail = classify_l(_f1(0.9), _f1(0.85), _f1(0.8), _f1(0.75), g_star=None,
                                 published_blocked_sd=None)
    assert letters == ["L3"], (letters, detail)


def test_l3_gstar_mismatch_vs_published():
    p50, p0, i_g, b_g = _f1(0.90), _f1(0.85), _f1(0.80), _f1(0.75)
    letters, detail = classify_l(p50, p0, i_g, b_g, g_star=4, published_blocked_sd=0.85)
    assert "L3" in letters, (letters, detail)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
