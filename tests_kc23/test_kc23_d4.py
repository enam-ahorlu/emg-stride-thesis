"""Synthetic tests for kc23_d4_invariance_stats.py: T1/T2/T3."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_d4_invariance_stats import classify_t, gainjitter_boundary

N_REAL = 3
DOSES = [0.40, 0.50, 0.60, 0.80, 1.00]


def _rep(row, noise=0.005, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([row + rng.normal(0, noise, len(row)) for _ in range(N_REAL)])


def test_t1_channel_measures_invariance():
    # subject probe falls monotonically with dose (invariance rises)
    probe = _rep(np.array([0.9, 0.8, 0.65, 0.45, 0.30]), seed=1)
    # F1 peaks early then falls >= 2pt by the top dose
    f1 = _rep(np.array([0.75, 0.76, 0.74, 0.70, 0.66]), seed=2)
    # class silhouette tracks F1 (falls together)
    sil = _rep(np.array([0.08, 0.075, 0.06, 0.04, 0.02]), seed=3)
    letter, detail = classify_t(probe, f1, sil, probe)
    assert letter == "T1", detail
    assert detail["probe_falls"], detail
    assert detail["f1_significant_fall"], detail


def test_t2_probe_does_not_fall():
    probe = _rep(np.array([0.5, 0.5, 0.5, 0.5, 0.5]), noise=0.01, seed=4)
    f1 = _rep(np.array([0.75, 0.74, 0.73, 0.70, 0.68]), seed=5)
    sil = _rep(np.array([0.08, 0.07, 0.06, 0.05, 0.04]), seed=6)
    letter, detail = classify_t(probe, f1, sil, probe)
    assert letter == "T2", detail
    assert not detail["probe_falls"]


def test_t3_shape_without_class_tracking():
    probe = _rep(np.array([0.9, 0.8, 0.65, 0.45, 0.30]), seed=7)
    f1 = _rep(np.array([0.75, 0.76, 0.74, 0.70, 0.66]), seed=8)
    # class silhouette does NOT track F1: it rises while F1 falls
    sil = _rep(np.array([0.02, 0.03, 0.05, 0.07, 0.09]), seed=9)
    letter, detail = classify_t(probe, f1, sil, probe)
    assert letter == "T3", detail
    assert not detail["tracks_class"], detail


def test_gainjitter_boundary_found():
    gj = {0.30: 0.75, 0.40: 0.76, 0.50: 0.74, 0.80: 0.70, 1.00: 0.65}
    note = gainjitter_boundary(gj)
    assert "boundary found" in note, note


def test_gainjitter_no_boundary():
    gj = {0.30: 0.70, 0.40: 0.72, 0.50: 0.74, 0.80: 0.76, 1.00: 0.78}
    note = gainjitter_boundary(gj)
    assert "no boundary" in note, note


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
