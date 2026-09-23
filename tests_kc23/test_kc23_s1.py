"""Synthetic tests for kc23_s1_scripted_stats.py: D-S/D-T/D-L and the repro gate."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_s1_scripted_stats import classify_d, choose_best_supervised, reproduction_gate

RNG = np.random.default_rng(5)
N = 40


def _f1(base, noise=0.01):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def test_ds_supervised_ahead():
    l0 = _f1(0.817, noise=0.005)
    sup = _f1(0.840, noise=0.005)  # 2.3pt ahead
    letter, t = classify_d(sup, l0)
    assert letter == "D-S", t


def test_dt_within_band():
    l0 = _f1(0.817, noise=0.01)
    sup = _f1(0.820, noise=0.01)
    letter, t = classify_d(sup, l0)
    assert letter == "D-T", t


def test_dl_label_free_ahead():
    l0 = _f1(0.850, noise=0.005)
    sup = _f1(0.820, noise=0.005)  # 3pt behind
    letter, t = classify_d(sup, l0)
    assert letter == "D-L", t


def test_choose_best_supervised():
    e1 = _f1(0.80)
    e2 = _f1(0.85)
    name, arr = choose_best_supervised(e1, e2)
    assert name == "S-ens2"
    assert np.array_equal(arr, e2)


def test_reproduction_gate_pass():
    l0 = np.full(40, 0.817)
    assert reproduction_gate(l0, 0.817)


def test_reproduction_gate_fail():
    l0 = np.full(40, 0.80)
    assert not reproduction_gate(l0, 0.817)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
