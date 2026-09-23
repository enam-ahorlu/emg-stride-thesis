"""Synthetic tests for kc23_c4_feature_stats.py: F-A/F-B/F-C."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c4_feature_stats import classify_set, classify_fa

RNG = np.random.default_rng(2)
N = 40


def _f1(base, noise=0.01):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def test_fa_extends():
    freq72 = _f1(0.777, noise=0.005)
    results = {}
    for feat in ["tdpsd54", "rich126"]:
        persubj = _f1(0.780, noise=0.005)  # within 1pt of freq72
        glob = _f1(0.720, noise=0.005)     # clear norm gain
        results[feat] = classify_set(persubj, glob, freq72)
    letter = classify_fa(results)
    assert letter == "F-A", results


def test_fb_moderate_gain():
    freq72 = _f1(0.777, noise=0.005)
    results = {}
    results["tdpsd54"] = classify_set(_f1(0.790, noise=0.005), _f1(0.72, noise=0.005), freq72)  # ~1.3pt gain
    results["rich126"] = classify_set(_f1(0.780, noise=0.005), _f1(0.72, noise=0.005), freq72)
    letter = classify_fa(results)
    assert letter == "F-B", results


def test_fc_large_gain():
    freq72 = _f1(0.777, noise=0.005)
    results = {}
    results["tdpsd54"] = classify_set(_f1(0.810, noise=0.005), _f1(0.72, noise=0.005), freq72)  # >2pt gain
    results["rich126"] = classify_set(_f1(0.780, noise=0.005), _f1(0.72, noise=0.005), freq72)
    letter = classify_fa(results)
    assert letter == "F-C", results


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
