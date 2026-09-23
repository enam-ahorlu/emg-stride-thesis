"""Synthetic tests for kc23_d5_replication_stats.py: E-R/E-N."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_d5_replication_stats import classify_er


def test_er_strong_agreement():
    diffs = np.array([1, 2, 3, 1, 0.5, 2, 1, -0.2, 1.5, 0.8])  # 9/10 positive
    letter, d = classify_er(diffs, expected_sign=1)
    assert letter == "E-R", d


def test_en_weak_agreement():
    diffs = np.array([1, -2, -3, 1, -0.5, -2, 1, -0.2, -1.5, 0.8])  # only 4/10 positive
    letter, d = classify_er(diffs, expected_sign=1)
    assert letter == "E-N", d


def test_en_wrong_direction_mean():
    diffs = np.array([-1, -2, -3, -1, -0.5, -2, -1, -0.2, -1.5, -0.8])  # mean negative, expected positive
    letter, d = classify_er(diffs, expected_sign=1)
    assert letter == "E-N", d


def test_er_negative_expected():
    diffs = np.array([-1, -2, -3, -1, -0.5, -2, -1, 0.2, -1.5, -0.8])  # 9/10 negative, expected negative
    letter, d = classify_er(diffs, expected_sign=-1)
    assert letter == "E-R", d


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
