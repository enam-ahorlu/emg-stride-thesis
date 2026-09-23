"""Synthetic tests for kc23_c3_tuning_stats.py: P1/P2/P3, N1/N2, E1/E2."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c3_tuning_stats import classify_p, classify_n, classify_e

RNG = np.random.default_rng(1)
N = 40


def _f1(base, noise=0.02):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def test_p1_lead_stands():
    resnet = _f1(0.840)
    classical = _f1(0.780)  # within 1pt of 77.7
    letter, t = classify_p(resnet, classical)
    assert letter == "P1", t


def test_p2_lead_narrows():
    resnet = _f1(0.840, noise=0.01)
    classical = _f1(0.797, noise=0.01)  # 2pt gain over 77.7, lead still >1pt
    letter, t = classify_p(resnet, classical)
    assert letter == "P2", t


def test_p3_classical_catches_up():
    resnet = _f1(0.840, noise=0.01)
    classical = _f1(0.838, noise=0.01)  # within 1pt of resnet
    letter, t = classify_p(resnet, classical)
    assert letter == "P3", t


def test_p3_classical_beats_deep():
    resnet = _f1(0.820)
    classical = _f1(0.850)
    letter, t = classify_p(resnet, classical)
    assert letter == "P3", t


def test_n1_every_family_extends():
    fams = {f: (_f1(0.80, 0.01), _f1(0.74, 0.01)) for f in ["svmx", "rfx", "hgb", "knn"]}
    letter, rows = classify_n(fams)
    assert letter == "N1", rows


def test_n2_one_family_nonpositive():
    fams = {f: (_f1(0.80, 0.01), _f1(0.74, 0.01)) for f in ["svmx", "rfx", "hgb"]}
    fams["knn"] = (_f1(0.70, 0.01), _f1(0.71, 0.01))  # negative gain
    letter, rows = classify_n(fams)
    assert letter == "N2", rows


def test_e1_close_to_headline():
    ens = _f1(0.858, noise=0.001)
    letter, s = classify_e(ens)
    assert letter == "E1", s


def test_e2_far_from_headline():
    ens = _f1(0.870, noise=0.001)
    letter, s = classify_e(ens)
    assert letter == "E2", s


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
