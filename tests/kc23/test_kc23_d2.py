"""Synthetic tests for src/kc23_d2_reliance_stats.py: reduction_factor and O-R/O-T/O-M."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from kc23_d2_reliance_stats import reduction_factor, classify_o


def test_reduction_factor_basic():
    r1 = np.array([10.0, 12.0, 8.0])
    r_aug = np.array([2.0, 3.0, 2.5])
    f = reduction_factor(r1, r_aug)
    assert abs(f - (10.0 / (2.5))) < 1e-6 or f > 3.0  # roughly 4x


def test_or_supported():
    letter = classify_o(occlusion_factor_gainjitter=4.0, permutation_factor_chandrop=2.5)
    assert letter == "O-R"


def test_ot_trained_in_robustness():
    letter = classify_o(occlusion_factor_gainjitter=1.2, permutation_factor_chandrop=1.1)
    assert letter == "O-T"


def test_om_mixed():
    letter = classify_o(occlusion_factor_gainjitter=2.0, permutation_factor_chandrop=1.0)
    assert letter == "O-M"


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
