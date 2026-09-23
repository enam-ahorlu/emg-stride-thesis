"""Synthetic tests for kc23_c6_ladder_stats.py: R1/R2."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c6_ladder_stats import classify_r

RNG = np.random.default_rng(4)
N = 10


def test_r1_replicates():
    f1_rung3 = np.full(N, 0.75) + RNG.normal(0, 0.01, N)
    f1_4lw = f1_rung3 - 0.05  # rung3 beats 4lw for all 10
    letter, d = classify_r(f1_rung3, f1_4lw, probe_rung0=0.9, probe_rung1=0.6, probe_rung3=0.3)
    assert letter == "R1", d


def test_r2_few_wins():
    f1_rung3 = np.array([0.7] * 3 + [0.6] * 7)
    f1_4lw = np.array([0.65] * 3 + [0.65] * 7)  # rung3 beats 4lw only 3/10
    letter, d = classify_r(f1_rung3, f1_4lw, probe_rung0=0.9, probe_rung1=0.6, probe_rung3=0.3)
    assert letter == "R2", d


def test_r2_centering_not_dominant():
    f1_rung3 = np.full(N, 0.75) + RNG.normal(0, 0.01, N)
    f1_4lw = f1_rung3 - 0.05  # beats 4lw 10/10
    # centering only removes a small fraction of the total reduction
    letter, d = classify_r(f1_rung3, f1_4lw, probe_rung0=0.9, probe_rung1=0.85, probe_rung3=0.3)
    assert letter == "R2", d


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
