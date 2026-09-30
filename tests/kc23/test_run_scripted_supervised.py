"""Synthetic tests for src/run_scripted_supervised.py: the 3-check reproduction
gate (rewritten 2026-09-24 -- see src/kc23_s1_scripted_stats.py's docstring for
why: SVM decision route, SVM_PROBA and soft are now three separate checks,
not one SVM check plus an "ensemble" figure that used to be compared with
itself when its file was missing) and balanced_buffer_indices (the
K-per-movement buffer selection, unchanged)."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from run_scripted_supervised import (reproduction_gate, balanced_buffer_indices,
                                     PUBLISHED_SVM, PUBLISHED_SVM_PROBA, PUBLISHED_SOFT)


def test_gate_pass():
    letter, d = reproduction_gate(PUBLISHED_SVM, PUBLISHED_SVM_PROBA, PUBLISHED_SOFT)
    assert letter == "PASS", d


def test_gate_fail_svm_drift():
    letter, d = reproduction_gate(0.70, PUBLISHED_SVM_PROBA, PUBLISHED_SOFT)  # SVM way off
    assert letter == "FAIL", d
    assert not d["svm_ok"]


def test_gate_fail_svm_proba_drift():
    letter, d = reproduction_gate(PUBLISHED_SVM, 0.70, PUBLISHED_SOFT)
    assert letter == "FAIL", d
    assert not d["svm_proba_ok"]


def test_gate_fail_soft_outside_band():
    letter, d = reproduction_gate(PUBLISHED_SVM, PUBLISHED_SVM_PROBA, 0.75)  # soft way off
    assert letter == "FAIL", d
    assert not d["soft_ok"]


def test_buffer_indices_first_k_per_movement():
    # 2 movements, times increasing, K=2
    tvals = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    movement = np.array(["WAK", "WAK", "WAK", "WAK", "UPS", "UPS", "UPS", "UPS"])
    idx = balanced_buffer_indices(tvals, movement, K=2)
    assert set(idx) == {0, 1, 4, 5}, idx  # first 2 (by time) of each movement


def test_buffer_indices_respects_time_order_not_row_order():
    tvals = np.array([3, 1, 2, 0])  # row order != time order
    movement = np.array(["WAK", "WAK", "WAK", "WAK"])
    idx = balanced_buffer_indices(tvals, movement, K=2)
    # the two EARLIEST by time are rows 3 (t=0) and 1 (t=1)
    assert set(idx) == {3, 1}, idx


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
