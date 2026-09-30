"""Synthetic tests for src/kc23_s2_transition_table.py: direct-change detection
on the raw per-sample Mode signal (not a time-gap heuristic), run-length
segments, and the window-agreement validator."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from kc23_s2_transition_table import find_transitions_one_circuit, label_runs, validate_against_windows


def test_direct_clean_transition_counted():
    labels = np.array(["LW"] * 5 + ["SA"] * 5)
    t = find_transitions_one_circuit(labels, fs=10.0)
    assert len(t) == 1
    t_s, a, b = t[0]
    assert (a, b) == ("LW", "SA")
    assert abs(t_s - 0.5) < 1e-9  # sample index 5 / fs 10


def test_change_through_dropped_mode_excluded():
    # LW -> OTHER (ramp) -> SA: neither side of either sample-to-sample change
    # is a same-pair-of-retained-modes direct change
    labels = np.array(["LW"] * 5 + ["OTHER"] * 3 + ["SA"] * 5)
    t = find_transitions_one_circuit(labels, fs=10.0)
    assert t == []


def test_change_through_sts_excluded():
    labels = np.array(["LW"] * 5 + ["STS"] * 3 + ["SD"] * 5)
    t = find_transitions_one_circuit(labels, fs=10.0)
    assert t == []


def test_multiple_direct_transitions():
    labels = np.array(["LW"] * 3 + ["SA"] * 3 + ["LW"] * 3 + ["SD"] * 3)
    t = find_transitions_one_circuit(labels, fs=10.0)
    pairs = [(a, b) for _, a, b in t]
    assert pairs == [("LW", "SA"), ("SA", "LW"), ("LW", "SD")]


def test_same_retained_mode_no_transition():
    labels = np.array(["LW"] * 10)
    assert find_transitions_one_circuit(labels, fs=10.0) == []


def test_label_runs_basic():
    labels = np.array(["LW"] * 4 + ["OTHER"] * 2 + ["SA"] * 3)
    runs = label_runs(labels, fs=10.0)
    assert runs == [(0.0, 0.4, "LW"), (0.4, 0.6, "OTHER"), (0.6, 0.9, "SA")]


def test_label_runs_single_run():
    labels = np.array(["LW"] * 5)
    assert label_runs(labels, fs=10.0) == [(0.0, 0.5, "LW")]


def test_label_runs_empty():
    assert label_runs(np.array([]), fs=10.0) == []


def _runs_df(rows):
    return pd.DataFrame(rows, columns=["subject", "circuit", "start_s", "end_s", "code"])


def test_validate_agrees_when_window_matches_segment():
    runs = _runs_df([(1, 1, 0.0, 10.0, "LW")])
    windows = pd.DataFrame([{"subject": 1, "circuit": 1, "movement": "WAK", "fs": 10.0,
                            "win_samples": 20, "t_start_circuit": 10}])  # 1.0s to 3.0s, inside [0,10)
    disagree = validate_against_windows(runs, windows)
    assert disagree.empty


def test_validate_flags_disagreement():
    runs = _runs_df([(1, 1, 0.0, 10.0, "SA")])  # segment is UPS
    windows = pd.DataFrame([{"subject": 1, "circuit": 1, "movement": "WAK", "fs": 10.0,
                            "win_samples": 20, "t_start_circuit": 10}])
    disagree = validate_against_windows(runs, windows)
    assert len(disagree) == 1
    assert disagree.iloc[0]["expected_movement"] == "UPS"


def test_validate_skips_stdup_windows():
    runs = _runs_df([(1, 1, 0.0, 10.0, "OTHER")])
    windows = pd.DataFrame([{"subject": 1, "circuit": 1, "movement": "STDUP", "fs": 10.0,
                            "win_samples": 20, "t_start_circuit": 10}])
    disagree = validate_against_windows(runs, windows)
    assert disagree.empty  # STDUP has no raw-mode segment to check against


def test_validate_skips_window_spanning_a_boundary():
    runs = _runs_df([(1, 1, 0.0, 2.0, "LW"), (1, 1, 2.0, 4.0, "SA")])
    windows = pd.DataFrame([{"subject": 1, "circuit": 1, "movement": "WAK", "fs": 10.0,
                            "win_samples": 30, "t_start_circuit": 10}])  # 1.0s to 4.0s -- crosses the boundary at 2.0
    disagree = validate_against_windows(runs, windows)
    assert disagree.empty  # not "wholly inside" one segment -- skipped, not counted either way
