"""Synthetic tests for kc23_s2_transitions.py: find_transition_times,
label_windows, decision_delay, error_by_zone."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_s2_transitions import find_transition_times, label_windows, decision_delay, error_by_zone

MODE_LW, MODE_SA = 1, 4
FS = 100.0


def test_find_transition_times_one_clean_transition():
    mode = np.array([MODE_LW] * 200 + [MODE_SA] * 200)
    trans = find_transition_times(mode, FS)
    assert len(trans) == 1
    t, frm, to = trans[0]
    assert frm == "LW" and to == "SA"
    assert abs(t - 2.0) < 1e-6  # sample 200 at 100Hz = 2.0s


def test_label_windows_zones():
    transitions = [(5.0, "LW", "SA")]
    win_end_times = np.array([1.0, 4.5, 5.2, 8.0])  # far, zone, zone, far(steady)
    df = label_windows(win_end_times, transitions)
    assert df.loc[0, "zone"] == "steady_state"
    assert df.loc[1, "zone"] == "transition_zone"
    assert df.loc[2, "zone"] == "transition_zone"
    assert df.loc[3, "zone"] == "steady_state"


def test_decision_delay_finds_first_stable_window():
    transitions = [(5.0, "LW", "SA")]
    win_end_times = np.array([4.9, 5.1, 5.3, 5.5, 5.7, 5.9, 6.1])
    y_true = np.array([0, 1, 1, 1, 1, 1, 1])
    # predictions wrong right after the transition, then correct for 3 in a row starting index 3
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1])
    d = decision_delay(win_end_times, y_true, y_pred, transitions, win_s=3)
    assert len(d) == 1
    assert abs(d.loc[0, "delay_s"] - (5.5 - 5.0)) < 1e-6


def test_error_by_zone_and_critical():
    zones = pd.DataFrame({
        "zone": ["steady_state", "steady_state", "transition_zone", "transition_zone"],
        "transition_type": [None, None, "SD_to_LW", "SD_to_LW"],
    })
    y_true = np.array([0, 0, 2, 2])  # 2 = DNS-ish code standing in for SD
    y_pred = np.array([0, 1, 3, 2])  # one steady-state error, one transition-zone (critical) error
    out = error_by_zone(zones, y_true, y_pred)
    assert abs(out["steady_state_error"] - 0.5) < 1e-6
    assert abs(out["transition_zone_error"] - 0.5) < 1e-6
    assert abs(out["dns_to_wak_critical_error_rate"] - 0.5) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
