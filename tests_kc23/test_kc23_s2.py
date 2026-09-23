"""Synthetic tests for kc23_s2_f0_feasibility.py: count_transitions_one_trial
and F-OK/F-MARGINAL/F-X."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_s2_f0_feasibility import count_transitions_one_trial, classify_f0

MODE_SIT, MODE_LW, MODE_RA, MODE_RD, MODE_SA, MODE_SD, MODE_STAND = 0, 1, 2, 3, 4, 5, 6


def _block(mode, n):
    return [mode] * n


def test_clean_transition_wak_to_ups():
    seq = _block(MODE_LW, 100) + _block(MODE_SA, 100)
    mode = np.array(seq)
    c = count_transitions_one_trial(mode)
    assert c[("LW", "SA")] == 1
    assert sum(c.values()) == 1


def test_ramp_voids_transition():
    # LW -> ramp ascent (dropped) -> SA: no clean transition, since a dropped
    # mode sits between the two retained runs.
    seq = _block(MODE_LW, 100) + _block(MODE_RA, 50) + _block(MODE_SA, 100)
    mode = np.array(seq)
    c = count_transitions_one_trial(mode)
    assert sum(c.values()) == 0, c


def test_multiple_transitions_counted():
    seq = (_block(MODE_LW, 50) + _block(MODE_SA, 50) + _block(MODE_LW, 50)
          + _block(MODE_SD, 50) + _block(MODE_LW, 50))
    mode = np.array(seq)
    c = count_transitions_one_trial(mode)
    assert c[("LW", "SA")] == 1
    assert c[("SA", "LW")] == 1
    assert c[("LW", "SD")] == 1
    assert c[("SD", "LW")] == 1


def _counts_df(per_subject_counts):
    rows = []
    for sid, counts in enumerate(per_subject_counts, start=1):
        rows.append({"subject": sid, "LW_to_SA": counts[0], "SA_to_LW": counts[1],
                    "LW_to_SD": counts[2], "SD_to_LW": counts[3]})
    return pd.DataFrame(rows)


def test_f_ok():
    df = _counts_df([[6, 6, 6, 6]] * 8 + [[1, 1, 1, 1]] * 2)  # 8/10 subjects have >=5 of each
    letter, d = classify_f0(df)
    assert letter == "F-OK", d


def test_f_marginal():
    df = _counts_df([[6, 6, 6, 6]] * 5 + [[1, 1, 1, 1]] * 5)  # only 5/10 subjects qualify
    letter, d = classify_f0(df)
    assert letter == "F-MARGINAL", d


def test_f_x_structurally_absent():
    df = _counts_df([[6, 6, 0, 0]] * 10)  # LW_to_SD/SD_to_LW always zero -- structurally absent
    letter, d = classify_f0(df)
    assert letter == "F-X", d


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
