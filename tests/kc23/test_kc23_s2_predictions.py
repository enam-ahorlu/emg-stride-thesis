"""Synthetic tests for src/kc23_s2_predictions.py: buffer_positions() for each
S2.3 condition, and load_common()'s row-alignment check."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from kc23_s2_predictions import buffer_positions, load_common, CONDITIONS


def test_transductive_has_no_buffer():
    assert buffer_positions("transductive", np.arange(10), None, None, None) is None


def test_causal100_takes_first_100_by_time():
    t_end = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    buf = buffer_positions("causal100", np.arange(5), None, t_end, None)
    # fewer than 100 windows here -- all of them, in time order
    assert list(buf) == [1, 3, 2, 4, 0]


def test_causal_balanced25_uses_movement_code():
    t_end = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=float)
    movement = np.array([1, 1, 1, 1, 4, 4, 4, 4])  # LW then SA raw codes
    buf = buffer_positions("causal_balanced25", np.arange(8), None, t_end, movement)
    # K=25 with only 4 windows/movement available takes all 4 of each (K just caps, never pads)
    assert set(buf) == {0, 1, 2, 3, 4, 5, 6, 7}


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        buffer_positions("bogus", np.arange(5), None, np.zeros(5), np.zeros(5))


def test_conditions_list_matches_plan():
    assert CONDITIONS == ["transductive", "causal100", "causal_balanced25"]


def test_load_common_raises_on_misaligned_inputs(tmp_path, monkeypatch):
    root = tmp_path
    (root / "data/features_out_ext").mkdir(parents=True)
    (root / "results/kc23_s2_adapter_circuitmeta").mkdir(parents=True)
    feat_meta = pd.DataFrame({"subject": [1, 2, 3], "movement": ["WAK", "UPS", "DNS"]})
    feat_meta.to_csv(root / "data/features_out_ext" /
                     "freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_meta.csv", index=False)
    np.savez(root / "data/features_out_ext" /
            "freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_ext.npz", X=np.zeros((3, 4)))
    cm_meta = pd.DataFrame({"subject": [1, 2], "movement": ["WAK", "UPS"], "circuit": [1, 1],
                           "t_start_circuit": [0, 100], "win_samples": [250, 250], "fs": [1000.0, 1000.0]})
    cm_meta.to_csv(root / "results/kc23_s2_adapter_circuitmeta" /
                  "windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_kc23s2_meta.csv", index=False)
    with pytest.raises(ValueError, match="row-aligned"):
        load_common(root)
