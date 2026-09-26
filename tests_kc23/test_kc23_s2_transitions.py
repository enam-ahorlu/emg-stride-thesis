"""Tests for kc23_s2_transitions.py (rewritten 2026-09-25). The first version
was tested only against its own toy assumptions; these pin the behaviour that
matters on the real files: per-model separation, window END TIMES in seconds
(not sample indices), the ground-truth table as the only source of transition
times, and fail-closed handling of every missing input."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_s2_transitions as m
from kc23_s2_transitions import (label_windows, decision_delay, causal_vote, analyse, run, CLASS_OF,
                                 CONDITIONS, MODELS, CODE_DNS, CODE_UPS, CODE_WAK)

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
STEP = 0.125


def test_label_windows_zones():
    transitions = [(5.0, "LW", "SA")]
    df = label_windows(np.array([1.0, 4.5, 5.2, 8.0, 6.5]), transitions)
    assert list(df["zone"]) == ["steady_state", "transition_zone", "transition_zone", "steady_state", "neither"]
    assert df.loc[1, "transition_type"] == "LW_to_SA"


def test_label_windows_no_transitions_is_all_steady():
    df = label_windows(np.array([1.0, 2.0]), [])
    assert list(df["zone"]) == ["steady_state", "steady_state"]


def test_decision_delay_uses_new_class_and_window_end_time():
    # true change at 5.0 to SA (UPS). Windows end every 0.125 s from 5.0. Predicted UPS stably from 5.5.
    t = 5.0 + STEP * np.arange(10)
    y = np.array([CODE_WAK, CODE_WAK, CODE_WAK, CODE_WAK, CODE_UPS, CODE_UPS, CODE_UPS, CODE_UPS, CODE_UPS, CODE_UPS])
    d = decision_delay(t, y, [(5.0, "LW", "SA")])
    assert abs(d.loc[0, "delay_s"] - 0.5) < 1e-9          # end time of the first of 3 consecutive correct windows


def test_decision_delay_does_not_count_old_class_as_correct():
    # the window label after the change may still be the OLD class; predicting old must not read as instant success
    t = 5.0 + STEP * np.arange(6)
    y = np.full(6, CODE_WAK)
    d = decision_delay(t, y, [(5.0, "LW", "SA")])
    assert np.isnan(d.loc[0, "delay_s"])


def test_decision_delay_run_must_be_adjacent_windows():
    t = np.array([5.0, 5.125, 6.0, 6.125, 6.25])          # gap between 5.125 and 6.0
    y = np.array([CODE_UPS, CODE_UPS, CODE_UPS, CODE_UPS, CODE_UPS])
    d = decision_delay(t, y, [(5.0, "LW", "SA")])
    assert abs(d.loc[0, "delay_s"] - 1.0) < 1e-9          # the run restarts after the gap


def test_decision_delay_stops_at_next_transition():
    t = 5.0 + STEP * np.arange(40)
    y = np.full(40, CODE_WAK)
    y[30:] = CODE_UPS                                       # reaches UPS only after the second transition at 6.0
    d = decision_delay(t, y, [(5.0, "LW", "SA"), (6.0, "SA", "LW")])
    assert np.isnan(d.loc[0, "delay_s"])


def test_causal_vote_plurality_and_tie_to_most_recent():
    y = np.array([3, 3, 3, 2, 2, 2, 2])
    v = causal_vote(y, n=5)
    assert list(v[:3]) == [3, 3, 3]
    assert v[5] == 2 or v[5] == 3                          # 3,3,3,2,2 -> plurality 3
    assert v[6] == 2                                        # 3,3,2,2,2 -> 2
    tie = causal_vote(np.array([3, 2]), n=5)
    assert tie[1] == 2                                      # tie -> most recent


def _preds(subject=156, circuit=1, n=120, seed=0):
    """One subject/circuit, windows every 0.125 s, three models."""
    rng = np.random.default_rng(seed)
    t = 3.5 + STEP * np.arange(n)
    rows = []
    for model, acc in (("SVM", 0.9), ("RESNET_SE_CD", 0.8), ("soft", 0.95)):
        y_true = np.where(t < 8.0, CODE_WAK, CODE_UPS)
        y_pred = np.where(rng.random(n) < acc, y_true, CODE_DNS)
        for i in range(n):
            rows.append({"model": model, "subject": subject, "circuit": circuit, "t_end": float(t[i]),
                         "y_true": int(y_true[i]), "y_pred": int(y_pred[i]), "mode_raw": -1, "fs": 1000.0})
    return pd.DataFrame(rows)


def _table(subject=156, circuit=1):
    return pd.DataFrame([{"subject": subject, "circuit": circuit, "t_change_s": 8.0, "from": "LW", "to": "SA"}])


def test_analyse_is_per_model_not_pooled():
    df = _preds()
    res = {mod: analyse("causal100", mod, df[df["model"] == mod], _table())[0] for mod in MODELS}
    assert res["soft"]["steady_error"] < res["RESNET_SE_CD"]["steady_error"]     # would be equal if pooled
    for r in res.values():
        assert r["n_windows"] == 120


def test_analyse_uses_end_time_seconds_for_zones():
    df = _preds()
    mod = df[df["model"] == "soft"]
    r, z, d = analyse("causal100", "soft", mod, _table())
    # windows ending within 1 s of 8.0 s: 7.0..9.0 at 0.125 spacing = 17 windows
    assert r["zone_n"] == 17
    assert r["steady_n"] == int(((mod["t_end"] < 6.0) | (mod["t_end"] > 10.0)).sum())


def _write(root: Path, drop=None):
    pdir = root / "preds"; pdir.mkdir()
    for cond in CONDITIONS:
        _preds().to_csv(pdir / f"s2_predictions_{cond}.csv", index=False)
    tpath = root / "table.csv"
    _table().to_csv(tpath, index=False)
    return pdir, tpath


def test_run_end_to_end_writes_measures_and_verdict(tmp_path):
    pdir, tpath = _write(tmp_path)
    out = tmp_path / "out"
    assert run(out, pdir, tpath) == 0
    meas = pd.read_csv(out / "s2_measures.csv")
    assert len(meas) == len(CONDITIONS) * len(MODELS)
    v = (out / "S2_VERDICT.md").read_text()
    assert "Descriptive only" in v and "NOT computed" in v and not LETTER_RE.search(v)
    assert "causal 100-window buffer collapses" in v and v.index("Reading 1") < v.index("causal 100-window buffer collapses")


def test_verdict_leads_with_the_transductive_and_balanced25_rows_and_takes_reading_1_from_them(tmp_path):
    pdir, tpath = _write(tmp_path)
    out = tmp_path / "out"
    assert run(out, pdir, tpath) == 0
    v = (out / "S2_VERDICT.md").read_text()
    rows = [l.split("|")[1].strip() for l in v.splitlines() if l.startswith("| ") and "---" not in l][1:]
    assert rows[:6] == ["transductive"] * 3 + ["causal_balanced25"] * 3 and rows[6:] == ["causal100"] * 3
    reading = v[v.index("Reading 1"):v.index("## Finding")]
    assert "transductive" in reading and "causal_balanced25" in reading and "causal100" not in reading


def test_reading_1_states_disagreement_between_the_lead_conditions():
    base = {"model": "soft", "vote_delta_zone_error": -0.01, "vote_added_delay_n_paired": 5}
    meas = pd.DataFrame([
        {**base, "condition": "transductive", "vote_delta_steady_error": -0.02, "vote_added_delay_paired_median_s": 0.1},
        {**base, "condition": "causal_balanced25", "vote_delta_steady_error": +0.02, "vote_added_delay_paired_median_s": 0.1}])
    assert "disagree" in m._reading(meas)


def test_the_causal100_collapse_is_quantified_per_model_and_not_labelled_a_reading():
    meas = pd.DataFrame([{"condition": c, "model": mo, "steady_error": e}
                         for c, e in (("transductive", 0.15), ("causal100", 0.60)) for mo in MODELS])
    t = m._causal100_finding(meas)
    assert "0.150 transductive" in t and "0.600 causal-100" in t and "D-6b" in t and "not a reading" in t


def test_s2b_reading_is_included_only_when_its_file_exists(tmp_path):
    assert "NOT computed" in m._s2b_reading(tmp_path)
    (tmp_path / "s2b_reading.txt").write_text("Reading 2 (S2b): measured.", encoding="utf-8")
    assert m._s2b_reading(tmp_path) == "Reading 2 (S2b): measured."


def test_decision_delay_gap_scales_with_the_window_step():
    t = np.array([5.0, 5.2, 5.4, 5.6])                                # 400 ms windows, 0.2 s step
    y = np.full(4, CODE_UPS)
    assert abs(decision_delay(t, y, [(5.0, "LW", "SA")], max_gap_s=0.4).loc[0, "delay_s"] - 0.0) < 1e-9
    t2 = np.array([5.0, 5.3, 5.6, 5.9])                               # a 0.3 s spacing breaks the default 0.25 s rule
    assert np.isnan(decision_delay(t2, np.full(4, CODE_UPS), [(5.0, "LW", "SA")]).loc[0, "delay_s"])


@pytest.mark.parametrize("victim", ["preds/s2_predictions_transductive.csv", "preds/s2_predictions_causal100.csv",
                                    "preds/s2_predictions_causal_balanced25.csv", "table.csv"])
def test_run_each_input_missing_fails_closed(tmp_path, victim):
    pdir, tpath = _write(tmp_path)
    out = tmp_path / "out"
    (tmp_path / victim).unlink()
    assert run(out, pdir, tpath) != 0
    assert not (out / "s2_measures.csv").exists()
    assert not LETTER_RE.search((out / "S2_VERDICT.md").read_text())


def test_run_stale_measures_file_is_removed_on_failure(tmp_path):
    pdir, tpath = _write(tmp_path)
    out = tmp_path / "out"
    assert run(out, pdir, tpath) == 0
    (pdir / "s2_predictions_causal100.csv").unlink()
    assert run(out, pdir, tpath) != 0
    assert not (out / "s2_measures.csv").exists()


def test_run_missing_model_in_predictions_fails(tmp_path):
    pdir, tpath = _write(tmp_path)
    p = pdir / "s2_predictions_causal100.csv"
    d = pd.read_csv(p); d[d["model"] != "soft"].to_csv(p, index=False)
    assert run(tmp_path / "out", pdir, tpath) != 0


def test_run_subject_absent_from_table_fails(tmp_path):
    pdir, tpath = _write(tmp_path)
    pd.DataFrame([{"subject": 999, "circuit": 1, "t_change_s": 8.0, "from": "LW", "to": "SA"}]).to_csv(tpath, index=False)
    assert run(tmp_path / "out", pdir, tpath) != 0


def test_run_no_preds_argument_no_placeholder_exit_zero(tmp_path):
    # the old script wrote "Awaiting real per-window predictions" and returned 0 when the file was absent
    assert run(tmp_path / "out", tmp_path / "nowhere", tmp_path / "no_table.csv") != 0
