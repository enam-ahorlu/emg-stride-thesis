"""kc23_s2b_window_trade.py (S2b) and the additive 400 ms path of kc23_s2_predictions.py."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_s2_transitions as s2
import kc23_s2b_window_trade as s2b
import kc23_s2_predictions as pred

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
CODE_WAK, CODE_UPS, CODE_DNS = 3, 2, 0
N_SUB = 3


def _preds(step, lag, err_rate=0.0, seed=0, subjects=range(1, N_SUB + 1)):
    """Windows every `step` s from 3.5 s; the true class changes WAK -> UPS at 8.0 s; the models keep the old class for `lag` s."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in subjects:
        t = np.arange(3.5, 14.0, step)
        y_true = np.where(t < 8.0, CODE_WAK, CODE_UPS)
        for model in s2.MODELS:
            y_pred = np.where((t >= 8.0) & (t < 8.0 + lag), CODE_WAK, y_true)
            flip = rng.random(len(t)) < err_rate
            y_pred = np.where(flip & (t < 6.0), CODE_DNS, y_pred)
            for i in range(len(t)):
                rows.append({"model": model, "subject": subject, "circuit": 1, "t_end": float(t[i]),
                             "y_true": int(y_true[i]), "y_pred": int(y_pred[i]), "mode_raw": -1, "fs": 1000.0})
    return pd.DataFrame(rows)


def _write(root: Path, err400=0.0, lag400=0.9):
    for name, step, lag, err in (("p250", 0.125, 0.4, 0.0), ("p400", 0.2, lag400, err400)):
        d = root / name
        d.mkdir()
        for cond in s2.CONDITIONS:
            _preds(step, lag, err).to_csv(d / f"s2_predictions_{cond}.csv", index=False)
    table = root / "table.csv"
    pd.DataFrame([{"subject": s, "circuit": 1, "t_change_s": 8.0, "from": "LW", "to": "SA"}
                  for s in range(1, N_SUB + 1)]).to_csv(table, index=False)
    return root / "p250", root / "p400", table


def test_the_trade_is_supported_when_400ms_adds_delay_for_no_accuracy_gain(tmp_path):
    p250, p400, table = _write(tmp_path)
    out = tmp_path / "out"
    assert s2b.run(out, p250, p400, table) == 0
    m = pd.read_csv(out / "s2b_measures.csv")
    assert len(m) == 9 and (m["n_paired"] == N_SUB).all()
    r = m[(m["condition"] == "transductive") & (m["model"] == "SVM")].iloc[0]
    assert r["added_delay_paired_median_ms"] > 100 and r["accuracy_gain_pp"] < 2
    text = (out / "s2b_reading.txt").read_text()
    assert "is supported by a lower-limb measurement" in text
    assert not LETTER_RE.search((out / "S2B_VERDICT.md").read_text())


def test_the_trade_is_reported_as_measured_when_400ms_buys_a_large_accuracy_gain(tmp_path):
    p250, p400, table = _write(tmp_path, err400=0.0)
    # make the 250 ms predictions worse in steady state, so 400 ms gains > 2 pt accuracy
    for cond in s2.CONDITIONS:
        f = p250 / f"s2_predictions_{cond}.csv"
        pd.concat([_preds(0.125, 0.4, err_rate=0.5, seed=3)]).to_csv(f, index=False)
    out = tmp_path / "out"
    assert s2b.run(out, p250, p400, table) == 0
    assert "reported as measured" in (out / "s2b_reading.txt").read_text()


def test_the_added_delay_is_paired_per_transition_and_uses_end_time(tmp_path):
    p250, p400, table = _write(tmp_path, lag400=0.9)
    out = tmp_path / "out"
    s2b.run(out, p250, p400, table)
    m = pd.read_csv(out / "s2b_measures.csv")
    r = m[(m["condition"] == "transductive") & (m["model"] == "soft")].iloc[0]
    # 250 ms: first new-class window ends at 8.5 (delay 0.5 at the 0.125 grid); 400 ms: first at 9.1 -> 1.1
    assert abs(r["delay_median_250_s"] - 0.5) < 0.13 and abs(r["delay_median_400_s"] - 1.1) < 0.21
    assert abs(r["added_delay_paired_median_ms"] - (r["delay_median_400_s"] - r["delay_median_250_s"]) * 1000) < 1e-6


def test_the_adjacency_allowance_is_twice_the_step():
    assert s2b.GAP_250_S == s2.MAX_GAP_S == 0.25 and s2b.GAP_400_S == 0.40


def test_the_s2_verdict_is_regenerated_with_reading_2_when_asked(tmp_path):
    p250, p400, table = _write(tmp_path)
    out, s2_out = tmp_path / "out", tmp_path / "s2"
    assert s2b.run(out, p250, p400, table, s2_out) == 0
    v = (s2_out / "S2_VERDICT.md").read_text()
    assert "Reading 2 (S2b, 400 ms against 250 ms; the locked SVM" in v and "NOT computed" not in v


@pytest.mark.parametrize("victim", ["p400/s2_predictions_transductive.csv", "p250/s2_predictions_causal100.csv", "table.csv"])
def test_each_missing_input_fails_closed_and_leaves_no_stale_output(tmp_path, victim):
    p250, p400, table = _write(tmp_path)
    out = tmp_path / "out"
    assert s2b.run(out, p250, p400, table) == 0
    (tmp_path / victim).unlink()
    assert s2b.run(out, p250, p400, table) == 2
    assert not (out / "s2b_measures.csv").exists() and not (out / "s2b_reading.txt").exists()
    v = (out / "S2B_VERDICT.md").read_text()
    assert "NO OUTCOME COMPUTED" in v and not LETTER_RE.search(v)


def test_different_subject_sets_fail_closed(tmp_path):
    p250, p400, table = _write(tmp_path)
    for cond in s2.CONDITIONS:
        _preds(0.2, 0.9, subjects=[1, 2]).to_csv(p400 / f"s2_predictions_{cond}.csv", index=False)
    assert s2b.run(tmp_path / "out", p250, p400, table) == 2


def test_the_400ms_prediction_spec_points_at_the_s2b_windows_and_the_default_is_the_250ms_job():
    assert pred.SPEC_250 == {"circuitmeta_dir": pred.CIRCUITMETA_DIR, "tag": pred.CIRCUITMETA_TAG,
                             "feat": pred.FEAT_ENABL3S, "meta_feat": pred.META_ENABL3S_FEAT}
    assert "w400" in pred.SPEC_400["tag"] and "s2b_adapter_400" in pred.SPEC_400["circuitmeta_dir"]
    assert "w400" in pred.SPEC_400["feat"] and "w400" in pred.SPEC_400["meta_feat"]
    import inspect
    assert inspect.signature(pred.load_common).parameters["spec"].default is None      # no argument: the 250 ms path
