"""KC-C3 edge-rule rerun in the gate: both runs are reported and the letters are recomputed with the rerun."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from kc23_c3_tuning_stats import run
from test_kc23_c3 import _make_tree, _write, _write_ensemble, N

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")


def _edge_rerun(root: Path, norm: str, mean: float, seed=11, c_choice=100.0):
    rng = np.random.default_rng(seed)
    d = root / f"results/kc23_c3_svm_{norm}_edge"
    _write(d, mean, rng, [str({"clf__C": c_choice, "clf__gamma": 0.01})] * N)
    pd.DataFrame({"heldout_subject": range(1, N + 1), "scale_value": 0.0139, "best_gamma": 0.0139, "best_gamma_mult": 1.0,
                  "c_grid": "0.01;0.03;0.1;0.3;1;3;10;30;100;300", "gamma_mult_grid": "0.01;0.1;0.3;1;3;10"}).to_csv(
        d / "svm_extended_gamma.csv", index=False)
    return d


def test_without_a_rerun_the_verdict_has_no_edge_rerun_section(tmp_path):
    _make_tree(tmp_path)
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    assert run(out, root=tmp_path) == 0
    assert "Edge-rule rerun" not in (out / "C3_VERDICT.md").read_text()


def test_both_runs_are_reported_and_the_letters_are_recomputed_with_the_rerun(tmp_path):
    _make_tree(tmp_path, resnet_mean=0.84, classical_mean=0.78, c_choice=30.0)      # the base run sits on the C edge
    _edge_rerun(tmp_path, "per_subject", 0.782)                                     # the rerun barely moves it
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    assert run(out, root=tmp_path) == 0
    v = (out / "C3_VERDICT.md").read_text()
    assert "Edge-rule rerun (both runs reported)" in v and "They match the base letters" in v
    assert "| per_subject |" in v and "not run" in v


def test_a_rerun_that_lands_on_an_escalating_letter_escalates_even_when_the_base_run_does_not(tmp_path):
    _make_tree(tmp_path, resnet_mean=0.84, classical_mean=0.78, c_choice=30.0)      # base: P1
    _edge_rerun(tmp_path, "per_subject", 0.835)                                     # rerun SVM-X within 1 pt of ResNet-SE+CD: P3
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    assert run(out, root=tmp_path) == 20
    v = (out / "C3_VERDICT.md").read_text()
    assert "**Outcomes: P1, N1, E1**" in v and "P1 -> P3" in v and "either escalating escalates" in v


def test_a_half_finished_rerun_is_an_error_not_something_to_ignore(tmp_path):
    _make_tree(tmp_path)
    d = _edge_rerun(tmp_path, "global", 0.75)
    f = next(d.glob("*_subjectwise.csv"))
    pd.read_csv(f).iloc[:25].to_csv(f, index=False)
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    assert run(out, root=tmp_path) == 20 and not LETTER_RE.search((out / "C3_VERDICT.md").read_text())
