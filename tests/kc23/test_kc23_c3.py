"""Tests for src/kc23_c3_tuning_stats.py: P1/P2/P3, N1/N2, E1/E2 (synthetic), plus
run()'s real-file discovery (fixed 2026-09-24: it originally expected
resnet_se_cd_persubj_subjectwise.csv / {fam}_persubj_subjectwise.csv /
ensemble_svmx_subjectwise.csv inside its own --out, none of which anything
writes there)."""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from kc23_c3_tuning_stats import (classify_p, classify_n, classify_e, run, load_ensemble_column,
                                  FAMILY_DIRS, RESNET_CD_DIR, ENSEMBLE_COLUMN)
from kc23_stats_common import read_subjectwise, require_complete

REPO_ROOT = Path(__file__).resolve().parents[2]
BEFORE_PERSUBJ = REPO_ROOT / "results/kc23_c3_before_persubj"
BEFORE_GLOBAL = REPO_ROOT / "results/kc23_c3_before_global"
AFTER_PERSUBJ = REPO_ROOT / "results/kc23_c3_after_persubj"

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
RNG = np.random.default_rng(1)
N = 40


def _f1(base, noise=0.02):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def test_p1_lead_stands():
    resnet = _f1(0.840)
    classical = _f1(0.780)  # within 1pt of 77.7
    letter, t = classify_p(resnet, classical)
    assert letter == "P1", t


def test_p2_lead_narrows():
    resnet = _f1(0.840, noise=0.01)
    classical = _f1(0.797, noise=0.01)  # 2pt gain over 77.7, lead still >1pt
    letter, t = classify_p(resnet, classical)
    assert letter == "P2", t


def test_p3_classical_catches_up():
    resnet = _f1(0.840, noise=0.01)
    classical = _f1(0.838, noise=0.01)  # within 1pt of resnet
    letter, t = classify_p(resnet, classical)
    assert letter == "P3", t


def test_p3_classical_beats_deep():
    resnet = _f1(0.820)
    classical = _f1(0.850)
    letter, t = classify_p(resnet, classical)
    assert letter == "P3", t


def test_n1_every_family_extends():
    fams = {f: (_f1(0.80, 0.01), _f1(0.74, 0.01)) for f in ["svmx", "rfx", "hgb", "knn"]}
    letter, rows = classify_n(fams)
    assert letter == "N1", rows


def test_n2_one_family_nonpositive():
    fams = {f: (_f1(0.80, 0.01), _f1(0.74, 0.01)) for f in ["svmx", "rfx", "hgb"]}
    fams["knn"] = (_f1(0.70, 0.01), _f1(0.71, 0.01))  # negative gain
    letter, rows = classify_n(fams)
    assert letter == "N2", rows


def test_e1_close_to_headline():
    ens = _f1(0.858, noise=0.001)
    letter, s = classify_e(ens)
    assert letter == "E1", s


def test_e2_far_from_headline():
    ens = _f1(0.870, noise=0.001)
    letter, s = classify_e(ens)
    assert letter == "E2", s


pytestmark_real = pytest.mark.skipif(not BEFORE_PERSUBJ.exists(), reason="real C3 fixture dirs not present")


@pytestmark_real
def test_real_fixture_reads_as_f1_array():
    # these are small inertness-proof smoke fixtures (2-3 subjects), not full
    # 40-subject runs -- the point is confirming read_subjectwise/model_token
    # correctly parses src/train_classical_loso.py's REAL column layout
    # (model, heldout_subject, ..., f1_macro, ...), not the subject count.
    df = read_subjectwise(BEFORE_PERSUBJ, model_token="SVM")
    assert "f1_macro" in df.columns
    assert (df["model"] == "SVM").all()
    assert len(df) >= 1
    assert ((df["f1_macro"] >= 0) & (df["f1_macro"] <= 1)).all()


@pytestmark_real
def test_real_fixture_before_after_both_readable():
    for d in (BEFORE_PERSUBJ, BEFORE_GLOBAL, AFTER_PERSUBJ):
        df = read_subjectwise(d, model_token="RF")
        assert "f1_macro" in df.columns
        assert len(df) >= 1


def test_p_out_when_classical_gains_over_3_but_still_trails_the_deep_model():
    letter, t = classify_p(_f1(0.900, 0.005), _f1(0.820, 0.005))    # +4.3 over the SVM, lead 8 pt
    assert letter == "P-OUT", t


def test_p_out_when_the_best_classical_is_more_than_1_pt_below_the_published_svm():
    letter, t = classify_p(_f1(0.840, 0.005), _f1(0.740, 0.005))
    assert letter == "P-OUT", t


def test_n_out_when_every_gain_is_positive_but_one_is_not_significant():
    fams = {f: (_f1(0.80, 0.01), _f1(0.74, 0.01)) for f in ["svmx", "rfx", "hgb"]}
    p = np.array(_f1(0.75, 0.05))
    noise = np.random.default_rng(5).normal(0, 0.02, N)
    noise = noise - noise.mean() + 0.0005                             # mean gain +0.05 pt, signs mixed
    fams["knn"] = (p + noise, p)
    letter, rows = classify_n(fams)
    assert letter == "N-OUT", rows


def _write(d: Path, mean, rng, best_params=None, fit=100.0):
    d.mkdir(parents=True, exist_ok=True)
    f1 = np.clip(mean + rng.normal(0, 0.01, N), 0.05, 0.99)
    df = pd.DataFrame({"model": "M", "heldout_subject": range(1, N + 1), "f1_macro": f1,
                       "best_params": best_params or ["{'clf__C': 1}"] * N, "fit_time_sec": fit})
    df.to_csv(d / "x__MODEL_nested_loso_subjectwise.csv", index=False)


def _proba(rng, y, sharp):
    p = rng.dirichlet(np.ones(4), len(y)) * (1 - sharp)
    p[np.arange(len(y)), y] += sharp
    return p / p.sum(1, keepdims=True)


def _make_tree(root: Path, resnet_mean=0.84, classical_mean=0.78, c_choice=1.0, mult_choice=1.0, seed=0,
               c_grid="0.01;0.03;0.1;0.3;1;3;10;30", g_grid="0.01;0.1;0.3;1;3;10", skip_rf_global=False):
    rng = np.random.default_rng(seed)
    _write(root / RESNET_CD_DIR, resnet_mean, rng)
    for fam, pattern in FAMILY_DIRS.items():
        bp = [str({"clf__C": c_choice, "clf__gamma": 0.01})] * N if fam == "svmx" else None
        _write(root / pattern.format(norm="per_subject"), classical_mean, rng, bp)
        if skip_rf_global and fam == "rfx":
            continue
        _write(root / pattern.format(norm="global"), classical_mean - 0.03, rng, bp)
    for norm in ("per_subject", "global"):
        d = root / FAMILY_DIRS["svmx"].format(norm=norm)
        pd.DataFrame({"heldout_subject": range(1, N + 1), "scale_value": 0.0139, "best_gamma": 0.0139 * mult_choice,
                      "best_gamma_mult": mult_choice, "c_grid": c_grid, "gamma_mult_grid": g_grid}).to_csv(
            d / "svm_extended_gamma.csv", index=False)
    pub = root / "results/ensemble_v2" / "proba_aug_chandrop"
    pub.mkdir(parents=True)
    for s in range(1, N + 1):
        y = rng.integers(0, 4, 24)
        np.savez(pub / f"RESNET_SE_sub{s:02d}.npz", proba=_proba(rng, y, 0.3), y_true=y.astype(np.int32))
        for model in ("SVM", "RF", "HGB"):
            fam = {"SVM": "svm", "RF": "rf", "HGB": "hgb"}[model]
            pd_ = root / f"results/kc23_c3_{fam}_per_subject" / "proba"
            pd_.mkdir(parents=True, exist_ok=True)
            np.savez(pd_ / f"{model}_sub{s:02d}.npz", proba=_proba(rng, y, 0.25), y_true=y.astype(np.int32))
    return root


def _write_ensemble(out_dir: Path, ens_mean=0.858):
    rng = np.random.default_rng(1)
    f1 = np.clip(ens_mean + rng.normal(0, 0.005, 40), 0.05, 0.99)
    df = pd.DataFrame({"subject": range(1, 41), ENSEMBLE_COLUMN: f1, "OTHER [hard]": f1 * 0.9})
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ensemble_v2_subjectwise.csv", index=False)


def test_run_fails_closed_when_resnet_reference_missing(tmp_path):
    out = tmp_path / "results/kc23_c3_ensemble"
    rc = run(out, root=tmp_path)
    assert rc == 20
    v = (out / "C3_VERDICT.md").read_text()
    assert "NO OUTCOME COMPUTED" in v and "**" not in v   # a missing input is not a result: no letter line


def test_run_fails_closed_when_ensemble_file_missing(tmp_path):
    _make_tree(tmp_path)
    out = tmp_path / "results/kc23_c3_ensemble"
    rc = run(out, root=tmp_path)
    assert rc == 20
    v = (out / "C3_VERDICT.md").read_text()
    assert "NO OUTCOME COMPUTED" in v and "**" not in v


@pytest.mark.parametrize("victim", ["results/kc23_c3_svm_per_subject/svm_extended_gamma.csv",
                                    "results/kc23_c3_hgb_global",
                                    "results/kc23_c3_rf_per_subject/proba/RF_sub09.npz",
                                    "results/ensemble_v2/proba_aug_chandrop/RESNET_SE_sub40.npz"])
def test_run_fails_closed_for_every_missing_input(tmp_path, victim):
    _make_tree(tmp_path)
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    v = tmp_path / victim
    shutil.rmtree(v) if v.is_dir() else v.unlink()
    assert run(out, root=tmp_path) == 20
    assert not LETTER_RE.search((out / "C3_VERDICT.md").read_text())
    assert not (out / "C3_tests.csv").exists()


def test_run_end_to_end_pass_writes_the_plan_reportables(tmp_path):
    _make_tree(tmp_path, resnet_mean=0.84, classical_mean=0.78)
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out, ens_mean=0.858)
    rc = run(out, root=tmp_path)
    assert rc == 0
    verdict = (out / "C3_VERDICT.md").read_text()
    assert LETTER_RE.search(verdict) and "**Outcomes: P1, N1, E1**" in verdict
    assert "Edge-hit table" in verdict and "Fit-time totals" in verdict and "best new classical member" in verdict
    edge = pd.read_csv(out / "C3_edge_hits.csv")
    assert set(edge["axis"]) == {"C", "gamma multiplier"} and not edge["triggered"].any()
    assert (pd.read_csv(out / "C3_fit_times.csv")["fit_time_hours"] > 0).all()


def test_the_edge_rule_triggers_when_more_than_10_folds_sit_on_an_edge(tmp_path):
    _make_tree(tmp_path, c_choice=30.0)                              # every fold at the C upper edge
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    assert run(out, root=tmp_path) == 0
    edge = pd.read_csv(out / "C3_edge_hits.csv")
    assert bool(edge[(edge["axis"] == "C") & (edge["norm"] == "per_subject")]["triggered"].iloc[0])
    assert "EDGE RULE TRIGGERED" in (out / "C3_VERDICT.md").read_text()


def test_an_already_extended_axis_does_not_trigger_again(tmp_path):
    _make_tree(tmp_path, c_choice=100.0, c_grid="0.1;0.3;1;3;10;30;100;300")     # extended once already
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    assert run(out, root=tmp_path) == 0
    edge = pd.read_csv(out / "C3_edge_hits.csv")
    assert not edge["triggered"].any() and edge["extension_already_applied"].any()


def test_optional_rf_global_may_be_absent_but_a_required_family_may_not(tmp_path):
    _make_tree(tmp_path, skip_rf_global=True)
    shutil.rmtree(tmp_path / "results/kc23_c3_rf_global", ignore_errors=True)
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    assert run(out, root=tmp_path) == 0
    assert "RF-X global run absent" in (out / "C3_VERDICT.md").read_text() or "rfx" in (out / "C3_VERDICT.md").read_text()


def test_exit_20_on_an_escalating_letter(tmp_path):
    _make_tree(tmp_path, resnet_mean=0.80, classical_mean=0.798)      # classical within 1 pt of the deep model: P3
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    assert run(out, root=tmp_path) == 20
    assert "P3" in (out / "C3_VERDICT.md").read_text()


def test_exit_10_on_an_out_letter(tmp_path):
    _make_tree(tmp_path, resnet_mean=0.90, classical_mean=0.82)       # gains > 3 pt, still trails by 8: P-OUT
    out = tmp_path / "results/kc23_c3_ensemble"
    _write_ensemble(out)
    assert run(out, root=tmp_path) == 10


def test_load_ensemble_column_missing_column_raises(tmp_path):
    p = tmp_path / "ens.csv"
    pd.DataFrame({"subject": [1, 2], "OTHER [hard]": [0.8, 0.8]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_ensemble_column(p, ENSEMBLE_COLUMN)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
