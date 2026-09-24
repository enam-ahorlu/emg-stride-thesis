"""Tests for kc23_c3_tuning_stats.py: P1/P2/P3, N1/N2, E1/E2 (synthetic), plus
run()'s real-file discovery (fixed 2026-09-24: it originally expected
resnet_se_cd_persubj_subjectwise.csv / {fam}_persubj_subjectwise.csv /
ensemble_svmx_subjectwise.csv inside its own --out, none of which anything
writes there)."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c3_tuning_stats import (classify_p, classify_n, classify_e, run, load_ensemble_column,
                                  FAMILY_DIRS, RESNET_CD_DIR, ENSEMBLE_COLUMN)
from kc23_stats_common import read_subjectwise, require_complete

REPO_ROOT = Path(__file__).resolve().parent.parent
BEFORE_PERSUBJ = REPO_ROOT / "results_kc23_c3_before_persubj"
BEFORE_GLOBAL = REPO_ROOT / "results_kc23_c3_before_global"
AFTER_PERSUBJ = REPO_ROOT / "results_kc23_c3_after_persubj"

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
    # correctly parses train_classical_loso.py's REAL column layout
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


def _make_family_tree(root: Path, resnet_mean=0.79, classical_mean=0.78, seed=0):
    rng = np.random.default_rng(seed)

    def write(d: Path, mean):
        d.mkdir(parents=True, exist_ok=True)
        f1 = np.clip(mean + rng.normal(0, 0.01, 40), 0.05, 0.99)
        pd.DataFrame({"subject": range(1, 41), "f1_macro": f1}).to_csv(
            d / "x__MODEL_nested_loso_subjectwise.csv", index=False)
        return f1

    write(root / RESNET_CD_DIR, resnet_mean)
    for fam, pattern in FAMILY_DIRS.items():
        write(root / pattern.format(norm="per_subject"), classical_mean)
        write(root / pattern.format(norm="global"), classical_mean - 0.03)  # persubj beats global
    return root


def _write_ensemble(out_dir: Path, ens_mean=0.858):
    rng = np.random.default_rng(1)
    f1 = np.clip(ens_mean + rng.normal(0, 0.005, 40), 0.05, 0.99)
    df = pd.DataFrame({"subject": range(1, 41), ENSEMBLE_COLUMN: f1, "OTHER [hard]": f1 * 0.9})
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ensemble_v2_subjectwise.csv", index=False)


def test_run_fails_closed_when_resnet_reference_missing(tmp_path):
    out = tmp_path / "results_kc23_c3_ensemble"
    rc = run(out, root=tmp_path)
    assert rc == 20
    assert "FAIL" in (out / "C3_VERDICT.md").read_text()


def test_run_fails_closed_when_ensemble_file_missing(tmp_path):
    _make_family_tree(tmp_path)
    out = tmp_path / "results_kc23_c3_ensemble"
    rc = run(out, root=tmp_path)
    assert rc == 20
    assert "FAIL" in (out / "C3_VERDICT.md").read_text()


def test_run_end_to_end_pass(tmp_path):
    _make_family_tree(tmp_path, resnet_mean=0.84, classical_mean=0.78)
    out = tmp_path / "results_kc23_c3_ensemble"
    _write_ensemble(out, ens_mean=0.858)
    rc = run(out, root=tmp_path)
    assert rc == 0
    verdict = (out / "C3_VERDICT.md").read_text()
    assert "P1" in verdict and "N1" in verdict and "E1" in verdict


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
