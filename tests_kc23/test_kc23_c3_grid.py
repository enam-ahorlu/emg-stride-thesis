"""KC-C3 SVM-X grid, conformance fix of 26 September 2026: gamma is the plan's MULTIPLES of the fitted `scale` value
(48 cells), not six absolute numbers plus 'scale' (56 cells). The default grid path must be untouched."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import train_classical_loso as t

ENV = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1")


def test_extended_grid_is_48_multiplicative_cells():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 72))
    grid, scale, mults = t.svm_extended_grid(X, [])
    assert len(grid["clf__C"]) * len(grid["clf__gamma"]) == 48
    assert grid["clf__C"] == [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    assert mults == [0.01, 0.1, 0.3, 1, 3, 10]
    assert scale == pytest.approx(1 / 72, rel=0.05)                  # standardized features: scale ~ 1/n_features
    assert grid["clf__gamma"] == pytest.approx([m * scale for m in mults])
    assert "scale" not in grid["clf__gamma"]                         # 'scale' is the multiple 1, not an extra cell


def test_the_old_absolute_gamma_grid_is_gone():
    assert not hasattr(t, "SVM_GRID_EXTENDED")


def test_scale_value_is_measured_after_the_pipelines_scaler():
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(1)
    X = rng.normal(5, 3, (400, 10))                                  # raw features far from unit variance
    _, scale_raw, _ = t.svm_extended_grid(X, [])
    _, scale_scaled, _ = t.svm_extended_grid(X, [("scaler", StandardScaler())])
    assert scale_scaled == pytest.approx(1 / 10, rel=0.02) and scale_raw < scale_scaled / 5


def test_axis_overrides_for_the_edge_rule():
    X = np.random.default_rng(2).normal(0, 1, (100, 8))
    grid, _, mults = t.svm_extended_grid(X, [], c_values=[0.001, 0.003, 0.01], mult_values=[0.001, 0.003, 0.01])
    assert grid["clf__C"] == [0.001, 0.003, 0.01] and mults == [0.001, 0.003, 0.01]
    assert t._csv_floats("0.1, 1,10") == [0.1, 1.0, 10.0] and t._csv_floats(None) is None


def _tiny(tmp: Path, n_sub=8, per_class=25, n_feat=12):
    rng = np.random.default_rng(0)
    rows, X = [], []
    for s in range(1, n_sub + 1):
        off = rng.normal(0, 0.5, n_feat)
        for ci, lab in enumerate(["DNS", "STDUP", "UPS", "WAK"]):
            for _ in range(per_class):
                rows.append({"subject": s, "movement": lab})
                X.append(off + ci * 0.8 + rng.normal(0, 1, n_feat))
    np.savez(tmp / "f.npz", X=np.array(X))
    pd.DataFrame(rows).to_csv(tmp / "m.csv", index=False)


def _run(script: Path, tmp: Path, out: str, *extra):
    cmd = [sys.executable, str(script), "--features", str(tmp / "f.npz"), "--meta", str(tmp / "m.csv"), "--models", "SVM",
           "--norm-mode", "per_subject", "--n-jobs", "1", "--out", str(tmp / out), *extra]
    r = subprocess.run(cmd, cwd=tmp, env=ENV, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return tmp / out


def test_extended_run_records_the_chosen_multiplier_in_a_sidecar_and_leaves_the_subjectwise_csv_schema_alone(tmp_path):
    _tiny(tmp_path)
    out = _run(REPO / "train_classical_loso.py", tmp_path, "ext", "--grid", "extended", "--search", "grid")
    sw = next(out.glob("*SVM_nested_loso_subjectwise.csv"))
    d = pd.read_csv(sw)
    assert len(d) == 8 and "gamma_mult" not in " ".join(d.columns)
    side = pd.read_csv(out / "svm_extended_gamma.csv")
    assert len(side) == 8 and set(side["best_gamma_mult"].round(6)) <= {0.01, 0.1, 0.3, 1.0, 3.0, 10.0}
    assert (side["scale_value"] > 0).all()


def test_default_grid_path_is_unchanged_against_git_head(tmp_path):
    r = subprocess.run(["git", "show", "HEAD:06_Code/train_classical_loso.py"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout:
        r = subprocess.run(["git", "show", "HEAD:train_classical_loso.py"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout:
        pytest.skip("git HEAD copy unavailable")
    if "svm_extended_grid" in r.stdout:
        pytest.skip("HEAD already contains the change; nothing to compare against")
    old = tmp_path / "old_train_classical_loso.py"
    old.write_text(r.stdout, encoding="utf-8")
    _tiny(tmp_path)
    a = _run(old, tmp_path, "old_out")
    b = _run(REPO / "train_classical_loso.py", tmp_path, "new_out")
    fa = pd.read_csv(next(a.glob("*SVM_nested_loso_subjectwise.csv"))).drop(columns=["fit_time_sec", "infer_ms_per_window"])
    fb = pd.read_csv(next(b.glob("*SVM_nested_loso_subjectwise.csv"))).drop(columns=["fit_time_sec", "infer_ms_per_window"])
    pd.testing.assert_frame_equal(fa, fb)                            # per-subject F1 and best params, exactly
    assert not (b / "svm_extended_gamma.csv").exists()
