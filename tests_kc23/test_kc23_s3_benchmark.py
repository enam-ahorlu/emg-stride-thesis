"""kc23_s3_benchmark.py: the active-only benchmark table and S3_VERDICT.md, on real-format fixtures (subjectwise csvs,
predictions_folds npy, proba npz, the published aonly ensemble csv). Fails closed on any missing cell."""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_s3_benchmark as b

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
N = 40


def _labels(rng, n=60):
    return rng.integers(0, 4, n).astype(np.int32)


def _preds(rng, y, dns_wak_rate):
    yp = y.copy()
    flip = rng.random(len(y)) < 0.25
    yp[flip] = rng.integers(0, 4, flip.sum())
    m = (y == b.DNS) & (rng.random(len(y)) < dns_wak_rate)
    yp[m] = b.WAK
    return yp


def _classical(root, dname, token, mean, rng, stem="feat", dns_wak=0.1):
    d = root / dname
    (d / "predictions_folds").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"model": token, "heldout_subject": range(1, N + 1), "f1_macro": mean + rng.normal(0, 0.01, N)}).to_csv(
        d / f"{stem}__{token}_nested_loso_subjectwise.csv", index=False)
    for s in range(1, N + 1):
        y = _labels(rng)
        np.save(d / "predictions_folds" / f"{stem}_{token}_sub{s:02d}_y_true.npy", y)
        np.save(d / "predictions_folds" / f"{stem}_{token}_sub{s:02d}_y_pred.npy", _preds(rng, y, dns_wak))


def _lda(root, dname, mean, rng):
    d = root / dname
    (d / "predictions_folds").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject": range(1, N + 1), "f1_macro": mean + rng.normal(0, 0.01, N)}).to_csv(d / "lda_subjectwise.csv", index=False)
    for s in range(1, N + 1):
        y = _labels(rng)
        np.save(d / "predictions_folds" / f"feat_LDA_sub{s:02d}_y_true.npy", y)
        np.save(d / "predictions_folds" / f"feat_LDA_sub{s:02d}_y_pred.npy", _preds(rng, y, 0.2))


def _cnn(root, dname, mean, rng, tag="TAG"):
    d = root / dname
    (d / "proba").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject": range(1, N + 1), "arch": "x", "f1_macro": mean + rng.normal(0, 0.01, N), "bal_acc": 0.5}).to_csv(
        d / "cnn_arch_subjectwise.csv", index=False)
    for s in range(1, N + 1):
        y = _labels(rng)
        p = rng.dirichlet(np.ones(4), len(y))
        p[np.arange(len(y)), y] += 1.0
        np.savez(d / "proba" / f"{tag}_sub{s:02d}.npz", proba=p / p.sum(1, keepdims=True), y_true=y)


def _ensemble(root, rng):
    d = root / "results_aonly_ensemble"
    d.mkdir(parents=True)
    rows = [{"subject": s, "member": m, "macro_f1": mean + rng.normal(0, 0.01), "crit_err": 0.05}
            for s in range(1, N + 1) for m, mean in (("SVM", 0.77), ("soft", 0.84))]
    pd.DataFrame(rows).to_csv(d / "aonly_ensemble_subjectwise.csv", index=False)


def build(root: Path, c3="**Outcomes: P1, N1, E1**", seed=0):
    rng = np.random.default_rng(seed)
    _classical(root, "results_aonly_global", "SVM", 0.70, rng); _classical(root, "results_aonly_global", "RF", 0.71, rng)
    _classical(root, "results_aonly_persubj", "SVM", 0.77, rng); _classical(root, "results_aonly_persubj", "RF", 0.76, rng)
    _cnn(root, "results_aonly_resnet_se_cd_persubj", 0.82, rng, "RESNET_SE_AONLY")
    _ensemble(root, rng)
    for norm, m in (("global", 0.60), ("per_subject", 0.68)):
        _lda(root, f"results_kc23_s3_lda_{norm}", m, rng)
    for arch, mg, mp in (("simple", 0.72, 0.74), ("resnet_se", 0.78, 0.81), ("resnet_se_cd", 0.79, None)):
        _cnn(root, f"results_kc23_s3_{arch}_global", mg, rng)
        if mp:
            _cnn(root, f"results_kc23_s3_{arch}_per_subject", mp, rng)
    # the rest-included benchmark for the hierarchy comparison
    _classical(root, "results_loso_freq_persubj", "SVM", 0.777, rng); _classical(root, "results_loso_freq_persubj", "RF", 0.773, rng)
    _lda(root, "results_lda_persubj", 0.687, rng)
    v = root / "results_kc23_c3_ensemble"
    v.mkdir()
    (v / "C3_VERDICT.md").write_text(f"# KC-C3 verdict\n\n{c3}\n", encoding="utf-8")
    return root


def test_builds_the_table_from_prior_and_new_cells(tmp_path):
    build(tmp_path)
    out = tmp_path / "out"
    assert b.run(out, tmp_path) == 0
    t = pd.read_csv(out / "benchmark_active_only.csv")
    assert len(t) == 13                                   # 7 families x 2 norms, minus the global ensemble
    assert set(t["provenance"]) == {"prior", "new"} and (t["n_subjects"] == N).all()
    assert t[(t["family"] == "SVM") & (t["normalization"] == "per_subject")]["source"].iloc[0] == "results_aonly_persubj"
    assert t[(t["family"] == "resnet_se_cd") & (t["normalization"] == "per_subject")]["source"].iloc[0] == "results_aonly_resnet_se_cd_persubj"
    assert t["dns_to_wak_subject_mean"].between(0, 1).all()
    v = (out / "S3_VERDICT.md").read_text()
    assert "Class hierarchy" in v and not LETTER_RE.search(v) and "LDA" in v


def test_dns_to_wak_rate_is_true_dns_windows_predicted_wak():
    y = np.array([0, 0, 0, 0, 3, 1]); p = np.array([3, 3, 0, 1, 3, 3])
    assert b.crit(y, p) == (2, 4)


def test_the_pooled_rate_is_computed_from_window_counts(tmp_path):
    build(tmp_path)
    assert b.run(tmp_path / "o", tmp_path) == 0
    r = pd.read_csv(tmp_path / "o" / "benchmark_active_only.csv")
    lda = r[(r["family"] == "LDA") & (r["normalization"] == "global")].iloc[0]
    assert 0.1 < lda["dns_to_wak_pooled"] < 0.45


def test_svmx_and_hgb_are_required_only_when_c3_lands_p2_or_p3(tmp_path):
    build(tmp_path, c3="**Outcomes: P2, N1, E1**")
    out = tmp_path / "o"
    assert b.run(out, tmp_path) == 2 and not (out / "benchmark_active_only.csv").exists()     # cells not there yet
    assert "results_kc23_s3_svmx" in (out / "S3_VERDICT.md").read_text() or "results_kc23_s3_" in (out / "S3_VERDICT.md").read_text()


def test_svmx_and_hgb_rows_appear_when_p3_and_their_cells_exist(tmp_path):
    build(tmp_path, c3="**Outcomes: P3, N1, E1**")
    rng = np.random.default_rng(9)
    for fam, token in (("svmx", "SVM"), ("hgb", "HGB")):
        for norm, m in (("global", 0.72), ("per_subject", 0.78)):
            _classical(tmp_path, f"results_kc23_s3_{fam}_{norm}", token, m, rng)
    assert b.run(tmp_path / "o", tmp_path) == 0
    t = pd.read_csv(tmp_path / "o" / "benchmark_active_only.csv")
    assert {"SVMX", "HGB"} <= set(t["family"]) and len(t) == 17


def test_the_c3_condition_cannot_be_decided_without_its_verdict(tmp_path):
    build(tmp_path)
    (tmp_path / "results_kc23_c3_ensemble" / "C3_VERDICT.md").unlink()
    assert b.run(tmp_path / "o", tmp_path) == 2


def test_a_c3_verdict_with_no_outcome_line_stops_the_benchmark(tmp_path):
    build(tmp_path, c3="NO OUTCOME COMPUTED. something is missing")
    assert b.run(tmp_path / "o", tmp_path) == 2


@pytest.mark.parametrize("victim", ["results_kc23_s3_lda_global/predictions_folds/feat_LDA_sub07_y_pred.npy",
                                    "results_kc23_s3_simple_per_subject/proba/TAG_sub12.npz",
                                    "results_aonly_persubj/predictions_folds/feat_RF_sub40_y_true.npy",
                                    "results_kc23_s3_resnet_se_global/cnn_arch_subjectwise.csv",
                                    "results_aonly_ensemble/aonly_ensemble_subjectwise.csv"])
def test_any_missing_cell_input_stops_the_benchmark_and_leaves_no_table(tmp_path, victim):
    build(tmp_path)
    out = tmp_path / "o"
    assert b.run(out, tmp_path) == 0
    (tmp_path / victim).unlink()
    assert b.run(out, tmp_path) == 2
    assert not (out / "benchmark_active_only.csv").exists()
    assert "NO OUTCOME COMPUTED" in (out / "S3_VERDICT.md").read_text()


def test_an_incomplete_subject_set_is_a_failure(tmp_path):
    build(tmp_path)
    f = next((tmp_path / "results_kc23_s3_lda_global").glob("lda_subjectwise.csv"))
    pd.read_csv(f).iloc[:-1].to_csv(f, index=False)
    assert b.run(tmp_path / "o", tmp_path) == 2


def test_order_changes_against_the_full_class_set_are_reported(tmp_path):
    build(tmp_path)
    # LDA per-subject beats SVM here (0.68 vs a full-class-set order SVM 0.777 > LDA 0.687): force it
    rng = np.random.default_rng(1)
    _lda(tmp_path, "results_kc23_s3_lda_per_subject", 0.90, rng)
    assert b.run(tmp_path / "o", tmp_path) == 0
    v = (tmp_path / "o" / "S3_VERDICT.md").read_text()
    assert "Order changes against the full-class-set benchmark" in v and "LDA" in v.split("Order changes")[1]


def test_the_lda_runner_writes_predictions_only_when_asked(tmp_path):
    import subprocess
    rng = np.random.default_rng(0)
    rows, X = [], []
    for s in range(1, 6):
        for ci, lab in enumerate(["DNS", "STDUP", "UPS", "WAK"]):
            for _ in range(30):
                rows.append({"subject": s, "movement": lab}); X.append(rng.normal(ci, 1, 6))
    np.savez(tmp_path / "f.npz", X=np.array(X)); pd.DataFrame(rows).to_csv(tmp_path / "m.csv", index=False)
    repo = Path(__file__).resolve().parent.parent
    env = {**__import__("os").environ, "PYTHONPATH": str(repo), "CUDA_VISIBLE_DEVICES": "-1"}

    def go(out, *extra):
        r = subprocess.run([sys.executable, str(repo / "run_lda_loso.py"), "--features", str(tmp_path / "f.npz"), "--meta",
                            str(tmp_path / "m.csv"), "--out", str(tmp_path / out), *extra], capture_output=True, text=True, env=env, timeout=300)
        assert r.returncode == 0, r.stderr[-800:]
        return tmp_path / out
    a, c = go("plain"), go("with", "--save-preds")
    assert not (a / "predictions_folds").exists() and len(list((c / "predictions_folds").glob("*_y_pred.npy"))) == 5
    pd.testing.assert_frame_equal(pd.read_csv(a / "lda_subjectwise.csv"), pd.read_csv(c / "lda_subjectwise.csv"))    # inert on the result
