"""src/kc23_c5_leak_stats.py: plateau rule, L1/L2/L3, the conformance additions (three models, per-subject cv-unit for every arm
per Enam's ruling of 26 Sept, L-OUT / L-NONE, every arm required) and the real-file layout
(b8_<tag>_<scheme>_g<g>_<cv_unit>_subjectwise.csv, one wide column per model, each job in its own subdirectory)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
import kc23_c5_leak_stats as mod
from kc23_c5_leak_stats import find_plateau, classify_l, tag_from_search_root, find_scheme_file, run

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
RNG = np.random.default_rng(3)
N = 40
TAG = "kc23_c5_siat"


def _f1(base, noise=0.005):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def _b(vals):
    return {g: _f1(v) for g, v in zip((1, 2, 4, 8, 16), vals)}


def test_plateau_found_early():
    g_star, detail = find_plateau(_b([0.80, 0.799, 0.798, 0.75, 0.70]))
    assert g_star == 1, detail


def test_plateau_later():
    g_star, _ = find_plateau(_b([0.90, 0.85, 0.80, 0.797, 0.70]))
    assert g_star == 4


def test_no_plateau():
    g_star, detail = find_plateau(_b([0.90, 0.85, 0.80, 0.75, 0.70]))
    assert g_star is None, detail


def _arms(p50, p0, i, b, g=1):
    return _f1(p50), _f1(p0), {gg: _f1(i) for gg in (1, 4, 16)}, {gg: _f1(b) for gg in (1, 2, 4, 8, 16)}


def test_l1_overlap_dominant():
    p50, p0, i, b = _arms(0.92, 0.80, 0.79, 0.78)               # total 14, overlap 12
    letters, _ = classify_l(p50, p0, i, b, 1, None)
    assert "L1" in letters


def test_l2_drift_dominant():
    p50, p0, i, b = _arms(0.90, 0.89, 0.88, 0.80)               # total 10, drift 8
    letters, _ = classify_l(p50, p0, i, b, 1, None)
    assert "L2" in letters and "L1" not in letters


def test_l3_no_plateau():
    p50, p0, i, b = _arms(0.9, 0.85, 0.8, 0.75)
    assert classify_l(p50, p0, i, b, None, None)[0] == ["L3"]


def test_l3_gstar_mismatch_vs_published():
    p50, p0, i, b = _arms(0.90, 0.85, 0.80, 0.75)
    letters, _ = classify_l(p50, p0, i, b, 4, 0.85)
    assert "L3" in letters


def test_l_none_when_neither_share_threshold_is_met():
    p50, p0, i, b = _arms(0.90, 0.86, 0.82, 0.80)               # total 10: overlap 4, drift 2
    assert classify_l(p50, p0, i, b, 1, None)[0] == ["L-NONE"]


def test_l_out_when_the_plateau_is_at_a_guard_with_no_interleaved_arm():
    p50, p0, i, b = _arms(0.90, 0.85, 0.80, 0.75)
    assert classify_l(p50, p0, i, b, 2, None)[0] == ["L-OUT"]      # I is registered at g in {1, 4, 16} only
    assert classify_l(p50, p0, i, b, 8, None)[0] == ["L-OUT"]


def test_l_out_when_blocked_is_not_below_pooled():
    p50, p0, i, b = _arms(0.80, 0.79, 0.80, 0.85)
    assert "L-OUT" in classify_l(p50, p0, i, b, 1, None)[0]


def test_tag_from_search_root():
    assert tag_from_search_root(Path("results/kc23_c5_leak_siat")) == "kc23_c5_siat"
    assert tag_from_search_root(Path("results/kc23_c5_leak_enabl3s")) == "kc23_c5_enabl3s"


def _write_scheme_csv(search_root: Path, subdir: str, scheme: str, guard: float, cv_unit: str, by_model: dict, tag=TAG):
    d = search_root / subdir
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"subject": list(range(1, N + 1))})
    for model, vals in by_model.items():
        df[model] = vals
    df.to_csv(d / f"b8_{tag}_{scheme}_g{guard:g}_{cv_unit}_subjectwise.csv", index=False)


def _tree(root: Path, p50=0.92, p0=0.80, i=0.79, b_base=0.78, models=("SVM", "RF", "LDA"), plateau_at_1=True):
    for pooled_scheme, sub, m in (("pooled_random", "p50", p50), ("pooled_random_nonoverlap", "p0", p0)):
        _write_scheme_csv(root, sub, pooled_scheme, 1.0, "per_subject", {mm: np.full(N, m) for mm in models})
    for g in (1, 2, 4, 8, 16):
        v = b_base - (0.0005 * g if plateau_at_1 else 0.02 * g)
        _write_scheme_csv(root, f"b{g}", "blocked", float(g), "per_subject", {mm: np.full(N, v) for mm in models})
    for g in (1, 4, 16):
        _write_scheme_csv(root, f"i{g}", "interleaved", float(g), "per_subject", {mm: np.full(N, i) for mm in models})
    return root


def test_find_scheme_file_reads_the_per_subject_arm_and_ignores_a_stray_pooled_file(tmp_path):
    root = _tree(tmp_path / "results/kc23_c5_leak_siat")
    _write_scheme_csv(root, "smoke", "blocked", 1.0, "pooled", {"SVM": np.zeros(N)})          # a pooled smoke-test output
    a = find_scheme_file(root, TAG, "blocked", 1.0)
    assert a.name.endswith("_g1_per_subject_subjectwise.csv") and a.parent.name == "b1"
    assert find_scheme_file(root, TAG, "blocked", 1.0, "pooled").parent.name == "smoke"


def test_run_end_to_end_all_three_models_per_subject(tmp_path):
    root = _tree(tmp_path / "results/kc23_c5_leak_siat")
    rc = run(root, published_blocked_sd=None)
    assert rc == 0
    v = (root / "C5_VERDICT.md").read_text()
    for m in ("SVM", "RF", "LDA"):
        assert f"**{m} outcome: L1**" in v
    assert "W-B1" not in v and "pooled" not in v.lower().replace("pooled_random", "") and LETTER_RE.search(v)
    assert "reproduction check" in v
    dec = pd.read_csv(root / "C5_decomposition.csv")
    assert list(dec["model"]) == ["SVM", "RF", "LDA"] and (dec["g_star"] == 1).all()


def test_the_gate_still_works_when_given_a_scheme_subdirectory(tmp_path):
    root = _tree(tmp_path / "results/kc23_c5_leak_siat")
    assert run(root / "b1", published_blocked_sd=None) == 0 and (root / "C5_VERDICT.md").exists()


def test_run_escalates_when_one_model_has_no_plateau(tmp_path):
    root = _tree(tmp_path / "results/kc23_c5_leak_siat")
    for g in (1, 2, 4, 8, 16):
        _write_scheme_csv(root, f"b{g}", "blocked", float(g), "per_subject",
                          {"SVM": np.full(N, 0.90 - 0.05 * g / 4), "RF": np.full(N, 0.78), "LDA": np.full(N, 0.78)})
    assert run(root, published_blocked_sd=None) == 20
    assert "**SVM outcome: L3**" in (root / "C5_VERDICT.md").read_text()


def test_run_exit_10_on_an_out_letter(tmp_path):
    root = _tree(tmp_path / "results/kc23_c5_leak_siat", p0=0.91, i=0.86, b_base=0.85)    # nothing fires
    assert run(root, published_blocked_sd=None) == 10
    assert "L-NONE" in (root / "C5_VERDICT.md").read_text()


@pytest.mark.parametrize("victim", ["pooled_random_nonoverlap_g1_per_subject", "pooled_random_g1_per_subject",
                                    "blocked_g8_per_subject", "blocked_g16_per_subject", "interleaved_g4_per_subject",
                                    "interleaved_g16_per_subject"])
def test_run_fails_closed_when_any_arm_is_missing(tmp_path, victim):
    root = _tree(tmp_path / "results/kc23_c5_leak_siat")
    next(root.glob(f"*/b8_{TAG}_{victim}_subjectwise.csv")).unlink()
    assert run(root, published_blocked_sd=None) == 20
    text = (root / "C5_VERDICT.md").read_text()
    assert "NO OUTCOME COMPUTED" in text and not LETTER_RE.search(text) and not (root / "C5_decomposition.csv").exists()


def test_a_missing_model_column_fails_closed(tmp_path):
    root = _tree(tmp_path / "results/kc23_c5_leak_siat", models=("SVM", "RF"))
    assert run(root, published_blocked_sd=None) == 20


def test_published_blocked_sd_enabl3s_is_reported_not_evaluated_never_silent():
    v, note = mod.load_published_blocked_sd("enabl3s", "SVM")
    assert v is None and "NOT evaluated" in note


def test_published_blocked_sd_siat_read_from_real_file():
    if not (mod.ROOT / mod.PUBLISHED_B8_FILE["siat"]).exists():
        pytest.skip("published b8 file not present")
    v, note = mod.load_published_blocked_sd("siat", "SVM")
    assert abs(v - 0.8801) < 1e-9 and "evaluated against" in note


def test_published_blocked_sd_for_a_model_the_file_lacks_is_reported_not_evaluated():
    if not (mod.ROOT / mod.PUBLISHED_B8_FILE["siat"]).exists():
        pytest.skip("published b8 file not present")
    v, note = mod.load_published_blocked_sd("siat", "LDA")
    assert v is None and "NOT evaluated for LDA" in note


def test_published_blocked_sd_missing_file_for_siat_is_an_error_not_a_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    with pytest.raises(mod.InputError):
        mod.load_published_blocked_sd("siat", "SVM")


def test_the_builder_runs_every_c5_arm_per_subject_and_has_no_w_b1_or_pooled_arm():
    import kc23_build_job_csvs as b
    for ds in ("siat", "enabl3s"):
        rows = {r["job_id"]: r["command"] for r in b.cpu_rows if r["job_id"].startswith(f"c5_{ds}_")}
        assert f"c5_{ds}_wb1" not in rows
        arms = {k: v for k, v in rows.items() if not k.endswith("_verdict")}
        assert sorted(arms) == sorted(f"c5_{ds}_{a}" for a in ["p50", "p0", "b1", "b2", "b4", "b8", "b16", "i1", "i4", "i16"])
        for jid, cmd in arms.items():
            assert "--cv-unit per_subject" in cmd and "pooled " not in cmd.replace("pooled_random", ""), jid
        verdict = next(r for r in b.cpu_rows if r["job_id"] == f"c5_{ds}_verdict")
        assert set(verdict["depends_on"].split(";")) == {f"c5_{ds}_{a}" for a in
                                                          ["p50", "p0", "b1", "b2", "b4", "b8", "b16", "i1", "i4", "i16"]}
