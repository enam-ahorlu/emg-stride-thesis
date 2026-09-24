"""Synthetic tests for kc23_c5_leak_stats.py: find_plateau, L1/L2/L3, and the
real-file discovery layer (fixed 2026-09-24: the gate originally assumed
p50_subjectwise.csv / b{g}_subjectwise.csv files that b8_movement_blocked_sd.py
never wrote -- its real --scheme output is b8_<tag>_<scheme>_g<g>_<cv_unit>_
subjectwise.csv with one wide column per model)."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c5_leak_stats import find_plateau, classify_l, tag_from_search_root, find_scheme_file, run

RNG = np.random.default_rng(3)
N = 40


def _f1(base, noise=0.005):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def test_plateau_found_early():
    b = {1: _f1(0.80), 2: _f1(0.799), 4: _f1(0.798), 8: _f1(0.75), 16: _f1(0.70)}
    g_star, detail = find_plateau(b)
    assert g_star == 1, detail


def test_no_plateau():
    b = {1: _f1(0.90), 2: _f1(0.85), 4: _f1(0.80), 8: _f1(0.75), 16: _f1(0.70)}
    g_star, detail = find_plateau(b)
    assert g_star is None, detail


def test_l1_overlap_dominant():
    p50 = _f1(0.92)
    p0 = _f1(0.80)     # big overlap drop
    i_g = _f1(0.79)
    b_g = _f1(0.78)    # total = p50-b_g = 14pt, overlap=12pt (>=60%)
    letters, detail = classify_l(p50, p0, i_g, b_g, g_star=1, published_blocked_sd=None)
    assert "L1" in letters, (letters, detail)


def test_l2_drift_dominant():
    p50 = _f1(0.90)
    p0 = _f1(0.89)     # small overlap drop (1pt)
    i_g = _f1(0.88)    # small autocorr drop (1pt)
    b_g = _f1(0.80)    # big drift drop (8pt); total=10pt, drift=8pt (>=40%)
    letters, detail = classify_l(p50, p0, i_g, b_g, g_star=1, published_blocked_sd=None)
    assert "L2" in letters, (letters, detail)


def test_l3_no_plateau():
    letters, detail = classify_l(_f1(0.9), _f1(0.85), _f1(0.8), _f1(0.75), g_star=None,
                                 published_blocked_sd=None)
    assert letters == ["L3"], (letters, detail)


def test_l3_gstar_mismatch_vs_published():
    p50, p0, i_g, b_g = _f1(0.90), _f1(0.85), _f1(0.80), _f1(0.75)
    letters, detail = classify_l(p50, p0, i_g, b_g, g_star=4, published_blocked_sd=0.85)
    assert "L3" in letters, (letters, detail)


def test_tag_from_search_root():
    assert tag_from_search_root(Path("results_kc23_c5_leak_siat")) == "kc23_c5_siat"
    assert tag_from_search_root(Path("results_kc23_c5_leak_enabl3s")) == "kc23_c5_enabl3s"


def _write_scheme_csv(search_root: Path, scheme_subdir: str, tag: str, scheme: str, guard: float,
                      cv_unit: str, f1_by_model: dict):
    """Mirrors the real layout: each job writes into its OWN subdirectory of
    the shared dataset root, never into the root itself (avoiding the
    is_complete() collision fixed 2026-09-24)."""
    d = search_root / scheme_subdir
    d.mkdir(parents=True, exist_ok=True)
    stem = f"b8_{tag}_{scheme}_g{guard:g}_{cv_unit}"
    df = pd.DataFrame({"subject": list(range(1, 41))})
    for model, vals in f1_by_model.items():
        df[model] = vals
    df.to_csv(d / f"{stem}_subjectwise.csv", index=False)


def test_find_scheme_file_matches_real_naming(tmp_path):
    root = tmp_path / "results_kc23_c5_leak_siat"
    _write_scheme_csv(root, "p50", "kc23_c5_siat", "pooled_random", 1.0, "per_subject",
                      {"SVM": [0.8] * 40})
    f = find_scheme_file(root, "kc23_c5_siat", "pooled_random")
    assert f is not None and f.name == "b8_kc23_c5_siat_pooled_random_g1_per_subject_subjectwise.csv"


def test_run_end_to_end_on_real_file_layout(tmp_path):
    root = tmp_path / "results_kc23_c5_leak_siat"
    _write_scheme_csv(root, "p50", "kc23_c5_siat", "pooled_random", 1.0, "per_subject", {"SVM": np.full(40, 0.92)})
    _write_scheme_csv(root, "p0", "kc23_c5_siat", "pooled_random_nonoverlap", 1.0, "per_subject",
                      {"SVM": np.full(40, 0.80)})
    for g in (1, 2, 4, 8, 16):
        _write_scheme_csv(root, f"b{g}", "kc23_c5_siat", "blocked", float(g), "pooled",
                          {"SVM": np.full(40, 0.78 - g * 0.001)})
    for g in (1, 4, 16):
        _write_scheme_csv(root, f"i{g}", "kc23_c5_siat", "interleaved", float(g), "pooled",
                          {"SVM": np.full(40, 0.79)})
    # the triggering job's own out_dir is one of the leaf subdirectories, not the shared root
    rc = run(root / "wb1", published_blocked_sd=None, model="SVM")
    assert (root / "C5_VERDICT.md").exists()
    assert rc in (0, 20)


def test_run_missing_p0_is_fail_not_fallback(tmp_path):
    root = tmp_path / "results_kc23_c5_leak_siat"
    _write_scheme_csv(root, "p50", "kc23_c5_siat", "pooled_random", 1.0, "per_subject", {"SVM": np.full(40, 0.92)})
    # p0 deliberately missing
    rc = run(root / "p50", published_blocked_sd=None, model="SVM")
    assert rc == 20
    assert "FAIL" in (root / "C5_VERDICT.md").read_text()


def test_different_schemes_do_not_collide_on_is_complete(tmp_path):
    """The bug this whole file layout exists to prevent: two jobs sharing one
    out_dir would make is_complete() mark the second "already done" the
    instant the first wrote any *subjectwise.csv there."""
    root = tmp_path / "results_kc23_c5_leak_siat"
    _write_scheme_csv(root, "p50", "kc23_c5_siat", "pooled_random", 1.0, "per_subject", {"SVM": np.full(40, 0.92)})
    p0_dir = root / "p0"
    assert not any(p0_dir.glob("*subjectwise.csv")) if p0_dir.exists() else True
    assert not list(root.glob("*subjectwise.csv")), "no file should land directly in the shared root"


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
