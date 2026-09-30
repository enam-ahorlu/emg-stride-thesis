"""KC-D4: classify_t against the plan's T1 to T3 as operationalised on 26 September 2026 (folds as Page blocks, Holm
across the two invariance meters, peak-then-fall by a paired Wilcoxon, per-fold Spearman with a sign test, T-OUT for
anything outside the grid), the gainjitter boundary, and the aggregator that feeds it (real-format fixtures)."""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kc23_d4_aggregate as agg
import kc23_d4_invariance_stats as st
from kc23_fixtures import make_run

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
N = 40
DOSES = st.MPCHANDROP_DOSES
RNG = np.random.default_rng(11)


def _mat(by_dose, noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    base = np.asarray(by_dose, float)[None, :]
    return base + rng.normal(0, noise, (N, len(by_dose))) + rng.normal(0, noise, (N, 1))


F1_PEAK_FALL = [0.80, 0.82, 0.83, 0.78, 0.72]
PROBE_FALLS = [0.90, 0.80, 0.70, 0.60, 0.50]
PERM_FALLS = [9.0, 8.0, 7.0, 6.0, 5.0]
SIL_TRACKS_F1 = [0.10, 0.11, 0.12, 0.08, 0.05]


def test_t1_all_four_conditions():
    letter, d = st.classify_t(_mat(F1_PEAK_FALL, seed=1), _mat(PROBE_FALLS, seed=2), _mat(SIL_TRACKS_F1, 0.002, 3),
                              _mat(PERM_FALLS, 0.05, 4))
    assert letter == "T1", d
    assert d["a_probe_falls"] and d["b_peaks_then_falls"] and d["c_tracks_class_silhouette"] and d["d_not_tracking_invariance"]


def test_t2_probe_does_not_fall():
    flat = [0.6] * 5
    letter, d = st.classify_t(_mat(F1_PEAK_FALL, seed=1), _mat(flat, seed=2), _mat(SIL_TRACKS_F1, 0.002, 3),
                              _mat(PERM_FALLS, 0.05, 4))
    assert letter == "T2", d


def test_t3_shape_shared_but_f1_does_not_track_class_silhouette():
    sil_flat = np.full((N, 5), 0.1)                        # constant per fold: no rank correlation exists
    letter, d = st.classify_t(_mat(F1_PEAK_FALL, seed=1), _mat(PROBE_FALLS, seed=2), sil_flat, _mat(PERM_FALLS, 0.05, 4))
    assert letter == "T3", d
    assert not d["c_tracks_class_silhouette"]


def test_t_out_when_f1_never_falls_even_though_the_probe_does():
    # a holds, b does not (F1 peaks at the highest dose): the old code defaulted this into T3
    f1_rising = [0.70, 0.74, 0.77, 0.80, 0.83]
    letter, d = st.classify_t(_mat(f1_rising, seed=1), _mat(PROBE_FALLS, seed=2), _mat([0.1, 0.11, 0.12, 0.13, 0.14], 0.002, 3),
                              _mat(PERM_FALLS, 0.05, 4))
    assert letter == "T-OUT" and not d["b_peaks_then_falls"], d


def test_t_out_when_f1_tracks_invariance_too():
    f1 = [0.70, 0.75, 0.80, 0.83, 0.79]                   # peaks at dose 4 of 5, rises with the falling probe
    letter, d = st.classify_t(_mat(f1, seed=1), _mat(PROBE_FALLS, seed=2), _mat([0.10, 0.11, 0.13, 0.14, 0.12], 0.002, 3),
                              _mat(PERM_FALLS, 0.05, 4))
    assert d["a_probe_falls"] and d["b_peaks_then_falls"] and d["c_tracks_class_silhouette"], d
    assert not d["d_not_tracking_invariance"]
    assert letter == "T-OUT"


def test_peak_at_the_lowest_dose_counts_as_peak_then_fall():
    # operationalisation (b): the argmax is not the highest dose (the earlier code's has_interior_peak was never used)
    f1 = [0.84, 0.82, 0.80, 0.78, 0.74]
    letter, d = st.classify_t(_mat(f1, seed=1), _mat(PROBE_FALLS, seed=2), _mat([0.14, 0.12, 0.10, 0.08, 0.05], 0.002, 3),
                              _mat(PERM_FALLS, 0.05, 4))
    assert d["peak_dose_idx"] == 0 and d["b_peaks_then_falls"]


def test_blocks_are_the_folds_not_the_realizations():
    # 3 blocks (the old design) could never exceed p = 1/6 for a perfect trend; 40 folds can be decisive
    from kc23_invariance_common import page_falls
    perfect = np.tile(np.array([0.9, 0.8, 0.7, 0.6, 0.5]), (40, 1)) + np.random.default_rng(0).normal(0, 0.001, (40, 5))
    assert page_falls(perfect)["p"] < 1e-6 and page_falls(perfect)["n_blocks"] == 40


def test_holm_across_the_two_meters_uses_the_subject_probe_adjusted_p():
    from kc23_invariance_common import holm_adjust
    assert holm_adjust([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm_adjust([0.04, 0.01]) == pytest.approx([0.04, 0.02])


def test_gainjitter_boundary_needs_a_significant_fall():
    doses = st.GAINJITTER_DOSES
    fall = _mat([0.83, 0.84, 0.85, 0.83, 0.79], seed=5)
    yes, line, _ = st.gainjitter_boundary(fall, doses)
    assert yes and "boundary" in line
    still_rising = _mat([0.80, 0.82, 0.83, 0.84, 0.85], seed=6)
    no, line2, _ = st.gainjitter_boundary(still_rising, doses)
    assert not no and "still highest" in line2
    noisy_dip = _mat([0.83, 0.84, 0.85, 0.845, 0.8449], 0.05, 7)
    assert not st.gainjitter_boundary(noisy_dip, doses)[0]


# ------------------------------------------------------------------- aggregator + stats, real-format fixtures
def _f1_fn(base, k):
    return lambda s: base + 0.01 * ((s * 7) % 5 - 2) / 2 + 0.0004 * ((s * k) % 3)


def _build_root(root: Path, f1_by_dose, probe_by_dose, sil_by_dose, perm_by_dose, gj_by_dose):
    for seed in agg.SEEDS:
        for i, dose in enumerate(DOSES):
            make_run(root, f"results/kc23_d4_mpchandrop_sd{dose:.2f}_s{seed}", augmentation="mpchandrop", seed=seed,
                     gain_sd=dose, f1=_f1_fn(f1_by_dose[i], seed), probe=_f1_fn(probe_by_dose[i], seed + 1),
                     sil=_f1_fn(sil_by_dose[i], seed + 2), perm=lambda s, v=perm_by_dose[i] / 900.0: v + 0.0002 * (s % 4))
        for i, (sd, pattern) in enumerate(agg.GAINJITTER_SOURCES.items()):
            make_run(root, pattern.format(seed=seed), augmentation="gainjitter", seed=seed, gain_sd=sd,
                     f1=_f1_fn(gj_by_dose[i], seed), instrumented=False)
        for arm, pattern, aug, p in agg.REFERENCE_ARMS:
            make_run(root, pattern.format(seed=seed), augmentation=aug, seed=seed, chandrop_p=p if p else 0.2)


@pytest.fixture
def root(tmp_path):
    _build_root(tmp_path, F1_PEAK_FALL, PROBE_FALLS, SIL_TRACKS_F1, PERM_FALLS, [0.83, 0.84, 0.85, 0.83, 0.79])
    return tmp_path


def test_end_to_end_t1_and_a_gainjitter_boundary(root):
    out = root / "d4_out"
    assert agg.run(root, out) == 0
    for f in ("d4_dose_sweep.csv", "d4_gainjitter_boundary.csv", "d4_reference_points.csv"):
        assert (out / f).exists()
    assert len(pd.read_csv(out / "d4_dose_sweep.csv")) == 3 * 5 * 40
    rc = st.run(out)
    v = (out / "D4_VERDICT.md").read_text()
    assert rc == 0 and "**Outcome: T1**" in v and "boundary" in v


def test_end_to_end_t2(tmp_path):
    _build_root(tmp_path, F1_PEAK_FALL, [0.6] * 5, SIL_TRACKS_F1, PERM_FALLS, [0.83, 0.84, 0.85, 0.83, 0.79])
    out = tmp_path / "o"
    assert agg.run(tmp_path, out) == 0
    assert st.run(out) == 0 and "**Outcome: T2**" in (out / "D4_VERDICT.md").read_text()


def test_end_to_end_t_out_exits_10_and_is_a_reported_outcome(tmp_path):
    _build_root(tmp_path, [0.70, 0.74, 0.77, 0.80, 0.83], PROBE_FALLS, [0.1, 0.11, 0.12, 0.13, 0.14], PERM_FALLS,
                [0.80, 0.82, 0.83, 0.84, 0.85])
    out = tmp_path / "o"
    assert agg.run(tmp_path, out) == 0
    assert st.run(out) == 10
    assert "T-OUT (outside the pre-registered grid)" in (out / "D4_VERDICT.md").read_text()


@pytest.mark.parametrize("victim", ["results/kc23_d4_mpchandrop_sd0.60_s7", "results/kc23_d1_r14_s123",
                                    "results/kc23_d4_gainjitter_sd1.00_s42", "results/kc23_d1_r15_s42"])
def test_aggregate_missing_run_fails_and_writes_nothing(root, victim):
    shutil.rmtree(root / victim)
    out = root / "o"
    assert agg.run(root, out) == 1
    assert not out.exists() or not list(out.glob("*.csv"))


def test_aggregate_mislabelled_dose_fails(root):
    import json
    p = root / "results/kc23_d4_mpchandrop_sd0.80_s7" / "run_config.json"
    cfg = json.loads(p.read_text()); cfg["args"]["aug_gain_sd"] = 0.5; p.write_text(json.dumps(cfg))
    assert agg.run(root, root / "o") == 1


@pytest.mark.parametrize("victim", ["instr/embed_probes.csv", "instr/permutation.csv", "run_config.json"])
def test_aggregate_missing_instrumentation_fails(root, victim):
    (root / "results/kc23_d4_mpchandrop_sd0.50_s42" / victim).unlink()
    assert agg.run(root, root / "o") == 1


def test_aggregate_nan_probe_fails(root):
    p = root / "results/kc23_d4_mpchandrop_sd0.40_s123" / "instr" / "embed_probes.csv"
    d = pd.read_csv(p); d.loc[3, "subject_probe_bacc"] = np.nan; d.to_csv(p, index=False)
    assert agg.run(root, root / "o") == 1


@pytest.mark.parametrize("missing", ["d4_dose_sweep.csv", "d4_gainjitter_boundary.csv"])
def test_stats_each_input_missing_fails_closed_with_no_letter(root, missing):
    out = root / "o"
    agg.run(root, out)
    (out / missing).unlink()
    assert st.run(out) == 20
    assert not LETTER_RE.search((out / "D4_VERDICT.md").read_text())


def test_stats_incomplete_fold_set_fails_closed(root):
    out = root / "o"
    agg.run(root, out)
    d = pd.read_csv(out / "d4_dose_sweep.csv")
    d[~((d["subject"] == 5) & (d["dose"] == 0.6))].to_csv(out / "d4_dose_sweep.csv", index=False)
    assert st.run(out) == 20


def test_stats_stale_verdict_is_replaced_when_an_input_disappears(root):
    out = root / "o"
    agg.run(root, out)
    assert st.run(out) == 0
    (out / "d4_gainjitter_boundary.csv").unlink()
    assert st.run(out) == 20 and not LETTER_RE.search((out / "D4_VERDICT.md").read_text())
    assert not (out / "D4_detail.csv").exists()
