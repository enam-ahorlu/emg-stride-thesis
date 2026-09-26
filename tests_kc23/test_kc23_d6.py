"""KC-D6 gates and outcomes, conforming to the plan text of 23 September (rewritten 26 September 2026): the FOLDS are the
Page blocks, Holm across the two meters, the falling limb needs >= 2 pts AND a paired Wilcoxon and is measured to the
largest NON-DIVERGED knob, the unseen-subject probe comes from embed_probes.csv, and anything outside the grid is X-OUT."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_d6_aggregate as agg
import kc23_d6_stats as st

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
N = 40


def _mat(by_knob, noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    base = np.asarray(by_knob, float)[None, :]
    return base + rng.normal(0, noise, (N, len(by_knob))) + rng.normal(0, noise, (N, 1))


# ---------------------------------------------------------------- sanity
def test_sanity_pass():
    assert st.classify_sanity(0.825)[0] == "PASS"


def test_sanity_fail():
    assert st.classify_sanity(0.80)[0] == "FAIL"


# ---------------------------------------------------------------- manipulation gate
DOM_FALLS = [0.95, 0.85, 0.75, 0.60]
SUBJ_FALLS = [0.90, 0.70, 0.50, 0.30]


def test_g_pass():
    letter, d = st.classify_manipulation(_mat(DOM_FALLS, seed=1), _mat(SUBJ_FALLS, seed=2))
    assert letter == "G-PASS", d
    assert d["domain_fall_pts"] > 30 and d["subject_moves"]


def test_g_weak_partial_fall():
    letter, d = st.classify_manipulation(_mat([0.95, 0.92, 0.90, 0.88], seed=1), _mat(SUBJ_FALLS, seed=2))
    assert letter == "G-WEAK", d               # a 7 pt fall


def test_g_weak_only_one_meter_moves():
    letter, d = st.classify_manipulation(_mat(DOM_FALLS, seed=1), _mat([0.6] * 4, seed=2))
    assert letter == "G-WEAK" and not d["subject_moves"], d


def test_g_fail_only_when_both_meters_fail():
    letter, d = st.classify_manipulation(_mat([0.95, 0.945, 0.94, 0.94], seed=1), _mat([0.9] * 4, seed=2))
    assert letter == "G-FAIL" and not d["subject_moves"], d


def test_domain_fall_under_two_points_with_a_significant_subject_trend_is_g_weak():
    # Enam's ruling of 26 Sept: G-WEAK covers "only one of the two meters moves", so the plan's two wordings no longer overlap
    letter, d = st.classify_manipulation(_mat([0.95, 0.945, 0.94, 0.94], seed=1), _mat(SUBJ_FALLS, seed=2))
    assert d["domain_fall_pts"] < 2 and d["subject_moves"] and letter == "G-WEAK", d


def test_a_domain_fall_of_two_to_ten_points_with_a_flat_subject_probe_is_g_weak_not_g_fail():
    letter, d = st.classify_manipulation(_mat([0.95, 0.92, 0.90, 0.88], seed=1), _mat([0.9] * 4, seed=2))
    assert letter == "G-WEAK", d


def test_manipulation_page_blocks_are_the_folds():
    _, d = st.classify_manipulation(_mat(DOM_FALLS, seed=1), _mat(SUBJ_FALLS, seed=2))
    # p-values from 40 blocks reach far below what 3 realization-blocks could (the smallest possible is 1/6)
    assert d["subject_p_raw"] < 1e-6 and d["domain_p_raw"] < 1e-6


def test_manipulation_holm_is_applied_across_the_two_meters():
    _, d = st.classify_manipulation(_mat(DOM_FALLS, seed=1), _mat(SUBJ_FALLS, seed=2))
    lo, hi = sorted([d["domain_p_raw"], d["subject_p_raw"]])
    assert min(d["domain_p_holm"], d["subject_p_holm"]) == pytest.approx(min(1.0, 2 * lo))
    assert max(d["domain_p_holm"], d["subject_p_holm"]) >= max(d["domain_p_holm"], d["subject_p_holm"])


def test_sfc_unit_norm_is_the_normalised_embedding_the_loss_sees():
    raw = np.tile([9.35, 8.5, 7.9, 7.2, 6.65], (N, 1))                          # the D2c shrink in the RAW norm
    ok, d = st.sfc_unit_norm(np.full((N, 5), 3e-9), raw)
    assert ok and d["normalised_norm_dev_max"] < 1e-5 and d["raw_norm_by_weight"].startswith("9.3500;8.5000")


def test_a_broken_normalisation_is_the_stop_condition():
    bad = np.full((N, 5), 1e-9); bad[3, 2] = 2e-3
    ok, d = st.sfc_unit_norm(bad, np.full((N, 5), 7.0))
    assert not ok and d["normalised_norm_dev_max"] == pytest.approx(2e-3)


def test_raw_scale_alone_is_not_a_stop_condition_and_the_old_ratio_is_gone():
    assert not hasattr(st, "SFC_NORM_MAX_RATIO") and not hasattr(st, "sfc_norm_flat")
    ok, _ = st.sfc_unit_norm(np.zeros((N, 3)), np.tile([100.0, 10.0, 1.0], (N, 1)))
    assert ok


# ---------------------------------------------------------------- outcome, X1 to X4 and X-OUT
KNOBS7 = 7
INV = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
SIL_TRACK = [0.10, 0.11, 0.12, 0.13, 0.12, 0.08, 0.04]
F1_PEAK = [0.80, 0.82, 0.84, 0.86, 0.85, 0.80, 0.72]       # interior peak at knob 3, then falls


def _outcome(f1, sil=None, cprobe=None, dom=None, subj=None, diverged=None, noise=0.003):
    k = len(f1)
    inv = np.linspace(0.9, 0.3, k)
    return st.classify_outcome(_mat(f1, noise, 1), _mat(dom if dom is not None else inv, 0.004, 2),
                               _mat(subj if subj is not None else inv, 0.004, 3),
                               _mat(sil if sil is not None else np.interp(np.arange(k), np.arange(len(SIL_TRACK)), SIL_TRACK), 0.0015, 4),
                               _mat(cprobe if cprobe is not None else np.linspace(0.5, 0.9, k) * 0 + np.array(f1) * 1.0, 0.003, 5),
                               diverged)


def test_x1_a_third_axis_is_measured():
    letter, d = _outcome(F1_PEAK)
    assert letter == "X1", d
    assert d["interior_peak"] and d["falling_limb"] and d["tracks_silhouette"] and d["tracks_class_probe"]
    assert not d["tracks_subject_probe"] and not d["tracks_domain_probe"]


def test_x2_f1_flat_or_rising_to_the_largest_knob():
    letter, d = _outcome([0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86])
    assert letter == "X2" and not d["falling_limb"], d


def test_x2_when_the_dip_is_smaller_than_two_points_even_if_significant():
    letter, d = _outcome([0.80, 0.82, 0.84, 0.86, 0.85, 0.855, 0.85], noise=0.0005)   # 1 pt below the peak
    assert letter == "X2", d


def test_x2_when_a_fall_of_two_points_or_more_is_not_significant():
    from kc23_invariance_common import peak_then_fall
    rng = np.random.default_rng(7)
    f1 = np.full((N, 5), 0.85) + rng.normal(0, 0.10, (N, 5))
    f1[:, 2] += 0.03                                     # a 3 pt bump in the mean, buried in fold-to-fold noise
    r = peak_then_fall(f1, min_gap_pp=2.0)
    assert r["gap_to_highest_pp"] >= 0 and not r["falls"] and r["wilcoxon_p"] >= 0.05


def test_x3_f1_falls_from_the_first_step():
    f1 = [0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74]
    letter, d = _outcome(f1, sil=[0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02])
    assert letter == "X3", d


def test_x4_shape_appears_but_f1_does_not_track_class_information():
    flat_sil = np.full(7, 0.1)
    letter, d = _outcome(F1_PEAK, sil=flat_sil, cprobe=np.full(7, 0.8))
    assert d["falling_limb"] and d["interior_peak"] and not d["tracks_silhouette"]
    assert letter == "X4"


def test_x_out_when_invariance_does_not_rise():
    letter, d = _outcome(F1_PEAK, dom=[0.7] * 7, subj=[0.6] * 7)
    assert letter == "X-OUT" and not d["invariance_rises"]        # the old code called this X4


def test_x_out_when_f1_tracks_the_invariance_meters():
    # peak late in the sweep so F1 rises with invariance for most of it, then a significant fall to the last knob
    f1 = [0.70, 0.74, 0.78, 0.82, 0.86, 0.90, 0.85]
    letter, d = _outcome(f1, sil=[0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.09])
    assert d["interior_peak"] and d["falling_limb"] and d["tracks_silhouette"]
    assert d["tracks_subject_probe"] and letter == "X-OUT", d


def test_diverged_top_arm_is_excluded_and_the_fall_is_measured_to_the_largest_non_diverged_knob():
    f1 = [0.80, 0.82, 0.84, 0.86, 0.85, 0.80, 0.40]                  # the last arm collapsed (diverged)
    diverged = np.array([False] * 6 + [True])
    letter, d = _outcome(f1, diverged=diverged)
    assert d["n_knobs_kept"] == 6 and d["gap_to_last_kept_pp"] == pytest.approx(6.0, abs=1.0), d


def test_too_few_non_diverged_knobs_is_an_input_error():
    with pytest.raises(st.InputError):
        _outcome(F1_PEAK, diverged=np.array([True] * 5 + [False] * 2))


def test_mechanism_letters_unchanged():
    assert st.classify_mechanism(0.85, 0.80, 0.85, 0.9, 0.8)[0] == "C-M1"
    assert st.classify_mechanism(0.80, 0.795, 0.85, 0.9, 0.8)[0] == "C-M2"
    assert st.classify_mechanism(0.81, 0.79, 0.85, 0.9, 0.8)[0] == "C-M3"


# ---------------------------------------------------------------- aggregator + stats on real-format arm directories
def make_arm(root: Path, family: str, knob, seed: int, *, f1, dom, subj, sil, cprobe, norm=7.0, loss=0.5,
             nonfinite=(), normdev=2e-9, retry=False, within=None):
    d = root / (agg.FAMILY_DIR[family](knob, seed) + ("__retry" if retry else ""))
    (d / "instr").mkdir(parents=True, exist_ok=True)
    cfg = {"seed": seed, "arch": "resnet_se", "augmentation": "chandrop", "epochs": 40, "batch": 256}
    if retry:
        cfg["grad_clip"] = 5.0
    if family == "sfc":
        cfg.update(coral_lambda=knob, coral_normalize="l2", target_pass="train")
    else:
        cfg.update(adv_lambda=knob, adv_mode="marginal", norm_mode="global" if family == "adv_marginal" else "per_subject")
    (d / "run_config.json").write_text(json.dumps({"args": cfg}), encoding="utf-8")
    subs = list(range(1, N + 1))
    jit = lambda s, k: 0.002 * (((s * 5 + k) % 7) - 3) / 3
    pd.DataFrame({"subject": subs, "adv_lambda": knob, "coral_lambda": knob,
                  "f1_macro": [f1 + jit(s, 1) for s in subs]}).to_csv(d / agg.SUBJECTWISE[family], index=False)
    al = pd.DataFrame({"subject": subs, "domain_probe_bacc": [dom + jit(s, 2) for s in subs],
                       "class_probe_tgt_bacc": 0.9, "feat_norm_src": [norm + jit(s, 3) for s in subs]})
    if family == "sfc":
        al["coral_embed_normdev_max"] = normdev
    al.to_csv(d / "alignment_subjectwise.csv", index=False)
    ep = pd.DataFrame({"subject": subs, "subject_probe_bacc": [subj + jit(s, 4) for s in subs],
                       "held_out_class_silhouette": [sil + jit(s, 5) * 0.1 for s in subs],
                       "held_out_class_probe_bacc": [cprobe + jit(s, 6) for s in subs]})
    if within is not None:
        ep["subject_probe_within_class_bacc"] = [within + jit(s, 7) for s in subs]
        ep["n_within_class_probes"] = 4
    ep.to_csv(d / "instr" / "embed_probes.csv", index=False)
    rows = []
    for s in subs:
        for ep in (1, 2, 3):
            bad = s in nonfinite and ep == 2
            rows.append({"epoch": ep, "val_loss": float("nan") if bad else loss + (3 - ep) * 0.05, "best": int(ep == 3 and not bad),
                         "subject": s})
    pd.DataFrame(rows).to_csv(d / "training_log.csv", index=False)


def build_family(root: Path, family: str, seeds, dom_fn, subj_fn, f1_fn, sil_fn, cprobe_fn, norm_fn=lambda i: 7.0,
                 loss_fn=lambda i: 0.5, nonfinite_at=None):
    for seed in seeds:
        for i, knob in enumerate(agg.FAMILY_KNOBS[family]):
            make_arm(root, family, knob, seed, f1=f1_fn(i), dom=dom_fn(i), subj=subj_fn(i), sil=sil_fn(i),
                     cprobe=cprobe_fn(i), norm=norm_fn(i), loss=loss_fn(i),
                     nonfinite=range(1, 41) if nonfinite_at == i else ())


def _shape(vals, k):
    v = np.asarray(vals, float)
    return np.interp(np.linspace(0, len(v) - 1, k), np.arange(len(v)), v)


def build_all(root, seeds, pass_shape=True):
    for fam in agg.KNOWN_FAMILIES:
        k = len(agg.FAMILY_KNOBS[fam])
        dom = _shape([0.95, 0.85, 0.7, 0.5], k) if pass_shape else _shape([0.95, 0.949, 0.948, 0.947], k)
        subj = _shape([0.9, 0.7, 0.5, 0.3], k)
        f1 = _shape([0.80, 0.84, 0.86, 0.85, 0.78, 0.72], k)
        sil = _shape([0.10, 0.12, 0.13, 0.12, 0.08, 0.04], k)
        build_family(root, fam, seeds, lambda i: dom[i], lambda i: subj[i], lambda i: f1[i], lambda i: sil[i],
                     lambda i: f1[i], norm_fn=(lambda i: 7.0 + 0.01 * i) if fam == "sfc" else (lambda i: 7.0))


@pytest.fixture
def root(tmp_path):
    build_all(tmp_path, [42])
    return tmp_path


def test_manipulation_end_to_end(root):
    out = root / "results_kc23_d6_manipulation_check"
    assert agg.run(out, root, "manipulation") == 0
    assert (out / "d6_family_adv_marginal.csv").exists() and (out / "d6_arm_divergence.csv").exists()
    rc = st.run(out)
    v = (out / "D6_VERDICT.md").read_text()
    assert rc == 0 and "manipulation_adv_marginal: G-PASS" in v and "manipulation_sfc: G-PASS" in v
    g = pd.read_csv(out / "D6_gates.csv")
    assert g.loc[g["item"] == "manipulation_sfc", "norm_ok"].iloc[0]


def test_the_subject_probe_is_the_embedding_probe_not_the_class_probe(root):
    out = root / "results_kc23_d6_manipulation_check"
    agg.run(out, root, "manipulation")
    fam = pd.read_csv(out / "d6_family_adv_marginal.csv")
    assert (fam["class_probe_bacc"].between(0.6, 1.0)).all()
    assert fam["subject_probe_bacc"].min() < 0.4 and fam["subject_probe_bacc"].max() > 0.8   # falls 0.9 -> 0.3, not the constant 0.9 class probe


def test_a_flat_domain_probe_with_a_falling_subject_probe_is_weak_and_still_runs_stage_two(tmp_path):
    build_all(tmp_path, [42], pass_shape=False)               # domain probe flat, unseen-subject probe falls
    out = tmp_path / "results_kc23_d6_manipulation_check"
    assert agg.run(out, tmp_path, "manipulation") == 0
    st.run(out)
    v = (out / "D6_VERDICT.md").read_text()
    assert v.count("G-WEAK") >= 3 and "G-FAIL" not in v.split("```")[0]
    assert set(agg.passing_families(out / "D6_gates.csv")) == set(agg.KNOWN_FAMILIES)


def test_gfail_family_is_reported_and_stops_stage_two(tmp_path):
    build_all(tmp_path, [42], pass_shape=False)
    for fam in agg.KNOWN_FAMILIES:                            # now the unseen-subject probe is flat as well: BOTH meters fail
        for w in agg.FAMILY_KNOBS[fam]:
            make_arm(tmp_path, fam, w, 42, f1=0.8, dom=0.95, subj=0.9, sil=0.1, cprobe=0.9)
    out = tmp_path / "results_kc23_d6_manipulation_check"
    assert agg.run(out, tmp_path, "manipulation") == 0
    st.run(out)
    v = (out / "D6_VERDICT.md").read_text()
    assert v.count("G-FAIL") >= 3 and "Every family is G-FAIL" in v
    assert agg.passing_families(out / "D6_gates.csv") == {}


def test_a_raw_norm_shrink_with_unit_normalised_embeddings_does_not_stop_sfc(tmp_path):
    build_all(tmp_path, [42])
    for i, w in enumerate(agg.FAMILY_KNOBS["sfc"]):                               # the D2c shrink in the RAW norm only
        make_arm(tmp_path, "sfc", w, 42, f1=0.8, dom=0.95 - 0.1 * i, subj=0.9 - 0.15 * i, sil=0.1, cprobe=0.9, norm=9.35 - 0.7 * i)
    out = tmp_path / "results_kc23_d6_manipulation_check"
    agg.run(out, tmp_path, "manipulation")
    assert st.run(out) == 0
    v = (out / "D6_VERDICT.md").read_text()
    assert "descriptive, not a stop condition" in v and "sfc" in agg.passing_families(out / "D6_gates.csv")


def test_a_non_unit_normalised_embedding_stops_the_family(tmp_path):
    build_all(tmp_path, [42])
    make_arm(tmp_path, "sfc", 10, 42, f1=0.8, dom=0.5, subj=0.3, sil=0.1, cprobe=0.9, normdev=4e-3)
    out = tmp_path / "results_kc23_d6_manipulation_check"
    agg.run(out, tmp_path, "manipulation")
    assert st.run(out) == 10
    assert "is not unit norm" in (out / "D6_VERDICT.md").read_text()
    assert "sfc" not in agg.passing_families(out / "D6_gates.csv")


def test_the_unit_norm_column_is_required_for_sfc(tmp_path):
    build_all(tmp_path, [42])
    p = tmp_path / agg.FAMILY_DIR["sfc"](10, 42) / "alignment_subjectwise.csv"
    pd.read_csv(p).drop(columns=["coral_embed_normdev_max"]).to_csv(p, index=False)
    assert agg.run(tmp_path / "o", tmp_path, "manipulation") == 1


def test_outcome_end_to_end_for_passing_families(tmp_path):
    build_all(tmp_path, [42, 7, 123])
    man = tmp_path / "results_kc23_d6_manipulation_check"
    agg.run(man, tmp_path, "manipulation"); st.run(man)
    out = tmp_path / "results_kc23_d6_outcome_check"
    assert agg.run(out, tmp_path, "outcome", man / "D6_gates.csv") == 0
    rc = st.run(out)
    v = (out / "D6_VERDICT.md").read_text()
    assert "outcome_adv_marginal:" in v and "outcome_sfc:" in v and rc in (0, 10)


def test_outcome_with_no_passing_family_is_reported_not_silent(tmp_path):
    build_all(tmp_path, [42], pass_shape=False)
    man = tmp_path / "results_kc23_d6_manipulation_check"
    agg.run(man, tmp_path, "manipulation"); st.run(man)
    g = pd.read_csv(man / "D6_gates.csv"); g["letter"] = "G-FAIL"; g.to_csv(man / "D6_gates.csv", index=False)
    out = tmp_path / "results_kc23_d6_outcome_check"
    assert agg.run(out, tmp_path, "outcome", man / "D6_gates.csv") == 0
    assert st.run(out) == 10 and "NO-STAGE-2" in (out / "D6_VERDICT.md").read_text()


# ---------------------------------------------------------------- divergence, the plan's rule, and the one retry
def _diverging_adv(tmp_path):
    build_family(tmp_path, "adv_marginal", [42], lambda i: 0.9 - 0.08 * i, lambda i: 0.9 - 0.08 * i, lambda i: 0.8,
                 lambda i: 0.1, lambda i: 0.9, loss_fn=lambda i: 0.5 if i < 6 else 1.6)


def test_adv_arm_diverges_when_its_best_epoch_loss_exceeds_twice_the_lambda0_arm(tmp_path):
    _diverging_adv(tmp_path)
    assert [a[0] for a in agg.diverged_arms(tmp_path, "adv_marginal", [42])] == [10]
    assert agg.diverged_arms(tmp_path, "adv_marginal", [42])[0][2] == "results_kc23_d6_adv_marginal_l10_s42"   # the real directory name


def test_a_diverged_arm_without_its_retry_fails_closed_and_names_the_retry(tmp_path):
    _diverging_adv(tmp_path)
    with pytest.raises(ValueError, match="__retry directory; run the one retry with --grad-clip 5.0"):
        agg.build_family(tmp_path, "adv_marginal", [42])
    assert agg.run(tmp_path / "o", tmp_path, "manipulation") == 1


def test_a_successful_retry_replaces_the_arm_and_is_flagged(tmp_path):
    _diverging_adv(tmp_path)
    make_arm(tmp_path, "adv_marginal", 10, 42, f1=0.7, dom=0.2, subj=0.1, sil=0.1, cprobe=0.9, loss=0.55, retry=True)
    fam, div = agg.build_family(tmp_path, "adv_marginal", [42])
    row = div.set_index("knob").loc[10]
    assert row["diverged_original"] and row["retried"] and not row["diverged"]
    assert fam[fam["knob"] == 10]["f1"].mean() == pytest.approx(0.7, abs=0.01)        # the retry's data, not the original's
    assert not div.set_index("knob").loc[0.3]["retried"]


def test_a_retry_that_diverges_again_stays_diverged_and_is_never_dropped(tmp_path):
    _diverging_adv(tmp_path)
    make_arm(tmp_path, "adv_marginal", 10, 42, f1=0.3, dom=0.2, subj=0.1, sil=0.1, cprobe=0.9, loss=1.7, retry=True)
    fam, div = agg.build_family(tmp_path, "adv_marginal", [42])
    row = div.set_index("knob").loc[10]
    assert row["diverged"] and row["retried"] and row["diverged_original"]
    assert (fam["knob"] == 10).any() and fam[fam["knob"] == 10]["diverged"].all()


def test_a_retry_directory_must_have_been_run_with_grad_clip_5(tmp_path):
    _diverging_adv(tmp_path)
    make_arm(tmp_path, "adv_marginal", 10, 42, f1=0.7, dom=0.2, subj=0.1, sil=0.1, cprobe=0.9, loss=0.55, retry=True)
    p = tmp_path / (agg.FAMILY_DIR["adv_marginal"](10, 42) + "__retry") / "run_config.json"
    cfg = json.loads(p.read_text()); cfg["args"]["grad_clip"] = 1.0; p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="grad-clip 5.0"):
        agg.build_family(tmp_path, "adv_marginal", [42])


def test_an_original_arm_must_not_carry_grad_clip(tmp_path):
    _diverging_adv(tmp_path)
    p = tmp_path / agg.FAMILY_DIR["adv_marginal"](0.3, 42) / "run_config.json"
    cfg = json.loads(p.read_text()); cfg["args"]["grad_clip"] = 5.0; p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="not a retry"):
        agg.diverged_arms(tmp_path, "adv_marginal", [42])


def test_a_nan_loss_marks_the_arm_diverged(tmp_path):
    build_family(tmp_path, "adv_marginal", [42], lambda i: 0.9 - 0.08 * i, lambda i: 0.9 - 0.08 * i, lambda i: 0.8,
                 lambda i: 0.1, lambda i: 0.9, nonfinite_at=4)
    assert [a[0] for a in agg.diverged_arms(tmp_path, "adv_marginal", [42])] == [1]


# ---------------------------------------------------------------- fail closed
def test_aggregate_requires_a_mode(tmp_path):
    assert agg.run(tmp_path / "o", tmp_path, None) == 1


def test_aggregate_missing_probe_file_fails_and_writes_nothing(root):
    (root / agg.FAMILY_DIR["sfc"](10, 42) / "instr" / "embed_probes.csv").unlink()
    out = root / "o"
    assert agg.run(out, root, "manipulation") == 1
    assert not list(out.glob("*.csv"))


def test_aggregate_mislabelled_knob_fails(root):
    p = root / agg.FAMILY_DIR["adv_marginal"](0.3, 42) / "run_config.json"
    cfg = json.loads(p.read_text()); cfg["args"]["adv_lambda"] = 3; p.write_text(json.dumps(cfg))
    assert agg.run(root / "o", root, "manipulation") == 1


def test_no_class_probe_fallback_for_the_subject_probe(root):
    p = root / agg.FAMILY_DIR["adv_marginal"](0.3, 42) / "instr" / "embed_probes.csv"
    d = pd.read_csv(p).drop(columns=["subject_probe_bacc"]); d.to_csv(p, index=False)
    assert agg.run(root / "o", root, "manipulation") == 1


@pytest.mark.parametrize("name", ["results_kc23_d6_sanity_check", "results_kc23_d6_manipulation_check",
                                  "results_kc23_d6_outcome_check"])
def test_stats_with_no_inputs_fails_closed_with_no_letter(tmp_path, name):
    out = tmp_path / name
    assert st.run(out) == 20
    assert not LETTER_RE.search((out / "D6_VERDICT.md").read_text())


def test_stats_unknown_directory_is_an_error_not_a_permissive_default(tmp_path):
    assert st.run(tmp_path / "somewhere") == 20


def test_sanity_dir_end_to_end(tmp_path):
    make_arm(tmp_path, "adv_marginal", 0, 42, f1=0.83, dom=0.9, subj=0.8, sil=0.1, cprobe=0.9)
    out = tmp_path / "results_kc23_d6_sanity_check"
    assert agg.run(out, tmp_path, "sanity") == 0
    assert st.run(out) == 0 and "sanity: PASS" in (out / "D6_VERDICT.md").read_text()


def test_sanity_dir_drift_escalates(tmp_path):
    make_arm(tmp_path, "adv_marginal", 0, 42, f1=0.70, dom=0.9, subj=0.8, sil=0.1, cprobe=0.9)
    out = tmp_path / "results_kc23_d6_sanity_check"
    agg.run(out, tmp_path, "sanity")
    assert st.run(out) == 20 and "sanity: FAIL" in (out / "D6_VERDICT.md").read_text()


def test_sanity_has_no_fallback_to_nonzero_lambda_rows(tmp_path):
    make_arm(tmp_path, "adv_marginal", 0, 42, f1=0.83, dom=0.9, subj=0.8, sil=0.1, cprobe=0.9)
    p = tmp_path / agg.FAMILY_DIR["adv_marginal"](0, 42) / "adv_subjectwise.csv"
    d = pd.read_csv(p); d["adv_lambda"] = 1.0; d.to_csv(p, index=False)
    assert agg.run(tmp_path / "o", tmp_path, "sanity") == 1
