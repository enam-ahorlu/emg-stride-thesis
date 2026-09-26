"""KC-D1 aggregator and stats after the 26 September 2026 conformance pass, on a FULL real-format tree: every arm at every
realization, the per-seed ensembles, the published runs (with the means the plan quotes) and the instrumented arms.
Covers: every registered contrast is produced; C16 is three comparisons (two nulls and a fall); the published run is
a fifth realization only where every arm of a contrast has one; BH runs over the FIXED registered family; the
sensitivity without the published run; the headline arms (ensemble and global included); the run-variance
deliverable; C17; and fail-closed behaviour."""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kc23_d1_aggregate as agg
import kc23_d1_replicate_stats as st
from kc23_fixtures import make_run

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
N = 40
SUBJ = np.arange(1, N + 1)
PERM = np.random.default_rng(3).permutation(N)
DEV = (np.linspace(-1, 1, N) * 0.02)[PERM]           # zero-mean per-subject effects: means are exact


def f1_of(mean, seed_shift=0.0, k=1.0):
    d = dict(zip(SUBJ, DEV * k))
    return lambda s: mean + seed_shift + d[s]


SEED_SHIFT = {42: 0.0, 7: 0.002, 123: -0.002, 1001: 0.001}
BASE = {"R1": 0.782, "R2": 0.8395, "R3": 0.848, "R4": 0.760, "R5": 0.825, "R6": 0.750, "R7": 0.812, "R8": 0.760,
        "R9": 0.760, "R10": 0.818, "R10pre": 0.772, "R11": 0.826, "R12": 0.860, "R13": 0.835, "R14": 0.837, "R15": 0.815,
        "R16": 0.849, "R17": 0.838}
ENS = {"soft": 0.858, "stacking": 0.860}
OCC = {"R1": 6.0, "R2": 3.0, "R4": 6.0, "R5": 3.0}


def build_tree(root: Path, base=None, ens=None, occ=None):
    base = {**BASE, **(base or {})}
    ens = {**ENS, **(ens or {})}
    occ = {**OCC, **(occ or {})}
    for arm, (fname, col, cfg) in agg.ARM_SPEC.items():
        if arm == "R10pre":
            continue
        for seed in agg.arm_seeds(arm):
            name = f"results_kc23_d1_{arm.lower()}_s{seed}"
            d = Path(root) / name
            shift = SEED_SHIFT[seed]
            f = f1_of(base[arm], shift)
            if isinstance(cfg, dict):
                kw = dict(augmentation=cfg["augmentation"], seed=seed, arch=cfg["arch"], f1=f,
                          chandrop_p=cfg.get("chandrop_p", 0.2), gain_sd=cfg.get("gain_sd", 0.4),
                          instrumented=arm in agg.INSTRUMENTED, occ=(lambda s, a=arm: occ[a]) if arm in occ else 5.0,
                          npz="windows_w400_x.npz" if arm == "R12" else "windows_w250_x.npz")
                make_run(root, name, **kw)
            elif arm in ("R8", "R9"):
                d.mkdir(parents=True, exist_ok=True)
                (d / "run_config.json").write_text(json.dumps({"args": {"seed": seed, "norm_mode": "per_subject"}}))
                pd.DataFrame({"model": "CNN", "subject": SUBJ, "f1_macro": [f(s) for s in SUBJ]}).to_csv(
                    d / fname, index=False)
            elif arm == "R10":
                d.mkdir(parents=True, exist_ok=True)
                fp = f1_of(base["R10pre"], shift)
                pd.DataFrame({"subject": SUBJ, "f1_pre_adabn": [fp(s) for s in SUBJ], "f1_macro": [f(s) for s in SUBJ]}
                             ).to_csv(d / fname, index=False)
            elif arm == "R11":
                d.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"subject": SUBJ, "f1_macro": [f(s) for s in SUBJ]}).to_csv(d / fname, index=False)
    for seed in agg.TIER_A_SEEDS:
        d = Path(root) / f"results_kc23_d1_ensemble_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        fs, fk = f1_of(ens["soft"], SEED_SHIFT[seed]), f1_of(ens["stacking"], SEED_SHIFT[seed])
        pd.DataFrame({"subject": SUBJ, "soft": [fs(s) for s in SUBJ], "stacking": [fk(s) for s in SUBJ]}).to_csv(
            d / f"ensemble_s{seed}.csv", index=False)
    # the published runs, with exactly the means the plan quotes
    pub = pd.read_csv(agg.PUBLISHED_RUNS_CSV)
    for _, r in pub.iterrows():
        d = Path(root) / r["directory"]
        d.mkdir(parents=True, exist_ok=True)
        p = d / r["subjectwise_file"]
        cur = pd.read_csv(p) if p.exists() else pd.DataFrame({"subject": SUBJ})
        target = float(r["quoted_value"]) if pd.notna(r["quoted_value"]) else {"R10": 0.818, "ENS_STACK": 0.8605}[r["arm"]]
        cur[r["f1_column"]] = [f1_of(target, 0.0)(s) for s in SUBJ]
        if r["subjectwise_file"] == "cnn_arch_subjectwise.csv":
            cur["arch"] = "resnet_se"
        cur.to_csv(p, index=False)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("d1tree")
    build_tree(root)
    return root


@pytest.fixture(scope="module")
def agg_out(tree):
    out = tree / "results_kc23_d1_stats"
    assert agg.run(out, tree, "full") == 0
    return out


def test_every_registered_contrast_is_produced(agg_out):
    df = pd.read_csv(agg_out / "d1_contrasts.csv")
    assert set(df["contrast"]) == set(agg.ALL_CONTRAST_IDS)
    for c in ("C10", "C11", "C13", "C13b", "C15", "C16a", "C16b", "C16c", "C17a", "C17b"):
        assert c in set(df["contrast"]), c            # the ones the first version left out
    assert set(agg.NULL_CONTRASTS) == {"C15", "C16a", "C16b"} and "C16c" in agg.REGISTERED_BH_FAMILY


def test_realizations_tier_a_tier_b_and_the_published_fifth(agg_out):
    df = pd.read_csv(agg_out / "d1_contrasts.csv")
    real = lambda c: set(df[df.contrast == c].realization.astype(str))
    assert real("C1") == {"42", "7", "123", "1001", "published"}          # R2 and R1 both have a published run
    assert real("C9") == {"42", "7", "123", "1001", "published"}          # R2 and R10(post)
    assert real("C14") == {"42", "7", "123", "1001", "published"}          # R12 and R2
    assert real("C13") == {"42", "7", "123", "1001", "published"}          # ensemble and R2
    assert real("C2") == {"42", "7", "123", "1001"}                        # R3 has no mapped published run
    assert real("C3") == {"42", "7", "123"} and set(df[df.contrast == "C3"].tier) == {"B"}
    assert real("C12") == {"42", "7", "123", "1001", "published"}
    assert set(df[df.contrast == "C16c"].tier) == {"B"}                    # R15 is Tier B


def test_contrast_values(agg_out):
    df = pd.read_csv(agg_out / "d1_contrasts.csv")
    m = lambda c, r=42: df[(df.contrast == c) & (df.realization.astype(str) == str(r))]["diff"].mean()
    assert m("C1") == pytest.approx(BASE["R2"] - BASE["R1"], abs=1e-9)
    assert m("C12") == pytest.approx(BASE["R2"] - agg.PUBLISHED_SVM, abs=1e-9)
    assert m("C13") == pytest.approx(ENS["soft"] - BASE["R2"], abs=1e-9)
    assert m("C13b") == pytest.approx(ENS["stacking"] - ENS["soft"], abs=1e-9)
    assert m("C6") == pytest.approx((BASE["R2"] - BASE["R1"]) - (BASE["R5"] - BASE["R4"]), abs=1e-9)
    assert m("C16a") == pytest.approx(BASE["R13"] - BASE["R2"], abs=1e-9)
    assert m("C16c") == pytest.approx(BASE["R15"] - BASE["R2"], abs=1e-9)
    assert m("C17a") == pytest.approx(9 * (OCC["R1"] - OCC["R2"]), abs=1e-6)   # summed over 9 channels, R1 - R2
    assert m("C9") == pytest.approx(BASE["R2"] - BASE["R10"], abs=1e-9)        # R10 POST, not pre


def test_headline_arms_ensemble_and_global(agg_out):
    h = pd.read_csv(agg_out / "d1_headline_inputs.csv").set_index("arm")
    assert set(h.index) == {"R2", "ensemble", "R12", "global"}
    assert h.loc["global", "realization_mean"] == pytest.approx(BASE["R10pre"] + np.mean(list(SEED_SHIFT.values())))
    assert h.loc["ensemble", "realization_mean"] == pytest.approx(ENS["soft"] + np.mean(list(SEED_SHIFT.values())))
    assert (h["realization_sd"] > 0).all()


def test_c17_factors_are_per_realization(agg_out):
    f = pd.read_csv(agg_out / "d1_c17_factors.csv")
    assert set(f.comparison) == {"R2 against R1", "R5 against R4"} and len(f) == 8
    assert f["factor"].between(1.9, 2.1).all()


def test_stats_end_to_end_letters(agg_out, tree):
    rc = st.run(agg_out, st.required_inputs(agg_out))
    v = (agg_out / "D1_VERDICT.md").read_text()
    assert "**reproduction: PASS**" in v
    res = pd.read_csv(agg_out / "D1_contrasts_verdict.csv").set_index("contrast")
    assert res.loc["C1", "letter"] == "ESTABLISHED" and res.loc["C12", "letter"] == "ESTABLISHED"
    assert res.loc["C15", "letter"] == "EQUIVALENT"                  # R9 - R8 == 0
    assert res.loc["C16a", "letter"] == "EQUIVALENT"                 # R13 - R2 = -0.45 pt: inside the +/-1 pt TOST bound
    assert res.loc["C16b", "letter"] == "EQUIVALENT"                 # R17 - R2 = -0.15 pt
    assert res.loc["C13b", "letter"] == "descriptive"
    assert "headline (global): H1" in v and "headline (ensemble): H1" in v
    assert rc == 0


def test_null_contrasts_use_tost_and_c16c_is_a_difference_test(agg_out):
    res = pd.read_csv(agg_out / "D1_contrasts_verdict.csv").set_index("contrast")
    assert res.loc["C16a", "letter"] in ("EQUIVALENT", "NOT_EQUIVALENT")
    assert res.loc["C16b", "letter"] in ("EQUIVALENT", "NOT_EQUIVALENT")
    assert res.loc["C16c", "letter"] in ("ESTABLISHED", "AMBIGUOUS", "NOT ESTABLISHED")
    assert res.loc["C15", "letter"] in ("EQUIVALENT", "NOT_EQUIVALENT")


def test_bh_runs_over_the_fixed_registered_family_of_17(agg_out):
    res = pd.read_csv(agg_out / "D1_contrasts_verdict.csv").set_index("contrast")
    fam = agg.REGISTERED_BH_FAMILY
    assert len(fam) == 17
    expected = st.benjamini_hochberg([res.loc[c, "p_raw"] for c in fam])
    assert [res.loc[c, "p_bh"] for c in fam] == pytest.approx(expected)
    assert res.loc[list(agg.NULL_CONTRASTS), "p_bh"].isna().all()       # nulls are not in the difference-test family


def test_sensitivity_without_the_published_run_is_reported(agg_out):
    res = pd.read_csv(agg_out / "D1_contrasts_verdict.csv").set_index("contrast")
    assert res.loc["C1", "n_realizations"] == 5 and res.loc["C1", "n_realizations_without_published"] == 4
    assert res.loc["C2", "n_realizations"] == res.loc["C2", "n_realizations_without_published"] == 4
    assert "Sensitivity (published run excluded)" in (agg_out / "D1_VERDICT.md").read_text()


def test_run_variance_deliverable(agg_out):
    pooled = pd.read_csv(agg_out / "D1_run_variance_pooled.csv").iloc[0]
    assert pooled["df"] > 20 and pooled["chi2_lo_pt"] < pooled["pooled_sd_pt"] < pooled["chi2_hi_pt"]
    per = pd.read_csv(agg_out / "D1_run_variance.csv")
    assert {"R1", "R2", "ENS_SOFT"} <= set(per.arm) and (per["sd_of_40fold_mean"] > 0).all()
    assert "Run variance: pooled SD" in (agg_out / "D1_VERDICT.md").read_text()


def test_secondary_model_is_stated_not_silently_dropped(agg_out):
    v = (agg_out / "D1_VERDICT.md").read_text()
    assert "Secondary mixed model" in v and ("NOT computed" in v or "fitted" in v)


def test_c17_factor_is_in_the_verdict(agg_out):
    assert "C17, occlusion reduction factor R2 against R1" in (agg_out / "D1_VERDICT.md").read_text()


# ------------------------------------------------------------------------------------------ escalation and fail closed
def _fresh(tmp_path, **kw):
    build_tree(tmp_path, **kw)
    out = tmp_path / "results_kc23_d1_stats"
    assert agg.run(out, tmp_path, "full") == 0
    return out


def test_c1_not_established_escalates(tmp_path):
    out = _fresh(tmp_path, base={"R2": 0.782})                 # R2 == R1: no effect
    assert st.run(out, st.required_inputs(out)) == 20


def test_headline_h2_escalates(tmp_path):
    out = _fresh(tmp_path, base={"R12": 0.83})                 # R12 realizations far below the published 0.860
    rc = st.run(out, st.required_inputs(out))
    assert rc == 20 and "headline (R12): H2" in (out / "D1_VERDICT.md").read_text()


@pytest.mark.parametrize("victim", ["results_kc23_d1_ensemble_s7", "results_kc23_d1_r17_s123", "results_kc23_d1_r9_s1001",
                                    "results_kc23_d1_r11_s42", "results_kc23_d1_r10_s1001"])
def test_aggregate_missing_input_fails_and_leaves_no_output(tmp_path, victim):
    build_tree(tmp_path)
    shutil.rmtree(tmp_path / victim)
    out = tmp_path / "o"
    assert agg.run(out, tmp_path, "full") == 1
    assert not list(out.glob("*.csv"))


def test_aggregate_instrumentation_missing_for_c17_fails(tmp_path):
    build_tree(tmp_path)
    (tmp_path / "results_kc23_d1_r5_s42" / "instr" / "occlusion.csv").unlink()
    assert agg.run(tmp_path / "o", tmp_path, "full") == 1


def test_aggregate_mislabelled_arm_fails(tmp_path):
    build_tree(tmp_path)
    p = tmp_path / "results_kc23_d1_r15_s7" / "run_config.json"
    cfg = json.loads(p.read_text()); cfg["args"]["aug_chandrop_p"] = 0.2; p.write_text(json.dumps(cfg))
    assert agg.run(tmp_path / "o", tmp_path, "full") == 1


def test_published_run_with_the_wrong_mean_is_refused_not_used(tmp_path):
    build_tree(tmp_path)
    p = tmp_path / "results_cnn_aug_resnet_se_none" / "cnn_arch_subjectwise.csv"
    d = pd.read_csv(p); d["f1_macro"] = d["f1_macro"] - 0.03; d.to_csv(p, index=False)
    assert agg.run(tmp_path / "o", tmp_path, "full") == 1


def test_published_run_missing_fails(tmp_path):
    build_tree(tmp_path)
    shutil.rmtree(tmp_path / "results_win400_cnn_persubj")
    assert agg.run(tmp_path / "o", tmp_path, "full") == 1


@pytest.mark.parametrize("victim", ["d1_contrasts.csv", "d1_headline_inputs.csv", "d1_arm_subject_f1.csv", "d1_c17_factors.csv"])
def test_stats_each_input_missing_fails_closed_with_no_letter(tmp_path, victim):
    out = _fresh(tmp_path)
    (out / victim).unlink()
    assert st.run(out, st.required_inputs(out)) == 20
    assert not LETTER_RE.search((out / "D1_VERDICT.md").read_text())


def test_stats_missing_contrast_rows_fail_closed(tmp_path):
    out = _fresh(tmp_path)
    d = pd.read_csv(out / "d1_contrasts.csv")
    d[d.contrast != "C17b"].to_csv(out / "d1_contrasts.csv", index=False)
    assert st.run(out, st.required_inputs(out)) == 20


def test_repro_mode_needs_only_the_seed_42_triplet(tmp_path):
    build_tree(tmp_path)
    out = tmp_path / "results_kc23_d1_repro_check"
    assert agg.run(out, tmp_path, "repro") == 0
    assert [p.name for p in out.glob("*.csv")] == ["d1_reproduction_inputs.csv"]
    assert st.run(out, st.required_inputs(out)) == 0 and "**reproduction: PASS**" in (out / "D1_VERDICT.md").read_text()
