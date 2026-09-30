"""KC-D6 mechanism test (ADV-C against ADV at the ADV collapse lambda_max), the secondary contrasts and the through-line
matrix: aggregator and stats on real-format run directories, fail closed."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
import kc23_d6_aggregate as agg
import kc23_d6_stats as st
from kc23_fixtures import make_run
from test_kc23_d6 import make_arm, N

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
SEEDS = [42, 7, 123]
ADV_F1 = [0.80, 0.84, 0.86, 0.85, 0.83, 0.78, 0.70]     # knobs 0, .03, .1, .3, 1, 3, 10: peak at 0.1, collapse at 1 (0.83 <= 0.84), next 3
KNOBS = agg.FAMILY_KNOBS["adv_marginal"]


def build_adv(root, seeds=SEEDS, f1s=ADV_F1, within=0.5):
    for sd in seeds:
        for i, k in enumerate(KNOBS):
            make_arm(root, "adv_marginal", k, sd, f1=f1s[i], dom=0.9 - 0.1 * i, subj=0.9 - 0.1 * i, sil=0.1, cprobe=0.9, within=within)


def make_mech_arm(root, kind, knob, seed, *, f1, within, oracle=None, mode=None, dom=0.3, subj=0.3, loss=0.5):
    d = root / agg.MECH_DIR[kind](knob, seed)
    (d / "instr").mkdir(parents=True, exist_ok=True)
    m, o = agg.MECH_MODE[kind]
    oracle = o if oracle is None else oracle
    mode = m if mode is None else mode
    cfg = {"seed": seed, "arch": "resnet_se", "augmentation": "chandrop", "epochs": 40, "batch": 256, "norm_mode": "global",
           "adv_mode": mode, "adv_lambda": knob, "oracle_target_labels": oracle}
    (d / "run_config.json").write_text(json.dumps({"args": cfg}), encoding="utf-8")
    subs = list(range(1, N + 1))
    jit = lambda s, k: 0.002 * (((s * 5 + k) % 7) - 3) / 3
    pd.DataFrame({"subject": subs, "adv_lambda": knob, "oracle": oracle, "f1_macro": [f1 + jit(s, 1) for s in subs]}).to_csv(
        d / "adv_subjectwise.csv", index=False)
    pd.DataFrame({"subject": subs, "domain_probe_bacc": [dom + jit(s, 2) for s in subs], "class_probe_tgt_bacc": 0.9,
                  "feat_norm_src": 7.0}).to_csv(d / "alignment_subjectwise.csv", index=False)
    pd.DataFrame({"subject": subs, "subject_probe_bacc": [subj + jit(s, 4) for s in subs],
                  "held_out_class_silhouette": 0.1, "held_out_class_probe_bacc": 0.9,
                  "subject_probe_within_class_bacc": [within + jit(s, 7) for s in subs], "n_within_class_probes": 4}).to_csv(
        d / "instr" / "embed_probes.csv", index=False)
    pd.DataFrame([{"epoch": e, "val_loss": loss + (3 - e) * 0.05, "best": int(e == 3), "subject": s}
                  for s in subs for e in (1, 2, 3)]).to_csv(d / "training_log.csv", index=False)


def build_advc(root, f1_at_collapse, f1_next, within, seeds=SEEDS, kind="advc"):
    for sd in seeds:
        make_mech_arm(root, kind, 1, sd, f1=f1_at_collapse, within=within)
        make_mech_arm(root, kind, 3, sd, f1=f1_next, within=within)


def run_mech(root):
    out = root / "results/kc23_d6_mechanism_check"
    rc_a = agg.run(out, root, "mechanism")
    return out, rc_a


# ---------------------------------------------------------------- collapse point
def test_collapse_point_is_the_first_value_above_the_peak_two_points_below_it():
    mean = dict(zip(KNOBS, ADV_F1))
    cp = agg.collapse_point(mean, KNOBS)
    assert cp["peak_knob"] == 0.1 and cp["collapse_knob"] == 1 and cp["next_knob"] == 3


def test_a_low_value_below_the_peak_is_not_a_collapse():
    mean = dict(zip(KNOBS, [0.70, 0.80, 0.86, 0.86, 0.85, 0.855, 0.85]))        # lambda 0 is 16 pts below the peak: rising, not collapsing
    assert agg.collapse_point(mean, KNOBS)["collapse_knob"] is None


def test_exactly_two_points_below_counts_and_the_last_value_has_no_next():
    mean = dict(zip(KNOBS, [0.80, 0.82, 0.84, 0.86, 0.86, 0.86, 0.84]))
    cp = agg.collapse_point(mean, KNOBS)
    assert cp["collapse_knob"] == 10 and cp["next_knob"] is None


def test_the_directory_names_use_the_grid_objects_not_floats(tmp_path):
    assert agg._orig_knob("adv_marginal", 10.0) == 10 and str(agg._orig_knob("adv_marginal", 10.0)) == "10"
    assert agg.MECH_DIR["advc"](agg._orig_knob("adv_marginal", 3.0), 42) == "results/kc23_d6_advc_l3_s42"


# ---------------------------------------------------------------- mechanism letters
def test_c_m1_conditional_alignment_keeps_class_structure(tmp_path):
    build_adv(tmp_path)
    build_advc(tmp_path, f1_at_collapse=0.86, f1_next=0.85, within=0.30)         # ADV-C at the peak, more invariant inside classes
    out, rc = run_mech(tmp_path)
    assert rc == 0
    assert st.run(out) == 0
    v = (out / "D6_VERDICT.md").read_text()
    assert "**mechanism: C-M1**" in v and "oracle target labels" in v and "diagnostic only, never deployable" in v
    m = pd.read_csv(out / "D6_mechanism.csv")
    assert list(m["lambda_max"]) == [1.0, 3.0] and m.loc[0, "advc_minus_adv_pp"] > 2 and m.loc[0, "wilcoxon_p"] < 0.05


def test_c_m2_advc_collapses_like_adv(tmp_path):
    build_adv(tmp_path)
    build_advc(tmp_path, f1_at_collapse=0.832, f1_next=0.78, within=0.30)         # within 1 pt of ADV at the collapse value
    out, rc = run_mech(tmp_path)
    assert rc == 0 and st.run(out) == 10
    assert "**mechanism: C-M2**" in (out / "D6_VERDICT.md").read_text()


def test_c_m3_in_between(tmp_path):
    build_adv(tmp_path)
    build_advc(tmp_path, f1_at_collapse=0.845, f1_next=0.84, within=0.30)         # +1.5 pt over ADV at the collapse value: neither M1 nor M2
    out, _ = run_mech(tmp_path)
    assert st.run(out) == 10 and "**mechanism: C-M3**" in (out / "D6_VERDICT.md").read_text()


def test_c_m1_needs_advc_to_be_at_least_as_invariant_inside_classes(tmp_path):
    build_adv(tmp_path, within=0.30)
    build_advc(tmp_path, f1_at_collapse=0.86, f1_next=0.85, within=0.60)          # F1 fine, but MORE subject-identifiable inside classes
    out, _ = run_mech(tmp_path)
    st.run(out)
    assert "**mechanism: C-M1**" not in (out / "D6_VERDICT.md").read_text()


def test_no_collapse_means_advc_is_not_run_and_that_is_reported(tmp_path):
    build_adv(tmp_path, f1s=[0.80, 0.82, 0.84, 0.85, 0.86, 0.86, 0.855])
    out, rc = run_mech(tmp_path)
    assert rc == 0 and pd.read_csv(out / "d6_mechanism_meta.csv")["not_run"].iloc[0] and not (out / "d6_mechanism_arms.csv").exists()
    assert st.run(out) == 10
    v = (out / "D6_VERDICT.md").read_text()
    assert "**mechanism: C-NOT-RUN**" in v and "ADV-C is not run" in v and LETTER_RE.search(v)


# ---------------------------------------------------------------- fail closed
@pytest.mark.parametrize("victim", ["results/kc23_d6_advc_l1_s7", "results/kc23_d6_advc_l3_s123"])
def test_a_missing_advc_run_fails_and_writes_nothing(tmp_path, victim):
    import shutil
    build_adv(tmp_path)
    build_advc(tmp_path, 0.86, 0.85, 0.3)
    shutil.rmtree(tmp_path / victim)
    out, rc = run_mech(tmp_path)
    assert rc == 1 and not list(out.glob("*.csv"))


def test_an_advc_arm_without_the_oracle_flag_is_refused(tmp_path):
    build_adv(tmp_path)
    build_advc(tmp_path, 0.86, 0.85, 0.3)
    make_mech_arm(tmp_path, "advc", 1, 42, f1=0.86, within=0.3, oracle=False)
    assert run_mech(tmp_path)[1] == 1


def test_an_advc_arm_with_the_wrong_adversary_mode_is_refused(tmp_path):
    build_adv(tmp_path)
    build_advc(tmp_path, 0.86, 0.85, 0.3)
    make_mech_arm(tmp_path, "advc", 3, 7, f1=0.85, within=0.3, mode="marginal")
    assert run_mech(tmp_path)[1] == 1


def test_adv_arms_without_the_within_class_probe_cannot_feed_the_mechanism_test(tmp_path):
    build_adv(tmp_path, within=None)
    build_advc(tmp_path, 0.86, 0.85, 0.3)
    assert run_mech(tmp_path)[1] == 1


def test_a_diverged_adv_arm_without_its_retry_blocks_the_mechanism_check(tmp_path):
    for sd in SEEDS:
        for i, k in enumerate(KNOBS):
            make_arm(tmp_path, "adv_marginal", k, sd, f1=ADV_F1[i], dom=0.5, subj=0.5, sil=0.1, cprobe=0.9, within=0.5,
                     loss=0.5 if i < 6 else 1.7)
    build_advc(tmp_path, 0.86, 0.85, 0.3)
    out, rc = run_mech(tmp_path)
    assert rc == 1 and not list(out.glob("*.csv"))


def test_mechanism_stats_with_no_inputs_fails_closed_with_no_letter(tmp_path):
    out = tmp_path / "results/kc23_d6_mechanism_check"
    assert st.run(out) == 20 and not LETTER_RE.search((out / "D6_VERDICT.md").read_text())


# ---------------------------------------------------------------- ADV-CDAN (optional)
def test_cdan_rows_are_tabulated_beside_advc_when_present_and_carry_no_letter(tmp_path):
    build_adv(tmp_path)
    build_advc(tmp_path, 0.86, 0.85, 0.3)
    build_advc(tmp_path, 0.84, 0.83, 0.32, kind="cdan")
    out, rc = run_mech(tmp_path)
    assert rc == 0 and st.run(out) == 0
    v = (out / "D6_VERDICT.md").read_text()
    assert "ADV-CDAN (deployable, no target labels; no letter)" in v
    assert "**mechanism: C-M1**" in v


def test_a_partly_run_cdan_is_an_error_not_a_silent_omission(tmp_path):
    import shutil
    build_adv(tmp_path)
    build_advc(tmp_path, 0.86, 0.85, 0.3)
    build_advc(tmp_path, 0.84, 0.83, 0.32, kind="cdan")
    shutil.rmtree(tmp_path / "results/kc23_d6_cdan_l3_s7")
    assert run_mech(tmp_path)[1] == 1


def test_a_cdan_arm_must_not_carry_the_oracle_flag(tmp_path):
    build_adv(tmp_path)
    build_advc(tmp_path, 0.86, 0.85, 0.3)
    build_advc(tmp_path, 0.84, 0.83, 0.32, kind="cdan")
    make_mech_arm(tmp_path, "cdan", 1, 42, f1=0.84, within=0.3, oracle=True)
    assert run_mech(tmp_path)[1] == 1


# ---------------------------------------------------------------- secondary contrasts
def build_d1_refs(root, seeds, r2=0.8395, r10=0.82, r11=0.826):
    for sd in seeds:
        make_run(root, f"results/kc23_d1_r2_s{sd}", augmentation="chandrop", seed=sd, f1=r2, chandrop_p=0.2, instrumented=False)
        d = root / f"results/kc23_d1_r10_s{sd}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"subject": range(1, N + 1), "f1_macro": r10, "f1_pre_adabn": r10 - 0.05}).to_csv(d / "adabn_subjectwise.csv", index=False)
        d = root / f"results/kc23_d1_r11_s{sd}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"subject": range(1, N + 1), "f1_macro": r11}).to_csv(d / "deep_coral_subjectwise.csv", index=False)


def build_advps(root, seeds, f1=0.85):
    for sd in seeds:
        for k in agg.FAMILY_KNOBS["advps"]:
            make_arm(root, "advps", k, sd, f1=f1 - 0.005 * k, dom=0.5, subj=0.5, sil=0.1, cprobe=0.9)


def test_secondary_contrasts_end_to_end(tmp_path):
    build_adv(tmp_path)
    build_advps(tmp_path, SEEDS)
    build_d1_refs(tmp_path, SEEDS)
    out = tmp_path / "results/kc23_d6_secondary_check"
    assert agg.run(out, tmp_path, "secondary") == 0
    df = pd.read_csv(out / "d6_secondary.csv")
    assert set(df["contrast"]) == {"ADV(best lambda=0.1)-R2", "ADV(best lambda=0.1)-R10", "ADV(best lambda=0.1)-R11",
                                   "ADV-PS(lambda=0.1)-R2", "ADV-PS(lambda=1)-R2", "ADV-PS(lambda=10)-R2"}
    assert st.run(out) == 0
    v = (out / "D6_VERDICT.md").read_text()
    assert "**secondary: reported**" in v and "batch differs" in v
    t = pd.read_csv(out / "D6_secondary_contrasts.csv").set_index("contrast")
    assert t.loc["ADV(best lambda=0.1)-R2", "mean_diff_pp"] == pytest.approx((0.86 - 0.8395) * 100, abs=0.3)
    assert (t["n_realizations"] == 3).all()


def test_secondary_uses_the_seeds_that_exist_and_says_how_many(tmp_path):
    build_adv(tmp_path, seeds=[42])
    build_advps(tmp_path, [42])
    build_d1_refs(tmp_path, [42])
    out = tmp_path / "results/kc23_d6_secondary_check"
    assert agg.run(out, tmp_path, "secondary") == 0 and st.run(out) == 0
    assert (pd.read_csv(out / "D6_secondary_contrasts.csv")["n_realizations"] == 1).all()


def test_secondary_needs_seed_42_of_both_families_and_the_d1_references(tmp_path):
    build_adv(tmp_path, seeds=[42])
    assert agg.run(tmp_path / "o", tmp_path, "secondary") == 1                 # no ADV-PS
    build_advps(tmp_path, [42])
    assert agg.run(tmp_path / "o", tmp_path, "secondary") == 1                 # no D1 references
    assert not list((tmp_path / "o").glob("*.csv"))


# ---------------------------------------------------------------- the combined through-line matrix
def _outcome_dir(tmp_path):
    from test_kc23_d6 import build_all
    build_all(tmp_path, [42, 7, 123])
    man = tmp_path / "results/kc23_d6_manipulation_check"
    agg.run(man, tmp_path, "manipulation")
    st.run(man)
    out = tmp_path / "results/kc23_d6_outcome_check"
    assert agg.run(out, tmp_path, "outcome", man / "D6_gates.csv") == 0
    return out


def test_the_outcome_verdict_ends_with_the_through_line_matrix_and_says_what_is_not_available(tmp_path):
    out = _outcome_dir(tmp_path)
    st.run(out)
    v = (out / "D6_VERDICT.md").read_text()
    assert "## Combined through-line matrix" in v
    assert "| axis | invariance measured | shape shown | mechanism measured | replicated on ENABL3S where run |" in v
    assert "ladder (C2, C6)" in v and "channel (D4)" in v and "learned (D6)" in v
    assert "not available" in v and "not run on ENABL3S" in v


def test_the_matrix_cells_carry_the_letters_the_other_stages_produced(tmp_path):
    out = _outcome_dir(tmp_path)
    for rel, text in (("results/kc23_c2_whitening_w400/C2_VERDICT.md", "**Endpoint 1 outcome: W2**\n**Endpoint 2 (mechanism) outcome: M1**\n"),
                      ("results/kc23_d4_stats/D4_VERDICT.md", "**Outcome: T1**\n"),
                      ("results/kc23_c6_ladder_enabl3s/C6_VERDICT.md", "**Outcome: R1**\n"),
                      ("results/kc23_d5_stats/D5_VERDICT.md", "- **chandrop_gain: E-R**\n")):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# v\n\n" + text, encoding="utf-8")
    st.run(out)
    v = (out / "D6_VERDICT.md").read_text()
    matrix = v.split("## Combined through-line matrix")[1]
    assert "Endpoint 1 outcome: W2" in matrix and "Endpoint 2 (mechanism) outcome: M1" in matrix
    assert "Outcome: T1" in matrix and "Outcome: R1" in matrix and "chandrop_gain: E-R" in matrix
