"""KC-D2 and KC-D3 aggregators and stats (26 September 2026), on real-format run directories. D2: per-subject SUMMED
drops with negative drops kept, the reduction factor computed PER REALIZATION with its mean and SD, the paired Wilcoxon on
realization-averaged sums, and the O-R/O-T/O-M letters. D3: realization means of X1 to X4 against R1 and R3, and the
A1 to A4 letters."""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kc23_d2_aggregate as d2a
import kc23_d2_reliance_stats as d2s
import kc23_d3_aggregate as d3a
import kc23_d3_axis_stats as d3s
from kc23_fixtures import make_run
from kc23_run_loader import load_run, reduction_factor

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")


def _subj(base, k=0.02):
    return lambda s: base * (1 + k * (((s * 3) % 5) - 2) / 2)


# ----------------------------------------------------------------------------------- D2
def build_d2(root: Path, occ, att, perm):
    """occ/att/perm: {arm: per-channel value}. R1..R5 x 4 seeds."""
    for arm, (pattern, arch, aug, kw) in d2a.ARMS.items():
        for seed in d2a.SEEDS:
            make_run(root, pattern.format(seed=seed), arch=arch, augmentation=aug, seed=seed,
                     occ=_subj(occ[arm] * (1 + (0.0 if arm in ('R1', 'R4') else 0.12 * ((seed % 3) - 1)))), att=_subj(att[arm]),
                     perm=_subj(perm[arm]), chandrop_p=kw.get("chandrop_p", 0.2), gain_sd=kw.get("gain_sd", 0.4))


OCC_R = {"R1": 6.0, "R2": 3.0, "R3": 1.5, "R4": 6.0, "R5": 3.0}         # gain jitter 4-fold, channel dropout 2-fold
ATT_R = {"R1": 2.0, "R2": 1.0, "R3": 1.0, "R4": 2.0, "R5": 1.0}
PERM_R = {"R1": 0.10, "R2": 0.04, "R3": 0.09, "R4": 0.10, "R5": 0.05}   # permutation reliance falls 2.5-fold


@pytest.fixture
def d2root(tmp_path):
    build_d2(tmp_path, OCC_R, ATT_R, PERM_R)
    return tmp_path


def test_d2_end_to_end_o_r(d2root):
    out = d2root / "results/kc23_d2_reliance"
    assert d2a.run(d2root, out) == 0
    assert len(pd.read_csv(out / "d2_persubject_sums.csv")) == 5 * 5 * 40
    assert d2s.run(out) == 0
    v = (out / "D2_VERDICT.md").read_text()
    assert "**Outcome: O-R**" in v and "Residual caveat" in v
    det = pd.read_csv(out / "D2_detail.csv")
    occ = det[det["comparison"].str.startswith("zeroing occlusion, R3")].iloc[0]
    assert occ["factor_mean"] == pytest.approx(4.0, rel=0.10) and occ["factor_sd"] > 0     # a per-realization SD exists
    assert len(occ["factor_per_realization"].split(";")) == 5


def test_d2_o_t_when_neither_measure_falls(tmp_path):
    build_d2(tmp_path, {**OCC_R, "R3": 5.6}, ATT_R, {**PERM_R, "R2": 0.095})
    out = tmp_path / "o"
    d2a.run(tmp_path, out); d2s.run(out)
    assert "**Outcome: O-T**" in (out / "D2_VERDICT.md").read_text()


def test_d2_o_m_in_between(tmp_path):
    build_d2(tmp_path, {**OCC_R, "R3": 1.5}, ATT_R, {**PERM_R, "R2": 0.09})      # occlusion 4x but permutation 1.1x
    out = tmp_path / "o"
    d2a.run(tmp_path, out); d2s.run(out)
    assert "**Outcome: O-M**" in (out / "D2_VERDICT.md").read_text()


def test_d2_factor_is_computed_per_realization_not_from_pooled_data(tmp_path):
    # realizations with very different scales: the mean of per-realization factors differs from the pooled ratio
    build_d2(tmp_path, OCC_R, ATT_R, PERM_R)
    out = tmp_path / "o"
    d2a.run(tmp_path, out)
    df = pd.read_csv(out / "d2_persubject_sums.csv")
    pooled = df[df.arm == "R1"].occlusion_sum.mean() / df[df.arm == "R3"].occlusion_sum.mean()
    det = d2s.compare(df, "R3", "R1", "occlusion_sum")
    per = [float(x) for x in det["factor_per_realization"].split(";")]
    assert det["factor_mean"] == pytest.approx(np.mean(per), abs=2e-3) and len(per) == 5
    assert det["factor_sd"] == pytest.approx(np.std(per, ddof=1), abs=2e-3)
    assert abs(det["factor_mean"] - pooled) > 1e-6          # they are different quantities


def test_d2_negative_drops_are_kept_not_clipped(tmp_path):
    make_run(tmp_path, "r", occ=lambda s: -2.0 if s == 1 else 3.0, seed=42)
    r = load_run(tmp_path / "r", augmentation="none", seed=42, chandrop_p=0.2)
    assert r["occlusion_sum"][1] == pytest.approx(-2.0 * 9)          # summed over the 9 channels, negative kept
    assert r["occlusion_sum"][2] == pytest.approx(27.0)


def test_d2_attenuation_uses_alpha_half_only(tmp_path):
    make_run(tmp_path, "r", att=lambda s: 4.0, seed=42)              # fixture: drop_pp = att * (1 - alpha)
    r = load_run(tmp_path / "r", augmentation="none", seed=42, chandrop_p=0.2)
    assert r["attenuation_sum"][3] == pytest.approx(4.0 * 0.5 * 9)


def test_d2_permutation_is_in_percentage_points(tmp_path):
    make_run(tmp_path, "r", perm=lambda s: 0.10, seed=42)
    r = load_run(tmp_path / "r", augmentation="none", seed=42, chandrop_p=0.2)
    assert r["permutation_sum"][5] == pytest.approx(0.10 * 100 * 9)


def test_reduction_factor_is_mean_over_mean_and_infinite_when_the_arm_is_not_positive():
    a, b = pd.Series([6.0, 6.0]), pd.Series([2.0, 4.0])
    assert reduction_factor(a, b) == pytest.approx(2.0)
    assert reduction_factor(a, pd.Series([0.0, -1.0])) == float("inf")


@pytest.mark.parametrize("victim", ["results/kc23_d1_r3_s7", "results/kc23_d1_r5_s1001"])
def test_d2_aggregate_missing_run_fails_and_writes_nothing(d2root, victim):
    shutil.rmtree(d2root / victim)
    out = d2root / "o"
    assert d2a.run(d2root, out) == 1 and not out.exists()


@pytest.mark.parametrize("victim", ["instr/attenuation.csv", "instr/permutation.csv", "instr/occlusion.csv", "run_config.json"])
def test_d2_aggregate_missing_instrumentation_fails(d2root, victim):
    (d2root / "results/kc23_d1_r2_s42" / victim).unlink()
    assert d2a.run(d2root, d2root / "o") == 1


def test_d2_aggregate_mislabelled_arm_fails(d2root):
    p = d2root / "results/kc23_d1_r4_s42" / "run_config.json"
    cfg = json.loads(p.read_text()); cfg["args"]["arch"] = "resnet_se"; p.write_text(json.dumps(cfg))
    assert d2a.run(d2root, d2root / "o") == 1


def test_d2_stats_missing_or_incomplete_input_fails_with_no_letter(d2root):
    out = d2root / "o"
    d2a.run(d2root, out)
    d = pd.read_csv(out / "d2_persubject_sums.csv")
    d[~((d.arm == "R3") & (d.realization == 7) & (d.subject == 3))].to_csv(out / "d2_persubject_sums.csv", index=False)
    assert d2s.run(out) == 20 and not LETTER_RE.search((out / "D2_VERDICT.md").read_text())
    (out / "d2_persubject_sums.csv").unlink()
    assert d2s.run(out) == 20 and not LETTER_RE.search((out / "D2_VERDICT.md").read_text())


def test_d2_classify_o_unchanged_from_the_plan():
    assert d2s.classify_o(3.0, 2.0) == "O-R" and d2s.classify_o(2.99, 2.0) == "O-M"
    assert d2s.classify_o(1.49, 1.49) == "O-T" and d2s.classify_o(1.5, 1.2) == "O-M"


# ----------------------------------------------------------------------------------- D3
def build_d3(root: Path, means: dict):
    for arm, (pattern, aug, kw) in d3a.ARMS.items():
        for seed in d3a.SEEDS:
            cfg = dict(augmentation=aug, seed=seed, gain_sd=kw.get("gain_sd", 0.4), instrumented=False,
                       f1=_subj(means[arm], 0.01))
            make_run(root, pattern.format(seed=seed), **cfg)
            if arm in d3a.SIGMA:
                p = Path(root) / pattern.format(seed=seed) / "run_config.json"
                c = json.loads(p.read_text()); c["args"]["aug_sigma"] = d3a.SIGMA[arm]; p.write_text(json.dumps(c))


BASE = {"R1": 0.80, "R3": 0.84, "X1": 0.80, "X2": 0.80, "X3": 0.80, "X4": 0.80}


@pytest.fixture
def d3root(tmp_path):
    build_d3(tmp_path, BASE)
    return tmp_path


def test_d3_end_to_end_a1(d3root):
    out = d3root / "results/kc23_d3_stats"
    assert d3a.run(d3root, out) == 0
    m = pd.read_csv(out / "d3_realization_means.csv").set_index("arm")
    assert set(m.index) == set(d3s.ARMS) and (m["n_realizations"] == 3).all()
    assert d3s.run(out) == 0
    v = (out / "D3_VERDICT.md").read_text()
    assert "**Outcome(s): A1**" in v and "X1 (Gaussian sigma 0.10" in v


def test_d3_a2_a3_a4_can_co_occur(tmp_path):
    build_d3(tmp_path, {**BASE, "X2": 0.835, "X3": 0.84, "X4": 0.835})
    out = tmp_path / "o"
    d3a.run(tmp_path, out); d3s.run(out)
    v = (out / "D3_VERDICT.md").read_text()
    assert "A2" in v and "A3" in v and "A4" in v and "A1" not in v.split("\n\n")[1]


def test_d3_no_letter_fired_is_stated_and_still_a_computed_outcome(tmp_path):
    build_d3(tmp_path, {**BASE, "X2": 0.82, "X3": 0.825, "X4": 0.80})
    out = tmp_path / "o"
    d3a.run(tmp_path, out)
    assert d3s.run(out) == 0
    v = (out / "D3_VERDICT.md").read_text()
    assert "**Outcome(s): none fired**" in v and LETTER_RE.search(v)


@pytest.mark.parametrize("victim", ["results/kc23_d3_x3_s7", "results/kc23_d1_r3_s123", "results/kc23_d3_x1_s42"])
def test_d3_aggregate_missing_run_fails(d3root, victim):
    shutil.rmtree(d3root / victim)
    out = d3root / "o"
    assert d3a.run(d3root, out) == 1 and not out.exists()


def test_d3_aggregate_mislabelled_sigma_fails(d3root):
    p = d3root / "results/kc23_d3_x2_s42" / "run_config.json"
    c = json.loads(p.read_text()); c["args"]["aug_sigma"] = 0.1; p.write_text(json.dumps(c))
    assert d3a.run(d3root, d3root / "o") == 1


def test_d3_aggregate_mislabelled_augmentation_fails(d3root):
    p = d3root / "results/kc23_d3_x4_s123" / "run_config.json"
    c = json.loads(p.read_text()); c["args"]["augmentation"] = "chanoffset"; p.write_text(json.dumps(c))
    assert d3a.run(d3root, d3root / "o") == 1


def test_d3_stats_missing_or_wrong_arms_fails_with_no_letter(d3root):
    out = d3root / "o"
    d3a.run(d3root, out)
    m = pd.read_csv(out / "d3_realization_means.csv")
    m[m.arm != "X4"].to_csv(out / "d3_realization_means.csv", index=False)
    assert d3s.run(out) == 20 and not LETTER_RE.search((out / "D3_VERDICT.md").read_text())
    (out / "d3_realization_means.csv").unlink()
    assert d3s.run(out) == 20


def test_d3_classify_a_boundaries_unchanged_from_the_plan():
    assert d3s.classify_a(0.80, 0.84, 0.80, 0.81, 0.822, 0.822) == ["A1"]           # X2 <= R1+1, X3,X4 < R3-1.5
    assert "A2" in d3s.classify_a(0.80, 0.84, 0.80, 0.83, 0.80, 0.80)               # X2 >= R3-1
    assert "A1" not in d3s.classify_a(0.80, 0.84, 0.80, 0.812, 0.80, 0.80)          # X2 above R1+1
