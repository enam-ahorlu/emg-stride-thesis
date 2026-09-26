"""Tests for kc23_d5_replication_stats.py (E-R/E-N, the D-6c directions, fail
closed) and kc23_d5_aggregate.py (the producer of the two CSVs the stats
script reads, which did not exist before 2026-09-25)."""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_d5_replication_stats import classify_er, EXPECTED_SIGN, run as stats_run
import kc23_d5_aggregate as agg

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")


def test_er_strong_agreement():
    diffs = np.array([1, 2, 3, 1, 0.5, 2, 1, -0.2, 1.5, 0.8])  # 9/10 positive
    letter, d = classify_er(diffs, expected_sign=1)
    assert letter == "E-R", d


def test_en_weak_agreement():
    diffs = np.array([1, -2, -3, 1, -0.5, -2, 1, -0.2, -1.5, 0.8])  # only 4/10 positive
    letter, d = classify_er(diffs, expected_sign=1)
    assert letter == "E-N", d


def test_en_wrong_direction_mean():
    diffs = np.array([-1, -2, -3, -1, -0.5, -2, -1, -0.2, -1.5, -0.8])  # mean negative, expected positive
    letter, d = classify_er(diffs, expected_sign=1)
    assert letter == "E-N", d


def test_er_negative_expected():
    diffs = np.array([-1, -2, -3, -1, -0.5, -2, -1, 0.2, -1.5, -0.8])  # 9/10 negative, expected negative
    letter, d = classify_er(diffs, expected_sign=-1)
    assert letter == "E-R", d


def test_d6c_directions_are_all_positive():
    # D-6c: gain jitter AHEAD of channel dropout (+), permutation REDUCTION under channel dropout (+)
    assert EXPECTED_SIGN == {"chandrop_gain": 1, "gainjitter_vs_chandrop": 1,
                             "occlusion_reduction": 1, "permutation_reduction": 1}


# --------------------------------------------------------------------------- synthetic 15-run root
SEEDS = agg.SEEDS
N = 10


def _write_run(root: Path, arm: str, seed: int, f1: np.ndarray, occ: float, perm: float, aug=None):
    d = root / f"results_kc23_d5_{arm}_s{seed}"
    (d / "instr").mkdir(parents=True, exist_ok=True)
    (d / "run_config.json").write_text(json.dumps(
        {"args": {"augmentation": aug or agg.ARMS[arm], "seed": seed}}), encoding="utf-8")
    subs = np.arange(101, 101 + N)
    pd.DataFrame({"subject": subs, "arch": "resnet_se", "f1_macro": f1, "bal_acc": f1}).to_csv(
        d / "cnn_arch_subjectwise.csv", index=False)
    pd.DataFrame([{"subject": s, "channel": c, "f1_full": 0.6, "f1_occluded": 0.5, "drop_pp": occ}
                  for s in subs for c in range(3)]).to_csv(d / "instr" / "occlusion.csv", index=False)
    pd.DataFrame([{"subject": s, "channel": c, "r": 5, "f1_drop_mean": perm, "f1_drop_sd": 0.01}
                  for s in subs for c in range(3)]).to_csv(d / "instr" / "permutation.csv", index=False)


def _make_root(root: Path, *, gj_delta=0.005, perm_e2=0.10):
    rng = np.random.default_rng(3)
    base = 0.55 + rng.normal(0, 0.02, N)
    for seed in SEEDS:
        _write_run(root, "e1", seed, base, occ=6.0, perm=0.10)
        _write_run(root, "e2", seed, base + 0.05, occ=3.0, perm=perm_e2)
        _write_run(root, "e3", seed, base + 0.05 + gj_delta, occ=5.5, perm=0.10)


@pytest.fixture
def root(tmp_path):
    _make_root(tmp_path)
    return tmp_path


def test_aggregate_then_stats_end_to_end(root):
    out = root / "d5_out"
    assert agg.run(root, out) == 0
    for f in ("d5_finding_diffs.csv", "d5_resnet_vs_svm.csv", "d5_magnitudes.csv", "d5_arm_seed_means.csv"):
        assert (out / f).exists()
    assert stats_run(out) == 0
    text = (out / "D5_VERDICT.md").read_text()
    letters = dict(re.findall(r"\*\*(\w+): (E-[RN])\*\*", text))
    assert letters["chandrop_gain"] == "E-R"                # +5 pt on 10/10
    assert letters["occlusion_reduction"] == "E-R"          # 6.0 -> 3.0 on 10/10
    assert letters["permutation_reduction"] == "E-N"        # no reduction
    assert "2.00x" in text                                   # magnitude printed beside the letter
    assert "six-fold" in text and "must not" in text
    assert "0.6" in text                                     # ResNet-SE+CD vs SVM number


def test_gainjitter_ahead_is_positive_direction(tmp_path):
    # gain jitter clearly AHEAD of channel dropout on every subject -> E-R under the D-6c direction
    _make_root(tmp_path, gj_delta=0.02)
    out = tmp_path / "o"
    agg.run(tmp_path, out)
    stats_run(out)
    assert "**gainjitter_vs_chandrop: E-R**" in (out / "D5_VERDICT.md").read_text()


def test_permutation_reduction_positive_when_e2_lower(tmp_path):
    _make_root(tmp_path, perm_e2=0.05)
    out = tmp_path / "o"
    agg.run(tmp_path, out)
    stats_run(out)
    assert "**permutation_reduction: E-R**" in (out / "D5_VERDICT.md").read_text()


# --------------------------------------------------------------------------- fail closed
def test_stats_with_no_inputs_exits_20_and_writes_no_letter(tmp_path):
    rc = stats_run(tmp_path / "empty")
    assert rc == 20
    v = (tmp_path / "empty" / "D5_VERDICT.md").read_text()
    assert not LETTER_RE.search(v)


@pytest.mark.parametrize("missing", ["d5_finding_diffs.csv", "d5_resnet_vs_svm.csv", "d5_magnitudes.csv"])
def test_stats_each_input_missing_fails(root, missing):
    out = root / "o"
    agg.run(root, out)
    (out / missing).unlink()
    assert stats_run(out) == 20
    assert not LETTER_RE.search((out / "D5_VERDICT.md").read_text())


def test_stats_stale_verdict_is_replaced_not_kept(root):
    out = root / "o"
    agg.run(root, out)
    assert stats_run(out) == 0
    (out / "d5_finding_diffs.csv").unlink()
    assert stats_run(out) == 20
    assert not LETTER_RE.search((out / "D5_VERDICT.md").read_text())   # the earlier real verdict is gone


def test_stats_missing_finding_fails(root):
    out = root / "o"
    agg.run(root, out)
    d = pd.read_csv(out / "d5_finding_diffs.csv")
    d[d["finding"] != "permutation_reduction"].to_csv(out / "d5_finding_diffs.csv", index=False)
    assert stats_run(out) == 20


def test_stats_wrong_subject_count_fails(root):
    out = root / "o"
    agg.run(root, out)
    d = pd.read_csv(out / "d5_finding_diffs.csv")
    d.drop(d[d["finding"] == "chandrop_gain"].index[:1]).to_csv(out / "d5_finding_diffs.csv", index=False)
    assert stats_run(out) == 20


def test_stats_csv_sign_disagreeing_with_encoded_direction_fails(root):
    out = root / "o"
    agg.run(root, out)
    d = pd.read_csv(out / "d5_finding_diffs.csv")
    d.loc[d["finding"] == "gainjitter_vs_chandrop", "expected_sign"] = -1
    d.to_csv(out / "d5_finding_diffs.csv", index=False)
    assert stats_run(out) == 20


@pytest.mark.parametrize("victim", ["cnn_arch_subjectwise.csv", "instr/occlusion.csv", "instr/permutation.csv",
                                    "run_config.json"])
def test_aggregate_each_input_missing_fails_and_writes_nothing(root, victim):
    (root / "results_kc23_d5_e2_s7" / victim).unlink()
    out = root / "o"
    assert agg.run(root, out) == 1
    assert not out.exists() or not list(out.glob("*.csv"))


def test_aggregate_missing_run_dir_fails(root):
    shutil.rmtree(root / "results_kc23_d5_e3_s2026")
    assert agg.run(root, root / "o") == 1


def test_aggregate_mislabelled_run_fails(root):
    (root / "results_kc23_d5_e2_s42" / "run_config.json").write_text(
        json.dumps({"args": {"augmentation": "gainjitter", "seed": 42}}), encoding="utf-8")
    assert agg.run(root, root / "o") == 1


def test_aggregate_wrong_subject_count_fails(root):
    p = root / "results_kc23_d5_e1_s123" / "cnn_arch_subjectwise.csv"
    pd.read_csv(p).iloc[:6].to_csv(p, index=False)
    assert agg.run(root, root / "o") == 1


def test_aggregate_subject_set_mismatch_fails(root):
    p = root / "results_kc23_d5_e1_s123" / "cnn_arch_subjectwise.csv"
    d = pd.read_csv(p); d.loc[0, "subject"] = 999
    d.to_csv(p, index=False)
    assert agg.run(root, root / "o") == 1


def test_aggregate_nan_f1_fails(root):
    p = root / "results_kc23_d5_e1_s7" / "cnn_arch_subjectwise.csv"
    d = pd.read_csv(p); d.loc[2, "f1_macro"] = np.nan
    d.to_csv(p, index=False)
    assert agg.run(root, root / "o") == 1
