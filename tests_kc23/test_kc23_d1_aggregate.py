"""Synthetic tests for kc23_d1_aggregate.py: the missing raw-to-aggregate
extractor for KC-D1, written after discovering (2026-09-24) that
kc23_d1_replicate_stats.py had nothing to read -- it was wired to a single
run's own out_dir, which never held its four expected input csvs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_d1_aggregate import (build_reproduction_inputs, build_contrast_rows,
                               build_headline_rows, load_arm_post, run)

SUBJECTS = list(range(1, 41))


def write_cnn_arch(root: Path, arm: str, seed: int, f1_by_subject: dict, arch="resnet_se"):
    out = root / f"results_kc23_d1_{arm.lower()}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"subject": s, "arch": arch, "f1_macro": f1_by_subject[s], "bal_acc": f1_by_subject[s]}
                       for s in sorted(f1_by_subject)])
    df.to_csv(out / "cnn_arch_subjectwise.csv", index=False)


def write_adabn(root: Path, seed: int, pre_by_subject: dict, post_by_subject: dict):
    out = root / f"results_kc23_d1_r10_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"subject": s, "arch": "resnet_se", "f1_pre_adabn": pre_by_subject[s],
                       "f1_macro": post_by_subject[s], "bal_acc": post_by_subject[s],
                       "delta_pp": (post_by_subject[s] - pre_by_subject[s]) * 100}
                       for s in sorted(pre_by_subject)])
    df.to_csv(out / "adabn_subjectwise.csv", index=False)


def uniform_f1(value, rng=None, jitter=0.0):
    if rng is None:
        return {s: value for s in SUBJECTS}
    return {s: value + rng.uniform(-jitter, jitter) for s in SUBJECTS}


def test_load_arm_post_missing_returns_none(tmp_path):
    assert load_arm_post(tmp_path, "R1", 42) is None


def test_load_arm_post_incomplete_returns_none(tmp_path):
    write_cnn_arch(tmp_path, "R1", 42, {s: 0.7 for s in SUBJECTS[:39]})  # only 39 of 40
    assert load_arm_post(tmp_path, "R1", 42) is None


def test_load_arm_post_complete_returns_series(tmp_path):
    write_cnn_arch(tmp_path, "R1", 42, uniform_f1(0.78))
    s = load_arm_post(tmp_path, "R1", 42)
    assert s is not None
    assert len(s) == 40
    assert abs(s.mean() - 0.78) < 1e-9


def test_reproduction_inputs_none_until_all_three_present(tmp_path):
    write_cnn_arch(tmp_path, "R1", 42, uniform_f1(0.782))
    assert build_reproduction_inputs(tmp_path) is None  # R2, R10 still missing
    write_cnn_arch(tmp_path, "R2", 42, uniform_f1(0.8395))
    assert build_reproduction_inputs(tmp_path) is None  # R10 still missing
    write_adabn(tmp_path, 42, pre_by_subject=uniform_f1(0.787), post_by_subject=uniform_f1(0.84))
    df = build_reproduction_inputs(tmp_path)
    assert df is not None
    got = df.set_index("arm")["f1_mean"]
    assert abs(got["R1"] - 0.782) < 1e-9
    assert abs(got["R2"] - 0.8395) < 1e-9
    assert abs(got["R10_pre"] - 0.787) < 1e-9


def test_contrast_c1_not_ready_until_all_four_tier_a_seeds(tmp_path):
    for seed in [42, 7, 123]:  # missing 1001
        write_cnn_arch(tmp_path, "R1", seed, uniform_f1(0.78))
        write_cnn_arch(tmp_path, "R2", seed, uniform_f1(0.80))
    rows = build_contrast_rows(tmp_path)
    assert not any(r["contrast"] == "C1" for r in rows)
    write_cnn_arch(tmp_path, "R1", 1001, uniform_f1(0.78))
    write_cnn_arch(tmp_path, "R2", 1001, uniform_f1(0.80))
    rows = build_contrast_rows(tmp_path)
    c1 = [r for r in rows if r["contrast"] == "C1"]
    assert len(c1) == 4 * 40  # 4 realizations x 40 subjects
    assert all(abs(r["diff"] - 0.02) < 1e-9 for r in c1)
    assert all(r["tier"] == "A" for r in c1)


def test_contrast_c3_is_tier_b_three_seeds(tmp_path):
    for seed in [42, 7, 123]:
        write_cnn_arch(tmp_path, "R13", seed, uniform_f1(0.80))
        write_cnn_arch(tmp_path, "R14", seed, uniform_f1(0.81))
    rows = build_contrast_rows(tmp_path)
    c3 = [r for r in rows if r["contrast"] == "C3"]
    assert len(c3) == 3 * 40
    assert all(r["tier"] == "B" for r in c3)
    assert all(abs(r["diff"] - 0.01) < 1e-9 for r in c3)


def test_contrast_c9_uses_r10_post_not_pre(tmp_path):
    for seed in [42, 7, 123, 1001]:
        write_cnn_arch(tmp_path, "R2", seed, uniform_f1(0.84))
        write_adabn(tmp_path, seed, pre_by_subject=uniform_f1(0.70), post_by_subject=uniform_f1(0.80))
    rows = build_contrast_rows(tmp_path)
    c9 = [r for r in rows if r["contrast"] == "C9"]
    assert len(c9) == 4 * 40
    assert all(abs(r["diff"] - (0.84 - 0.80)) < 1e-9 for r in c9), "C9 must use R10's post-AdaBN f1, not pre"


def test_contrast_c12_uses_published_svm_constant(tmp_path):
    for seed in [42, 7, 123, 1001]:
        write_cnn_arch(tmp_path, "R2", seed, uniform_f1(0.8395))
    rows = build_contrast_rows(tmp_path)
    c12 = [r for r in rows if r["contrast"] == "C12"]
    assert len(c12) == 4 * 40
    assert all(abs(r["diff"] - (0.8395 - 0.777)) < 1e-9 for r in c12)


def test_interaction_contrast_c6(tmp_path):
    for seed in [42, 7, 123, 1001]:
        write_cnn_arch(tmp_path, "R1", seed, uniform_f1(0.70))
        write_cnn_arch(tmp_path, "R2", seed, uniform_f1(0.74))  # R2-R1 = +0.04
        write_cnn_arch(tmp_path, "R4", seed, uniform_f1(0.60))
        write_cnn_arch(tmp_path, "R5", seed, uniform_f1(0.61))  # R5-R4 = +0.01
    rows = build_contrast_rows(tmp_path)
    c6 = [r for r in rows if r["contrast"] == "C6"]
    assert len(c6) == 4 * 40
    assert all(abs(r["diff"] - (0.04 - 0.01)) < 1e-6 for r in c6)


def test_headline_rows_only_once_all_tier_a_seeds_present(tmp_path):
    for seed in [42, 7, 123]:
        write_cnn_arch(tmp_path, "R2", seed, uniform_f1(0.8395))
    rows = build_headline_rows(tmp_path)
    assert rows == []
    write_cnn_arch(tmp_path, "R2", 1001, uniform_f1(0.8395))
    rows = build_headline_rows(tmp_path)
    r2 = [r for r in rows if r["arm"] == "R2"][0]
    assert abs(r2["realization_mean"] - 0.8395) < 1e-9
    assert abs(r2["realization_sd"] - 0.0) < 1e-9


def test_run_writes_only_the_csvs_that_are_ready(tmp_path):
    root = tmp_path / "root"
    out = tmp_path / "out"
    for seed in [42, 7, 123, 1001]:
        write_cnn_arch(root, "R1", seed, uniform_f1(0.78))
        write_cnn_arch(root, "R2", seed, uniform_f1(0.84))
    # R10 missing entirely -- reproduction inputs and C9 must not be written
    rc = run(out, root)
    assert rc == 0
    assert not (out / "d1_reproduction_inputs.csv").exists()
    assert (out / "d1_contrasts.csv").exists()
    df = pd.read_csv(out / "d1_contrasts.csv")
    assert "C9" not in set(df["contrast"])
    assert "C1" in set(df["contrast"])
    assert (out / "d1_headline_inputs.csv").exists()


def test_run_is_idempotent_and_additive(tmp_path):
    root = tmp_path / "root"
    out = tmp_path / "out"
    write_cnn_arch(root, "R1", 42, uniform_f1(0.78))
    write_cnn_arch(root, "R2", 42, uniform_f1(0.84))
    write_adabn(root, 42, pre_by_subject=uniform_f1(0.787), post_by_subject=uniform_f1(0.80))
    run(out, root)
    assert (out / "d1_reproduction_inputs.csv").exists()
    assert not (out / "d1_contrasts.csv").exists()  # C1 needs all 4 Tier A seeds
    for seed in [7, 123, 1001]:
        write_cnn_arch(root, "R1", seed, uniform_f1(0.78))
        write_cnn_arch(root, "R2", seed, uniform_f1(0.84))
    run(out, root)
    assert (out / "d1_contrasts.csv").exists()
