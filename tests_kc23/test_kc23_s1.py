"""Synthetic tests for kc23_s1_scripted_stats.py: D-S/D-T/D-L, the 3-check
reproduction gate (rewritten 2026-09-24: the previous version compared the
published ensemble figure with itself when its file was missing -- see
run_scripted_supervised.reproduction_gate, reused here, never redefined)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_s1_scripted_stats import classify_d, choose_best_supervised, load_all_seeds, run
from run_scripted_supervised import reproduction_gate, PUBLISHED_SVM, PUBLISHED_SVM_PROBA, PUBLISHED_SOFT

RNG = np.random.default_rng(5)
N = 40


def _f1(base, noise=0.01):
    return np.clip(base + RNG.normal(0, noise, N), 0.05, 0.99)


def test_ds_supervised_ahead():
    l0 = _f1(0.817, noise=0.005)
    sup = _f1(0.840, noise=0.005)  # 2.3pt ahead
    letter, t = classify_d(sup, l0)
    assert letter == "D-S", t


def test_dt_within_band():
    l0 = _f1(0.817, noise=0.01)
    sup = _f1(0.820, noise=0.01)
    letter, t = classify_d(sup, l0)
    assert letter == "D-T", t


def test_dl_label_free_ahead():
    l0 = _f1(0.850, noise=0.005)
    sup = _f1(0.820, noise=0.005)  # 3pt behind
    letter, t = classify_d(sup, l0)
    assert letter == "D-L", t


def test_choose_best_supervised():
    e1 = _f1(0.80)
    e2 = _f1(0.85)
    name, arr = choose_best_supervised(e1, e2)
    assert name == "S-ens2"
    assert np.array_equal(arr, e2)


def test_reproduction_gate_pass_all_three_exact():
    letter, d = reproduction_gate(PUBLISHED_SVM, PUBLISHED_SVM_PROBA, PUBLISHED_SOFT)
    assert letter == "PASS", d


def test_reproduction_gate_fails_on_svm_drift():
    letter, d = reproduction_gate(PUBLISHED_SVM - 0.01, PUBLISHED_SVM_PROBA, PUBLISHED_SOFT)
    assert letter == "FAIL", d
    assert not d["svm_ok"]


def test_reproduction_gate_fails_on_soft_outside_band():
    letter, d = reproduction_gate(PUBLISHED_SVM, PUBLISHED_SVM_PROBA, PUBLISHED_SOFT - 0.03)
    assert letter == "FAIL", d
    assert not d["soft_ok"]


def _write_seed_csv(root: Path, seed: int, overrides: dict | None = None):
    """40-subject rows for every (K, arm) at the published/reproducing values,
    unless overridden. overrides: {(K, arm): value} or {(K, arm): None} to omit."""
    overrides = overrides or {}
    rows = []
    for K in (5, 10, 25):
        for arm, val in [("SVM_decision", PUBLISHED_SVM), ("SVM_PROBA", PUBLISHED_SVM_PROBA),
                        ("RESNET_SE", 0.79), ("L0", PUBLISHED_SOFT), ("S-only", 0.5),
                        ("S-pool", 0.83), ("S-ft", 0.82), ("S-ens1", PUBLISHED_SOFT),
                        ("S-ens2", PUBLISHED_SOFT)]:
            key = (K, arm)
            if key in overrides and overrides[key] is None:
                continue
            v = overrides.get(key, val)
            for subj in range(1, 41):
                rows.append({"subject": subj, "seed": seed, "K": K, "arm": arm,
                            "f1_macro": v, "n_excl": 100})
    d = root / f"results_kc23_s1_scripted_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "s1_subjectwise.csv", index=False)


def test_run_missing_seed_is_fail_not_fallback(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    _write_seed_csv(tmp_path, 42)
    # seeds 7 and 123 deliberately absent
    rc = run(tmp_path / "out")
    assert rc == 20
    assert "FAIL" in (tmp_path / "out" / "S1_VERDICT.md").read_text()


def test_run_passes_when_all_three_seeds_complete_and_reproduces(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for seed in (42, 7, 123):
        _write_seed_csv(tmp_path, seed)
    rc = run(tmp_path / "out")
    assert rc == 0
    verdict = (tmp_path / "out" / "S1_VERDICT.md").read_text()
    assert "PASS" in verdict


def test_run_fails_on_svm_drift_even_with_all_seeds_present(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for seed in (42, 7, 123):
        _write_seed_csv(tmp_path, seed, overrides={(25, "SVM_decision"): 0.69})
    rc = run(tmp_path / "out")
    assert rc == 20
