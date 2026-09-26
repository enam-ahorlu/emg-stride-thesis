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
    v = (tmp_path / "out" / "S1_VERDICT.md").read_text()
    assert "NO OUTCOME COMPUTED" in v        # a missing input is not a result: no '**...: LETTER**' line
    assert "**" not in v


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


# ---------------------------------------------------------------------------
# 2026-09-25: secondary analyses (K curve, S-ft against L1, determinism),
# --report-only, and letter-free failure verdicts.
import re
LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")


def _seed_rows(seed, ft_boost=0.0, pool_offset=0.0, ens1=None):
    rows = []
    for K in (5, 10, 25):
        vals = [("SVM_decision", PUBLISHED_SVM), ("SVM_PROBA", PUBLISHED_SVM_PROBA), ("RESNET_SE", 0.79),
                ("L0", PUBLISHED_SOFT), ("S-only", 0.5), ("S-pool", 0.83 + pool_offset),
                ("S-ft", 0.79 + ft_boost), ("S-ens1", ens1 if ens1 is not None else PUBLISHED_SOFT),
                ("S-ens2", PUBLISHED_SOFT)]
        for arm, v in vals:
            for subj in range(1, 41):
                jitter = 0.0 if arm in ("SVM_decision", "SVM_PROBA") else 0.0001 * (subj % 7)   # gate arms stay exact
                rows.append({"subject": subj, "seed": seed, "K": K, "arm": arm,
                             "f1_macro": v + jitter, "n_excl": 100})
    return rows


def _write_all(root, **kw):
    for seed in (42, 7, 123):
        d = root / f"results_kc23_s1_scripted_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        extra = {k: (v(seed) if callable(v) else v) for k, v in kw.items()}
        pd.DataFrame(_seed_rows(seed, **extra)).to_csv(d / "s1_subjectwise.csv", index=False)


def test_verdict_starts_with_letter_and_has_all_secondary_sections(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    _write_all(tmp_path)
    assert run(tmp_path / "out") == 0
    v = (tmp_path / "out" / "S1_VERDICT.md").read_text()
    first_bold = LETTER_RE.search(v)
    assert first_bold and first_bold.group(0).startswith("**Outcome:")
    for section in ("Secondary 1: K curve", "Secondary 2: S-ft against L1", "Secondary 3"):
        assert section in v
    for f in ("S1_k_curve.csv", "S1_sft_vs_l1.csv", "S1_determinism.csv"):
        assert (tmp_path / "out" / f).exists()


def test_k_curve_reports_smallest_k_that_reaches_l0(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    df_rows = []
    for seed in (42, 7, 123):
        rows = _seed_rows(seed)
        for r in rows:   # S-ens1 behind L0 at K=5, level at 10, ahead at 25
            if r["arm"] == "S-ens1":
                r["f1_macro"] = {5: PUBLISHED_SOFT - 0.02, 10: PUBLISHED_SOFT, 25: PUBLISHED_SOFT + 0.03}[r["K"]] + 0.0001 * (r["subject"] % 7)
        d = tmp_path / f"results_kc23_s1_scripted_s{seed}"; d.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(d / "s1_subjectwise.csv", index=False)
    kc, smallest = mod.k_curve(mod.load_all_seeds())
    assert smallest["S-ens1"]["smallest_K_reaching_l0"] == 10
    assert smallest["S-ens1"]["smallest_K_significantly_ahead"] == 25


def test_sft_vs_l1_pairs_inside_each_seed(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    _write_all(tmp_path, ft_boost=lambda seed: {42: 0.01, 7: 0.02, 123: 0.03}[seed])
    rows = mod.sft_vs_l1(mod.load_all_seeds())
    assert [r["n_seeds_positive"] for r in rows] == [3, 3, 3]
    assert abs(rows[0]["delta_pp"] - 2.0) < 0.2


def test_determinism_check_confirms_invariant_arms_and_flags_a_seed_dependent_one(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    _write_all(tmp_path)
    rows = mod.determinism_check(mod.load_all_seeds())
    inv = {(r["arm"], r["K"]): r["seed_invariant"] for r in rows}
    assert inv[("S-pool", 25)] and inv[("S-only", 5)]
    # S-pool made seed-dependent -> the check must catch it
    _write_all(tmp_path, pool_offset=lambda seed: 0.001 * seed)
    rows = mod.determinism_check(mod.load_all_seeds())
    assert not {(r["arm"], r["K"]): r["seed_invariant"] for r in rows}[("S-pool", 10)]


def test_report_only_exits_zero_on_ds_but_gate_mode_exits_20(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    _write_all(tmp_path, ens1=PUBLISHED_SOFT + 0.03)
    assert run(tmp_path / "a", report_only=True) == 0
    assert "**Outcome: D-S**" in (tmp_path / "a" / "S1_VERDICT.md").read_text()
    assert run(tmp_path / "b") == 20


def test_missing_secondary_input_fails_with_no_letter_in_verdict(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for seed in (42, 7, 123):
        rows = [r for r in _seed_rows(seed) if not (r["arm"] == "S-ft" and r["K"] == 10)]
        d = tmp_path / f"results_kc23_s1_scripted_s{seed}"; d.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(d / "s1_subjectwise.csv", index=False)
    assert run(tmp_path / "out") == 20
    assert not LETTER_RE.search((tmp_path / "out" / "S1_VERDICT.md").read_text())


def test_stale_verdict_is_replaced_when_an_input_disappears(tmp_path, monkeypatch):
    import kc23_s1_scripted_stats as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    _write_all(tmp_path)
    assert run(tmp_path / "out") == 0
    (tmp_path / "results_kc23_s1_scripted_s123" / "s1_subjectwise.csv").unlink()
    assert run(tmp_path / "out") == 20
    assert not LETTER_RE.search((tmp_path / "out" / "S1_VERDICT.md").read_text())
