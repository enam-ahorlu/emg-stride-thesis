"""Fail-closed behaviour of kc23_d1_aggregate.py (--require) and
kc23_d1_replicate_stats.py (required inputs by directory), added 2026-09-25
after the fail-open sweep found both exiting 0 with every input missing."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_d1_aggregate as agg
import kc23_d1_replicate_stats as st

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")


def _arm(root: Path, arm: str, seed: int, mean: float):
    d = root / f"results_kc23_d1_{arm.lower()}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    f1 = np.full(40, mean) + np.linspace(-0.01, 0.01, 40)
    if arm == "R10":
        pd.DataFrame({"subject": range(1, 41), "f1_macro": f1, "f1_pre_adabn": f1 - 0.05}).to_csv(
            d / "adabn_subjectwise.csv", index=False)
    else:
        pd.DataFrame({"subject": range(1, 41), "f1_macro": f1}).to_csv(d / "cnn_arch_subjectwise.csv", index=False)


def _repro_root(root: Path):
    _arm(root, "R1", 42, 0.782)
    _arm(root, "R2", 42, 0.8395)
    _arm(root, "R10", 42, 0.79)


def test_require_repro_fails_with_no_inputs_and_writes_nothing(tmp_path):
    out = tmp_path / "o"
    assert agg.run(out, tmp_path, require="repro") == 1
    assert not list(out.glob("*.csv"))


def test_require_repro_passes_when_seed42_triplet_present(tmp_path):
    _repro_root(tmp_path)
    assert agg.run(tmp_path / "o", tmp_path, require="repro") == 0
    assert (tmp_path / "o" / "d1_reproduction_inputs.csv").exists()


def test_require_full_fails_when_contrasts_incomplete(tmp_path):
    _repro_root(tmp_path)
    assert agg.run(tmp_path / "o", tmp_path, require="full") == 1
    assert not list((tmp_path / "o").glob("*.csv"))


def test_default_mode_fails_when_nothing_ready(tmp_path):
    assert agg.run(tmp_path / "o", tmp_path) == 1


def test_stale_outputs_are_removed_when_a_run_fails(tmp_path):
    _repro_root(tmp_path)
    out = tmp_path / "o"
    assert agg.run(out, tmp_path, require="repro") == 0
    (tmp_path / "results_kc23_d1_r10_s42" / "adabn_subjectwise.csv").unlink()
    assert agg.run(out, tmp_path, require="repro") == 1
    assert not (out / "d1_reproduction_inputs.csv").exists()


def _stats_dir(tmp_path, name, with_repro=True):
    d = tmp_path / name
    d.mkdir()
    if with_repro:
        pd.DataFrame([{"arm": "R1", "f1_mean": 0.782}, {"arm": "R2", "f1_mean": 0.8395},
                      {"arm": "R10_pre", "f1_mean": 0.787}]).to_csv(d / "d1_reproduction_inputs.csv", index=False)
    return d


def test_stats_repro_dir_needs_only_the_reproduction_inputs(tmp_path):
    d = _stats_dir(tmp_path, "results_kc23_d1_repro_check")
    assert st.run(d, st.required_inputs(d)) == 0
    assert "**reproduction: PASS**" in (d / "D1_VERDICT.md").read_text()


def test_stats_with_no_inputs_exits_20_no_letter(tmp_path):
    d = _stats_dir(tmp_path, "results_kc23_d1_repro_check", with_repro=False)
    assert st.run(d, st.required_inputs(d)) == 20
    assert not LETTER_RE.search((d / "D1_VERDICT.md").read_text())


def test_stats_full_dir_missing_contrasts_fails_even_though_repro_present(tmp_path):
    d = _stats_dir(tmp_path, "results_kc23_d1_stats")
    assert st.required_inputs(d) == {"repro", "contrasts", "headline"}
    assert st.run(d, st.required_inputs(d)) == 20
    assert not LETTER_RE.search((d / "D1_VERDICT.md").read_text())


def test_stats_unknown_directory_name_has_no_permissive_default(tmp_path):
    assert st.required_inputs(tmp_path / "somewhere_else") is None


def test_stats_full_verdict_names_what_was_not_computed(tmp_path):
    d = _stats_dir(tmp_path, "results_kc23_d1_stats")
    rows = [{"contrast": c, "tier": "A", "realization": s, "subject": sub, "diff": 0.05 + 0.001 * sub}
            for c in agg.EXTRACTABLE_CONTRASTS for s in (42, 7, 123, 1001) for sub in range(1, 41)]
    pd.DataFrame(rows).to_csv(d / "d1_contrasts.csv", index=False)
    pd.DataFrame([{"arm": a, "realization_mean": 0.84, "realization_sd": 0.005} for a in agg.HEADLINE_ARMS]).to_csv(
        d / "d1_headline_inputs.csv", index=False)
    rc = st.run(d, st.required_inputs(d))
    text = (d / "D1_VERDICT.md").read_text()
    assert "NOT computed" in text and "C17" in text and "C13b" in text and "global" in text
    assert rc in (0, 20)
