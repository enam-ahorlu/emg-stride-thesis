"""Synthetic-gate-output tests for src/kc23_d6_stage2_job_gen.py (docs/plans/RUN_ORDER_KC23.md
item 11c): passing_families() reads a D6_gates.csv correctly, and
build_stage2_rows()/append_rows() generate exactly the right seed-7/123 rows
for passing families only, idempotently."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from kc23_d6_stage2_job_gen import passing_families, build_stage2_rows, append_rows, FAMILY_SPECS


def _gates_csv(tmp_path, rows):
    p = tmp_path / "D6_gates.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_passing_families_mixed_letters(tmp_path):
    p = _gates_csv(tmp_path, [
        {"item": "sanity", "letter": "PASS"},
        {"item": "manipulation_adv_marginal", "letter": "G-PASS"},
        {"item": "manipulation_sfc", "letter": "G-WEAK"},
        {"item": "manipulation_advps", "letter": "G-FAIL"},
    ])
    fams = passing_families(p)
    assert fams == {"adv_marginal": "G-PASS", "sfc": "G-WEAK"}
    assert "advps" not in fams


def test_all_fail_yields_nothing(tmp_path):
    p = _gates_csv(tmp_path, [
        {"item": "manipulation_adv_marginal", "letter": "G-FAIL"},
        {"item": "manipulation_sfc", "letter": "G-FAIL"},
        {"item": "manipulation_advps", "letter": "G-FAIL"},
    ])
    assert passing_families(p) == {}


def test_build_stage2_rows_correct_counts_and_seeds():
    rows = build_stage2_rows({"adv_marginal": "G-PASS", "advps": "G-WEAK"})
    # 7 knobs x 2 seeds for adv_marginal, 3 knobs x 2 seeds for advps
    assert len(rows) == 7 * 2 + 3 * 2
    seeds = {r["seed"] for r in rows}
    assert seeds == {"7", "123"}
    job_ids = {r["job_id"] for r in rows}
    assert "d6_adv_marginal_l0_s7" in job_ids
    assert "d6_adv_marginal_l10_s123" in job_ids
    assert "d6_advps_l0.1_s7" in job_ids
    assert not any("sfc" in j for j in job_ids)  # sfc was not passing


def test_build_stage2_rows_unknown_family_is_skipped_not_guessed():
    rows = build_stage2_rows({"mystery_family": "G-PASS"})
    assert rows == []


def test_append_rows_idempotent(tmp_path):
    gpu_csv = tmp_path / "jobs/kc23_jobs_gpu.csv"
    gpu_csv.parent.mkdir(parents=True, exist_ok=True)
    gpu_csv.write_text("job_id,stage,seed,command,out_dir,depends_on,gate_script\n"
                      "existing_job,X,,cmd,out,,\n", encoding="utf-8")
    rows = build_stage2_rows({"advps": "G-PASS"})
    n1 = append_rows(gpu_csv, rows)
    assert n1 == len(rows)
    n2 = append_rows(gpu_csv, rows)  # re-running must not duplicate
    assert n2 == 0
    df = pd.read_csv(gpu_csv)
    assert len(df) == 1 + len(rows)
    assert df["job_id"].duplicated().sum() == 0


def test_family_specs_cover_every_stage1_family():
    assert set(FAMILY_SPECS) == {"adv_marginal", "sfc", "advps"}


def test_stage2_rows_are_instrumented_and_declare_their_outputs():
    import kc23_d6_stage2_job_gen as g
    rows = g.build_stage2_rows({"adv_marginal": "G-PASS", "sfc": "G-WEAK", "advps": "G-PASS"})
    assert rows and all("--instrument" in r["command"] and r["expected_outputs"].endswith("|40") for r in rows)


def test_the_outcome_pseudo_row_waits_for_every_stage2_row_and_is_gated_on_a_letter():
    import kc23_d6_stage2_job_gen as g
    rows = g.build_stage2_rows({"sfc": "G-PASS"})
    o = g.outcome_row(rows, "results/kc23_d6_manipulation_check/D6_gates.csv")
    assert set(o["depends_on"].split(";")) == {r["job_id"] for r in rows}
    assert "--require outcome" in o["command"] and o["gate_script"] == "src/kc23_d6_stats.py"
    assert o["expected_outputs"] == "*_VERDICT.md|LETTER" and "outcome" in o["out_dir"]
