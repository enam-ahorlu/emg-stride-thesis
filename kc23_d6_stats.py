#!/usr/bin/env python3
"""
kc23_d6_stats.py
==================
KC-D6 gates and outcomes. EXPERIMENT_PLAN_KC23_DEEP.md "KC-D6. A learned
alignment axis that actually moves invariance", sections 6.5 to 6.7. Four
independent decision points, each with its own classify function so
tests_kc23/ can drive every letter:

1. Sanity gate (classify_sanity): after Stage 1, ADV at lambda_max=0, F1
   within +/-1.5pt of the D2e weight-0 target-pass arm (83.0%). FAIL -> ESCALATE
   (only sanity halts; everything else below is claim-level, report-and-continue).

2. Manipulation gate, per family (classify_manipulation):
     G-PASS: source-target domain probe falls >= 10pt AND unseen-subject
             probe falls with a significant Page trend (Holm < 0.05) -> run Stage 2
     G-WEAK: fall 2-10pt, or only one meter moves                    -> run Stage 2, flagged weak
     G-FAIL: fall < 2pt                                              -> no Stage 2

3. Primary outcome, ADV/SFC family (classify_outcome):
     X1: invariance rises monotonically (both meters, Page trend), F1 has an
         interior peak >= 2pt above the largest non-diverged knob (paired,
         significant), F1 tracks class info but not invariance meters
     X2: invariance rises, F1 flat/rising to the largest knob (no falling limb)
     X3: invariance rises, F1 falls from the first step (falling limb only)
     X4: shape appears but F1 does not track class information

4. Mechanism test, ADV-C against ADV at the collapse knob (classify_mechanism):
     C-M1: ADV-C >= ADV peak - 1pt, AND ADV-C - ADV >= 2pt at the collapse
           knob, AND ADV-C's within-class invariance >= ADV's
     C-M2: ADV-C collapses like ADV (|ADV-C - ADV| < 1pt at the collapse knob)
     C-M3: in between

Exit codes: only the sanity gate returns 20. G-FAIL returns 0 (a result, not
a failure). X2-X4 and C-M2/C-M3 return 10 (report, continue).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from kc23_stats_common import page_trend_test, paired_test, print_gate_header, write_no_outcome_verdict

SANITY_PUBLISHED_F1 = 0.830
SANITY_TOL = 0.015


def classify_sanity(f1_lambda0: float) -> tuple[str, dict]:
    diff = abs(f1_lambda0 - SANITY_PUBLISHED_F1)
    ok = diff <= SANITY_TOL
    return ("PASS" if ok else "FAIL"), {"f1_lambda0": f1_lambda0, "published": SANITY_PUBLISHED_F1, "diff": diff}


def classify_manipulation(domain_probe_by_knob: np.ndarray, subject_probe_trend_matrix: np.ndarray) -> tuple[str, dict]:
    """domain_probe_by_knob: 1D, lowest to highest knob (already realization-
    averaged). subject_probe_trend_matrix: (n_realizations, n_knobs), for the
    Page trend test (falling probe = rising invariance)."""
    domain_fall_pts = (domain_probe_by_knob[0] - domain_probe_by_knob[-1]) * 100.0
    page = page_trend_test(-subject_probe_trend_matrix)
    subject_probe_trend_significant = page["p_one_sided"] < 0.05

    domain_moves = domain_fall_pts >= 10.0
    subject_moves = subject_probe_trend_significant
    # G-FAIL is checked FIRST: the plan states it as a threshold purely on the
    # domain probe ("a fall < 2 points, the D2c threshold"), independent of
    # what the subject probe does.
    if domain_fall_pts < 2.0:
        letter = "G-FAIL"
    elif domain_moves and subject_moves:
        letter = "G-PASS"
    else:
        # 2-10pt fall, or >=10pt fall but the subject probe trend is not
        # significant (only one of the two meters moved): weak manipulation.
        letter = "G-WEAK"
    return letter, {"domain_fall_pts": domain_fall_pts, "subject_probe_page_p": page["p_one_sided"],
                    "domain_moves": domain_moves, "subject_moves": subject_moves}


def classify_outcome(invariance_by_knob: np.ndarray, f1_by_knob: np.ndarray,
                     class_metric_by_knob: np.ndarray, invariance_trend_matrix: np.ndarray) -> tuple[str, dict]:
    """All *_by_knob arrays: realization-averaged, lowest to highest knob.
    invariance_trend_matrix: (n_realizations, n_knobs) for the Page test
    (rising invariance == falling subject-probe accuracy, so pass the probe
    directly and this function negates internally, matching classify_manipulation)."""
    page = page_trend_test(-invariance_trend_matrix)
    invariance_rises = page["p_one_sided"] < 0.05

    peak_idx = int(np.argmax(f1_by_knob))
    interior_peak = 0 <= peak_idx < len(f1_by_knob) - 1
    drop_to_last_pp = (f1_by_knob[peak_idx] - f1_by_knob[-1]) * 100.0
    falling_limb = interior_peak and drop_to_last_pp >= 2.0
    rising_limb = peak_idx > 0

    rho_class, _ = stats.spearmanr(f1_by_knob, class_metric_by_knob)
    tracks_class = rho_class is not None and not np.isnan(rho_class) and rho_class > 0
    rho_inv, _ = stats.spearmanr(f1_by_knob, -invariance_by_knob)
    tracks_invariance = rho_inv is not None and not np.isnan(rho_inv) and rho_inv > 0.3

    detail = {"invariance_rises": invariance_rises, "peak_idx": peak_idx, "falling_limb": falling_limb,
             "rising_limb": rising_limb, "drop_to_last_pp": drop_to_last_pp,
             "tracks_class": tracks_class, "tracks_invariance": tracks_invariance}

    if not invariance_rises:
        return "X4", detail  # no invariance movement at all: report as the weakest case
    if falling_limb and rising_limb and tracks_class and not tracks_invariance:
        return "X1", detail
    if not falling_limb:
        return "X2", detail
    if falling_limb and not rising_limb:
        return "X3", detail
    return "X4", detail


def classify_mechanism(advc_at_collapse: float, adv_at_collapse: float, adv_peak: float,
                       advc_invariance: float, adv_invariance: float) -> tuple[str, dict]:
    cond1 = advc_at_collapse >= adv_peak - 0.01
    cond2 = (advc_at_collapse - adv_at_collapse) >= 0.02
    cond3 = advc_invariance >= adv_invariance
    detail = {"advc_ge_peak_minus_1pt": cond1, "advc_minus_adv_pp": (advc_at_collapse - adv_at_collapse) * 100.0,
             "advc_invariance_at_least_adv": cond3}
    if cond1 and cond2 and cond3:
        return "C-M1", detail
    if abs(advc_at_collapse - adv_at_collapse) < 0.01:
        return "C-M2", detail
    return "C-M3", detail


EXPECTED_FAMILIES = ("adv_marginal", "sfc", "advps")


def _fail_closed(out_dir: Path, reason: str) -> int:
    print(f"[D6] MISSING (fail closed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(out_dir / "D6_VERDICT.md", "KC-D6 verdict", reason)
    return 20


def run(out_dir: Path) -> int:
    """Which checks are expected here is driven by out_dir's own name (the
    queue only ever invokes a gate_script as `python <gate> --out <out_dir>`,
    no other arguments) -- 'sanity' or 'manipulation' in the name, matching
    kc23_build_job_csvs.py's d6_sanity_check / d6_manipulation_check pseudo-
    jobs. Fixed 2026-09-24: a missing expected file used to mean "nothing to
    check, continue" (exit 0) -- now it fails closed (exit 20), since these
    gates are only ever invoked by a job whose own dependencies guarantee the
    aggregator (kc23_d6_aggregate.py) already ran and should have produced it."""
    exit_code = 0
    results = {}
    name = out_dir.name

    expect_sanity = "sanity" in name
    expect_manipulation = "manipulation" in name
    if not expect_sanity and not expect_manipulation:
        # Unknown invocation context. The old code checked "whatever is present", which lets a partial set of
        # families through as if it were complete. There is no permissive default (2026-09-25).
        return _fail_closed(out_dir, f"{out_dir.name!r} names neither the sanity nor the manipulation check, so "
                            f"it is not known which inputs are required")

    if expect_sanity:
        sanity_path = out_dir / "d6_sanity.csv"
        if not sanity_path.exists():
            return _fail_closed(out_dir, f"{sanity_path} not found")
        f1_l0 = float(pd.read_csv(sanity_path)["f1"].mean())
        s_letter, s_detail = classify_sanity(f1_l0)
        print_gate_header("KC-D6 sanity", s_letter, "Continue." if s_letter == "PASS" else
                          "ESCALATE: the harness does not match the published Deep CORAL runs.")
        results["sanity"] = {"letter": s_letter, **s_detail}
        if s_letter == "FAIL":
            exit_code = 20

    if expect_manipulation:
        for family in EXPECTED_FAMILIES:
            family_path = out_dir / f"d6_manipulation_{family}.csv"
            if not family_path.exists():
                return _fail_closed(out_dir, f"{family_path} not found (family {family!r} incomplete "
                                    f"or never run)")
            df = pd.read_csv(family_path)  # cols: realization, knob, domain_probe, subject_probe
            knobs = sorted(df["knob"].unique())
            domain_mean = df.groupby("knob")["domain_probe"].mean().reindex(knobs).to_numpy()
            subj_mat = df.pivot(index="realization", columns="knob", values="subject_probe").reindex(columns=knobs).to_numpy()
            letter, detail = classify_manipulation(domain_mean, subj_mat)
            results[f"manipulation_{family}"] = {"letter": letter, **detail}
            print_gate_header(f"KC-D6 manipulation ({family})", letter,
                              {"G-PASS": "Run Stage 2.", "G-WEAK": "Run Stage 2, flagged weak.",
                               "G-FAIL": "No Stage 2."}[letter])

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"item": k, **v} for k, v in results.items()]
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "D6_gates.csv", index=False)
    verdict_lines = ["# KC-D6 verdict\n"]
    for item, v in results.items():
        verdict_lines.append(f"- **{item}: {v['letter']}**")
    (out_dir / "D6_VERDICT.md").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    if exit_code == 0 and any(v["letter"] in ("X2", "X3", "X4", "C-M2", "C-M3")
                              for v in results.values() if "letter" in v):
        exit_code = 10
    return exit_code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
