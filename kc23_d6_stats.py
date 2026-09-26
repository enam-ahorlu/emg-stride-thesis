#!/usr/bin/env python3
"""
kc23_d6_stats.py
==================
KC-D6 gates and outcomes. EXPERIMENT_PLAN_KC23_DEEP.md "KC-D6. A learned
alignment axis that actually moves invariance", sections 6.5 to 6.7. Rewritten
26 September 2026 to conform to the plan text of 23 September
(KC23_PREREG_CONFORMANCE.md). The earlier version used the 3 realizations as
Page blocks, applied no Holm correction, took X1's falling limb as a bare
2.0 pt gap with no test and measured it to the last knob whether or not that arm
had diverged, counted a peak at the lowest knob as interior, and defaulted any
unmatched case to X4.

Mode is chosen by the out_dir name (the queue passes only --out):
  *sanity*        d6_sanity.csv
  *manipulation*  d6_family_<family>.csv for adv_marginal, sfc and advps (seed 42)
  *outcome*       d6_family_<family>.csv for every family that passed the gate (seeds 42, 7, 123)

Statistics (operationalised by Enam, 26 September 2026, matching KC-D4; see
kc23_invariance_common.py). Matrices are (40 folds, knobs), realization-averaged
per fold, and the FOLDS are the Page blocks:
  the two invariance meters   the source-target domain probe and the unseen-subject probe
  meters fall / rise          Page trend across the knob, Holm-adjusted across the two meters
  falling limb                the argmax of the mean F1 is not the highest non-diverged knob, F1 at that knob is at
                              least 2 pts below the peak AND below it by a paired Wilcoxon over the folds (p < 0.05)
  F1 tracks class / invariance  per-fold Spearman across the knobs, mean > 0 and a sign test over the folds

1. Sanity gate (classify_sanity): ADV at lambda_max 0, F1 within +/-1.5 pt of the D2e weight-0 target-pass arm (83.0%).
   FAIL -> ESCALATE. Only this gate halts.
2. Manipulation gate per family (classify_manipulation), seed 42:
     G-PASS  the domain probe falls >= 10 pts from the lowest to the highest knob AND the unseen-subject probe falls
             with a significant Page trend (Holm < 0.05)                    -> run Stage 2
     G-WEAK  a fall of 2 to 10 pts, or only one of the two meters moves    -> run Stage 2, flagged weak
     G-FAIL  a fall < 2 pts (the D2c threshold)                             -> no Stage 2
   For SFC the embedding norm must stay flat across the weights; if it does not the normalization is broken and the
   family stops (norm_ok = False, no Stage 2).
3. Primary outcome, ADV (and SFC on the same grid), realization-averaged (classify_outcome). With "invariance rises" =
   both meters rise (Holm < 0.05):
     X1  invariance rises, F1 has an INTERIOR peak with a falling limb (>= 2 pts and paired significant), F1 tracks the
         class probe AND the silhouette, and does not track either invariance meter
     X2  invariance rises, and F1 is flat or rising up to the largest non-diverged knob (no such falling limb)
     X3  invariance rises and F1 falls from the first step (peak at the lowest knob), with no rising limb
     X4  the shape appears (interior peak and falling limb) but F1 does not track class information
     X-OUT  anything else (for example invariance does not rise, or F1 tracks the invariance meters): the label for a
            case outside the pre-registered grid; exit 10; never defaulted into X4
4. Mechanism (classify_mechanism), ADV-C against ADV: unchanged decision rule; it needs a within-class subject-invariance
   measure and the ADV-C runs, neither of which exists yet (KC23_PREREG_CONFORMANCE.md, open).

Exit codes: only the sanity gate returns 20. G-FAIL returns 0 (a result). X2, X3, X4, X-OUT return 10.
A missing or malformed input returns 20 with a verdict that carries no outcome line.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_invariance_common import (ALPHA, holm_adjust, page_falls, peak_then_fall, tracks, tracks_invariance)
from kc23_stats_common import print_gate_header, write_no_outcome_verdict

SANITY_PUBLISHED_F1 = 0.830
SANITY_TOL = 0.015
FALL_MIN_PP = 2.0
DOMAIN_FALL_PASS_PP = 10.0
DOMAIN_FALL_FAIL_PP = 2.0
SFC_NORM_MAX_RATIO = 1.10       # operationalisation of "the embedding norm must stay flat" (max/min of the fold mean)
EXPECTED_FAMILIES = ("adv_marginal", "sfc", "advps")
N_FOLDS = 40


class InputError(Exception):
    pass


def classify_sanity(f1_lambda0: float) -> tuple[str, dict]:
    diff = abs(f1_lambda0 - SANITY_PUBLISHED_F1)
    ok = diff <= SANITY_TOL
    return ("PASS" if ok else "FAIL"), {"f1_lambda0": f1_lambda0, "published": SANITY_PUBLISHED_F1, "diff": diff}


def meters_fall(domain_mat: np.ndarray, subject_mat: np.ndarray) -> dict:
    """Page trend on each meter, folds as blocks, Holm across the two meters."""
    p_dom, p_sub = page_falls(domain_mat)["p"], page_falls(subject_mat)["p"]
    h_dom, h_sub = holm_adjust([p_dom, p_sub])
    return {"domain_p_raw": p_dom, "domain_p_holm": h_dom, "subject_p_raw": p_sub, "subject_p_holm": h_sub,
            "domain_falls": bool(h_dom < ALPHA), "subject_falls": bool(h_sub < ALPHA)}


def classify_manipulation(domain_mat: np.ndarray, subject_mat: np.ndarray) -> tuple[str, dict]:
    """Matrices (folds, knobs), lowest knob first."""
    domain_fall_pts = float((domain_mat[:, 0].mean() - domain_mat[:, -1].mean()) * 100.0)
    m = meters_fall(domain_mat, subject_mat)
    domain_moves = domain_fall_pts >= DOMAIN_FALL_PASS_PP
    subject_moves = m["subject_falls"]
    # G-FAIL is checked first: the plan states it as a threshold on the fall alone ("a fall < 2 points, the D2c
    # threshold"), independent of what the unseen-subject probe does. Where a fall < 2 pts coincides with a moving
    # subject probe the plan's G-FAIL and G-WEAK wordings overlap; the plan's explicit threshold wins (recorded in
    # KC23_PREREG_CONFORMANCE.md).
    if domain_fall_pts < DOMAIN_FALL_FAIL_PP:
        letter = "G-FAIL"
    elif domain_moves and subject_moves:
        letter = "G-PASS"
    else:
        letter = "G-WEAK"
    return letter, {"domain_fall_pts": domain_fall_pts, "domain_moves": bool(domain_moves),
                    "subject_moves": bool(subject_moves), **m}


def sfc_norm_flat(norm_mat: np.ndarray) -> tuple[bool, dict]:
    per_weight = norm_mat.mean(axis=0)
    ratio = float(per_weight.max() / per_weight.min()) if per_weight.min() > 0 else float("inf")
    return bool(ratio <= SFC_NORM_MAX_RATIO), {"norm_max_over_min": ratio, "max_allowed": SFC_NORM_MAX_RATIO}


def classify_outcome(f1: np.ndarray, domain: np.ndarray, subject: np.ndarray, sil: np.ndarray, cprobe: np.ndarray,
                     diverged: np.ndarray | None = None) -> tuple[str, dict]:
    """All matrices (folds, knobs), lowest knob first, realization-averaged. `diverged` (per knob) removes those arms
    from the sweep: the falling limb is measured to the LARGEST NON-DIVERGED knob."""
    k = f1.shape[1]
    keep = np.ones(k, bool) if diverged is None else ~np.asarray(diverged, bool)
    if keep.sum() < 3:
        raise InputError(f"only {int(keep.sum())} non-diverged knobs: too few to judge a shape")
    f1, domain, subject, sil, cprobe = (a[:, keep] for a in (f1, domain, subject, sil, cprobe))
    m = meters_fall(domain, subject)
    inv_rises = bool(m["domain_falls"] and m["subject_falls"])
    pf = peak_then_fall(f1, min_gap_pp=FALL_MIN_PP)
    kk = f1.shape[1]
    interior = bool(0 < pf["peak_idx"] < kk - 1)
    cls_sil, cls_probe = tracks(f1, sil), tracks(f1, cprobe)
    tracks_class = bool(cls_sil["tracks"] and cls_probe["tracks"])
    inv_subj = tracks_invariance(f1, subject)
    inv_dom = tracks_invariance(f1, domain)
    tracks_inv = bool(inv_subj["tracks_invariance"] or inv_dom["tracks_invariance"])
    detail = {"n_knobs_kept": int(keep.sum()), **m, "invariance_rises": inv_rises,
              "peak_idx": pf["peak_idx"], "interior_peak": interior, "falling_limb": pf["falls"],
              "gap_to_last_kept_pp": pf["gap_to_highest_pp"], "f1_wilcoxon_p": pf["wilcoxon_p"],
              "tracks_silhouette": cls_sil["tracks"], "tracks_class_probe": cls_probe["tracks"],
              "tracks_subject_probe": inv_subj["tracks_invariance"], "tracks_domain_probe": inv_dom["tracks_invariance"]}
    if not inv_rises:
        return "X-OUT", detail
    if not pf["falls"]:
        return "X2", detail
    if pf["peak_is_lowest"]:
        return "X3", detail
    if interior:
        if tracks_class and not tracks_inv:
            return "X1", detail
        if not tracks_class:
            return "X4", detail
    return "X-OUT", detail


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


# --------------------------------------------------------------------------- matrices and I/O
def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise InputError(f"required input missing: {path}")
    return pd.read_csv(path)


def family_matrices(df: pd.DataFrame, seeds: list[int], label: str):
    """(knobs, {measure: (folds, knobs)}, diverged per knob). Every (fold, knob, seed) exactly once, no NaN."""
    need = {"knob", "realization", "subject", "f1", "domain_probe_bacc", "subject_probe_bacc", "class_silhouette",
            "class_probe_bacc", "feat_norm_src", "diverged"}
    if not need <= set(df.columns):
        raise InputError(f"{label}: lacks columns {sorted(need - set(df.columns))}")
    if set(df["realization"].unique()) != set(seeds):
        raise InputError(f"{label}: realizations {sorted(df['realization'].unique())}, required {seeds}")
    knobs = sorted(df["knob"].unique(), key=float)
    counts = df.groupby(["subject", "knob"]).size().unstack("knob")
    if counts.shape[0] != N_FOLDS or (counts != len(seeds)).any().any():
        raise InputError(f"{label}: needs {N_FOLDS} folds x {len(knobs)} knobs x {len(seeds)} realizations, no gaps")
    mats = {}
    for col in ("f1", "domain_probe_bacc", "subject_probe_bacc", "class_silhouette", "class_probe_bacc", "feat_norm_src"):
        if df[col].isna().any():
            raise InputError(f"{label}: NaN in {col}")
        piv = df.groupby(["subject", "knob"])[col].mean().unstack("knob")
        mats[col] = piv[knobs].to_numpy(float)
    diverged = np.array([bool(df.loc[df["knob"] == k, "diverged"].any()) for k in knobs])
    return knobs, mats, diverged


def _fail(out_dir: Path, name: str, reason: str) -> int:
    print(f"[D6] MISSING (fail closed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(out_dir / "D6_VERDICT.md", "KC-D6 verdict", reason)
    (out_dir / "D6_gates.csv").unlink(missing_ok=True)
    return 20


def _per_knob_table(knobs, mats, diverged) -> str:
    t = pd.DataFrame({"knob": knobs, "f1": mats["f1"].mean(0), "domain_probe": mats["domain_probe_bacc"].mean(0),
                      "subject_probe": mats["subject_probe_bacc"].mean(0), "silhouette": mats["class_silhouette"].mean(0),
                      "class_probe": mats["class_probe_bacc"].mean(0), "embed_norm": mats["feat_norm_src"].mean(0),
                      "diverged": diverged})
    return "```\n" + t.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n```"


def run(out_dir: Path) -> int:
    name = out_dir.name
    if "sanity" in name:
        mode = "sanity"
    elif "manipulation" in name:
        mode = "manipulation"
    elif "outcome" in name:
        mode = "outcome"
    else:
        return _fail(out_dir, name, f"{name!r} names neither the sanity, the manipulation nor the outcome check, so it "
                                    f"is not known which inputs are required")
    results, tables, exit_code = {}, [], 0
    try:
        if mode == "sanity":
            sp = _read(out_dir / "d6_sanity.csv")
            if sp["subject"].duplicated().any() or len(sp) != N_FOLDS or sp["f1"].isna().any():
                raise InputError(f"d6_sanity.csv needs {N_FOLDS} unique non-NaN folds, found {len(sp)}")
            s_letter, s_detail = classify_sanity(float(sp["f1"].mean()))
            print_gate_header("KC-D6 sanity", s_letter, "Continue." if s_letter == "PASS" else
                              "ESCALATE: the harness does not match the published Deep CORAL runs.")
            results["sanity"] = {"letter": s_letter, **s_detail}
            if s_letter == "FAIL":
                exit_code = 20
        elif mode == "manipulation":
            for fam in EXPECTED_FAMILIES:
                df = _read(out_dir / f"d6_family_{fam}.csv")
                knobs, mats, div = family_matrices(df, [42], fam)
                letter, detail = classify_manipulation(mats["domain_probe_bacc"], mats["subject_probe_bacc"])
                row = {"letter": letter, "norm_ok": True, **detail}
                if fam == "sfc":
                    ok, nd = sfc_norm_flat(mats["feat_norm_src"])
                    row.update(nd); row["norm_ok"] = ok
                    if not ok:
                        exit_code = max(exit_code, 10)
                results[f"manipulation_{fam}"] = row
                tables.append(f"### {fam} (seed 42)\n\n" + _per_knob_table(knobs, mats, div))
                print_gate_header(f"KC-D6 manipulation ({fam})", letter,
                                  {"G-PASS": "Run Stage 2.", "G-WEAK": "Run Stage 2, flagged weak.",
                                   "G-FAIL": "No Stage 2."}[letter])
        else:  # outcome
            marker = out_dir / "d6_no_passing_family.csv"
            fams = [f for f in EXPECTED_FAMILIES if (out_dir / f"d6_family_{f}.csv").exists()]
            if not fams and not marker.exists():
                raise InputError("neither a d6_family_<family>.csv nor d6_no_passing_family.csv is present")
            for fam in fams:
                knobs, mats, div = family_matrices(_read(out_dir / f"d6_family_{fam}.csv"), [42, 7, 123], fam)
                letter, detail = classify_outcome(mats["f1"], mats["domain_probe_bacc"], mats["subject_probe_bacc"],
                                                  mats["class_silhouette"], mats["class_probe_bacc"], div)
                results[f"outcome_{fam}"] = {"letter": letter, **detail}
                tables.append(f"### {fam} (seeds 42, 7, 123)\n\n" + _per_knob_table(knobs, mats, div))
                print_gate_header(f"KC-D6 outcome ({fam})", letter, "")
                if letter != "X1":
                    exit_code = max(exit_code, 10)
            if not fams:
                results["outcome"] = {"letter": "NO-STAGE-2"}
                exit_code = max(exit_code, 10)
    except InputError as e:
        return _fail(out_dir, name, str(e))
    except ValueError as e:
        return _fail(out_dir, name, str(e))

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"item": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(out_dir / "D6_gates.csv", index=False)
    lines = ["# KC-D6 verdict\n"]
    for item, v in results.items():
        label = v["letter"] if v["letter"] != "X-OUT" else "X-OUT (outside the pre-registered grid)"
        lines.append(f"- **{item}: {label}**")
    if mode == "manipulation":
        lets = [v["letter"] for k, v in results.items()]
        if all(l == "G-FAIL" for l in lets):
            lines.append("\nEvery family is G-FAIL: this axis cannot be tested with these knobs (plan 6.7); the "
                         "Section 5.3 limitation is restated with the new evidence.")
        if not results["manipulation_sfc"]["norm_ok"]:
            lines.append("\nSFC: the embedding norm did not stay flat across the weights, so the normalization is "
                         "broken and the family stops (no Stage 2).")
    if tables:
        lines.append("\n" + "\n\n".join(tables))
    (out_dir / "D6_VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if exit_code == 0 and mode == "outcome" and any(v["letter"] != "X1" for v in results.values()):
        exit_code = 10
    return exit_code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
