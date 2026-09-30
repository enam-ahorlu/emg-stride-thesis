#!/usr/bin/env python3
"""
src/kc23_d6_stats.py
==================
KC-D6 gates and outcomes. docs/plans/EXPERIMENT_PLAN_KC23_DEEP.md "KC-D6. A learned
alignment axis that actually moves invariance", sections 6.5 to 6.7. Rewritten
26 September 2026 to conform to the plan text of 23 September
(docs/kc23/KC23_PREREG_CONFORMANCE.md). The earlier version used the 3 realizations as
Page blocks, applied no Holm correction, took X1's falling limb as a bare
2.0 pt gap with no test and measured it to the last knob whether or not that arm
had diverged, counted a peak at the lowest knob as interior, and defaulted any
unmatched case to X4.

Mode is chosen by the out_dir name (the queue passes only --out):
  *sanity*        d6_sanity.csv
  *manipulation*  d6_family_<family>.csv for adv_marginal, sfc and advps (seed 42)
  *outcome*       d6_family_<family>.csv for every family that passed the gate (seeds 42, 7, 123); the verdict ends with the
                  combined through-line matrix (axis x invariance measured / shape shown / mechanism measured / replicated on
                  ENABL3S), whose cells hold the letters the other stages produced (no new judgment is made in it)
  *mechanism*     d6_mechanism_meta.csv, d6_mechanism_arms.csv (ADV against ADV-C at the ADV collapse lambda, plan 6.6)
  *secondary*     d6_secondary.csv (ADV at its best lambda against R2, R10, R11; ADV-PS at each lambda against R2)

Statistics (operationalised by Enam, 26 September 2026, matching KC-D4; see
src/kc23_invariance_common.py). Matrices are (40 folds, knobs), realization-averaged
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
     G-FAIL  BOTH meters fail: a domain fall < 2 pts (the D2c threshold) AND no significant subject-probe trend
                                                                             -> no Stage 2   (Enam, 26 Sept)
   For SFC the L2-normalised embedding the CORAL term sees must be unit norm (max |norm - 1| < 1e-5); if it is not the
   normalization is broken and the family stops (norm_ok = False, no Stage 2). The raw penultimate norm is reported only.
3. Primary outcome, ADV (and SFC on the same grid), realization-averaged (classify_outcome). With "invariance rises" =
   both meters rise (Holm < 0.05):
     X1  invariance rises, F1 has an INTERIOR peak with a falling limb (>= 2 pts and paired significant), F1 tracks the
         class probe AND the silhouette, and does not track either invariance meter
     X2  invariance rises, and F1 is flat or rising up to the largest non-diverged knob (no such falling limb)
     X3  invariance rises and F1 falls from the first step (peak at the lowest knob), with no rising limb
     X4  the shape appears (interior peak and falling limb) but F1 does not track class information
     X-OUT  anything else (for example invariance does not rise, or F1 tracks the invariance meters): the label for a
            case outside the pre-registered grid; exit 10; never defaulted into X4
4. Mechanism (classify_mechanism), ADV-C against ADV at the ADV collapse lambda_max (the smallest lambda_max above the peak
   whose realization-mean F1 is 2 pts or more below it), realization-averaged and paired. Within-class subject invariance is
   1 minus the within-class subject probe (subject_probe_within_class_bacc: the probe inside each movement class, averaged over
   classes); ADV-C must be at least as invariant as ADV. C-M1 exits 0; C-M2, C-M3 and "ADV has no collapse, ADV-C not run"
   (plan 6.2) exit 10 (claim-level, reported). ADV-CDAN, when run (only after C-M1), is tabulated beside them, no letter.
5. Secondary contrasts: reported, no letter (exit 0): the batch size differs from KC-D1's runs (256 here, the D1 default there),
   so the contrasts against R2, R10 and R11 are stated with that difference.

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
from kc23_stats_common import paired_test, print_gate_header, write_no_outcome_verdict

SANITY_PUBLISHED_F1 = 0.830
SANITY_TOL = 0.015
FALL_MIN_PP = 2.0
DOMAIN_FALL_PASS_PP = 10.0
DOMAIN_FALL_FAIL_PP = 2.0
SFC_UNIT_NORM_TOL = 1e-5        # Enam, 26 Sept: the L2-normalised embedding the CORAL term sees must be unit norm, max |norm - 1| < 1e-5
EXPECTED_FAMILIES = ("adv_marginal", "sfc", "advps")
N_FOLDS = 40
WITHIN = "subject_probe_within_class_bacc"


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
    # Ruling (Enam, 26 Sept): G-WEAK explicitly covers "only one of the two meters moves", so G-FAIL needs BOTH to fail:
    # the domain-probe fall under 2 pts AND no significant Page trend on the unseen-subject probe. A domain fall under
    # 2 pts with a significant subject-probe trend is G-WEAK.
    if domain_fall_pts < DOMAIN_FALL_FAIL_PP and not subject_moves:
        letter = "G-FAIL"
    elif domain_moves and subject_moves:
        letter = "G-PASS"
    else:
        letter = "G-WEAK"
    return letter, {"domain_fall_pts": domain_fall_pts, "domain_moves": bool(domain_moves),
                    "subject_moves": bool(subject_moves), **m}


def sfc_unit_norm(normdev_mat: np.ndarray, raw_norm_mat: np.ndarray) -> tuple[bool, dict]:
    """Ruling (Enam, 26 Sept). The stop condition is the normalisation itself: the embedding the CORAL term sees is L2
    normalised, so its norm must be 1 (max |norm - 1| < 1e-5 over every fold, weight and batch). Failing it is the
    broken-normalisation stop. The RAW penultimate norm across the weights is reported, descriptively, and is NOT a stop
    condition: once the loss is scale-free, raw scale no longer lowers it."""
    worst = float(np.max(normdev_mat))
    return bool(np.isfinite(worst) and worst < SFC_UNIT_NORM_TOL), {
        "normalised_norm_dev_max": worst, "unit_norm_tol": SFC_UNIT_NORM_TOL,
        "raw_norm_by_weight": ";".join(f"{v:.4f}" for v in raw_norm_mat.mean(axis=0))}


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
            "class_probe_bacc", "feat_norm_src", "diverged"} | ({"coral_embed_normdev"} if label == "sfc" else set())
    if not need <= set(df.columns):
        raise InputError(f"{label}: lacks columns {sorted(need - set(df.columns))}")
    if set(df["realization"].unique()) != set(seeds):
        raise InputError(f"{label}: realizations {sorted(df['realization'].unique())}, required {seeds}")
    knobs = sorted(df["knob"].unique(), key=float)
    counts = df.groupby(["subject", "knob"]).size().unstack("knob")
    if counts.shape[0] != N_FOLDS or (counts != len(seeds)).any().any():
        raise InputError(f"{label}: needs {N_FOLDS} folds x {len(knobs)} knobs x {len(seeds)} realizations, no gaps")
    mats = {}
    for col in ("f1", "domain_probe_bacc", "subject_probe_bacc", "class_silhouette", "class_probe_bacc", "feat_norm_src") + (
            ("coral_embed_normdev",) if label == "sfc" else ()):
        if df[col].isna().any():
            raise InputError(f"{label}: NaN in {col}")
        piv = df.groupby(["subject", "knob"])[col].mean().unstack("knob")
        mats[col] = piv[knobs].to_numpy(float)
    diverged = np.array([bool(df.loc[df["knob"] == k, "diverged"].any()) for k in knobs])
    return knobs, mats, diverged


def mechanism_letter(meta: pd.Series, arms: pd.DataFrame) -> tuple[str, dict, list[dict]]:
    """ADV-C against ADV at the collapse lambda_max. Realization-averaged per subject, paired over the 40 subjects."""
    ck, nk = float(meta["collapse_knob"]), (None if pd.isna(meta["next_knob"]) else float(meta["next_knob"]))

    def per_subject(kind, knob, col):
        d = arms[(arms["kind"] == kind) & (arms["knob"].astype(float) == knob)]
        if d.empty or d["realization"].nunique() != len(str(meta["seeds"]).split(";")):
            raise InputError(f"{kind} at lambda_max {knob}: realizations {sorted(d['realization'].unique())}, need {meta['seeds']}")
        piv = d.groupby(["subject", "realization"])[col].mean().unstack("realization")
        if len(piv) != N_FOLDS or piv.isna().any().any():
            raise InputError(f"{kind} at lambda_max {knob}: needs {N_FOLDS} folds in every realization")
        return piv.mean(axis=1).sort_index().to_numpy(float)

    rows = []
    for knob in [ck] + ([nk] if nk is not None else []):
        f_c, f_a = per_subject("advc", knob, "f1"), per_subject("adv", knob, "f1")
        w_c, w_a = per_subject("advc", knob, WITHIN), per_subject("adv", knob, WITHIN)
        letter, detail = classify_mechanism(float(f_c.mean()), float(f_a.mean()), float(meta["peak_f1"]),
                                            float(1 - w_c.mean()), float(1 - w_a.mean()))
        t = paired_test(f_c, f_a, f"advc_minus_adv_lambda{knob:g}", "D6-mechanism")
        rows.append({"lambda_max": knob, "letter_if_this_value": letter, "advc_f1": float(f_c.mean()), "adv_f1": float(f_a.mean()),
                     "advc_minus_adv_pp": t["delta_pp"], "wilcoxon_p": t["p_raw"], "dz": t["cohens_d" if "cohens_d" in t else "cohens_dz"],
                     "advc_within_class_probe": float(w_c.mean()), "adv_within_class_probe": float(w_a.mean()),
                     "n_improved": t["n_improved"], **detail})
    return rows[0]["letter_if_this_value"], rows[0], rows


def cdan_rows(meta: pd.Series, arms: pd.DataFrame) -> list[dict]:
    d = arms[arms["kind"] == "cdan"]
    out = []
    for knob in sorted(d["knob"].astype(float).unique()):
        x = d[d["knob"].astype(float) == knob]
        out.append({"lambda_max": knob, "cdan_f1": float(x.groupby("subject")["f1"].mean().mean()),
                    "cdan_within_class_probe": float(x.groupby("subject")[WITHIN].mean().mean()),
                    "cdan_domain_probe": float(x.groupby("subject")["domain_probe_bacc"].mean().mean())})
    return out


def secondary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in df.groupby("contrast", sort=False):
        avg = g.groupby("subject")["diff"].mean().sort_index().to_numpy(float)
        if len(avg) != N_FOLDS:
            raise InputError(f"secondary contrast {cid}: needs {N_FOLDS} subjects, found {len(avg)}")
        t = paired_test(avg, np.zeros_like(avg), cid, "D6-secondary")
        by_real = g.groupby("realization")["diff"].mean()
        rows.append({"contrast": cid, "mean_diff_pp": t["delta_pp"], "wilcoxon_p": t["p_raw"], "dz": t["cohens_dz"],
                     "bca_lo_pp": t["bca_lo_pp"], "bca_hi_pp": t["bca_hi_pp"], "n_improved": t["n_improved"],
                     "n_realizations": int(len(by_real)), "n_realizations_same_sign": int((np.sign(by_real) == np.sign(by_real.mean())).sum())})
    return pd.DataFrame(rows)


def _bold_lines(path: Path) -> list[str]:
    """The '**label: LETTER**' lines of another stage's verdict, or [] when it does not exist yet."""
    import re
    if not path.exists():
        return []
    return [m.group(1).strip() for m in re.finditer(r"\*\*([^*\n]*:\s*[A-Za-z0-9][^*\n]*)\*\*", path.read_text(encoding="utf-8"))]


def throughline_matrix(root: Path, d6_manip: list[str], d6_outcome: list[str], d6_mech: list[str]) -> str:
    """Plan 6 output: 'a combined through-line matrix: axis (ladder from C2, channel from D4, learned from D6) x {invariance
    measured, shape shown, mechanism measured, replicated on ENABL3S where run}'. Each cell lists the letters the stage that
    bears on it produced (or says it is not available yet); the matrix makes no judgment of its own."""
    def cell(lines):
        return "; ".join(lines) if lines else "not available"
    c2 = _bold_lines(root / "results/kc23_c2_whitening_w400" / "C2_VERDICT.md")
    c6 = _bold_lines(root / "results/kc23_c6_ladder_enabl3s" / "C6_VERDICT.md")
    d4 = _bold_lines(root / "results/kc23_d4_stats" / "D4_VERDICT.md")
    d5 = _bold_lines(root / "results/kc23_d5_stats" / "D5_VERDICT.md")
    w = [l for l in c2 if "Endpoint 1" in l]
    m = [l for l in c2 if "Endpoint 2" in l]
    rows = [("ladder (C2, C6)", "geometry rows and probes (src/kc23_c6_geometry.py), no letter", cell(w), cell(m), cell(c6)),
            ("channel (D4)", cell(d4), cell(d4), cell(d4), cell(d5)),
            ("learned (D6)", cell(d6_manip), cell(d6_outcome), cell(d6_mech), "not run on ENABL3S")]
    return ("| axis | invariance measured | shape shown | mechanism measured | replicated on ENABL3S where run |\n"
            "|---|---|---|---|---|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows))


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
    elif "mechanism" in name:
        mode = "mechanism"
    elif "secondary" in name:
        mode = "secondary"
    else:
        return _fail(out_dir, name, f"{name!r} names neither the sanity, manipulation, outcome, mechanism nor secondary check, "
                                    f"so it is not known which inputs are required")
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
                    ok, nd = sfc_unit_norm(mats["coral_embed_normdev"], mats["feat_norm_src"])
                    row.update(nd); row["norm_ok"] = ok
                    if not ok:
                        exit_code = max(exit_code, 10)
                results[f"manipulation_{fam}"] = row
                tables.append(f"### {fam} (seed 42)\n\n" + _per_knob_table(knobs, mats, div))
                print_gate_header(f"KC-D6 manipulation ({fam})", letter,
                                  {"G-PASS": "Run Stage 2.", "G-WEAK": "Run Stage 2, flagged weak.",
                                   "G-FAIL": "No Stage 2."}[letter])
        elif mode == "outcome":
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
        elif mode == "mechanism":
            meta_df = _read(out_dir / "d6_mechanism_meta.csv")
            if len(meta_df) != 1:
                raise InputError("d6_mechanism_meta.csv must hold exactly one row")
            meta = meta_df.iloc[0]
            if bool(meta["not_run"]):
                results["mechanism"] = {"letter": "C-NOT-RUN", "peak_knob": meta["peak_knob"], "peak_f1": meta["peak_f1"]}
                tables.append("ADV has no collapse (no lambda_max above the peak is 2 pts or more below it), so ADV-C is not run "
                              "(plan 6.2). ADV F1 by lambda_max: " + str(meta["adv_f1_by_knob"]))
                exit_code = max(exit_code, 10)
            else:
                arms = _read(out_dir / "d6_mechanism_arms.csv")
                letter, first, rows_m = mechanism_letter(meta, arms)
                results["mechanism"] = {"letter": letter, **first}
                exit_code = max(exit_code, 0 if letter == "C-M1" else 10)
                pd.DataFrame(rows_m).to_csv(out_dir / "D6_mechanism.csv", index=False)
                cd = cdan_rows(meta, arms)
                tab = pd.DataFrame(rows_m)[["lambda_max", "letter_if_this_value", "advc_f1", "adv_f1", "advc_minus_adv_pp",
                                            "wilcoxon_p", "advc_within_class_probe", "adv_within_class_probe"]]
                tables.append(f"ADV peak F1 {float(meta['peak_f1']):.4f}; collapse lambda_max {meta['collapse_knob']}. ADV-C "
                              f"(oracle target labels, diagnostic only, never deployable) against ADV:\n\n```\n"
                              + tab.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n```\n"
                              + ("\nADV-CDAN (deployable, no target labels; no letter):\n\n```\n"
                                 + pd.DataFrame(cd).to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n```" if cd else ""))
        elif mode == "secondary":
            sec = secondary_table(_read(out_dir / "d6_secondary.csv"))
            sec.to_csv(out_dir / "D6_secondary_contrasts.csv", index=False)
            results["secondary"] = {"letter": "reported", "n_contrasts": int(len(sec))}
            tables.append("Paired over the 40 subjects, realization-averaged (seeds 42, 7, 123); no letter and no halt. The D6 arms "
                          "use batch 256 and KC-D1's use the run script's default, so the batch differs (as in C11).\n\n```\n"
                          + sec.to_string(index=False, float_format=lambda v: f"{v:.4g}") + "\n```")
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
            lines.append("\nSFC: the L2-normalised embedding the CORAL term sees is not unit norm (max |norm - 1| "
                         f"{results['manipulation_sfc']['normalised_norm_dev_max']:.2e}, tolerance 1e-5), so the "
                         "normalization is broken and the family stops (no Stage 2).")
        if "raw_norm_by_weight" in results["manipulation_sfc"]:
            lines.append("\nSFC raw penultimate norm by weight (descriptive, not a stop condition): "
                         + results["manipulation_sfc"]["raw_norm_by_weight"])
    if mode == "outcome":
        root = out_dir.parent.parent
        man = [f"{k}: {v['letter']}" for k, v in results.items() if k.startswith("manipulation_")]
        if not man:
            man = _bold_lines(root / "results/kc23_d6_manipulation_check" / "D6_VERDICT.md")
        outc = [f"{k[len('outcome_'):]}: {v['letter']}" for k, v in results.items() if k.startswith("outcome_")]
        mech = _bold_lines(root / "results/kc23_d6_mechanism_check" / "D6_VERDICT.md")
        lines.append("\n## Combined through-line matrix\n\n" + throughline_matrix(root, man, outc, mech))
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
