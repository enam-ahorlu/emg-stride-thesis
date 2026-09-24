#!/usr/bin/env python3
"""
kc23_d1_replicate_stats.py
=============================
KC-D1 stats/gate. EXPERIMENT_PLAN_KC23_DEEP.md "KC-D1. The replicate
programme", D1.4 to D1.7. The long pole of the programme: the reproduction
gate, 17 registered contrasts (C1-C17), the KC-D6-adjacent descriptive
stacking row C13b (decision D-6a, KC23_HALT.md), and the headline gate.

D1.4 reproduction gate (seed 42 re-run only): R2 within +/-1.5pt of 0.8395,
R1 within +/-1.5pt of 0.782, R10 pre-adaptation within +/-1.5pt of 0.787 or
0.772. FAIL -> ESCALATE ("code drift since publication").

D1.5 per contrast:
  ESTABLISHED:    the realization-averaged Wilcoxon survives BH within the
                   KC-D1 family, AND the same sign holds in >= 4/5
                   realizations (Tier A) or 3/3 (Tier B)
  AMBIGUOUS:       exactly one of the two holds
  NOT ESTABLISHED: neither holds
Null contrasts (C15, the C16 plateau pairs) use TOST equivalence (+/-1.0pt)
instead of a two-sided difference test.

D1.6 headline gate: published R2/ensemble/R12/global values within the
realization mean +/- 2SD -> H1 (stays as version of record); outside -> H2
(ESCALATE).

D1.7: C1 or C12 landing NOT ESTABLISHED -> ESCALATE. Any other established
contrast landing AMBIGUOUS/NOT ESTABLISHED is reported, not halted.

C13b (DECISION D-6a, KC23_HALT.md, 24 September 2026): a DESCRIPTIVE row
beside C13, comparing the published stacking combiner (SVM + ResNet-SE+CD,
logistic-regression meta-learner fit on the other 39 subjects) against the
soft-vote ensemble, read against the measured realization SD -- not a
registered hypothesis, no ESTABLISHED/AMBIGUOUS label, no gate.

Scope note: D1.5 item 3 ("secondary model: F1 ~ arm + (1|subject) + (1|seed)
via statsmodels MixedLM") is NOT computed by this script. The pre-registered
ESTABLISHED/AMBIGUOUS/NOT-ESTABLISHED verdict depends only on items 1
(realization-averaged Wilcoxon + BH) and 2 (seed-level sign consistency), so
this omission does not affect any letter here; the MixedLM fit is a
supplementary cross-check the real KC-D1 write-up should still add before
Table 4.6/4.8/4.9 text is finalized, and it is flagged here rather than
silently skipped.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from kc23_stats_common import cohens_d_paired, bca_ci, tost_equivalence, print_gate_header

PUBLISHED_R2 = 0.8395
PUBLISHED_R1 = 0.782
PUBLISHED_R10_PRE = (0.787, 0.772)  # either published figure is an acceptable gate target
REPRO_TOL = 0.015

PUBLISHED_ENSEMBLE = 0.858
PUBLISHED_R12 = 0.860
PUBLISHED_GLOBAL = 0.772

NULL_CONTRASTS = {"C15", "C16"}

CONTRASTS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
            "C12", "C13", "C14", "C15", "C16", "C17"]


def reproduction_gate(r1_mean: float, r2_mean: float, r10_pre_mean: float) -> tuple[str, dict]:
    ok_r2 = abs(r2_mean - PUBLISHED_R2) <= REPRO_TOL
    ok_r1 = abs(r1_mean - PUBLISHED_R1) <= REPRO_TOL
    ok_r10 = any(abs(r10_pre_mean - p) <= REPRO_TOL for p in PUBLISHED_R10_PRE)
    ok = ok_r2 and ok_r1 and ok_r10
    return ("PASS" if ok else "FAIL"), {"r2_ok": ok_r2, "r1_ok": ok_r1, "r10_ok": ok_r10,
                                        "r2_mean": r2_mean, "r1_mean": r1_mean, "r10_pre_mean": r10_pre_mean}


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    return out.tolist()


def classify_contrast(realization_avg_diff: np.ndarray, per_realization_sign_agree: int,
                      n_realizations: int, p_bh: float, tier: str, is_null: bool = False) -> tuple[str, dict]:
    if is_null:
        eq = tost_equivalence(realization_avg_diff, np.zeros_like(realization_avg_diff))
        letter = "EQUIVALENT" if eq["equivalent"] else "NOT_EQUIVALENT"
        return letter, eq

    need_agree = 4 if tier == "A" else 3
    survives_bh = p_bh < 0.05
    sign_consistent = per_realization_sign_agree >= need_agree

    if survives_bh and sign_consistent:
        letter = "ESTABLISHED"
    elif survives_bh or sign_consistent:
        letter = "AMBIGUOUS"
    else:
        letter = "NOT ESTABLISHED"
    return letter, {"survives_bh": survives_bh, "sign_consistent": sign_consistent,
                    "n_realizations_agree": per_realization_sign_agree, "n_realizations": n_realizations,
                    "p_bh": p_bh}


def classify_headline(published: float, realization_mean: float, realization_sd: float) -> tuple[str, dict]:
    lo, hi = realization_mean - 2 * realization_sd, realization_mean + 2 * realization_sd
    within = lo <= published <= hi
    return ("H1" if within else "H2"), {"published": published, "realization_mean": realization_mean,
                                        "realization_sd": realization_sd, "lo": lo, "hi": hi, "within": within}


def stacking_descriptive(f1_stacking: np.ndarray, f1_softvote: np.ndarray, realization_sd: float) -> dict:
    diff = f1_stacking - f1_softvote
    edge_pp = float(diff.mean() * 100.0)
    return {"stacking_mean": float(f1_stacking.mean()), "softvote_mean": float(f1_softvote.mean()),
           "edge_pp": edge_pp, "realization_sd_pp": realization_sd * 100.0,
           "edge_within_run_variance": abs(edge_pp) < realization_sd * 100.0}


def run(out_dir: Path) -> int:
    exit_code = 0

    # Fixed 2026-09-24: this letter used to be computed, used to set exit_code,
    # and then silently dropped -- the local `letter` name was reused (and
    # overwritten) by the per-contrast and per-headline-arm loops below, so
    # D1_VERDICT.md never recorded it even though the gate had genuinely run
    # and genuinely passed or failed. Found live: results_kc23_d1_repro_check/
    # D1_VERDICT.md existed with only its header, no letter, even though the
    # job had actually run and produced real reproduction_inputs.
    repro_letter, repro_detail = None, None
    gate_path = out_dir / "d1_reproduction_inputs.csv"
    if gate_path.exists():
        g = pd.read_csv(gate_path).set_index("arm")["f1_mean"]
        repro_letter, repro_detail = reproduction_gate(float(g["R1"]), float(g["R2"]), float(g["R10_pre"]))
        print_gate_header("KC-D1 reproduction gate", repro_letter,
                          "Continue." if repro_letter == "PASS" else "ESCALATE: code drift since publication.")
        print(f"  {repro_detail}")
        if repro_letter == "FAIL":
            exit_code = 20

    contrast_results = {}
    contrasts_path = out_dir / "d1_contrasts.csv"  # cols: contrast, tier, realization, subject, diff
    if contrasts_path.exists() and exit_code != 20:
        df = pd.read_csv(contrasts_path)
        rows = []
        for cname, g in df.groupby("contrast"):
            tier = g["tier"].iloc[0]
            is_null = cname in NULL_CONTRASTS
            avg = g.groupby("subject")["diff"].mean().to_numpy()
            if is_null:
                letter, detail = classify_contrast(avg, 0, 0, 1.0, tier, is_null=True)
                rows.append({"contrast": cname, "tier": tier, "letter": letter, **detail})
                continue
            if np.allclose(avg, 0):
                p_raw = 1.0
            else:
                p_raw = float(stats.wilcoxon(avg, zero_method="wilcox", alternative="two-sided").pvalue)
            by_real = g.groupby("realization")["diff"].mean()
            n_agree = int((np.sign(by_real) == np.sign(by_real.mean())).sum())
            rows.append({"contrast": cname, "tier": tier, "p_raw": p_raw, "n_agree": n_agree,
                        "n_realizations": len(by_real), "mean_diff_pp": float(avg.mean() * 100),
                        "cohens_dz": cohens_d_paired(avg, np.zeros_like(avg))})
        # BH within the whole KC-D1 family of non-null contrasts
        nonnull = [r for r in rows if "p_raw" in r]
        if nonnull:
            p_bh = benjamini_hochberg([r["p_raw"] for r in nonnull])
            for r, pb in zip(nonnull, p_bh):
                letter, detail = classify_contrast(None, r["n_agree"], r["n_realizations"], pb, r["tier"])
                r["p_bh"] = pb
                r["letter"] = letter
        for r in rows:
            contrast_results[r["contrast"]] = r
            print_gate_header(f"KC-D1 {r['contrast']}", r["letter"], "")

        for critical in ("C1", "C12"):
            if critical in contrast_results and contrast_results[critical]["letter"] == "NOT ESTABLISHED":
                exit_code = 20
                print(f"[D1] {critical} NOT ESTABLISHED -> ESCALATE (Finding C's core changes).")

    headline_path = out_dir / "d1_headline_inputs.csv"
    headline_checks = {}
    if headline_path.exists() and exit_code != 20:
        h = pd.read_csv(headline_path)
        checks = headline_checks
        for arm, published in [("R2", PUBLISHED_R2), ("ensemble", PUBLISHED_ENSEMBLE),
                               ("R12", PUBLISHED_R12), ("global", PUBLISHED_GLOBAL)]:
            row = h[h["arm"] == arm]
            if len(row):
                letter, detail = classify_headline(published, float(row["realization_mean"].iloc[0]),
                                                    float(row["realization_sd"].iloc[0]))
                checks[arm] = {"letter": letter, **detail}
                print_gate_header(f"KC-D1 headline ({arm})", letter, "")
        if any(v["letter"] == "H2" for v in checks.values()):
            exit_code = 20

    stacking_path = out_dir / "d1_stacking_c13b.csv"
    stacking_result = None
    if stacking_path.exists():
        s = pd.read_csv(stacking_path)
        stacking_result = stacking_descriptive(s["f1_stacking"].to_numpy(), s["f1_softvote"].to_numpy(),
                                               float(s["realization_sd"].iloc[0]))
        print(f"[D1] C13b (descriptive, decision D-6a): {stacking_result}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if contrast_results:
        pd.DataFrame(list(contrast_results.values())).to_csv(out_dir / "D1_contrasts_verdict.csv", index=False)
    if stacking_result:
        pd.DataFrame([stacking_result]).to_csv(out_dir / "D1_C13b_stacking.csv", index=False)

    verdict_lines = ["# KC-D1 verdict\n"]
    if repro_letter is not None:
        verdict_lines.append(f"- **reproduction: {repro_letter}**")
    for k, v in contrast_results.items():
        verdict_lines.append(f"- **{k}: {v['letter']}**")
    for arm, v in headline_checks.items():
        verdict_lines.append(f"- **headline ({arm}): {v['letter']}**")
    if not (repro_letter is not None or contrast_results or headline_checks):
        verdict_lines.append("(no inputs present yet)")
    (out_dir / "D1_VERDICT.md").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    return exit_code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
