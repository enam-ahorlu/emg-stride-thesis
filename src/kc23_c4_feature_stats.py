#!/usr/bin/env python3
"""
src/kc23_c4_feature_stats.py
==========================
KC-C4 stats/gate. docs/plans/EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C4. Richer established
feature sets", C4.3 and C4.4.

  F-A: both sets within 1 pt of Freq-72 under per-subject normalization, and the
       normalization gain present on both            -> the claim extends
  F-B: one set gains 1 to 2 pts                        -> report; the text notes it
  F-C: one set gains > 2 pts                           -> ESCALATE (could change the classical member)

Conformance pass, 26 September 2026 (docs/kc23/KC23_PREREG_CONFORMANCE.md):
  - The plan runs "SVM-X (the KC-C3 grid) and LDA" on each set. The comparator for the SVM-X reading is therefore SVM-X on
    Freq-72 (results/kc23_c3_svm_per_subject), not the published default-grid SVM (results/loso_freq_persubj) the earlier
    code used; the builder rows now run --grid extended --search grid. The LDA reading uses the published Freq-72 LDA
    (results/lda_persubj). The letter is the SVM-X reading (the classical member is the SVM); the LDA reading is computed
    the same way and reported beside it, and either one escalating escalates.
  - The plan's letters do not cover a set that loses more than 1 pt, or a set within 1 pt with no normalization gain
    (the earlier code returned F-B for both). Both are now "F-OUT (outside the pre-registered grid)", exit 10.
  - "Normalization gain present" is not defined in the plan. Operationalised here, and flagged for the author: per set,
    per-subject minus global F1 on the new set is positive and Holm-significant (Holm over the two sets, Wilcoxon).
A per-set gain "1 to 2" is 1 < gain <= 2 and "within 1 pt" is |gain| <= 1; ties at the boundaries follow the code's existing
convention. With both sets gaining in the F-B band the letter is still F-B ("one set" is read as "at least one").
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import (paired_test, holm, print_gate_header, read_subjectwise, require_complete,
                               write_no_outcome_verdict)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SETS = ["tdpsd54", "rich126"]
N_SUBJECTS = 40
# model -> (Freq-72 per-subject comparator dir, new-set dir pattern)
READINGS = {
    "SVM-X": ("results/kc23_c3_svm_per_subject", "results/kc23_c4_{feat}_svm_{norm}"),
    "LDA": ("results/lda_persubj", "results/kc23_c4_{feat}_lda_{norm}"),
}


def classify_set(f1_persubj_new: np.ndarray, f1_global_new: np.ndarray, f1_persubj_freq72: np.ndarray) -> dict:
    gain_vs_freq72 = paired_test(f1_persubj_new, f1_persubj_freq72, "new_minus_freq72", "C4")
    norm_gain = paired_test(f1_persubj_new, f1_global_new, "persubj_minus_global", "C4")
    delta_pp = gain_vs_freq72["delta_pp"]
    if abs(delta_pp) <= 1.0:
        set_letter = "within_1pt"
    elif 1.0 < delta_pp <= 2.0:
        set_letter = "gain_1_2pt"
    elif delta_pp > 2.0:
        set_letter = "gain_over_2pt"
    else:
        set_letter = "loss"
    return {"set_letter": set_letter, "delta_vs_freq72_pp": delta_pp, "norm_gain_pp": norm_gain["delta_pp"],
            "norm_gain_p_raw": norm_gain["p_raw"]}


def mark_norm_gain(results: dict[str, dict]) -> dict[str, dict]:
    """Holm across the sets, then 'present' = positive and Holm p < 0.05."""
    rows = holm([{"p_raw": r["norm_gain_p_raw"]} for r in results.values()])
    for r, h in zip(results.values(), rows):
        r["norm_gain_p_holm"] = h["p_holm"]
        r["norm_gain_present"] = bool(r["norm_gain_pp"] > 0 and h["p_holm"] < 0.05)
    return results


def classify_fa(results: dict[str, dict]) -> str:
    if any(r["set_letter"] == "gain_over_2pt" for r in results.values()):
        return "F-C"
    if any(r["set_letter"] == "gain_1_2pt" for r in results.values()):
        return "F-B"
    if all(r["set_letter"] == "within_1pt" and r.get("norm_gain_present", False) for r in results.values()):
        return "F-A"
    return "F-OUT"          # a set loses > 1 pt, or a set is within 1 pt with no normalization gain: no letter in the plan


def _load(d: Path, label: str) -> np.ndarray:
    return require_complete(read_subjectwise(d), N_SUBJECTS, label)


def run(out_dir: Path, root: Path | None = None) -> int:
    """Fail closed. Every input is the real sibling directory a C4 row writes; none is optional, and nothing stands in
    for a missing one (an earlier version fell back to the published 0.777)."""
    root = root or ROOT
    letters, tables, lines = {}, [], []
    try:
        for reading, (comp_dir, pattern) in READINGS.items():
            f72 = _load(root / comp_dir, f"freq72 {reading}")
            res = {}
            for feat in FEATURE_SETS:
                res[feat] = classify_set(_load(root / pattern.format(feat=feat, norm="per_subject"), f"{feat} persubj {reading}"),
                                         _load(root / pattern.format(feat=feat, norm="global"), f"{feat} global {reading}"), f72)
            res = mark_norm_gain(res)
            letters[reading] = classify_fa(res)
            for feat, r in res.items():
                tables.append({"reading": reading, "feature_set": feat, **r})
                lines.append(f"| {reading} | {feat} | {r['delta_vs_freq72_pp']:+.2f} | {r['set_letter']} | "
                             f"{r['norm_gain_pp']:+.2f} | {r['norm_gain_p_holm']:.3g} | {'yes' if r['norm_gain_present'] else 'no'} |")
    except (FileNotFoundError, ValueError) as e:
        print(f"[C4] FAIL (no outcome computed): {e}", file=sys.stderr)
        write_no_outcome_verdict(out_dir / "C4_VERDICT.md", "KC-C4 verdict", str(e))
        (out_dir / "C4_tests.csv").unlink(missing_ok=True)
        return 20

    letter = letters["SVM-X"]
    lda_letter = letters["LDA"]
    print_gate_header("KC-C4", letter, {
        "F-A": "The claim extends to established richer sets.", "F-B": "Report; the text notes it.",
        "F-C": "ESCALATE. Could change the classical member.",
        "F-OUT": "Outside the pre-registered grid: reported, not defaulted into a letter.",
    }[letter])
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tables).to_csv(out_dir / "C4_tests.csv", index=False)
    (out_dir / "C4_VERDICT.md").write_text(
        f"# KC-C4 verdict\n\n**Outcome: {letter}**\n\n"
        f"SVM-X reading (the classical member; comparator SVM-X on Freq-72): {letter}. LDA reading (comparator the published "
        f"Freq-72 LDA), computed the same way and reported, not used for the letter: {lda_letter}.\n\n"
        "| reading | set | gain vs Freq-72 (pt) | band | per-subject minus global (pt) | Holm p | normalization gain present |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n", encoding="utf-8")
    if "F-C" in letters.values():
        return 20
    if "F-OUT" in letters.values():
        return 10
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
