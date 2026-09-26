#!/usr/bin/env python3
"""
kc23_c4_feature_stats.py
==========================
KC-C4 stats/gate. EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C4. Richer established
feature sets", C4.4.

  F-A: both sets (TDPSD-54, Rich-126) within 1pt of Freq-72 under per-subject
       normalization, AND the normalization gain (per-subject minus global)
       present on both -> the claim extends
  F-B: one set gains 1 to 2pt over Freq-72                -> report
  F-C: one set gains > 2pt over Freq-72                   -> ESCALATE

"Gains" is the per-subject-normalization mean F1 delta of the new set minus
Freq-72's per-subject mean F1 (both already-published or freshly run at
--grid default under KC-C3's harness). Per set, the more severe letter wins
if both F-B and F-C conditions could apply (a >2pt gain also satisfies "1 to
2pt or more", so F-C is checked first).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import (paired_test, print_gate_header, read_subjectwise, require_complete,
                               write_no_outcome_verdict)

ROOT = Path(__file__).resolve().parent
FREQ72_DIR = "results_loso_freq_persubj"    # the published Freq-72 per-subject SVM run (a real directory, not a number)
FEATURE_SETS = ["tdpsd54", "rich126"]


def classify_set(f1_persubj_new: np.ndarray, f1_global_new: np.ndarray,
                 f1_persubj_freq72: np.ndarray) -> dict:
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
    return {"set_letter": set_letter, "delta_vs_freq72_pp": delta_pp,
           "norm_gain_pp": norm_gain["delta_pp"], "norm_gain_significant": norm_gain["p_raw"] < 0.05}


def classify_fa(results: dict[str, dict]) -> str:
    if any(r["set_letter"] == "gain_over_2pt" for r in results.values()):
        return "F-C"
    if any(r["set_letter"] == "gain_1_2pt" for r in results.values()):
        return "F-B"
    all_within_1pt = all(r["set_letter"] == "within_1pt" for r in results.values())
    all_norm_gain = all(r["norm_gain_pp"] > 0 for r in results.values())
    if all_within_1pt and all_norm_gain:
        return "F-A"
    return "F-B"  # within-1pt but no norm gain on one set, or a mild mixed case: report, don't escalate


def run(out_dir: Path) -> int:
    """Fail closed (rewritten 2026-09-25). The previous version (a) fell back
    to the published 0.777 for Freq-72 when its comparator directory was
    missing, (b) looked for '<feat>_svm_persubj_subjectwise.csv' in its OWN
    out_dir, a name no job writes, and (c) printed 'no results' and exited 0
    when it found none, so the gate could never compute a letter and never
    said so. Inputs are now the real sibling directories the C4 rows write,
    located relative to this script (the queue passes only --out); every one
    is required, and nothing stands in for a missing one."""
    needed = [("freq72", ROOT / FREQ72_DIR)]
    for feat in FEATURE_SETS:
        needed.append((f"{feat}_persubj", ROOT / f"results_kc23_c4_{feat}_svm_per_subject"))
        needed.append((f"{feat}_global", ROOT / f"results_kc23_c4_{feat}_svm_global"))
    try:
        f1 = {label: require_complete(read_subjectwise(d, model_token="SVM"), 40, label) for label, d in needed}
    except (FileNotFoundError, ValueError) as e:
        print(f"[C4] FAIL (no outcome computed): {e}", file=sys.stderr)
        write_no_outcome_verdict(out_dir / "C4_VERDICT.md", "KC-C4 verdict", str(e))
        (out_dir / "C4_tests.csv").unlink(missing_ok=True)
        return 20

    results = {feat: classify_set(f1[f"{feat}_persubj"], f1[f"{feat}_global"], f1["freq72"])
               for feat in FEATURE_SETS}
    letter = classify_fa(results)
    print_gate_header("KC-C4", letter, {
        "F-A": "The claim extends to established richer sets.",
        "F-B": "Report; the text notes it.",
        "F-C": "ESCALATE. Could change the classical member.",
    }[letter])
    for feat, r in results.items():
        print(f"  {feat}: delta={r['delta_vs_freq72_pp']:+.3f}pp, norm_gain={r['norm_gain_pp']:+.3f}pp")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"feature_set": k, **v} for k, v in results.items()]).to_csv(out_dir / "C4_tests.csv", index=False)
    (out_dir / "C4_VERDICT.md").write_text(f"# KC-C4 verdict\n\n**Outcome: {letter}**\n", encoding="utf-8")

    return 20 if letter == "F-C" else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
