#!/usr/bin/env python3
"""
kc23_c2_whitening_stats.py
============================
KC-C2 stats/gate. EXPERIMENT_PLAN_KC23_CLASSICAL.md section "KC-C2. Whitening
under principled regularizers, plus an oracle mechanism test", C2.4.

Endpoint 1 (headline variant 4lw, pre-registered): F1(rung 3) - F1(4lw),
paired over 40 subjects.
  W1: penalty >= 5 pts, significant       -> the finding stands, corrected size
  W2: penalty 1 to 5 pts, significant     -> ordering holds, size revised
  W3: penalty < 1 pt or not significant   -> ESCALATE

Endpoint 2 (mechanism, 4o oracle against rung 3 and 4lw):
  M1: 4o >= F1(rung3) - 1pt, AND 4o - 4lw >= 2pt  -> mechanism supported
  M2: 4o falls about as far as 4lw (|4o - 4lw| < 1pt) -> mechanism unsupported
  M3: in between                                       -> partial support

Classification functions are pure (take numpy arrays), so tests_kc23/ can
drive every letter with synthetic per-subject F1 vectors.

Exit codes: W1/W2 -> 0, M2/M3 -> 0; W3 -> 20 (ESCALATE); M1 is reported (10)
since it is the "cleanest possible illustration for Section 4.6" per the plan.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import paired_test, print_gate_header, read_subjectwise, require_complete

ROOT = Path(__file__).resolve().parent


def classify_w(f1_rung3: np.ndarray, f1_4lw: np.ndarray) -> tuple[str, dict]:
    t = paired_test(f1_rung3, f1_4lw, "rung3_minus_4lw", "C2-W")
    penalty_pp = t["delta_pp"]  # rung3 - 4lw; positive = 4lw worse (a penalty)
    significant = t["p_raw"] < 0.05
    if penalty_pp >= 5.0 and significant:
        letter = "W1"
    elif 1.0 <= penalty_pp < 5.0 and significant:
        letter = "W2"
    else:
        letter = "W3"
    return letter, t


def classify_m(f1_rung3: np.ndarray, f1_4o: np.ndarray, f1_4lw: np.ndarray) -> tuple[str, dict, dict]:
    t_r3 = paired_test(f1_4o, f1_rung3, "4o_minus_rung3", "C2-M")
    t_lw = paired_test(f1_4o, f1_4lw, "4o_minus_4lw", "C2-M")
    oracle_vs_rung3 = t_r3["delta_pp"]     # 4o - rung3, in pp
    oracle_vs_4lw = t_lw["delta_pp"]       # 4o - 4lw, in pp
    if oracle_vs_rung3 >= -1.0 and oracle_vs_4lw >= 2.0:
        letter = "M1"
    elif abs(oracle_vs_4lw) < 1.0:
        letter = "M2"
    else:
        letter = "M3"
    return letter, t_r3, t_lw


def run(out_dir: Path, w250: bool = True) -> int:
    tag = "w250" if w250 else "w400"
    ladder_dir = out_dir
    f1_r3 = require_complete(read_subjectwise(ladder_dir / "ladder_loso_3_SVM_subjectwise.csv"), 40, "rung3")
    f1_4lw = require_complete(read_subjectwise(ladder_dir / "ladder_loso_4lw_SVM_subjectwise.csv"), 40, "4lw")
    f1_4o = require_complete(read_subjectwise(ladder_dir / "ladder_loso_4o_SVM_subjectwise.csv"), 40, "4o")

    w_letter, w_stats = classify_w(f1_r3, f1_4lw)
    m_letter, m_r3, m_lw = classify_m(f1_r3, f1_4o, f1_4lw)

    reading = {
        "W1": "The finding stands with a corrected size. Table 4.7 reports 4lw.",
        "W2": "The ordering claim holds, the size claim is revised.",
        "W3": "ESCALATE. Over-alignment becomes a regularization artifact.",
    }[w_letter]
    print_gate_header(f"KC-C2 ({tag}) Endpoint 1", w_letter, reading)
    print(f"  penalty(rung3-4lw) = {w_stats['delta_pp']:+.3f} pp, p={w_stats['p_raw']:.4g}, "
         f"dz={w_stats['cohens_dz']:+.3f}")

    m_reading = {
        "M1": "Mechanism supported: the damage comes from removing class-pooled covariance.",
        "M2": "Mechanism unsupported: the damage is estimation or something else.",
        "M3": "Partial support.",
    }[m_letter]
    print_gate_header(f"KC-C2 ({tag}) Endpoint 2 (mechanism)", m_letter, m_reading)
    print(f"  4o-rung3 = {m_r3['delta_pp']:+.3f} pp, 4o-4lw = {m_lw['delta_pp']:+.3f} pp")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([w_stats, m_r3, m_lw]).to_csv(out_dir / "C2_tests.csv", index=False)
    verdict = f"""# KC-C2 verdict ({tag})

**Endpoint 1 outcome: {w_letter}**
**Endpoint 2 (mechanism) outcome: {m_letter}**

{reading}

{m_reading}
"""
    (out_dir / "C2_VERDICT.md").write_text(verdict, encoding="utf-8")

    if w_letter == "W3":
        return 20
    if m_letter == "M1":
        return 10
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    # The window tag was hard-wired to w250 here, so the w400 directory's verdict was labelled (w250). The queue
    # passes only --out, so the label comes from the directory name (results_kc23_c2_whitening_w250 / _w400).
    sys.exit(run(out, w250=not out.name.endswith("w400")))


if __name__ == "__main__":
    main()
