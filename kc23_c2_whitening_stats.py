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

from kc23_stats_common import (paired_test, print_gate_header, read_subjectwise, require_complete,
                               write_no_outcome_verdict)

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


VARIANTS = ["4b", "4c", "4d", "4lw"]     # the deployable whitening variants; 4lw is the pre-registered headline


def _fail(out_dir: Path, reason: str) -> int:
    print(f"[C2] FAIL (no outcome computed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(out_dir / "C2_VERDICT.md", "KC-C2 verdict", reason)
    (out_dir / "C2_tests.csv").unlink(missing_ok=True)
    return 20


def run(out_dir: Path, w250: bool = True) -> int:
    """Conformance pass, 26 September 2026: classify_w and classify_m already matched the plan and are unchanged. The
    plan's Endpoint 1 is "F1(rung 3) minus F1(EACH deployable whitening variant)", paired, with Wilcoxon, dz, BCa and
    subjects counted, letters conditioned on 4lw and REPORTED for 4b to 4d; only 4lw was computed. The subject probe
    under 4o ("because an oracle that keeps class structure and still removes subject identity would be the cleanest
    possible illustration") was not reported. Both are added; every input is required, and a missing one exits 20 with a
    verdict that carries no outcome line."""
    tag = "w250" if w250 else "w400"
    ladder_dir = out_dir
    try:
        f1_r3 = require_complete(read_subjectwise(ladder_dir / "ladder_loso_3_SVM_subjectwise.csv"), 40, "rung3")
        f1_var = {v: require_complete(read_subjectwise(ladder_dir / f"ladder_loso_{v}_SVM_subjectwise.csv"), 40, v)
                  for v in VARIANTS}
        f1_4o = require_complete(read_subjectwise(ladder_dir / "ladder_loso_4o_SVM_subjectwise.csv"), 40, "4o")
    except (FileNotFoundError, ValueError) as e:
        return _fail(out_dir, str(e))
    f1_4lw = f1_var["4lw"]

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

    # Endpoint 1 for EVERY deployable variant (the letter is descriptive for 4b to 4d; only 4lw governs the exit code)
    var_rows, var_lines = [], []
    for v in VARIANTS:
        lt, tt = classify_w(f1_r3, f1_var[v])
        var_rows.append({**tt, "variant": v, "letter_if_headline": lt})
        var_lines.append(f"| {v} | {tt['delta_pp']:+.2f} | {tt['p_raw']:.3g} | {tt['cohens_dz']:+.2f} | "
                         f"[{tt['bca_lo_pp']:+.2f}, {tt['bca_hi_pp']:+.2f}] | {tt['n_improved']}/{tt['n']} | {lt} |")
    # the subject probe under 4o, when the geometry rows exist (kc23_c6_geometry.py); said outright when they do not
    geo = out_dir / "ladder_geometry.csv"
    probe_line = "Subject probe under 4o: NOT reported, ladder_geometry.csv has not been produced for this directory."
    if geo.exists():
        g = pd.read_csv(geo)
        g["rung"] = g["rung"].astype(str)
        col = "subject_probe_linear"
        if col in g.columns and "4o" in set(g["rung"]):
            by = g.set_index("rung")[col]
            probe_line = (f"Subject probe under 4o (linear, class-pooled): {by['4o']:.3f}"
                          + (f", against rung 3 {by['3']:.3f} and rung 0 {by['0']:.3f}" if {"3", "0"} <= set(by.index) else "")
                          + f" (chance {float(g['chance_floor'].iloc[0]):.3f}).")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([w_stats, m_r3, m_lw]).to_csv(out_dir / "C2_tests.csv", index=False)
    pd.DataFrame(var_rows).to_csv(out_dir / "C2_variant_penalties.csv", index=False)
    verdict = f"""# KC-C2 verdict ({tag})

**Endpoint 1 outcome: {w_letter}**
**Endpoint 2 (mechanism) outcome: {m_letter}**

{reading}

{m_reading}

Endpoint 1 for every deployable variant, penalty = F1(rung 3) minus F1(variant), paired over 40 subjects
(the letter conditions on 4lw; the others are reported):

| variant | penalty (pt) | Wilcoxon p | dz | BCa 95% (pt) | subjects improved | letter if it were the headline |
|---|---|---|---|---|---|---|
{chr(10).join(var_lines)}

{probe_line}
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
