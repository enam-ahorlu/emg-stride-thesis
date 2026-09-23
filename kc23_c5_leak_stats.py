#!/usr/bin/env python3
"""
kc23_c5_leak_stats.py
=======================
KC-C5 stats/gate. EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C5. Decomposing the
subject-dependent 'leak'", C5.4.

Plateau guard g*: the smallest g in {1,2,4,8,16} at which B-g changes by less
than 0.5pt from B-(2g). No plateau by g=16 is itself outcome L3.

Delta_overlap = P50 - P0
Delta_autocorr = P0 - I-g*
Delta_drift = I-g* - B-g*

  L1: Delta_overlap >= 60% of (P50 - B-g*)         -> "overlap leak" stands, sized
  L2: Delta_drift >= 40% of the total (P50 - B-g*)  -> split-protocol effect, named components
  L3: no plateau by g=16, OR (g* != 1 AND B-g* differs from the published blocked
      SD figure by > 1pt)                            -> ESCALATE

L1 and L2 are not mutually exclusive (both can fire); L3 overrides both when
it fires (the decomposition itself is untrustworthy).

Published published_blocked_sd: the pre-KC23 published movement-blocked SD
figure (g implicitly 1, from results_b8_sd), passed in or read from the
existing results_b8_sd/b8_base_w250_compare.csv sd_new_blocked column
(mean over models given, or a single value for one model).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import paired_test, print_gate_header


def find_plateau(b_by_g: dict[int, np.ndarray], guards=(1, 2, 4, 8, 16)) -> tuple[int | None, dict]:
    """b_by_g: {guard: per-subject F1 array}. Returns (g_star or None, detail)."""
    detail = {}
    for i, g in enumerate(guards[:-1]):
        g2 = guards[i + 1]
        if g not in b_by_g or g2 not in b_by_g:
            continue
        change_pp = abs(b_by_g[g2].mean() - b_by_g[g].mean()) * 100.0
        detail[g] = change_pp
        if change_pp < 0.5:
            return g, detail
    return None, detail


def classify_l(p50: np.ndarray, p0: np.ndarray, i_gstar: np.ndarray, b_gstar: np.ndarray,
              g_star: int | None, published_blocked_sd: float | None) -> tuple[list[str], dict]:
    letters = []
    total_pp = (p50.mean() - b_gstar.mean()) * 100.0 if g_star is not None else float("nan")
    overlap_pp = (p50.mean() - p0.mean()) * 100.0
    autocorr_pp = (p0.mean() - i_gstar.mean()) * 100.0 if g_star is not None else float("nan")
    drift_pp = (i_gstar.mean() - b_gstar.mean()) * 100.0 if g_star is not None else float("nan")

    if g_star is None:
        letters.append("L3")
    else:
        if published_blocked_sd is not None and g_star != 1:
            gap = abs(b_gstar.mean() - published_blocked_sd) * 100.0
            if gap > 1.0:
                letters.append("L3")
        if total_pp and total_pp != 0 and not np.isnan(total_pp):
            if overlap_pp >= 0.60 * total_pp:
                letters.append("L1")
            if drift_pp >= 0.40 * total_pp:
                letters.append("L2")

    detail = {"g_star": g_star, "total_pp": total_pp, "overlap_pp": overlap_pp,
             "autocorr_pp": autocorr_pp, "drift_pp": drift_pp}
    return letters, detail


def run(out_dir: Path, published_blocked_sd: float | None = None) -> int:
    from kc23_stats_common import read_subjectwise, require_complete
    p50 = require_complete(read_subjectwise(out_dir / "p50_subjectwise.csv"), 40, "P50")
    p0 = require_complete(read_subjectwise(out_dir / "p0_subjectwise.csv"), 40, "P0")
    b_by_g = {}
    for g in (1, 2, 4, 8, 16):
        p = out_dir / f"b{g}_subjectwise.csv"
        if p.exists():
            b_by_g[g] = require_complete(read_subjectwise(p), 40, f"B-{g}")
    i_by_g = {}
    for g in (1, 4, 16):
        p = out_dir / f"i{g}_subjectwise.csv"
        if p.exists():
            i_by_g[g] = require_complete(read_subjectwise(p), 40, f"I-{g}")

    g_star, plateau_detail = find_plateau(b_by_g)
    print(f"[C5] plateau detail (delta pp between consecutive guards): {plateau_detail}, g*={g_star}")

    if g_star is not None and g_star in i_by_g and g_star in b_by_g:
        letters, detail = classify_l(p50, p0, i_by_g[g_star], b_by_g[g_star], g_star, published_blocked_sd)
    else:
        letters, detail = (["L3"], {"g_star": g_star, "note": "no matching I-g or B-g for g_star"})

    reading = []
    if "L1" in letters:
        reading.append("Overlap leak stands as named, with its size.")
    if "L2" in letters:
        reading.append("Reported as a split-protocol effect with components; 'leak' limited to overlap.")
    if "L3" in letters:
        reading.append("ESCALATE. Table 4.2 and the abstract's gap figure change.")
    if not letters:
        reading.append("Neither L1 nor L2 threshold met; report the raw decomposition.")

    print_gate_header("KC-C5", ",".join(letters) or "none", " ".join(reading))
    print(f"  {detail}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([detail]).to_csv(out_dir / "C5_decomposition.csv", index=False)
    (out_dir / "C5_VERDICT.md").write_text(
        f"# KC-C5 verdict\n\n**Outcome(s): {letters}**\n\n{' '.join(reading)}\n", encoding="utf-8")

    return 20 if "L3" in letters else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--published-blocked-sd", type=float, default=None)
    args = ap.parse_args()
    sys.exit(run(Path(args.out), args.published_blocked_sd))


if __name__ == "__main__":
    main()
