#!/usr/bin/env python3
"""
kc23_s2_f0_feasibility.py
============================
KC-S2 F0 feasibility gate. EXPERIMENT_PLAN_KC23_DEPLOYMENT.md "KC-S2 ... F0
feasibility gate, first", S2.2.

A transition is the sample where the Mode label changes between two RETAINED
classes (WAK/UPS/DNS; STDUP is a synthetic sit-to-stand window, not a
locomotion mode transition endpoint here), with no dropped mode (ramps,
standing, sitting, stand-to-sit) in between. Four transition types counted
per subject: WAK->UPS, UPS->WAK, WAK->DNS, DNS->WAK.

  F-OK:       at least 5 of each type for at least 8 of 10 subjects -> run S2
  F-MARGINAL: fewer                                                  -> ask Enam
  F-X:        transitions not recoverable (e.g. a retained-class type is
              structurally absent, such as the ramp drop removing every
              stair entry for every subject)                          -> stop S2

count_transitions() re-derives the raw per-sample Mode sequence via
adapt_external_dataset.load_subject_trials (reused, not reimplemented) rather
than the windowed meta, since a transition is defined at sample granularity.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import print_gate_header

TRANSITION_TYPES = [("LW", "SA"), ("SA", "LW"), ("LW", "SD"), ("SD", "LW")]  # WAK<->UPS, WAK<->DNS
RETAINED_CODES = {"LW", "SA", "SD"}  # STDUP is not a transition endpoint (a sit-to-stand window, not a mode)
DROPPED_MODES = {0, 2, 3, 6}  # Sitting, RampAscent, RampDescent, Standing (see adapt_external_dataset.py)


def count_transitions_one_trial(mode: np.ndarray) -> dict[tuple[str, str], int]:
    """mode: raw per-sample Mode integer array for one circuit CSV (one trial).
    Returns counts of each retained-class-to-retained-class transition, with
    no dropped mode in between (a run of consecutive dropped-mode samples
    between two retained runs still counts as "in between" and voids the
    transition, per the plan's definition)."""
    from adapt_external_dataset import MODE_LW, MODE_SA, MODE_SD
    code_of = {MODE_LW: "LW", MODE_SA: "SA", MODE_SD: "SD"}
    # collapse to a run-length sequence of (code_or_None, start_idx)
    runs = []
    prev = None
    for m in mode:
        cur = code_of.get(int(m), None) if int(m) not in DROPPED_MODES else "DROPPED"
        if cur != prev:
            runs.append(cur)
            prev = cur
    counts = {t: 0 for t in TRANSITION_TYPES}
    last_retained = None
    for r in runs:
        if r is None or r == "DROPPED":
            last_retained = None  # a dropped or unknown mode voids the pending transition
            continue
        if last_retained is not None and last_retained != r and (last_retained, r) in counts:
            counts[(last_retained, r)] += 1
        last_retained = r
    return counts


def count_transitions(root: Path) -> pd.DataFrame:
    # adapt_external_dataset.load_subject_trials only yields the mapped
    # WAK/UPS/DNS/STDUP label string, not the raw per-sample Mode integer a
    # transition needs, so the raw Mode column is read directly here, per
    # circuit CSV -- reusing the module's column name and mode-code constants
    # (aed.MODE_COL, MODE_LW/SA/SD), not reimplementing the label mapping.
    import re
    import adapt_external_dataset as aed
    subj_dirs = sorted([d for d in root.glob("AB*") if d.is_dir()])
    for sd in subj_dirs:
        m = re.search(r"AB(\d+)", sd.name)
        if not m:
            continue
        sid = int(m.group(1))
        raw_dir = sd / "Raw"
        csvs = sorted(raw_dir.glob("*_raw.csv")) if raw_dir.exists() else sorted(sd.glob("**/*_raw.csv"))
        subj_counts = {t: 0 for t in TRANSITION_TYPES}
        for csv in csvs:
            df = pd.read_csv(csv, usecols=[aed.MODE_COL])
            c = count_transitions_one_trial(df[aed.MODE_COL].to_numpy())
            for t in TRANSITION_TYPES:
                subj_counts[t] += c[t]
        rows.append({"subject": sid, **{f"{a}_to_{b}": v for (a, b), v in subj_counts.items()}})
    return pd.DataFrame(rows)


def classify_f0(counts_df: pd.DataFrame, min_count: int = 5, min_subjects: int = 8) -> tuple[str, dict]:
    type_cols = [c for c in counts_df.columns if c != "subject"]
    ok_matrix = counts_df[type_cols] >= min_count
    subjects_ok_per_type = ok_matrix.sum(axis=0)
    n_subjects = len(counts_df)
    all_types_ok = (subjects_ok_per_type >= min_subjects).all()

    structurally_absent = any((counts_df[c] == 0).all() for c in type_cols)

    if structurally_absent:
        letter = "F-X"
    elif all_types_ok:
        letter = "F-OK"
    else:
        letter = "F-MARGINAL"
    detail = {"n_subjects": n_subjects, "subjects_ok_per_type": subjects_ok_per_type.to_dict(),
             "structurally_absent": structurally_absent}
    return letter, detail


def run(out_dir: Path, root: Path | None = None) -> int:
    counts_path = out_dir / "s2_transition_counts.csv"
    if root is not None:
        df = count_transitions(root)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(counts_path, index=False)
    else:
        df = pd.read_csv(counts_path)

    letter, detail = classify_f0(df)
    reading = {"F-OK": "Run S2.", "F-MARGINAL": "Report the counts and ask Enam before running.",
              "F-X": "Stop S2. The limitation stays in Section 5.3."}[letter]
    print_gate_header("KC-S2 F0", letter, reading)
    print(f"  {detail}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "S2_F0_VERDICT.md").write_text(f"# KC-S2 F0 verdict\n\n**Outcome: {letter}**\n\n{reading}\n",
                                               encoding="utf-8")
    if letter == "F-MARGINAL":
        return 10
    if letter == "F-X":
        return 20
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="ENABL3S raw root (5362627); omit to reuse an existing counts csv")
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.root) if args.root else None))


if __name__ == "__main__":
    main()
