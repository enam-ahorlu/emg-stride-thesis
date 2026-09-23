#!/usr/bin/env python3
"""
kc23_c6_ladder_stats.py
=========================
KC-C6 stats/gate. EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C6. The alignment
ladder on ENABL3S". n=10 subjects.

  R1: z-scoring (rung 3) beats 4lw in F1 for at least 7 of 10 subjects, AND
      centering (rung 1) does most of the linear subject-identity-probe
      reduction (rung0 -> rung3)                    -> over-alignment replicates
  R2: otherwise                                       -> non-replication

"Centering does most of the work" is operationalized as: the probe reduction
from rung0 to rung1 (centering alone) is at least half of the total
reduction from rung0 to rung3 (mean+scale). This threshold is not given a
number in the plan; 50% is the natural reading of "most" and is stated here
explicitly as a judgment call.

With n=10, the plan states no significance claim is made unless it survives
the KC-F1 family -- this script reports counts and the reduction ratio, not
a p-value gate.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import read_subjectwise, require_complete, print_gate_header

N_SUBJECTS = 10


def classify_r(f1_rung3: np.ndarray, f1_4lw: np.ndarray,
               probe_rung0: float, probe_rung1: float, probe_rung3: float) -> tuple[str, dict]:
    n_beats = int((f1_rung3 > f1_4lw).sum())
    total_reduction = probe_rung0 - probe_rung3
    centering_reduction = probe_rung0 - probe_rung1
    centering_does_most = (centering_reduction / total_reduction) >= 0.5 if total_reduction != 0 else False
    letter = "R1" if (n_beats >= 7 and centering_does_most) else "R2"
    detail = {"n_beats_of_10": n_beats, "total_probe_reduction": total_reduction,
             "centering_probe_reduction": centering_reduction,
             "centering_fraction": (centering_reduction / total_reduction) if total_reduction else float("nan"),
             "centering_does_most": centering_does_most}
    return letter, detail


def run(out_dir: Path) -> int:
    f1_rung3 = require_complete(read_subjectwise(out_dir / "ladder_loso_3_SVM_subjectwise.csv"), N_SUBJECTS, "rung3")
    f1_4lw = require_complete(read_subjectwise(out_dir / "ladder_loso_4lw_SVM_subjectwise.csv"), N_SUBJECTS, "4lw")
    geometry = pd.read_csv(out_dir / "ladder_geometry.csv")  # rung, subject_probe_bal_acc
    g = geometry.set_index("rung")["subject_probe_bal_acc"]
    letter, detail = classify_r(f1_rung3, f1_4lw, float(g.loc[0]), float(g.loc[1]), float(g.loc[3]))

    reading = {
        "R1": "Over-alignment replicates in direction; the abstract may list it.",
        "R2": "Report non-replication. The abstract does not list it.",
    }[letter]
    print_gate_header("KC-C6", letter, reading)
    print(f"  {detail}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([detail]).to_csv(out_dir / "C6_detail.csv", index=False)
    (out_dir / "C6_VERDICT.md").write_text(f"# KC-C6 verdict\n\n**Outcome: {letter}**\n\n{reading}\n",
                                            encoding="utf-8")
    return 0  # C6 has no ESCALATE letter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
