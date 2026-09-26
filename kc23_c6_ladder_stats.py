#!/usr/bin/env python3
"""
kc23_c6_ladder_stats.py
=========================
KC-C6 stats/gate. EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C6. The alignment
ladder on ENABL3S". n=10 subjects.

  R1: z-scoring (rung 3) beats 4lw in F1 for at least 7 of 10 subjects, AND
      centering does most of the linear-probe work                 -> over-alignment replicates in direction
  R2: otherwise                                                     -> non-replication

"Centering does most of the linear-probe work" (Enam, 26 September 2026):
    (probe at rung 0 minus probe at rung 1) / (probe at rung 0 minus probe at rung 3) >= 0.5
on ENABL3S, using the linear class-pooled subject-identity probe. Rung 1 is
mean-centering alone, rung 3 is mean plus scale.

Conformance (KC23_PREREG_CONFORMANCE.md): classify_r already matched the plan and this
operationalisation and is unchanged. What was wrong was the input. ladder_geometry.csv was
read from the ladder job's directory and nothing wrote it; kc23_c6_geometry.py now does
(the published metric code, ENABL3S features, rungs 0 to 4 plus 4lw and 4o). It is
required: a missing file, a missing rung (0, 1 or 3), a non-finite probe, or a missing F1
file exits 20 with a verdict that carries no outcome line. Rungs are matched by id as text,
so "4lw" and "3" cannot be confused with a row order.

With n=10, the plan states no significance claim is made unless it survives the KC-F1
family, so this reports counts and the reduction ratio, not a p-value gate. C6 has no
ESCALATE letter: a computed result exits 0.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import read_subjectwise, require_complete, print_gate_header, write_no_outcome_verdict

N_SUBJECTS = 10
CENTERING_FRACTION = 0.5
PROBE_COLUMN = "subject_probe_linear"      # the linear, class-pooled subject-identity probe


def classify_r(f1_rung3: np.ndarray, f1_4lw: np.ndarray,
               probe_rung0: float, probe_rung1: float, probe_rung3: float) -> tuple[str, dict]:
    n_beats = int((f1_rung3 > f1_4lw).sum())
    total_reduction = probe_rung0 - probe_rung3
    centering_reduction = probe_rung0 - probe_rung1
    centering_does_most = (centering_reduction / total_reduction) >= CENTERING_FRACTION if total_reduction != 0 else False
    letter = "R1" if (n_beats >= 7 and centering_does_most) else "R2"
    detail = {"n_beats_of_10": n_beats, "total_probe_reduction": total_reduction,
             "centering_probe_reduction": centering_reduction,
             "centering_fraction": (centering_reduction / total_reduction) if total_reduction else float("nan"),
             "centering_does_most": centering_does_most}
    return letter, detail


class InputError(Exception):
    pass


def load_probe_by_rung(path: Path) -> dict:
    if not path.exists():
        raise InputError(f"required input missing: {path}")
    g = pd.read_csv(path)
    if "rung" not in g.columns or PROBE_COLUMN not in g.columns:
        raise InputError(f"ladder_geometry.csv lacks 'rung' or '{PROBE_COLUMN}'")
    g["rung"] = g["rung"].astype(str)
    if g["rung"].duplicated().any():
        raise InputError("ladder_geometry.csv has a rung listed twice")
    by = g.set_index("rung")[PROBE_COLUMN]
    for r in ("0", "1", "3"):
        if r not in by.index:
            raise InputError(f"ladder_geometry.csv has no row for rung {r}")
        if not np.isfinite(by[r]):
            raise InputError(f"the linear probe at rung {r} is not finite")
    return {r: float(by[r]) for r in ("0", "1", "3")}


def _fail(out_dir: Path, reason: str) -> int:
    print(f"[C6] FAIL (no outcome computed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(out_dir / "C6_VERDICT.md", "KC-C6 verdict", reason)
    (out_dir / "C6_detail.csv").unlink(missing_ok=True)
    return 20


def run(out_dir: Path) -> int:
    try:
        probes = load_probe_by_rung(out_dir / "ladder_geometry.csv")
        try:
            f1_rung3 = require_complete(read_subjectwise(out_dir / "ladder_loso_3_SVM_subjectwise.csv"), N_SUBJECTS, "rung3")
            f1_4lw = require_complete(read_subjectwise(out_dir / "ladder_loso_4lw_SVM_subjectwise.csv"), N_SUBJECTS, "4lw")
        except (FileNotFoundError, ValueError) as e:
            raise InputError(str(e))
    except InputError as e:
        return _fail(out_dir, str(e))

    letter, detail = classify_r(f1_rung3, f1_4lw, probes["0"], probes["1"], probes["3"])
    reading = {
        "R1": "Over-alignment replicates in direction; the abstract may list it.",
        "R2": "Report non-replication. The abstract does not list it.",
    }[letter]
    print_gate_header("KC-C6", letter, reading)
    print(f"  {detail}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**detail, "probe_rung0": probes["0"], "probe_rung1": probes["1"], "probe_rung3": probes["3"]}]
                 ).to_csv(out_dir / "C6_detail.csv", index=False)
    (out_dir / "C6_VERDICT.md").write_text(
        f"# KC-C6 verdict\n\n**Outcome: {letter}**\n\n{reading}\n\n"
        f"Rung 3 beats 4lw in F1 for {detail['n_beats_of_10']} of {N_SUBJECTS} subjects (needs at least 7). "
        f"Linear class-pooled subject probe: rung 0 {probes['0']:.3f}, rung 1 {probes['1']:.3f}, rung 3 "
        f"{probes['3']:.3f} (chance 0.1). Centering removes {detail['centering_fraction']:.2f} of the total "
        f"reduction (needs at least {CENTERING_FRACTION}). With n = {N_SUBJECTS}, no significance claim is made "
        f"unless it survives the KC-F1 family.\n", encoding="utf-8")
    return 0  # C6 has no ESCALATE letter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
