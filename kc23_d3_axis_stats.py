#!/usr/bin/env python3
"""
kc23_d3_axis_stats.py
=======================
KC-D3 stats/gate. EXPERIMENT_PLAN_KC23_DEEP.md "KC-D3. Axis against magnitude
of the augmentation", D3.3. Realization-averaged (3 realizations: 42 re-run,
7, 123). "X >= R3 - 1" means X's mean F1 is within 1pt of R3 or above it.

  A1: X2 <= R1 + 1pt, AND X3 and X4 both < R3 - 1.5pt  -> per-channel
      multiplicative is the active axis (Section 4.7's rule stands)
  A2: X2 >= R3 - 1pt   -> noise helps at matched magnitude; "wrong axis" withdrawn
  A3: X3 >= R3 - 1pt   -> "multiplicative" unsupported
  A4: X4 >= R3 - 1pt   -> per-channel independence not needed

A2, A3, A4 can co-occur with each other and are independent of A1 (A1 is the
clean positive result; A2-A4 are each an erosion of some part of the claim).
None halt the programme -- always exit 0, letters are purely descriptive.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

from kc23_stats_common import print_gate_header


def classify_a(r1: float, r3: float, x1: float, x2: float, x3: float, x4: float) -> list[str]:
    fired = []
    if x2 <= r1 + 0.01 and x3 < r3 - 0.015 and x4 < r3 - 0.015:
        fired.append("A1")
    if x2 >= r3 - 0.01:
        fired.append("A2")
    if x3 >= r3 - 0.01:
        fired.append("A3")
    if x4 >= r3 - 0.01:
        fired.append("A4")
    return fired


def run(out_dir: Path) -> int:
    means = pd.read_csv(out_dir / "d3_realization_means.csv").set_index("arm")["f1_mean"]
    r1, r3 = float(means["R1"]), float(means["R3"])
    x1, x2, x3, x4 = float(means["X1"]), float(means["X2"]), float(means["X3"]), float(means["X4"])
    fired = classify_a(r1, r3, x1, x2, x3, x4)

    readings = {
        "A1": "Per-channel multiplicative is the active axis, now with magnitude-matched evidence.",
        "A2": "Noise helps at matched magnitude; the 'wrong axis' claim is withdrawn.",
        "A3": "'Multiplicative' is unsupported; the claim becomes per-channel perturbation.",
        "A4": "Per-channel independence is not needed.",
    }
    print_gate_header("KC-D3", ",".join(fired) or "none",
                      " ".join(readings[l] for l in fired) or "No letter fired.")
    print(f"  R1={r1:.4f} R3={r3:.4f} X1={x1:.4f} X2={x2:.4f} X3={x3:.4f} X4={x4:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"letters": ",".join(fired), "R1": r1, "R3": r3, "X1": x1, "X2": x2, "X3": x3, "X4": x4}]
                ).to_csv(out_dir / "D3_detail.csv", index=False)
    (out_dir / "D3_VERDICT.md").write_text(
        f"# KC-D3 verdict\n\n**Outcome(s): {fired}**\n\n" + "\n".join(readings[l] for l in fired) + "\n",
        encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
