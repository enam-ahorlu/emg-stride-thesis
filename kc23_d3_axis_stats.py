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

A2, A3, A4 can co-occur with each other and are independent of A1. "Report every
letter that fires. None of them halts the programme": always exit 0.

Conformance (26 September 2026, KC23_PREREG_CONFORMANCE.md): classify_a already
matched the plan clause by clause and is unchanged. What changed is the I/O. The
input d3_realization_means.csv is now built by kc23_d3_aggregate.py (nothing
produced it before); it is REQUIRED, must hold exactly the six arms with 3
realizations each, and a missing or incomplete input exits 20 with a verdict that
has no outcome line. The verdict also reports X1 against R1 (the published
Gaussian setting, "now replicated") and each arm's SD across realizations, and
states "none fired" instead of an empty list when no letter fires.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

from kc23_stats_common import print_gate_header, write_no_outcome_verdict

ARMS = ["R1", "R3", "X1", "X2", "X3", "X4"]
N_REALIZATIONS = 3


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


READINGS = {
    "A1": "Per-channel multiplicative is the active axis, now with magnitude-matched evidence.",
    "A2": "Noise helps at matched magnitude; the 'wrong axis' claim is withdrawn.",
    "A3": "'Multiplicative' is unsupported; the claim becomes per-channel perturbation.",
    "A4": "Per-channel independence is not needed.",
}


def _fail(out_dir: Path, reason: str) -> int:
    print(f"[D3] FAIL (no outcome computed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(out_dir / "D3_VERDICT.md", "KC-D3 verdict", reason)
    (out_dir / "D3_detail.csv").unlink(missing_ok=True)
    return 20


def run(out_dir: Path) -> int:
    p = out_dir / "d3_realization_means.csv"
    if not p.exists():
        return _fail(out_dir, f"required input missing: {p}")
    df = pd.read_csv(p)
    need = {"arm", "n_realizations", "f1_mean", "f1_sd_across_realizations"}
    if not need <= set(df.columns) or set(df["arm"]) != set(ARMS) or df["arm"].duplicated().any():
        return _fail(out_dir, f"d3_realization_means.csv must hold exactly the arms {ARMS} with {sorted(need)}")
    if (df["n_realizations"] != N_REALIZATIONS).any() or df["f1_mean"].isna().any():
        return _fail(out_dir, f"every arm needs {N_REALIZATIONS} realizations and a non-NaN mean")
    means = df.set_index("arm")["f1_mean"]
    sds = df.set_index("arm")["f1_sd_across_realizations"]
    r1, r3 = float(means["R1"]), float(means["R3"])
    x1, x2, x3, x4 = (float(means[a]) for a in ("X1", "X2", "X3", "X4"))
    fired = classify_a(r1, r3, x1, x2, x3, x4)

    print_gate_header("KC-D3", ",".join(fired) or "none", " ".join(READINGS[l] for l in fired) or "No letter fired.")
    print(f"  R1={r1:.4f} R3={r3:.4f} X1={x1:.4f} X2={x2:.4f} X3={x3:.4f} X4={x4:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"letters": ",".join(fired), **{a: float(means[a]) for a in ARMS}}]).to_csv(
        out_dir / "D3_detail.csv", index=False)
    table = "\n".join(f"| {a} | {means[a] * 100:.2f} | {sds[a] * 100:.2f} |" for a in ARMS)
    outcome = ", ".join(fired) if fired else "none fired"
    (out_dir / "D3_VERDICT.md").write_text(
        f"# KC-D3 verdict\n\n**Outcome(s): {outcome}**\n\n"
        + ("\n".join(READINGS[l] for l in fired) if fired else "No letter fired (A1 to A4 all false).") + "\n\n"
        f"| arm | mean F1 (%) | SD across the 3 realizations (pt) |\n|---|---|---|\n{table}\n\n"
        f"X1 (Gaussian sigma 0.10, the published setting) against R1: {(x1 - r1) * 100:+.2f} pt "
        f"(replication of the Section 4.3.1 null). None of A1 to A4 halts the programme.\n", encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
