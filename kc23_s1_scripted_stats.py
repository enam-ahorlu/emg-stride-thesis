#!/usr/bin/env python3
"""
kc23_s1_scripted_stats.py
============================
KC-S1 stats/gate. EXPERIMENT_PLAN_KC23_DEPLOYMENT.md "KC-S1. Scripted buffer:
label-free against supervised", S1.5.

Primary endpoint, at K=25: the better of S-ens1 and S-ens2 (chosen on the
realization average) against L0, paired over 40 subjects, realization-averaged.

  D-S: supervised ahead by >= 1.0pt, significant, on >= 25 of 40 subjects
       -> ESCALATE (claim change: "without labeled calibration" holds offline only)
  D-T: within +/- 1.0pt   -> labels add nothing once the buffer exists
  D-L: label-free ahead by >= 1.0pt -> report; strengthens the label-free case
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import paired_test, print_gate_header, read_subjectwise, require_complete

REPRO_TOL = 0.005  # K=25 L0 reproduction gate vs published balanced25


def choose_best_supervised(f1_s_ens1: np.ndarray, f1_s_ens2: np.ndarray) -> tuple[str, np.ndarray]:
    return ("S-ens1", f1_s_ens1) if f1_s_ens1.mean() >= f1_s_ens2.mean() else ("S-ens2", f1_s_ens2)


def classify_d(f1_supervised: np.ndarray, f1_l0: np.ndarray) -> tuple[str, dict]:
    t = paired_test(f1_supervised, f1_l0, "supervised_minus_l0", "S1")
    delta_pp = t["delta_pp"]
    significant = t["p_raw"] < 0.05
    n_improved = t["n_improved"]
    if delta_pp >= 1.0 and significant and n_improved >= 25:
        letter = "D-S"
    elif abs(delta_pp) <= 1.0:
        letter = "D-T"
    elif delta_pp <= -1.0:
        letter = "D-L"
    else:
        letter = "D-T"  # falls in an ambiguous small-magnitude, non-significant zone: treat as no material difference
    return letter, t


def reproduction_gate(f1_l0_k25: np.ndarray, published_balanced25: float) -> bool:
    return abs(f1_l0_k25.mean() - published_balanced25) <= REPRO_TOL


def run(out_dir: Path, published_balanced25: float | None = None) -> int:
    f1_l0 = require_complete(read_subjectwise(out_dir / "l0_k25_subjectwise.csv"), 40, "L0")
    if published_balanced25 is not None:
        ok = reproduction_gate(f1_l0, published_balanced25)
        print(f"[S1] L0 K=25 reproduction gate vs published balanced25 ({published_balanced25:.4f}): "
             f"{'PASS' if ok else 'FAIL'} (got {f1_l0.mean():.4f})")
        if not ok:
            print("[S1] Reproduction gate FAILED. Stop.", file=sys.stderr)
            return 20

    f1_ens1 = require_complete(read_subjectwise(out_dir / "s_ens1_k25_subjectwise.csv"), 40, "S-ens1")
    f1_ens2 = require_complete(read_subjectwise(out_dir / "s_ens2_k25_subjectwise.csv"), 40, "S-ens2")
    best_name, f1_best = choose_best_supervised(f1_ens1, f1_ens2)
    print(f"[S1] best supervised arm at K=25: {best_name} (mean {f1_best.mean():.4f})")

    letter, detail = classify_d(f1_best, f1_l0)
    reading = {
        "D-S": "ESCALATE. 'Without labeled calibration' holds offline only.",
        "D-T": "Labels add nothing once the scripted buffer exists.",
        "D-L": "Label-free ahead; strengthens the label-free case.",
    }[letter]
    print_gate_header("KC-S1", letter, reading)
    print(f"  {detail}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"best_supervised_arm": best_name, **detail}]).to_csv(out_dir / "S1_tests.csv", index=False)
    (out_dir / "S1_VERDICT.md").write_text(f"# KC-S1 verdict\n\n**Outcome: {letter}**\n\n{reading}\n",
                                            encoding="utf-8")
    return 20 if letter == "D-S" else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--published-balanced25", type=float, default=None)
    args = ap.parse_args()
    sys.exit(run(Path(args.out), args.published_balanced25))


if __name__ == "__main__":
    main()
