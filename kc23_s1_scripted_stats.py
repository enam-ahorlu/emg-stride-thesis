#!/usr/bin/env python3
"""
kc23_s1_scripted_stats.py
============================
KC-S1 stats/gate. EXPERIMENT_PLAN_KC23_DEPLOYMENT.md "KC-S1. Scripted buffer:
label-free against supervised", S1.4-S1.5.

Aggregates across the three per-seed run_scripted_supervised.py output
directories (results_kc23_s1_scripted_s42/7/123, each holding
s1_subjectwise.csv), rather than reading a single --out directory -- the
queue always invokes a gate_script as `python <gate> --out <out_dir>`, so the
directory discovery is done here, glob'd relative to this script's own
location, not passed through --out. --out is only where THIS gate writes its
own S1_VERDICT.md / S1_tests.csv.

Reproduction gate (S1.4), fixed 2026-09-24 after finding the previous version
compared a single ensemble figure against itself when the ensemble file was
missing (always a trivial pass): three checks, all against seed 42's K=25
results (SVM/SVM_PROBA are seed-invariant by construction; RESNET_SE and the
soft vote are not, so seed 42 -- the anchor realization used throughout this
programme -- is the one checked):
  - SVM decision route == 0.7472 (published balanced25), exact to 4 dp
  - SVM_PROBA == 0.7286, exact to 4 dp
  - soft (L0) within +/-1.5pt of 0.8152 (or the KC-D1 measured band, once
    available)
A missing input is a FAIL, never a fallback to the published value.

Primary endpoint (S1.5), at K=25: the better of S-ens1 and S-ens2 (chosen on
the realization average) against L0, paired over 40 subjects,
realization-averaged across the 3 base-training seeds.

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

from kc23_stats_common import paired_test, print_gate_header, require_complete
from run_scripted_supervised import reproduction_gate, K_LIST

ROOT = Path(__file__).resolve().parent
SEED_ANCHOR = 42
K_PRIMARY = 25
SEEDS = [42, 7, 123]


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


def load_all_seeds() -> pd.DataFrame:
    dfs = []
    for seed in SEEDS:
        f = ROOT / f"results_kc23_s1_scripted_s{seed}" / "s1_subjectwise.csv"
        if f.exists():
            dfs.append(pd.read_csv(f))
    if not dfs:
        return pd.DataFrame(columns=["subject", "seed", "K", "arm", "f1_macro", "n_excl"])
    return pd.concat(dfs, ignore_index=True)


def arm_at(df: pd.DataFrame, seed: int, K: int, arm: str) -> pd.DataFrame:
    return df[(df["seed"] == seed) & (df["K"] == K) & (df["arm"] == arm)]


def realization_averaged(df: pd.DataFrame, K: int, arm: str, seeds: list[int]) -> np.ndarray | None:
    """Per-subject mean over the given seeds' realizations at (K, arm). None
    (never a fallback) if any seed/subject is missing for this arm."""
    per_seed = []
    for seed in seeds:
        sub = arm_at(df, seed, K, arm)
        try:
            f1 = require_complete(sub, 40, f"{arm} K={K} seed={seed}")
        except ValueError as e:
            print(f"[S1] MISSING: {e}", file=sys.stderr)
            return None
        per_seed.append(f1)
    return np.mean(np.vstack(per_seed), axis=0)


def run(out_dir: Path) -> int:
    df = load_all_seeds()

    # ---- S1.4 reproduction gate: seed 42, K=25 ----
    def scalar_mean(K, arm, seed=SEED_ANCHOR):
        sub = arm_at(df, seed, K, arm)
        try:
            f1 = require_complete(sub, 40, f"{arm} K={K} seed={seed}")
        except ValueError as e:
            print(f"[S1] MISSING (gate input): {e}", file=sys.stderr)
            return None
        return float(f1.mean())

    svm_f1 = scalar_mean(K_PRIMARY, "SVM_decision")
    svm_proba_f1 = scalar_mean(K_PRIMARY, "SVM_PROBA")
    soft_f1 = scalar_mean(K_PRIMARY, "L0")

    if svm_f1 is None or svm_proba_f1 is None or soft_f1 is None:
        print("[S1] reproduction gate: FAIL (missing input, not computed -- see above). Stop.",
             file=sys.stderr)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "S1_VERDICT.md").write_text(
            "# KC-S1 verdict\n\n**Outcome: reproduction gate FAIL (missing input)**\n", encoding="utf-8")
        return 20

    letter_gate, detail_gate = reproduction_gate(svm_f1, svm_proba_f1, soft_f1)
    print_gate_header("KC-S1 reproduction gate", letter_gate,
                      "Continue." if letter_gate == "PASS" else "ESCALATE: code drift since publication.")
    print(f"  {detail_gate}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([detail_gate]).to_csv(out_dir / "S1_reproduction_gate.csv", index=False)
    if letter_gate == "FAIL":
        (out_dir / "S1_VERDICT.md").write_text(
            f"# KC-S1 verdict\n\n**Outcome: reproduction gate FAIL**\n\n{detail_gate}\n", encoding="utf-8")
        return 20

    # ---- S1.5 primary endpoint: K=25, realization-averaged over the 3 seeds ----
    f1_l0 = realization_averaged(df, K_PRIMARY, "L0", SEEDS)
    f1_ens1 = realization_averaged(df, K_PRIMARY, "S-ens1", SEEDS)
    f1_ens2 = realization_averaged(df, K_PRIMARY, "S-ens2", SEEDS)
    if f1_l0 is None or f1_ens1 is None or f1_ens2 is None:
        print("[S1] endpoint: FAIL (missing input, not computed -- see above). Stop.", file=sys.stderr)
        (out_dir / "S1_VERDICT.md").write_text(
            "# KC-S1 verdict\n\n**Outcome: endpoint FAIL (missing input)**\n", encoding="utf-8")
        return 20

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

    pd.DataFrame([{"best_supervised_arm": best_name, **detail}]).to_csv(out_dir / "S1_tests.csv", index=False)
    (out_dir / "S1_VERDICT.md").write_text(
        f"# KC-S1 verdict\n\n**Reproduction gate: {letter_gate}**\n\n**Outcome: {letter}**\n\n{reading}\n",
        encoding="utf-8")
    return 20 if letter == "D-S" else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
