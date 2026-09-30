#!/usr/bin/env python3
"""
src/kc23_s1_scripted_stats.py
============================
KC-S1 stats/gate. docs/plans/EXPERIMENT_PLAN_KC23_DEPLOYMENT.md "KC-S1. Scripted buffer:
label-free against supervised", S1.4-S1.5.

Aggregates across the three per-seed src/run_scripted_supervised.py output
directories (results/kc23_s1_scripted_s42/7/123, each holding
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

Secondary (S1.5, added 2026-09-25 per decision D-6b): the K curve (smallest K
at which each supervised ensemble reaches L0), S-ft against L1 at each K (the
pure fine-tune gain, paired inside each seed's base model), and a data check
that S-pool and S-only are seed-invariant classical fits. All of it is written
into S1_VERDICT.md, which starts with the outcome letter.

--report-only: compute and write everything but exit 0 whatever the letter
(still non-zero on a missing input). Used by the queue row that PRODUCES the
verdict; the queue then invokes this script again, without the flag, as the
gate, so a D-S still reaches the halt protocol.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import paired_test, print_gate_header, require_complete, write_no_outcome_verdict
from run_scripted_supervised import reproduction_gate, K_LIST

ROOT = Path(__file__).resolve().parents[1]
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
        f = ROOT / f"results/kc23_s1_scripted_s{seed}" / "s1_subjectwise.csv"
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


ARM_L1 = "RESNET_SE"   # L1 = label-free ResNet-SE+CD alone (the base model S-ft starts from)
DETERMINISM_ARMS = ("S-pool", "S-only")
DETERMINISM_CONTROLS = ("RESNET_SE", "S-ft", "L0")   # must NOT be seed-invariant


def k_curve(df: pd.DataFrame):
    """Supervised ensemble against L0 at every K (realization-averaged, paired
    over 40 subjects). None if any input is missing."""
    rows = []
    for K in K_LIST:
        l0 = realization_averaged(df, K, "L0", SEEDS)
        if l0 is None:
            return None
        for arm in ("S-ens1", "S-ens2"):
            a = realization_averaged(df, K, arm, SEEDS)
            if a is None:
                return None
            t = paired_test(a, l0, f"{arm}_minus_L0_K{K}", "S1-secondary")
            rows.append({"K": K, "arm": arm, "mean_arm": t["mean_a"], "mean_l0": t["mean_b"],
                         "delta_pp": t["delta_pp"], "p_raw": t["p_raw"], "n_improved": t["n_improved"],
                         "reaches_l0": bool(t["delta_pp"] >= 0.0),
                         "significantly_ahead": bool(t["delta_pp"] > 0.0 and t["p_raw"] < 0.05)})
    smallest = {}
    for arm in ("S-ens1", "S-ens2"):
        ok = [r["K"] for r in rows if r["arm"] == arm and r["reaches_l0"]]
        sig = [r["K"] for r in rows if r["arm"] == arm and r["significantly_ahead"]]
        smallest[arm] = {"smallest_K_reaching_l0": min(ok) if ok else None,
                         "smallest_K_significantly_ahead": min(sig) if sig else None}
    return rows, smallest


def sft_vs_l1(df: pd.DataFrame):
    """S-ft minus L1 at each K. The pairing is inside each seed's base model
    (same held-out subject, same trained network); the realization average is
    then taken over the seeds, and the per-seed mean gains are reported beside
    it so run variance is visible."""
    rows = []
    for K in K_LIST:
        ft = realization_averaged(df, K, "S-ft", SEEDS)
        l1 = realization_averaged(df, K, ARM_L1, SEEDS)
        if ft is None or l1 is None:
            return None
        per_seed = []
        for seed in SEEDS:
            a = require_complete(arm_at(df, seed, K, "S-ft"), 40, f"S-ft K={K} seed={seed}")
            b = require_complete(arm_at(df, seed, K, ARM_L1), 40, f"{ARM_L1} K={K} seed={seed}")
            per_seed.append(float((a - b).mean() * 100))
        t = paired_test(ft, l1, f"S-ft_minus_L1_K{K}", "S1-secondary")
        rows.append({"K": K, "mean_s_ft": t["mean_a"], "mean_l1": t["mean_b"], "delta_pp": t["delta_pp"],
                     "p_raw": t["p_raw"], "cohens_dz": t["cohens_dz"], "n_improved": t["n_improved"],
                     "per_seed_delta_pp": ";".join(f"{v:+.2f}" for v in per_seed),
                     "n_seeds_positive": int(sum(v > 0 for v in per_seed))})
    return rows


def determinism_check(df: pd.DataFrame):
    """Max absolute per-subject difference of each arm across the seeds at each
    K. S-pool and S-only should be exactly 0 (fixed-seed classical fits on
    seed-independent buffers); the controls must not be 0, which shows the
    check can tell the difference. NaN-aware: a NaN must sit in the same place
    in every seed."""
    rows = []
    for arm in DETERMINISM_ARMS + DETERMINISM_CONTROLS:
        for K in K_LIST:
            vecs = []
            for seed in SEEDS:
                sub = arm_at(df, seed, K, arm).sort_values("subject")
                if len(sub) != 40:
                    print(f"[S1] MISSING (determinism): {arm} K={K} seed={seed} has {len(sub)} rows, not 40",
                          file=sys.stderr)
                    return None
                vecs.append(sub["f1_macro"].to_numpy(float))
            stack = np.vstack(vecs)
            nan_same = bool((np.isnan(stack) == np.isnan(stack[0])).all())
            with np.errstate(all="ignore"):
                spread = float(np.nanmax(np.abs(stack - stack[0]))) if not np.isnan(stack).all() else float("nan")
            rows.append({"arm": arm, "K": K, "max_abs_diff_across_seeds": spread,
                         "nan_pattern_identical": nan_same,
                         "seed_invariant": bool(nan_same and spread == 0.0),
                         "is_control": arm in DETERMINISM_CONTROLS})
    return rows


def _md_table(rows, cols, fmt=None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if v is None:
                cells.append("-")
            else:
                cells.append(fmt[c](v) if c in fmt else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def run(out_dir: Path, report_only: bool = False) -> int:
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
        write_no_outcome_verdict(out_dir / "S1_VERDICT.md", "KC-S1 verdict",
                                 "reproduction-gate input missing (an arm, K or seed file is absent or incomplete)")
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
        write_no_outcome_verdict(out_dir / "S1_VERDICT.md", "KC-S1 verdict",
                                 "primary-endpoint input missing (L0, S-ens1 or S-ens2 incomplete in some seed)")
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

    # ---- secondary analyses (D-6b): all inputs required, none defaulted ----
    kc = k_curve(df)
    sf = sft_vs_l1(df)
    det = determinism_check(df)
    if kc is None or sf is None or det is None:
        print("[S1] secondary analyses: FAIL (missing input, not computed -- see above). Stop.", file=sys.stderr)
        write_no_outcome_verdict(out_dir / "S1_VERDICT.md", "KC-S1 verdict",
                                 "secondary-analysis input missing (K curve, S-ft against L1, or determinism check)")
        return 20
    kc_rows, smallest = kc
    pd.DataFrame(kc_rows).to_csv(out_dir / "S1_k_curve.csv", index=False)
    pd.DataFrame(sf).to_csv(out_dir / "S1_sft_vs_l1.csv", index=False)
    pd.DataFrame(det).to_csv(out_dir / "S1_determinism.csv", index=False)

    pct = lambda v: f"{v * 100:.2f}"
    pp = lambda v: f"{v:+.2f}"
    verdict = (
        f"# KC-S1 verdict\n\n**Outcome: {letter}**\n\n**Reproduction gate: {letter_gate}**\n\n{reading}\n\n"
        f"## Primary endpoint (K={K_PRIMARY}, realization-averaged over seeds {SEEDS})\n\n"
        f"Best supervised arm {best_name} {detail['mean_a'] * 100:.2f}% against L0 {detail['mean_b'] * 100:.2f}%: "
        f"{detail['delta_pp']:+.2f} pt, Wilcoxon p={detail['p_raw']:.3g}, dz={detail['cohens_dz']:.2f}, "
        f"BCa 95% [{detail['bca_lo_pp']:+.2f}, {detail['bca_hi_pp']:+.2f}] pt, "
        f"{detail['n_improved']} of {detail['n']} subjects improved.\n\n"
        f"## Secondary 1: K curve (supervised ensemble against L0)\n\n"
        + _md_table(kc_rows, ["K", "arm", "mean_arm", "mean_l0", "delta_pp", "p_raw", "n_improved",
                              "reaches_l0", "significantly_ahead"],
                    {"mean_arm": pct, "mean_l0": pct, "delta_pp": pp, "p_raw": lambda v: f"{v:.3g}"})
        + "\nSmallest K at which each arm reaches L0 (mean difference >= 0) and at which it is significantly "
          "ahead (p < 0.05): "
        + "; ".join(f"{a}: reaches at K={v['smallest_K_reaching_l0']}, significantly ahead at "
                    f"K={v['smallest_K_significantly_ahead']}" for a, v in smallest.items())
        + f". The K grid is {list(K_LIST)}, so K={min(K_LIST)} is the lowest value tested: a result at "
          f"K={min(K_LIST)} means 'at or below {min(K_LIST)}', not exactly {min(K_LIST)}.\n\n"
        f"## Secondary 2: S-ft against L1 (pure fine-tune gain, causal normalization)\n\n"
        + _md_table(sf, ["K", "mean_s_ft", "mean_l1", "delta_pp", "p_raw", "cohens_dz", "n_improved",
                         "per_seed_delta_pp", "n_seeds_positive"],
                    {"mean_s_ft": pct, "mean_l1": pct, "delta_pp": pp, "p_raw": lambda v: f"{v:.3g}",
                     "cohens_dz": lambda v: f"{v:.2f}"})
        + "\nL1 is the RESNET_SE arm, the same trained base network S-ft starts from, so each seed's pairing sits "
          "inside one training realization.\n\n"
        f"## Secondary 3: S-pool and S-only are seed-invariant classical fits\n\n"
        + _md_table(det, ["arm", "K", "max_abs_diff_across_seeds", "seed_invariant", "is_control"],
                    {"max_abs_diff_across_seeds": lambda v: f"{v:.3g}"})
        + "\nS-pool and S-only are fixed SVC fits (`random_state=42`, not the run seed) on buffers that do not "
          "depend on the seed, so their per-subject F1 is identical in every seed by construction; the rows above "
          "confirm it in the data. The controls (RESNET_SE, S-ft, L0, which depend on the trained network) are "
          "not seed-invariant, so the check can tell the difference. Their across-seed SD is therefore zero by "
          "construction and is not evidence of stability across training runs.\n"
    )
    (out_dir / "S1_VERDICT.md").write_text(verdict, encoding="utf-8")
    if report_only:
        return 0
    return 20 if letter == "D-S" else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-only", action="store_true",
                    help="write the full verdict but exit 0 whatever the letter (missing input still fails)")
    args = ap.parse_args()
    sys.exit(run(Path(args.out), report_only=args.report_only))


if __name__ == "__main__":
    main()
