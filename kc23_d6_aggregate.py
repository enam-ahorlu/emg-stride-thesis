#!/usr/bin/env python3
"""
kc23_d6_aggregate.py
======================
Turns completed KC-D6 Stage-1 per-knob run folders into the two input files
kc23_d6_stats.py reads: d6_sanity.csv (f1 per subject, from the ADV-marginal
lambda_max=0 arm) and d6_manipulation_<family>.csv (realization, knob,
domain_probe, subject_probe -- realization = LOSO subject, per
EXPERIMENT_PLAN_KC23_DEEP.md 6.2/6.5: "40 folds, seed 42 re-run").

Written 2026-09-24 for the same reason kc23_d1_aggregate.py exists: the D6
sanity/manipulation gates were wired to fire on a single knob's own private
out_dir, which never held these files.

Real output schemas (confirmed against results_kc23_d6_smoke_marginal/
_classcond/_cdan and results_kc23_d05_smoke_l2, the fixtures used to test
this):
  adv_subjectwise.csv        (ADV family, run_adv_align_loso.py):
    subject, arch, adv_lambda, adv_mode, oracle, diverged, f1_macro, bal_acc
  deep_coral_subjectwise.csv (SFC family, run_deep_coral_align_loso.py):
    subject, target_pass, ..., coral_lambda, ...
  alignment_subjectwise.csv  (both families, the D2c alignment_metrics +
    D0.3 probes instrumentation):
    subject, ..., domain_probe_bacc, class_probe_tgt_bacc, class_probe_src_bacc
  "domain probe" = domain_probe_bacc; "unseen-subject probe" =
  class_probe_tgt_bacc (measured on the held-out subject's own validation
  windows, per 6.2's "neither the classifier nor the adversary ever sees").

Idempotent and safe to re-run: a family's manipulation input is written only
once EVERY knob in its grid has produced a real alignment_subjectwise.csv;
never a partial subset. Sanity is written once the lambda_max=0 run exists.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

# Must match kc23_build_job_csvs.py's Stage-1 generation and
# kc23_d6_stage2_job_gen.py's FAMILY_SPECS exactly.
FAMILY_KNOBS = {
    "adv_marginal": [0, 0.03, 0.1, 0.3, 1, 3, 10],
    "sfc": [0.1, 1, 10, 100, 1000],
    "advps": [0.1, 1, 10],
}
FAMILY_OUT_DIR = {
    "adv_marginal": lambda knob: f"results_kc23_d6_adv_marginal_l{knob}_s42",
    "sfc": lambda knob: f"results_kc23_d6_sfc_w{knob}_s42",
    "advps": lambda knob: f"results_kc23_d6_advps_l{knob}_s42",
}
KNOWN_FAMILIES = list(FAMILY_KNOBS)


def load_alignment(root: Path, family: str, knob) -> pd.DataFrame | None:
    d = root / FAMILY_OUT_DIR[family](knob)
    f = d / "alignment_subjectwise.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    needed = {"subject", "domain_probe_bacc", "class_probe_tgt_bacc"}
    if not needed.issubset(df.columns):
        print(f"[d6-aggregate] {f}: missing columns {needed - set(df.columns)}", file=sys.stderr)
        return None
    return df[["subject", "domain_probe_bacc", "class_probe_tgt_bacc"]]


def build_sanity(root: Path) -> pd.DataFrame | None:
    d = root / FAMILY_OUT_DIR["adv_marginal"](0)
    f = d / "adv_subjectwise.csv"
    if not f.exists():
        print(f"[d6-aggregate] sanity: {f} not found -- lambda_max=0 arm not done yet")
        return None
    df = pd.read_csv(f)
    lam0 = df[df["adv_lambda"] == 0]
    if lam0.empty:
        lam0 = df  # single-knob smoke fixtures may not carry adv_lambda==0 explicitly
    return lam0[["subject", "f1_macro"]].rename(columns={"f1_macro": "f1"})


def build_manipulation(root: Path, family: str) -> pd.DataFrame | None:
    knobs = FAMILY_KNOBS[family]
    rows = []
    for knob in knobs:
        df = load_alignment(root, family, knob)
        if df is None:
            print(f"[d6-aggregate] manipulation/{family}: knob {knob} not complete yet "
                 f"({FAMILY_OUT_DIR[family](knob)}/alignment_subjectwise.csv missing) -- "
                 f"not writing this family's manipulation input yet")
            return None
        for _, r in df.iterrows():
            rows.append({"realization": int(r["subject"]), "knob": knob,
                        "domain_probe": float(r["domain_probe_bacc"]),
                        "subject_probe": float(r["class_probe_tgt_bacc"])})
    return pd.DataFrame(rows)


def run(out_dir: Path, root: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    sanity = build_sanity(root)
    if sanity is not None:
        sanity.to_csv(out_dir / "d6_sanity.csv", index=False)
        print(f"[d6-aggregate] wrote d6_sanity.csv ({len(sanity)} subjects)")

    for family in KNOWN_FAMILIES:
        manip = build_manipulation(root, family)
        if manip is not None:
            manip.to_csv(out_dir / f"d6_manipulation_{family}.csv", index=False)
            print(f"[d6-aggregate] wrote d6_manipulation_{family}.csv "
                 f"({manip['realization'].nunique()} subjects x {manip['knob'].nunique()} knobs)")

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.root)))


if __name__ == "__main__":
    main()
