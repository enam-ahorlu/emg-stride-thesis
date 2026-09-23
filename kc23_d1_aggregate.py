#!/usr/bin/env python3
"""
kc23_d1_aggregate.py
======================
Turns completed KC-D1 per-realization run folders (results_kc23_d1_r<N>_s<seed>/)
into the input CSVs kc23_d1_replicate_stats.py reads from a shared --out
directory: d1_reproduction_inputs.csv, d1_contrasts.csv, d1_headline_inputs.csv.
EXPERIMENT_PLAN_KC23_DEEP.md D1.4-D1.6.

Written 2026-09-24 after discovering the pre-registered gate had nothing to
read: kc23_d1_replicate_stats.py was wired (kc23_build_job_csvs.py) to fire
on a single run's own out_dir, which never held these four csvs -- this
script is the missing piece that actually produces them, from the real,
confirmed run output schemas.

Scope (read before extending): only arms whose on-disk output schema has been
directly confirmed against a real completed run, or against the run script's
own source, are wired here -- R1-R7, R10 (pre and post), R12-R17, all reading
run_cnn_arch_loso.py's cnn_arch_subjectwise.csv (subject, arch, f1_macro,
bal_acc) or run_adabn_cnn_loso.py's adabn_subjectwise.csv (subject, arch,
f1_pre_adabn, f1_macro, bal_acc, delta_pp). R8/R9 (train_cnn_loso.py) and R11
(run_deep_coral_align_loso.py) had not completed a single run when this was
written and their subjectwise output schema was not directly confirmed, so
they are deliberately NOT wired rather than guessed at. Consequences:
  - d1_contrasts.csv does NOT include C10 (R11-R10 post), C11 (R2-R11), or
    C15 (R9-R8).
  - The per-seed ensemble (C13, and the descriptive C13b/decision-D-6a row)
    is NOT computed here -- it needs ensemble_v2_combine.py's proba-directory
    contract, which was not confirmed either.
  - C16 (the three-way plateau/fall comparison) and C17 (the occlusion
    reduction factor, which also needs KC-D2's occlusion instrumentation) are
    structurally different from a plain two-run diff and are also not
    computed here.
  - d1_headline_inputs.csv only ever gets R2 and R12 rows; "ensemble" and
    "global" (D1.6) need the same ensemble/occlusion wiring above and are
    skipped for the same reason.
This mirrors the MixedLM scope note already in kc23_d1_replicate_stats.py:
flagged in this docstring and in the run-time output, not silently dropped.
Extend the ARM_SCHEMA / SIMPLE_CONTRASTS / INTERACTION_CONTRASTS tables below
once R8/R9/R11 have produced a real subjectwise csv to confirm column names
against, and once the ensemble/occlusion wiring is designed.

Idempotent and safe to re-run: for each of the three output csvs, this only
writes rows for a reproduction check / contrast / headline arm once ALL of
its required realizations (4 seeds for a Tier A arm, 3 for Tier B) are
actually on disk and complete (40 distinct subjects, no duplicates) -- never
a partial subset -- so a re-run after more seeds land can only ever add
contrasts, never emit a premature verdict on an incomplete one. Prints which
contrasts/headline arms are and are not yet ready every time it runs.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kc23_stats_common import require_complete

N_SUBJECTS = 40
TIER_A_SEEDS = [42, 7, 123, 1001]
TIER_B_SEEDS = [42, 7, 123]
TIER_B_ARMS = {"R13", "R14", "R15", "R16", "R17"}
PUBLISHED_SVM = 0.777  # C12: R2 - SVM (fixed, 77.7), EXPERIMENT_PLAN_KC23_DEEP.md D1.5

# arm -> subjectwise csv filename within results_kc23_d1_<arm.lower()>_s<seed>/
ARM_SCHEMA = {arm: "cnn_arch_subjectwise.csv" for arm in
             ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R12", "R13", "R14", "R15", "R16", "R17"]}
ARM_SCHEMA["R10"] = "adabn_subjectwise.csv"  # post-adaptation column is f1_macro; pre is f1_pre_adabn

SIMPLE_CONTRASTS = [  # (id, a, b): diff = f1(a) - f1(b), same seed, matched subject
    ("C1", "R2", "R1"), ("C2", "R3", "R2"), ("C3", "R14", "R13"), ("C4", "R16", "R15"),
    ("C5", "R1", "R4"), ("C8", "R4", "R6"), ("C9", "R2", "R10"), ("C14", "R12", "R2"),
]
INTERACTION_CONTRASTS = [  # (id, a, b, c, d): diff = (f1(a)-f1(b)) - (f1(c)-f1(d))
    ("C6", "R2", "R1", "R5", "R4"),
    ("C7", "R5", "R4", "R7", "R6"),
]
HEADLINE_ARMS = ["R2", "R12"]


def arm_seeds(arm: str) -> list[int]:
    return TIER_B_SEEDS if arm in TIER_B_ARMS else TIER_A_SEEDS


def load_arm_post(root: Path, arm: str, seed: int) -> pd.Series | None:
    """Post-adaptation / plain f1_macro per subject for one arm+seed realization,
    or None if that run isn't complete yet (missing, or not all 40 subjects)."""
    if arm not in ARM_SCHEMA:
        return None
    p = root / f"results_kc23_d1_{arm.lower()}_s{seed}" / ARM_SCHEMA[arm]
    if not p.exists():
        return None
    df = pd.read_csv(p)
    try:
        f1 = require_complete(df, N_SUBJECTS, label=f"{arm} s{seed}")
    except ValueError as e:
        print(f"[d1-aggregate] {e} -- not using this realization yet")
        return None
    return pd.Series(f1, index=sorted(df["subject"].unique()))


def load_r10_pre(root: Path, seed: int) -> pd.Series | None:
    p = root / f"results_kc23_d1_r10_s{seed}" / "adabn_subjectwise.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df["subject"].duplicated().any() or len(df) != N_SUBJECTS:
        print(f"[d1-aggregate] R10 s{seed} pre-AdaBN: incomplete/duplicated -- not using yet")
        return None
    return df.sort_values("subject").set_index("subject")["f1_pre_adabn"]


def build_reproduction_inputs(root: Path) -> pd.DataFrame | None:
    r1 = load_arm_post(root, "R1", 42)
    r2 = load_arm_post(root, "R2", 42)
    r10_pre = load_r10_pre(root, 42)
    if r1 is None or r2 is None or r10_pre is None:
        print("[d1-aggregate] reproduction inputs (R1/R2/R10-pre, seed 42): not all complete yet")
        return None
    return pd.DataFrame([{"arm": "R1", "f1_mean": float(r1.mean())},
                        {"arm": "R2", "f1_mean": float(r2.mean())},
                        {"arm": "R10_pre", "f1_mean": float(r10_pre.mean())}])


def build_contrast_rows(root: Path) -> list[dict]:
    rows = []
    ready, not_ready = [], []

    for cid, a, b in SIMPLE_CONTRASTS:
        seeds = arm_seeds(a)
        a_by_seed = {s: load_arm_post(root, a, s) for s in seeds}
        b_by_seed = {s: load_arm_post(root, b, s) for s in seeds}
        if any(v is None for v in a_by_seed.values()) or any(v is None for v in b_by_seed.values()):
            not_ready.append(cid)
            continue
        tier = "B" if a in TIER_B_ARMS or b in TIER_B_ARMS else "A"
        for s in seeds:
            diff = a_by_seed[s] - b_by_seed[s]
            for subj, d in diff.items():
                rows.append({"contrast": cid, "tier": tier, "realization": s, "subject": int(subj), "diff": float(d)})
        ready.append(cid)

    for cid, a, b, c, d in INTERACTION_CONTRASTS:
        seeds = TIER_A_SEEDS
        parts = {arm: {s: load_arm_post(root, arm, s) for s in seeds} for arm in (a, b, c, d)}
        if any(v is None for m in parts.values() for v in m.values()):
            not_ready.append(cid)
            continue
        for s in seeds:
            diff = (parts[a][s] - parts[b][s]) - (parts[c][s] - parts[d][s])
            for subj, dv in diff.items():
                rows.append({"contrast": cid, "tier": "A", "realization": s, "subject": int(subj), "diff": float(dv)})
        ready.append(cid)

    # C12: R2 - SVM (fixed constant), Tier A
    r2_by_seed = {s: load_arm_post(root, "R2", s) for s in TIER_A_SEEDS}
    if all(v is not None for v in r2_by_seed.values()):
        for s in TIER_A_SEEDS:
            for subj, f1 in r2_by_seed[s].items():
                rows.append({"contrast": "C12", "tier": "A", "realization": s, "subject": int(subj),
                            "diff": float(f1 - PUBLISHED_SVM)})
        ready.append("C12")
    else:
        not_ready.append("C12")

    print(f"[d1-aggregate] contrasts ready: {sorted(set(ready))}")
    print(f"[d1-aggregate] contrasts not yet complete: {sorted(set(not_ready))}")
    print("[d1-aggregate] out of scope for this extractor (see module docstring): "
          "C10, C11, C13, C13b, C15, C16, C17")
    return rows


def build_headline_rows(root: Path) -> list[dict]:
    rows = []
    for arm in HEADLINE_ARMS:
        means = []
        complete = True
        for s in TIER_A_SEEDS:
            f1 = load_arm_post(root, arm, s)
            if f1 is None:
                complete = False
                break
            means.append(float(f1.mean()))
        if not complete:
            print(f"[d1-aggregate] headline arm {arm}: not all Tier A seeds complete yet")
            continue
        rows.append({"arm": arm, "realization_mean": float(np.mean(means)),
                    "realization_sd": float(np.std(means, ddof=1))})
    return rows


def run(out_dir: Path, root: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    repro = build_reproduction_inputs(root)
    if repro is not None:
        repro.to_csv(out_dir / "d1_reproduction_inputs.csv", index=False)

    contrast_rows = build_contrast_rows(root)
    if contrast_rows:
        pd.DataFrame(contrast_rows).to_csv(out_dir / "d1_contrasts.csv", index=False)

    headline_rows = build_headline_rows(root)
    if headline_rows:
        pd.DataFrame(headline_rows).to_csv(out_dir / "d1_headline_inputs.csv", index=False)

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.root)))


if __name__ == "__main__":
    main()
