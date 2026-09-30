#!/usr/bin/env python3
"""
src/kc23_d2_reliance_stats.py
============================
KC-D2 stats/gate (analysis only, on KC-D1's instrumented runs).
docs/plans/EXPERIMENT_PLAN_KC23_DEEP.md "KC-D2. Reliance against trained-in robustness",
D2.2 and D2.3. Rewritten 26 September 2026 (docs/kc23/KC23_PREREG_CONFORMANCE.md): the
classification (classify_o) already matched the plan and is unchanged, but the
analysis layer did not. It computed each reduction factor once, from all
realizations pooled; the plan says "each is computed per realization, with the
across-realization mean and SD, plus the paired Wilcoxon on realization-averaged
per-subject sums". It also never reported the attenuation factor, and it read
inputs no job produced (src/kc23_d2_aggregate.py now does).

Primary quantities (plan D2.2):
  reduction factor of permutation reliance   R2 against R1
  reduction factor of zeroing occlusion      R3 against R1   (the key cross-check)
  reduction factor of attenuation at alpha 0.5   R2 against R1
each computed PER REALIZATION as mean over subjects of the reference's summed drop
divided by the mean of the arm's (kc23_run_loader.reduction_factor; the summed
drop is KC-D1's C17 definition, negative drops kept), then averaged over the 4
realizations with its SD, plus a paired Wilcoxon on the realization-averaged
per-subject sums (n = 40). The letters use the across-realization MEAN factor.
Also reported, descriptively, the same three measures for R2 and R5 against their
references (C17: R2 against R1 and R5 against R4).

  O-R: zeroing occlusion falls >= 3-fold under gain jitter AND permutation reliance
       falls >= 2-fold under channel dropout
  O-T: zeroing occlusion falls < 1.5-fold under gain jitter AND permutation
       reliance falls < 1.5-fold under channel dropout
  O-M: anything else
Residual caveat, recorded in the verdict as the plan requires: permuting one channel
breaks cross-channel coherence, so it is not perfectly in-distribution either; it is
the closest standard probe.

Input (required): d2_persubject_sums.csv. A missing or incomplete input exits 20 with a
verdict that carries no outcome line. D2 has no ESCALATE letter: a computed result exits 0.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from kc23_stats_common import print_gate_header, write_no_outcome_verdict

ARMS = ["R1", "R2", "R3", "R4", "R5"]
REALIZATIONS = [42, 7, 123, 1001, 2026]
N_SUBJECTS = 40
QUANTITIES = ["occlusion_sum", "attenuation_sum", "permutation_sum"]
# (label, arm, reference, quantity, primary?)
COMPARISONS = [
    ("permutation reliance, R2 against R1", "R2", "R1", "permutation_sum", True),
    ("zeroing occlusion, R3 against R1 (the cross-check)", "R3", "R1", "occlusion_sum", True),
    ("attenuation at alpha 0.5, R2 against R1", "R2", "R1", "attenuation_sum", True),
    ("zeroing occlusion, R2 against R1 (C17)", "R2", "R1", "occlusion_sum", False),
    ("zeroing occlusion, R5 against R4 (C17)", "R5", "R4", "occlusion_sum", False),
    ("permutation reliance, R5 against R4", "R5", "R4", "permutation_sum", False),
    ("attenuation at alpha 0.5, R5 against R4", "R5", "R4", "attenuation_sum", False),
]


# Kept for src/kc23_d5_aggregate.py, which imports both (the definition D5 shares with D2).
def reduction_factor(cost_r1: np.ndarray, cost_r_aug: np.ndarray) -> float:
    denom = cost_r_aug.mean()
    return float(cost_r1.mean() / denom) if denom > 0 else float("inf")


def _mean_per_subject_cost(df: pd.DataFrame, value_col: str) -> pd.Series:
    return df.groupby("subject")[value_col].mean()


def classify_o(occlusion_factor_gainjitter: float, permutation_factor_chandrop: float) -> str:
    if occlusion_factor_gainjitter >= 3.0 and permutation_factor_chandrop >= 2.0:
        return "O-R"
    if occlusion_factor_gainjitter < 1.5 and permutation_factor_chandrop < 1.5:
        return "O-T"
    return "O-M"


class InputError(Exception):
    pass


def load_sums(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise InputError(f"required input missing: {path}")
    df = pd.read_csv(path)
    need = {"arm", "realization", "subject", *QUANTITIES}
    if not need <= set(df.columns):
        raise InputError(f"d2_persubject_sums.csv lacks {sorted(need - set(df.columns))}")
    if set(df["arm"]) != set(ARMS) or set(df["realization"]) != set(REALIZATIONS):
        raise InputError(f"arms {sorted(set(df['arm']))} / realizations {sorted(set(df['realization']))} differ from the "
                         f"required {ARMS} / {REALIZATIONS}")
    counts = df.groupby(["arm", "realization"]).size()
    if (counts != N_SUBJECTS).any() or df.duplicated(["arm", "realization", "subject"]).any():
        raise InputError(f"every (arm, realization) needs {N_SUBJECTS} unique subjects")
    if df[QUANTITIES].isna().any().any():
        raise InputError("NaN in a summed drop")
    return df


def compare(df: pd.DataFrame, arm: str, ref: str, q: str) -> dict:
    per_real = []
    for r in REALIZATIONS:
        a = df[(df["arm"] == arm) & (df["realization"] == r)].set_index("subject")[q]
        b = df[(df["arm"] == ref) & (df["realization"] == r)].set_index("subject")[q]
        per_real.append(reduction_factor(b.to_numpy(), a.to_numpy()))
    avg = df.groupby(["arm", "subject"])[q].mean().unstack("arm")
    diff = (avg[ref] - avg[arm]).to_numpy()
    p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(avg[ref], avg[arm], zero_method="wilcox",
                                                              alternative="two-sided").pvalue)
    return {"factor_mean": float(np.mean(per_real)), "factor_sd": float(np.std(per_real, ddof=1)),
            "factor_per_realization": ";".join(f"{v:.3f}" for v in per_real),
            "wilcoxon_p": p, "mean_ref_sum": float(avg[ref].mean()), "mean_arm_sum": float(avg[arm].mean()),
            "n_subjects_ref_higher": int((diff > 0).sum())}


def _fail(out_dir: Path, reason: str) -> int:
    print(f"[D2] FAIL (no outcome computed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(out_dir / "D2_VERDICT.md", "KC-D2 verdict", reason)
    (out_dir / "D2_detail.csv").unlink(missing_ok=True)
    return 20


def run(out_dir: Path) -> int:
    try:
        df = load_sums(out_dir / "d2_persubject_sums.csv")
    except InputError as e:
        return _fail(out_dir, str(e))
    rows = []
    for label, arm, ref, q, primary in COMPARISONS:
        rows.append({"comparison": label, "primary": primary, "arm": arm, "reference": ref, "quantity": q,
                     **compare(df, arm, ref, q)})
    res = pd.DataFrame(rows)
    occ = res.loc[res["comparison"].str.startswith("zeroing occlusion, R3"), "factor_mean"].iloc[0]
    perm = res.loc[res["comparison"].str.startswith("permutation reliance, R2"), "factor_mean"].iloc[0]
    letter = classify_o(float(occ), float(perm))
    reading = {
        "O-R": "Reduced reliance is supported by two measures that do not share channel dropout's training perturbation.",
        "O-T": "Occlusion measured trained-in robustness; 'draws on what was there all along' goes.",
        "O-M": "Report per measure; permutation reliance is the reliance measure, occlusion the robustness measure.",
    }[letter]
    print_gate_header("KC-D2", letter, reading)
    print(f"  occlusion reduction factor under gain jitter (R3 vs R1): {occ:.2f}x")
    print(f"  permutation reduction factor under channel dropout (R2 vs R1): {perm:.2f}x")

    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "D2_detail.csv", index=False)
    lines = "\n".join(f"| {r.comparison} | {r.factor_mean:.2f} | {r.factor_sd:.2f} | {r.factor_per_realization} | "
                      f"{r.wilcoxon_p:.3g} |" for r in res.itertuples())
    (out_dir / "D2_VERDICT.md").write_text(
        f"# KC-D2 verdict\n\n**Outcome: {letter}**\n\n{reading}\n\n"
        f"Reduction factors (per realization, then mean and SD over {len(REALIZATIONS)} realizations; paired Wilcoxon on "
        f"realization-averaged per-subject sums, n = {N_SUBJECTS}):\n\n"
        f"| comparison | mean factor | SD | per realization | Wilcoxon p |\n|---|---|---|---|---|\n{lines}\n\n"
        f"Residual caveat: permuting one channel breaks cross-channel coherence, so it is not perfectly "
        f"in-distribution either. It is the closest standard probe.\n", encoding="utf-8")
    return 0  # D2 has no ESCALATE letter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
