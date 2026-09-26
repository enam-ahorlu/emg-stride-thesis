#!/usr/bin/env python3
"""
kc23_d5_replication_stats.py
==============================
KC-D5 stats/gate. EXPERIMENT_PLAN_KC23_DEEP.md "KC-D5. ENABL3S deep mechanism
replication", n=10 subjects, 5 realizations (42 re-run, 7, 123, 1001, 2026).

Per finding (channel-dropout gain E2-E1, gain jitter vs channel dropout
E3-E2, occlusion reduction, permutation reduction):
  E-R: the finding's direction holds in the realization average, with >= 7 of
       10 subjects agreeing on sign                    -> may be listed as replicated
  E-N: otherwise                                         -> listed as not replicated

"The finding's direction" is the direction the thesis states for that finding
on SIAT-LLMD (decision D-6c, 25 September 2026), encoded below in
EXPECTED_SIGN and never read from the input CSV (a CSV sign that disagrees is
an error). It is not chosen from the ENABL3S numbers. Both contested readings
take the reading less favourable to the thesis:
  gainjitter_vs_chandrop  +1  (Section 4.3.3: gain jitter is AHEAD of channel dropout)
  permutation_reduction   +1  (the reliance claim is a REDUCTION under channel dropout)

ResNet-SE+CD against the SVM (65.7) is reported as a number with its
across-seed SD, not a letter -- feeds the abstract's replication sentence
directly. The occlusion magnitude is reported beside its letter: the thesis
may say the direction replicates, never "six-fold", on ENABL3S.

Fail closed (rewritten 2026-09-25). The previous version treated both input
files as optional, so with neither present it wrote a header-only verdict and
exited 0. Nothing produced those files. Now they come from
kc23_d5_aggregate.py, and any missing or malformed input, any finding absent
or with a wrong subject count, exits 20 with a verdict that contains no
letter. No published number is substituted for a missing input.

With n=10, no letter here implies a halt; KC-D5 has no ESCALATE outcome, so a
computed result exits 0 and only a failure exits 20.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import print_gate_header, write_no_outcome_verdict

N_SUBJECTS = 10
SVM_ENABL3S = 0.657
EXPECTED_SIGN = {
    "chandrop_gain": 1,
    "gainjitter_vs_chandrop": 1,
    "occlusion_reduction": 1,
    "permutation_reduction": 1,
}
OCCLUSION_MAG = "occlusion_reduction_factor_E1_over_E2"
SIAT_OCCLUSION_REFERENCE = "about 6x on SIAT-LLMD"


def classify_er(realization_avg_diff_per_subject: np.ndarray, expected_sign: int = 1) -> tuple[str, dict]:
    """realization_avg_diff_per_subject: length-10 array, each subject's diff
    already averaged across realizations. expected_sign: +1 if the finding
    predicts a positive diff, -1 if negative."""
    agree = int(((np.sign(realization_avg_diff_per_subject) == np.sign(expected_sign))
                | (realization_avg_diff_per_subject == 0)).sum())
    direction_holds = np.sign(realization_avg_diff_per_subject.mean()) == np.sign(expected_sign)
    letter = "E-R" if (direction_holds and agree >= 7) else "E-N"
    return letter, {"n_agree": agree, "n_subjects": len(realization_avg_diff_per_subject),
                    "mean_diff": float(realization_avg_diff_per_subject.mean()), "direction_holds": bool(direction_holds)}


class MissingInput(Exception):
    pass


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MissingInput(f"required input missing: {path}")
    return pd.read_csv(path)


def load_inputs(out_dir: Path):
    diffs = _read(out_dir / "d5_finding_diffs.csv")
    resnet = _read(out_dir / "d5_resnet_vs_svm.csv")
    mags = _read(out_dir / "d5_magnitudes.csv")
    need = {"finding", "subject", "diff", "expected_sign"}
    if not need <= set(diffs.columns):
        raise MissingInput(f"d5_finding_diffs.csv lacks columns {sorted(need - set(diffs.columns))}")
    if "f1_resnet" not in resnet.columns or len(resnet) < 2:
        raise MissingInput("d5_resnet_vs_svm.csv needs an f1_resnet column with at least 2 realizations")
    if not {"quantity", "mean", "sd_across_seeds"} <= set(mags.columns):
        raise MissingInput("d5_magnitudes.csv lacks quantity/mean/sd_across_seeds")
    found = set(diffs["finding"])
    if found != set(EXPECTED_SIGN):
        raise MissingInput(f"findings present {sorted(found)} differ from the required {sorted(EXPECTED_SIGN)}")
    for name, g in diffs.groupby("finding"):
        if len(g) != N_SUBJECTS or g["subject"].duplicated().any() or g["diff"].isna().any():
            raise MissingInput(f"finding {name}: needs {N_SUBJECTS} unique non-NaN subjects, found {len(g)} rows")
        if int(g["expected_sign"].iloc[0]) != EXPECTED_SIGN[name]:
            raise MissingInput(f"finding {name}: CSV expected_sign {g['expected_sign'].iloc[0]} disagrees with the "
                               f"direction encoded here ({EXPECTED_SIGN[name]}, decision D-6c)")
    if OCCLUSION_MAG not in set(mags["quantity"]):
        raise MissingInput(f"d5_magnitudes.csv lacks {OCCLUSION_MAG}")
    return diffs, resnet, mags


def run(out_dir: Path) -> int:
    try:
        diffs, resnet, mags = load_inputs(out_dir)
    except MissingInput as e:
        print(f"[D5] FAIL (no outcome computed): {e}", file=sys.stderr)
        write_no_outcome_verdict(out_dir / "D5_VERDICT.md", "KC-D5 verdict", str(e))
        return 20

    findings = {}
    for finding in EXPECTED_SIGN:
        g = diffs[diffs["finding"] == finding]
        letter, detail = classify_er(g.sort_values("subject")["diff"].to_numpy(), EXPECTED_SIGN[finding])
        findings[finding] = {"letter": letter, "expected_sign": EXPECTED_SIGN[finding], **detail}
        print_gate_header(f"KC-D5 {finding}", letter,
                          "May be listed as replicated." if letter == "E-R" else "Listed as not replicated.")

    mean_f1 = float(resnet["f1_resnet"].mean())
    sd_f1 = float(resnet["f1_resnet"].std(ddof=1))
    resnet_summary = {"resnet_se_cd_mean": mean_f1, "resnet_se_cd_sd": sd_f1,
                      "svm_enabl3s": SVM_ENABL3S, "delta_pp": (mean_f1 - SVM_ENABL3S) * 100.0}
    print(f"[D5] ResNet-SE+CD {mean_f1:.4f} +/- {sd_f1:.4f} vs SVM {SVM_ENABL3S:.4f} "
          f"(delta {resnet_summary['delta_pp']:+.2f}pp)")

    occ = mags[mags["quantity"] == OCCLUSION_MAG].iloc[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"finding": k, **v} for k, v in findings.items()]).to_csv(out_dir / "D5_findings.csv", index=False)

    lines = ["# KC-D5 verdict\n"]
    for finding, v in findings.items():
        lines.append(f"- **{finding}: {v['letter']}** (mean diff {v['mean_diff']:+.3f}, "
                     f"{v['n_agree']} of {v['n_subjects']} subjects agree with the expected sign "
                     f"{'+' if v['expected_sign'] > 0 else '-'})")
    lines.append("")
    lines.append(f"Occlusion magnitude beside its letter: reduction factor {occ['mean']:.2f}x "
                 f"(SD {occ['sd_across_seeds']:.2f} across the 5 realizations) on ENABL3S, against "
                 f"{SIAT_OCCLUSION_REFERENCE}. The thesis may say the direction replicates; it must not say "
                 f"'six-fold' for ENABL3S.")
    lines.append("")
    lines.append(f"ResNet-SE+CD {resnet_summary['resnet_se_cd_mean']:.4f} +/- "
                 f"{resnet_summary['resnet_se_cd_sd']:.4f} (SD across 5 realizations) against the ENABL3S SVM "
                 f"{SVM_ENABL3S:.4f}: {resnet_summary['delta_pp']:+.2f} pp. Reported as a number, not a letter.")
    lines.append("")
    lines.append("Directions are those the thesis states on SIAT-LLMD (decision D-6c), encoded in the script: "
                 "gain jitter ahead of channel dropout (positive) and a permutation-reliance reduction under "
                 "channel dropout (positive). Both take the reading less favourable to the thesis.")
    (out_dir / "D5_VERDICT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0  # KC-D5 has no ESCALATE outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
