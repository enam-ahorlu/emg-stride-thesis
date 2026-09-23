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

ResNet-SE+CD against the SVM (65.7) is reported as a number with its
across-seed SD, not a letter -- feeds the abstract's replication sentence
directly, computed here but not classified.

With n=10, no letter here implies a halt; KC-D5 has no ESCALATE outcome.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import print_gate_header

N_SUBJECTS = 10
SVM_ENABL3S = 0.657


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


def run(out_dir: Path) -> int:
    findings = {}
    diffs_path = out_dir / "d5_finding_diffs.csv"  # cols: finding, subject, diff, expected_sign
    if diffs_path.exists():
        df = pd.read_csv(diffs_path)
        for finding, g in df.groupby("finding"):
            sign = int(g["expected_sign"].iloc[0])
            letter, detail = classify_er(g.sort_values("subject")["diff"].to_numpy(), sign)
            findings[finding] = {"letter": letter, **detail}
            print_gate_header(f"KC-D5 {finding}", letter,
                              "May be listed as replicated." if letter == "E-R" else "Listed as not replicated.")

    resnet_vs_svm_path = out_dir / "d5_resnet_vs_svm.csv"  # cols: realization/seed, f1_resnet
    resnet_summary = None
    if resnet_vs_svm_path.exists():
        r = pd.read_csv(resnet_vs_svm_path)
        mean_f1 = float(r["f1_resnet"].mean())
        sd_f1 = float(r["f1_resnet"].std(ddof=1)) if len(r) > 1 else 0.0
        resnet_summary = {"resnet_se_cd_mean": mean_f1, "resnet_se_cd_sd": sd_f1,
                          "svm_enabl3s": SVM_ENABL3S, "delta_pp": (mean_f1 - SVM_ENABL3S) * 100.0}
        print(f"[D5] ResNet-SE+CD {mean_f1:.4f} +/- {sd_f1:.4f} vs SVM {SVM_ENABL3S:.4f} "
             f"(delta {resnet_summary['delta_pp']:+.2f}pp)")

    out_dir.mkdir(parents=True, exist_ok=True)
    if findings:
        pd.DataFrame([{"finding": k, **v} for k, v in findings.items()]).to_csv(out_dir / "D5_findings.csv", index=False)
    verdict_lines = ["# KC-D5 verdict\n"]
    for finding, v in findings.items():
        verdict_lines.append(f"- **{finding}: {v['letter']}**")
    if resnet_summary:
        verdict_lines.append(f"\nResNet-SE+CD {resnet_summary['resnet_se_cd_mean']:.4f} +/- "
                             f"{resnet_summary['resnet_se_cd_sd']:.4f} vs SVM {SVM_ENABL3S:.4f}")
    (out_dir / "D5_VERDICT.md").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")
    return 0  # KC-D5 has no ESCALATE outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
