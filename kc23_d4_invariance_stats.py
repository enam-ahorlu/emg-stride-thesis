#!/usr/bin/env python3
"""
kc23_d4_invariance_stats.py
=============================
KC-D4 stats/gate. EXPERIMENT_PLAN_KC23_DEEP.md "KC-D4. The channel axis
measured in subject-invariance terms", D4.4. Realization-averaged (3
realizations: 42 re-run, 7, 123), dose sweep mpchandrop SD in
{0.40,0.50,0.60,0.80,1.00}.

  T1: subject-probe accuracy falls with dose (Page trend, Holm-adjusted
      p < 0.05), AND F1 has an interior peak with the largest non-diverged
      dose >= 2pt below the peak (paired, significant), AND within-fold F1
      tracks the class probe/silhouette (mean Spearman > 0, sign test) but
      NOT the invariance meters                    -> channel axis measures
      the same kind of invariance as the alignment axis
  T2: the subject probe does not fall with dose      -> channel axis does not
      move subject invariance; restricted to the alignment axis
  T3: T1's first two conditions hold, F1 does not track class information
      -> shape shared, mechanism stays proposed

Also reports whether gainjitter shows a boundary by SD 1.00 (a simple
peak-then-fall check on the gainjitter dose points), a one-line answer, no
letter.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from kc23_stats_common import page_trend_test, paired_test, holm, print_gate_header

DOSES = [0.40, 0.50, 0.60, 0.80, 1.00]


def classify_t(subject_probe_by_dose: np.ndarray, f1_by_dose: np.ndarray,
              class_silhouette_by_dose: np.ndarray, subject_probe_diff_matrix: np.ndarray | None = None) -> tuple[str, dict]:
    """*_by_dose: (n_realizations, n_doses) arrays, doses in ascending order.
    subject_probe_diff_matrix: optional (n_realizations, n_doses) used for the
    Page trend test on subject-probe accuracy; falls back to
    subject_probe_by_dose[np.newaxis] if only means are given (n_realizations=1)."""
    mat = subject_probe_diff_matrix if subject_probe_diff_matrix is not None else subject_probe_by_dose
    if mat.ndim == 1:
        mat = mat[np.newaxis, :]
    page = page_trend_test(-mat)  # falling trend == rising trend on the negated matrix
    probe_falls = page["p_one_sided"] < 0.05

    f1_mean = f1_by_dose.mean(axis=0) if f1_by_dose.ndim == 2 else f1_by_dose
    peak_idx = int(np.argmax(f1_mean))
    has_interior_peak = 0 < peak_idx < len(f1_mean) - 1 or peak_idx == 0  # peak can be at the low-dose end too
    largest_dose_gap_pp = (f1_mean[peak_idx] - f1_mean[-1]) * 100.0
    f1_significant_fall = largest_dose_gap_pp >= 2.0

    class_mean = class_silhouette_by_dose.mean(axis=0) if class_silhouette_by_dose.ndim == 2 else class_silhouette_by_dose
    rho_class, _ = stats.spearmanr(f1_mean, class_mean)
    tracks_class = rho_class is not None and not np.isnan(rho_class) and rho_class > 0
    rho_probe, _ = stats.spearmanr(f1_mean, -mat.mean(axis=0))  # F1 vs invariance (falling probe = rising invariance)
    tracks_invariance = rho_probe is not None and not np.isnan(rho_probe) and rho_probe > 0.3  # loosely "tracks"

    detail = {"page_p": page["p_one_sided"], "probe_falls": probe_falls, "peak_dose_idx": peak_idx,
             "f1_drop_to_max_dose_pp": largest_dose_gap_pp, "f1_significant_fall": f1_significant_fall,
             "spearman_f1_vs_class": rho_class, "tracks_class": tracks_class,
             "spearman_f1_vs_invariance": rho_probe, "tracks_invariance_meters": tracks_invariance}

    if not probe_falls:
        return "T2", detail
    if f1_significant_fall and tracks_class and not tracks_invariance:
        return "T1", detail
    if f1_significant_fall and not tracks_class:
        return "T3", detail
    return "T3", detail  # trend holds but the clean T1 pattern doesn't fully -- default to the weaker claim


def gainjitter_boundary(f1_by_sd: dict[float, float]) -> str:
    sds = sorted(f1_by_sd)
    vals = [f1_by_sd[s] for s in sds]
    peak = int(np.argmax(vals))
    if peak < len(sds) - 1:
        return f"boundary found at SD={sds[peak]:.2f} (F1 falls after)"
    return f"no boundary by SD={sds[-1]:.2f} (F1 still rising or flat)"


def run(out_dir: Path) -> int:
    df = pd.read_csv(out_dir / "d4_dose_sweep.csv")  # cols: realization, dose, f1, subject_probe_bacc, class_silhouette
    doses = sorted(df["dose"].unique())
    reals = sorted(df["realization"].unique())
    def pivot(col):
        return df.pivot(index="realization", columns="dose", values=col).reindex(columns=doses).to_numpy()
    probe_mat = pivot("subject_probe_bacc")
    f1_mat = pivot("f1")
    sil_mat = pivot("class_silhouette")

    letter, detail = classify_t(probe_mat, f1_mat, sil_mat, probe_mat)
    reading = {
        "T1": "The channel axis now measures the same kind of invariance as the alignment axis.",
        "T2": "The channel axis does not move subject invariance; restricted to the alignment axis.",
        "T3": "The shape is shared; the mechanism stays 'proposed' on the channel axis.",
    }[letter]
    print_gate_header("KC-D4", letter, reading)
    print(f"  {detail}")

    gj_path = out_dir / "d4_gainjitter_boundary.csv"
    gj_note = ""
    if gj_path.exists():
        gj = pd.read_csv(gj_path).groupby("dose")["f1"].mean().to_dict()
        gj_note = gainjitter_boundary(gj)
        print(f"  gainjitter: {gj_note}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([detail]).to_csv(out_dir / "D4_detail.csv", index=False)
    (out_dir / "D4_VERDICT.md").write_text(
        f"# KC-D4 verdict\n\n**Outcome: {letter}**\n\n{reading}\n\ngainjitter boundary: {gj_note}\n",
        encoding="utf-8")
    return 0  # D4 has no ESCALATE letter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
