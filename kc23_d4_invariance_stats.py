#!/usr/bin/env python3
"""
kc23_d4_invariance_stats.py
=============================
KC-D4 stats/gate. EXPERIMENT_PLAN_KC23_DEEP.md "KC-D4. The channel axis
measured in subject-invariance terms", D4.3 and D4.4. Rewritten 26 September
2026 to conform to the plan text of 23 September (KC23_PREREG_CONFORMANCE.md);
the earlier version did not match it (one Spearman over 5 dose means, an
invented rho > 0.3 cutoff, a 2.0 pt gap borrowed from D6's X1, Page blocks =
3 realizations, no Holm, and any unmatched case silently defaulted to T3).

Plan, D4.4:
  T1  Subject-probe accuracy falls with dose along `mpchandrop` (Page trend,
      Holm < 0.05), F1 peaks and then falls, and within-fold F1 tracks class
      silhouette (mean Spearman > 0, sign test) but not the subject probe.
  T2  The subject probe does not fall with dose.
  T3  T1's first two conditions hold but F1 does not track class information.
  Also: whether `gainjitter` shows a boundary by SD 1.00 (a one-line answer, no
  letter).

Operationalisation (Enam, 26 September 2026), applied to matrices of shape
(40 folds, 5 doses), realization-averaged per fold:
  (a) the unseen-subject probe falls with dose: Page trend, the 40 FOLDS as
      blocks, the 5 doses as treatments, Holm-adjusted across the two
      invariance meters (subject probe and permutation reliance), the
      subject-probe adjusted p < 0.05;
  (b) F1 peaks and then falls: the argmax of the mean F1 is not the highest
      dose, and F1 at the highest dose is below the peak by a paired Wilcoxon
      over the 40 folds, p < 0.05;
  (c) F1 tracks class information: per fold, Spearman across the 5 doses
      between F1 and class silhouette; mean > 0 and a sign test over the folds,
      positive, p < 0.05;
  (d) F1 does not track invariance: the same per-fold Spearman between F1 and
      the NEGATED subject probe is not significantly positive by the sign test.
  T1 = a, b, c and d.   T2 = not a.   T3 = a and b, but not c.
  Any other combination is T-OUT ("outside the pre-registered grid"): exit 10,
  reported, never defaulted into T3.
  Gainjitter boundary: over SD {0.30, 0.40, 0.50, 0.80, 1.00}, the argmax of the
  mean F1 is not at 1.00 and F1 at 1.00 is below that peak by a paired Wilcoxon,
  p < 0.05.

Inputs (kc23_d4_aggregate.py writes them; nothing here is optional):
  d4_dose_sweep.csv          realization, subject, dose, f1, subject_probe_bacc,
                             class_silhouette, class_probe_bacc, permutation_sum
  d4_gainjitter_boundary.csv realization, subject, dose, f1
A missing or incomplete input exits 20 with a verdict that carries no outcome
line. D4 has no ESCALATE letter: T1 to T3 exit 0, T-OUT exits 10.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_invariance_common import (ALPHA, holm_adjust, page_falls, peak_then_fall, tracks, tracks_invariance)
from kc23_stats_common import print_gate_header, write_no_outcome_verdict

MPCHANDROP_DOSES = [0.40, 0.50, 0.60, 0.80, 1.00]
GAINJITTER_DOSES = [0.30, 0.40, 0.50, 0.80, 1.00]
N_FOLDS = 40
REALIZATIONS = [42, 7, 123]
SWEEP_COLS = ["realization", "subject", "dose", "f1", "subject_probe_bacc", "class_silhouette",
              "class_probe_bacc", "permutation_sum"]


class InputError(Exception):
    pass


def _matrix(df: pd.DataFrame, col: str, doses: list[float], label: str) -> np.ndarray:
    """(folds, doses) matrix of `col`, averaged over realizations per fold. Every (realization, dose) must hold
    the same complete set of folds."""
    if set(np.round(df["dose"].unique(), 4)) != set(np.round(doses, 4)):
        raise InputError(f"{label}: doses {sorted(df['dose'].unique())} differ from the required {doses}")
    if set(df["realization"].unique()) != set(REALIZATIONS):
        raise InputError(f"{label}: realizations {sorted(df['realization'].unique())} differ from {REALIZATIONS}")
    if df[col].isna().any():
        raise InputError(f"{label}: NaN in {col}")
    piv = df.groupby(["subject", "dose"])[col].mean().unstack("dose")
    counts = df.groupby(["subject", "dose"]).size().unstack("dose")
    if piv.shape[0] != N_FOLDS or (counts != len(REALIZATIONS)).any().any():
        raise InputError(f"{label}: needs {N_FOLDS} folds x {len(doses)} doses x {len(REALIZATIONS)} realizations "
                         f"with no gaps or duplicates")
    piv = piv.reindex(columns=sorted(piv.columns))
    return piv.to_numpy(float)


def classify_t(f1: np.ndarray, probe: np.ndarray, sil: np.ndarray, perm: np.ndarray) -> tuple[str, dict]:
    """All four matrices (n_folds, n_doses), realization-averaged, doses ascending."""
    page_probe = page_falls(probe)
    page_perm = page_falls(perm)
    p_probe_holm, p_perm_holm = holm_adjust([page_probe["p"], page_perm["p"]])
    a = p_probe_holm < ALPHA

    pf = peak_then_fall(f1)
    b = pf["falls"]

    cls = tracks(f1, sil)
    c = cls["tracks"]
    inv = tracks_invariance(f1, probe)
    d = not inv["tracks_invariance"]

    detail = {
        "a_probe_falls": bool(a), "probe_page_p_raw": page_probe["p"], "probe_page_p_holm": p_probe_holm,
        "perm_page_p_raw": page_perm["p"], "perm_page_p_holm": p_perm_holm,
        "b_peaks_then_falls": bool(b), "peak_dose_idx": pf["peak_idx"], "peak_not_highest": pf["peak_not_highest"],
        "gap_to_highest_pp": pf["gap_to_highest_pp"], "f1_wilcoxon_p": pf["wilcoxon_p"],
        "c_tracks_class_silhouette": bool(c), "class_mean_rho": cls["mean_rho"], "class_n_pos": cls["n_pos"],
        "class_n_valid": cls["n_valid"], "class_sign_p": cls["sign_p"],
        "d_not_tracking_invariance": bool(d), "inv_mean_rho": inv["mean_rho"], "inv_n_pos": inv["n_pos"],
        "inv_n_valid": inv["n_valid"], "inv_sign_p": inv["sign_p"],
    }
    if not a:
        return "T2", detail
    if a and b and c and d:
        return "T1", detail
    if a and b and not c:
        return "T3", detail
    return "T-OUT", detail


def gainjitter_boundary(f1_mat: np.ndarray, doses: list[float]) -> tuple[bool, str, dict]:
    pf = peak_then_fall(f1_mat)
    peak_sd = doses[pf["peak_idx"]]
    if pf["falls"]:
        return True, (f"gainjitter shows a boundary: F1 peaks at SD {peak_sd:.2f} and is {pf['gap_to_highest_pp']:.2f} pt "
                      f"lower at SD {doses[-1]:.2f} (paired Wilcoxon p = {pf['wilcoxon_p']:.3g})"), pf
    if pf["peak_not_highest"]:
        return False, (f"no boundary shown by SD {doses[-1]:.2f}: F1 peaks at SD {peak_sd:.2f} but the fall to "
                       f"SD {doses[-1]:.2f} is {pf['gap_to_highest_pp']:.2f} pt, not significant (p = "
                       f"{pf['wilcoxon_p']:.3g})"), pf
    return False, f"no boundary by SD {doses[-1]:.2f}: F1 is still highest at the top dose", pf


def _fail(out_dir: Path, reason: str) -> int:
    print(f"[D4] FAIL (no outcome computed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(out_dir / "D4_VERDICT.md", "KC-D4 verdict", reason)
    (out_dir / "D4_detail.csv").unlink(missing_ok=True)
    return 20


def run(out_dir: Path) -> int:
    try:
        sweep_path, gj_path = out_dir / "d4_dose_sweep.csv", out_dir / "d4_gainjitter_boundary.csv"
        for p in (sweep_path, gj_path):
            if not p.exists():
                raise InputError(f"required input missing: {p}")
        sw = pd.read_csv(sweep_path)
        gj = pd.read_csv(gj_path)
        if not set(SWEEP_COLS) <= set(sw.columns) or not {"realization", "subject", "dose", "f1"} <= set(gj.columns):
            raise InputError("d4_dose_sweep.csv or d4_gainjitter_boundary.csv lacks required columns")
        f1 = _matrix(sw, "f1", MPCHANDROP_DOSES, "mpchandrop f1")
        probe = _matrix(sw, "subject_probe_bacc", MPCHANDROP_DOSES, "mpchandrop subject probe")
        sil = _matrix(sw, "class_silhouette", MPCHANDROP_DOSES, "mpchandrop class silhouette")
        perm = _matrix(sw, "permutation_sum", MPCHANDROP_DOSES, "mpchandrop permutation reliance")
        cprobe = _matrix(sw, "class_probe_bacc", MPCHANDROP_DOSES, "mpchandrop class probe")
        gj_f1 = _matrix(gj, "f1", GAINJITTER_DOSES, "gainjitter f1")
    except InputError as e:
        return _fail(out_dir, str(e))

    letter, detail = classify_t(f1, probe, sil, perm)
    reading = {
        "T1": "The channel axis now measures the same kind of invariance as the alignment axis.",
        "T2": "The channel axis does not move subject invariance; restricted to the alignment axis.",
        "T3": "The shape is shared; the mechanism stays 'proposed' on the channel axis.",
        "T-OUT": "Outside the pre-registered grid (T1 to T3): reported, not defaulted into a letter.",
    }[letter]
    print_gate_header("KC-D4", letter, reading)
    print(f"  {detail}")
    has_boundary, gj_line, gj_detail = gainjitter_boundary(gj_f1, GAINJITTER_DOSES)
    print(f"  gainjitter: {gj_line}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"letter": letter, **detail}]).to_csv(out_dir / "D4_detail.csv", index=False)
    per_dose = pd.DataFrame({"dose": MPCHANDROP_DOSES, "f1": f1.mean(0), "subject_probe_bacc": probe.mean(0),
                             "permutation_sum_pp": perm.mean(0), "class_silhouette": sil.mean(0),
                             "class_probe_bacc": cprobe.mean(0)})
    per_dose.to_csv(out_dir / "D4_per_dose.csv", index=False)
    table = per_dose.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    outcome = letter if letter != "T-OUT" else "T-OUT (outside the pre-registered grid)"
    (out_dir / "D4_VERDICT.md").write_text(
        f"# KC-D4 verdict\n\n**Outcome: {outcome}**\n\n{reading}\n\n"
        f"Conditions (folds are the blocks; 3 realizations averaged per fold): "
        f"(a) subject probe falls with dose, Page Holm p = {detail['probe_page_p_holm']:.3g} -> {detail['a_probe_falls']}; "
        f"(b) F1 peaks then falls (peak at dose index {detail['peak_dose_idx']}, {detail['gap_to_highest_pp']:.2f} pt above the "
        f"highest dose, paired Wilcoxon p = {detail['f1_wilcoxon_p']:.3g}) -> {detail['b_peaks_then_falls']}; "
        f"(c) F1 tracks class silhouette (mean rho {detail['class_mean_rho']:.3f}, {detail['class_n_pos']} of "
        f"{detail['class_n_valid']} folds positive, sign p = {detail['class_sign_p']:.3g}) -> {detail['c_tracks_class_silhouette']}; "
        f"(d) F1 does not track invariance (mean rho {detail['inv_mean_rho']:.3f}, {detail['inv_n_pos']} of "
        f"{detail['inv_n_valid']} folds positive, sign p = {detail['inv_sign_p']:.3g}) -> {detail['d_not_tracking_invariance']}.\n\n"
        f"Per dose (mean over folds and realizations):\n\n```\n{table}\n```\n\n"
        f"Gainjitter boundary: {gj_line}.\n", encoding="utf-8")
    return 10 if letter == "T-OUT" else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
