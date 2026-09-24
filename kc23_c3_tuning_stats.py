#!/usr/bin/env python3
"""
kc23_c3_tuning_stats.py
=========================
KC-C3 stats/gate. EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C3. Classical tuning
parity and two new families", C3.5.

Endpoint 1, the deep lead (ResNet-SE+CD - best tuned classical, per-subject norm):
  P1: best tuned classical within 1pt of published SVM (77.7%)       -> lead stands
  P2: best tuned classical gains 1 to 3pt                            -> ESCALATE (framing)
  P3: best tuned classical within 1pt of ResNet-SE+CD, or above it   -> ESCALATE

Endpoint 2, Finding A on the new families (per-subject minus global gain):
  N1: positive and significant on every family  -> extends to six classical families
  N2: any family with a gain <= 0                -> ESCALATE

Endpoint 3, the ensemble (soft vote with SVM-X probabilities against 85.8%):
  E1: |delta| < 0.5pt   -> report
  E2: |delta| >= 0.5pt  -> ESCALATE
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import paired_test, holm, print_gate_header, read_subjectwise, require_complete

PUBLISHED_SVM = 0.777
PUBLISHED_ENSEMBLE = 0.858

# Real, confirmed paths (fixed 2026-09-24 -- this gate originally expected
# resnet_se_cd_persubj_subjectwise.csv / {fam}_persubj_subjectwise.csv /
# ensemble_svmx_subjectwise.csv inside its OWN --out, none of which anything
# writes there). train_classical_loso.py's real subjectwise file is
# {features_stem}__{MODEL}_nested_loso_subjectwise.csv, one per
# results_kc23_c3_<model>_<norm>/ directory (columns: model, heldout_subject,
# ..., f1_macro, ...; read_subjectwise already tolerates heldout_subject).
# The published ResNet-SE+CD reference lives in the pre-existing
# results_cnn_aug_resnet_se_chandrop/ (not a KC23 job -- C3 is classical-only).
# The tuned-SVM ensemble figure is the "SVM+RESNET_SE [soft]" column of
# results_kc23_c3_ensemble/ensemble_v2_subjectwise.csv (wide: one column per
# combiner x subset, produced by kc23_c3_merge_proba.py + ensemble_v2_combine.py)
# -- the published headline combiner (KC23_HALT.md decision D-6a), now fed the
# KC-C3-tuned SVM proba in place of the original.
FAMILY_DIRS = {
    "svmx": "results_kc23_c3_svm_{norm}",
    "rfx": "results_kc23_c3_rf_{norm}",
    "hgb": "results_kc23_c3_hgb_{norm}",
    "knn": "results_kc23_c3_knn_{norm}",
}
RESNET_CD_DIR = "results_cnn_aug_resnet_se_chandrop"
ENSEMBLE_COLUMN = "SVM+RESNET_SE [soft]"


def load_ensemble_column(path: Path, column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"{path}: column {column!r} not found (have {list(df.columns)})")
    return df[["subject", column]].rename(columns={column: "f1_macro"})


def classify_p(f1_resnet_cd: np.ndarray, f1_best_classical: np.ndarray) -> tuple[str, dict]:
    best_mean = float(f1_best_classical.mean())
    gain_over_svm_pp = (best_mean - PUBLISHED_SVM) * 100.0
    t = paired_test(f1_resnet_cd, f1_best_classical, "resnet_cd_minus_best_classical", "C3-P")
    lead_pp = t["delta_pp"]
    if gain_over_svm_pp < 1.0:
        letter = "P1"
    elif 1.0 <= gain_over_svm_pp < 3.0:
        letter = "P2"
    else:
        letter = "P3"
    # P3 also fires directly if the classical model is within 1pt of, or beats, the deep model
    if lead_pp < 1.0:
        letter = "P3"
    return letter, {**t, "gain_over_published_svm_pp": gain_over_svm_pp}


def classify_n(family_gains: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[str, list[dict]]:
    """family_gains: {family_name: (f1_persubj, f1_global)}."""
    rows = []
    for fam, (persubj, glob) in family_gains.items():
        t = paired_test(persubj, glob, fam, "C3-N")
        rows.append(t)
    rows = holm(rows)
    all_positive_significant = all(r["delta_pp"] > 0 and r["p_holm"] < 0.05 for r in rows)
    any_nonpositive = any(r["delta_pp"] <= 0 for r in rows)
    letter = "N2" if any_nonpositive else ("N1" if all_positive_significant else "N2")
    return letter, rows


def classify_e(f1_ensemble_svmx: np.ndarray) -> tuple[str, dict]:
    mean_f1 = float(f1_ensemble_svmx.mean())
    delta_pp = (mean_f1 - PUBLISHED_ENSEMBLE) * 100.0
    letter = "E1" if abs(delta_pp) < 0.5 else "E2"
    return letter, {"mean_f1": mean_f1, "delta_pp": delta_pp, "published": PUBLISHED_ENSEMBLE}


def _fail_closed(out_dir: Path, reason: str) -> int:
    print(f"[C3] MISSING (fail closed): {reason}", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "C3_VERDICT.md").write_text(
        f"# KC-C3 verdict\n\n**Outcome: FAIL (missing input)**\n\n{reason}\n", encoding="utf-8")
    return 20


def run(out_dir: Path, root: Path | None = None) -> int:
    """root: repo root the family/ensemble/resnet_cd directories live under
    (defaults to out_dir's own parent, since results_kc23_c3_ensemble is a
    sibling of results_kc23_c3_svm_per_subject etc., not a child of it)."""
    root = root or out_dir.parent
    fired = []
    rows_all = []

    resnet_dir = root / RESNET_CD_DIR
    if not resnet_dir.exists():
        return _fail_closed(out_dir, f"{resnet_dir} not found (published ResNet-SE+CD reference)")
    resnet_cd = require_complete(read_subjectwise(resnet_dir), 40, "resnet_cd")

    candidates = {}
    for fam, pattern in FAMILY_DIRS.items():
        d = root / pattern.format(norm="per_subject")
        if not d.exists():
            return _fail_closed(out_dir, f"{d} not found (family {fam!r}, per_subject)")
        candidates[fam] = require_complete(read_subjectwise(d), 40, fam)
    best_fam = max(candidates, key=lambda k: candidates[k].mean())
    best_classical = candidates[best_fam]
    print(f"[C3] best tuned classical family: {best_fam} (mean F1 {best_classical.mean():.4f})")

    p_letter, p_stats = classify_p(resnet_cd, best_classical)
    fired.append(p_letter)
    rows_all.append(p_stats)
    print_gate_header("KC-C3 Endpoint 1", p_letter, {
        "P1": "The 6.3pt lead stands.", "P2": "ESCALATE: the lead narrows, framing choice.",
        "P3": "ESCALATE: Finding C's strongest single model changes.",
    }[p_letter])

    family_gains = {}
    for fam, pattern in FAMILY_DIRS.items():
        gd = root / pattern.format(norm="global")
        if not gd.exists():
            return _fail_closed(out_dir, f"{gd} not found (family {fam!r}, global)")
        family_gains[fam] = (candidates[fam], require_complete(read_subjectwise(gd), 40, fam + "_global"))
    n_letter, n_rows = classify_n(family_gains)
    fired.append(n_letter)
    rows_all.extend(n_rows)
    print_gate_header("KC-C3 Endpoint 2", n_letter, {
        "N1": "Every model tried extends to six classical families.",
        "N2": "ESCALATE: Finding A's scope wording changes.",
    }[n_letter])

    ens_p = out_dir / "ensemble_v2_subjectwise.csv"
    if not ens_p.exists():
        return _fail_closed(out_dir, f"{ens_p} not found (run kc23_c3_merge_proba.py + "
                            f"ensemble_v2_combine.py first)")
    f1_ens = require_complete(load_ensemble_column(ens_p, ENSEMBLE_COLUMN), 40, "ensemble_svmx")
    e_letter, e_stats = classify_e(f1_ens)
    fired.append(e_letter)
    rows_all.append(e_stats)
    print_gate_header("KC-C3 Endpoint 3", e_letter, {
        "E1": "Report.", "E2": "ESCALATE: touches the headline.",
    }[e_letter])

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows_all:
        pd.DataFrame(rows_all).to_csv(out_dir / "C3_tests.csv", index=False)
    verdict = f"# KC-C3 verdict\n\n**Outcomes fired: {fired}**\n"
    (out_dir / "C3_VERDICT.md").write_text(verdict, encoding="utf-8")

    escalate = {"P2", "P3", "N2", "E2"}
    if any(l in escalate for l in fired):
        return 20
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
