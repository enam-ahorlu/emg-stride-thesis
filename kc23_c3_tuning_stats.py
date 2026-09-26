#!/usr/bin/env python3
"""
kc23_c3_tuning_stats.py
=========================
KC-C3 stats/gate. EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C3. Classical tuning
parity and two new families", C3.3 to C3.5.

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

Conformance pass, 26 September 2026 (KC23_PREREG_CONFORMANCE.md). The plan's tables do not cover every case, and
the earlier code silently put the uncovered ones into a letter:
  - Endpoint 1: a best tuned classical that gains 3 or more points over the SVM but is still more than 1 point behind
    ResNet-SE+CD was called P3; the plan's P3 is "within 1pt of ResNet-SE+CD, or above it", and P2 stops at 3 pts. A
    best classical MORE than 1 pt below the published SVM was called P1 ("within 1pt"). Both are now "P-OUT (outside
    the pre-registered grid)", exit 10, reported.
  - Endpoint 2: every family positive but not every gain significant was called N2 ("any family with a gain <= 0").
    It is now "N-OUT", exit 10, reported. Significance keeps the house convention (Holm within the family).
Also added, as the plan requires of C3_VERDICT.md: the edge-hit table (the edge rule: if the selected SVM C or gamma
sits on a grid edge in MORE THAN 10 of 40 folds, extend that axis by two steps once and rerun; report both runs) and
fit-time totals; and the soft vote using the best new classical member (RF-X or HGB or SVM-X, whichever has the best
per-subject-normalization mean F1 among the families that save probabilities), reported beside Endpoint 3. RF-X global
is optional in the plan and, if absent, is skipped with the fact stated.

Real paths (fixed 2026-09-24): train_classical_loso.py's subjectwise file is {features_stem}__{MODEL}_nested_loso_
subjectwise.csv, one per results_kc23_c3_<model>_<norm>/ directory. The published ResNet-SE+CD reference is the
pre-existing results_cnn_aug_resnet_se_chandrop/. The tuned-SVM ensemble is the "SVM+RESNET_SE [soft]" column of
results_kc23_c3_ensemble/ensemble_v2_subjectwise.csv, built by kc23_c3_merge_proba.py (the tuned SVM in place of the
published one, with RF, CNN and ResNet-SE+CD from results_ensemble_v2/proba_aug_chandrop, the probabilities behind the
85.8% headline) and ensemble_v2_combine.py. Any missing input exits 20 with a verdict that carries no outcome line.
"""
from __future__ import annotations
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import (paired_test, holm, print_gate_header, read_subjectwise, require_complete,
                               write_no_outcome_verdict)

PUBLISHED_SVM = 0.777
PUBLISHED_ENSEMBLE = 0.858

FAMILY_DIRS = {
    "svmx": "results_kc23_c3_svm_{norm}",
    "rfx": "results_kc23_c3_rf_{norm}",
    "hgb": "results_kc23_c3_hgb_{norm}",
    "knn": "results_kc23_c3_knn_{norm}",
}
RESNET_CD_DIR = "results_cnn_aug_resnet_se_chandrop"
ENSEMBLE_COLUMN = "SVM+RESNET_SE [soft]"
OPTIONAL_GLOBAL = {"rfx"}        # the plan: "RF-X ... global if time allows (optional, marked)"
PROBA_FAMILIES = {"svmx": ("SVM", "results_kc23_c3_svm_per_subject/proba"),
                  "rfx": ("RF", "results_kc23_c3_rf_per_subject/proba"),
                  "hgb": ("HGB", "results_kc23_c3_hgb_per_subject/proba")}
RESNET_PROBA_DIR = "results_ensemble_v2/proba_aug_chandrop"
N_SUBJECTS = 40
SVM_C_BASE = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
SVM_GAMMA_MULT_BASE = [0.01, 0.1, 0.3, 1, 3, 10]
EDGE_FOLDS_MORE_THAN = 10
EDGE_RERUN_DIR = "results_kc23_c3_svm_{norm}_edge"     # written by the edge-rule rerun rows (kc23_c3_edge_job_gen.py)


class InputError(Exception):
    pass


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
    if lead_pp < 1.0:                       # within 1pt of ResNet-SE+CD, or above it
        letter = "P3"
    elif abs(gain_over_svm_pp) < 1.0:       # within 1pt of the published SVM
        letter = "P1"
    elif 1.0 <= gain_over_svm_pp < 3.0:
        letter = "P2"
    else:
        letter = "P-OUT"                    # e.g. gains >= 3 pts yet still > 1 pt behind ResNet-SE+CD; or > 1 pt worse than the SVM
    return letter, {**t, "gain_over_published_svm_pp": gain_over_svm_pp}


def classify_n(family_gains: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[str, list[dict]]:
    """family_gains: {family_name: (f1_persubj, f1_global)}."""
    rows = []
    for fam, (persubj, glob) in family_gains.items():
        t = paired_test(persubj, glob, fam, "C3-N")
        rows.append(t)
    rows = holm(rows)
    all_positive = all(r["delta_pp"] > 0 for r in rows)
    all_significant = all(r["p_holm"] < 0.05 for r in rows)
    any_nonpositive = any(r["delta_pp"] <= 0 for r in rows)
    if any_nonpositive:
        letter = "N2"
    elif all_positive and all_significant:
        letter = "N1"
    else:
        letter = "N-OUT"        # every gain positive, but not every one significant: the plan has no letter for it
    return letter, rows


def classify_e(f1_ensemble_svmx: np.ndarray) -> tuple[str, dict]:
    mean_f1 = float(f1_ensemble_svmx.mean())
    delta_pp = (mean_f1 - PUBLISHED_ENSEMBLE) * 100.0
    letter = "E1" if abs(delta_pp) < 0.5 else "E2"
    return letter, {"mean_f1": mean_f1, "delta_pp": delta_pp, "published": PUBLISHED_ENSEMBLE}


# --------------------------------------------------------------------------------------------- edge rule, fit time
def _subjectwise(d: Path, label: str) -> pd.DataFrame:
    if not d.exists():
        raise InputError(f"{d} not found ({label})")
    try:
        df = read_subjectwise(d, model_token="")
    except FileNotFoundError as e:
        raise InputError(str(e))
    if df["subject"].duplicated().any() or len(df) != N_SUBJECTS:
        raise InputError(f"{label}: needs {N_SUBJECTS} unique subjects, found {len(df)} rows")
    return df.sort_values("subject").reset_index(drop=True)


def svm_edge_hits(svm_dir: Path, norm: str) -> dict:
    """The C3.3 edge rule for the SVM-X run in one directory. The multiplier chosen per fold is in svm_extended_gamma.csv
    (train_classical_loso.py writes it next to the subjectwise csv). If the run already used an EXTENDED axis (its grid
    differs from the plan's base grid) the rule has been applied once and is not triggered again."""
    sw = _subjectwise(svm_dir, f"SVM-X {norm}")
    if "best_params" not in sw.columns:
        raise InputError(f"{svm_dir.name}: no best_params column")
    side_path = svm_dir / "svm_extended_gamma.csv"
    if not side_path.exists():
        raise InputError(f"{side_path} missing: the chosen gamma multiplier is needed for the edge-hit table")
    side = pd.read_csv(side_path).drop_duplicates("heldout_subject").sort_values("heldout_subject")
    if len(side) != N_SUBJECTS:
        raise InputError(f"{side_path.name}: needs {N_SUBJECTS} folds, found {len(side)}")
    cs = np.array([float(ast.literal_eval(p)["clf__C"]) for p in sw["best_params"]])
    mult = side["best_gamma_mult"].to_numpy(float)
    c_grid = sorted(float(x) for x in str(side["c_grid"].iloc[0]).split(";"))
    g_grid = sorted(float(x) for x in str(side["gamma_mult_grid"].iloc[0]).split(";"))
    extended = (c_grid != SVM_C_BASE) or (g_grid != SVM_GAMMA_MULT_BASE)
    rows = []
    for axis, vals, grid in (("C", cs, c_grid), ("gamma multiplier", mult, g_grid)):
        lo, hi = grid[0], grid[-1]
        n_lo = int(np.isclose(vals, lo, rtol=1e-6).sum())
        n_hi = int(np.isclose(vals, hi, rtol=1e-6).sum())
        rows.append({"norm": norm, "axis": axis, "low_edge": lo, "folds_at_low_edge": n_lo, "high_edge": hi,
                     "folds_at_high_edge": n_hi,
                     "triggered": bool((not extended) and (n_lo > EDGE_FOLDS_MORE_THAN or n_hi > EDGE_FOLDS_MORE_THAN)),
                     "extension_already_applied": bool(extended)})
    return {"rows": rows}


def load_edge_rerun(root: Path, norm: str):
    """The edge-rule rerun of SVM-X for one normalization, or None when the rule was not run for it. A directory that
    exists but is incomplete is an error (a half-finished rerun must not be silently ignored)."""
    d = root / EDGE_RERUN_DIR.format(norm=norm)
    if not d.exists():
        return None
    return _subjectwise(d, f"SVM-X {norm} edge rerun")


def fit_time_totals(root: Path, norms=("per_subject", "global")) -> pd.DataFrame:
    rows = []
    for fam, pattern in FAMILY_DIRS.items():
        for norm in norms:
            d = root / pattern.format(norm=norm)
            if not d.exists():
                continue
            sw = _subjectwise(d, f"{fam} {norm}")
            if "fit_time_sec" not in sw.columns:
                raise InputError(f"{d.name}: no fit_time_sec column")
            rows.append({"family": fam, "norm": norm, "fit_time_hours": float(sw["fit_time_sec"].sum() / 3600.0)})
    return pd.DataFrame(rows)


def soft_vote_f1(cls_proba_dir: Path, cls_model: str, resnet_dir: Path) -> np.ndarray:
    """Per-subject macro F1 of the soft vote (mean of the two probability matrices) of one classical family and the
    published ResNet-SE+CD, from saved per-window probabilities. Row alignment is asserted."""
    from sklearn.metrics import f1_score
    out = []
    for s in range(1, N_SUBJECTS + 1):
        fc = cls_proba_dir / f"{cls_model}_sub{s:02d}.npz"
        fr = resnet_dir / f"RESNET_SE_sub{s:02d}.npz"
        for f in (fc, fr):
            if not f.exists():
                raise InputError(f"missing input: {f}")
        with np.load(fc) as zc, np.load(fr) as zr:
            if zc["proba"].shape != zr["proba"].shape or not np.array_equal(zc["y_true"], zr["y_true"]):
                raise InputError(f"subject {s}: {cls_model} and ResNet-SE+CD probabilities are not row-aligned")
            p = (zc["proba"] + zr["proba"]) / 2.0
            y = zc["y_true"]
        out.append(float(f1_score(y, p.argmax(1), average="macro", labels=[0, 1, 2, 3], zero_division=0)))
    return np.array(out)


def _fail_closed(out_dir: Path, reason: str) -> int:
    print(f"[C3] MISSING (fail closed): {reason}", file=sys.stderr)
    write_no_outcome_verdict(out_dir / "C3_VERDICT.md", "KC-C3 verdict", reason)
    (out_dir / "C3_tests.csv").unlink(missing_ok=True)
    return 20


def run(out_dir: Path, root: Path | None = None) -> int:
    """root: repo root the family/ensemble/resnet_cd directories live under (defaults to out_dir's own parent, since
    results_kc23_c3_ensemble is a sibling of results_kc23_c3_svm_per_subject etc., not a child of it)."""
    root = root or out_dir.parent
    try:
        resnet_dir = root / RESNET_CD_DIR
        if not resnet_dir.exists():
            raise InputError(f"{resnet_dir} not found (published ResNet-SE+CD reference)")
        resnet_cd = require_complete(read_subjectwise(resnet_dir), N_SUBJECTS, "resnet_cd")

        candidates = {fam: _subjectwise(root / pattern.format(norm="per_subject"), f"{fam} per_subject")["f1_macro"].to_numpy(float)
                      for fam, pattern in FAMILY_DIRS.items()}
        family_gains, skipped = {}, []
        for fam, pattern in FAMILY_DIRS.items():
            gd = root / pattern.format(norm="global")
            if not gd.exists():
                if fam in OPTIONAL_GLOBAL:
                    skipped.append(fam)
                    continue
                raise InputError(f"{gd} not found (family {fam!r}, global)")
            family_gains[fam] = (candidates[fam], _subjectwise(gd, f"{fam} global")["f1_macro"].to_numpy(float))

        edge_rows = []
        for norm in ("per_subject", "global"):
            edge_rows += svm_edge_hits(root / FAMILY_DIRS["svmx"].format(norm=norm), norm)["rows"]
        times = fit_time_totals(root)
        edge_rerun = {n: load_edge_rerun(root, n) for n in ("per_subject", "global")}
        edge_rerun_hits = {n: svm_edge_hits(root / EDGE_RERUN_DIR.format(norm=n), n)["rows"]
                           for n, v in edge_rerun.items() if v is not None}

        ens_p = out_dir / "ensemble_v2_subjectwise.csv"
        if not ens_p.exists():
            raise InputError(f"{ens_p} not found (run kc23_c3_merge_proba.py + ensemble_v2_combine.py first)")
        f1_ens = require_complete(load_ensemble_column(ens_p, ENSEMBLE_COLUMN), N_SUBJECTS, "ensemble_svmx")

        for fam, (model, pdir) in PROBA_FAMILIES.items():     # every candidate member must be complete, not just the winner
            for s in range(1, N_SUBJECTS + 1):
                if not (root / pdir / f"{model}_sub{s:02d}.npz").exists():
                    raise InputError(f"missing input: {root / pdir / f'{model}_sub{s:02d}.npz'}")
        means = {f: float(candidates[f].mean()) for f in PROBA_FAMILIES}
        best_member = max(means, key=means.get)
        model, pdir = PROBA_FAMILIES[best_member]
        f1_best_member = soft_vote_f1(root / pdir, model, root / RESNET_PROBA_DIR)
    except (InputError, FileNotFoundError, ValueError) as e:
        return _fail_closed(out_dir, str(e))

    best_fam = max(candidates, key=lambda k: candidates[k].mean())
    best_classical = candidates[best_fam]
    print(f"[C3] best tuned classical family: {best_fam} (mean F1 {best_classical.mean():.4f})")
    fired, rows_all = [], []

    p_letter, p_stats = classify_p(resnet_cd, best_classical)
    fired.append(p_letter)
    rows_all.append(p_stats)
    print_gate_header("KC-C3 Endpoint 1", p_letter, {
        "P1": "The 6.3pt lead stands.", "P2": "ESCALATE: the lead narrows, framing choice.",
        "P3": "ESCALATE: Finding C's strongest single model changes.",
        "P-OUT": "Outside the pre-registered grid: reported, not defaulted into a letter.",
    }[p_letter])

    n_letter, n_rows = classify_n(family_gains)
    fired.append(n_letter)
    rows_all.extend(n_rows)
    print_gate_header("KC-C3 Endpoint 2", n_letter, {
        "N1": "Every model tried extends to six classical families.",
        "N2": "ESCALATE: Finding A's scope wording changes.",
        "N-OUT": "Outside the pre-registered grid: reported, not defaulted into a letter.",
    }[n_letter])

    e_letter, e_stats = classify_e(f1_ens)
    fired.append(e_letter)
    rows_all.append(e_stats)
    print_gate_header("KC-C3 Endpoint 3", e_letter, {"E1": "Report.", "E2": "ESCALATE: touches the headline."}[e_letter])

    # ---- the edge-rule rerun: "report both runs". The letters above are the BASE run; when a rerun exists the same letters are
    # recomputed with the rerun SVM-X in its place, and either reading escalating escalates (as in C4).
    sens_fired, edge_run_md = [], ""
    if any(v is not None for v in edge_rerun.values()):
        ps2 = edge_rerun["per_subject"]["f1_macro"].to_numpy(float) if edge_rerun["per_subject"] is not None else candidates["svmx"]
        cands2 = {**candidates, "svmx": ps2}
        best2 = max(cands2, key=lambda k: cands2[k].mean())
        p2, _ = classify_p(resnet_cd, cands2[best2])
        gains2 = dict(family_gains)
        g2 = edge_rerun["global"]["f1_macro"].to_numpy(float) if edge_rerun["global"] is not None else family_gains["svmx"][1]
        gains2["svmx"] = (ps2, g2)
        n2, _ = classify_n(gains2)
        sens_fired = [p2, n2]
        base_means = {"per_subject": candidates["svmx"].mean(), "global": family_gains["svmx"][1].mean()}
        lines = []
        for n, v in edge_rerun.items():
            if v is None:
                lines.append(f"| {n} | {base_means[n]:.4f} | not run | | |")
                continue
            hits = edge_rerun_hits[n]
            lines.append(f"| {n} | {base_means[n]:.4f} | {v['f1_macro'].mean():.4f} | {(v['f1_macro'].mean() - base_means[n]) * 100:+.2f} | "
                         + "; ".join(f"{h['axis']}: {h['folds_at_low_edge']} low, {h['folds_at_high_edge']} high" for h in hits) + " |")
        differs = [f"{a} -> {b}" for a, b in ((p_letter, p2), (n_letter, n2)) if a != b]
        edge_run_md = ("\n\n## Edge-rule rerun (both runs reported)\n\n| norm | base SVM-X F1 | rerun SVM-X F1 | change (pt) | rerun edge hits |\n"
                       "|---|---|---|---|---|\n" + "\n".join(lines) + f"\n\nLetters recomputed with the rerun SVM-X in place of the base: "
                       f"Endpoint 1 {p2}, Endpoint 2 {n2}. "
                       + (f"They differ from the base letters ({'; '.join(differs)}): both readings are reported and either "
                          f"escalating escalates." if differs else "They match the base letters."))

    edge = pd.DataFrame(edge_rows)
    triggered = bool(edge["triggered"].any())
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_all).to_csv(out_dir / "C3_tests.csv", index=False)
    edge.to_csv(out_dir / "C3_edge_hits.csv", index=False)
    times.to_csv(out_dir / "C3_fit_times.csv", index=False)

    bm_delta = (float(f1_best_member.mean()) - PUBLISHED_ENSEMBLE) * 100.0
    edge_md = "| norm | axis | low edge | folds at low | high edge | folds at high | rule triggered |\n|---|---|---|---|---|---|---|\n" + \
        "\n".join(f"| {r.norm} | {r.axis} | {r.low_edge:g} | {r.folds_at_low_edge} | {r.high_edge:g} | {r.folds_at_high_edge} | "
                  f"{'yes' if r.triggered else ('extension already applied' if r.extension_already_applied else 'no')} |"
                  for r in edge.itertuples())
    time_md = "| family | norm | fit time (h) |\n|---|---|---|\n" + "\n".join(
        f"| {r.family} | {r.norm} | {r.fit_time_hours:.2f} |" for r in times.itertuples())
    notes = []
    if skipped:
        notes.append(f"Optional family global run absent and skipped: {skipped} (the plan marks RF-X global optional).")
    if triggered:
        notes.append("EDGE RULE TRIGGERED (more than 10 of 40 folds on a grid edge): extend that axis by two steps, once, "
                     "and rerun; report both runs (kc23_c3_edge_job_gen.py appends the rerun rows).")
    outcomes = ", ".join(fired)
    (out_dir / "C3_VERDICT.md").write_text(
        f"# KC-C3 verdict\n\n**Outcomes: {outcomes}**\n\n"
        f"Endpoint 1 {p_letter} (best tuned classical: {best_fam}, gain over the published SVM "
        f"{p_stats['gain_over_published_svm_pp']:+.2f} pt, ResNet-SE+CD lead {p_stats['delta_pp']:+.2f} pt); "
        f"Endpoint 2 {n_letter}; Endpoint 3 {e_letter} (soft vote with SVM-X {e_stats['mean_f1']:.4f} against "
        f"{PUBLISHED_ENSEMBLE:.3f}, {e_stats['delta_pp']:+.2f} pt).\n\n"
        f"Soft vote with the best new classical member ({best_member.upper()}, per-subject mean F1 {means[best_member]:.4f}) "
        f"and ResNet-SE+CD: {f1_best_member.mean():.4f}, {bm_delta:+.2f} pt against {PUBLISHED_ENSEMBLE:.3f} "
        f"(reported, no letter).\n\n## Edge-hit table (SVM-X)\n\n{edge_md}\n\n## Fit-time totals\n\n{time_md}\n\n"
        + edge_run_md + "\n\n" + ("\n\n".join(notes) + "\n" if notes else ""), encoding="utf-8")

    escalate = {"P2", "P3", "N2", "E2"}
    if any(l in escalate for l in fired + sens_fired):
        return 20
    if any(l.endswith("-OUT") for l in fired + sens_fired):
        return 10
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
