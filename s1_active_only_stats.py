#!/usr/bin/env python3
"""
s1_active_only_stats.py
=======================
S-1 of EXPERIMENT_PLAN_PREPROC_AUDIT.md. Active-only STDUP re-run: does the
published class hierarchy STDUP >> UPS > WAK > DNS survive when STDUP is
restricted to active windows (support 1982, 14.5%) instead of 86.5% rest?

Reads the four classical LOSO runs written by run_s1_classical.sh:
  results_aonly_persubj/   SVM, RF, per-subject norm
  results_aonly_global/    SVM, RF, global norm
plus the deep arm if present (results_aonly_resnet_se_cd_persubj/).

Reports:
  1. per-class F1 (SVM, RF, per-subject, active-only) beside the published
     Freq-72 LOSO per-class F1, with the resulting hierarchy stated.
  2. macro-F1 for all four runs and the per-subject minus global delta per model
     (paired Wilcoxon, BCa 95%, paired Cohen's d).
  3. per-subject class counts (from the active-only meta) and any cell < 30.

Grid C / H / M in the plan. INDEPENDENTLY of C/H/M, the normalization delta is
reported; if per-subject no longer beats global by a similar margin to the
published +6.9 pp (SVM) / +5.1 pp (RF), STOP AND ESCALATE.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from window_ablation_stats import bca_ci, cohens_d_paired  # noqa: E402

OUT = ROOT / "results_locus"
OUT.mkdir(exist_ok=True)
LABELS = ["DNS", "STDUP", "UPS", "WAK"]  # alphabetical == y_int order
META_AONLY = ROOT / "windows_WAK_UPS_DNS_STDUP_v1_meta_w250_ov50_conf60_Aonly.csv"

# published Freq-72 LOSO per-subject per-class F1 (report_figs/freq72_error_analysis/)
PUB_PERCLASS = {
    "SVM": {"DNS": 0.677, "STDUP": 0.961, "UPS": 0.763, "WAK": 0.702},
    "RF":  {"DNS": 0.662, "STDUP": 0.960, "UPS": 0.754, "WAK": 0.709},
}
PUB_MACRO = {"SVM": {"persubj": 0.777, "global": 0.708},
             "RF":  {"persubj": 0.773, "global": 0.722}}
PUB_NORM_DELTA = {"SVM": 0.777 - 0.708, "RF": 0.773 - 0.722}  # +6.9 pp, +5.1 pp
ESCALATE_TOL_PP = 3.0


def sw_path(d: str, model: str) -> Path | None:
    hits = sorted((ROOT / d).glob(f"*{model}*subjectwise*.csv"))
    return hits[0] if hits else None


def load_macro(d: str, model: str) -> pd.Series | None:
    p = sw_path(d, model)
    if not p:
        return None
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    col = "f1_macro" if "f1_macro" in df.columns else "f1"
    return df.set_index(sc)[col].sort_index()


def load_perclass(d: str, model: str) -> dict | None:
    """Pooled per-class F1 from the per-subject fold predictions.

    --save-preds writes a full-LOSO predictions/ dir only when every fold ran in one
    process; a run resumed from checkpoints skips it. The per-fold files under
    predictions_folds/ (..._{MODEL}_subNN_y_{pred,true}.npy) are always written with
    --flush-preds, so concatenate those in subject order.
    """
    for sub in ("predictions", "predictions_folds"):
        base = ROOT / d / sub
        pps = sorted(base.glob(f"*_{model}_sub*_y_pred.npy"))
        pts = sorted(base.glob(f"*_{model}_sub*_y_true.npy"))
        if pps and pts and len(pps) == len(pts):
            y_pred = np.concatenate([np.load(f) for f in pps])
            y_true = np.concatenate([np.load(f) for f in pts])
            f1s = f1_score(y_true, y_pred, average=None, labels=list(range(len(LABELS))))
            return {LABELS[i]: float(f1s[i]) for i in range(len(LABELS))}
    return None


def hierarchy(perclass: dict) -> list[str]:
    return [k for k, _ in sorted(perclass.items(), key=lambda kv: -kv[1])]


def main() -> int:
    print("=" * 78)
    print("S-1: active-only STDUP re-run")
    print("=" * 78)

    # ---- per-subject class counts ----
    if META_AONLY.exists():
        m = pd.read_csv(META_AONLY)
        ct = m.groupby("movement").size()
        pc = m.groupby(["subject", "movement"]).size().unstack(fill_value=0)
        low = pc.where(pc < 30).stack().dropna()
        print(f"\nactive-only windows: {len(m)} total  {ct.to_dict()}  STDUP {100*ct.get('STDUP',0)/len(m):.1f}%")
        print(f"subject-by-class cells < 30 windows: {len(low)}"
              + (f"  {dict(low)}" if len(low) else " (none)"))
    else:
        print("  (active-only meta not found; class-count check skipped)")

    # ---- per-class F1, active-only, per-subject norm ----
    print("\n--- per-class F1: active-only (per-subject norm) vs published Freq-72 LOSO ---")
    perclass_new = {}
    for model in ("SVM", "RF"):
        pcnew = load_perclass("results_aonly_persubj", model)
        if pcnew is None:
            print(f"  {model}: no predictions yet in results_aonly_persubj/")
            continue
        perclass_new[model] = pcnew
        print(f"  {model}:")
        print(f"    {'class':7} {'published F1':>12} {'active-only F1':>14}  {'delta':>8}")
        for c in LABELS:
            print(f"    {c:7} {PUB_PERCLASS[model][c]:>12.3f} {pcnew[c]:>14.3f}  {pcnew[c]-PUB_PERCLASS[model][c]:>+8.3f}")
        print(f"    published hierarchy: {' > '.join(hierarchy(PUB_PERCLASS[model]))}")
        print(f"    active-only hierarchy: {' > '.join(hierarchy(pcnew))}")

    # ---- macro-F1 and normalization delta ----
    print("\n--- macro-F1 and per-subject minus global delta ---")
    norm_rows = []
    escalate = False
    for model in ("SVM", "RF"):
        ps = load_macro("results_aonly_persubj", model)
        gl = load_macro("results_aonly_global", model)
        if ps is None or gl is None:
            print(f"  {model}: runs not both present yet")
            continue
        ix = ps.index.intersection(gl.index)
        a, b = ps.loc[ix].to_numpy(), gl.loc[ix].to_numpy()
        d = a - b
        lo, hi = bca_ci(d)
        w = stats.wilcoxon(a, b)
        dz = cohens_d_paired(a, b)
        pub_d = PUB_NORM_DELTA[model] * 100
        drift = d.mean() * 100 - pub_d
        flag = abs(drift) > ESCALATE_TOL_PP or d.mean() <= 0 or w.pvalue >= 0.05
        escalate |= flag
        print(f"  {model}: per-subject {a.mean():.4f}  global {b.mean():.4f}  "
              f"delta {d.mean()*100:+.2f} pp  95% BCa [{lo*100:+.2f}, {hi*100:+.2f}]  "
              f"p = {w.pvalue:.4g}  d = {dz:+.2f}   (published delta {pub_d:+.2f} pp; drift {drift:+.2f} pp) "
              f"{'  << ESCALATE' if flag else ''}")
        norm_rows.append({"model": model, "persubj_mean": float(a.mean()), "global_mean": float(b.mean()),
                          "delta_pp": float(d.mean() * 100), "bca_lo_pp": lo * 100, "bca_hi_pp": hi * 100,
                          "wilcoxon_p": float(w.pvalue), "cohens_d": dz,
                          "published_delta_pp": pub_d, "drift_pp": drift, "escalate": bool(flag)})

    # ---- deep arm, if present ----
    deep = load_macro("results_aonly_resnet_se_cd_persubj", "resnet_se") or \
        load_macro("results_aonly_resnet_se_cd_persubj", "")
    deep_perclass = None
    proba_dir = ROOT / "results_aonly_resnet_se_cd_persubj" / "proba"
    pf = sorted(proba_dir.glob("*sub*.npz")) if proba_dir.exists() else []
    if pf:
        yp_all, yt_all = [], []
        for f in pf:
            z = np.load(f)
            yp_all.append(z["proba"].argmax(1))
            yt_all.append(z["y_true"])
        yp_all = np.concatenate(yp_all)
        yt_all = np.concatenate(yt_all)
        f1s = f1_score(yt_all, yp_all, average=None, labels=list(range(len(LABELS))))
        deep_perclass = {LABELS[i]: float(f1s[i]) for i in range(len(LABELS))}
    if deep is not None:
        print(f"\n  deep arm (resnet_se + CD, per-subject, active-only): mean macro-F1 {deep.mean():.4f}  (n={len(deep)})")
    if deep_perclass is not None:
        print(f"    per-class F1 (pooled over {len(pf)} subjects): "
              + "  ".join(f"{c} {deep_perclass[c]:.3f}" for c in LABELS))
        print(f"    deep-arm hierarchy: {' > '.join(hierarchy(deep_perclass))}"
              + ("   [STDUP stays top -> consistent with H]" if hierarchy(deep_perclass)[0] == "STDUP"
                 else "   [STDUP NOT top -> deep arm diverges from classical H]"))

    # ---- grid C / H / M ----
    verdict = None
    if len(perclass_new) == 2:
        tops = {model: hierarchy(pc)[0] for model, pc in perclass_new.items()}
        if all(t == "STDUP" for t in tops.values()):
            verdict = "H"
            meaning = ("Hierarchy holds. STDUP stays top even restricted to active windows, so the "
                       "biomechanical reading was right after all and the current section 4.3 / 5.3 "
                       "correction is over-stated. Report as prominently as outcome C would be.")
        elif all(hierarchy(pc)[-1] == "STDUP" or hierarchy(pc)[0] != "STDUP" for pc in perclass_new.values()) \
                and all(t != "STDUP" for t in tops.values()):
            # STDUP not top for either model
            not_bottom = any(hierarchy(pc)[-1] != "STDUP" for pc in perclass_new.values())
            verdict = "M" if not_bottom else "C"
            if verdict == "C":
                meaning = ("Composition. STDUP is no longer top of the hierarchy under the "
                           "active-only definition (expected given the AUC 0.47). The section 4.3 / "
                           "5.3 wording stands; Chapter 4 gains a short subsection reporting both "
                           "hierarchies.")
            else:
                meaning = ("Mixed. STDUP falls but stays above at least one of UPS / WAK / DNS. "
                           "Report the ordering as measured and make no causal claim beyond it.")
        else:
            verdict = "M"
            meaning = ("Mixed. The two models disagree on where STDUP lands, or it is neither "
                       "clearly top nor clearly bottom. Report the orderings as measured.")
        print("\n" + "=" * 78)
        print(f"S-1 GRID OUTCOME {verdict}: {meaning}")
        print("=" * 78)
    else:
        print("\n(grid deferred: per-class F1 not available for both models yet)")

    if escalate:
        print("\n*** NORMALIZATION-DELTA ESCALATION ***")
        print("The per-subject vs global margin under active-only STDUP has moved from the "
              "published +6.9 pp (SVM) / +5.1 pp (RF) by more than the 3.0 pp tolerance, or is no "
              "longer positive/significant. Per the plan this means STOP AND ESCALATE: it would "
              "imply Finding A is partly carried by the rest windows. Reported; not resolved.")
    else:
        print("\nNormalization delta: per-subject still beats global by a margin close to the "
              "published one. Finding A is NOT class-definition-dependent. No escalation.")

    res = {"stage": "S-1", "grid_outcome": verdict, "grid_meaning": (meaning if verdict else None),
           "per_class_f1_active_only": perclass_new,
           "published_per_class_f1": PUB_PERCLASS,
           "normalization_delta": norm_rows,
           "escalate_normalization_delta": bool(escalate),
           "deep_arm_mean_f1": (float(deep.mean()) if deep is not None else None),
           "deep_arm_n": (int(len(deep)) if deep is not None else None),
           "deep_arm_per_class_f1": deep_perclass,
           "deep_arm_hierarchy": (hierarchy(deep_perclass) if deep_perclass is not None else None),
           "deep_arm_consistent_with_H": (bool(hierarchy(deep_perclass)[0] == "STDUP")
                                          if deep_perclass is not None else None),
           "class_counts_ok": (len(low) == 0 if META_AONLY.exists() else None),
           "thesis_files_edited": False, "section_4_17_touched": False}
    json.dump(res, open(OUT / "s1_outcome.json", "w"), indent=2)
    print(f"\nwrote {OUT/'s1_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
