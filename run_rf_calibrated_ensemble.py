#!/usr/bin/env python3
"""
run_rf_calibrated_ensemble.py
==============================
EXPERIMENT_PLAN_CRITIQUE.md E5 (triage T15/T16) -- is RF dropped from the best
ensemble because it's genuinely weaker, or because its probabilities are poorly
calibrated and soft voting punishes miscalibration rather than weakness?

Step 1: quantify miscalibration of SVM/RF/ResNet-SE+CD from the existing
        headline probabilities (results_ensemble_v2/proba_aug_chandrop) --
        ECE (15 equal-mass bins), Brier score, mean max-confidence, pooled and
        per class. Reliability diagrams.
Step 2: recalibrate RF INSIDE the LOSO loop: CalibratedClassifierCV(RF, cv=
        GroupKFold(5) grouped by TRAINING subject), method in {isotonic,sigmoid}.
        Calibration never sees the held-out subject.
Step 3: re-vote (SVM + RF_calibrated + ResNet-SE+CD via ensemble_v2_combine.py)
        -- does it reach/beat SVM+RESNET_SE soft = 0.8579 (or stacking 0.8604)?

Output: results_rf_calibrated/{calibration_metrics.csv,proba_isotonic/,
        proba_sigmoid/,isotonic/,sigmoid/}
        report_figs/new_experiments/reliability_diagrams.png
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from train_classical_loso import load_features_npz, encode_labels
from run_streaming_norm_loso import per_subject_transductive

ROOT = Path(__file__).parent
OUT = ROOT / "results_rf_calibrated"; OUT.mkdir(exist_ok=True)
FIGDIR = ROOT / "report_figs" / "new_experiments"; FIGDIR.mkdir(parents=True, exist_ok=True)
FEAT = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
LABELS = ["DNS", "STDUP", "UPS", "WAK"]
SEED = 42
HEADLINE_PROBA = ROOT / "results_ensemble_v2" / "proba_aug_chandrop"
SUBS = list(range(1, 41))


# ============================================================================
# STEP 1 -- calibration metrics from existing proba
# ============================================================================
def ece_score(y_true_onehot, proba, n_bins=15):
    """Expected Calibration Error, equal-MASS (quantile) bins on max-confidence,
    following the plan's "15 equal-mass bins" spec (not equal-width)."""
    conf = proba.max(1)
    pred = proba.argmax(1)
    correct = (pred == y_true_onehot).astype(float)
    order = np.argsort(conf)
    conf_s, correct_s = conf[order], correct[order]
    n = len(conf_s)
    bins = np.array_split(np.arange(n), n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        acc_bin = correct_s[b].mean()
        conf_bin = conf_s[b].mean()
        ece += (len(b) / n) * abs(acc_bin - conf_bin)
    return float(ece)


def brier_score_multiclass(y_true, proba, n_classes):
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def load_model_proba(tag, proba_dir=HEADLINE_PROBA, subs=SUBS):
    yt_all, proba_all, subj_all = [], [], []
    for s in subs:
        f = proba_dir / f"{tag}_sub{s:02d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        yt_all.append(z["y_true"].astype(int))
        proba_all.append(z["proba"].astype(float))
        subj_all.append(np.full(len(z["y_true"]), s))
    return np.concatenate(yt_all), np.concatenate(proba_all), np.concatenate(subj_all)


def step1_calibration_metrics():
    rows = []
    reliability_data = {}
    for tag in ["SVM", "RF", "RESNET_SE"]:
        yt, proba, subj = load_model_proba(tag)
        ece = ece_score(yt, proba)
        brier = brier_score_multiclass(yt, proba, len(LABELS))
        mean_conf = float(proba.max(1).mean())
        acc = float((proba.argmax(1) == yt).mean())
        rows.append(dict(model=tag, scope="pooled", ece=round(ece, 4), brier=round(brier, 4),
                         mean_max_confidence=round(mean_conf, 4), accuracy=round(acc, 4), n=len(yt)))
        for c in range(len(LABELS)):
            m = yt == c
            if m.sum() == 0:
                continue
            rows.append(dict(model=tag, scope=LABELS[c], ece=round(ece_score(yt[m], proba[m]), 4),
                             brier=round(brier_score_multiclass(yt[m], proba[m], len(LABELS)), 4),
                             mean_max_confidence=round(float(proba[m].max(1).mean()), 4),
                             accuracy=round(float((proba[m].argmax(1) == yt[m]).mean()), 4), n=int(m.sum())))
        reliability_data[tag] = (yt, proba)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration_metrics.csv", index=False)
    print(df[df.scope == "pooled"].to_string(index=False))
    print(f"[save] {OUT / 'calibration_metrics.csv'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    for ax, tag in zip(axes, ["SVM", "RF", "RESNET_SE"]):
        yt, proba = reliability_data[tag]
        conf = proba.max(1); pred = proba.argmax(1); correct = (pred == yt).astype(float)
        order = np.argsort(conf)
        bins = np.array_split(order, 15)
        xs = [conf[b].mean() for b in bins if len(b)]
        ys = [correct[b].mean() for b in bins if len(b)]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
        ax.plot(xs, ys, "o-", color="#C44E52")
        pooled_ece = df[(df.model == tag) & (df.scope == "pooled")]["ece"].values[0]
        ax.set_title(f"{tag} (ECE={pooled_ece:.3f})")
        ax.set_xlabel("mean predicted confidence (bin)")
    axes[0].set_ylabel("empirical accuracy (bin)")
    fig.suptitle("Reliability diagrams, 15 equal-mass bins (headline proba_aug_chandrop)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "reliability_diagrams.png", dpi=150)
    plt.close(fig)
    print(f"[save] {FIGDIR / 'reliability_diagrams.png'}")


# ============================================================================
# STEP 2 -- recalibrate RF inside the LOSO loop
# ============================================================================
def step2_calibrate_rf(args):
    X = load_features_npz(FEAT).astype(np.float64)
    meta = pd.read_csv(META)
    y, _ = encode_labels(meta["movement"].astype(str).to_numpy())
    subjects = meta["subject"].astype(int).to_numpy()
    bp = json.loads(open(ROOT / "_bestparams.json").read())["RF"]
    bp = {int(k): v for k, v in bp.items()}

    for method in ["isotonic", "sigmoid"]:
        proba_dir = OUT / f"proba_{method}"; proba_dir.mkdir(exist_ok=True)
        for heldout in SUBS:
            out_npz = proba_dir / f"RF_sub{heldout:02d}.npz"
            if args.resume and out_npz.exists():
                continue
            te = (subjects == heldout); tr = ~te
            # per_subject_transductive(..., mask) only normalises rows where mask
            # is True -- passing `tr` alone leaves the held-out subject's rows
            # RAW (unnormalised), since the loop inside only iterates
            # np.unique(subjects[mask]). Must normalise ALL subjects (mask=all-True)
            # so the held-out subject gets its own transductive z-score too,
            # matching the headline per-subject pipeline (see run_within_subject_
            # baseline.py which does this correctly; this was a bug here that
            # silently fed the model raw-scale test features).
            Xn = per_subject_transductive(X, subjects, np.ones_like(tr, dtype=bool))
            Xtr, ytr, gtr = Xn[tr], y[tr], subjects[tr]
            Xte, yte = Xn[te], y[te]

            params = bp[heldout]
            base_rf = RandomForestClassifier(n_estimators=params["clf__n_estimators"],
                                             max_depth=params["clf__max_depth"],
                                             class_weight="balanced", random_state=SEED,
                                             n_jobs=args.rf_n_jobs)
            n_groups = len(np.unique(gtr))
            cv = GroupKFold(n_splits=min(5, n_groups))
            splits = list(cv.split(Xtr, ytr, groups=gtr))
            calib = CalibratedClassifierCV(base_rf, method=method, cv=splits)
            calib.fit(Xtr, ytr)

            classes = calib.classes_.astype(int)
            raw = calib.predict_proba(Xte)
            proba_full = np.zeros((raw.shape[0], len(LABELS)))
            proba_full[:, classes] = raw
            np.savez(out_npz, proba=proba_full, y_true=yte.astype(np.int32))
            f1 = f1_score(yte, proba_full.argmax(1), average="macro", zero_division=0)
            print(f"[{method}] Sub{heldout:02d}: f1={f1:.4f}", flush=True)

        # copy unchanged SVM/RESNET_SE headline proba so ensemble_v2_combine.py sees a complete set
        for tag in ["SVM", "RESNET_SE"]:
            for s in SUBS:
                src = HEADLINE_PROBA / f"{tag}_sub{s:02d}.npz"
                dst = proba_dir / f"{tag}_sub{s:02d}.npz"
                if src.exists() and not dst.exists():
                    shutil.copyfile(src, dst)
        print(f"[done] {method} -> {proba_dir}")


# ============================================================================
# STEP 3 -- re-vote
# ============================================================================
def step3_combine():
    for method in ["isotonic", "sigmoid"]:
        proba_dir = OUT / f"proba_{method}"
        out_dir = OUT / method
        if not proba_dir.exists():
            print(f"[skip] {proba_dir} missing"); continue
        subprocess.run([sys.executable, "ensemble_v2_combine.py",
                        "--proba-dir", str(proba_dir.relative_to(ROOT)),
                        "--out", str(out_dir.relative_to(ROOT)),
                        "--ref", "SVM+RESNET_SE [soft]"], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["metrics", "calibrate", "combine"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rf-n-jobs", type=int, default=1)
    args = ap.parse_args()
    if args.stage == "metrics":
        step1_calibration_metrics()
    elif args.stage == "calibrate":
        step2_calibrate_rf(args)
    elif args.stage == "combine":
        step3_combine()


if __name__ == "__main__":
    main()
