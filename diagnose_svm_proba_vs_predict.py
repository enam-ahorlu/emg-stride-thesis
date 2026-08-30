#!/usr/bin/env python3
"""
diagnose_svm_proba_vs_predict.py
==================================
Follow-up diagnostic on E3 (causal ensemble). E3's honesty check validated
predict_proba().argmax(1) against the published SVM figure at the
TRANSDUCTIVE test-normalisation (0.7768 vs 0.7767, near-exact match). It did
NOT check whether that same probability-route survives at the CAUSAL
calib-100 test-normalisation, where the test distribution is shifted away
from what the internal 5-fold Platt-scaling CV saw.

For subjects 1-3 only: fit the SVM exactly as run_causal_ensemble.py's
stage_svm does (per_subject_transductive on training subjects, best_params
C from _bestparams.json, probability=True), then on the SAME fitted model
and the SAME calib-100-normalised test features, compare:
  (a) f1_macro of clf.predict(Xte)                   -- decision_function/OvO route
  (b) f1_macro of clf.predict_proba(Xte).argmax(1)    -- Platt + pairwise-coupling route
both buffer-included and buffer-excluded, plus the disagreement rate between
the two predicted-label vectors.

Expected (per the plan this diagnostic answers): (a) ~= published streaming
calib-100 figure (~0.748 buffer-excluded, decision-function SVM, no Platt
scaling involved at all); (b) ~= E3's causal-ensemble SVM-solo calib100
figure (0.7319, buffer-excluded, from results_causal_ensemble/calib100_subjectwise.csv
"SVM_excl"). If they do NOT split that way, the gap is a protocol difference
(e.g. causal normalisation itself, or a best_params/config mismatch) rather
than decision-vs-probability routing, and the E3 ensemble number needs
re-examination before it is trusted further.

Does NOT retrain the full 40-subject leg -- only subjects 1-3, calib100 only.

Output: results_causal_ensemble/svm_proba_vs_predict_diagnostic.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from run_causal_ensemble import load_common, buffer_mask, WARMUP, SEED, ROOT, OUT
from run_streaming_norm_loso import per_subject_transductive, normalise_test_subject

SUBJECTS = [1, 2, 3]
K = 100


def main():
    X, y, subjects, tvals = load_common()
    bp = json.loads(open(ROOT / "_bestparams.json").read())["SVM"]
    bp = {int(k): v for k, v in bp.items()}

    rows = []
    for heldout in SUBJECTS:
        te = (subjects == heldout); tr = ~te
        Xn_tr = per_subject_transductive(X, subjects, tr)
        Xtr, ytr = Xn_tr[tr], y[tr]
        yte = y[te]
        order = np.argsort(tvals[te], kind="stable")
        n_te = int(te.sum())

        params = bp[heldout]
        C = params["clf__C"]; gamma = params.get("clf__gamma", "scale")
        clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced",
                  probability=True, random_state=SEED, cache_size=500)
        clf.fit(Xtr, ytr)
        classes = clf.classes_.astype(int)

        Xte_c = normalise_test_subject(X[te], order, "calib", K, WARMUP)
        isb = buffer_mask(order, n_te, K)
        excl = ~isb

        # (a) decision_function / OvO voting route -- NOT affected by Platt scaling
        y_pred_decision = clf.predict(Xte_c)

        # (b) Platt-scaled probability + pairwise-coupling route
        proba_raw = clf.predict_proba(Xte_c)
        proba_full = np.zeros((proba_raw.shape[0], 4))
        proba_full[:, classes] = proba_raw
        y_pred_proba = proba_full.argmax(1)

        f1_decision_incl = f1_score(yte, y_pred_decision, average="macro", zero_division=0)
        f1_decision_excl = f1_score(yte[excl], y_pred_decision[excl], average="macro", zero_division=0)
        f1_proba_incl = f1_score(yte, y_pred_proba, average="macro", zero_division=0)
        f1_proba_excl = f1_score(yte[excl], y_pred_proba[excl], average="macro", zero_division=0)

        disagree_all = float((y_pred_decision != y_pred_proba).mean())
        disagree_excl = float((y_pred_decision[excl] != y_pred_proba[excl]).mean())

        rows.append(dict(
            subject=int(heldout), config=f"calib{K}", C=C, gamma=gamma, n_test=n_te,
            n_buffer_excluded=int(excl.sum()),
            f1_decision_route_incl=round(float(f1_decision_incl), 4),
            f1_decision_route_excl=round(float(f1_decision_excl), 4),
            f1_proba_route_incl=round(float(f1_proba_incl), 4),
            f1_proba_route_excl=round(float(f1_proba_excl), 4),
            disagreement_rate_incl=round(disagree_all, 4),
            disagreement_rate_excl=round(disagree_excl, 4),
        ))
        print(f"Sub{heldout:02d}: decision_excl={f1_decision_excl:.4f}  proba_excl={f1_proba_excl:.4f}  "
              f"disagree_excl={disagree_excl:.4f}", flush=True)

    df = pd.DataFrame(rows)
    mean_row = dict(subject="MEAN", config=f"calib{K}", C=np.nan, gamma="", n_test=df.n_test.mean(),
                    n_buffer_excluded=df.n_buffer_excluded.mean(),
                    f1_decision_route_incl=round(df.f1_decision_route_incl.mean(), 4),
                    f1_decision_route_excl=round(df.f1_decision_route_excl.mean(), 4),
                    f1_proba_route_incl=round(df.f1_proba_route_incl.mean(), 4),
                    f1_proba_route_excl=round(df.f1_proba_route_excl.mean(), 4),
                    disagreement_rate_incl=round(df.disagreement_rate_incl.mean(), 4),
                    disagreement_rate_excl=round(df.disagreement_rate_excl.mean(), 4))
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    out_path = OUT / "svm_proba_vs_predict_diagnostic.csv"
    df.to_csv(out_path, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n[save] {out_path}")

    mean_decision_excl = rows and np.mean([r["f1_decision_route_excl"] for r in rows])
    mean_proba_excl = rows and np.mean([r["f1_proba_route_excl"] for r in rows])
    print(f"\n[check] decision route mean (n=3) = {mean_decision_excl:.4f} "
          f"(expect ~0.748, published streaming calib-100 buffer-excluded)")
    print(f"[check] proba route mean    (n=3) = {mean_proba_excl:.4f} "
          f"(expect ~0.7319, E3 causal-ensemble SVM-solo calib100 buffer-excluded)")
    if mean_decision_excl > mean_proba_excl + 0.005:
        print("[verdict] decision route clearly beats proba route on these 3 subjects -- "
              "consistent with Platt-scaling/pairwise-coupling degradation under causal "
              "distribution shift. Proceed to check the full 40-subject pattern.")
    else:
        print("[verdict] decision route does NOT clearly beat proba route here -- "
              "the gap is likely NOT pairwise coupling. Stop and re-examine the causal "
              "protocol (normalisation, best_params, buffer definition) before trusting "
              "the E3 ensemble number further.")


if __name__ == "__main__":
    main()
