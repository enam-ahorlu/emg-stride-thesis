# rescore_streaming_buffer.py
# ---------------------------------------------------------------------------
# Airtight protocol check for Section 4.14 (causal / streaming normalisation).
#
# The streaming run scores F1 over the WHOLE test sequence, including the first
# K "calibration buffer" windows that were used to estimate the frozen calib-K
# normaliser. That is not label leakage (the buffer stats are unsupervised), but
# it is mildly optimistic versus a strict "calibrate on the first K, score only
# the remaining windows" protocol. This script re-scores every buffered config
# BOTH ways so the gap can be reported.
#
# Crucial efficiency fact: for a given held-out subject the fitted model does NOT
# depend on the test-norm config (training subjects are always transductively
# normalised). So we fit each (subject, model) ONCE and reuse it across all
# configs. f1_incl reproduces the published streaming number (a faithfulness
# check); f1_excl drops the first `buffer` windows (in acquisition-time order)
# from scoring only.
# ---------------------------------------------------------------------------
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from run_streaming_norm_loso import (
    per_subject_transductive, normalise_test_subject, build_search,
    TIME_COL_CANDIDATES,
)
from train_classical_loso import load_features_npz, encode_labels

FEAT = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
MODELS = ["SVM", "RF"]
# (mode, k, name) — buffer excluded = k for calib, warmup for running
CONFIGS = [("calib", 25, "calib25"), ("calib", 50, "calib50"),
           ("calib", 100, "calib100"), ("running", 0, "running")]
SEED = 42
WARMUP = 16
OUT = Path("results_loso_freq_streaming") / "streaming_buffer_rescore.csv"


def main():
    X = load_features_npz(Path(FEAT)).astype(np.float64)
    meta = pd.read_csv(META)
    label_col = next((c for c in ["movement", "label", "y_str", "status_mode", "y"] if c in meta.columns), None)
    subj_col = next((c for c in ["subject", "subject_id", "subject_int", "sid"] if c in meta.columns), None)
    y, _ = encode_labels(meta[label_col].astype(str).to_numpy())
    subjects = meta[subj_col].astype(int).to_numpy()
    time_col = next((c for c in TIME_COL_CANDIDATES if c in meta.columns), None)
    time_vals = meta[time_col].to_numpy() if time_col else np.arange(len(y))
    subjects_u = sorted(np.unique(subjects).tolist())

    done = set()
    if OUT.exists():
        d = pd.read_csv(OUT)
        done = set(zip(d["subject"].astype(int), d["model"]))
        print(f"[resume] {len(done)} (subject,model) rows already done")

    for heldout in subjects_u:
        te = (subjects == heldout); tr = ~te
        Xn = per_subject_transductive(X, subjects, tr)
        Xtr, ytr, gtr = Xn[tr], y[tr], subjects[tr]
        order = np.argsort(time_vals[te], kind="stable")
        yte_ord = y[te][order]
        inner_cv = GroupKFold(n_splits=min(5, len(np.unique(gtr))))
        for m in MODELS:
            if (heldout, m) in done:
                continue
            t0 = time.time()
            search = build_search(m, inner_cv, Xtr, ytr, gtr, SEED, n_jobs=1, rf_n_jobs=4)
            search.fit(Xtr, ytr)
            est = search.best_estimator_
            rows = []
            for (mode, k, name) in CONFIGS:
                Xte_norm = normalise_test_subject(X[te], order, mode, k, WARMUP)
                yhat_ord = est.predict(Xte_norm)[order]
                buf = k if mode == "calib" else WARMUP
                f1_incl = f1_score(yte_ord, yhat_ord, average="macro", zero_division=0)
                f1_excl = f1_score(yte_ord[buf:], yhat_ord[buf:], average="macro", zero_division=0)
                rows.append({"subject": int(heldout), "model": m, "config": name,
                             "buffer": buf, "n_test": int(te.sum()),
                             "f1_incl": round(float(f1_incl), 6),
                             "f1_excl": round(float(f1_excl), 6)})
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"[fold] Sub{heldout:02d} {m} done in {time.time()-t0:.1f}s :: " +
                  ", ".join(f"{r['config']} incl={r['f1_incl']:.3f}/excl={r['f1_excl']:.3f}" for r in rows), flush=True)

    # summary
    d = pd.read_csv(OUT).drop_duplicates(["subject", "model", "config"])
    g = (d.groupby(["config", "model"])[["f1_incl", "f1_excl"]].mean().round(4).reset_index())
    g["delta_pp"] = ((g["f1_excl"] - g["f1_incl"]) * 100).round(2)
    g.to_csv(Path("results_loso_freq_streaming") / "streaming_buffer_rescore_summary.csv", index=False)
    print("\n================  BUFFER-EXCLUDED RE-SCORE SUMMARY  ================")
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
