# rescore_streaming_buffer_v2.py
# ---------------------------------------------------------------------------
# Faithful FAST version of rescore_streaming_buffer.py. Instead of re-running
# GridSearchCV per subject, it fits each subject's model directly with the
# hyperparameters GridSearchCV already selected in the authoritative per-subject
# LOSO run (results_loso_freq_persubj/*best_params). Because GridSearchCV refits
# best_estimator_ on the full training set, SVC(C=best) / RF(best) fit on the
# same training data reproduce exactly the same model — so f1_incl still matches
# the published streaming numbers (validated below on Sub01/Sub02). ~15x faster
# for the SVM (1 fit vs 15).
# ---------------------------------------------------------------------------
from __future__ import annotations
import time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from run_streaming_norm_loso import per_subject_transductive, normalise_test_subject, TIME_COL_CANDIDATES
from train_classical_loso import load_features_npz, encode_labels

FEAT = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META = "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
CONFIGS = [("calib", 25, "calib25"), ("calib", 50, "calib50"),
           ("calib", 100, "calib100"), ("running", 0, "running")]
SEED, WARMUP = 42, 16
OUT = Path("results_loso_freq_streaming") / "streaming_buffer_rescore_v2.csv"

# known incl values from the faithful GridSearch run (for validation)
CHECK = {(1, "SVM"): {"calib25": 0.575, "calib50": 0.644, "calib100": 0.657, "running": 0.644},
         (2, "SVM"): {"calib25": 0.766, "calib50": 0.763, "calib100": 0.752, "running": 0.768}}


def fit_model(m, params, Xtr, ytr):
    if m == "SVM":
        est = SVC(kernel="rbf", class_weight="balanced", cache_size=500,
                  C=params["clf__C"], gamma=params.get("clf__gamma", "scale"))
    else:
        est = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=4,
                                     n_estimators=params["clf__n_estimators"],
                                     max_depth=params["clf__max_depth"])
    est.fit(Xtr, ytr)
    return est


def main():
    bp = json.loads(open("_bestparams.json").read())
    bp = {m: {int(k): v for k, v in d.items()} for m, d in bp.items()}
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
        d = pd.read_csv(OUT); done = set(zip(d["subject"].astype(int), d["model"]))

    for heldout in subjects_u:
        te = (subjects == heldout); tr = ~te
        Xn = per_subject_transductive(X, subjects, tr)
        Xtr, ytr = Xn[tr], y[tr]
        order = np.argsort(time_vals[te], kind="stable")
        yte_ord = y[te][order]
        for m in ["SVM", "RF"]:
            if (heldout, m) in done:
                continue
            t0 = time.time()
            est = fit_model(m, bp[m][heldout], Xtr, ytr)
            rows = []
            for (mode, k, name) in CONFIGS:
                Xte_norm = normalise_test_subject(X[te], order, mode, k, WARMUP)
                yhat_ord = est.predict(Xte_norm)[order]
                buf = k if mode == "calib" else WARMUP
                f1_incl = f1_score(yte_ord, yhat_ord, average="macro", zero_division=0)
                f1_excl = f1_score(yte_ord[buf:], yhat_ord[buf:], average="macro", zero_division=0)
                rows.append({"subject": int(heldout), "model": m, "config": name, "buffer": buf,
                             "n_test": int(te.sum()), "f1_incl": round(float(f1_incl), 6),
                             "f1_excl": round(float(f1_excl), 6)})
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            chk = ""
            if (heldout, m) in CHECK:
                exp = CHECK[(heldout, m)]
                diffs = {r["config"]: round(r["f1_incl"] - exp[r["config"]], 3) for r in rows}
                chk = " VALIDATE(incl-expected)=" + str(diffs)
            print(f"[fold] Sub{heldout:02d} {m} {time.time()-t0:.1f}s :: " +
                  ", ".join(f"{r['config']} incl={r['f1_incl']:.3f}/excl={r['f1_excl']:.3f}" for r in rows) + chk, flush=True)

    d = pd.read_csv(OUT).drop_duplicates(["subject", "model", "config"])
    g = d.groupby(["config", "model"])[["f1_incl", "f1_excl"]].mean().round(4).reset_index()
    g["delta_pp"] = ((g["f1_excl"] - g["f1_incl"]) * 100).round(2)
    g.to_csv(Path("results_loso_freq_streaming") / "streaming_buffer_rescore_summary.csv", index=False)
    print("\n===== BUFFER-EXCLUDED RE-SCORE SUMMARY (mean over 40 subjects) =====")
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
