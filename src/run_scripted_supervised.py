#!/usr/bin/env python3
"""
src/run_scripted_supervised.py
=============================
KC-S1 runner. docs/plans/EXPERIMENT_PLAN_KC23_DEPLOYMENT.md "KC-S1. Scripted buffer:
label-free against supervised", S1.2 to S1.4. Reuses src/run_buffer_composition.py
(the balanced25 SVM/SVM_PROBA/RESNET_SE/soft fits and buffer machinery),
src/run_within_subject_baseline.py (the regime-C pooling protocol, unchanged) and
src/run_cnn_calibration_loso.py (the 3-epoch fine-tune schedule) by IMPORT, never
copied.

Protocol (S1.2, fixed): for each held-out subject, the scripted buffer B_K is
the first K windows of EACH movement's own recording (the balanced25
construction of src/run_buffer_composition.py, generalized here to K in
{5, 10, 25} via balanced_buffer_indices). Every arm normalizes the held-out
subject from B_K only (causal) and is scored on the same set -- all windows
not in B_25 -- so every K is scored on identical windows.

Arms (S1.3), all evaluated on the B_25-excluded windows:
  L0      label-free soft vote: SVM_PROBA + RESNET_SE+CD  (reproduces balanced25)
  L1      label-free ResNet-SE+CD alone
  L2      label-free SVM, probability route (== SVM_PROBA)
  S-pool  SVM on source + B_K labels: vstack(source transductive, B_K causal),
          run_within_subject_baseline.make_svm(bp["SVM"][h]["clf__C"]) refit
          (gamma="scale", unchanged -- that script's own regime-C construction,
          not "fixed" here)
  S-only  SVM on B_K alone (make_svm(), fixed C=1, regime-A style: too few
          labels to tune)
  S-ft    ResNet-SE+CD fine-tuned on B_K (run_cnn_calibration_loso.finetune,
          --ft-epochs 3 --ft-lr 5e-4), starting from THE SAME base model as L1
  S-ens1  soft vote of S-ft and S-pool
  S-ens2  soft vote of S-ft and L2

The SVM (decision route, no probability) is fit alongside SVM_PROBA purely to
feed the reproduction gate (S1.4) -- src/run_buffer_composition.py's own published
gate target is defined on the decision route, not the probability route.

Base training runs once per held-out subject per seed (--seed): SVM (decision
+ proba) via a single pair of SVC fits on the transductive-normalized source,
and ResNet-SE+CD via run_cnn_arch_loso.train_fold on the source, exactly
matching run_buffer_composition.stage_cnn's recipe (so L0 reproduces
balanced25). Every K's causal re-normalization and re-inference reuses that
one fit/training, per run_buffer_composition's own efficiency note.

Output: one shared results/kc23_s1_scripted_s<seed>/s1_subjectwise.csv per
seed (long format: subject, seed, K, arm, f1_macro, n_excl), resumable per
(subject, K) -- a subject's SVM/CNN base is retrained only if ANY of its K
rows are missing. src/kc23_s1_scripted_stats.py aggregates across the three
per-seed directories and computes the reproduction gate + S1.5 endpoint.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_buffer_composition import (
    load_common, norm_2d, norm_3d, is_buffer_mask, f1_incl_excl, LABELS, ROOT as BC_ROOT,
    META as BC_META, NPZ as BC_NPZ,
)
from run_streaming_norm_loso import per_subject_transductive
from run_within_subject_baseline import make_svm
from run_cnn_calibration_loso import finetune as cnn_finetune

K_LIST = [5, 10, 25]
B25_K = 25  # the fixed exclusion buffer for scoring, regardless of which K is being tested
PUBLISHED_SVM = 0.7472        # decision route, results/buffer_composition/buffer_composition_summary.csv (balanced25)
PUBLISHED_SVM_PROBA = 0.7286  # probability route, same source
PUBLISHED_SOFT = 0.8152       # soft ensemble, same source
GATE_TOL_EXACT = 5e-5   # "exact to 4 dp" -- both deterministic at fixed params
GATE_TOL_ENSEMBLE_DEFAULT = 0.015  # placeholder until KC-D1's own realization SD is measured


def balanced_buffer_indices(tvals_subject: np.ndarray, movement_subject: np.ndarray, K: int) -> np.ndarray:
    """The first K windows of EACH movement's own recording, by time order --
    the same construction as run_buffer_composition.buffer_indices("balanced25",
    ...), generalized from a fixed 25 to any K."""
    idx_out = []
    for mv in np.unique(movement_subject):
        m = np.where(movement_subject == mv)[0]
        order = np.argsort(tvals_subject[m], kind="stable")
        idx_out.append(m[order[:K]])
    return np.concatenate(idx_out) if idx_out else np.array([], dtype=int)


def reproduction_gate(svm_f1: float, svm_proba_f1: float, soft_f1: float,
                      ensemble_tol: float = GATE_TOL_ENSEMBLE_DEFAULT) -> tuple[str, dict]:
    svm_ok = abs(svm_f1 - PUBLISHED_SVM) <= GATE_TOL_EXACT
    proba_ok = abs(svm_proba_f1 - PUBLISHED_SVM_PROBA) <= GATE_TOL_EXACT
    soft_ok = abs(soft_f1 - PUBLISHED_SOFT) <= ensemble_tol
    letter = "PASS" if (svm_ok and proba_ok and soft_ok) else "FAIL"
    return letter, {"svm_ok": svm_ok, "svm_proba_ok": proba_ok, "soft_ok": soft_ok,
                    "svm_f1": svm_f1, "svm_proba_f1": svm_proba_f1, "soft_f1": soft_f1,
                    "published_svm": PUBLISHED_SVM, "published_svm_proba": PUBLISHED_SVM_PROBA,
                    "published_soft": PUBLISHED_SOFT}


def load_cnn_common():
    meta = pd.read_csv(BC_META)
    from train_cnn_loso import normalize_label_to_str
    data = np.load(BC_NPZ)
    X = data["X_env"].astype(np.float32)
    y = np.array([LABELS.index(s) for s in meta["movement"].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).to_numpy()
    return X, y, subjects


def fit_svm_subject(X, y, subjects, heldout, bp):
    from sklearn.svm import SVC
    te = (subjects == heldout); tr = ~te
    Xn_tr = per_subject_transductive(X, subjects, tr)
    Xtr, ytr = Xn_tr[tr], y[tr]
    C = bp[heldout]["clf__C"]; gamma = bp[heldout].get("clf__gamma", "scale")
    clf_plain = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", cache_size=500)
    clf_proba = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", cache_size=500,
                    probability=True, random_state=42)
    clf_plain.fit(Xtr, ytr)
    clf_proba.fit(Xtr, ytr)
    return clf_plain, clf_proba, Xn_tr, tr


def train_cnn_base(X3d, y, subjects, heldout, device, seed, hp):
    """hp: the base-training hyperparameters (an argparse Namespace or
    anything with .base_epochs/.base_patience/.base_batch/.base_lr/
    .base_chandrop_p attributes) -- explicit CLI args, not bare literals, so
    run_config.json actually records what a run used. Defaults match the
    published base training (epochs 40, patience 7, batch 512, chandrop 0.2),
    NOT src/run_cnn_calibration_loso.py's defaults (25/5), which V6 showed
    produces a materially different (81.8%) run if used by mistake."""
    from train_cnn_loso import per_subject_zscore_3d, choose_val_subjects
    from cnn_architectures import build_model
    from run_cnn_arch_loso import train_fold
    te = (subjects == heldout); tr = ~te
    in_ch = X3d.shape[1]
    Xtr_all = per_subject_zscore_3d(X3d[tr], subjects[tr])
    ytr_all, subtr = y[tr], subjects[tr]
    tr_subs, va_subs = choose_val_subjects(subtr, 0.15, seed + heldout)
    m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)
    model = build_model("resnet_se", in_ch, len(LABELS)).to(device)
    model = train_fold(model, Xtr_all[m_tr], ytr_all[m_tr], Xtr_all[m_va], ytr_all[m_va],
                       device, epochs=hp.base_epochs, batch=hp.base_batch, lr=hp.base_lr,
                       patience=hp.base_patience, seed=seed, aug_mode="chandrop", aug_sigma=0.1,
                       aug_chandrop_p=hp.base_chandrop_p, aug_timemask_frac=0.15)
    return model, in_ch


def proba_from_cnn(model, X3d_subset, y_subset, in_ch, device):
    from train_cnn_loso import WindowsDataset
    from run_cnn_arch_loso import evaluate_with_proba
    from torch.utils.data import DataLoader
    if len(y_subset) == 0:
        return np.zeros((0, len(LABELS)))
    dl = DataLoader(WindowsDataset(X3d_subset, y_subset), batch_size=512, shuffle=False)
    _, _, proba = evaluate_with_proba(model, dl, device)
    return proba


def f1_excl(y_true_excl, y_pred_excl):
    from sklearn.metrics import f1_score
    if len(y_true_excl) == 0:
        return float("nan")
    return float(f1_score(y_true_excl, y_pred_excl, average="macro", zero_division=0))


def run_subject(heldout, X_svm, y_svm, subjects_svm, tvals_svm, X_cnn, y_cnn, subjects_cnn,
                bp, device, seed, done_keys, hp):
    """Returns a list of row dicts (subject, seed, K, arm, f1_macro, n_excl) for
    every (K, arm) not already in done_keys."""
    needed = any((int(heldout), int(K)) not in done_keys for K in K_LIST)
    if not needed:
        return []

    clf_plain, clf_proba, Xn_tr_svm, tr_svm = fit_svm_subject(X_svm, y_svm, subjects_svm, heldout, bp)
    classes = clf_proba.classes_.astype(int)

    def proba_full_svm(raw):
        out = np.zeros((raw.shape[0], len(LABELS))); out[:, classes] = raw
        return out

    cnn_model, in_ch = train_cnn_base(X_cnn, y_cnn, subjects_cnn, heldout, device, seed, hp)

    te_svm = (subjects_svm == heldout)
    X_te_svm, y_te_svm, t_te_svm = X_svm[te_svm], y_svm[te_svm], tvals_svm[te_svm]
    te_cnn = (subjects_cnn == heldout)
    X_te_cnn, y_te_cnn = X_cnn[te_cnn], y_cnn[te_cnn]
    assert len(y_te_svm) == len(y_te_cnn) and np.array_equal(y_te_svm, y_te_cnn), \
        f"Sub{heldout}: SVM-feature and CNN-feature window sets/labels must be row-aligned"
    n_te = len(y_te_svm)

    b25 = balanced_buffer_indices(t_te_svm, y_te_svm, B25_K)
    excl = ~is_buffer_mask(b25, n_te)
    y_excl = y_te_svm[excl]

    rows = []
    for K in K_LIST:
        if (int(heldout), int(K)) in done_keys:
            continue
        bk = balanced_buffer_indices(t_te_svm, y_te_svm, K)

        Xte_n_svm = norm_2d(X_te_svm, bk)
        Xte_n_cnn = norm_3d(X_te_cnn, bk)

        pred_svm_decision = clf_plain.predict(Xte_n_svm[excl])
        proba_svm = proba_full_svm(clf_proba.predict_proba(Xte_n_svm[excl]))
        proba_cnn = proba_from_cnn(cnn_model, Xte_n_cnn[excl], y_excl, in_ch, device)

        f1_svm_decision = f1_excl(y_excl, pred_svm_decision)
        f1_svm_proba = f1_excl(y_excl, proba_svm.argmax(1))       # L2
        f1_resnet = f1_excl(y_excl, proba_cnn.argmax(1))          # L1
        proba_l0 = (proba_svm + proba_cnn) / 2.0
        f1_l0 = f1_excl(y_excl, proba_l0.argmax(1))               # L0

        # S-only: SVM fit on B_K alone (fixed C=1, regime-A style; too few
        # labels at small K to tune)
        Xa, ya = Xte_n_svm[bk], y_te_svm[bk]
        f1_s_only = float("nan")
        if len(np.unique(ya)) >= 2 and len(ya) >= len(LABELS):
            clf_only = make_svm()
            clf_only.fit(Xa, ya)
            f1_s_only = f1_excl(y_excl, clf_only.predict(Xte_n_svm[excl]))

        # S-pool: run_within_subject_baseline's regime-C protocol, unchanged --
        # vstack(source transductive, B_K causal), refit at the published C.
        proba_pool = None
        f1_s_pool = float("nan")
        if len(np.unique(ya)) >= 1 and len(ya) >= 1:
            from sklearn.svm import SVC
            Xtr_pool = np.vstack([Xn_tr_svm[tr_svm], Xa])
            ytr_pool = np.concatenate([y_svm[tr_svm], ya])
            C = bp[heldout]["clf__C"]
            clf_pool = SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced",
                          cache_size=500, probability=True, random_state=42)
            clf_pool.fit(Xtr_pool, ytr_pool)
            pool_classes = clf_pool.classes_.astype(int)
            f1_s_pool = f1_excl(y_excl, clf_pool.predict(Xte_n_svm[excl]))
            raw_pool = clf_pool.predict_proba(Xte_n_svm[excl])
            proba_pool = np.zeros((raw_pool.shape[0], len(LABELS))); proba_pool[:, pool_classes] = raw_pool
            del clf_pool

        # S-ft: fine-tune the SAME CNN base model used for L1, on B_K (causal-normalized).
        proba_ft = None
        f1_s_ft = float("nan")
        Xcal_cnn, ycal_cnn = Xte_n_cnn[bk], y_te_cnn[bk]
        if len(np.unique(ycal_cnn)) >= 1 and len(ycal_cnn) >= 1:
            ft_model = cnn_finetune(cnn_model, Xcal_cnn, ycal_cnn, in_ch, device,
                                    epochs=hp.ft_epochs, lr=hp.ft_lr, batch=min(512, max(1, len(ycal_cnn))))
            proba_ft = proba_from_cnn(ft_model, Xte_n_cnn[excl], y_excl, in_ch, device)
            f1_s_ft = f1_excl(y_excl, proba_ft.argmax(1))
            del ft_model

        f1_s_ens1 = (f1_excl(y_excl, ((proba_ft + proba_pool) / 2.0).argmax(1))
                    if proba_ft is not None and proba_pool is not None else float("nan"))
        f1_s_ens2 = (f1_excl(y_excl, ((proba_ft + proba_svm) / 2.0).argmax(1))
                    if proba_ft is not None else float("nan"))

        n_excl = int(excl.sum())
        for arm, f1 in [("SVM_decision", f1_svm_decision), ("SVM_PROBA", f1_svm_proba),
                       ("RESNET_SE", f1_resnet), ("L0", f1_l0), ("S-only", f1_s_only),
                       ("S-pool", f1_s_pool), ("S-ft", f1_s_ft), ("S-ens1", f1_s_ens1),
                       ("S-ens2", f1_s_ens2)]:
            rows.append({"subject": int(heldout), "seed": int(seed), "K": int(K), "arm": arm,
                        "f1_macro": round(float(f1), 6) if f1 == f1 else float("nan"), "n_excl": n_excl})
        print(f"  Sub{heldout:02d} K={K}: SVM={f1_svm_decision:.4f} SVM_PROBA={f1_svm_proba:.4f} "
              f"RESNET_SE={f1_resnet:.4f} L0={f1_l0:.4f} S-pool={f1_s_pool:.4f} S-ft={f1_s_ft:.4f} "
              f"S-ens1={f1_s_ens1:.4f} S-ens2={f1_s_ens2:.4f}", flush=True)

    del cnn_model
    gc.collect()
    import torch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def run(out_dir: Path, args) -> int:
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S1] seed={args.seed} device={device}", flush=True)

    from run_config_dump import dump_run_config
    dump_run_config(out_dir, args)

    csv_path = out_dir / "s1_subjectwise.csv"
    done_keys = set()
    if args.resume and csv_path.exists():
        prev = pd.read_csv(csv_path)
        prev = prev[prev["seed"] == args.seed]
        by_subject_k = prev.groupby(["subject", "K"])["arm"].nunique()
        done_keys = {tuple(k) for k, n in by_subject_k.items() if n >= 9}
        print(f"[resume] {len(done_keys)} (subject,K) pairs already complete for seed {args.seed}", flush=True)

    X_svm, y_svm, subjects_svm, tvals_svm = load_common()
    X_cnn, y_cnn, subjects_cnn = load_cnn_common()
    bp = {int(k): v for k, v in json.loads(open(BC_ROOT / "results/_bestparams.json").read())["SVM"].items()}

    subs_u = sorted(np.unique(subjects_svm).tolist())
    if args.subjects:
        want = {int(s) for s in args.subjects.split(",") if s.strip()}
        subs_u = [s for s in subs_u if s in want]
    for heldout in subs_u:
        t0 = time.time()
        rows = run_subject(heldout, X_svm, y_svm, subjects_svm, tvals_svm, X_cnn, y_cnn, subjects_cnn,
                           bp, device, args.seed, done_keys, args)
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            print(f"[fold] Sub{heldout:02d} done ({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"[resume] skip Sub{heldout:02d} (already complete)", flush=True)

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/kc23_s1_scripted")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--subjects", default=None,
                    help="comma-separated subject ids to restrict to (integration testing only)")
    # Base-training hyperparameters, explicit (not bare literals) so run_config.json
    # actually records what a run used. Defaults are the published base training,
    # NOT src/run_cnn_calibration_loso.py's defaults (epochs=25, patience=5), which V6
    # showed produces a materially different (81.8%) run if used by mistake.
    ap.add_argument("--base-epochs", type=int, default=40)
    ap.add_argument("--base-patience", type=int, default=7)
    ap.add_argument("--base-batch", type=int, default=512)
    ap.add_argument("--base-lr", type=float, default=1e-3)
    ap.add_argument("--base-chandrop-p", type=float, default=0.2)
    ap.add_argument("--ft-epochs", type=int, default=3)
    ap.add_argument("--ft-lr", type=float, default=5e-4)
    args = ap.parse_args()
    sys.exit(run(Path(args.out), args))


if __name__ == "__main__":
    main()
