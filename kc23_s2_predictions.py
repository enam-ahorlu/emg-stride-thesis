#!/usr/bin/env python3
"""
kc23_s2_predictions.py
=========================
KC-S2 S2.3 producing job. EXPERIMENT_PLAN_KC23_DEPLOYMENT.md: "Per-window
LOSO predictions, with circuit and time, for the locked SVM, the locked
ResNet-SE+CD and the locked soft vote on ENABL3S, under: transductive
per-subject normalization; the causal 100-window buffer. Also score the
scripted balanced25 buffer, for parity with KC-S1."

"Locked" here means the SAME architecture, augmentation and fitting PROCEDURE
as the main SIAT pipeline's R2/D5-e2 arm (resnet_se, chandrop 0.2) and the
default (non-extended) SVM grid {C: [1,5,10], gamma: [scale]} -- NOT SIAT's
per-subject best_params (ENABL3S subjects were never part of that tuning).
Each subject's SVM and CNN are each fit ONCE (fresh, LOSO over ENABL3S's own
9 source subjects, matching D5's own precedent of training fresh on ENABL3S
rather than transferring a frozen SIAT model), then re-normalized and
re-inferred 3x (transductive, causal-100, causal-balanced25), reusing
run_buffer_composition.py's norm_2d/norm_3d/buffer machinery -- the same
efficiency pattern it already uses for SIAT.

Row alignment: results_kc23_s2_adapter_circuitmeta's --with-circuit-meta
windows are BYTE-IDENTICAL in count and order to the published (non-circuit-
meta) ENABL3S windows this adapter tag produces (confirmed 2026-09-24: same
45525 rows, same subject sequence) -- so the published Freq-72 features
(features_out_ext/...features_ext.npz) can be used directly for SVM, aligned
row-for-row with the circuit-meta meta CSV's circuit/t_start_circuit columns,
without re-extracting features from the circuit-meta npz.

Output: subject, circuit, t_end, y_true, y_pred, mode_raw, fs -- one CSV per
(model, condition), the exact schema kc23_s2_transitions.py already expects.
mode_raw is the raw Mode code majority-voted per window (LW/SA/SD=1/4/5;
STDUP windows, which have no single raw-mode equivalent, are marked -1).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CIRCUITMETA_DIR = "results_kc23_s2_adapter_circuitmeta"
CIRCUITMETA_TAG = "ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_kc23s2"
FEAT_ENABL3S = "features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_ext.npz"
META_ENABL3S_FEAT = "features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_meta.csv"
LABELS = ["DNS", "STDUP", "UPS", "WAK"]
MOVEMENT_TO_MODE_RAW = {"WAK": 1, "UPS": 4, "DNS": 5, "STDUP": -1}  # MODE_LW/SA/SD; STDUP has no raw-mode equivalent
CONDITIONS = ["transductive", "causal100", "causal_balanced25"]


# The S2b (400 ms) run is this same pipeline on the 400 ms circuit-meta windows: every path below has the 250 ms value as its
# default, so a call without SPEC is byte-for-byte the S2.3 job.
SPEC_250 = {"circuitmeta_dir": CIRCUITMETA_DIR, "tag": CIRCUITMETA_TAG, "feat": FEAT_ENABL3S, "meta_feat": META_ENABL3S_FEAT}
SPEC_400 = {"circuitmeta_dir": "results_kc23_s2b_adapter_400", "tag": "ENABL3S_WAK_UPS_DNS_STDUP_w400_ov50_conf60_kc23s2b",
            "feat": "results_kc23_s2b_features/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w400_ov50_conf60_kc23s2b_features_ext.npz",
            "meta_feat": "results_kc23_s2b_features/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w400_ov50_conf60_kc23s2b_features_meta.csv"}


def load_common(root: Path, spec: dict | None = None):
    spec = spec or SPEC_250
    from train_classical_loso import load_features_npz, encode_labels
    X_feat = load_features_npz(root / spec["feat"]).astype(np.float64)
    meta_feat = pd.read_csv(root / spec["meta_feat"])
    meta_cm = pd.read_csv(root / spec["circuitmeta_dir"] / f"windows_{spec['tag']}_meta.csv")
    if len(meta_feat) != len(meta_cm) or not (meta_feat["subject"].to_numpy() == meta_cm["subject"].to_numpy()).all():
        raise ValueError("features_ext and circuit-meta window sets are not row-aligned -- "
                         "re-check both were built with the same adapter tag/params")
    y, _ = encode_labels(meta_feat["movement"].astype(str).to_numpy())
    subjects = meta_cm["subject"].astype(int).to_numpy()
    circuits = meta_cm["circuit"].to_numpy()
    t_end_s = (meta_cm["t_start_circuit"].to_numpy() + meta_cm["win_samples"].to_numpy()) / meta_cm["fs"].to_numpy()
    fs = meta_cm["fs"].to_numpy()
    mode_raw = meta_cm["movement"].map(MOVEMENT_TO_MODE_RAW).to_numpy()
    npz = np.load(root / spec["circuitmeta_dir"] / f"windows_{spec['tag']}.npz")
    X_env = npz["X_env"].astype(np.float32)
    return X_feat, X_env, y, subjects, circuits, t_end_s, fs, mode_raw


def buffer_positions(condition: str, subject_positions: np.ndarray, subject_circuits: np.ndarray,
                     subject_t_end: np.ndarray, subject_movement_code: np.ndarray) -> np.ndarray | None:
    """Positions (into the held-out subject's own window array) forming the
    buffer for one condition. None for "transductive" (no buffer -- the whole
    session normalizes itself)."""
    if condition == "transductive":
        return None
    if condition == "causal100":
        order = np.argsort(subject_t_end, kind="stable")
        return order[:100]
    if condition == "causal_balanced25":
        from run_scripted_supervised import balanced_buffer_indices
        return balanced_buffer_indices(subject_t_end, subject_movement_code, K=25)
    raise ValueError(condition)


def run_subject(heldout, X_feat, X_env, y, subjects, circuits, t_end_s, fs, mode_raw, bp_svm_grid, device):
    from sklearn.svm import SVC
    from run_streaming_norm_loso import per_subject_transductive
    from run_buffer_composition import norm_2d, norm_3d
    from train_cnn_loso import per_subject_zscore_3d, choose_val_subjects
    from cnn_architectures import build_model
    from run_cnn_arch_loso import train_fold, evaluate_with_proba
    from torch.utils.data import DataLoader
    from train_cnn_loso import WindowsDataset
    from sklearn.model_selection import GroupKFold, GridSearchCV

    te = (subjects == heldout); tr = ~te
    n_te = int(te.sum())

    # ---- SVM: fixed (non-extended) grid, fresh per fold, transductive-normalized source ----
    Xn_tr_svm = per_subject_transductive(X_feat, subjects, tr)
    inner_cv = GroupKFold(n_splits=min(5, len(np.unique(subjects[tr]))))
    search = GridSearchCV(SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42,
                             cache_size=500),
                          {"C": [1, 5, 10], "gamma": ["scale"]}, scoring="f1_macro",
                          cv=list(inner_cv.split(Xn_tr_svm[tr], y[tr], groups=subjects[tr])), n_jobs=1)
    search.fit(Xn_tr_svm[tr], y[tr])
    svm = search.best_estimator_
    svm_classes = svm.classes_.astype(int)

    # ---- CNN: resnet_se, chandrop 0.2, fresh per fold (matches D5-e2 exactly) ----
    in_ch = X_env.shape[1]
    Xtr_all = per_subject_zscore_3d(X_env[tr], subjects[tr])
    tr_subs, va_subs = choose_val_subjects(subjects[tr], 0.15, 42 + heldout)
    m_tr, m_va = np.isin(subjects[tr], tr_subs), np.isin(subjects[tr], va_subs)
    cnn = build_model("resnet_se", in_ch, len(LABELS)).to(device)
    cnn = train_fold(cnn, Xtr_all[m_tr], y[tr][m_tr], Xtr_all[m_va], y[tr][m_va], device,
                     epochs=40, batch=512, lr=1e-3, patience=7, seed=42,
                     aug_mode="chandrop", aug_sigma=0.1, aug_chandrop_p=0.2, aug_timemask_frac=0.15)

    X_te_feat, X_te_env = X_feat[te], X_env[te]
    y_te, circ_te, t_end_te, fs_te, mode_te = y[te], circuits[te], t_end_s[te], fs[te], mode_raw[te]

    rows_by_cond = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        buf = buffer_positions(condition, np.arange(n_te), circ_te, t_end_te, mode_te)
        if buf is None:
            Xn_svm = per_subject_transductive(X_feat, subjects, np.ones_like(tr))[te]
            Xn_env = per_subject_zscore_3d(X_env[te], subjects[te])
        else:
            Xn_svm = norm_2d(X_te_feat, buf)
            Xn_env = norm_3d(X_te_env, buf)

        proba_svm_raw = svm.predict_proba(Xn_svm)
        proba_svm = np.zeros((n_te, len(LABELS))); proba_svm[:, svm_classes] = proba_svm_raw
        dl = DataLoader(WindowsDataset(Xn_env, y_te), batch_size=512, shuffle=False)
        _, _, proba_cnn = evaluate_with_proba(cnn, dl, device)
        proba_soft = (proba_svm + proba_cnn) / 2.0

        for model_name, proba in [("SVM", proba_svm), ("RESNET_SE_CD", proba_cnn), ("soft", proba_soft)]:
            y_pred = proba.argmax(1)
            for i in range(n_te):
                rows_by_cond[condition].append({
                    "model": model_name, "subject": int(heldout), "circuit": circ_te[i],
                    "t_end": float(t_end_te[i]), "y_true": int(y_te[i]), "y_pred": int(y_pred[i]),
                    "mode_raw": int(mode_te[i]), "fs": float(fs_te[i])})
    del cnn
    import gc; gc.collect()
    import torch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows_by_cond


def run(out_dir: Path, root: Path, subjects_filter: set | None = None, spec: dict | None = None) -> int:
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_feat, X_env, y, subjects, circuits, t_end_s, fs, mode_raw = load_common(root, spec)
    subs_u = sorted(np.unique(subjects).tolist())
    if subjects_filter:
        subs_u = [s for s in subs_u if s in subjects_filter]

    paths = {c: out_dir / f"s2_predictions_{c}.csv" for c in CONDITIONS}
    done = {c: set() for c in CONDITIONS}
    for c in CONDITIONS:
        if paths[c].exists():
            done[c] = set(pd.read_csv(paths[c])["subject"].astype(int).unique().tolist())

    for heldout in subs_u:
        if all(heldout in done[c] for c in CONDITIONS):
            print(f"[S2.3] skip Sub{heldout} (already complete)", flush=True)
            continue
        rows_by_cond = run_subject(heldout, X_feat, X_env, y, subjects, circuits, t_end_s, fs, mode_raw,
                                   None, device)
        for c in CONDITIONS:
            if heldout in done[c]:
                continue
            pd.DataFrame(rows_by_cond[c]).to_csv(paths[c], mode="a", header=not paths[c].exists(), index=False)
        print(f"[S2.3] Sub{heldout} done", flush=True)

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results_kc23_s2_predictions")
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--window-ms", type=int, default=250, choices=[250, 400],
                    help="250 (default, the S2.3 job) or 400 (S2b, the window trade)")
    args = ap.parse_args()
    subj_filter = {int(s) for s in args.subjects.split(",")} if args.subjects else None
    sys.exit(run(Path(args.out), Path(args.root), subj_filter, SPEC_400 if args.window_ms == 400 else SPEC_250))


if __name__ == "__main__":
    main()
