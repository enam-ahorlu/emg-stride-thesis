#!/usr/bin/env python3
"""
run_causal_ensemble.py
=======================
EXPERIMENT_PLAN_CRITIQUE.md E3 (triage T2) -- the deployable (causal) score for
the headline SVM+ResNet-SE+CD ensemble (transductive 0.858). Critics extrapolated
a causal figure ("~81%") from SVM/RF alone; this measures it directly on the
actual headline ensemble.

Both members are trained ONCE per held-out subject (identical to the headline
procedure: SVM best-params refit, ResNet-SE+CD 40ep/chandrop/per-subject norm
on TRAINING subjects) and then evaluated at calib25/50/100 by swapping only the
held-out subject's normalisation for a causal (first-K-windows) estimate -- so
training cost is 1x per subject, not 3x per (subject,K).

Stages (run with --stage):
  svm      : causal SVM probabilities (CPU). Also produces the transductive
             probability-SVM honesty check (Step 0.5 in the plan) as a byproduct
             since the same fitted model is scored at the transductive config too.
  cnn      : causal ResNet-SE+CD probabilities (GPU).
  combine  : soft-vote SVM+RESNET_SE per K, buffer-included AND buffer-excluded,
             plus solo-model attribution, into results_causal_ensemble/report.csv.

Example:
  python run_causal_ensemble.py --stage svm --resume
  python run_causal_ensemble.py --stage cnn --resume
  python run_causal_ensemble.py --stage combine
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from train_classical_loso import load_features_npz, encode_labels
from run_streaming_norm_loso import per_subject_transductive, normalise_test_subject, TIME_COL_CANDIDATES

ROOT = Path(__file__).parent
OUT = ROOT / "results_causal_ensemble"; OUT.mkdir(exist_ok=True)
LABELS = ["DNS", "STDUP", "UPS", "WAK"]
SEED = 42
WARMUP = 16
KS = [25, 50, 100]
FEAT = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
NPZ = ROOT / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"
HEADLINE_TRANSDUCTIVE_SVM_F1 = 0.7767  # results_loso_freq_persubj SVM summary


def load_common():
    X = load_features_npz(FEAT).astype(np.float64)
    meta = pd.read_csv(META)
    y, _ = encode_labels(meta["movement"].astype(str).to_numpy())
    # "subject" (1..40) is authoritative -- matches _bestparams.json keys and
    # results_loso_freq_persubj heldout_subject; "subject_int" is 0-indexed
    # and used only internally by some scripts, NOT for subject identity here.
    subjects = meta["subject"].astype(int).to_numpy()
    time_col = next((c for c in TIME_COL_CANDIDATES if c in meta.columns), None)
    tvals = meta[time_col].to_numpy() if time_col else np.arange(len(y))
    return X, y, subjects, tvals


def buffer_mask(order, n, k):
    """is_buffer in ORIGINAL row order: True for the first k time-ordered windows."""
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    return rank < k


# ============================================================================
# STAGE: SVM
# ============================================================================
def stage_svm(args):
    X, y, subjects, tvals = load_common()
    bp = json.loads(open(ROOT / "_bestparams.json").read())["SVM"]
    bp = {int(k): v for k, v in bp.items()}
    subs_u = sorted(np.unique(subjects).tolist())

    honesty_rows = []
    for K in KS:
        (OUT / f"proba_calib{K}").mkdir(exist_ok=True)
    (OUT / "proba_transductive_check").mkdir(exist_ok=True)

    for heldout in subs_u:
        out_paths = {K: OUT / f"proba_calib{K}" / f"SVM_sub{heldout:02d}.npz" for K in KS}
        check_path = OUT / "proba_transductive_check" / f"SVM_sub{heldout:02d}.npz"
        if args.resume and all(p.exists() for p in out_paths.values()) and check_path.exists():
            print(f"[resume] skip SVM Sub{heldout:02d}")
            continue

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

        def proba_full(Xte):
            raw = clf.predict_proba(Xte)
            out = np.zeros((raw.shape[0], len(LABELS)))
            out[:, classes] = raw
            return out

        # honesty check: transductive test-subject normalisation, probability SVM
        Xte_trans = normalise_test_subject(X[te], order, "transductive", 0, WARMUP)
        proba_trans = proba_full(Xte_trans)
        f1_trans = f1_score(yte, proba_trans.argmax(1), average="macro", zero_division=0)
        np.savez(check_path, proba=proba_trans, y_true=yte.astype(np.int32))
        honesty_rows.append(dict(subject=int(heldout), f1_macro=float(f1_trans)))

        for K in KS:
            Xte_c = normalise_test_subject(X[te], order, "calib", K, WARMUP)
            proba = proba_full(Xte_c)
            isb = buffer_mask(order, n_te, K)
            np.savez(out_paths[K], proba=proba, y_true=yte.astype(np.int32), is_buffer=isb)

        print(f"[fold] Sub{heldout:02d} done (transductive-check f1={f1_trans:.4f})", flush=True)

    # write/append honesty check summary
    hc_csv = OUT / "honesty_check_transductive_svm.csv"
    if honesty_rows:
        df_new = pd.DataFrame(honesty_rows)
        if hc_csv.exists() and args.resume:
            prev = pd.read_csv(hc_csv)
            df_new = pd.concat([prev, df_new]).drop_duplicates("subject", keep="last")
        df_new.sort_values("subject").to_csv(hc_csv, index=False)
    if hc_csv.exists():
        df = pd.read_csv(hc_csv)
        mean_f1 = df["f1_macro"].mean()
        diff = mean_f1 - HEADLINE_TRANSDUCTIVE_SVM_F1
        print(f"\n[HONESTY CHECK] probability-SVM transductive F1 = {mean_f1:.4f} "
              f"(headline decision-function SVM = {HEADLINE_TRANSDUCTIVE_SVM_F1:.4f}, "
              f"diff = {diff:+.4f}, n={len(df)})")
        if abs(diff) <= 0.005:
            print("[HONESTY CHECK] PASS (within +/-0.005) -- using predict_proba() for the ensemble.")
        else:
            print("[HONESTY CHECK] *** FAIL *** (diff exceeds 0.005) -- "
                  "fall back to decision_function->softmax; do not trust causal numbers yet.")


# ============================================================================
# STAGE: CNN (ResNet-SE+CD)
# ============================================================================
def causal_calib_stats_3d(Xsub_ordered, k):
    k = max(1, min(k, Xsub_ordered.shape[0]))
    mu = Xsub_ordered[:k].mean(axis=(0, 2), keepdims=True)
    sd = Xsub_ordered[:k].std(axis=(0, 2), keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def causal_normalise_test_subject_3d(Xsub, order, k):
    Xord = Xsub[order]
    mu, sd = causal_calib_stats_3d(Xord, k)
    out_ord = (Xord - mu) / sd
    out = np.empty_like(out_ord)
    out[order] = out_ord
    return out


def stage_cnn(args):
    import torch
    from train_cnn_loso import (per_subject_zscore_3d, choose_val_subjects, normalize_label_to_str,
                                 WindowsDataset, LABELS as CNN_LABELS)
    from torch.utils.data import DataLoader
    from cnn_architectures import build_model, count_params
    from run_cnn_arch_loso import train_fold, evaluate_with_proba

    assert CNN_LABELS == LABELS
    meta = pd.read_csv(META)
    data = np.load(NPZ)
    X = data["X_env"].astype(np.float32)
    y = np.array([LABELS.index(s) for s in meta["movement"].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).to_numpy() if "subject" in meta.columns \
        else meta["subject_int"].astype(int).to_numpy()
    time_col = next((c for c in TIME_COL_CANDIDATES if c in meta.columns), None)
    tvals = meta[time_col].to_numpy() if time_col else np.arange(len(y))
    in_ch = X.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}; ResNet-SE params={count_params(build_model('resnet_se', in_ch, len(LABELS))):,}")

    for K in KS:
        (OUT / f"proba_calib{K}").mkdir(exist_ok=True)

    subs_u = sorted(np.unique(subjects).tolist())
    for heldout in subs_u:
        out_paths = {K: OUT / f"proba_calib{K}" / f"RESNET_SE_sub{heldout:02d}.npz" for K in KS}
        if args.resume and all(p.exists() for p in out_paths.values()):
            print(f"[resume] skip CNN Sub{heldout:02d}")
            continue

        te = (subjects == heldout); tr = ~te
        Xtr_all = per_subject_zscore_3d(X[tr], subjects[tr])   # training subjects: transductive (offline, legit)
        ytr_all, subtr = y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, 0.15, SEED + heldout)
        m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)

        model = build_model("resnet_se", in_ch, len(LABELS)).to(device)
        model = train_fold(model, Xtr_all[m_tr], ytr_all[m_tr], Xtr_all[m_va], ytr_all[m_va],
                           device, epochs=40, batch=512, lr=1e-3, patience=7, seed=SEED,
                           aug_mode="chandrop", aug_sigma=0.1, aug_chandrop_p=0.2, aug_timemask_frac=0.15)

        yte = y[te]
        order = np.argsort(tvals[te], kind="stable")
        n_te = int(te.sum())
        for K in KS:
            Xte_c = causal_normalise_test_subject_3d(X[te], order, K)
            dl = DataLoader(WindowsDataset(Xte_c, yte), batch_size=512, shuffle=False)
            yt, yp, proba = evaluate_with_proba(model, dl, device)
            isb = buffer_mask(order, n_te, K)
            np.savez(out_paths[K], proba=proba.astype(np.float64), y_true=yt.astype(np.int32), is_buffer=isb)
            f1 = f1_score(yt, yp, average="macro", zero_division=0)
            print(f"  Sub{heldout:02d} calib{K}: f1={f1:.4f}", flush=True)

        del model
        import gc; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[fold] Sub{heldout:02d} CNN done", flush=True)


# ============================================================================
# STAGE: combine
# ============================================================================
def stage_combine(args):
    subs_u = list(range(1, 41))

    def load_model(tag, K):
        d = {}
        for s in subs_u:
            f = OUT / f"proba_calib{K}" / f"{tag}_sub{s:02d}.npz"
            if f.exists():
                z = np.load(f)
                d[s] = (z["y_true"].astype(int), z["proba"].astype(float), z["is_buffer"].astype(bool))
        return d

    rows = []
    for K in KS:
        svm = load_model("SVM", K)
        cnn = load_model("RESNET_SE", K)
        common = sorted(set(svm) & set(cnn))
        if len(common) < 40:
            print(f"[warn] calib{K}: only {len(common)}/40 subjects have BOTH SVM and RESNET_SE proba so far")
        per_subj = {"SVM": [], "RESNET_SE": [], "soft": []}
        per_subj_excl = {"SVM": [], "RESNET_SE": [], "soft": []}
        for s in common:
            yt_s, p_s, isb_s = svm[s]
            yt_c, p_c, isb_c = cnn[s]
            n = min(len(yt_s), len(yt_c))
            yt, isb = yt_s[:n], isb_s[:n]
            p_svm, p_cnn = p_s[:n], p_c[:n]
            p_soft = (p_svm + p_cnn) / 2.0
            for name, p in [("SVM", p_svm), ("RESNET_SE", p_cnn), ("soft", p_soft)]:
                yhat = p.argmax(1)
                per_subj[name].append(f1_score(yt, yhat, average="macro", zero_division=0))
                excl = ~isb
                if excl.sum() > 0:
                    per_subj_excl[name].append(f1_score(yt[excl], yhat[excl], average="macro", zero_division=0))
        for name in ["SVM", "RESNET_SE", "soft"]:
            rows.append(dict(config=f"calib{K}", model=name,
                             f1_incl_mean=round(float(np.mean(per_subj[name])), 4) if per_subj[name] else np.nan,
                             f1_incl_sd=round(float(np.std(per_subj[name], ddof=1)), 4) if len(per_subj[name]) > 1 else np.nan,
                             f1_excl_mean=round(float(np.mean(per_subj_excl[name])), 4) if per_subj_excl[name] else np.nan,
                             f1_excl_sd=round(float(np.std(per_subj_excl[name], ddof=1)), 4) if len(per_subj_excl[name]) > 1 else np.nan,
                             n=len(per_subj[name])))
        # subjectwise for stats (soft ensemble, buffer-excluded and included)
        pd.DataFrame({f"soft_incl": per_subj["soft"], f"soft_excl": per_subj_excl["soft"],
                      f"SVM_incl": per_subj["SVM"], f"SVM_excl": per_subj_excl["SVM"],
                      f"RESNET_SE_incl": per_subj["RESNET_SE"], f"RESNET_SE_excl": per_subj_excl["RESNET_SE"],
                      "subject": common}).to_csv(OUT / f"calib{K}_subjectwise.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "report.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n[save] {OUT / 'report.csv'}")
    print("Reference: transductive (upper bound) SVM+RESNET_SE soft = 0.8579")
    print("Reference: causal AdaBN calib100 = 0.7679 | classical causal SVM calib100 = 0.745")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["svm", "cnn", "combine"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stage == "svm":
        stage_svm(args)
    elif args.stage == "cnn":
        stage_cnn(args)
    elif args.stage == "combine":
        stage_combine(args)


if __name__ == "__main__":
    main()
