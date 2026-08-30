#!/usr/bin/env python3
"""
run_buffer_composition.py
=========================
EXPERIMENT_PLAN_CAUSALITY.md E-C3 -- how much of the 81.7% deployable figure
depends on the calibration buffer containing all four movements?

t_start restarts at 0 for each of the four movement recordings, so sorting a
held-out subject's windows by t_start (what normalise_test_subject does)
interleaves the movements: the "first 100 time-ordered windows" is really a
near-class-balanced sample from the opening seconds of all four recordings
(measured: 22.0 WAK / 25.4 UPS / 25.9 DNS / 26.7 STDUP on average, all four
classes present for all 40/40 subjects). That is a short scripted unlabeled
calibration set, not a contiguous session-start buffer -- a more favourable
protocol than the thesis describes.

This measures the deployable F1 under buffer-selection modes that vary ONLY
which windows fit the test-subject normalizer (buffer windows are always
excluded from the F1, via run_causal_ensemble.buffer_mask logic; scoring is
over all remaining windows of all four classes):

  mixed100       first 100 by t_start  (published calib100 -- reproduction gate)
  single_WAK     first 100 windows of the WAK recording only  (true contiguous
  single_UPS       single-activity buffer)
  single_DNS
  single_STDUP
  balanced25     25 from the start of each movement  (the explicitly scripted
                 protocol, stated as such)

Efficiency (per the plan): the buffer mode changes nothing about training, so
each model is trained ONCE per held-out subject and all six modes are scored
from that one fitted model.
  SVM arm  : cached _bestparams.json refit (legitimate -- identical model
             across modes, within-model comparison). Two SVCs are fit:
               SVM        plain decision-function SVC   -> mirrors
                          rescore_streaming_buffer_v2.py (gate 0.7476)
               SVM_PROBA  probability=True SVC          -> mirrors
                          run_causal_ensemble.stage_svm (gate 0.7319), feeds
                          the soft vote.
  ensemble : ResNet-SE+CD trained once per held-out subject exactly as
             run_causal_ensemble.stage_cnn, then re-normalized + re-inferred
             per mode (gate 0.7995; soft ensemble gate 0.8168).

Reuses run_streaming_norm_loso.per_subject_transductive and
run_causal_ensemble.buffer_mask; does not reimplement them. The buffer index
set is made explicit (buffer_indices) rather than implicitly the first K of a
time sort.

Stages (--stage):
  svm      : CPU. plain + probability SVC per subject, all 6 modes.
  cnn      : GPU. ResNet-SE+CD per subject, all 6 modes.
  combine  : soft-vote + subjectwise/summary CSVs + gates + paired Wilcoxon.

Output dir: results_buffer_composition/
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from train_classical_loso import load_features_npz, encode_labels
from run_streaming_norm_loso import per_subject_transductive, TIME_COL_CANDIDATES
from run_causal_ensemble import buffer_mask

ROOT = Path(__file__).parent
OUT = ROOT / "results_buffer_composition"; OUT.mkdir(exist_ok=True)
LABELS = ["DNS", "STDUP", "UPS", "WAK"]
SEED = 42
K = 100
PER_MOV_BAL = 25
FEAT = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
NPZ = ROOT / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"

MODES = ["mixed100", "single_WAK", "single_UPS", "single_DNS", "single_STDUP", "balanced25"]

# published gate values, buffer-excluded LOSO mean F1 at calib100 / mixed100
GATES = {
    "SVM":        0.7476,   # results_loso_freq_streaming/streaming_buffer_rescore_summary.csv
    "SVM_PROBA":  0.7319,   # results_causal_ensemble/report.csv  (calib100 SVM)
    "RESNET_SE":  0.7995,   # results_causal_ensemble/report.csv  (calib100 RESNET_SE)
    "soft":       0.8168,   # results_causal_ensemble/report.csv  (calib100 soft)
}
GATE_TOL = 0.005


def load_common():
    X = load_features_npz(FEAT).astype(np.float64)
    meta = pd.read_csv(META)
    y, _ = encode_labels(meta["movement"].astype(str).to_numpy())
    subjects = meta["subject"].astype(int).to_numpy()
    time_col = next((c for c in TIME_COL_CANDIDATES if c in meta.columns), None)
    tvals = meta[time_col].to_numpy() if time_col else np.arange(len(y))
    return X, y, subjects, tvals


def buffer_indices(mode, y_te, t_te):
    """Explicit buffer index set (positions into the test subject's windows,
    original row order). mixed100 == the published first-K-by-t_start buffer."""
    order = np.argsort(t_te, kind="stable")
    if mode == "mixed100":
        return order[:K]
    if mode.startswith("single_"):
        mov = mode.split("_", 1)[1]
        pos = np.where(y_te == LABELS.index(mov))[0]
        mo = pos[np.argsort(t_te[pos], kind="stable")]
        return mo[:K]
    if mode == "balanced25":
        idx = []
        for mov in ["WAK", "UPS", "DNS", "STDUP"]:
            pos = np.where(y_te == LABELS.index(mov))[0]
            mo = pos[np.argsort(t_te[pos], kind="stable")]
            idx.append(mo[:PER_MOV_BAL])
        return np.concatenate(idx)
    raise ValueError(mode)


def is_buffer_mask(buf_pos, n):
    m = np.zeros(n, dtype=bool)
    m[buf_pos] = True
    return m


def norm_2d(X_te, buf_pos):
    b = X_te[buf_pos]
    mu = b.mean(axis=0, keepdims=True)
    sd = b.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_te - mu) / sd


def norm_3d(X_te, buf_pos):
    b = X_te[buf_pos]
    mu = b.mean(axis=(0, 2), keepdims=True)
    sd = b.std(axis=(0, 2), keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_te - mu) / sd


def f1_incl_excl(y_true, y_pred, isb):
    excl = ~isb
    f_incl = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f_excl = f1_score(y_true[excl], y_pred[excl], average="macro", zero_division=0) if excl.sum() else np.nan
    return float(f_incl), float(f_excl)


# ---------------------------------------------------------------------------
def stage_svm(args):
    X, y, subjects, tvals = load_common()
    bp = {int(k): v for k, v in json.loads(open(ROOT / "_bestparams.json").read())["SVM"].items()}
    subs_u = sorted(np.unique(subjects).tolist())
    for mode in MODES:
        (OUT / f"proba_{mode}").mkdir(exist_ok=True)

    for h in subs_u:
        paths = {(tag, mode): OUT / f"proba_{mode}" / f"{tag}_sub{h:02d}.npz"
                 for tag in ("SVM", "SVM_PROBA") for mode in MODES}
        if args.resume and all(p.exists() for p in paths.values()):
            print(f"[resume] skip SVM Sub{h:02d}")
            continue
        t0 = time.time()
        te = (subjects == h); tr = ~te
        Xn_tr = per_subject_transductive(X, subjects, tr)
        Xtr, ytr = Xn_tr[tr], y[tr]
        X_te, y_te, t_te = X[te], y[te], tvals[te]
        n_te = int(te.sum())

        C = bp[h]["clf__C"]; gamma = bp[h].get("clf__gamma", "scale")
        clf_plain = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", cache_size=500)
        clf_proba = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", cache_size=500,
                        probability=True, random_state=SEED)
        clf_plain.fit(Xtr, ytr)
        clf_proba.fit(Xtr, ytr)
        cls = clf_proba.classes_.astype(int)

        def proba_full(raw):
            out = np.zeros((raw.shape[0], len(LABELS)))
            out[:, cls] = raw
            return out

        for mode in MODES:
            buf = buffer_indices(mode, y_te, t_te)
            isb = is_buffer_mask(buf, n_te)
            Xte_n = norm_2d(X_te, buf)
            pred_plain = clf_plain.predict(Xte_n)
            proba_p = proba_full(clf_proba.predict_proba(Xte_n))
            np.savez(paths[("SVM", mode)], pred=pred_plain.astype(np.int32),
                     y_true=y_te.astype(np.int32), is_buffer=isb, buf_n=len(buf))
            np.savez(paths[("SVM_PROBA", mode)], proba=proba_p,
                     y_true=y_te.astype(np.int32), is_buffer=isb, buf_n=len(buf))
        print(f"[fold] SVM Sub{h:02d} done ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
def stage_cnn(args):
    import torch
    from torch.utils.data import DataLoader
    from train_cnn_loso import (per_subject_zscore_3d, choose_val_subjects,
                                normalize_label_to_str, WindowsDataset, LABELS as CNN_LABELS)
    from cnn_architectures import build_model, count_params
    from run_cnn_arch_loso import train_fold, evaluate_with_proba
    assert CNN_LABELS == LABELS

    meta = pd.read_csv(META)
    data = np.load(NPZ)
    X = data["X_env"].astype(np.float32)
    y = np.array([LABELS.index(s) for s in meta["movement"].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).to_numpy()
    time_col = next((c for c in TIME_COL_CANDIDATES if c in meta.columns), None)
    tvals = meta[time_col].to_numpy() if time_col else np.arange(len(y))
    in_ch = X.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}; ResNet-SE params={count_params(build_model('resnet_se', in_ch, len(LABELS))):,}")
    for mode in MODES:
        (OUT / f"proba_{mode}").mkdir(exist_ok=True)

    subs_u = sorted(np.unique(subjects).tolist())
    for h in subs_u:
        paths = {mode: OUT / f"proba_{mode}" / f"RESNET_SE_sub{h:02d}.npz" for mode in MODES}
        if args.resume and all(p.exists() for p in paths.values()):
            print(f"[resume] skip CNN Sub{h:02d}")
            continue
        t0 = time.time()
        te = (subjects == h); tr = ~te
        Xtr_all = per_subject_zscore_3d(X[tr], subjects[tr])
        ytr_all, subtr = y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, 0.15, SEED + h)
        m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)

        model = build_model("resnet_se", in_ch, len(LABELS)).to(device)
        model = train_fold(model, Xtr_all[m_tr], ytr_all[m_tr], Xtr_all[m_va], ytr_all[m_va],
                           device, epochs=40, batch=512, lr=1e-3, patience=7, seed=SEED,
                           aug_mode="chandrop", aug_sigma=0.1, aug_chandrop_p=0.2, aug_timemask_frac=0.15)

        X_te, y_te, t_te = X[te], y[te], tvals[te]
        n_te = int(te.sum())
        for mode in MODES:
            buf = buffer_indices(mode, y_te, t_te)
            isb = is_buffer_mask(buf, n_te)
            Xte_n = norm_3d(X_te, buf)
            dl = DataLoader(WindowsDataset(Xte_n, y_te), batch_size=512, shuffle=False)
            yt, yp, proba = evaluate_with_proba(model, dl, device)
            np.savez(paths[mode], proba=proba.astype(np.float64), y_true=yt.astype(np.int32),
                     is_buffer=isb, buf_n=len(buf))
            fi, fe = f1_incl_excl(yt, yp, isb)
            print(f"  Sub{h:02d} {mode}: f1_excl={fe:.4f} (buf_n={len(buf)})", flush=True)

        del model
        import gc; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[fold] CNN Sub{h:02d} done ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
def stage_combine(args):
    subs_u = list(range(1, 41))
    rows = []
    for mode in MODES:
        def load(tag):
            d = {}
            for s in subs_u:
                f = OUT / f"proba_{mode}" / f"{tag}_sub{s:02d}.npz"
                if f.exists():
                    d[s] = np.load(f)
            return d
        svm, svmp, cnn = load("SVM"), load("SVM_PROBA"), load("RESNET_SE")
        common = sorted(set(svm) & set(svmp) & set(cnn))
        if len(common) < 40:
            print(f"[warn] {mode}: {len(common)}/40 subjects have all three model outputs")
        for s in common:
            yt = svmp[s]["y_true"].astype(int)
            isb = svmp[s]["is_buffer"].astype(bool)
            n = len(yt)
            p_svmp = svmp[s]["proba"].astype(float)[:n]
            p_cnn = cnn[s]["proba"].astype(float)[:n]
            pred_svm = svm[s]["pred"].astype(int)[:n]
            preds = {
                "SVM": pred_svm,
                "SVM_PROBA": p_svmp.argmax(1),
                "RESNET_SE": p_cnn.argmax(1),
                "soft": ((p_svmp + p_cnn) / 2.0).argmax(1),
            }
            for m, yh in preds.items():
                fi, fe = f1_incl_excl(yt, yh, isb)
                rows.append(dict(subject=s, model=m, mode=mode, f1_incl=round(fi, 6),
                                 f1_excl=round(fe, 6), buf_n=int(svmp[s]["buf_n"])))
    sw = pd.DataFrame(rows)
    sw.to_csv(OUT / "buffer_composition_subjectwise.csv", index=False)

    summ = (sw.groupby(["model", "mode"])
              .agg(f1_incl_mean=("f1_incl", "mean"), f1_incl_sd=("f1_incl", "std"),
                   f1_excl_mean=("f1_excl", "mean"), f1_excl_sd=("f1_excl", "std"),
                   n=("subject", "nunique"), buf_n_mean=("buf_n", "mean"))
              .reset_index())
    for c in ["f1_incl_mean", "f1_incl_sd", "f1_excl_mean", "f1_excl_sd", "buf_n_mean"]:
        summ[c] = summ[c].round(4)
    summ.to_csv(OUT / "buffer_composition_summary.csv", index=False)

    print("\n================  E-C3 BUFFER COMPOSITION (buffer-excluded F1)  ================")
    piv = summ.pivot(index="mode", columns="model", values="f1_excl_mean").reindex(MODES)
    print(piv.to_string())

    # ---- validation gate: mixed100 reproduces published calib100 ----
    print("\n================  VALIDATION GATE (mixed100 vs published)  ================")
    gate_ok = True
    g = summ[summ["mode"] == "mixed100"].set_index("model")
    for m, ref in GATES.items():
        if m in g.index:
            val = float(g.loc[m, "f1_excl_mean"])
            ok = abs(val - ref) <= GATE_TOL
            gate_ok &= ok
            print(f"  {m:<10} mixed100 f1_excl={val:.4f}  vs published {ref:.4f}  "
                  f"|diff|={abs(val-ref):.4f}  -> {'PASS' if ok else 'FAIL'}")
        else:
            gate_ok = False
            print(f"  {m:<10} MISSING")
    print(f"[GATE {'PASS' if gate_ok else 'FAIL'}]")

    # ---- paired Wilcoxon: each non-mixed mode vs mixed100 ----
    print("\n================  PAIRED WILCOXON vs mixed100 (f1_excl)  ================")
    stat_rows = []
    for model in ["SVM", "SVM_PROBA", "RESNET_SE", "soft"]:
        base = sw[(sw["model"] == model) & (sw["mode"] == "mixed100")].set_index("subject")["f1_excl"]
        for mode in MODES:
            if mode == "mixed100":
                continue
            cur = sw[(sw["model"] == model) & (sw["mode"] == mode)].set_index("subject")["f1_excl"]
            j = pd.concat([base.rename("b"), cur.rename("c")], axis=1).dropna()
            if len(j) >= 2 and np.any(j.b.values != j.c.values):
                w = wilcoxon(j.c.values, j.b.values)
                dm = float((j.c - j.b).mean())
                stat_rows.append(dict(model=model, mode=mode, n=len(j),
                                      mean_delta_vs_mixed=round(dm, 4),
                                      wilcoxon_p=round(float(w.pvalue), 5)))
    st = pd.DataFrame(stat_rows)
    st.to_csv(OUT / "buffer_composition_wilcoxon.csv", index=False)
    print(st.to_string(index=False))

    # ---- verdict ----
    soft_mixed = float(summ[(summ["model"] == "soft") & (summ["mode"] == "mixed100")]["f1_excl_mean"].iloc[0])
    singles = summ[(summ["model"] == "soft") & (summ["mode"].str.startswith("single_"))]["f1_excl_mean"]
    bal = float(summ[(summ["model"] == "soft") & (summ["mode"] == "balanced25")]["f1_excl_mean"].iloc[0])
    worst = float(singles.min()); best = float(singles.max())
    print("\n================  VERDICT  ================")
    print(f"soft ensemble buffer-excluded F1:  mixed100={soft_mixed:.4f}  "
          f"balanced25={bal:.4f}  single-movement range=[{worst:.4f}, {best:.4f}]")
    drop = soft_mixed - worst
    if drop < 0.02:
        print(f"Single-movement buffers sit within {drop:.4f} of mixed100 -> deployable figure is "
              f"robust to buffer composition (strong sentence).")
    else:
        print(f"Worst single-movement buffer is {drop:.4f} below mixed100 -> Section 4.14.1 should "
              f"report a RANGE bounded by the worst single-movement buffer ({worst:.4f}) and the "
              f"scripted balanced buffer ({bal:.4f}), not the single point.")
    print(f"\n[save] {OUT/'buffer_composition_subjectwise.csv'}")
    print(f"[save] {OUT/'buffer_composition_summary.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["svm", "cnn", "combine"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    {"svm": stage_svm, "cnn": stage_cnn, "combine": stage_combine}[args.stage](args)


if __name__ == "__main__":
    main()
