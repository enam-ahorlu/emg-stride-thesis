#!/usr/bin/env python3
"""
run_aonly_ensemble.py
=====================
EXPERIMENT_PLAN_AONLY_ENSEMBLE.md -- does the 85.8% soft-vote headline survive the
active-only class control that Section 4.3.1 already put the members through?

CPU only, no GPU. The deep member is already on disk
(results_aonly_resnet_se_cd_persubj/proba/RESNET_SE_AONLY_sub{01..40}.npz);
the classical member is the probability SVM re-run written to
results_aonly_persubj_proba/proba/SVM_sub{01..40}.npz (Phase 1 command in the plan;
run separately with --save-proba --reuse-params-dir results_aonly_persubj).

Class order [DNS, STDUP, UPS, WAK] throughout. Does not fabricate numbers, does not
touch results_aonly_persubj/, does not edit any chapter.

Outputs:
  results_aonly_ensemble/aonly_ensemble_summary.csv
  results_aonly_ensemble/aonly_ensemble_subjectwise.csv
  report_figs/new_experiments/aonly_ensemble.png
"""
from __future__ import annotations
import sys, io, ast, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm
from sklearn.metrics import f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
OUT = ROOT / "results_aonly_ensemble"; OUT.mkdir(exist_ok=True)
FIGOUT = ROOT / "report_figs" / "new_experiments"; FIGOUT.mkdir(parents=True, exist_ok=True)

AONLY_META = ROOT / "features_out" / "freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_meta.csv"
AONLY_STEM = "freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_ext"
SVM_PERSUBJ = ROOT / "results_aonly_persubj"
SVM_GLOBAL = ROOT / "results_aonly_global"
SVM_PROBA = ROOT / "results_aonly_persubj_proba" / "proba"           # Phase 1 output
RES_PROBA = ROOT / "results_aonly_resnet_se_cd_persubj" / "proba"
CNN_SUMMARY = ROOT / "results_aonly_resnet_se_cd_persubj" / "cnn_arch_summary.csv"
PREDF = SVM_PERSUBJ / "predictions_folds"                            # published SVM/RF preds
TRANS_DIR = ROOT / "results_ensemble_v2" / "proba_aug_chandrop"      # offline SVM+ResNet-SE+CD, 0.8579 soft

LABELS = ["DNS", "STDUP", "UPS", "WAK"]
DNS, WAK = 0, 3
SUBS = list(range(1, 41))

PUB_STDUP_F1 = {"SVM": 0.84, "RF": 0.83, "RESNET_SE": 0.88}
PUB_RESNET_MACRO = 0.823
PUB_MARGIN_PP = {"SVM": 9.54, "RF": 6.66}                            # per-subject minus global
PUB_SVM_ACTIVE_DF = 0.7667                                          # decision-function SVM, published active-only
PUB_HEADLINE = 0.858                                                # transductive SVM+ResNet-SE+CD soft
PLATT_STOP_PP = 2.0
TOL_MEAN = 0.001


def macro_f1(yt, yp): return f1_score(yt, yp, average="macro", zero_division=0)
def perclass_f1(yt, yp): return f1_score(yt, yp, average=None, labels=[0, 1, 2, 3], zero_division=0)
def crit_err(yt, yp):
    m = (yt == DNS)
    return float(np.mean(yp[m] == WAK)) if m.any() else np.nan


def bca_ci(x, n_boot=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3: return (np.nan, np.nan)
    seed = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=4).digest(), "big")
    rng = np.random.default_rng(seed)
    th = x.mean()
    bs = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    z0 = norm.ppf(min(max((bs < th).mean(), 1e-4), 1 - 1e-4))
    jk = np.array([np.delete(x, i).mean() for i in range(len(x))]); jm = jk.mean()
    den = 6 * (((jm - jk) ** 2).sum() ** 1.5)
    a = (((jm - jk) ** 3).sum() / den) if den else 0.0
    def q(al):
        z = z0 + norm.ppf(al)
        return np.percentile(bs, 100 * norm.cdf(z0 + z / (1 - a * z)))
    return float(q(.025)), float(q(.975))


def cohen_dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan


def holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def load_pred_folds(model):
    """published SVM/RF active-only per-subject (y_true,y_pred) from predictions_folds."""
    out = {}
    for s in SUBS:
        yt = np.load(PREDF / f"{AONLY_STEM}_{model}_sub{s:02d}_y_true.npy", allow_pickle=True).astype(int)
        yp = np.load(PREDF / f"{AONLY_STEM}_{model}_sub{s:02d}_y_pred.npy", allow_pickle=True).astype(int)
        out[s] = (yt, yp)
    return out


# ----------------------------------------------------------------------------
def phase0():
    print("\n" + "=" * 78 + "\nPHASE 0  reproduce Section 4.3.1's published active-only figures\n" + "=" * 78)
    ok = True
    meta = pd.read_csv(AONLY_META)

    # window composition
    n_tot = len(meta); vc = meta["movement"].value_counts()
    stdup_frac = vc["STDUP"] / n_tot
    cell = meta.groupby(["subject", "movement"]).size()
    min_cell = int(cell.min())
    print(f"  windows total = {n_tot} (expect 13643)  {'OK' if n_tot == 13643 else 'MISS'}")
    print(f"  STDUP windows = {vc['STDUP']} (expect 1982), fraction = {stdup_frac:.4f} (expect ~0.145)  "
          f"{'OK' if vc['STDUP'] == 1982 else 'MISS'}")
    print(f"  min subject x class cell = {min_cell} (expect >= 32), cells < 32 = {int((cell < 32).sum())}  "
          f"{'OK' if min_cell >= 32 else 'MISS'}")
    ok &= (n_tot == 13643 and vc["STDUP"] == 1982 and min_cell >= 32)

    # per-class STDUP F1 + ResNet macro
    stdup = {}
    for model in ("SVM", "RF"):
        pf = load_pred_folds(model)
        per = np.array([perclass_f1(*pf[s]) for s in SUBS]).mean(0)
        stdup[model] = per[1]
    res_per, res_macro = [], []
    for s in SUBS:
        z = np.load(RES_PROBA / f"RESNET_SE_AONLY_sub{s:02d}.npz")
        yt, yp = z["y_true"].astype(int), z["proba"].argmax(1)
        res_per.append(perclass_f1(yt, yp)); res_macro.append(macro_f1(yt, yp))
    stdup["RESNET_SE"] = np.array(res_per).mean(0)[1]
    res_macro_mean = float(np.mean(res_macro))
    cnn_reported = float(pd.read_csv(CNN_SUMMARY)["f1_macro_mean"].iloc[0])
    for m in ("SVM", "RF", "RESNET_SE"):
        hit = abs(round(stdup[m], 2) - PUB_STDUP_F1[m]) < 1e-9
        print(f"  {m:10s} active-only STDUP F1 = {stdup[m]:.4f} -> {round(stdup[m],2):.2f} (expect {PUB_STDUP_F1[m]:.2f})  {'OK' if hit else 'MISS'}")
        ok &= hit
    print(f"  ResNet-SE+CD active-only macro-F1 = {res_macro_mean:.4f} (cnn_arch_summary {cnn_reported:.4f}, expect 0.823)  "
          f"{'OK' if abs(res_macro_mean - PUB_RESNET_MACRO) < 0.0015 else 'MISS'}")
    ok &= abs(res_macro_mean - PUB_RESNET_MACRO) < 0.0015

    # normalization margins (per-subject minus global), paired mean over 40
    def sw(dirp, model):
        p = sorted(dirp.glob(f"*{model}_nested_loso_subjectwise.csv"))[0]
        return pd.read_csv(p).sort_values("heldout_subject")["f1_macro"].to_numpy(float)
    for model in ("SVM", "RF"):
        ps = sw(SVM_PERSUBJ, model); gl = sw(SVM_GLOBAL, model)
        margin_pp = float((ps - gl).mean() * 100)
        hit = abs(margin_pp - PUB_MARGIN_PP[model]) < 0.05
        print(f"  {model} per-subject - global margin = {margin_pp:+.2f} pp (expect {PUB_MARGIN_PP[model]:+.2f})  {'OK' if hit else 'MISS'}")
        ok &= hit

    print(f"\n  PHASE 0 GATE: {'PASS' if ok else 'FAIL -- stop, the whole experiment is a comparison against these.'}")
    return ok


# ----------------------------------------------------------------------------
def phase1():
    print("\n" + "=" * 78 + "\nPHASE 1  probability SVM on the active-only features (Platt-routing check)\n" + "=" * 78)
    n_have = len(list(SVM_PROBA.glob("SVM_sub*.npz")))
    if n_have < 40:
        print(f"  *** only {n_have}/40 SVM proba files present at {SVM_PROBA} ***")
        print("  Run the Phase 1 command first:")
        print("    python train_classical_loso.py --features features_out/%s.npz \\" % AONLY_STEM)
        print("      --meta features_out/%s ... --save-proba --reuse-params-dir results_aonly_persubj" %
              AONLY_META.name)
        return None
    f1s = []
    for s in SUBS:
        z = np.load(SVM_PROBA / f"SVM_sub{s:02d}.npz")
        f1s.append(macro_f1(z["y_true"].astype(int), z["proba"].argmax(1)))
    proba_mean = float(np.mean(f1s))
    shift_pp = (proba_mean - PUB_SVM_ACTIVE_DF) * 100
    print(f"  probability SVM (predict_proba route)  active-only macro-F1 = {proba_mean:.4f}")
    print(f"  published SVM  (decision_function route)               = {PUB_SVM_ACTIVE_DF:.4f}")
    print(f"  Platt-routing shift = {shift_pp:+.2f} pp   (Section 4.13.1 reports 73.2 vs 74.8 = -1.6 pp on the causal ensemble)")
    if abs(shift_pp) > PLATT_STOP_PP:
        print(f"  *** shift exceeds {PLATT_STOP_PP:.0f} pp -- STOP and report (larger than the routing effect measured elsewhere) ***")
        return dict(proba_mean=proba_mean, shift_pp=shift_pp, stop=True)
    print(f"  shift within +-{PLATT_STOP_PP:.0f} pp -- consistent with pairwise-coupling routing, not an error. Continuing.")
    return dict(proba_mean=proba_mean, shift_pp=shift_pp, stop=False, f1s=np.array(f1s))


# ----------------------------------------------------------------------------
def hard_vote_2(p_a, p_b):
    """2-member hard vote; ties broken toward member B (ResNet-SE+CD, higher solo F1)."""
    a, b = p_a.argmax(1), p_b.argmax(1)
    out = b.copy()
    agree = (a == b)
    out[agree] = a[agree]
    return out


def phase2():
    print("\n" + "=" * 78 + "\nPHASE 2  combine, score on the active-only class set\n" + "=" * 78)
    rows = []
    yt_all = {"soft": [], "hard": [], "SVM": [], "RESNET_SE": []}
    yp_all = {"soft": [], "hard": [], "SVM": [], "RESNET_SE": []}
    for s in SUBS:
        zs = np.load(SVM_PROBA / f"SVM_sub{s:02d}.npz")
        zr = np.load(RES_PROBA / f"RESNET_SE_AONLY_sub{s:02d}.npz")
        yt = zs["y_true"].astype(int)
        assert np.array_equal(yt, zr["y_true"].astype(int)), f"y_true mismatch sub{s}"
        ps, pr = zs["proba"], zr["proba"]
        preds = {"SVM": ps.argmax(1), "RESNET_SE": pr.argmax(1),
                 "soft": ((ps + pr) / 2).argmax(1), "hard": hard_vote_2(ps, pr)}
        for name, yp in preds.items():
            pc = perclass_f1(yt, yp)
            rows.append(dict(subject=s, member=name, macro_f1=macro_f1(yt, yp),
                             f1_DNS=pc[0], f1_STDUP=pc[1], f1_UPS=pc[2], f1_WAK=pc[3],
                             crit_err=crit_err(yt, yp)))
            yt_all[name].append(yt); yp_all[name].append(yp)
    sw = pd.DataFrame(rows)
    sw.to_csv(OUT / "aonly_ensemble_subjectwise.csv", index=False)

    summ = []
    for name in ("SVM", "RESNET_SE", "hard", "soft"):
        g = sw[sw.member == name]
        yt = np.concatenate(yt_all[name]); yp = np.concatenate(yp_all[name])
        pooled_ce = float(np.mean(yp[yt == DNS] == WAK))
        summ.append(dict(member=name, n_subjects=len(g),
                         macro_f1_mean=g.macro_f1.mean(), macro_f1_sd=g.macro_f1.std(ddof=1),
                         f1_DNS=g.f1_DNS.mean(), f1_STDUP=g.f1_STDUP.mean(),
                         f1_UPS=g.f1_UPS.mean(), f1_WAK=g.f1_WAK.mean(),
                         crit_err_pooled=pooled_ce, crit_err_persubj_mean=np.nanmean(g.crit_err),
                         n_active_windows=len(yt)))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "aonly_ensemble_summary.csv", index=False)
    print(summ.to_string(index=False))
    return sw, summ


# ----------------------------------------------------------------------------
def phase3(sw, summ):
    print("\n" + "=" * 78 + "\nPHASE 3  paired stats\n" + "=" * 78)
    wide = sw.pivot(index="subject", columns="member", values="macro_f1").sort_index()
    recs, pvals = [], []
    for other in ("RESNET_SE", "SVM"):
        a = wide["soft"].to_numpy(); b = wide[other].to_numpy()
        W, p = wilcoxon(a, b)
        lo, hi = bca_ci(a - b)
        recs.append(dict(contrast=f"aonly soft vs aonly {other}", mean_delta_pp=float((a - b).mean() * 100),
                         cohen_dz=cohen_dz(a, b), bca_lo_pp=lo * 100, bca_hi_pp=hi * 100,
                         n_improved=int((a > b).sum()), W=float(W), p=float(p)))
        pvals.append(p)
    hp = holm(pvals)
    for r, h in zip(recs, hp):
        r["p_holm"] = float(h); r["sig_holm_0.05"] = "Yes" if h < 0.05 else "No"
    st = pd.DataFrame(recs)
    st.to_csv(OUT / "aonly_ensemble_wilcoxon.csv", index=False)
    print(st.to_string(index=False))

    soft_mean = summ.loc[summ.member == "soft", "macro_f1_mean"].iloc[0]
    print(f"\n  Level difference vs the published headline (NOT a paired test -- different window sets):")
    print(f"    published transductive SVM+ResNet-SE+CD soft = {PUB_HEADLINE:.3f}  on 26,347 AorR windows (full class def)")
    print(f"    active-only ensemble                          = {soft_mean:.4f}  on 13,643 active-only windows")
    print(f"    level drop = {(PUB_HEADLINE - soft_mean) * 100:+.2f} pp  (reported the way Section 4.3.1 reports STDUP 0.96 -> 0.84)")
    print(f"\n  New BH family members added: 2 (aonly soft vs each member).")
    return st, soft_mean


# ----------------------------------------------------------------------------
def published_perclass_from_trans():
    """Per-subject-mean per-class F1 under the published class definition, computed
    consistently from results_ensemble_v2/proba_aug_chandrop for all four models."""
    acc = {m: [] for m in ("SVM", "RF", "RESNET_SE", "soft")}
    for s in SUBS:
        z = {t: np.load(TRANS_DIR / f"{t}_sub{s:02d}.npz") for t in ("SVM", "RF", "RESNET_SE")}
        n = min(len(z[t]["y_true"]) for t in z)
        yt = z["SVM"]["y_true"][:n].astype(int)
        ps, pr = z["SVM"]["proba"][:n], z["RESNET_SE"]["proba"][:n]
        preds = {"SVM": ps.argmax(1), "RF": z["RF"]["proba"][:n].argmax(1),
                 "RESNET_SE": pr.argmax(1), "soft": ((ps + pr) / 2).argmax(1)}
        for m, yp in preds.items():
            acc[m].append(perclass_f1(yt, yp))
    return {m: np.array(v).mean(0) for m, v in acc.items()}


def phase4(sw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 200})
    CLS = ["STDUP", "UPS", "WAK", "DNS"]
    idx = {c: LABELS.index(c) for c in CLS}
    pub = published_perclass_from_trans()

    # active-only per-class: SVM/RF from predictions_folds, ResNet from proba, ensemble from sw
    act = {}
    for model in ("SVM", "RF"):
        pf = load_pred_folds(model)
        act[model] = np.array([perclass_f1(*pf[s]) for s in SUBS]).mean(0)
    res = []
    for s in SUBS:
        z = np.load(RES_PROBA / f"RESNET_SE_AONLY_sub{s:02d}.npz")
        res.append(perclass_f1(z["y_true"].astype(int), z["proba"].argmax(1)))
    act["RESNET_SE"] = np.array(res).mean(0)
    g = sw[sw.member == "soft"]
    act["soft"] = np.array([g.f1_DNS.mean(), g.f1_STDUP.mean(), g.f1_UPS.mean(), g.f1_WAK.mean()])  # DNS,STDUP,UPS,WAK order
    act_by_lab = {"soft": {LABELS[i]: act["soft"][i] for i in range(4)}}

    models = [("SVM", "SVM"), ("RF", "RF"), ("RESNET_SE", "ResNet-SE+CD"), ("soft", "Soft-vote ensemble")]
    cols = {"SVM": "#e07b39", "RF": "#f0b27a", "RESNET_SE": "#2e6f9e", "soft": "#3a923a"}
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3), sharey=True)
    x = np.arange(len(CLS)); w = 0.38
    for ax, (key, title) in zip(axes, models):
        p = [pub[key][idx[c]] for c in CLS]
        if key == "soft":
            a = [act_by_lab["soft"][c] for c in CLS]
        else:
            a = [act[key][idx[c]] for c in CLS]
        ax.bar(x - w / 2, p, w, color="#cfcfcf", edgecolor="black", lw=0.3, label="published (STDUP incl. rest)")
        ax.bar(x + w / 2, a, w, color=cols[key], edgecolor="black", lw=0.3, label="active-only STDUP")
        for i in range(len(CLS)):
            d = (a[i] - p[i]) * 100
            ax.annotate("0" if abs(d) < 0.5 else f"{d:+.0f}", xy=(i + w / 2, a[i] + 0.012),
                        ha="center", fontsize=8, color="#c0392b" if d < -1 else "#1e7d34")
        ax.set_xticks(x); ax.set_xticklabels(CLS, fontsize=9); ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("per-class LOSO F1"); axes[0].set_ylim(0.55, 1.03)
    axes[0].legend(loc="lower left", fontsize=7.5, frameon=False)
    fig.suptitle("Active-only class control extended to the soft-vote ensemble "
                 "(per-class LOSO F1, published vs active-only STDUP)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGOUT / "aonly_ensemble.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [save] {FIGOUT / 'aonly_ensemble.png'}")


# ----------------------------------------------------------------------------
def main():
    if not phase0():
        print("\nSTOP after Phase 0.")
        return
    p1 = phase1()
    if p1 is None:
        print("\nSTOP: Phase 1 SVM proba not yet available.")
        return
    if p1.get("stop"):
        print("\nSTOP after Phase 1 (Platt-routing shift exceeds 2 pp).")
        return
    sw, summ = phase2()
    st, soft_mean = phase3(sw, summ)
    phase4(sw)

    print("\n" + "=" * 78 + "\nGATE B verdict\n" + "=" * 78)
    res_alone = summ.loc[summ.member == "RESNET_SE", "macro_f1_mean"].iloc[0]
    svm_alone = summ.loc[summ.member == "SVM", "macro_f1_mean"].iloc[0]
    print(f"  active-only ensemble macro-F1 = {soft_mean:.4f}   "
          f"(ResNet-SE+CD alone {res_alone:.4f}, SVM alone {svm_alone:.4f})")
    if abs(soft_mean - res_alone) <= 0.01 and soft_mean >= 0.815:
        v = ("A", f"the ensemble lands near its deep member ({soft_mean:.3f} vs {res_alone:.3f}); the control "
                  "costs the headline about what it cost every other model; Section 4.3.1 gains one row and the "
                  "abstract's class-set sentence gains a number.")
    elif soft_mean < res_alone - 0.01:
        v = ("B", f"the ensemble ({soft_mean:.3f}) falls further than its members; the soft vote was drawing part "
                  "of its advantage from the quiescent class -- needs saying in Section 4.3.1, Section 5.7 and the abstract.")
    else:
        v = ("C", f"the ensemble ({soft_mean:.3f}) holds up better than its members; strengthens Section 5.7's "
                  "argument that the two members make different errors.")
    print(f"\n  OUTCOME {v[0]}: {v[1]}")


if __name__ == "__main__":
    main()
