#!/usr/bin/env python3
"""
run_filter_phase345.py
=======================
EXPERIMENT_PLAN_FILTER.md Phases 3-5. Scores each of the 4 preprocessing arms
(A/B/C/D) x 2 models (SVM/RF), unsmoothed and causal-k=5-probability-averaged
smoothed, on macro-F1 and the POOLED DNS->WAK critical-error rate. Reuses
causal_probavg / subject_meta_order / macro_f1 / crit_err_pooled from
run_causal_smoothing.py UNCHANGED, per the plan's explicit instruction not to
write a second smoothing function.

Then Phase 4 stats (B vs A, C vs A, D vs A, D vs best-of-{B,C}; Holm within
each (model, scoring-mode, metric) family of 4) and Phase 5 figure.

Does not touch results_loso_freq_persubj/ or any published directory. Does not
edit any thesis chapter. Stops before Phase 6 -- Gate C is reported, not
decided here.
"""
from __future__ import annotations
import sys, io, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from run_causal_smoothing import causal_probavg, subject_meta_order, macro_f1, crit_err_pooled, LABELS, DNS, WAK

ROOT = Path(__file__).parent
FILTER_DIR = ROOT / "results_filter_causal"
FEAT_DIR = ROOT / "features_out_filter"
FIGDIR = ROOT / "report_figs" / "new_experiments"; FIGDIR.mkdir(parents=True, exist_ok=True)

ARM_STEMS = {
    "A": "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbFalse_ceFalse_features_ext",
    "B": "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbTrue_ceFalse_features_ext",
    "C": "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbFalse_ceTrue_features_ext",
    "D": "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbTrue_ceTrue_features_ext",
}
ARMS = ["A", "B", "C", "D"]
SUBS = list(range(1, 41))
K = 5
PUB_SVM, PUB_RF = 0.7767, 0.7732
NONDET_PP = 0.5


def bca_ci(x, n_boot=10000):
    x = np.asarray(x, float)
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


def cohen_dz(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan


def holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def meta_path(arm):
    stem = ARM_STEMS[arm]
    return FEAT_DIR / f"{stem[:-len('_features_ext')]}_features_meta.csv"


def phase3():
    print("=" * 78 + "\nPHASE 3  score every arm x model, unsmoothed and k=5 smoothed (pooled crit-error)\n" + "=" * 78)
    rows = []
    subj_rows = []
    for arm in ARMS:
        meta = pd.read_csv(meta_path(arm))
        for model in ("SVM", "RF"):
            proba_dir = FILTER_DIR / arm / "proba"
            yts_u, preds_u, yts_s, preds_s = [], [], [], []
            for s in SUBS:
                f = proba_dir / f"{model}_sub{s:02d}.npz"
                z = np.load(f)
                yt, proba = z["y_true"].astype(int), z["proba"]
                order, trial, y_meta = subject_meta_order(meta, s)
                assert np.array_equal(y_meta, yt), f"row-order mismatch arm {arm} {model} sub{s}"
                yt_o = yt[order]
                raw_o = proba[order].argmax(1)
                smoothed_o = causal_probavg(proba[order], trial, K)
                yts_u.append(yt_o); preds_u.append(raw_o)
                yts_s.append(yt_o); preds_s.append(smoothed_o)
                subj_rows.append(dict(arm=arm, model=model, subject=s, mode="unsmoothed",
                                      f1=macro_f1(yt_o, raw_o),
                                      crit_err=float(np.mean(raw_o[yt_o == DNS] == WAK)) if (yt_o == DNS).any() else np.nan))
                subj_rows.append(dict(arm=arm, model=model, subject=s, mode="smoothed",
                                      f1=macro_f1(yt_o, smoothed_o),
                                      crit_err=float(np.mean(smoothed_o[yt_o == DNS] == WAK)) if (yt_o == DNS).any() else np.nan))
            for mode, yts, preds in (("unsmoothed", yts_u, preds_u), ("smoothed", yts_s, preds_s)):
                f1_mean = np.mean([macro_f1(a, b) for a, b in zip(yts, preds)])
                ce_pool, n_dns = crit_err_pooled(yts, preds)
                rows.append(dict(arm=arm, model=model, mode=mode, macro_f1_mean=f1_mean,
                                 crit_err_pooled=ce_pool, n_dns=n_dns))
                print(f"  arm {arm} {model:3s} {mode:10s}: macro-F1={f1_mean:.4f}  DNS->WAK(pooled)={ce_pool:.4f}")

    summ = pd.DataFrame(rows)
    sw = pd.DataFrame(subj_rows)
    summ.to_csv(FILTER_DIR / "causal_filter_summary.csv", index=False)
    sw.to_csv(FILTER_DIR / "causal_filter_subjectwise.csv", index=False)
    print(f"\n  [save] {FILTER_DIR / 'causal_filter_summary.csv'}")
    print(f"  [save] {FILTER_DIR / 'causal_filter_subjectwise.csv'}")

    print("\n  Arm A vs published Freq-72 reference (unsmoothed):")
    for model, pub in (("SVM", PUB_SVM), ("RF", PUB_RF)):
        v = summ[(summ.arm == "A") & (summ.model == model) & (summ["mode"] == "unsmoothed")]["macro_f1_mean"].values[0]
        print(f"    {model}: {v:.4f}  (published {pub:.4f}, diff {v-pub:+.4f})")
    return summ, sw


def phase4(sw):
    print("\n" + "=" * 78 + "\nPHASE 4  paired stats (Holm within family of 4 per model/mode/metric)\n" + "=" * 78)
    rows = []
    for model in ("SVM", "RF"):
        for mode in ("unsmoothed", "smoothed"):
            for metric in ("f1", "crit_err"):
                wide = sw[(sw.model == model) & (sw["mode"] == mode)].pivot(index="subject", columns="arm", values=metric)
                best_bc = wide[["B", "C"]].mean().idxmax()  # 'B' or 'C', whichever has the higher subject-mean
                contrasts = [("B", "A"), ("C", "A"), ("D", "A"), ("D", best_bc)]
                pvals, recs = [], []
                for arm_x, arm_ref in contrasts:
                    a, b = wide[arm_x].to_numpy(), wide[arm_ref].to_numpy()
                    mask = ~(np.isnan(a) | np.isnan(b))
                    a, b = a[mask], b[mask]
                    d = a - b
                    if np.allclose(d, 0):
                        W, p = np.nan, 1.0
                    else:
                        W, p = wilcoxon(a, b, zero_method="wilcox")
                    lo, hi = bca_ci(d)
                    label = f"{arm_x} vs {arm_ref}" + (" (best of B/C)" if arm_ref == best_bc and arm_x == "D" else "")
                    recs.append(dict(model=model, mode=mode, metric=metric, contrast=label,
                                     mean_delta_pp=float(d.mean() * 100), cohen_dz=cohen_dz(d),
                                     bca_lo_pp=lo * 100, bca_hi_pp=hi * 100,
                                     nondet_multiple=(d.mean() * 100) / NONDET_PP,
                                     clears_nondet_band=bool(min(abs(lo), abs(hi)) > (NONDET_PP / 100) and np.sign(lo) == np.sign(hi)),
                                     n=int(mask.sum()), W=float(W) if W == W else np.nan, p=float(p)))
                    pvals.append(p)
                hp = holm(pvals)
                for r, h in zip(recs, hp):
                    r["p_holm"] = float(h); r["sig_holm_0.05"] = "Yes" if h < 0.05 else "No"
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(FILTER_DIR / "causal_filter_wilcoxon.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n  [save] {FILTER_DIR / 'causal_filter_wilcoxon.csv'}")
    n_bh = len(df)
    print(f"\n  New BH family members added: {n_bh} (4 contrasts x 2 models x 2 modes x 2 metrics).")
    return df, n_bh


def phase5(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    colors = {"SVM": "#e07b39", "RF": "#2e6f9e"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    x = np.arange(len(ARMS))
    for model in ("SVM", "RF"):
        for mode, ls in (("unsmoothed", "-"), ("smoothed", "--")):
            sub = summ[(summ.model == model) & (summ["mode"] == mode)].set_index("arm").reindex(ARMS)
            axL.plot(x, sub["macro_f1_mean"], marker="o", ms=7, lw=2, ls=ls, color=colors[model],
                     label=f"{model} {mode}")
            axR.plot(x, sub["crit_err_pooled"], marker="o", ms=7, lw=2, ls=ls, color=colors[model],
                     label=f"{model} {mode}")
    for ax, title, ylab in ((axL, "Macro-F1 by arm", "LOSO macro-F1"),
                            (axR, "DNS→WAK critical-error rate by arm", "pooled critical-error rate")):
        ax.set_xticks(x); ax.set_xticklabels(ARMS)
        ax.set_xlabel("preprocessing arm")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axL.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle("EXPERIMENT_PLAN_FILTER: cost of the two acausal preprocessing stages", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "causal_filter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [save] {FIGDIR / 'causal_filter.png'}")


def main():
    summ, sw = phase3()
    wil, n_bh = phase4(sw)
    phase5(summ)

    print("\n" + "=" * 78 + "\nWhich outcome, and Gate C\n" + "=" * 78)
    gate_open = False
    for model in ("SVM", "RF"):
        a_u = summ[(summ.arm == "A") & (summ.model == model) & (summ["mode"] == "unsmoothed")]["macro_f1_mean"].values[0]
        d_u = summ[(summ.arm == "D") & (summ.model == model) & (summ["mode"] == "unsmoothed")]["macro_f1_mean"].values[0]
        drop_pp = (a_u - d_u) * 100
        print(f"  {model} unsmoothed: arm A={a_u:.4f} arm D={d_u:.4f}  drop={drop_pp:+.2f} pp")
        if drop_pp >= 1.0:
            gate_open = True
    print(f"\n  GATE C: {'OPEN' if gate_open else 'CLOSED'} (>=1 pp classical drop for either model, unsmoothed, "
          f"arm A vs arm D)")
    print("  Per the plan: stop and report before Phase 6 either way. The deep-arm run is NOT started here.")
    return n_bh


if __name__ == "__main__":
    main()
