#!/usr/bin/env python3
"""
run_causal_smoothing.py
=======================
EXPERIMENT_PLAN_SMOOTHING.md -- does a causal (past-only) majority vote change the
deployable causal-ensemble figure, and does it move the DNS->WAK critical-error
rate that Section 5.7.1's "upper bound on per-transition risk" claim is about?

No retraining, no GPU. Post-hoc pass over probabilities already on disk:
  results_causal_ensemble/proba_calib{25,50,100}/{SVM,RESNET_SE}_sub{01..40}.npz
each carrying proba (n,4), y_true (n,), is_buffer (n,), in LABELS order
[DNS, STDUP, UPS, WAK] and -- per the Phase 0 gate below -- in that subject's
meta-CSV row order.

Phases (all CPU):
  0  row-order mapping gate  (HARD: stop on any subject mismatch)
  1  reproduce the published unsmoothed causal ensemble (HARD gate: 0.786/0.811/0.817 +-0.003)
  2  causal majority vote (vote-of-argmax) and causal probability-averaging, k = 1,3,5,7
     + DNS->WAK critical-error rate at every k
     + transductive (offline) smoothed reference at k = 5
  3  paired stats on calib100 (Wilcoxon, Cohen's d, BCa 95%, Holm)
  4  figure

Outputs:
  results_causal_smoothing/smoothing_by_k.csv
  results_causal_smoothing/smoothing_subjectwise.csv
  results_causal_smoothing/smoothing_wilcoxon.csv
  report_figs/new_experiments/causal_smoothing.png

Does NOT fabricate numbers, edit results_causal_ensemble/, or touch any chapter.
"""
from __future__ import annotations
import sys, io, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm
from sklearn.metrics import f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
CE = ROOT / "results_causal_ensemble"
OUT = ROOT / "results_causal_smoothing"; OUT.mkdir(exist_ok=True)
FIGOUT = ROOT / "report_figs" / "new_experiments"; FIGOUT.mkdir(parents=True, exist_ok=True)
META = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
TRANS_DIR = ROOT / "results_ensemble_v2" / "proba_aug_chandrop"   # offline SVM+ResNet-SE+CD, 0.8579 soft

LABELS = ["DNS", "STDUP", "UPS", "WAK"]
DNS, WAK = 0, 3
KS_BUFFER = [25, 50, 100]
K_VOTE = [1, 3, 5, 7]
SUBS = list(range(1, 41))
PUBLISHED_SOFT = {25: 0.786, 50: 0.811, 100: 0.817}          # buffer-excluded 40-fold mean macro-F1
PUBLISHED_MEMBER_C100 = {"SVM": 0.732, "RESNET_SE": 0.800}
TOL = 0.003
TRANSDUCTIVE_UPPER_BOUND = 0.8579
RNG_SEED = 42


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def macro_f1(yt, yp):
    return f1_score(yt, yp, average="macro", zero_division=0)


def crit_err_rate(yt, yp):
    """Proportion of true-DNS windows predicted WAK (Table 4.6 definition). NaN if no DNS.
    Table 4.6 / Section 5.7.1 report this POOLED over all windows; the per-subject value
    here is used only for the paired stats. NaN if the subject has no true-DNS window."""
    m = (yt == DNS)
    return float(np.mean(yp[m] == WAK)) if m.any() else np.nan


def crit_err_pooled(yt_list, yp_list):
    yt = np.concatenate(yt_list); yp = np.concatenate(yp_list)
    m = (yt == DNS)
    return float(np.mean(yp[m] == WAK)), int(m.sum())


def subject_meta_order(meta, s):
    """That subject's rows in meta file order, plus a trial id per row.
    Trial = (subject, movement); t_start is strictly monotonic within each."""
    sm = meta[meta["subject"] == s]
    mv = sm["movement"].to_numpy()
    tstart = sm["t_start"].to_numpy()
    yint = sm["y_int"].to_numpy().astype(np.int32)
    # stable sort to (movement, t_start): regroups rows into contiguous trials
    codes = {m: i for i, m in enumerate(sorted(set(mv)))}
    mvcode = np.array([codes[m] for m in mv])
    order = np.lexsort((tstart, mvcode))          # primary = movement, secondary = t_start
    return order, mvcode[order], yint


def causal_vote(raw, trial, k):
    """Vote-of-argmax over windows i-k+1..i within the same trial, past-only.
    Tie broken toward the most recent window's raw prediction."""
    n = len(raw)
    out = raw.copy()
    for i in range(n):
        lo = max(0, i - k + 1)
        js = [j for j in range(lo, i + 1) if trial[j] == trial[i]]
        c = np.bincount(raw[js], minlength=4)
        mx = c.max()
        winners = np.where(c == mx)[0]
        if winners.size == 1:
            out[i] = winners[0]
        else:
            for j in reversed(js):                # most recent first
                if raw[j] in winners:
                    out[i] = raw[j]
                    break
    return out


def causal_probavg(proba, trial, k):
    """Mean the probability rows over i-k+1..i within the same trial, then argmax."""
    n = len(proba)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        lo = max(0, i - k + 1)
        js = [j for j in range(lo, i + 1) if trial[j] == trial[i]]
        out[i] = proba[js].mean(axis=0).argmax()
    return out


def short_vote_flags(trial, k):
    """True where the effective same-trial causal window is shorter than k."""
    n = len(trial)
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - k + 1)
        cnt = sum(1 for j in range(lo, i + 1) if trial[j] == trial[i])
        flags[i] = cnt < k
    return flags


def bca_ci(x, n_boot=10000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
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
    d = d[~np.isnan(d)]
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


# ----------------------------------------------------------------------------
# PHASE 0 -- row-order mapping gate
# ----------------------------------------------------------------------------
def phase0(meta):
    print("\n" + "=" * 78 + "\nPHASE 0  row-order mapping gate\n" + "=" * 78)
    fail = []
    for s in SUBS:
        sm = meta[meta["subject"] == s]
        y_meta = sm["y_int"].to_numpy().astype(np.int32)                      # meta file order
        z = np.load(CE / "proba_calib100" / f"RESNET_SE_sub{s:02d}.npz")
        y_npz = z["y_true"].astype(np.int32)
        if len(y_meta) != len(y_npz):
            fail.append((s, "count", len(y_meta), len(y_npz), None)); break
        if not np.array_equal(y_meta, y_npz):
            diff = np.where(y_meta != y_npz)[0][:10]
            pairs = [(int(i), int(y_meta[i]), int(y_npz[i])) for i in diff]
            fail.append((s, "labels", len(y_meta), len(y_npz), pairs)); break
    if fail:
        s, kind, c_meta, c_npz, pairs = fail[0]
        print(f"  *** GATE FAILED *** first mismatching subject = {s} ({kind})")
        print(f"      meta count = {c_meta}   npz count = {c_npz}")
        if pairs:
            print(f"      first disagreeing (row_idx, y_meta, y_npz): {pairs}")
        print("  Per plan Phase 0.3: stopping. Do NOT realign by sorting/guessing.")
        print("  Fallback is a separate run that re-emits causal probabilities with a window index.")
        return False

    # SVM vs RESNET_SE must agree on y_true and is_buffer at every buffer length
    agree = True
    for K in KS_BUFFER:
        for s in SUBS:
            zs = np.load(CE / f"proba_calib{K}" / f"SVM_sub{s:02d}.npz")
            zr = np.load(CE / f"proba_calib{K}" / f"RESNET_SE_sub{s:02d}.npz")
            if not (np.array_equal(zs["y_true"], zr["y_true"]) and np.array_equal(zs["is_buffer"], zr["is_buffer"])):
                agree = False
                print(f"  *** SVM/ResNet-SE disagree on y_true/is_buffer at calib{K} sub{s:02d} ***")
    print(f"  Mapping gate: PASS for all 40 subjects (counts + label vectors identical).")
    print(f"  SVM vs ResNet-SE agree on y_true and is_buffer for all 40 subjects at "
          f"calib25/50/100: {'PASS' if agree else 'FAIL'}")
    return agree


# ----------------------------------------------------------------------------
# PHASE 1 -- reproduce the published unsmoothed causal ensemble
# ----------------------------------------------------------------------------
def load_pair(K, s):
    zs = np.load(CE / f"proba_calib{K}" / f"SVM_sub{s:02d}.npz")
    zr = np.load(CE / f"proba_calib{K}" / f"RESNET_SE_sub{s:02d}.npz")
    n = min(len(zs["y_true"]), len(zr["y_true"]))
    return (zs["y_true"][:n].astype(int), zs["proba"][:n], zr["proba"][:n],
            zs["is_buffer"][:n].astype(bool))


def phase1():
    print("\n" + "=" * 78 + "\nPHASE 1  reproduce unsmoothed causal ensemble (buffer-excluded)\n" + "=" * 78)
    repro = {}
    member = {25: {}, 50: {}, 100: {}}
    for K in KS_BUFFER:
        fs, fv, fr = [], [], []
        for s in SUBS:
            yt, ps, pr, isb = load_pair(K, s)
            keep = ~isb
            soft = (ps + pr) / 2.0
            fs.append(macro_f1(yt[keep], soft.argmax(1)[keep]))
            fv.append(macro_f1(yt[keep], ps.argmax(1)[keep]))
            fr.append(macro_f1(yt[keep], pr.argmax(1)[keep]))
        repro[K] = float(np.mean(fs))
        member[K]["SVM"] = float(np.mean(fv))
        member[K]["RESNET_SE"] = float(np.mean(fr))
        print(f"  calib{K:<3}  soft reproduced = {repro[K]:.4f}   published = {PUBLISHED_SOFT[K]:.3f}   "
              f"delta = {repro[K]-PUBLISHED_SOFT[K]:+.4f}   {'OK' if abs(repro[K]-PUBLISHED_SOFT[K])<=TOL else 'MISS'}")
    print(f"  calib100 members: SVM = {member[100]['SVM']:.4f} (pub {PUBLISHED_MEMBER_C100['SVM']:.3f}), "
          f"ResNet-SE+CD = {member[100]['RESNET_SE']:.4f} (pub {PUBLISHED_MEMBER_C100['RESNET_SE']:.3f})")
    ok = all(abs(repro[K] - PUBLISHED_SOFT[K]) <= TOL for K in KS_BUFFER)
    if not ok:
        print("  *** PHASE 1 GATE FAILED *** reproduction outside +-0.003; stopping before any delta.")
    else:
        print("  Phase 1 gate: PASS (all three buffer lengths within +-0.003).")
    return ok, repro, member


# ----------------------------------------------------------------------------
# PHASE 2 -- causal smoothing
# ----------------------------------------------------------------------------
def phase2(meta):
    print("\n" + "=" * 78 + "\nPHASE 2  causal smoothing: vote-of-argmax and probability-averaging\n" + "=" * 78)
    orders = {s: subject_meta_order(meta, s) for s in SUBS}

    subj_rows = []          # per (buffer,k,variant,subject)
    short_frac_rows = []
    pooled = {}             # (K,k,variant) -> ([yt_scored...],[pred_scored...])
    for K in KS_BUFFER:
        for k in K_VOTE:
            for variant in ("vote", "probavg"):
                yts, preds = [], []
                for s in SUBS:
                    yt, ps, pr, isb = load_pair(K, s)
                    order, trial, _ = orders[s]
                    yt_o, isb_o = yt[order], isb[order]
                    soft_o = ((ps + pr) / 2.0)[order]
                    raw_o = soft_o.argmax(1).astype(np.int64)
                    if variant == "vote":
                        pred_o = causal_vote(raw_o, trial, k) if k > 1 else raw_o
                    else:
                        pred_o = causal_probavg(soft_o, trial, k) if k > 1 else raw_o
                    keep = ~isb_o
                    yts.append(yt_o[keep]); preds.append(pred_o[keep])
                    subj_rows.append(dict(buffer=K, k=k, variant=variant, subject=s,
                                          f1=macro_f1(yt_o[keep], pred_o[keep]),
                                          crit_err=crit_err_rate(yt_o[keep], pred_o[keep]),
                                          n_scored=int(keep.sum())))
                pooled[(K, k, variant)] = (yts, preds)
            # short-vote fraction at this (K) is variant-independent; record for k=5 (plan) + all k
            for k in K_VOTE:
                tot = 0; short = 0
                for s in SUBS:
                    yt, ps, pr, isb = load_pair(K, s)
                    order, trial, _ = orders[s]
                    keep = ~isb[order]
                    sf = short_vote_flags(trial, k)
                    tot += int(keep.sum()); short += int((sf & keep).sum())
                short_frac_rows.append(dict(buffer=K, k=k, n_scored=tot, n_short=short,
                                            short_frac=short / tot if tot else np.nan))

    subj = pd.DataFrame(subj_rows)
    subj.to_csv(OUT / "smoothing_subjectwise.csv", index=False)

    # transductive (offline) smoothed reference at k = 5
    trans_rows = []
    for k in [1, 5]:
        for variant in ("vote", "probavg"):
            f1s, yts, preds = [], [], []
            for s in SUBS:
                zs = np.load(TRANS_DIR / f"SVM_sub{s:02d}.npz")
                zr = np.load(TRANS_DIR / f"RESNET_SE_sub{s:02d}.npz")
                n = min(len(zs["y_true"]), len(zr["y_true"]))
                yt = zs["y_true"][:n].astype(int)
                soft = ((zs["proba"][:n] + zr["proba"][:n]) / 2.0)
                order, trial, y_meta = subject_meta_order(meta, s)
                assert np.array_equal(y_meta[:n], yt), f"transductive y_true mismatch sub{s}"
                soft_o = soft[order]
                raw_o = soft_o.argmax(1).astype(np.int64)
                if k == 1:
                    pred_o = raw_o
                elif variant == "vote":
                    pred_o = causal_vote(raw_o, trial, k)
                else:
                    pred_o = causal_probavg(soft_o, trial, k)
                f1s.append(macro_f1(yt[order], pred_o))
                yts.append(yt[order]); preds.append(pred_o)
            ce_pool, n_dns = crit_err_pooled(yts, preds)
            trans_rows.append(dict(buffer="transductive", k=k, variant=variant,
                                   f1_mean=float(np.mean(f1s)), f1_se=float(np.std(f1s, ddof=1) / np.sqrt(len(f1s))),
                                   crit_err_pooled=ce_pool, n_dns=n_dns, n=len(f1s)))

    # aggregate causal (macro-F1 as 40-fold mean; critical-error rate POOLED, per Table 4.6)
    agg_rows = []
    for (K, k, v), g in subj.groupby(["buffer", "k", "variant"]):
        ce_pool, n_dns = crit_err_pooled(*pooled[(K, k, v)])
        agg_rows.append(dict(buffer=K, k=k, variant=v,
                             f1_mean=g["f1"].mean(),
                             f1_se=g["f1"].std(ddof=1) / np.sqrt(len(g)),
                             crit_err_pooled=ce_pool, n_dns=n_dns,
                             crit_err_persubj_mean=np.nanmean(g["crit_err"]),
                             crit_err_persubj_se=np.nanstd(g["crit_err"], ddof=1) / np.sqrt(g["crit_err"].notna().sum()),
                             n=len(g)))
    agg = pd.DataFrame(agg_rows)
    sf = pd.DataFrame(short_frac_rows)
    by_k = pd.concat([agg, pd.DataFrame(trans_rows)], ignore_index=True)
    by_k.to_csv(OUT / "smoothing_by_k.csv", index=False)
    sf.to_csv(OUT / "smoothing_short_vote_fraction.csv", index=False)

    for v in ("vote", "probavg"):
        print(f"\n  [{v}]  macro-F1 (40-fold mean, buffer-excluded) / DNS->WAK critical-error rate (pooled)")
        print("    buf   " + "   ".join(f"k={k}" for k in K_VOTE))
        for K in KS_BUFFER:
            f1line = "  ".join(f"{agg[(agg.buffer==K)&(agg.k==k)&(agg.variant==v)].f1_mean.values[0]:.4f}" for k in K_VOTE)
            celine = "  ".join(f"{agg[(agg.buffer==K)&(agg.k==k)&(agg.variant==v)].crit_err_pooled.values[0]:.4f}" for k in K_VOTE)
            print(f"    c{K:<4} F1  {f1line}")
            print(f"    c{K:<4} CE  {celine}")
    print("\n  transductive smoothed reference (F1 = 40-fold mean, CE = pooled):")
    for r in trans_rows:
        print(f"    k={r['k']} [{r['variant']}]  F1 = {r['f1_mean']:.4f}   CE = {r['crit_err_pooled']:.4f}")
    print("\n  short-vote fraction (windows within k-1 of a trial start, of the scored set):")
    for K in KS_BUFFER:
        row = sf[(sf.buffer == K) & (sf.k == 5)].iloc[0]
        print(f"    calib{K} @ k=5 : {row.n_short}/{row.n_scored} = {row.short_frac:.4f}")

    return subj, by_k, sf, trans_rows


# ----------------------------------------------------------------------------
# PHASE 3 -- paired stats on calib100
# ----------------------------------------------------------------------------
def phase3(subj):
    print("\n" + "=" * 78 + "\nPHASE 3  paired stats, calib100\n" + "=" * 78)
    rows = []
    contrasts = [(3, 1), (5, 1), (5, 3)]
    for variant in ("probavg", "vote"):
        for metric in ("f1", "crit_err"):
            wide = subj[(subj.buffer == 100) & (subj.variant == variant)].pivot(
                index="subject", columns="k", values=metric).sort_index()
            pvals = []; recs = []
            for (ka, kb) in contrasts:
                a = wide[ka].to_numpy(); b = wide[kb].to_numpy()
                mask = ~(np.isnan(a) | np.isnan(b))
                a, b = a[mask], b[mask]
                d = a - b
                if np.allclose(d, 0):
                    p = 1.0; W = np.nan
                else:
                    W, p = wilcoxon(a, b, zero_method="wilcox")
                lo, hi = bca_ci(d)
                recs.append(dict(variant=variant, metric=metric, contrast=f"k{ka}_vs_k{kb}",
                                 mean_delta=float(np.mean(d)), cohen_dz=cohen_dz(a, b),
                                 bca_lo=lo, bca_hi=hi, W=float(W) if W == W else np.nan,
                                 p=float(p), n=int(mask.sum())))
                pvals.append(p)
            hp = holm(pvals)
            for r, hpi in zip(recs, hp):
                r["p_holm"] = float(hpi)
                r["sig_holm_0.05"] = "Yes" if hpi < 0.05 else "No"
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "smoothing_wilcoxon.csv", index=False)
    print(out.to_string(index=False))
    n_bh = int(((out.variant == "probavg")).sum())
    print(f"\n  BH family members added (primary variant = probavg): {n_bh}  "
          f"(3 macro-F1 contrasts + 3 critical-error contrasts). "
          f"vote-of-argmax contrasts are recorded but not entered into the BH family.")
    return out, n_bh


# ----------------------------------------------------------------------------
# PHASE 4 -- figure
# ----------------------------------------------------------------------------
def phase4(agg, trans_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200})
    COLORS = {25: "#e07b39", 50: "#c9457b", 100: "#2e6f9e"}
    variant = "probavg"
    a = agg[agg.variant == variant]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True)
    for K in KS_BUFFER:
        g = a[a.buffer == K].sort_values("k")
        axL.errorbar(g.k, g.f1_mean, yerr=g.f1_se, marker="o", ms=7, lw=2, capsize=3,
                     color=COLORS[K], label=f"calib{K}")
        axR.errorbar(g.k, g.crit_err_persubj_mean, yerr=g.crit_err_persubj_se, marker="o", ms=7,
                     lw=2, capsize=3, color=COLORS[K], label=f"calib{K}")
    tr = next(r for r in trans_rows if r["k"] == 5 and r["variant"] == variant)
    axL.axhline(tr["f1_mean"], ls="--", lw=1.4, color="#3a923a", alpha=0.8,
                label=f"transductive smoothed (k=5) = {tr['f1_mean']:.3f}")

    axL.set_xticks(K_VOTE)
    axL.set_xlabel("causal vote length k (windows)")
    axR.set_xlabel("causal vote length k (windows)")
    axL.set_ylabel("LOSO macro-F1 (buffer-excluded)")
    axR.set_ylabel("DNS→WAK critical-error rate (per-subject mean)")
    axL.set_title("Macro-F1 vs causal vote length")
    axR.set_title("Critical-error rate vs causal vote length")
    axL.legend(fontsize=8.5, frameon=False, loc="upper left")
    axR.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.suptitle("Causal majority-vote smoothing of the causal ensemble "
                 "(probability-averaged; error bars = 1 SE, n = 40)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGOUT / "causal_smoothing.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [save] {FIGOUT / 'causal_smoothing.png'}")


# ----------------------------------------------------------------------------
def main():
    meta = pd.read_csv(META)
    if not phase0(meta):
        print("\nSTOP after Phase 0 (Outcome D).")
        return
    ok, repro, member = phase1()
    if not ok:
        print("\nSTOP after Phase 1 (reproduction gate).")
        return
    subj, by_k, sf, trans_rows = phase2(meta)
    agg = by_k[by_k.buffer.isin(KS_BUFFER)].copy()
    agg["buffer"] = agg["buffer"].astype(int)
    wil, n_bh = phase3(subj)
    phase4(agg, trans_rows)

    # ---- Gate A verdict ----
    print("\n" + "=" * 78 + "\nGATE A verdict\n" + "=" * 78)
    pv = agg[agg.variant == "probavg"]
    def get(K, k, col):
        return float(pv[(pv.buffer == K) & (pv.k == k)][col].values[0])
    base_f1 = get(100, 1, "f1_mean"); k5_f1 = get(100, 5, "f1_mean")
    base_ce = get(100, 1, "crit_err_pooled"); k5_ce = get(100, 5, "crit_err_pooled")
    d_f1 = (k5_f1 - base_f1) * 100
    d_ce = (k5_ce - base_ce) * 100
    print(f"  calib100 probability-averaged:  macro-F1 k1={base_f1:.4f} -> k5={k5_f1:.4f}  (Δ {d_f1:+.2f} pp)")
    print(f"  calib100 probability-averaged:  DNS→WAK  k1={base_ce:.4f} -> k5={k5_ce:.4f}  (Δ {d_ce:+.2f} pp)")
    if d_f1 >= 2.0 and d_ce <= 0:
        verdict = ("A", "smoothing lifts the causal figure by >=2 pp at k=5; quote the smoothed number; "
                        "the causal-filter experiment must be scored BOTH smoothed and unsmoothed.")
    elif d_f1 < 1.0:
        verdict = ("B", "smoothing moves macro-F1 <1 pp; Section 5.7.1 stands as written; "
                        "the causal-filter experiment is scored UNSMOOTHED only.")
    elif d_f1 >= 1.0 and d_ce >= 0:
        verdict = ("C", "macro-F1 rises but the critical-error rate does not fall (or rises): "
                        "Section 5.7.1's upper-bound claim is wrong in the direction that matters and needs rewriting; "
                        "score the causal-filter experiment BOTH smoothed and unsmoothed and foreground the critical-error result.")
    else:
        verdict = ("A/B boundary", f"macro-F1 Δ = {d_f1:+.2f} pp (between 1 and 2 pp), critical-error Δ = {d_ce:+.2f} pp; "
                        "treat as the cheaper branch (unsmoothed) unless the write-up wave decides the 1-2 pp band is material.")
    print(f"\n  OUTCOME {verdict[0]}: {verdict[1]}")
    print(f"\n  New BH family members added by this experiment: {n_bh} (primary variant only).")


if __name__ == "__main__":
    main()
