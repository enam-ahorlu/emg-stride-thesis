#!/usr/bin/env python3
"""
run_featureset_loso_analysis.py
===============================
EXPERIMENT_PLAN_FEATURESETS.md Phases 0, 2, 3 (analysis + figure). Phase 1 (the two
nested LOSO runs for Extended-54 and Combined-81) is run separately by
train_classical_loso.py; this reads their subjectwise CSVs plus the two existing
anchor runs and assembles the four-way comparison.

Anchors already on disk:
  Base-36  results_loso_norm_persubj   SVM 0.7769  RF 0.7711
  Freq-72  results_loso_freq_persubj   SVM 0.7767  RF 0.7732   <- Phase 0 gate: must be 0.777 / 0.773
New (Phase 1):
  Extended-54  results_featureset_loso/ext54
  Combined-81  results_featureset_loso/combined81

Does not fabricate numbers, does not touch results_loso_norm_persubj / results_loso_freq_persubj,
does not edit any chapter.

Outputs:
  results_featureset_loso/featureset_loso_summary.csv
  results_featureset_loso/featureset_loso_subjectwise.csv
  results_featureset_loso/featureset_loso_wilcoxon.csv
  report_figs/new_experiments/featureset_loso.png
"""
from __future__ import annotations
import sys, io, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent
FS = ROOT / "results_featureset_loso"; FS.mkdir(exist_ok=True)
FIGOUT = ROOT / "report_figs" / "new_experiments"; FIGOUT.mkdir(parents=True, exist_ok=True)

SETS = {
    "Base-36": dict(dim=36, dir=ROOT / "results_loso_norm_persubj",
                    stem="windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_base"),
    "Extended-54": dict(dim=54, dir=FS / "ext54",
                        stem="windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext"),
    "Freq-72": dict(dim=72, dir=ROOT / "results_loso_freq_persubj",
                    stem="freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext"),
    "Combined-81": dict(dim=81, dir=FS / "combined81",
                        stem="combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_full"),
}
FEAT_NPZ = {
    "Base-36": "features_out/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_base.npz",
    "Extended-54": "features_out/windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz",
    "Freq-72": "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz",
    "Combined-81": "features_out/combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_full.npz",
}
# Subject-dependent macro-F1 from Section 4.1.1 (results_classical_* subjdep_cv.csv)
SUBJDEP = {
    "Base-36": {"SVM": 0.8591, "RF": 0.8392},
    "Extended-54": {"SVM": 0.8415, "RF": 0.8405},
    "Freq-72": {"SVM": 0.8734, "RF": 0.8434},
    "Combined-81": {"SVM": 0.8722, "RF": 0.8482},
}
GATE_FREQ72 = {"SVM": 0.777, "RF": 0.773}
ORDER = ["Base-36", "Extended-54", "Freq-72", "Combined-81"]


def sw_vec(setname, model):
    d = SETS[setname]
    p = d["dir"] / f"{d['stem']}__{model}_nested_loso_subjectwise.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).sort_values("heldout_subject")
    return df["heldout_subject"].to_numpy(int), df["f1_macro"].to_numpy(float)


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


def phase0():
    print("=" * 78 + "\nPHASE 0  confirm the two existing runs are what the plan assumes\n" + "=" * 78)
    ok = True
    # feature dims
    from train_classical_loso import load_features_npz
    for s in ORDER:
        w = load_features_npz(Path(FEAT_NPZ[s])).shape[1]
        hit = (w == SETS[s]["dim"])
        print(f"  {s:12s} npz width = {w} (expect {SETS[s]['dim']})  {'OK' if hit else 'MISS'}")
        ok &= hit
    # anchor means + Freq-72 gate
    for s in ("Base-36", "Freq-72"):
        for m in ("SVM", "RF"):
            r = sw_vec(s, m)
            if r is None:
                print(f"  {s} {m}: subjectwise MISSING"); ok = False; continue
            subs, v = r
            mean = v.mean()
            line = f"  {s:9s} {m}: LOSO macro-F1 = {mean:.4f}  (n={len(v)})"
            if s == "Freq-72":
                hit = abs(round(mean, 3) - GATE_FREQ72[m]) < 1e-9
                line += f"   gate expects {GATE_FREQ72[m]:.3f}  {'OK' if hit else 'MISS -- STOP, this is the anchor'}"
                ok &= hit
            print(line)
    print("  NOTE: neither anchor dir carries run_config.json; --norm-mode/--cv-scheme/--inner-splits/"
          "--seed cannot be read back. Directory names ('_persubj', LOSO subjectwise schema) and the "
          "reproduced Freq-72 means are the evidence they were run under the canonical settings the two "
          "new runs use (--norm-mode per_subject --cv-scheme loso --inner-splits 5 --seed 42).")
    print(f"\n  PHASE 0 GATE: {'PASS' if ok else 'FAIL'}")
    return ok


def collect():
    """returns dict[set][model] -> (subs, vec) for all four sets, or None where a run is incomplete."""
    out = {}
    for s in ORDER:
        out[s] = {}
        for m in ("SVM", "RF"):
            r = sw_vec(s, m)
            if r is not None and len(r[1]) != 40:
                print(f"  [warn] {s} {m}: {len(r[1])}/40 folds -- run incomplete")
                r = None
            out[s][m] = r
    return out


def phase2(data):
    print("\n" + "=" * 78 + "\nPHASE 2  four-way comparison\n" + "=" * 78)
    ready = all(data[s][m] is not None for s in ORDER for m in ("SVM", "RF"))
    if not ready:
        missing = [f"{s}/{m}" for s in ORDER for m in ("SVM", "RF") if data[s][m] is None]
        print(f"  Phase 1 runs not complete: missing {missing}. Re-run this script when they finish.")
        return None

    # subjectwise wide
    rows = []
    for s in ORDER:
        subs = data[s]["SVM"][0]
        for i, subj in enumerate(subs):
            rows.append(dict(feature_set=s, subject=int(subj),
                             SVM=data[s]["SVM"][1][i], RF=data[s]["RF"][1][i]))
    sw = pd.DataFrame(rows)
    sw.to_csv(FS / "featureset_loso_subjectwise.csv", index=False)

    # summary table
    srows = []
    for s in ORDER:
        for m in ("SVM", "RF"):
            v = data[s][m][1]
            srows.append(dict(feature_set=s, dim=SETS[s]["dim"], model=m,
                              loso_macro_f1_mean=v.mean(), loso_inter_subject_sd=v.std(ddof=1),
                              subj_dependent_macro_f1=SUBJDEP[s][m],
                              loso_minus_subjdep_pp=(v.mean() - SUBJDEP[s][m]) * 100))
    summ = pd.DataFrame(srows)
    summ.to_csv(FS / "featureset_loso_summary.csv", index=False)
    print(summ.to_string(index=False))

    # paired contrasts within model, Holm within family of 6 per model
    pairs = [("Freq-72", "Base-36"), ("Freq-72", "Extended-54"), ("Freq-72", "Combined-81"),
             ("Combined-81", "Base-36"), ("Combined-81", "Extended-54"), ("Extended-54", "Base-36")]
    wrows = []
    for m in ("SVM", "RF"):
        pvals, recs = [], []
        for a, b in pairs:
            va, vb = data[a][m][1], data[b][m][1]
            d = va - vb
            W, p = wilcoxon(va, vb)
            lo, hi = bca_ci(d)
            recs.append(dict(model=m, contrast=f"{a} vs {b}", mean_delta_pp=d.mean() * 100,
                             cohen_dz=cohen_dz(d), bca_lo_pp=lo * 100, bca_hi_pp=hi * 100,
                             n_better=int((d > 0).sum()), W=float(W), p=float(p)))
            pvals.append(p)
        hp = holm(pvals)
        for r, h in zip(recs, hp):
            r["p_holm"] = float(h); r["sig_holm_0.05"] = "Yes" if h < 0.05 else "No"
            wrows.append(r)
    wil = pd.DataFrame(wrows)
    wil.to_csv(FS / "featureset_loso_wilcoxon.csv", index=False)
    print("\n" + wil.to_string(index=False))

    # ordering statements
    print("\n  ORDERING under each protocol (macro-F1, descending):")
    for m in ("SVM", "RF"):
        lo = sorted(ORDER, key=lambda s: -data[s][m][1].mean())
        sd = sorted(ORDER, key=lambda s: -SUBJDEP[s][m])
        print(f"    {m}  subject-dependent : {' > '.join(sd)}")
        print(f"    {m}  cross-subject LOSO: {' > '.join(lo)}")
    return summ, sw, wil, data


def phase3_fig(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(ORDER)); w = 0.36
    cols = {"SVM": "#e07b39", "RF": "#2e6f9e"}
    for j, m in enumerate(("SVM", "RF")):
        means = [data[s][m][1].mean() for s in ORDER]
        ses = [data[s][m][1].std(ddof=1) / np.sqrt(40) for s in ORDER]
        ax.bar(x + (j - 0.5) * w, means, w, yerr=ses, capsize=3, color=cols[m],
               edgecolor="black", lw=0.3, label=f"{m} LOSO (cross-subject)")
        ax.scatter(x + (j - 0.5) * w, [SUBJDEP[s][m] for s in ORDER], facecolor="none",
                   edgecolor="black", s=55, zorder=4,
                   label="subject-dependent (Section 4.1.1)" if j == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n({SETS[s]['dim']}-dim)" for s in ORDER])
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0.74, 0.90)
    ax.set_title("Feature-set ordering: cross-subject (LOSO) bars vs subject-dependent markers", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGOUT / "featureset_loso.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [save] {FIGOUT / 'featureset_loso.png'}")


def main():
    if not phase0():
        print("\nSTOP after Phase 0.")
        return
    data = collect()
    res = phase2(data)
    if res is None:
        return
    summ, sw, wil, data = res
    phase3_fig(data)

    # ---- outcome verdict ----
    print("\n" + "=" * 78 + "\nOutcome (Section 4.1.1 caveat)\n" + "=" * 78)
    lead = {}
    for m in ("SVM", "RF"):
        means = {s: data[s][m][1].mean() for s in ORDER}
        top = max(means, key=means.get)
        second = sorted(means.values())[-2]
        lead[m] = (top, means[top] - second, means)
    combined_leads = all(lead[m][0] == "Combined-81" for m in ("SVM", "RF"))
    combined_margin = min(lead[m][1] for m in ("SVM", "RF")) if combined_leads else None
    # "clearly" = leads both models AND the Freq-72 vs Combined-81 contrast is Holm-significant for at least one model
    fc_sig = ((wil.contrast == "Freq-72 vs Combined-81") & (wil["sig_holm_0.05"] == "Yes")).any()
    subjdep_ordering_survives = all(
        sorted(ORDER, key=lambda s: -data[s][m][1].mean()) == sorted(ORDER, key=lambda s: -SUBJDEP[s][m])
        for m in ("SVM", "RF"))
    print(f"  Does the subject-dependent ordering survive the cross-subject protocol?  "
          f"{'YES' if subjdep_ordering_survives else 'NO'}")
    if combined_leads and fc_sig and combined_margin and combined_margin > 0.005:
        print(f"  OUTCOME B: Combined-81 leads clearly under LOSO (both models; margin >= "
              f"{combined_margin*100:.2f} pp; Freq-72 vs Combined-81 Holm-significant). "
              f"Per the plan: REPORT AND STOP -- do not re-run the thesis on Combined-81.")
    elif not subjdep_ordering_survives and not (combined_leads and fc_sig):
        # four sets statistically indistinguishable is Outcome A
        any_sig = (wil["sig_holm_0.05"] == "Yes").any()
        if not any_sig:
            print("  OUTCOME A: the four sets are statistically indistinguishable under LOSO. "
                  "Section 4.1.1's caveat closes in the best way -- the discredited-protocol choice did not matter.")
        else:
            print("  Mixed: some contrasts significant but Combined-81 does not lead both models clearly. "
                  "Report the table; reported results stay on Freq-72.")
    else:
        print("  OUTCOME C (or near-A): Freq-72 among the top; the original choice stands. "
              "Reported results stay on Freq-72; only Section 4.1.1's account of how it was chosen changes.")
    n_bh = len(wil)
    print(f"\n  New BH family members added: {n_bh} (6 paired contrasts x 2 models).")


if __name__ == "__main__":
    main()
