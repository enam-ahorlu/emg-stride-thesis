#!/usr/bin/env python3
"""
run_deepcoral_d2_analysis.py
=============================
EXPERIMENT_PLAN_DEEPCORAL.md, D2 AMENDMENT, Phase D2.4. Analyzes the six-point
lambda sweep (0.1, 1[repro], 3, 10, 30, 100) against the two references
(global-norm ~0.772, per-subject normalization 0.839490) and against AdaBN
(0.8182). Reads deep_coral_subjectwise.csv for every arm, never the rounded
deep_coral_summary.csv.

Outputs:
  results_deep_coral_d2/sweep_table.csv
  results_deep_coral_d2/paired_contrasts.csv
  report_figs/new_experiments/deep_coral_lambda.png
"""
from __future__ import annotations
import sys, io, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).parent
D2OUT = ROOT / "results_deep_coral_d2"; D2OUT.mkdir(exist_ok=True)
FIGDIR = ROOT / "report_figs" / "new_experiments"; FIGDIR.mkdir(parents=True, exist_ok=True)

ARMS = {
    0.1: "results_deep_coral_lam0p1",
    1.0: "results_deep_coral_lam1p0_repro",
    3.0: "results_deep_coral_lam3",
    10.0: "results_deep_coral_lam10",
    30.0: "results_deep_coral_lam30",
    100.0: "results_deep_coral_lam100",
}
NEW_LAMBDAS = [0.1, 3.0, 10.0, 30.0, 100.0]   # excludes 1.0, the reproduction/comparator
PERSUBJ_PATH = ROOT / "results_cnn_aug_resnet_se_chandrop" / "cnn_arch_subjectwise.csv"
ADABN_PATH = ROOT / "results_adabn_chandrop" / "adabn_subjectwise.csv"
GLOBAL_NORM_REF = 0.772
PUBLISHED_LAM1 = 0.825712
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


def load_vec(dirname):
    df = pd.read_csv(ROOT / dirname / "deep_coral_subjectwise.csv").sort_values("subject")
    assert df["subject"].nunique() == 40 and len(df) == 40, f"{dirname}: not 40 unique subjects"
    return df.set_index("subject")["f1_macro"]


def main():
    print("=" * 78 + "\nPhase D2.4  sweep table\n" + "=" * 78)
    vecs = {lam: load_vec(d) for lam, d in ARMS.items()}
    persubj = pd.read_csv(PERSUBJ_PATH).sort_values("subject").set_index("subject")["f1_macro"]
    assert persubj.nunique() >= 1 and len(persubj) == 40
    persubj_mean = float(persubj.mean())
    print(f"  per-subject comparator mean (6dp): {persubj_mean:.6f}  "
          f"(source: {PERSUBJ_PATH.parent.name}) {'OK' if abs(persubj_mean-0.839490) < 1e-6 else '*** MISMATCH ***'}")
    if abs(persubj_mean - 0.839490) >= 1e-6:
        print("  *** ABORT: comparator does not equal 0.839490 to six decimals. ***")
        return
    adabn = pd.read_csv(ADABN_PATH).sort_values("subject").set_index("subject")["f1_macro"]

    rows = []
    for lam in sorted(vecs):
        v = vecs[lam]
        rows.append(dict(lambda_=lam, mean=float(v.mean()), sd=float(v.std(ddof=1)), n=len(v)))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(D2OUT / "sweep_table.csv", index=False)
    print(sweep.to_string(index=False))
    print(f"\n  references: global-norm ~{GLOBAL_NORM_REF}, per-subject norm {persubj_mean:.6f}, "
          f"AdaBN {float(adabn.mean()):.6f}")

    best_lam = sweep.loc[sweep["mean"].idxmax(), "lambda_"]
    print(f"\n  best lambda by mean: {best_lam}")

    print("\n" + "=" * 78 + "\nPhase D2.4  paired contrasts (Holm within family of 11)\n" + "=" * 78)
    tests = []
    for lam in NEW_LAMBDAS:
        tests.append((f"lambda={lam} vs per-subject norm", vecs[lam], persubj))
    for lam in NEW_LAMBDAS:
        tests.append((f"lambda={lam} vs lambda=1", vecs[lam], vecs[1.0]))
    tests.append((f"best lambda={best_lam} vs AdaBN", vecs[best_lam], adabn))

    recs, pvals = [], []
    for label, a_s, b_s in tests:
        common = a_s.index.intersection(b_s.index)
        a, b = a_s.loc[common].to_numpy(), b_s.loc[common].to_numpy()
        d = a - b
        W, p = wilcoxon(a, b, zero_method="wilcox")
        lo, hi = bca_ci(d)
        clears = (np.sign(lo) == np.sign(hi)) and (min(abs(lo), abs(hi)) > NONDET_PP / 100)
        recs.append(dict(contrast=label, mean_delta_pp=float(d.mean() * 100), cohen_dz=cohen_dz(d),
                         bca_lo_pp=lo * 100, bca_hi_pp=hi * 100, n_improved=int((d > 0).sum()),
                         n=len(common), W=float(W), p=float(p),
                         nondet_multiple=(d.mean() * 100) / NONDET_PP,
                         clears_nondet_band=bool(clears)))
        pvals.append(p)
    hp = holm(pvals)
    for r, h in zip(recs, hp):
        r["p_holm"] = float(h); r["sig_holm_0.05"] = "Yes" if h < 0.05 else "No"
    contrasts = pd.DataFrame(recs)
    contrasts.to_csv(D2OUT / "paired_contrasts.csv", index=False)
    print(contrasts.to_string(index=False))

    print("\n" + "=" * 78 + "\nInterior-maximum prediction check\n" + "=" * 78)
    lam1_v = float(sweep[sweep.lambda_ == 1.0]["mean"].iloc[0])
    print(f"  lambda=1 mean = {lam1_v:.6f} (published 0.825712 for context; this run's own repro used above)")
    print(f"  global-norm baseline (not re-run here) ~ {GLOBAL_NORM_REF}")
    rising_to_1 = lam1_v > GLOBAL_NORM_REF
    print(f"  Does macro-F1 rise from global-norm (~0.772) to lambda=1 ({lam1_v:.4f})? {rising_to_1} "
          f"(established structurally: lambda=0 degenerates to the global-norm arm, not re-run as part of this sweep)")
    monotone_fall_above_1 = all(sweep[sweep.lambda_ == l]["mean"].iloc[0] <= lam1_v
                                for l in (3.0, 10.0, 30.0, 100.0))
    print(f"  Does macro-F1 fall monotonically for every lambda > 1 tested? {monotone_fall_above_1}")
    print(f"  Sweep values above lambda=1: " +
          ", ".join(f"lambda={l}: {sweep[sweep.lambda_==l]['mean'].iloc[0]:.6f}" for l in (3.0,10.0,30.0,100.0)))
    print(f"  Peak of the full sweep sits at lambda={best_lam} (mean {sweep['mean'].max():.6f}), "
          f"{'ABOVE' if best_lam > 1.0 else 'AT OR BELOW'} lambda=1.")

    # ---------------- figure ----------------
    fig_lambda_sweep(sweep, persubj_mean, float(adabn.mean()))

    # ---------------- outcome ----------------
    print("\n" + "=" * 78 + "\nOutcome rule (threshold = 0.5 pp band)\n" + "=" * 78)
    best_vs_ps = contrasts[contrasts.contrast == f"lambda={best_lam} vs per-subject norm"].iloc[0]
    delta = best_vs_ps.mean_delta_pp   # best - per-subject
    holm_sig = best_vs_ps["sig_holm_0.05"] == "Yes"
    print(f"  best lambda ({best_lam}) vs per-subject: delta = {delta:+.3f} pp, Holm p = {best_vs_ps.p_holm:.4f}, "
          f"clears 0.5pp band = {best_vs_ps.clears_nondet_band}")
    if delta < -0.5 and holm_sig and best_vs_ps.clears_nondet_band:
        outcome = "A"
        text = "best lambda stays below per-subject normalization by more than 0.5pp with a paired test surviving Holm -- the expected case, claim strengthens to 'at its best setting among those tested'."
    elif abs(delta) <= 0.5 or not best_vs_ps.clears_nondet_band:
        outcome = "B"
        text = "best lambda closes the gap to inside the 0.5pp band -- per-subject normalization and a tuned Deep CORAL are indistinguishable on this backbone. REPORT AND STOP. Do not edit any chapter."
    else:
        outcome = "C"
        text = "best lambda overtakes per-subject normalization by more than 0.5pp with a paired test surviving Holm -- the CNN-side comparison inverts. STOP IMMEDIATELY. Enam's decision, not the runner's."
    print(f"\n  OUTCOME {outcome}: {text}")


def fig_lambda_sweep(sweep, persubj_mean, adabn_mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = sweep["lambda_"].to_numpy()
    y = sweep["mean"].to_numpy()
    yerr = sweep["sd"].to_numpy() / np.sqrt(sweep["n"].to_numpy())
    ax.errorbar(x, y, yerr=yerr, marker="o", ms=8, lw=2, capsize=4, color="#41ab5d",
               label="Deep CORAL (this sweep)", zorder=3)
    ax.set_xscale("log")
    ax.axhline(persubj_mean, ls="--", lw=1.5, color="#2c7fb8", label=f"per-subject norm ({persubj_mean:.4f})")
    ax.axhline(0.772, ls=":", lw=1.5, color="#9e9e9e", label="global-norm baseline (~0.772)")
    ax.axhline(adabn_mean, ls="-.", lw=1.3, color="#c44e52", alpha=0.8, label=f"AdaBN ({adabn_mean:.4f})")
    ax.axvline(1.0, ls=":", lw=1, color="#666", alpha=0.5)
    ax.annotate("published\nfixed setting", xy=(1.0, y[list(x).index(1.0)]), xytext=(1.4, 0.80),
               fontsize=8, color="#444", arrowprops=dict(arrowstyle="->", color="#666", lw=0.8))
    ax.set_xlabel("coral_lambda (log scale)")
    ax.set_ylabel("LOSO macro-F1")
    ax.set_ylim(0.75, 0.86)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.set_title("Deep CORAL lambda sweep (ResNet-SE+CD, chandrop): is the published\n"
                 "fixed setting near an interior maximum?", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "deep_coral_lambda.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [save] {FIGDIR / 'deep_coral_lambda.png'}")


if __name__ == "__main__":
    main()
