#!/usr/bin/env python3
"""
window_ablation_stats.py
========================
Statistics and verdict for the window-length ablation (W-1).

Encodes the pre-registered decision rule of EXPERIMENT_PLAN_WINDOW_LENGTH.md
section 1.3 in code, so the verdict cannot drift once the numbers are in.

Conventions follow the thesis (section 3.6): paired Wilcoxon signed-rank across
the 40 subjects, paired Cohen's d, BCa bootstrap 95% CI with 10,000 resamples,
Holm correction applied WITHIN each family and never across the two.

Usage:
  python window_ablation_stats.py --out results_window_ablation
  python window_ablation_stats.py --out results_window_ablation --skip-gates   # diagnostics only

Reads only. Writes only into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
SEED = 42
N_BOOT = 10_000

# --------------------------------------------------------------------------
# Arms. Each maps to a directory and the column layout its driver writes.
# 250 ms arms are the published runs and are reused, never re-run.
# --------------------------------------------------------------------------
CLASSICAL_ARMS = {
    # (window, model, norm): (results dir, expected published F1 or None)
    (150, "SVM", "global"):      ("results_win150_global",  None),
    (150, "SVM", "per_subject"): ("results_win150_persubj", None),
    (250, "SVM", "global"):      ("results_loso_freq",         0.708),
    (250, "SVM", "per_subject"): ("results_loso_freq_persubj", 0.777),
    (400, "SVM", "global"):      ("results_win400_global",  None),
    (400, "SVM", "per_subject"): ("results_win400_persubj", None),
}

CNN_ARMS = {
    (150, "ResNet-SE+CD", "global"):      ("results_win150_cnn_global",  None),
    (150, "ResNet-SE+CD", "per_subject"): ("results_win150_cnn_persubj", None),
    # W-1 Stage 0 fix (31 Aug 2026): the plan's original dirs held SimpleEMGCNN
    # runs (0.757 / 0.681), not ResNet-SE+CD. Per-subject 250 ms repointed to the
    # deep model of record (reproduces 0.8395). No stored ResNet-SE+CD global-norm
    # 250 ms run existed, so it is generated fresh into results_win250_cnn_global
    # and carries no published gate value.
    (250, "ResNet-SE+CD", "global"):      ("results_win250_cnn_global",          None),
    (250, "ResNet-SE+CD", "per_subject"): ("results_cnn_aug_resnet_se_chandrop",  0.840),
    (400, "ResNet-SE+CD", "global"):      ("results_win400_cnn_global",  None),
    (400, "ResNet-SE+CD", "per_subject"): ("results_win400_cnn_persubj", None),
}

GATE_TOL = 0.002


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _read_subjectwise(d: Path, model: str = "") -> pd.DataFrame:
    """Find the per-subject CSV a driver wrote, whatever it called it.

    `model` disambiguates dirs that hold several models' files: results_loso_freq/
    and results_loso_freq_persubj/ each carry both an __RF_ and an __SVM_
    subjectwise CSV, and a bare sorted() glob returns RF first. When the requested
    model's token appears in some filename, keep only those files.
    """
    mtok = model.split("-")[0].upper()  # "SVM"; "RESNET" for the CNN arms
    for pat in ("*subjectwise*.csv", "per_subject_metrics*.csv", "*_subjectwise_ckpt.csv"):
        hits = sorted(d.glob(pat)) or sorted(d.glob("checkpoints/" + pat))
        if mtok and any(mtok in h.name.upper() for h in hits):
            hits = [h for h in hits if mtok in h.name.upper()]
        for h in hits:
            df = pd.read_csv(h)
            cols = {c.lower(): c for c in df.columns}
            subj = cols.get("heldout_subject") or cols.get("subject")
            f1 = cols.get("f1_macro")
            if subj and f1:
                return df.rename(columns={subj: "subject", f1: "f1_macro"})
    raise FileNotFoundError(f"no per-subject results with subject and f1_macro columns under {d}")


def load_arm(dirname: str, model: str) -> np.ndarray:
    """Return f1_macro indexed by subject 1..40, as a length-40 array."""
    d = ROOT / dirname
    if not d.is_dir():
        raise FileNotFoundError(f"missing results directory: {d}")
    df = _read_subjectwise(d, model)
    if "model" in df.columns and df["model"].nunique() > 1:
        df = df[df["model"].astype(str).str.upper().str.startswith(model.split("-")[0].upper())]
    df = df[["subject", "f1_macro"]].dropna()

    dupes = df["subject"].duplicated().sum()
    if dupes:
        raise ValueError(
            f"{dirname}: {dupes} duplicated subject rows. A resumed run has double-appended; "
            "this is a real failure mode of append-mode checkpointing. Deduplicate the "
            "checkpoint before trusting any number from it."
        )
    if len(df) != 40:
        raise ValueError(f"{dirname}: expected 40 subjects, found {len(df)}. Run is incomplete.")

    return df.sort_values("subject")["f1_macro"].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def bca_ci(diff: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05,
           seed: int = SEED) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI for the mean of `diff`."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    theta = diff.mean()

    boot = np.array([rng.choice(diff, n, replace=True).mean() for _ in range(n_boot)])

    prop = np.mean(boot < theta)
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)   # keep z0 finite
    z0 = stats.norm.ppf(prop)

    # jackknife acceleration
    jack = np.array([np.delete(diff, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a_hat * (z0 + z)))

    lo = adj(stats.norm.ppf(alpha / 2))
    hi = adj(stats.norm.ppf(1 - alpha / 2))
    return (float(np.quantile(boot, lo)), float(np.quantile(boot, hi)))


def paired_test(a: np.ndarray, b: np.ndarray, label: str, family: str) -> dict:
    diff = a - b
    if np.allclose(diff, 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    lo, hi = bca_ci(diff)
    return {
        "family": family,
        "comparison": label,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "delta_pp": float(diff.mean() * 100),
        "p_raw": p,
        "cohens_d": cohens_d_paired(a, b),
        "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
        "n": len(a),
    }


def holm(rows: list[dict]) -> list[dict]:
    """Holm-Bonferroni within a family. Adds p_holm, monotonic and capped at 1."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    m = len(rows)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * rows[i]["p_raw"])
        running = max(running, adj)          # enforce monotonicity
        rows[i]["p_holm"] = running
    return rows


# --------------------------------------------------------------------------
# The pre-registered decision rule (plan section 1.3). Do not edit after seeing data.
# --------------------------------------------------------------------------
def classify(delta_pp: float, p_holm: float, d: float) -> str:
    ad, adl = abs(delta_pp), abs(d)
    if p_holm >= 0.05 or (ad < 1.0 and adl < 0.3):
        return "A"
    if p_holm < 0.05 and ad >= 2.5 and adl >= 0.8:
        return "C"
    return "B"


VERDICT_TEXT = {
    "A": ("No material difference. The 250 ms window stands. Write the ablation up as a "
          "robustness result. No escalation."),
    "B": ("Material but tolerable. The 250 ms window stands, reported with the trade-off "
          "stated honestly. Flag to Enam in the summary, but the plan does not stop here."),
    "C": ("Material and large. STOP. Escalate to Enam and do not begin rewriting anything. "
          "The decision between re-baselining and reframing 250 ms as a latency-constrained "
          "choice is his, not yours."),
}


def main() -> int:
    ap = argparse.ArgumentParser("Window-length ablation statistics and verdict.")
    ap.add_argument("--out", default="results_window_ablation")
    ap.add_argument("--skip-gates", action="store_true",
                    help="Run without the 250 ms reproduction gates. Diagnostics only; "
                         "a verdict produced this way is not reportable.")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    arms, failures = {}, []
    for table in (CLASSICAL_ARMS, CNN_ARMS):
        for key, (dirname, expected) in table.items():
            try:
                arms[key] = load_arm(dirname, key[1])
            except Exception as e:
                failures.append(f"{key}: {e}")

    if failures:
        print("COULD NOT LOAD:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 2

    # ---- gates -----------------------------------------------------------
    print("=== reproduction gates (plan section 6) ===")
    gate_ok = True
    for table in (CLASSICAL_ARMS, CNN_ARMS):
        for key, (dirname, expected) in table.items():
            if expected is None:
                continue
            got = arms[key].mean()
            ok = abs(got - expected) <= GATE_TOL
            gate_ok &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  {key[1]:<13} {key[0]:>3} ms {key[2]:<11} "
                  f"got {got:.4f}, published {expected:.4f}")
    if not gate_ok and not args.skip_gates:
        print("\nA gate failed. Either the wrong directory is being read or the pipeline has "
              "drifted. Every new number is untrustworthy until this is resolved. Stopping.\n"
              "Do not adjust the expected value to match what was found.", file=sys.stderr)
        return 3

    # ---- summary ---------------------------------------------------------
    rows = []
    for (w, model, norm), vals in sorted(arms.items()):
        lo, hi = bca_ci(vals)
        rows.append({"window_ms": w, "model": model, "norm": norm, "n": len(vals),
                     "f1_mean": vals.mean(), "f1_sd": vals.std(ddof=1),
                     "ci_low": lo, "ci_high": hi})
    pd.DataFrame(rows).to_csv(out / "window_ablation_summary.csv", index=False)
    make_figure(rows, out / "fig_window_ablation.png")

    # ---- family 1, primary: the normalization gap at each window ---------
    fam1 = []
    for model in ("SVM", "ResNet-SE+CD"):
        for w in (150, 250, 400):
            fam1.append(paired_test(arms[(w, model, "per_subject")],
                                    arms[(w, model, "global")],
                                    f"{model} @ {w} ms: per-subject vs global", "1-primary"))
    holm(fam1)

    # ---- family 2, secondary: window comparisons under per-subject norm --
    fam2 = []
    for model in ("SVM", "ResNet-SE+CD"):
        for wa, wb in ((150, 250), (400, 250), (150, 400)):
            fam2.append(paired_test(arms[(wa, model, "per_subject")],
                                    arms[(wb, model, "per_subject")],
                                    f"{model}: {wa} ms vs {wb} ms (per-subject)", "2-secondary"))
    holm(fam2)

    pd.DataFrame(fam1 + fam2).to_csv(out / "window_ablation_tests.csv", index=False)

    # ---- verdict ---------------------------------------------------------
    decisive = [r for r in fam2 if r["comparison"].startswith(("SVM: 400", "ResNet-SE+CD: 400"))]
    outcomes = {r["comparison"]: classify(r["delta_pp"], r["p_holm"], r["cohens_d"])
                for r in decisive}
    worst = "C" if "C" in outcomes.values() else ("B" if "B" in outcomes.values() else "A")

    gap_fail = [r for r in fam1 if r["p_holm"] >= 0.05 or r["delta_pp"] <= 0]

    lines = [f"# Window-length ablation: verdict\n",
             f"## Outcome {worst}\n", VERDICT_TEXT[worst], "",
             "## Primary endpoint: does the normalization finding survive?\n"]
    for r in fam1:
        lines.append(f"- {r['comparison']}: {r['delta_pp']:+.2f} pp "
                     f"(Holm p = {r['p_holm']:.4g}, d = {r['cohens_d']:.2f}, "
                     f"95% CI [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] pp)")
    lines.append("")
    if gap_fail:
        lines.append("**The normalization gap does not hold everywhere.** This is a threat to the "
                     "thesis's central claim, not a question about window choice, and it escalates "
                     "regardless of the outcome letter above:")
        for r in gap_fail:
            lines.append(f"  - {r['comparison']}")
    else:
        lines.append("The gap is positive and significant at every window and for both models. "
                     "The thesis's central claim is robust to window length.")

    lines += ["", "## Secondary endpoint: window comparisons under per-subject normalization\n"]
    for r in fam2:
        lines.append(f"- {r['comparison']}: {r['delta_pp']:+.2f} pp "
                     f"(Holm p = {r['p_holm']:.4g}, d = {r['cohens_d']:.2f}, "
                     f"95% CI [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] pp)")
    lines += ["", "## Decision rule applied\n",
              "Per EXPERIMENT_PLAN_WINDOW_LENGTH.md section 1.3, fixed before the runs:", ""]
    for k, v in outcomes.items():
        lines.append(f"- {k} -> outcome {v}")
    if args.skip_gates:
        lines += ["", "**Produced with --skip-gates. Not reportable.**"]

    (out / "window_ablation_verdict.md").write_text("\n".join(lines), encoding="utf8")
    (out / "window_ablation_outcome.json").write_text(
        json.dumps({"outcome": worst, "per_comparison": outcomes,
                    "primary_gap_holds": not gap_fail}, indent=2), encoding="utf8")

    print("\n" + "\n".join(lines))
    print(f"\nwrote {out}/window_ablation_summary.csv, _tests.csv, _verdict.md, _outcome.json")
    return 0


def make_figure(rows, out_path):
    """fig_window_ablation.png -- black and white, square-cornered, no colour, to
    sit beside Figures 3.1 and 5.2. Two panels (SVM | ResNet-SE+CD): LOSO
    macro-F1 against window length, per-subject vs global normalization, with
    95% CI bars. The vertical span between the two lines at each window is the
    primary endpoint. Point sizes are scaled by FIG_W / PAGE_W so the numbers
    in this source are the sizes seen at page width (see make_fig5_2.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
        "text.color": "black", "axes.edgecolor": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })
    FIG_W, FIG_H, PAGE_W = 9.0, 4.1, 6.15
    S = FIG_W / PAGE_W
    def pt(x): return x * S

    wins = [150, 250, 400]
    def series(model, norm):
        d = {r["window_ms"]: r for r in rows if r["model"] == model and r["norm"] == norm}
        y = [d[w]["f1_mean"] for w in wins]
        lo = [d[w]["f1_mean"] - d[w]["ci_low"] for w in wins]
        hi = [d[w]["ci_high"] - d[w]["f1_mean"] for w in wins]
        return y, [lo, hi]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=300, sharey=True)
    for ax, model in zip(axes, ("SVM", "ResNet-SE+CD")):
        yp, ep = series(model, "per_subject")
        yg, eg = series(model, "global")
        ax.errorbar(wins, yp, yerr=ep, color="black", linewidth=1.3, linestyle="-",
                    marker="o", markersize=pt(4), markerfacecolor="black",
                    markeredgecolor="black", capsize=pt(2.5), capthick=1.0, zorder=3,
                    label="per-subject normalization")
        ax.errorbar(wins, yg, yerr=eg, color="black", linewidth=1.3, linestyle="--",
                    marker="s", markersize=pt(4), markerfacecolor="white",
                    markeredgecolor="black", capsize=pt(2.5), capthick=1.0, zorder=3,
                    label="global normalization")
        for w, a, b in zip(wins, yp, yg):                      # gap annotation
            ax.annotate("", xy=(w, a), xytext=(w, b),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=0.9))
            ax.text(w + 6, (a + b) / 2, f"{(a - b) * 100:+.1f} pp",
                    ha="left", va="center", fontsize=pt(6.6))
        ax.set_title(model, fontsize=pt(9), fontweight="bold")
        ax.set_xlabel("analysis window (ms)", fontsize=pt(8))
        ax.set_xticks(wins); ax.set_xticklabels(wins, fontsize=pt(7.5))
        ax.tick_params(axis="y", labelsize=pt(7.5))
        ax.set_xlim(120, 445)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("LOSO macro-F1", fontsize=pt(8))
    axes[0].legend(loc="lower right", fontsize=pt(6.8), frameon=True,
                   edgecolor="black", handlelength=2.6)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
