# make_alignment_ladder_fig.py
# ---------------------------------------------------------------------------
# Figure A.19, regenerated from the MEASURED ladder in
# results_alignment_ladder_loso/alignment_ladder_full.csv. Nothing hardcoded.
#
# The earlier version drew MMD on one axis and put class silhouette AND
# downstream F1 together on a twin axis. Their ranges differ by two orders of
# magnitude, so the silhouette line rendered flat and the peak-then-fall it
# exists to show was invisible. Four panels on a shared x-axis instead: one
# measure per axis.
# ---------------------------------------------------------------------------
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
SRC  = ROOT / "results_alignment_ladder_loso" / "alignment_ladder_full.csv"
OUT  = ROOT / "report_figs" / "new_experiments" / "alignment_ladder.png"

GREEN, GREY, RED, BLUE = "#55A868", "#8C8C8C", "#C44E52", "#2C7FB8"
INK, MUTED = "#222222", "#666666"

df = pd.read_csv(SRC).sort_values("rung").reset_index(drop=True)
names = ["global\nz-score", "mean\ncentering", "scale\nonly",
         "mean + scale\n(per-subject z)", "full\nwhitening"]
x = np.arange(len(df))
peak = int(df["silhouette_by_class"].idxmax())

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#CCCCCC",
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(4, 1, figsize=(8.2, 9.4), sharex=True,
                         gridspec_kw={"height_ratios": [1.15, 1, 1, 1.15], "hspace": 0.28})
a0, a1, a2, a3 = axes

def tidy(ax):
    ax.grid(axis="y", alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.axvline(peak, color=MUTED, ls=":", lw=1.0, alpha=0.55, zorder=0)

a0.bar(x, df["mmd_removed_pct"], color=GREEN, width=0.58, edgecolor="white", linewidth=1.2)
a0.axhline(0, color="#999999", lw=0.9)
a0.set_ylabel("MMD removed (%)")
a0.set_title("Subject discrepancy removed rises with alignment strength", fontsize=10.5, loc="left", pad=8)
tidy(a0)

a1.plot(x, df["subject_probe_bal_acc"], color=GREY, marker="o", ms=8, lw=2)
a1.axhline(0.025, color=MUTED, ls="--", lw=1.0)
a1.annotate("chance floor 0.025", (4.05, 0.025), fontsize=8.5, color=MUTED, va="bottom", ha="right")
a1.set_ylabel("subject probe\n(balanced acc.)")
a1.set_title("Subject identity is lowest at full whitening, where accuracy is worst", fontsize=10.5, loc="left", pad=8)
tidy(a1)

a2.plot(x, df["silhouette_by_class"], color=RED, marker="o", ms=8, lw=2)
a2.axhline(0, color="#999999", lw=0.9)
a2.set_ylabel("class silhouette")
a2.set_ylim(df["silhouette_by_class"].min() - 0.006, df["silhouette_by_class"].max() + 0.009)
a2.set_title("Class separability peaks at per-subject z, then falls back", fontsize=10.5, loc="left", pad=8)
for i in (peak, 4):
    a2.annotate(f"{df['silhouette_by_class'][i]:+.3f}", (x[i], df["silhouette_by_class"][i]),
                textcoords="offset points", xytext=(0, 11 if i == peak else -18),
                ha="center", fontsize=9, color=RED)
tidy(a2)

a3.plot(x, df["f1_macro_mean"], color=BLUE, marker="s", ms=8, lw=2)
a3.set_ylabel("LOSO macro-F1\n(SVM)")
a3.set_title("Downstream accuracy follows separability, not alignment strength", fontsize=10.5, loc="left", pad=8)
for i in (peak, 4):
    a3.annotate(f"{df['f1_macro_mean'][i]:.3f}", (x[i], df["f1_macro_mean"][i]),
                textcoords="offset points", xytext=(0, 11 if i == peak else -19),
                ha="center", fontsize=9, color=BLUE)
a3.set_ylim(df["f1_macro_mean"].min() - 0.035, df["f1_macro_mean"].max() + 0.035)
tidy(a3)

a3.set_xticks(x); a3.set_xticklabels(names, fontsize=9.5, color=INK)
a3.set_xlabel("alignment operator", labelpad=8)
fig.align_ylabels(axes)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"[save] {OUT}")