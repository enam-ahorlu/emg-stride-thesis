from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).parent
OUT  = ROOT / "report_figs" / "new_experiments"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
BLUE, LBLUE = "#2e6f9e", "#8fb8de"
GREEN, GREY = "#3a923a", "#9e9e9e"
LGREY, RED  = "#bbbbbb", "#c44e52"
INK, MUTED  = "#222222", "#666666"

# ------------------------------------------------------------------ Fig 5: B3
d = pd.read_csv(ROOT / "results_locus" / "b3_per_subject_silhouette.csv")
sil = d[[f"sil_rung{i}" for i in range(5)]].to_numpy()
names = ["global\nz-score", "mean\ncentering", "scale\nonly",
         "mean + scale\n(per-subject z)", "full\nwhitening"]
x = np.arange(5)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})
for row in sil:
    a1.plot(x, row, color=LBLUE, alpha=0.30, lw=0.8, zorder=1)
a1.plot(x, sil.mean(0), color=BLUE, lw=2.4, marker="o", ms=6, zorder=3, label="mean of 40 subjects")
a1.axvspan(1.6, 3.4, color=GREEN, alpha=0.09, zorder=0)
a1.set_xticks(x); a1.set_xticklabels(names, fontsize=8.6)
a1.set_ylabel("within-subject class silhouette")
a1.set_title("Per-subject class separability along the alignment ladder", fontsize=10.5)
a1.legend(fontsize=8.5, frameon=False, loc="lower left")
hi = sil.max()
a1.set_ylim(sil.min() - 0.012, hi + 0.040)
a1.annotate("centering is invisible\nwithin a subject", xy=(2.5, hi + 0.030),
            ha="center", va="top", fontsize=8.2, color=MUTED)
a1.grid(axis="y", alpha=0.25, lw=0.7); a1.set_axisbelow(True)

d34 = sil[:, 3] - sil[:, 4]; d30 = sil[:, 3] - sil[:, 0]
rng = np.random.default_rng(42)
for i, (vals, lbl, p, dd) in enumerate([(d34, "vs full whitening", "1.8e-12", "1.91"),
                                        (d30, "vs global z-score", "9.6e-6", "0.78")]):
    jit = rng.normal(0, 0.055, len(vals))
    a2.scatter(vals, np.full(len(vals), i) + jit, s=22, color=BLUE, alpha=0.55,
               edgecolor="none", zorder=3)
    m = vals.mean()
    a2.plot([m, m], [i - 0.22, i + 0.22], color=INK, lw=2.2, zorder=4)
    a2.annotate(f"mean {m:+.3f}   p = {p}, d = {dd}   {(vals>0).sum()}/40 positive",
                xy=(m, i + 0.36), ha="center", fontsize=8.2, color=INK)
a2.axvline(0, color=RED, ls="--", lw=1.1, zorder=1)
a2.set_yticks([0, 1]); a2.set_yticklabels(["vs full\nwhitening", "vs global\nz-score"], fontsize=9)
a2.set_ylim(-0.5, 1.72); a2.set_xlabel("silhouette advantage of the mean + scale rung")
a2.set_title("Paired across 40 subjects", fontsize=10.5)
a2.grid(axis="x", alpha=0.25, lw=0.7); a2.set_axisbelow(True)
fig.suptitle("The rung-3 peak holds within subjects, not only across five pooled points", fontsize=11.5)
fig.tight_layout(); fig.savefig(OUT / "b3_per_subject_silhouette.png", bbox_inches="tight"); plt.close()
print("saved b3_per_subject_silhouette.png")
print("mean sil by rung:", np.round(sil.mean(0), 4))
