# make_new_result_figs.py
# Figures for the results added after 28 August 2026. Thesis plot theme:
# same rcParams and palette as make_deeptier_figs_cd.py / make_frontier_figs_cd.py.
from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT  = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
ORANGE, LORANGE = "#e07b39", "#f0a860"      # SVM, RF
BLUE,  LBLUE    = "#2e6f9e", "#8fb8de"      # ResNet-SE / primary, CNN / secondary
GREEN, GREY     = "#3a923a", "#9e9e9e"      # improvement, baseline
LGREY, RED      = "#bbbbbb", "#c44e52"
INK,   MUTED    = "#222222", "#666666"

def tidy(ax):
    ax.grid(axis="y", alpha=0.25, lw=0.7); ax.set_axisbelow(True)

# ---------------------------------------------------------------- Fig 1: P-10
p10 = json.load(open(ROOT / "results_locus" / "p10_outcome.json"))
cur = pd.DataFrame(p10["curve"]); n = p10["n"]
noaug = round(cur.loc[cur.sd == 1.0, "mean_f1"].iloc[0] - cur.loc[cur.sd == 1.0, "vs_noaug_pp"].iloc[0] / 100, 4)
sem = cur.sd_f1 / np.sqrt(n)
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axvspan(0.38, 0.62, color=LBLUE, alpha=0.16, zorder=0)
ax.axhline(noaug, color=GREY, ls="--", lw=1.1, zorder=1)
ax.annotate(f"no augmentation ({noaug*100:.1f}%)", xy=(0.385, noaug + 0.004),
            fontsize=8.5, color=MUTED, ha="left", va="bottom")
ax.errorbar(cur.sd, cur.mean_f1, yerr=sem, color=BLUE, lw=1.8, marker="o", ms=6,
            capsize=3, elinewidth=0.9, zorder=3, label="mean-preserving channel dropout")
pk = cur.mean_f1.idxmax()
ax.scatter([cur.sd[pk]], [cur.mean_f1[pk]], s=120, facecolor="none", edgecolor=GREEN, lw=1.8, zorder=4)
ax.annotate("peak", xy=(cur.sd[pk] + 0.022, cur.mean_f1[pk] + 0.010),
            ha="left", fontsize=9, color=GREEN)
last = cur.iloc[-1]
ax.scatter([last.sd], [last.mean_f1], s=52, color=RED, zorder=5)
for sd_, dpp, ph, dx, ha_ in [(0.8, 4.63, "3.9e-09", -0.018, "right"), (1.0, 10.65, "1.1e-11", -0.022, "right")]:
    y = float(cur.loc[cur.sd == sd_, "mean_f1"].iloc[0])
    ax.annotate(f"-{dpp:.2f} pp\nHolm p = {ph}", xy=(sd_ + dx, y + 0.004),
                ha=ha_, va="bottom", fontsize=8.2, color=RED)
ax.text(0.50, 0.8665, "operating range", ha="center", fontsize=8.5, color=MUTED)
ax.set_xlabel("perturbation standard deviation (mean-preserving, p' = SD$^2$/(1+SD$^2$))")
ax.set_ylabel("LOSO macro-F1"); ax.set_xticks(list(cur.sd)); ax.set_ylim(0.705, 0.872)
ax.set_title("Over-invariance boundary on the mean-preserving family", fontsize=11)
tidy(ax); fig.tight_layout()
fig.savefig(OUT / "p10_dose_response.png", bbox_inches="tight"); plt.close()
print("saved p10_dose_response.png   baseline =", noaug)

# ----------------------------------------------------------------- Fig 2: S-1
s1 = json.load(open(ROOT / "results_locus" / "s1_outcome.json"))
CLS = ["STDUP", "UPS", "WAK", "DNS"]
pub = {"SVM": {"STDUP": .9607, "UPS": .7628, "WAK": .7023, "DNS": .6768},
       "RF":  {"STDUP": .9595, "UPS": .7540, "WAK": .7094, "DNS": .6619},
       "ResNet-SE+CD": {"STDUP": .971, "UPS": .800, "WAK": .778, "DNS": .800}}
act = {"SVM": s1["per_class_f1_active_only"]["SVM"],
       "RF":  s1["per_class_f1_active_only"]["RF"],
       "ResNet-SE+CD": s1["deep_arm_per_class_f1"]}
cols = {"SVM": ORANGE, "RF": LORANGE, "ResNet-SE+CD": BLUE}
fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3), sharey=True)
x = np.arange(len(CLS)); w = 0.38
for ax, m in zip(axes, ["SVM", "RF", "ResNet-SE+CD"]):
    p = [pub[m][c] for c in CLS]; a = [act[m][c] for c in CLS]
    ax.bar(x - w/2, p, w, color=LGREY, edgecolor="black", lw=0.3, label="published (STDUP includes rest)")
    ax.bar(x + w/2, a, w, color=cols[m], edgecolor="black", lw=0.3, label="active-only STDUP")
    for i in range(len(CLS)):
        d = (a[i] - p[i]) * 100
        lbl = "0" if abs(d) < 0.5 else f"{d:+.0f}"
        ax.annotate(lbl, xy=(i + w/2, a[i] + 0.012), ha="center", fontsize=8.4,
                    color=RED if d < -1 else GREEN)
    ax.set_xticks(x); ax.set_xticklabels(CLS, fontsize=9.5); ax.set_title(m, fontsize=10.5); tidy(ax)
axes[0].set_ylabel("per-class LOSO F1"); axes[0].set_ylim(0.55, 1.03)
axes[0].legend(loc="lower left", fontsize=8, frameon=False)
fig.suptitle("STDUP loses its margin but keeps its rank when the rest windows are removed", fontsize=11.5)
fig.tight_layout(); fig.savefig(OUT / "s1_active_only_per_class.png", bbox_inches="tight"); plt.close()
print("saved s1_active_only_per_class.png")
