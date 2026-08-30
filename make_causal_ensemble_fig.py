#!/usr/bin/env python3
"""
make_causal_ensemble_fig.py
=============================
E3 (triage T2) causal-ensemble figure -- missing entirely until now;
causal_retention.png (12 July) only covers the older classical SVM/RF study.
Reads results_causal_ensemble/report.csv (soft ensemble, SVM solo, ResNet-SE+CD
solo, all buffer-excluded) and plots the deployable retention curve against:
  - the transductive (offline) upper bound, 0.8579
  - causal AdaBN calib100 (0.7679, deep single-model alternative)
  - classical causal SVM calib100 (0.748, prior study's classical figure --
    NOTE this is the decision_function-route SVM number; the causal-ensemble
    leg's own SVM-solo series uses predict_proba() for the soft vote, which
    diagnose_svm_proba_vs_predict.py showed lands lower for the same reason
    -- both are plotted as what they actually are, not reconciled here).

The point the figure must carry: at every causal buffer size tested, the
ensemble retains more of the offline upper bound than either of its own
members, or any single-model alternative (classical or deep).

Layout notes (this went through two broken iterations before landing here):
  - reference values are NOT put next to their reference line as inline text
    -- with 3 series x 3 K-values packed into a ~0.15 F1 range, any inline
    label long enough to be readable sweeps far enough horizontally to hit
    something else. They are pulled into ONE axes-fraction-anchored box
    (immune to data-coordinate collisions, same fix used for
    alignment_ladder.png) with the line style explained alongside each value.
  - per-point value labels are shown ONLY at K=100 (rightmost, most spread
    out) rather than at all three K, for the same reason.

Style matched to make_frontier_figs_cd.py (colors, spines, dpi). CPU only,
no retraining -- reads report.csv directly.

Output: report_figs/new_experiments/causal_ensemble_retention.png
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
C = {"SVM": "#e07b39", "ResNet-SE+CD": "#2e6f9e", "Ensemble": "#3a923a"}
REF_COLOR = "#777777"

TRANSDUCTIVE_UPPER_BOUND = 0.8579          # SVM+ResNet-SE+CD soft, offline (results_ensemble_v2_chandrop)
ADABN_CALIB100 = 0.7679                    # causal AdaBN, CD backbone (results_adabn_causal_chandrop)
CLASSICAL_SVM_CALIB100 = 0.748             # prior streaming-norm study, decision_function route

report = pd.read_csv(ROOT / "results_causal_ensemble" / "report.csv")
Ks = [25, 50, 100]
series = {
    "Ensemble (SVM+ResNet-SE+CD soft)": ("soft", C["Ensemble"], "o"),
    "ResNet-SE+CD solo": ("RESNET_SE", C["ResNet-SE+CD"], "s"),
    "SVM solo": ("SVM", C["SVM"], "^"),
}

fig, ax = plt.subplots(figsize=(8.2, 5.4))

k100_vals = {}
for label, (model_key, color, marker) in series.items():
    sub = report[report.model == model_key].set_index("config")
    y = [sub.loc[f"calib{k}", "f1_excl_mean"] for k in Ks]
    yerr = [sub.loc[f"calib{k}", "f1_excl_sd"] for k in Ks]
    ax.errorbar(Ks, y, yerr=yerr, marker=marker, ms=8, lw=2, capsize=3,
               color=color, label=label, zorder=3)
    k100_vals[label] = y[-1]
    ax.text(Ks[-1] + 4, y[-1], f"{y[-1]:.3f}", ha="left", va="center", fontsize=9, color=color, fontweight="bold")

# three reference lines -- undecorated (no inline text, see module docstring)
ax.axhline(TRANSDUCTIVE_UPPER_BOUND, ls="--", lw=1.3, color=C["Ensemble"], alpha=0.55, zorder=1)
ax.axhline(ADABN_CALIB100, ls=":", lw=1.6, color=REF_COLOR, alpha=0.9, zorder=1)
ax.axhline(CLASSICAL_SVM_CALIB100, ls="-.", lw=1.3, color=REF_COLOR, alpha=0.9, zorder=1)

# single fixed-position reference box, stacked BELOW the axes (not in any
# corner of the plot itself) -- guarantees no collision with plotted data
# regardless of where the curves/error bars happen to land.
ref_text = (
    "Reference (dashed lines):\n"
    f"  - -   transductive upper bound        {TRANSDUCTIVE_UPPER_BOUND:.3f}\n"
    f"  ...   causal AdaBN, calib100          {ADABN_CALIB100:.3f}\n"
    f"  -.-   classical causal SVM, calib100  {CLASSICAL_SVM_CALIB100:.3f}"
)

ax.set_xlim(15, 118)
ax.set_xticks(Ks)
ax.set_xlabel("causal calibration buffer size K (windows), buffer-excluded scoring")
ax.set_ylabel("LOSO macro-F1")
ax.set_ylim(0.655, 0.895)
ax.set_title("Deployable retention under the causal constraint:\nthe ensemble beats every single-model alternative at every K",
             fontsize=10.5)

# legend and reference text both live BELOW the axes, stacked with a clear
# gap, in the SAME (axes-fraction) coordinate system so their relative
# spacing is predictable -- mixing axes-fraction and figure-fraction here
# previously let the two land on top of each other.
ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
ax.text(0.5, -0.46, ref_text, transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="#444",
        family="monospace", multialignment="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#999", lw=0.5))

fig.savefig(OUT / "causal_ensemble_retention.png", bbox_inches="tight")
plt.close(fig)
print(f"[save] {OUT / 'causal_ensemble_retention.png'}")
