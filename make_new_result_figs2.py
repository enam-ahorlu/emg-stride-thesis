from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).parent
OUT  = ROOT / "report_figs" / "new_experiments"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
ORANGE, LORANGE = "#e07b39", "#f0a860"
BLUE,  LBLUE    = "#2e6f9e", "#8fb8de"
GREEN, GREY     = "#3a923a", "#9e9e9e"
LGREY, RED      = "#bbbbbb", "#c44e52"
INK,   MUTED    = "#222222", "#666666"
def tidy(ax):
    ax.grid(axis="y", alpha=0.25, lw=0.7); ax.set_axisbelow(True)

# ------------------------------------------------------------------ Fig 3: B8
df = pd.read_csv(ROOT / "results_b8_sd" / "b8_all_configs.csv")
MC = {"SVM": ORANGE, "RF": LORANGE, "LDA": GREY, "CNN": LBLUE}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.2, 4.5),
                               gridspec_kw={"width_ratios": [1.45, 1]})
cfgs = list(dict.fromkeys(df.config)); xs = np.arange(len(cfgs))
mods = ["SVM", "RF", "LDA", "CNN"]; w = 0.2
for j, m in enumerate(mods):
    sub = df[df.model == m]
    if sub.empty: continue
    xv, yv = [], []
    for i, c in enumerate(cfgs):
        r = sub[sub.config == c]
        if not r.empty: xv.append(i + (j - 1.5) * w); yv.append(float(r.delta_pp.iloc[0]))
    axL.bar(xv, yv, w, color=MC[m], edgecolor="black", lw=0.3, label=m)
axL.axhline(0, color=INK, lw=0.8)
axL.set_xticks(xs); axL.set_xticklabels([c.replace("_", "\n") for c in cfgs], fontsize=8.5)
axL.set_ylabel("change in subject-dependent macro-F1 (pp)")
axL.set_title("Cost of blocking the split, by configuration", fontsize=10.5)
axL.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
axL.set_ylim(-7.8, 3.2); tidy(axL)
axL.annotate("every comparison p < $10^{-9}$", xy=(0.015, 0.035), xycoords="axes fraction",
             fontsize=8.5, color=MUTED)

# Right panel: the ranking flip is against the PUBLISHED subject-dependent figures
# (Table 4.7), not against this re-run's own pooled numbers, in which the CNN never led.
# Two harnesses, labelled as such: that is the honest form of the reproduction caveat.
PUB = {"SVM": 87.4, "RF": 84.3, "CNN": 90.4}
blk = {}
for _, r in df[df.config.isin(["freq72_w250", "cnn_w250"])].iterrows():
    if r.model in PUB: blk[r.model] = float(r.sd_new_blocked)
mods_r = ["SVM", "RF", "CNN"]
xr = np.arange(len(mods_r)); wr = 0.38
axR.bar(xr - wr/2, [PUB[m] for m in mods_r], wr, color=LGREY, edgecolor="black", lw=0.3,
        hatch="//", label="as reported (Table 4.7)")
axR.bar(xr + wr/2, [blk[m] for m in mods_r], wr,
        color=[MC[m] for m in mods_r], edgecolor="black", lw=0.3,
        label="common movement-blocked protocol")
for i, m in enumerate(mods_r):
    axR.annotate(f"{PUB[m]:.1f}", xy=(i - wr/2, PUB[m] + 0.35), ha="center", fontsize=8.4, color=MUTED)
    axR.annotate(f"{blk[m]:.1f}", xy=(i + wr/2, blk[m] + 0.35), ha="center", fontsize=8.6, color=INK)
axR.annotate("CNN first", xy=(2 - wr/2, PUB["CNN"] + 1.6), ha="center", fontsize=8.2, color=MUTED)
axR.annotate("CNN last", xy=(2 + wr/2, blk["CNN"] + 1.6), ha="center", fontsize=8.2, color=INK)
axR.set_xticks(xr); axR.set_xticklabels(mods_r, fontsize=9.5)
axR.set_ylim(80, 95.5); axR.set_ylabel("subject-dependent macro-F1 (%)")
axR.set_title("The ranking flips on a common protocol (250 ms)", fontsize=10.5)
axR.legend(fontsize=8.2, frameon=False, loc="upper left"); tidy(axR)
axR.set_xlabel("two harnesses; see the reproduction caveat in Section 4.1.3",
               fontsize=8.2, color=MUTED, labelpad=8)
fig.suptitle("The subject-dependent split: a 50% window overlap inflates every model, and reorders them", fontsize=11.5)
fig.tight_layout(); fig.savefig(OUT / "b8_pooled_vs_blocked.png", bbox_inches="tight"); plt.close()
print("saved b8_pooled_vs_blocked.png  |  blocked:", blk, " published:", PUB)

# ------------------------------------------------------------------ Fig 4: P-9
p9 = json.load(open(ROOT / "results_locus" / "p9_outcome.json"))
cost = p9["summed_cost_pp"]; rf = p9["reduction_factors"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
labs = ["ResNet\n(no SE)", "ResNet-SE"]
noaug = [cost["A1"], cost["A3"]]; cd = [cost["A2"], cost["A4"]]
x = np.arange(2); w = 0.36
a1.bar(x - w/2, noaug, w, color=LGREY, edgecolor="black", lw=0.3, label="no augmentation")
a1.bar(x + w/2, cd, w, color=BLUE, edgecolor="black", lw=0.3, label="channel dropout")
for i in range(2):
    a1.annotate(f"{noaug[i]:.1f}", xy=(i - w/2, noaug[i] + 1.6), ha="center", fontsize=8.6)
    a1.annotate(f"{cd[i]:.1f}", xy=(i + w/2, cd[i] + 1.6), ha="center", fontsize=8.6, color=BLUE)
a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=9.5)
a1.set_ylabel("summed single-electrode occlusion cost (pp)")
a1.set_title("Electrode reliance, both backbones", fontsize=10.5); a1.set_ylim(0, 112)
a1.legend(fontsize=8.5, frameon=False, loc="upper center"); tidy(a1)

ests = [rf["resnet_A1_A2"]["est"], rf["resnet_se_A3_A4"]["est"]]
los  = [rf["resnet_A1_A2"]["ci"][0], rf["resnet_se_A3_A4"]["ci"][0]]
his  = [rf["resnet_A1_A2"]["ci"][1], rf["resnet_se_A3_A4"]["ci"][1]]
a2.errorbar(ests, [1, 0], xerr=[np.subtract(ests, los), np.subtract(his, ests)], fmt="o",
            ms=8, color=BLUE, capsize=4, elinewidth=1.2, zorder=3)
a2.axvline(rf["original"], color=GREY, ls="--", lw=1.1, zorder=1)
a2.annotate(f"as published ({rf['original']:.2f}x)", xy=(rf["original"], 1.42),
            fontsize=8.5, color=MUTED, ha="center")
for y, e in zip([1, 0], ests):
    a2.annotate(f"{e:.2f}x", xy=(e, y + 0.13), ha="center", fontsize=9, color=BLUE)
a2.set_yticks([1, 0]); a2.set_yticklabels(labs, fontsize=9.5); a2.set_ylim(-0.5, 1.7)
a2.set_xlabel("occlusion-reduction factor (95% BCa)")
a2.set_title("Indistinguishable between backbones", fontsize=10.5)
a2.annotate("paired difference p = 0.35", xy=(0.03, 0.06), xycoords="axes fraction",
            fontsize=8.5, color=MUTED)
a2.grid(axis="x", alpha=0.25, lw=0.7); a2.set_axisbelow(True)
fig.suptitle("Channel dropout flattens electrode reliance by the same factor on either backbone", fontsize=11.5)
fig.tight_layout(); fig.savefig(OUT / "p9_backbone_reduction.png", bbox_inches="tight"); plt.close()
print("saved p9_backbone_reduction.png")
