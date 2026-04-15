#!/usr/bin/env python3
"""
generate_subject_line_plots.py
===============================
Per-subject trajectory line plots across 4 optimisation stages.

Outputs:
  report_figs/subject_lines_svm_rf.png         (Plot A)
  report_figs/subject_lines_cnn.png             (Plot B)
  report_figs/subject_lines_all_models.png      (Plot C)
  report_figs/subject_improvement_heatmap.png   (Plot D)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ROOT    = Path(__file__).parent
RPT_DIR = ROOT / "report_figs"
RPT_DIR.mkdir(exist_ok=True)

# ============================================================
# Helpers (same as generate_ci_plots.py)
# ============================================================

def _get_subj_col(df):
    return "heldout_subject" if "heldout_subject" in df.columns else "subject"


def load_classical_subjectwise(results_dir, model):
    rdir = ROOT / results_dir
    files = sorted(rdir.glob(f"*{model}*subjectwise*.csv"))
    if not files:
        ckpt = rdir / "checkpoints"
        if ckpt.exists():
            files = sorted(ckpt.glob(f"*{model}*ckpt.csv"))
    if not files:
        raise FileNotFoundError(f"No {model} subjectwise CSV in {rdir}")
    df = pd.read_csv(files[0])
    sc = _get_subj_col(df)
    df = df.drop_duplicates(subset=[sc]).sort_values(sc)
    return (np.array(df["f1_macro"].tolist()),
            df[sc].astype(int).tolist())


def load_cnn_subjectwise(results_dir):
    path = ROOT / results_dir / "per_subject_metrics_cnn_loso.csv"
    df = pd.read_csv(path)
    sc = _get_subj_col(df)
    df = df.drop_duplicates(subset=[sc]).sort_values(sc)
    return (np.array(df["f1_macro"].tolist()),
            df[sc].astype(int).tolist())


def align_to_subjects(arr, subj_list, target_subjects):
    mapping = {s: v for s, v in zip(subj_list, arr)}
    return np.array([mapping[s] for s in target_subjects])


# ============================================================
# Step 1: Load all stages
# ============================================================
print("=" * 60)
print("LOADING PER-SUBJECT F1 FOR ALL STAGES")
print("=" * 60)

svm_base_arr, svm_base_subj = load_classical_subjectwise("results_loso_freq", "SVM")
rf_base_arr,  rf_base_subj  = load_classical_subjectwise("results_loso_freq", "RF")
cnn_base_arr, cnn_base_subj = load_cnn_subjectwise("results_cnn_loso")

all_subjects = sorted(
    set(svm_base_subj) & set(rf_base_subj) & set(cnn_base_subj)
)
n_subj = len(all_subjects)
print(f"  Subjects: {n_subj}")

svm_base = align_to_subjects(svm_base_arr, svm_base_subj, all_subjects)
rf_base  = align_to_subjects(rf_base_arr,  rf_base_subj,  all_subjects)
cnn_base = align_to_subjects(cnn_base_arr, cnn_base_subj, all_subjects)

svm_norm_arr, svm_norm_subj = load_classical_subjectwise("results_loso_freq_persubj", "SVM")
rf_norm_arr,  rf_norm_subj  = load_classical_subjectwise("results_loso_freq_persubj", "RF")
cnn_norm_arr, cnn_norm_subj = load_cnn_subjectwise("results_cnn_loso_norm_persubj")

svm_norm = align_to_subjects(svm_norm_arr, svm_norm_subj, all_subjects)
rf_norm  = align_to_subjects(rf_norm_arr,  rf_norm_subj,  all_subjects)
cnn_norm = align_to_subjects(cnn_norm_arr, cnn_norm_subj, all_subjects)

svm_fs_arr, svm_fs_subj = load_classical_subjectwise("results_loso_freq_rfe36", "SVM")
svm_fs = align_to_subjects(svm_fs_arr, svm_fs_subj, all_subjects)

rf_fs  = rf_norm.copy()

cnn_aug_arr, cnn_aug_subj = load_cnn_subjectwise("results_cnn_loso_aug_gaussian")
cnn_aug = align_to_subjects(cnn_aug_arr, cnn_aug_subj, all_subjects)

ens_df = pd.read_csv(RPT_DIR / "ensemble_3way_per_subject.csv")
sc_ens = _get_subj_col(ens_df)
ens_df = ens_df.drop_duplicates(subset=[sc_ens]).sort_values(sc_ens)
ens_f1 = align_to_subjects(
    np.array(ens_df["f1_macro"].tolist()),
    ens_df[sc_ens].astype(int).tolist(),
    all_subjects
)

# Build subject x stage matrix for each model: shape (n_subj, 4)
stage_labels = ["Baseline", "Norm\n(per-subj)", "Feat Sel/Aug\n(+per-subj)", "Ensemble\n(3-way)"]
stage_keys   = ["Baseline", "Norm", "FeatSel_Aug", "Ensemble"]

data_mat = {
    "SVM": np.column_stack([svm_base, svm_norm, svm_fs, ens_f1]),
    "RF":  np.column_stack([rf_base,  rf_norm,  rf_fs,  ens_f1]),
    "CNN": np.column_stack([cnn_base, cnn_norm,  cnn_aug, ens_f1]),
}

# For improvement highlight: compare stage 0 (Baseline) vs stage 2 (Feat Sel/Aug)
for model, mat in data_mat.items():
    delta = mat[:, 2] - mat[:, 0]
    n_imp = (delta > 0).sum()
    n_deg = (delta < 0).sum()
    print(f"  {model}: {n_imp} subjects improve, {n_deg} degrade (baseline -> feat-sel/aug)")


# ============================================================
# Helper: draw a single subplot with per-subject trajectories
# ============================================================
colors_model = {"SVM": "#2196F3", "RF": "#4CAF50", "CNN": "#FF9800"}

def draw_trajectory_ax(ax, mat, model, stage_labels, title):
    """
    mat: shape (n_subj, n_stages)
    Draws thin grey lines per subject, highlighted green/red, thick mean line.
    """
    n_stages = mat.shape[1]
    x = np.arange(n_stages)

    # Highlight: improvement = baseline->combined (col 2)
    delta = mat[:, 2] - mat[:, 0]

    for i in range(mat.shape[0]):
        if delta[i] > 0:
            lc = "#A5D6A7"   # light green
            lw = 0.9
            zorder = 2
        elif delta[i] < 0:
            lc = "#EF9A9A"   # light red
            lw = 0.9
            zorder = 2
        else:
            lc = "#BDBDBD"
            lw = 0.7
            zorder = 1
        ax.plot(x, mat[i], color=lc, linewidth=lw, alpha=0.85, zorder=zorder)

    mean_line = mat.mean(axis=0)
    ax.plot(x, mean_line, color=colors_model[model], linewidth=3.0, zorder=5, label="Mean")

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=9)
    ax.set_ylabel("F1 Macro", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0.30, 1.02)
    ax.grid(axis="y", alpha=0.35)

    # Legend for highlights
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=colors_model[model], linewidth=2.5, label=f"Mean ({model})"),
        Line2D([0], [0], color="#4CAF50", linewidth=1.2, label="Improves (Baseline→FeatSel)"),
        Line2D([0], [0], color="#F44336", linewidth=1.2, label="Degrades"),
        Line2D([0], [0], color="#9E9E9E", linewidth=0.8, label="No change"),
    ]
    ax.legend(handles=legend_elems, fontsize=7.5, loc="lower right")


# ============================================================
# Plot A: SVM (left) + RF (right)
# ============================================================
print("\n" + "=" * 60)
print("PLOT A: subject_lines_svm_rf.png")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
draw_trajectory_ax(axes[0], data_mat["SVM"], "SVM", stage_labels, "SVM — Per-subject F1 Trajectory")
draw_trajectory_ax(axes[1], data_mat["RF"],  "RF",  stage_labels, "RF — Per-subject F1 Trajectory")
axes[1].set_ylabel("")
fig.suptitle("Per-subject F1 Across Optimisation Stages (n=40)", fontsize=13, fontweight="bold")
plt.tight_layout()
out_a = RPT_DIR / "subject_lines_svm_rf.png"
plt.savefig(out_a, dpi=300)
plt.close()
print(f"  [saved] {out_a.name}")

# ============================================================
# Plot B: CNN only
# ============================================================
print("\n" + "=" * 60)
print("PLOT B: subject_lines_cnn.png")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 6))
draw_trajectory_ax(ax, data_mat["CNN"], "CNN", stage_labels, "CNN — Per-subject F1 Trajectory")
fig.suptitle("CNN: Per-subject F1 Across Optimisation Stages (n=40)", fontsize=13, fontweight="bold")
plt.tight_layout()
out_b = RPT_DIR / "subject_lines_cnn.png"
plt.savefig(out_b, dpi=300)
plt.close()
print(f"  [saved] {out_b.name}")

# ============================================================
# Plot C: SVM + RF + CNN in a row, shared y-axis
# ============================================================
print("\n" + "=" * 60)
print("PLOT C: subject_lines_all_models.png")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, model in zip(axes, ["SVM", "RF", "CNN"]):
    draw_trajectory_ax(ax, data_mat[model], model, stage_labels,
                       f"{model} — Per-subject F1 Trajectory")
for ax in axes[1:]:
    ax.set_ylabel("")
fig.suptitle("Per-subject F1 Trajectories Across Optimisation Stages (n=40 subjects)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out_c = RPT_DIR / "subject_lines_all_models.png"
plt.savefig(out_c, dpi=300)
plt.close()
print(f"  [saved] {out_c.name}")

# ============================================================
# Plot D: Heatmap — delta F1 (combined - baseline) per subject x model
# ============================================================
print("\n" + "=" * 60)
print("PLOT D: subject_improvement_heatmap.png")
print("=" * 60)

# 'combined' = FeatSel/Aug stage (col 2)
delta_mat = np.column_stack([
    data_mat["SVM"][:, 2] - data_mat["SVM"][:, 0],
    data_mat["RF"][:,  2] - data_mat["RF"][:,  0],
    data_mat["CNN"][:, 2] - data_mat["CNN"][:, 0],
])  # shape (40, 3)

# Sort by mean delta across models
sort_idx = np.argsort(delta_mat.mean(axis=1))
delta_sorted   = delta_mat[sort_idx]
subjects_sorted = np.array(all_subjects)[sort_idx]

# Print summary
print(f"\n  Delta F1 (FeatSel/Aug - Baseline):")
for mi, model in enumerate(["SVM", "RF", "CNN"]):
    vals = delta_mat[:, mi]
    print(f"    {model}: mean={vals.mean():+.4f}, min={vals.min():+.4f}, max={vals.max():+.4f}")

# Diverging colormap centred at 0
vmax = max(abs(delta_sorted).max(), 0.05)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, ax = plt.subplots(figsize=(6, 14))
im = ax.imshow(delta_sorted, aspect="auto", cmap="RdYlGn",
               norm=norm, interpolation="nearest")

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["SVM", "RF", "CNN"], fontsize=12, fontweight="bold")
ax.set_yticks(np.arange(n_subj))
ax.set_yticklabels([f"S{s:02d}" for s in subjects_sorted], fontsize=7)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Subject (sorted by mean delta)", fontsize=11)
ax.set_title("Delta F1: FeatSel/Aug vs Baseline\n(green = improvement, red = degradation)",
             fontsize=11, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Delta F1", fontsize=10)

# Annotate cells
for i in range(n_subj):
    for j in range(3):
        v = delta_sorted[i, j]
        txt_color = "black" if abs(v) < vmax * 0.5 else "white"
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                fontsize=5.5, color=txt_color)

plt.tight_layout()
out_d = RPT_DIR / "subject_improvement_heatmap.png"
plt.savefig(out_d, dpi=300)
plt.close()
print(f"  [saved] {out_d.name}")

print("\n=== generate_subject_line_plots.py COMPLETE ===")
