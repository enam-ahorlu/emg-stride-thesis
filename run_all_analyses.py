#!/usr/bin/env python3
"""
Run all supplementary analyses — uses scipy for Wilcoxon signed-rank test.
"""
import os, json, csv, math
import numpy as np
from scipy.stats import wilcoxon as scipy_wilcoxon

PROJ = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJ, "analysis_results.json")

results = {}

# --- Wilcoxon signed-rank test via scipy (two-sided) ---
def wilcoxon_signed_rank(x, y):
    """Two-sided Wilcoxon signed-rank test using scipy.
    Returns (W_stat, W_stat, z=nan, p, n_eff) to preserve call signature."""
    d = x - y
    n_eff = int(np.sum(d != 0))  # effective n after dropping zeros
    if n_eff == 0:
        return 0.0, 0.0, float('nan'), 1.0, 0
    stat, p = scipy_wilcoxon(x, y, alternative='two-sided')
    return stat, stat, float('nan'), p, n_eff

# --- Pearson and Spearman correlation ---
def pearson_r(x, y):
    n = len(x)
    mx, my = x.mean(), y.mean()
    num = np.sum((x - mx) * (y - my))
    den = math.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2))
    return num / den if den > 0 else 0

def spearman_rho(x, y):
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    return pearson_r(rx, ry)

# ============================================================
# 1. Load per-subject LOSO F1 scores
# ============================================================
print("=" * 60)
print("1. LOADING PER-SUBJECT LOSO METRICS")
print("=" * 60)

loso_path = os.path.join(PROJ, "results_loso_light", "per_subject_metrics_250_base_loso.csv")
rf_f1, svm_f1 = {}, {}

with open(loso_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        subj = int(row['subject'])
        model = row['model']
        f1 = float(row['f1_macro'])
        if model == 'SVM':
            svm_f1[subj] = f1
        elif model == 'RF':
            rf_f1[subj] = f1

cnn_path = os.path.join(PROJ, "results_cnn_loso", "per_subject_metrics_cnn_loso.csv")
cnn_f1 = {}
with open(cnn_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        subj = int(row['subject'])
        cnn_f1[subj] = float(row['f1_macro'])

subjects = sorted(set(rf_f1.keys()) & set(svm_f1.keys()) & set(cnn_f1.keys()))
rf_arr = np.array([rf_f1[s] for s in subjects])
svm_arr = np.array([svm_f1[s] for s in subjects])
cnn_arr = np.array([cnn_f1[s] for s in subjects])

print(f"N subjects: {len(subjects)}")
print(f"RF  LOSO F1: {rf_arr.mean():.4f} +/- {rf_arr.std(ddof=1):.4f}")
print(f"SVM LOSO F1: {svm_arr.mean():.4f} +/- {svm_arr.std(ddof=1):.4f}")
print(f"CNN LOSO F1: {cnn_arr.mean():.4f} +/- {cnn_arr.std(ddof=1):.4f}")

# ============================================================
# 2. WILCOXON SIGNED-RANK TESTS
# ============================================================
print("\n" + "=" * 60)
print("2. WILCOXON SIGNED-RANK TESTS (two-sided)")
print("=" * 60)

comparisons = {
    "RF_vs_SVM": (rf_arr, svm_arr),
    "RF_vs_CNN": (rf_arr, cnn_arr),
    "SVM_vs_CNN": (svm_arr, cnn_arr),
}

stat_results = {}
for name, (a, b) in comparisons.items():
    Wp, Wm, z, p, n_eff = wilcoxon_signed_rank(a, b)
    mean_diff = (a - b).mean()
    sig = "YES" if p < 0.05 else "NO"
    print(f"{name}: W+={Wp:.1f}, W-={Wm:.1f}, z={z:.3f}, p={p:.6f}, mean_diff={mean_diff:.4f}, sig={sig}")
    stat_results[name] = {
        "W_plus": float(Wp), "W_minus": float(Wm),
        "z_approx": round(float(z), 4),
        "p_value": round(float(p), 6),
        "mean_difference": round(float(mean_diff), 4),
        "n_effective": int(n_eff),
        "significant_at_005": p < 0.05,
    }

results["wilcoxon_tests"] = stat_results

# ============================================================
# 3. CONFIDENCE INTERVALS
# ============================================================
print("\n" + "=" * 60)
print("3. 95% CONFIDENCE INTERVALS")
print("=" * 60)

def ci_95(arr):
    n = len(arr)
    mean = arr.mean()
    se = arr.std(ddof=1) / math.sqrt(n)
    return mean, se, mean - 1.96 * se, mean + 1.96 * se

ci_results = {}
for name, arr in [("RF", rf_arr), ("SVM", svm_arr), ("CNN", cnn_arr)]:
    mean, se, lo, hi = ci_95(arr)
    print(f"{name} LOSO F1: {mean*100:.2f}% [{lo*100:.2f}%, {hi*100:.2f}%]")
    ci_results[name] = {
        "mean": round(float(mean), 4), "std": round(float(arr.std(ddof=1)), 4),
        "se": round(float(se), 4),
        "ci_95_low": round(float(lo), 4), "ci_95_high": round(float(hi), 4),
        "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
    }

results["confidence_intervals"] = ci_results

# ============================================================
# 4. SUBJECT DIFFICULTY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("4. SUBJECT DIFFICULTY ANALYSIS")
print("=" * 60)

subj_avg_f1 = {s: (rf_f1[s] + svm_f1[s] + cnn_f1[s]) / 3.0 for s in subjects}
sorted_subjs = sorted(subj_avg_f1.items(), key=lambda x: x[1])

print("\n--- 5 HARDEST SUBJECTS ---")
hard5 = []
for s, f1 in sorted_subjs[:5]:
    print(f"  Subject {s}: avg_f1={f1*100:.1f}% (RF={rf_f1[s]*100:.1f}%, SVM={svm_f1[s]*100:.1f}%, CNN={cnn_f1[s]*100:.1f}%)")
    hard5.append({"subject": s, "avg_f1": round(f1*100, 1), "rf": round(rf_f1[s]*100, 1), "svm": round(svm_f1[s]*100, 1), "cnn": round(cnn_f1[s]*100, 1)})

print("\n--- 5 EASIEST SUBJECTS ---")
easy5 = []
for s, f1 in sorted_subjs[-5:]:
    print(f"  Subject {s}: avg_f1={f1*100:.1f}% (RF={rf_f1[s]*100:.1f}%, SVM={svm_f1[s]*100:.1f}%, CNN={cnn_f1[s]*100:.1f}%)")
    easy5.append({"subject": s, "avg_f1": round(f1*100, 1), "rf": round(rf_f1[s]*100, 1), "svm": round(svm_f1[s]*100, 1), "cnn": round(cnn_f1[s]*100, 1)})

# Cross-model correlation
r_rf_svm = pearson_r(rf_arr, svm_arr)
r_rf_cnn = pearson_r(rf_arr, cnn_arr)
r_svm_cnn = pearson_r(svm_arr, cnn_arr)
rho_rf_svm = spearman_rho(rf_arr, svm_arr)

print(f"\nPearson r: RF-SVM={r_rf_svm:.3f}, RF-CNN={r_rf_cnn:.3f}, SVM-CNN={r_svm_cnn:.3f}")
print(f"Spearman rho RF-SVM: {rho_rf_svm:.3f}")

f1_range = (sorted_subjs[-1][1] - sorted_subjs[0][1]) * 100
print(f"F1 range across subjects: {f1_range:.1f} pp")

results["subject_difficulty"] = {
    "hardest_5": hard5, "easiest_5": easy5,
    "range_pp": round(f1_range, 1),
    "correlations": {"pearson_rf_svm": round(r_rf_svm, 3), "pearson_rf_cnn": round(r_rf_cnn, 3), "pearson_svm_cnn": round(r_svm_cnn, 3)},
}

# ============================================================
# 5. CLASS IMBALANCE
# ============================================================
print("\n" + "=" * 60)
print("5. CLASS IMBALANCE")
print("=" * 60)

json_250 = os.path.join(PROJ, "windows_WAK_UPS_DNS_STDUP_v1_meta_w250_ov50_conf60_AorR_summary.json")
with open(json_250) as f:
    w250 = json.load(f)

total = w250['summary']['total_windows']
by_move = w250['windows_by_movement']
imb = max(by_move.values()) / min(by_move.values())

for cls, cnt in sorted(by_move.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {cnt} ({cnt/total*100:.1f}%)")
print(f"Total: {total}, Imbalance ratio: {imb:.1f}:1")

results["class_imbalance"] = {
    "total_windows_250ms": total,
    "windows_by_class": by_move,
    "imbalance_ratio": round(imb, 2),
    "percentages": {k: round(v/total*100, 1) for k, v in by_move.items()},
}

# ============================================================
# 6. GENERALIZATION GAP WITH STATS
# ============================================================
print("\n" + "=" * 60)
print("6. GENERALIZATION GAP")
print("=" * 60)

gap_path = os.path.join(PROJ, "results_loso_light", "generalization_gap.csv")
rf_sd, svm_sd = {}, {}
with open(gap_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = int(row['subject'])
        if row['model'] == 'RF':
            rf_sd[s] = float(row['f1_sd'])
        elif row['model'] == 'SVM':
            svm_sd[s] = float(row['f1_sd'])

cnn_gap_path = os.path.join(PROJ, "results_cnn_loso", "generalization_gap_cnn.csv")
cnn_sd = {}
if os.path.exists(cnn_gap_path):
    with open(cnn_gap_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cnn_sd[int(row['subject'])] = float(row['f1_sd'])

rf_sd_arr = np.array([rf_sd[s] for s in subjects])
svm_sd_arr = np.array([svm_sd[s] for s in subjects])
cnn_sd_arr = np.array([cnn_sd.get(s, np.nan) for s in subjects])

rf_gap = rf_sd_arr - rf_arr
svm_gap = svm_sd_arr - svm_arr
cnn_gap = cnn_sd_arr - cnn_arr

# One-sided Wilcoxon: SD > LOSO
for name, sd_a, loso_a in [("RF", rf_sd_arr, rf_arr), ("SVM", svm_sd_arr, svm_arr)]:
    Wp, Wm, z, p, n_eff = wilcoxon_signed_rank(sd_a, loso_a)
    p_one = p / 2  # one-sided
    gap_a = sd_a - loso_a
    print(f"{name}: gap={gap_a.mean()*100:.1f}pp +/- {gap_a.std(ddof=1)*100:.1f}pp, p(one-sided)={p_one:.2e}")

# CNN
mask = ~np.isnan(cnn_sd_arr)
Wp, Wm, z, p, n_eff = wilcoxon_signed_rank(cnn_sd_arr[mask], cnn_arr[mask])
p_one = p / 2
cnn_g = cnn_gap[mask]
print(f"CNN: gap={cnn_g.mean()*100:.1f}pp +/- {cnn_g.std(ddof=1)*100:.1f}pp, p(one-sided)={p_one:.2e}")

results["generalization_gap_stats"] = {
    "RF": {"sd_mean": round(rf_sd_arr.mean()*100, 2), "loso_mean": round(rf_arr.mean()*100, 2),
           "gap_mean_pp": round(rf_gap.mean()*100, 2), "gap_std_pp": round(rf_gap.std(ddof=1)*100, 2)},
    "SVM": {"sd_mean": round(svm_sd_arr.mean()*100, 2), "loso_mean": round(svm_arr.mean()*100, 2),
            "gap_mean_pp": round(svm_gap.mean()*100, 2), "gap_std_pp": round(svm_gap.std(ddof=1)*100, 2)},
    "CNN": {"sd_mean": round(float(np.nanmean(cnn_sd_arr))*100, 2), "loso_mean": round(cnn_arr.mean()*100, 2),
            "gap_mean_pp": round(float(np.nanmean(cnn_gap))*100, 2), "gap_std_pp": round(float(np.nanstd(cnn_gap, ddof=1))*100, 2)},
}

# ============================================================
# 7. CHECK LDA
# ============================================================
print("\n" + "=" * 60)
print("7. LDA STATUS")
print("=" * 60)

classical_dir = os.path.join(PROJ, "results_classical")
lda_found = False
if os.path.exists(classical_dir):
    for fn in os.listdir(classical_dir):
        if 'LDA' in fn.upper() or 'lda' in fn:
            lda_found = True
            print(f"  Found: {fn}")

lda_loso = [f for f in os.listdir(os.path.join(PROJ, "results_loso_light")) if 'lda' in f.lower()]
print(f"LDA SD results: {'Found' if lda_found else 'Not found'}")
print(f"LDA LOSO results: {lda_loso if lda_loso else 'Not found'}")

results["lda_status"] = {"sd_found": lda_found, "loso_found": bool(lda_loso)}

# SAVE — custom encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)
print(f"\nAll results saved to {OUT}")
