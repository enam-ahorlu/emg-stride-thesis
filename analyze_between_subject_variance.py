#!/usr/bin/env python3
"""
analyze_between_subject_variance.py
====================================
EXPERIMENT_PLAN_CRITIQUE.md E1 (triage T25) -- between-subject variance
decomposition. Explains WHY per-subject z-scoring works and why CORAL adds
little on top: (A) per-feature intraclass correlation (ICC), class-conditional
then pooled; (B) an alignment ladder (rung 0-4) measuring how much
between-subject discrepancy (MMD, Wasserstein-1, subject-identity probe) each
normalisation operator removes; (C) whether distributional outlierness under
global vs per-subject norm predicts LOSO subject difficulty.

CPU only, no training (classical LOSO subjectwise CSVs + ResNet-SE+CD proba
subjectwise CSV are reused, not retrained).

Outputs: results_variance_decomposition/{variance_components,alignment_ladder,
subject_distance_vs_f1}.csv + report_figs/new_experiments/{icc_histogram,
alignment_ladder,distance_vs_f1}.png
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from train_classical_loso import load_features_npz, encode_labels, per_subject_zscore

ROOT = Path(__file__).parent
OUT = ROOT / "results_variance_decomposition"; OUT.mkdir(exist_ok=True)
FIGDIR = ROOT / "report_figs" / "new_experiments"; FIGDIR.mkdir(parents=True, exist_ok=True)

FEAT = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
LABELS = ["DNS", "STDUP", "UPS", "WAK"]
SEED = 42
N_CHANNELS = 9

# 72 = MAV,RMS,WL,ZC (base, 9 ch each) + WAMP (9) + MNF,MDF,SP (9 ch each);
# order fixed by extract_features.py's extract_one_window_features().
FAMILY_BLOCKS = ["MAV", "RMS", "WL", "ZC", "WAMP", "MNF", "MDF", "SP"]
FEATURE_FAMILY = np.repeat(FAMILY_BLOCKS, N_CHANNELS)
FEATURE_CHANNEL = np.tile(np.arange(1, N_CHANNELS + 1), len(FAMILY_BLOCKS))
assert len(FEATURE_FAMILY) == 72


def sym_sqrt(M, eps=1e-6):
    w, V = eigh((M + M.T) / 2.0)
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T


def sym_invsqrt(M, eps=1e-6):
    w, V = eigh((M + M.T) / 2.0)
    w = np.clip(w, eps, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def load_data():
    X = load_features_npz(FEAT).astype(np.float64)
    meta = pd.read_csv(META)
    y, label_map = encode_labels(meta["movement"].astype(str).to_numpy())
    # "subject" (1..40) is authoritative -- matches _bestparams.json keys and
    # every *_subjectwise.csv's heldout_subject/subject column. "subject_int"
    # is an unrelated 0-indexed internal relabeling (NOT subject-1) and must
    # not be used for subject identity or merges against other result files.
    subjects = meta["subject"].astype(int).to_numpy()
    return X, y, subjects, meta


# ============================================================================
# PART A -- per-feature ICC, class-conditional then pooled
# ============================================================================
def icc_one_way(x: np.ndarray, groups: np.ndarray):
    """Standard unequal-n one-way random-effects ICC(1) components.
    Returns (s2_between, s2_within, ssb, ssw, dfb, dfw)."""
    uniq = np.unique(groups)
    k = len(uniq)
    n_i = np.array([np.sum(groups == g) for g in uniq])
    means_i = np.array([x[groups == g].mean() for g in uniq])
    N = n_i.sum()
    grand_mean = x.mean()
    ssb = float((n_i * (means_i - grand_mean) ** 2).sum())
    ssw = float(sum(((x[groups == g] - means_i[i]) ** 2).sum() for i, g in enumerate(uniq)))
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    n0 = (N - (n_i ** 2).sum() / N) / dfb
    s2_within = msw
    s2_between = max((msb - msw) / n0, 0.0)
    return s2_between, s2_within, ssb, ssw, dfb, dfw


def part_a_icc(X, y, subjects):
    print("\n[Part A] per-feature ICC, class-conditional then pooled")
    rows = []
    for f in range(X.shape[1]):
        s2b_list, s2w_list, dfb_list, dfw_list = [], [], [], []
        for c in range(len(LABELS)):
            mask = y == c
            s2b, s2w, ssb, ssw, dfb, dfw = icc_one_way(X[mask, f], subjects[mask])
            s2b_list.append(s2b); s2w_list.append(s2w)
            dfb_list.append(dfb); dfw_list.append(dfw)
        # pool across classes: simple mean of the class-conditional variance
        # components (removes class-composition confound per class, then
        # combines with equal class weight so no single class dominates).
        s2b_pooled = float(np.mean(s2b_list))
        s2w_pooled = float(np.mean(s2w_list))
        icc_pooled = s2b_pooled / (s2b_pooled + s2w_pooled) if (s2b_pooled + s2w_pooled) > 0 else 0.0
        rows.append(dict(feature_idx=f, family=FEATURE_FAMILY[f], channel=int(FEATURE_CHANNEL[f]),
                         s2_between=s2b_pooled, s2_within=s2w_pooled, ICC=icc_pooled,
                         **{f"ICC_{LABELS[c]}": s2b_list[c] / (s2b_list[c] + s2w_list[c])
                            if (s2b_list[c] + s2w_list[c]) > 0 else 0.0 for c in range(len(LABELS))}))
    df = pd.DataFrame(rows).sort_values("ICC", ascending=False)
    df.to_csv(OUT / "variance_components.csv", index=False)
    print(df.groupby("family")["ICC"].mean().sort_values(ascending=False).to_string())
    print(f"[save] {OUT / 'variance_components.csv'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(df["ICC"], bins=24, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("ICC (fraction of variance between-subject)")
    axes[0].set_ylabel("count of features (of 72)")
    axes[0].set_title("ICC distribution, all 72 features")
    fam_order = df.groupby("family")["ICC"].mean().sort_values(ascending=False).index
    data_by_fam = [df[df.family == fam]["ICC"].to_numpy() for fam in fam_order]
    axes[1].boxplot(data_by_fam, tick_labels=fam_order)
    axes[1].set_ylabel("ICC")
    axes[1].set_title("ICC by feature family")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGDIR / "icc_histogram.png", dpi=160)
    plt.close(fig)
    print(f"[save] {FIGDIR / 'icc_histogram.png'}")
    return df


# ============================================================================
# PART B -- alignment ladder
# ============================================================================
def rung0_global_z(X):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


def rung1_mean_center(X, subjects):
    """Rung-0 baseline (global z-score, so all 72 features share a common
    scale) PLUS per-subject mean-centering -- isolates the location-only
    effect. Applying mean-centering to RAW (unscaled) features would make
    MMD/W1 incomparable across rungs, since features have wildly different
    native units (e.g. ZC counts vs MAV fractions)."""
    Xg = rung0_global_z(X)
    Xo = Xg.copy()
    for s in np.unique(subjects):
        m = subjects == s
        Xo[m] = Xg[m] - Xg[m].mean(0, keepdims=True)
    return Xo


def rung2_scale_only(X, subjects):
    """Rung-0 baseline PLUS per-subject scaling only (no re-centering) --
    isolates the scale-only effect, same common-scale rationale as rung 1."""
    Xg = rung0_global_z(X)
    Xo = Xg.copy()
    for s in np.unique(subjects):
        m = subjects == s
        sd = Xg[m].std(0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xo[m] = Xg[m] / sd
    return Xo


def rung3_mean_scale(X, subjects):
    return per_subject_zscore(X, subjects)


def rung4_full_whiten_recolor(X, subjects, lam=1.0):
    """Per-subject whitening (Cs^-1/2), recoloured to the pooled covariance of
    the rung-3 (mean+scale) output -- the per-subject analogue of CORAL's
    full second-order alignment, vs. rung 3's diagonal-only alignment."""
    X3 = rung3_mean_scale(X, subjects)
    Ctarget = np.cov(X3, rowvar=False) + lam * np.eye(X.shape[1])
    Ctarget_sqrt = sym_sqrt(Ctarget)
    Xo = X.copy()
    for s in np.unique(subjects):
        m = subjects == s
        Xs = X[m]
        mu = Xs.mean(0, keepdims=True)
        Cs = np.cov(Xs, rowvar=False) + lam * np.eye(X.shape[1])
        A = sym_invsqrt(Cs) @ Ctarget_sqrt
        Xo[m] = (Xs - mu) @ A
    return Xo


RUNGS = {
    0: ("global_z", rung0_global_z, False),
    1: ("mean_center", rung1_mean_center, True),
    2: ("scale_only", rung2_scale_only, True),
    3: ("mean_scale", rung3_mean_scale, True),
    4: ("full_whiten_recolor", rung4_full_whiten_recolor, True),
}

N_PER_SUBJECT = 200  # stratified subsample size, shared with make_feature_space_viz.py (E2)


def stratified_subsample(y, subjects, n_per_subject=N_PER_SUBJECT, seed=SEED):
    """Balanced-by-class, capped-per-subject subsample (200/subject, seed 42).
    Lives here (not in make_feature_space_viz.py) so E1's silhouette-by-class
    computation and E2's embeddings use the IDENTICAL subsample -- E2 imports
    this rather than defining its own copy, which would risk drifting out of
    sync and make the two experiments' numbers not directly comparable."""
    rng = np.random.default_rng(seed)
    n_classes = len(LABELS)
    per_class_target = n_per_subject // n_classes
    idx_out = []
    for s in np.unique(subjects):
        for c in range(n_classes):
            m = np.where((subjects == s) & (y == c))[0]
            take = min(per_class_target, len(m))
            if take > 0:
                idx_out.append(rng.choice(m, take, replace=False))
    return np.concatenate(idx_out)


def rbf_mmd2(A, B, gamma):
    """Unbiased-ish RBF MMD^2 between two samples (biased estimator, standard
    in practice: includes diagonal terms)."""
    def rbf(U, V):
        su = (U ** 2).sum(1, keepdims=True)
        sv = (V ** 2).sum(1, keepdims=True)
        d2 = su + sv.T - 2.0 * (U @ V.T)
        return np.exp(-gamma * np.clip(d2, 0, None))
    Kaa = rbf(A, A).mean()
    Kbb = rbf(B, B).mean()
    Kab = rbf(A, B).mean()
    return float(Kaa + Kbb - 2 * Kab)


def median_heuristic_gamma(Z):
    n = min(len(Z), 1500)
    idx = np.random.default_rng(SEED).choice(len(Z), n, replace=False)
    Zs = Z[idx]
    d2 = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def part_b_ladder(X, y, subjects, cap_per_subject_class=100):
    print("\n[Part B] alignment ladder")
    rng = np.random.default_rng(SEED)
    subs_u = np.unique(subjects)
    # fixed subsample of window indices per subject-class, reused across all rungs
    sub_idx = {}
    for s in subs_u:
        for c in range(len(LABELS)):
            m = np.where((subjects == s) & (y == c))[0]
            if len(m) > cap_per_subject_class:
                m = rng.choice(m, cap_per_subject_class, replace=False)
            sub_idx[(s, c)] = m

    # SAME stratified subsample as E2 (make_feature_space_viz.py), fixed across
    # all 5 rungs, so silhouette_by_class is computed on identical points every
    # time -- isolates the effect of the transform, not the sample.
    class_sil_idx = stratified_subsample(y, subjects)
    ys_sil = y[class_sil_idx]

    rows = []
    for rung_id, (name, fn, needs_subjects) in RUNGS.items():
        t0 = time.time()
        Xr = fn(X, subjects) if needs_subjects else fn(X)
        gamma = median_heuristic_gamma(Xr)

        # (i) mean pairwise MMD, class-conditional then averaged
        mmd_per_class = []
        for c in range(len(LABELS)):
            samples = {s: Xr[sub_idx[(s, c)]] for s in subs_u if len(sub_idx[(s, c)]) >= 3}
            keys = list(samples.keys())
            vals = []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    vals.append(rbf_mmd2(samples[keys[i]], samples[keys[j]], gamma))
            mmd_per_class.append(np.mean(vals))
        mmd_mean = float(np.mean(mmd_per_class))

        # (ii) mean per-feature Wasserstein-1 between subject pairs (pooled
        # across classes for tractability; MMD above is the class-conditional metric)
        w1_vals = []
        subj_samples = {s: Xr[subjects == s] for s in subs_u}
        pair_count = 0
        for i in range(len(subs_u)):
            for j in range(i + 1, len(subs_u)):
                a, b = subj_samples[subs_u[i]], subj_samples[subs_u[j]]
                pair_count += 1
                if pair_count % 3 != 0:  # subsample pairs for tractability (~260/780)
                    continue
                w1_vals.append(np.mean([wasserstein_distance(a[:, f], b[:, f]) for f in range(Xr.shape[1])]))
        w1_mean = float(np.mean(w1_vals))

        # (iii) subject-identity probe: 5-fold CV balanced accuracy
        clf = LogisticRegression(max_iter=300)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        probe_acc = float(cross_val_score(clf, Xr, subjects, cv=skf, scoring="balanced_accuracy", n_jobs=1).mean())

        # (iv) class separability on the SAME points, raw 72-D space (not an
        # embedding) -- the counterweight to (i)-(iii): those all measure how
        # much SUBJECT structure a rung removes, this measures what happens to
        # CLASS structure along the way (the over-alignment trade-off).
        sil_class = float(silhouette_score(Xr[class_sil_idx], ys_sil))

        rows.append(dict(rung=rung_id, name=name, mmd_mean=mmd_mean, wasserstein1_mean=w1_mean,
                         subject_probe_bal_acc=probe_acc, silhouette_by_class=sil_class,
                         chance_floor=1.0 / len(subs_u), elapsed_sec=round(time.time() - t0, 1)))
        print(f"  rung {rung_id} ({name}): MMD={mmd_mean:.4f} W1={w1_mean:.4f} "
              f"probe_bal_acc={probe_acc:.4f} silhouette_by_class={sil_class:.4f} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df["mmd_removed_pct"] = (1 - df["mmd_mean"] / df.loc[df.rung == 0, "mmd_mean"].values[0]) * 100
    df["w1_removed_pct"] = (1 - df["wasserstein1_mean"] / df.loc[df.rung == 0, "wasserstein1_mean"].values[0]) * 100
    df.to_csv(OUT / "alignment_ladder.csv", index=False)
    print(df.to_string(index=False))
    print(f"[save] {OUT / 'alignment_ladder.csv'}")

    # Dual-axis: subject-discrepancy-removed (bars, keeps climbing through full
    # whitening) against class-separability (line, peaks at mean_scale and
    # falls at full_whiten_recolor) -- makes the over-alignment trade-off
    # visible on the figure itself, not just inferable from a bar chart that
    # only shows the subject-alignment side (which would read as an argument
    # FOR more alignment, the opposite of the thesis conclusion).
    #
    # Downstream LOSO F1 (SVM) is reported for three rungs: global norm
    # (rung 0), per-subject z-score (rung 3), and the actual CORAL baseline
    # (closest real-world analogue of rung 4's full covariance alignment --
    # CORAL itself, not rung 4's exact transform). These are placed in ONE
    # axes-fraction-anchored box (not per-bar floating callouts) precisely
    # because per-bar callouts collide unpredictably with the twin-axis line
    # whenever a bar and the line are both near the top of their own range at
    # the same x -- a fixed-position box cannot collide with data at all.
    LOSO_F1_SVM = [("global_z", 0.708), ("mean_scale", 0.777), ("full_whiten (~CORAL)", 0.724)]

    plt.rcParams.update({"font.size": 11})
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(df))
    ax1.bar(x, df["mmd_removed_pct"], color="#55A868", edgecolor="black",
           linewidth=0.3, width=0.6, label="subject discrepancy removed (MMD)")
    ax1.set_xticks(x); ax1.set_xticklabels(df["name"], rotation=25, ha="right")
    ax1.set_ylabel("% of rung-0 MMD removed", color="#2d6a3e")
    ax1.tick_params(axis="y", labelcolor="#2d6a3e")
    ax1.spines["top"].set_visible(False)
    ax1.set_ylim(min(0, df["mmd_removed_pct"].min() - 10), 100)

    ax2 = ax1.twinx()
    sil_lo, sil_hi = df["silhouette_by_class"].min(), df["silhouette_by_class"].max()
    sil_pad = max((sil_hi - sil_lo) * 0.25, 0.004)
    ax2.set_ylim(sil_lo - sil_pad, sil_hi + sil_pad)
    ax2.plot(x, df["silhouette_by_class"], color="#c44e52", marker="o", ms=7,
             lw=2, label="class separability (silhouette, raw 72-D)")
    ax2.set_ylabel("silhouette by class (raw 72-D)", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax2.spines["top"].set_visible(False)

    peak_idx = int(df["silhouette_by_class"].idxmax())
    peak_rung_name = RUNGS[int(df.loc[peak_idx, "rung"])][0]

    name_w = max(len(n) for n, _ in LOSO_F1_SVM) + 2
    summary_lines = ["Downstream LOSO F1 (SVM):"]
    for name, f1 in LOSO_F1_SVM:
        marker = "  <- peak" if name == peak_rung_name else ""
        summary_lines.append(f"  {name:<{name_w}}{f1:.3f}{marker}")
    ax1.text(0.02, 0.97, "\n".join(summary_lines), transform=ax1.transAxes, fontsize=8,
             va="top", ha="left", family="monospace", color="#333",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", lw=0.6))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    ax1.set_title(f"Alignment ladder: subject discrepancy removed vs. class separability\n"
                  f"class separability peaks at {peak_rung_name}, then falls at full whitening --\n"
                  f"the over-alignment trade-off, and it costs downstream F1 too",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / "alignment_ladder.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {FIGDIR / 'alignment_ladder.png'}")
    return df


# ============================================================================
# PART C -- distributional outlierness vs LOSO difficulty
# ============================================================================
def mahalanobis_dist(x_mean, pooled_mean, pooled_cov_inv):
    d = (x_mean - pooled_mean)
    return float(np.sqrt(max(d @ pooled_cov_inv @ d.T, 0.0)))


def part_c_distance_vs_f1(X, y, subjects):
    print("\n[Part C] distributional outlierness vs LOSO subject difficulty")
    subs_u = sorted(np.unique(subjects).tolist())
    rng = np.random.default_rng(SEED)

    X_global = StandardScaler().fit_transform(X)
    X_persubj = per_subject_zscore(X, subjects)

    def cap_idx(mask, cap=150):
        idx = np.where(mask)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        return idx

    rows = []
    for norm_name, Xn in [("global", X_global), ("per_subject", X_persubj)]:
        pooled_cov = np.cov(Xn, rowvar=False) + 1.0 * np.eye(Xn.shape[1])
        pooled_cov_inv = np.linalg.pinv(pooled_cov)
        pooled_mean = Xn.mean(0)
        gamma = median_heuristic_gamma(Xn)
        for s in subs_u:
            m_s = subjects == s
            m_other = ~m_s
            Xs_idx = cap_idx(m_s)
            Xother_idx = cap_idx(m_other, cap=1500)
            mmd = rbf_mmd2(Xn[Xs_idx], Xn[Xother_idx], gamma)
            maha = mahalanobis_dist(Xn[m_s].mean(0), pooled_mean, pooled_cov_inv)
            rows.append(dict(subject=int(s), norm=norm_name, mmd_to_others=mmd, mahalanobis=maha))
    dist_df = pd.DataFrame(rows)

    # merge with LOSO F1
    svm_f1 = pd.read_csv(ROOT / "results_loso_freq_persubj" /
                          "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext__SVM_nested_loso_subjectwise.csv")
    rf_f1 = pd.read_csv(ROOT / "results_loso_freq_persubj" /
                         "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext__RF_nested_loso_subjectwise.csv")
    f1_map = {"SVM": dict(zip(svm_f1.heldout_subject, svm_f1.f1_macro)),
              "RF": dict(zip(rf_f1.heldout_subject, rf_f1.f1_macro))}
    cd_path = ROOT / "results_cnn_aug_resnet_se_chandrop_proba" / "cnn_arch_subjectwise.csv"
    if cd_path.exists():
        cd_f1 = pd.read_csv(cd_path)
        f1_map["ResNet_SE_CD"] = dict(zip(cd_f1.subject, cd_f1.f1_macro))

    for model, m in f1_map.items():
        dist_df[f"f1_{model}"] = dist_df["subject"].map(m)

    dist_df.to_csv(OUT / "subject_distance_vs_f1.csv", index=False)
    print(f"[save] {OUT / 'subject_distance_vs_f1.csv'}")

    corr_rows = []
    for norm_name in ["global", "per_subject"]:
        sub = dist_df[dist_df.norm == norm_name]
        for model in f1_map:
            for dist_col in ["mmd_to_others", "mahalanobis"]:
                valid = sub.dropna(subset=[f"f1_{model}"])
                if len(valid) < 3:
                    continue
                rho, p = spearmanr(valid[dist_col], valid[f"f1_{model}"])
                corr_rows.append(dict(norm=norm_name, model=model, distance=dist_col,
                                      spearman_rho=round(float(rho), 4), p=float(p), n=len(valid)))
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "distance_vs_f1_correlations.csv", index=False)
    print(corr_df.to_string(index=False))
    print(f"[save] {OUT / 'distance_vs_f1_correlations.csv'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, norm_name in zip(axes, ["global", "per_subject"]):
        sub = dist_df[dist_df.norm == norm_name]
        ax.scatter(sub["mmd_to_others"], sub["f1_SVM"], alpha=0.7, label="SVM")
        rho_row = corr_df[(corr_df.norm == norm_name) & (corr_df.model == "SVM") & (corr_df.distance == "mmd_to_others")]
        rho = rho_row.spearman_rho.values[0] if len(rho_row) else float("nan")
        ax.set_title(f"{norm_name} norm (SVM, Spearman rho={rho:.2f})")
        ax.set_xlabel("MMD to other 39 subjects")
    axes[0].set_ylabel("subject LOSO macro-F1")
    fig.tight_layout()
    fig.savefig(FIGDIR / "distance_vs_f1.png", dpi=160)
    plt.close(fig)
    print(f"[save] {FIGDIR / 'distance_vs_f1.png'}")
    return dist_df, corr_df


def main():
    t0 = time.time()
    X, y, subjects, meta = load_data()
    print(f"[data] X={X.shape} subjects={len(np.unique(subjects))} classes={len(LABELS)}")
    part_a_icc(X, y, subjects)
    part_b_ladder(X, y, subjects)
    part_c_distance_vs_f1(X, y, subjects)
    print(f"\n[DONE] total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
