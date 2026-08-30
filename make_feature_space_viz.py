#!/usr/bin/env python3
"""
make_feature_space_viz.py
==========================
EXPERIMENT_PLAN_CRITIQUE.md E2 (triage T38) -- t-SNE/UMAP of the Freq-72
feature space under three normalisations (global z, per-subject z, CORAL/
per-subject-whiten-recolor), colored by subject and by class. The visual
companion to E1's alignment ladder; reuses E1's rung0/rung3/rung4 operators
and its subject-identity-probe numbers (results_variance_decomposition/
alignment_ladder.csv) so the figure and the quoted number agree exactly.

CPU only. Outputs: report_figs/new_experiments/feature_space_{by_subject,
by_class}.png + results_variance_decomposition/embedding_metrics.csv
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from train_classical_loso import load_features_npz, encode_labels, per_subject_zscore
from analyze_between_subject_variance import (
    FEAT, META, LABELS, SEED, rung0_global_z, rung3_mean_scale, rung4_full_whiten_recolor,
    stratified_subsample, N_PER_SUBJECT,
)

ROOT = Path(__file__).parent
OUT = ROOT / "results_variance_decomposition"; OUT.mkdir(exist_ok=True)
FIGDIR = ROOT / "report_figs" / "new_experiments"; FIGDIR.mkdir(parents=True, exist_ok=True)

try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False
    print("[warn] umap-learn not available; shipping t-SNE only")

CONDITIONS = [
    ("global_z", rung0_global_z, False),
    ("per_subject_z", rung3_mean_scale, True),
    ("CORAL_recolor", rung4_full_whiten_recolor, True),
]


def main():
    X = load_features_npz(FEAT).astype(np.float64)
    meta = pd.read_csv(META)
    y, _ = encode_labels(meta["movement"].astype(str).to_numpy())
    subjects = meta["subject"].astype(int).to_numpy()  # authoritative subject id (1..40)

    idx = stratified_subsample(y, subjects)
    print(f"[subsample] {len(idx)} points ({len(np.unique(subjects))} subjects, "
          f"target {N_PER_SUBJECT}/subject balanced by class)")
    ys, subs = y[idx], subjects[idx]

    probe_lookup = {}
    ladder_csv = OUT / "alignment_ladder.csv"
    if ladder_csv.exists():
        ladder = pd.read_csv(ladder_csv).set_index("name")["subject_probe_bal_acc"].to_dict()
        probe_lookup = {"global_z": ladder.get("global_z"), "per_subject_z": ladder.get("mean_scale"),
                        "CORAL_recolor": ladder.get("full_whiten_recolor")}

    methods = [("tsne", lambda Z: TSNE(n_components=2, perplexity=30, random_state=SEED,
                                       init="pca", max_iter=1000).fit_transform(Z))]
    if HAVE_UMAP:
        methods.append(("umap", lambda Z: umap.UMAP(n_components=2, random_state=SEED).fit_transform(Z)))

    embeddings = {}
    metric_rows = []
    for cond_name, fn, needs_subjects in CONDITIONS:
        Xfull = fn(X, subjects) if needs_subjects else fn(X)
        Xr = Xfull[idx]
        sil_subj = float(silhouette_score(Xr, subs))
        sil_class = float(silhouette_score(Xr, ys))
        metric_rows.append(dict(condition=cond_name, silhouette_by_subject=sil_subj,
                                silhouette_by_class=sil_class,
                                subject_probe_bal_acc=probe_lookup.get(cond_name, np.nan),
                                n_points=len(idx)))
        print(f"[{cond_name}] silhouette_subject={sil_subj:.3f} silhouette_class={sil_class:.3f} "
              f"probe_bal_acc={probe_lookup.get(cond_name, float('nan')):.3f}")
        for mname, mfn in methods:
            print(f"  embedding {mname} ...")
            embeddings[(cond_name, mname)] = mfn(Xr)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT / "embedding_metrics.csv", index=False)
    print(f"[save] {OUT / 'embedding_metrics.csv'}")

    n_methods = len(methods)
    n_conds = len(CONDITIONS)

    for color_by, labels_vec, title_tag, cmap_name in [
        ("subject", subs, "colored by subject (40)", "tab20"),
        ("class", ys, "colored by class (4)", "Set1"),
    ]:
        fig, axes = plt.subplots(n_methods, n_conds, figsize=(5 * n_conds, 5 * n_methods), squeeze=False)
        for mi, (mname, _) in enumerate(methods):
            for ci, (cond_name, _, _) in enumerate(CONDITIONS):
                ax = axes[mi][ci]
                emb = embeddings[(cond_name, mname)]
                if color_by == "subject":
                    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels_vec, cmap=cmap_name, s=4, alpha=0.6)
                else:
                    for c in range(len(LABELS)):
                        m = labels_vec == c
                        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.6, label=LABELS[c])
                    if mi == 0 and ci == n_conds - 1:
                        ax.legend(markerscale=3, fontsize=8, loc="best")
                ax.set_title(f"{mname} / {cond_name}", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Freq-72 feature space, {title_tag}", fontsize=13)
        fig.tight_layout()
        fname = FIGDIR / f"feature_space_by_{color_by}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"[save] {fname}")


if __name__ == "__main__":
    main()
