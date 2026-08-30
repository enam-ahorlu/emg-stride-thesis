"""Compute pooled training-distribution statistics for the MyoLens OOD guard.

Mirrors prepare_deployment_artifacts.py's load_all() + per_subject_zscore_2d() + the
holdout split, then computes the mean vector and a (ridge-regularised) inverse covariance
of the z-scored Freq-72 feature space over the 37 training subjects only. Held-out subjects
(10, 13, 22) never contribute, matching the deployment model's training/holdout split
exactly -- this statistic describes the distribution the model was actually trained on.

prepare_deployment_artifacts.py never computed or persisted this. Confirmed with Enam
14 Aug 2026 while building the calibration OOD guard (C4): the gap was real, not an
oversight in reading the manifest.

Run in the thesis venv from the 06_Code directory:

    python compute_ood_stats.py --out "../../Advanced Software Engineering/MyoLens/artifacts"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FEATURES = (
    "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
)
DEFAULT_META = (
    "features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def per_subject_zscore_2d(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Verbatim from prepare_deployment_artifacts.py -- must match the serving-time
    normalisation exactly, or the OOD stats describe a different distribution than the one
    the deployed model actually sees."""
    out = X.copy().astype(np.float64)
    for sid in np.unique(subjects):
        m = subjects == sid
        block = out[m]
        mu = block.mean(axis=0, keepdims=True)
        sd = block.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out[m] = (block - mu) / sd
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Compute MyoLens OOD-guard training statistics.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--out", required=True, help="Artefact output directory")
    ap.add_argument("--holdout", default="10,13,22")
    ap.add_argument(
        "--ridge", type=float, default=1e-3,
        help="Diagonal ridge added to the covariance before inversion, as a fraction of the "
             "mean feature variance, for numerical stability against near-collinear columns.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    fp, mp = root / args.features, root / args.meta
    X_feat = np.load(fp, allow_pickle=False)["X"]
    if X_feat.ndim != 2 or X_feat.shape[1] != 72:
        raise SystemExit(f"FATAL: expected (N,72) Freq-72 features, got {X_feat.shape}")

    md = pd.read_csv(mp)
    subjects = md["subject"].astype(int).to_numpy()
    if len(md) != len(X_feat):
        raise SystemExit(f"FATAL: row-count mismatch -- features {len(X_feat)}, meta {len(md)}")

    holdout = np.array(sorted(int(s) for s in args.holdout.split(",")))
    tr_mask = ~np.isin(subjects, holdout)
    train_subj = np.unique(subjects[tr_mask])
    print(f"[data] {len(X_feat):,} windows total; {int(tr_mask.sum()):,} in "
          f"{len(train_subj)} training subjects; holdout {holdout.tolist()} excluded")

    Xf = per_subject_zscore_2d(X_feat, subjects)
    Xtr = Xf[tr_mask].astype(np.float64)

    mean = Xtr.mean(axis=0)
    cov = np.cov(Xtr, rowvar=False, ddof=1)

    # Ridge regularisation: several Freq-72 blocks are amplitude proxies of the same nine
    # channels (MAV, RMS, WL are all highly correlated with each other), so the empirical
    # covariance is close to singular. A tiny determinant makes the Mahalanobis distance
    # wildly sensitive to noise in the covariance estimate rather than to genuine distance.
    diag_var = np.diag(cov).copy()
    cov_reg = cov + np.eye(72) * diag_var.mean() * args.ridge
    inv_cov = np.linalg.inv(cov_reg)

    # Sanity check 1: distance of the training pool against itself. This is the number
    # app/config.py's ood_threshold (currently 12.0, chosen before this script existed) should
    # be read against.
    centred = Xtr - mean
    self_d = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centred, inv_cov, centred), 0.0))
    print(f"[check] training self-distance: mean={self_d.mean():.3f} "
          f"p95={np.percentile(self_d, 95):.3f} p99={np.percentile(self_d, 99):.3f} "
          f"max={self_d.max():.3f}")

    # Sanity check 2: the three held-out subjects are exactly the demo/deployment-validation
    # subjects (TD-01). If their distance blew past the threshold, the shipped demo would
    # refuse itself on first use -- worth knowing now rather than after deploying.
    te_mask = ~tr_mask
    Xte = Xf[te_mask].astype(np.float64)
    centred_te = Xte - mean
    te_d = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centred_te, inv_cov, centred_te), 0.0))
    te_subjects = subjects[te_mask]
    holdout_report = {}
    for sid in holdout:
        d = te_d[te_subjects == sid]
        holdout_report[str(sid)] = {"mean": float(d.mean()), "max": float(d.max())}
        print(f"[check] holdout subject {sid}: mean distance {d.mean():.3f}, max {d.max():.3f}")

    stats_path = out / "ood_stats.npz"
    np.savez_compressed(
        stats_path,
        mean=mean.astype(np.float64),
        inverse_covariance=inv_cov.astype(np.float64),
        training_subjects=train_subj.astype(np.int64),
        holdout_subjects=holdout.astype(np.int64),
        ridge=np.float64(args.ridge),
    )
    print(f"[save] {stats_path} ({stats_path.stat().st_size / 1024:.1f} KB)")

    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest["ood_guard"] = {
        "file": stats_path.name,
        "sha256": sha256(stats_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "Mahalanobis distance in the z-scored Freq-72 feature space, mean over "
                  "calibration windows",
        "pooled_over": "training subjects only (holdout excluded), z-scored per-subject "
                        "exactly as at training time",
        "ridge_regularisation": args.ridge,
        "n_training_windows": int(tr_mask.sum()),
        "n_training_subjects": int(len(train_subj)),
    }
    manifest["ood_guard"]["training_pool_self_distance"] = {
        "mean": float(self_d.mean()),
        "p95": float(np.percentile(self_d, 95)),
        "p99": float(np.percentile(self_d, 99)),
        "max": float(self_d.max()),
    }
    manifest["ood_guard"]["holdout_subject_distances"] = holdout_report
    manifest["ood_guard"]["note"] = (
        "Computed post-hoc (14 Aug 2026), not part of the original "
        "prepare_deployment_artifacts.py run -- that script never persisted OOD statistics. "
        "app/config.py's ood_threshold=12.0 predates this computation; sanity-check it against "
        "the printed percentiles above before trusting it as more than a placeholder."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[save] {manifest_path} updated with an 'ood_guard' section")


if __name__ == "__main__":
    main()
