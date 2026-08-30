# run_seed_stability.py
# ---------------------------------------------------------------------------
# Action 2.3 — Seed-stability sweep.
#
# Re-runs the headline per-subject-normalised LOSO configuration across several
# random seeds and reports mean +/- SD of the LOSO macro-F1 across seeds, so the
# 0.777 / 0.773 / 0.754 numbers can be shown to be stable rather than a single
# lucky seed. SVM (RBF) is deterministic given the data, so its variation comes
# only from the inner CV split ordering; RF and the CNN are stochastic and are
# the real test of stability.
#
# This is a DRIVER: it shells out to the existing trainers (so the model config
# is identical) once per seed, then aggregates their summary CSVs. It is
# resumable — a seed whose output already exists is skipped.
#
# Example (classical only, 3 seeds):
#   python run_seed_stability.py \
#       --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz \
#       --meta     features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --seeds 42,7,123 --models classical
#
# Add the CNN (heavier) once you are happy with the classical numbers:
#   python run_seed_stability.py ... --models classical,cnn \
#       --npz windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import subprocess
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PY = sys.executable  # use the same interpreter that launched this driver


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")


def classical_summary_f1(out_dir: Path, model: str):
    hits = glob.glob(str(out_dir / f"*__{model}_nested_loso_summary.csv"))
    if not hits:
        return None
    return float(pd.read_csv(hits[0])["f1_macro_mean"].iloc[0])


def main():
    ap = argparse.ArgumentParser("Seed-stability sweep for the headline LOSO config.")
    ap.add_argument("--features", required=True, help="freq72 feature .npz (classical)")
    ap.add_argument("--meta", required=True, help="aligned meta CSV")
    ap.add_argument("--npz", default=None, help="raw windows .npz with X_env (needed only for --models including cnn)")
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--models", default="classical",
                    help="comma list from {classical, cnn}")
    ap.add_argument("--outroot", default="results_seed_stability")
    ap.add_argument("--resume", action="store_true",
                    help="No-op: this driver is always resume-safe (each seed is skipped "
                         "if already done; each trainer subcall uses --resume + per-subject "
                         "checkpoints). Accepted for compatibility with run_with_memory_guard.py.")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    do_classical = "classical" in args.models
    do_cnn = "cnn" in args.models
    outroot = ROOT / args.outroot
    outroot.mkdir(parents=True, exist_ok=True)

    records = []  # (model, seed, f1)

    for seed in seeds:
        if do_classical:
            cdir = outroot / f"classical_seed{seed}"
            need = (classical_summary_f1(cdir, "SVM") is None or
                    classical_summary_f1(cdir, "RF") is None)
            if need:
                run([PY, "train_classical_loso.py",
                     "--features", args.features, "--meta", args.meta,
                     "--out", str(cdir), "--models", "SVM,RF",
                     "--norm-mode", "per_subject", "--inner-splits", "5",
                     "--seed", str(seed), "--rf-n-jobs", "6", "--resume"])
            for m in ["SVM", "RF"]:
                f1 = classical_summary_f1(cdir, m)
                if f1 is not None:
                    records.append({"model": m, "seed": seed, "f1_macro": f1})

        if do_cnn:
            if not args.npz:
                raise ValueError("--npz is required when --models includes cnn")
            ndir = outroot / f"cnn_seed{seed}"
            summ = ndir / "cnn_loso_summary.csv"
            if not summ.exists():
                run([PY, "train_cnn_loso.py",
                     "--npz", args.npz, "--meta", args.meta, "--xkey", "X_env",
                     "--out", str(ndir), "--norm-mode", "per_subject",
                     "--seed", str(seed), "--resume"])
            if summ.exists():
                f1 = float(pd.read_csv(summ)["mean_f1"].iloc[0])
                records.append({"model": "CNN", "seed": seed, "f1_macro": f1})

    df = pd.DataFrame(records)
    df.to_csv(outroot / "seed_stability_raw.csv", index=False)

    rows = []
    for m, g in df.groupby("model"):
        rows.append({"model": m,
                     "f1_mean_over_seeds": round(g["f1_macro"].mean(), 4),
                     "f1_sd_over_seeds": round(g["f1_macro"].std(ddof=1), 4) if len(g) > 1 else 0.0,
                     "n_seeds": len(g),
                     "seeds": ",".join(map(str, sorted(g["seed"].tolist())))})
    summ = pd.DataFrame(rows)
    summ.to_csv(outroot / "seed_stability_summary.csv", index=False)
    print("\n================  SEED-STABILITY SUMMARY (paste this back)  ================")
    print(summ.to_string(index=False))
    print("\nHeadline (seed 42): SVM 0.7767, RF 0.7732, CNN 0.7537")
    print(f"[save] {outroot/'seed_stability_summary.csv'}")


if __name__ == "__main__":
    main()
