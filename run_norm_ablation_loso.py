#!/usr/bin/env python3
"""
run_norm_ablation_loso.py
=========================
Orchestrates normalization ablation for LOSO classical models.

Runs 4 conditions:
  1. none         — no scaling at all
  2. global       — StandardScaler (current default, results already exist)
  3. per_subject  — per-subject z-score before LOSO loop
  4. robust       — RobustScaler (median/IQR) in pipeline

Each condition runs SVM then RF separately (stop-start memory management).

Usage:
  python run_norm_ablation_loso.py [--skip-global] [--features FEATURES] [--meta META]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOSO_SCRIPT = str(ROOT / "train_classical_loso.py")

# Default feature files (base-36)
DEFAULT_FEATURES = str(ROOT / "features_out" /
    "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_base.npz")
DEFAULT_META = str(ROOT / "features_out" /
    "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv")

# Normalization conditions and output directories
CONDITIONS = [
    ("none",        "results_loso_norm_none"),
    ("global",      "results_loso_light"),       # existing results
    ("per_subject", "results_loso_norm_persubj"),
    ("robust",      "results_loso_norm_robust"),
]


def run_loso(norm_mode: str, out_dir: str, model: str,
             features: str, meta: str) -> bool:
    """Run a single LOSO training. Returns True on success."""
    cmd = [
        PYTHON, LOSO_SCRIPT,
        "--features", features,
        "--meta", meta,
        "--out", out_dir,
        "--models", model,
        "--norm-mode", norm_mode,
        "--inner-splits", "5",
        "--save-preds",
        "--cv-scheme", "loso",
        "--resume",
        "--flush-preds",
    ]

    print(f"\n{'='*60}")
    print(f"  NORM={norm_mode}  MODEL={model}  OUT={out_dir}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  [OK] {norm_mode}/{model} completed in {elapsed:.0f}s")
        return True
    else:
        print(f"  [FAIL] {norm_mode}/{model} returned {result.returncode} after {elapsed:.0f}s")
        return False


def main():
    ap = argparse.ArgumentParser("Normalization ablation runner for LOSO")
    ap.add_argument("--skip-global", action="store_true",
                    help="Skip global condition (results already in results_loso_light/)")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--conditions", default=None,
                    help="Comma-separated list of conditions to run (e.g. 'none,per_subject')")
    ap.add_argument("--models", default="SVM,RF",
                    help="Comma-separated models to run")
    args = ap.parse_args()

    models = [m.strip().upper() for m in args.models.split(",")]

    if args.conditions:
        run_conditions = [c.strip() for c in args.conditions.split(",")]
        conditions = [(nm, od) for nm, od in CONDITIONS if nm in run_conditions]
    else:
        conditions = CONDITIONS

    if args.skip_global:
        conditions = [(nm, od) for nm, od in conditions if nm != "global"]

    print(f"Normalization ablation: {len(conditions)} conditions x {len(models)} models")
    print(f"Features: {args.features}")
    print(f"Conditions: {[c[0] for c in conditions]}")
    print(f"Models: {models}")

    results = []
    for norm_mode, out_dir in conditions:
        for model in models:
            ok = run_loso(norm_mode, out_dir, model, args.features, args.meta)
            results.append((norm_mode, model, ok))

    # Summary
    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    for norm_mode, model, ok in results:
        status = "OK" if ok else "FAIL"
        print(f"  {norm_mode:15s} {model:5s} -> {status}")

    failed = [r for r in results if not r[2]]
    if failed:
        print(f"\n  {len(failed)} runs FAILED. Check logs above.")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} runs completed successfully!")
        print(f"  Now run: python compare_norm_ablation.py")


if __name__ == "__main__":
    main()
