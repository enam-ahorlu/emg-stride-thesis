#!/usr/bin/env python3
"""
run_cnn_norm_ablation_loso.py
==============================
Orchestrates normalization ablation for CNN LOSO.

Runs 3 new conditions (global already exists in results_cnn_loso/):
  1. none         — no scaling
  2. per_subject  — per-subject z-score before LOSO loop
  3. robust       — RobustScaler (median/IQR) per fold

Usage:
  python run_cnn_norm_ablation_loso.py [--include-global] [--npz NPZ] [--meta META]
  python run_cnn_norm_ablation_loso.py --conditions none,robust  # subset only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
CNN_SCRIPT = str(ROOT / "train_cnn_loso.py")

# Default data files (raw windows NPZ — not features)
DEFAULT_NPZ = str(ROOT / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz")
DEFAULT_META = str(ROOT / "features_out" /
    "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv")

# Normalization conditions and output directories
CONDITIONS = [
    ("none",        "results_cnn_loso_norm_none"),
    ("global",      "results_cnn_loso"),          # existing baseline — skipped by default
    ("per_subject", "results_cnn_loso_norm_persubj"),
    ("robust",      "results_cnn_loso_norm_robust"),
]


def run_cnn_loso(norm_mode: str, out_dir: str, npz: str, meta: str) -> bool:
    """Run a single CNN LOSO condition. Returns True on success."""
    cmd = [
        PYTHON, CNN_SCRIPT,
        "--npz", npz,
        "--meta", meta,
        "--out", out_dir,
        "--norm-mode", norm_mode,
        "--resume",
        "--amp",
    ]

    print(f"\n{'='*60}")
    print(f"  NORM={norm_mode}  OUT={out_dir}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  [OK] {norm_mode} completed in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        return True
    else:
        print(f"  [FAIL] {norm_mode} returned {result.returncode} after {elapsed:.0f}s")
        return False


def main():
    ap = argparse.ArgumentParser("CNN normalization ablation runner for LOSO")
    ap.add_argument("--include-global", action="store_true",
                    help="Also run the global condition (results_cnn_loso/). "
                         "Default: skip — global results already exist.")
    ap.add_argument("--npz", default=DEFAULT_NPZ,
                    help="Path to raw windows NPZ file")
    ap.add_argument("--meta", default=DEFAULT_META,
                    help="Path to meta CSV file")
    ap.add_argument("--conditions", default=None,
                    help="Comma-separated list of conditions to run (e.g. 'none,robust')")
    args = ap.parse_args()

    # Build condition list
    if args.conditions:
        run_set = {c.strip() for c in args.conditions.split(",")}
        conditions = [(nm, od) for nm, od in CONDITIONS if nm in run_set]
    else:
        conditions = list(CONDITIONS)

    # Skip global unless explicitly requested
    if not args.include_global:
        conditions = [(nm, od) for nm, od in conditions if nm != "global"]

    if not conditions:
        print("[ERROR] No conditions to run.")
        sys.exit(1)

    print(f"CNN normalization ablation: {len(conditions)} condition(s)")
    print(f"NPZ:        {args.npz}")
    print(f"Meta:       {args.meta}")
    print(f"Conditions: {[c[0] for c in conditions]}")
    print(f"Est. time:  ~{len(conditions) * 5}–{len(conditions) * 10} hours on GPU")

    results = []
    for norm_mode, out_dir in conditions:
        ok = run_cnn_loso(norm_mode, out_dir, args.npz, args.meta)
        results.append((norm_mode, ok))

    # Summary
    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    for norm_mode, ok in results:
        status = "OK" if ok else "FAIL"
        print(f"  {norm_mode:15s} -> {status}")

    failed = [r for r in results if not r[1]]
    if failed:
        print(f"\n  {len(failed)} condition(s) FAILED. Check logs above.")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} condition(s) completed successfully!")
        print(f"  Now run: python compare_cnn_norm_ablation.py")


if __name__ == "__main__":
    main()
