# run_cnn_aug_globnorm_loso.py
"""
Orchestrator: runs CNN LOSO with data augmentation across three conditions
(gaussian, chandrop, timemask) using GLOBAL z-score normalization.

This is the independent-baseline version — same augmentation experiments
but with global norm instead of per_subject, so the improvement from
augmentation can be measured independently from the normalization improvement.

No 'combined' condition — only individual augmentation strategies.

Usage
-----
    python run_cnn_aug_globnorm_loso.py
    python run_cnn_aug_globnorm_loso.py --skip-conditions timemask
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

# Use the project .venv which has torch+CUDA; fall back to sys.executable
_VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

# Fixed NPZ / meta paths
NPZ  = str(ROOT / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz")
META = str(ROOT / "windows_WAK_UPS_DNS_STDUP_v1_meta_w250_ov50_conf60_AorR.csv")

# Conditions: name -> --augment value (no 'combined')
CONDITIONS: dict[str, str] = {
    "gaussian": "gaussian",
    "chandrop": "chandrop",
    "timemask": "timemask",
}


def run_condition(condition_name: str, aug_value: str, skip_set: set[str]) -> float:
    """Launch one CNN augmentation condition. Returns wall-clock seconds elapsed."""
    if condition_name in skip_set:
        print(f"[skip] condition '{condition_name}' is in --skip-conditions list")
        return 0.0

    out_dir = ROOT / f"results_cnn_loso_aug_{condition_name}_globnorm"
    cmd = [
        PYTHON, str(ROOT / "train_cnn_loso.py"),
        "--npz",        NPZ,
        "--meta",       META,
        "--norm-mode",  "global",
        "--augment",    aug_value,
        "--out",        str(out_dir),
        "--resume",
        "--amp",
    ]

    print(f"\n{'='*70}")
    print(f"[run] condition={condition_name}  aug={aug_value}  norm=global")
    print(f"[run] out={out_dir}")
    print(f"[cmd] {' '.join(cmd)}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"[WARN] condition '{condition_name}' exited with code {result.returncode}")
    else:
        print(f"[done] condition '{condition_name}' finished in {elapsed/60:.1f} min")

    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Orchestrate CNN LOSO with data augmentation (GLOBAL norm)."
    )
    ap.add_argument(
        "--skip-conditions",
        default="",
        help="Comma-separated list of condition names to skip "
             "(choices: gaussian, chandrop, timemask). Default: empty (run all).",
    )
    args = ap.parse_args()

    skip_set = {c.strip() for c in args.skip_conditions.split(",") if c.strip()}

    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    for cond_name, aug_val in CONDITIONS.items():
        elapsed = run_condition(
            condition_name=cond_name,
            aug_value=aug_val,
            skip_set=skip_set,
        )
        timings[cond_name] = elapsed

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*70}")
    print("[summary] Per-condition timing:")
    for cond_name, secs in timings.items():
        print(f"  {cond_name:<12} {secs/60:6.1f} min")
    print(f"  {'TOTAL':<12} {total_elapsed/60:6.1f} min")
    print(f"{'='*70}")
    print("DONE — CNN augmentation ablation (global norm) complete.")


if __name__ == "__main__":
    main()
