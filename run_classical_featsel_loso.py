# run_classical_featsel_loso.py
"""
Orchestrator: runs classical LOSO (SVM + RF) with feature selection across
four conditions (rfe_36, rfe_27, mi_36, mi_27) using per-subject z-score
normalization and freq-72 features.

Usage
-----
    python run_classical_featsel_loso.py
    python run_classical_featsel_loso.py --skip-conditions rfe_27,mi_27
    python run_classical_featsel_loso.py --models SVM
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

# Fixed feature / meta paths (freq-72 extended feature set)
FEATURES = str(ROOT / "features_out" /
               "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz")
META = str(ROOT / "features_out" /
           "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv")

# Condition definitions: name -> (feat_sel, n_features)
CONDITIONS: dict[str, tuple[str, int]] = {
    "rfe36": ("rfe", 36),
    "rfe27": ("rfe", 27),
    "mi36":  ("mi",  36),
    "mi27":  ("mi",  27),
}


def run_condition(
    condition_name: str,
    feat_sel: str,
    n_features: int,
    models: str,
    skip_set: set[str],
) -> float:
    """Launch one condition and return wall-clock seconds elapsed."""
    if condition_name in skip_set:
        print(f"[skip] condition '{condition_name}' is in --skip-conditions list")
        return 0.0

    out_dir = ROOT / f"results_loso_freq_{condition_name}"
    cmd = [
        sys.executable, str(ROOT / "train_classical_loso.py"),
        "--features",    FEATURES,
        "--meta",        META,
        "--out",         str(out_dir),
        "--models",      models,
        "--norm-mode",   "per_subject",
        "--inner-splits", "5",
        "--feat-sel",    feat_sel,
        "--n-features",  str(n_features),
        "--resume",
    ]

    print(f"\n{'='*70}")
    print(f"[run] condition={condition_name}  feat_sel={feat_sel}  n_features={n_features}")
    print(f"[run] models={models}  out={out_dir}")
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
        description="Orchestrate classical LOSO with feature selection (freq-72, per_subject norm)."
    )
    ap.add_argument(
        "--skip-conditions",
        default="",
        help="Comma-separated list of condition names to skip "
             "(choices: rfe36, rfe27, mi36, mi27). Default: empty (run all).",
    )
    ap.add_argument(
        "--models",
        default="SVM,RF",
        help="Comma-separated models to pass to train_classical_loso.py (default: SVM,RF)",
    )
    args = ap.parse_args()

    skip_set = {c.strip() for c in args.skip_conditions.split(",") if c.strip()}

    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    for cond_name, (feat_sel, n_feat) in CONDITIONS.items():
        elapsed = run_condition(
            condition_name=cond_name,
            feat_sel=feat_sel,
            n_features=n_feat,
            models=args.models,
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
    print("DONE — feature selection ablation complete.")


if __name__ == "__main__":
    main()
