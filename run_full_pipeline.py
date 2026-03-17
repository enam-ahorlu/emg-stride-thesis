#!/usr/bin/env python3
"""
run_full_pipeline.py — Master orchestration script
====================================================
Runs the complete updated LOSO pipeline, one model at a time, sequentially.

Phases:
  Phase 1: Feature extraction (frequency-enhanced, with wavelet if pywt available)
  Phase 2: Merge wavelet + freq features into 81-dim set (if pywt available)
  Phase 3: CNN LOSO (uses raw signals, not features — unchanged pipeline)
  Phase 4: SVM LOSO on base (36-dim) + freq-enhanced (72-dim)  [StandardScaler ON]
  Phase 5: RF  LOSO on base (36-dim) + freq-enhanced (72-dim)  [StandardScaler ON]
  Phase 6: Feature set comparison (base vs freq-enhanced)
  Phase 7: Post-analysis (statistics, plots, generalization gaps)

Models run ONE AT A TIME, fully sequentially as requested.
Each model completes for ALL 40 subjects before the next model starts.
Uses stop-start batching to manage memory on long runs.

Usage:
  Activate your venv first, then:

  python run_full_pipeline.py                     # Run everything
  python run_full_pipeline.py --skip-cnn           # Skip CNN (if already done)
  python run_full_pipeline.py --skip-extract       # Skip feature extraction (if already done)
  python run_full_pipeline.py --only-compare       # Just run comparison + analysis (after LOSO done)
  python run_full_pipeline.py --phase 3            # Run only phase 3 (CNN)
  python run_full_pipeline.py --phase 4            # Run only phase 4 (SVM)
  python run_full_pipeline.py --phase 5            # Run only phase 5 (RF)
"""

import argparse
import subprocess
import sys
import os
import time
import gc
import glob
from pathlib import Path
from datetime import datetime

PROJ = Path(__file__).parent.resolve()
FEATURES_DIR = PROJ / "features_out"

# Raw windowed data (for CNN and feature extraction)
NPZ_250 = PROJ / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"
META_250 = PROJ / "windows_WAK_UPS_DNS_STDUP_v1_meta_w250_ov50_conf60_AorR.csv"

# Feature files for classical models
FEAT_BASE_250 = FEATURES_DIR / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_base.npz"
FEAT_BASE_META = FEATURES_DIR / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
FEAT_FREQ_250 = FEATURES_DIR / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
FEAT_FREQ_META = FEATURES_DIR / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
FEAT_FULL_250 = FEATURES_DIR / "combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_full.npz"

# Output directories (separate from original results to preserve baselines)
OUT_CNN       = PROJ / "results_cnn_loso_fixed"
OUT_BASE      = PROJ / "results_loso_base_fixed"    # base features with StandardScaler
OUT_FREQ      = PROJ / "results_loso_freq"          # freq-enhanced features
OUT_FULL      = PROJ / "results_loso_full"          # 81-dim (if available)

TOTAL_SUBJECTS = 40
LOG_FILE = PROJ / "pipeline_log.txt"


def log(msg: str):
    """Print and log to file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd, desc="", timeout=None):
    """Run a subprocess, print output live, return exit code."""
    log(f"START: {desc}")
    log(f"  CMD: {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJ), timeout=timeout)
    elapsed = time.perf_counter() - t0
    log(f"  DONE in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode


def check_subjects_done(out_dir, model_name, features_stem):
    """Check how many subjects have been completed from checkpoints."""
    ckpt = out_dir / "checkpoints" / f"{features_stem}__{model_name}_subjectwise_ckpt.csv"
    if ckpt.exists():
        import pandas as pd
        df = pd.read_csv(ckpt)
        if "heldout_subject" in df.columns:
            return set(df["heldout_subject"].astype(int).tolist())
    return set()


def check_cnn_subjects_done(out_dir):
    """Check how many subjects have been completed for CNN LOSO."""
    metrics = out_dir / "per_subject_metrics_cnn_loso.csv"
    if metrics.exists():
        import pandas as pd
        df = pd.read_csv(metrics)
        if "subject" in df.columns:
            return set(df["subject"].astype(int).tolist())
    return set()


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Feature Extraction
# ═══════════════════════════════════════════════════════════════════
def phase1_extract():
    """Extract frequency-enhanced features (with wavelet fallback)."""
    log("=" * 60)
    log("PHASE 1: Feature Extraction (frequency-enhanced)")
    log("=" * 60)

    if FEAT_FREQ_250.exists():
        log(f"  Freq features already exist: {FEAT_FREQ_250}")
        log("  Skipping extraction. Delete the file to force re-extraction.")
        return 0

    # Try with wavelet first
    rc = run_cmd([
        sys.executable, "extract_features.py",
        "--npz", str(NPZ_250),
        "--meta", str(META_250),
        "--out-dir", str(FEATURES_DIR),
        "--prefix", "freq",
        "--use", "raw",
        "--freq",
        "--fs", "2000.0",
    ], desc="Feature extraction (base+WAMP+wavelet+freq → 81 dims)")

    if rc != 0:
        log("  Wavelet extraction failed (likely missing pywt). Retrying without wavelet...")
        rc = run_cmd([
            sys.executable, "extract_features.py",
            "--npz", str(NPZ_250),
            "--meta", str(META_250),
            "--out-dir", str(FEATURES_DIR),
            "--prefix", "freq",
            "--use", "raw",
            "--freq",
            "--no-wavelet",
            "--fs", "2000.0",
        ], desc="Feature extraction (base+WAMP+freq → 72 dims, no wavelet)")

    return rc


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Merge wavelet + freq into 81-dim
# ═══════════════════════════════════════════════════════════════════
def phase2_merge():
    """Merge old extended (54-dim, has wavelet) + new freq features → 81-dim."""
    log("=" * 60)
    log("PHASE 2: Merge wavelet + freq → 81-dim combined features")
    log("=" * 60)

    if FEAT_FULL_250.exists():
        log(f"  Combined features already exist: {FEAT_FULL_250}")
        return 0

    rc = run_cmd([
        sys.executable, "merge_freq_wavelet_features.py",
    ], desc="Merge wavelet + freq features")

    if rc != 0:
        log("  Merge failed (likely missing pywt dependency). 81-dim set not available.")
        log("  Pipeline will proceed with 72-dim freq features instead.")
    return rc


# ═══════════════════════════════════════════════════════════════════
# Phase 3: CNN LOSO (raw signals, independent of features)
# ═══════════════════════════════════════════════════════════════════
def phase3_cnn():
    """CNN LOSO — runs on raw windowed signals, not feature vectors."""
    log("=" * 60)
    log("PHASE 3: CNN LOSO (all 40 subjects, sequential)")
    log("=" * 60)

    OUT_CNN.mkdir(parents=True, exist_ok=True)

    done = check_cnn_subjects_done(OUT_CNN)
    if len(done) >= TOTAL_SUBJECTS:
        log(f"  CNN LOSO already complete ({len(done)}/{TOTAL_SUBJECTS} subjects).")
        return 0

    log(f"  Starting CNN LOSO ({len(done)}/{TOTAL_SUBJECTS} already done)")

    batch = 0
    while True:
        batch += 1
        rc = run_cmd([
            sys.executable, "train_cnn_loso.py",
            "--npz", str(NPZ_250),
            "--meta", str(META_250),
            "--xkey", "X_env",
            "--label-col", "movement",
            "--out", str(OUT_CNN),
            "--epochs", "25",
            "--batch", "512",
            "--lr", "1e-3",
            "--val-frac", "0.15",
            "--seed", "42",
            "--resume",
            "--num-workers", "0",
        ], desc=f"CNN LOSO batch {batch}")

        if rc == 0:
            log("CNN LOSO COMPLETED SUCCESSFULLY!")
            break

        done = check_cnn_subjects_done(OUT_CNN)
        if len(done) >= TOTAL_SUBJECTS:
            log(f"  All {TOTAL_SUBJECTS} subjects done for CNN!")
            break
        log(f"  CNN batch {batch} exited (code={rc}). {len(done)}/{TOTAL_SUBJECTS} done. Retrying in 5s...")
        time.sleep(5)
        gc.collect()

        if batch > 20:
            log("[ABORT] Too many CNN batches. Check for errors.")
            return 1

    return 0


# ═══════════════════════════════════════════════════════════════════
# Phase 4 & 5: Classical LOSO (one model at a time, per feature set)
# ═══════════════════════════════════════════════════════════════════
def run_classical_loso(model_name, features_file, meta_file, out_dir, desc_label):
    """
    Run classical LOSO for one model on one feature set, with stop-start recovery.
    Fully utilizes n_jobs=-1 for RF (parallel trees), SVM uses cache_size=500.
    """
    log("-" * 50)
    log(f"  {model_name} on {desc_label}")
    log(f"  Features: {features_file}")
    log(f"  Output:   {out_dir}")
    log("-" * 50)

    if not Path(features_file).exists():
        log(f"  [SKIP] Feature file not found: {features_file}")
        return -1

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    feat_stem = Path(features_file).stem
    done = check_subjects_done(Path(out_dir), model_name, feat_stem)
    if len(done) >= TOTAL_SUBJECTS:
        log(f"  {model_name} already complete ({len(done)}/{TOTAL_SUBJECTS} subjects)")
        return 0

    log(f"  Starting {model_name} ({len(done)}/{TOTAL_SUBJECTS} already done)")

    batch = 0
    while True:
        batch += 1
        rc = run_cmd([
            sys.executable, "train_classical_loso.py",
            "--features", str(features_file),
            "--meta", str(meta_file),
            "--out", str(out_dir),
            "--models", model_name,
            "--inner-splits", "5",
            "--save-preds",
            "--cv-scheme", "loso",
            "--resume",
            "--flush-preds",
        ], desc=f"{model_name} LOSO ({desc_label}) batch {batch}")

        if rc == 0:
            log(f"  {model_name} LOSO ({desc_label}) COMPLETED!")
            break

        done = check_subjects_done(Path(out_dir), model_name, feat_stem)
        if len(done) >= TOTAL_SUBJECTS:
            log(f"  All {TOTAL_SUBJECTS} subjects done for {model_name}!")
            break
        log(f"  {model_name} batch {batch} exited (code={rc}). {len(done)}/{TOTAL_SUBJECTS} done. Retrying in 5s...")
        time.sleep(5)
        gc.collect()

        if batch > 20:
            log(f"[ABORT] Too many {model_name} batches on {desc_label}.")
            return 1

    return 0


def phase4_svm():
    """SVM LOSO on both base (36) and freq (72) feature sets."""
    log("=" * 60)
    log("PHASE 4: SVM LOSO (base-36 + freq-72, StandardScaler ON)")
    log("=" * 60)

    # SVM on base features (36-dim) — fixed pipeline
    run_classical_loso("SVM", FEAT_BASE_250, FEAT_BASE_META, OUT_BASE, "base-36dim")

    # SVM on freq-enhanced features (72-dim)
    run_classical_loso("SVM", FEAT_FREQ_250, FEAT_FREQ_META, OUT_FREQ, "freq-72dim")

    # SVM on full combined (81-dim) if available
    if FEAT_FULL_250.exists():
        full_meta = FEATURES_DIR / "combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
        if full_meta.exists():
            run_classical_loso("SVM", FEAT_FULL_250, full_meta, OUT_FULL, "full-81dim")


def phase5_rf():
    """RF LOSO on both base (36) and freq (72) feature sets."""
    log("=" * 60)
    log("PHASE 5: RF LOSO (base-36 + freq-72, StandardScaler ON)")
    log("=" * 60)

    # RF on base features (36-dim) — fixed pipeline
    run_classical_loso("RF", FEAT_BASE_250, FEAT_BASE_META, OUT_BASE, "base-36dim")

    # RF on freq-enhanced features (72-dim)
    run_classical_loso("RF", FEAT_FREQ_250, FEAT_FREQ_META, OUT_FREQ, "freq-72dim")

    # RF on full combined (81-dim) if available
    if FEAT_FULL_250.exists():
        full_meta = FEATURES_DIR / "combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
        if full_meta.exists():
            run_classical_loso("RF", FEAT_FULL_250, full_meta, OUT_FULL, "full-81dim")


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Feature Set Comparison
# ═══════════════════════════════════════════════════════════════════
def phase6_compare():
    """Compare base vs freq-enhanced LOSO results and produce comparison report."""
    log("=" * 60)
    log("PHASE 6: Feature Set Comparison (base-36 vs freq-72)")
    log("=" * 60)

    import pandas as pd
    import numpy as np

    report_dir = PROJ / "report_figs"
    report_dir.mkdir(exist_ok=True)

    rows = []

    # Collect results from each configuration
    configs = [
        ("base-36", OUT_BASE),
        ("freq-72", OUT_FREQ),
    ]
    if OUT_FULL.exists():
        configs.append(("full-81", OUT_FULL))

    for feat_label, result_dir in configs:
        for model in ["SVM", "RF"]:
            # Find summary CSV
            summaries = list(result_dir.glob(f"*__{model}_nested_loso_summary.csv"))
            if not summaries:
                log(f"  [MISSING] No summary for {model} in {result_dir}")
                continue
            s = pd.read_csv(summaries[0])
            rows.append({
                "feat_set": feat_label,
                "model": model,
                "f1_macro_mean": s["f1_macro_mean"].values[0],
                "f1_macro_sd": s["f1_macro_sd"].values[0],
                "bal_acc_mean": s["bal_acc_mean"].values[0],
                "acc_mean": s["acc_mean"].values[0],
            })

    # CNN (no feature comparison — single result)
    cnn_summary = OUT_CNN / "cnn_loso_summary.csv"
    if cnn_summary.exists():
        cs = pd.read_csv(cnn_summary)
        if "f1_macro_mean" in cs.columns:
            rows.append({
                "feat_set": "raw-signal",
                "model": "CNN",
                "f1_macro_mean": cs["f1_macro_mean"].values[0],
                "f1_macro_sd": cs.get("f1_macro_sd", cs.get("f1_macro_std", pd.Series([0.0]))).values[0],
                "bal_acc_mean": cs.get("bal_acc_mean", pd.Series([0.0])).values[0],
                "acc_mean": cs.get("acc_mean", pd.Series([0.0])).values[0],
            })

    if not rows:
        log("  No results found to compare. Run LOSO phases first.")
        return

    comp = pd.DataFrame(rows)
    comp.to_csv(report_dir / "loso_feature_comparison.csv", index=False)
    log(f"  Saved: {report_dir / 'loso_feature_comparison.csv'}")

    # Print comparison table
    log("\n  === LOSO Feature Set Comparison (w=250ms, StandardScaler ON) ===")
    log(comp.to_string(index=False))

    # Determine winner for each model
    for model in ["SVM", "RF"]:
        model_rows = comp[comp["model"] == model].sort_values("f1_macro_mean", ascending=False)
        if len(model_rows) >= 2:
            best = model_rows.iloc[0]
            second = model_rows.iloc[1]
            delta = best["f1_macro_mean"] - second["f1_macro_mean"]
            log(f"\n  {model}: Best = {best['feat_set']} (F1={best['f1_macro_mean']:.4f}), "
                f"vs {second['feat_set']} (F1={second['f1_macro_mean']:.4f}), "
                f"delta={delta:+.4f}")

    # Also compute deltas (freq - base) for each model
    delta_rows = []
    for model in ["SVM", "RF"]:
        base_r = comp[(comp["model"] == model) & (comp["feat_set"] == "base-36")]
        freq_r = comp[(comp["model"] == model) & (comp["feat_set"] == "freq-72")]
        if not base_r.empty and not freq_r.empty:
            delta_rows.append({
                "model": model,
                "delta_f1": freq_r["f1_macro_mean"].values[0] - base_r["f1_macro_mean"].values[0],
                "delta_balacc": freq_r["bal_acc_mean"].values[0] - base_r["bal_acc_mean"].values[0],
                "delta_acc": freq_r["acc_mean"].values[0] - base_r["acc_mean"].values[0],
            })

    if delta_rows:
        delta_df = pd.DataFrame(delta_rows)
        delta_df.to_csv(report_dir / "loso_feature_comparison_delta.csv", index=False)
        log("\n  Deltas (freq-72 minus base-36):")
        log(delta_df.to_string(index=False))

    # Generate bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feat_sets = comp["feat_set"].unique().tolist()
        models_in_comp = [m for m in ["SVM", "RF", "CNN"] if m in comp["model"].values]

        if len(feat_sets) >= 2 and len(models_in_comp) >= 2:
            colours = {"base-36": "#4C72B0", "freq-72": "#DD8452", "full-81": "#55A868", "raw-signal": "#C44E52"}
            x = np.arange(len(models_in_comp))
            width = 0.8 / len(feat_sets)

            fig, ax = plt.subplots(figsize=(10, 6))
            for i, fs in enumerate(feat_sets):
                vals = []
                stds = []
                for m in models_in_comp:
                    r = comp[(comp["model"] == m) & (comp["feat_set"] == fs)]
                    if not r.empty:
                        vals.append(r["f1_macro_mean"].values[0])
                        stds.append(r["f1_macro_sd"].values[0])
                    else:
                        vals.append(0)
                        stds.append(0)
                offset = (i - len(feat_sets) / 2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width, label=fs,
                              color=colours.get(fs, "#999999"), edgecolor="black",
                              linewidth=0.5, yerr=stds, capsize=4)
                for bar, v in zip(bars, vals):
                    if v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(models_in_comp, fontsize=11)
            ax.set_ylabel("Macro F1 (LOSO)")
            ax.set_title("Feature Set Comparison: LOSO Cross-Subject (w=250ms, StandardScaler ON)")
            ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(0.5, 0.85)
            plt.tight_layout()
            plt.savefig(report_dir / "loso_feature_comparison_bar.png", dpi=200, bbox_inches="tight")
            plt.close()
            log(f"  Saved: {report_dir / 'loso_feature_comparison_bar.png'}")
    except Exception as e:
        log(f"  [WARN] Could not generate comparison plot: {e}")


# ═══════════════════════════════════════════════════════════════════
# Phase 7: Post-analysis
# ═══════════════════════════════════════════════════════════════════
def phase7_analysis():
    """
    Post-analysis: compute per-subject metrics for new results,
    then run statistical tests and plot generation.

    NOTE: The existing run_all_analyses.py and merge_all_gaps.py hardcode
    results_loso_light/ and results_cnn_loso/ as input directories.
    After determining the winning feature set (Phase 6), you should either:
      (a) Copy winning results into results_loso_light/ (replacing old), OR
      (b) Update the analysis scripts to point to the winning result dir.
    This phase computes per-subject metrics for ALL new result directories
    so you can compare them before making that decision.
    """
    log("=" * 60)
    log("PHASE 7: Post-analysis (per-subject metrics, statistics, plots)")
    log("=" * 60)

    # Compute per-subject metrics for each new LOSO result directory
    for feat_label, out_dir, feat_file, meta_file in [
        ("base-36", OUT_BASE, FEAT_BASE_250, FEAT_BASE_META),
        ("freq-72", OUT_FREQ, FEAT_FREQ_250, FEAT_FREQ_META),
    ]:
        pred_dir = out_dir / "predictions"
        if pred_dir.exists() and list(pred_dir.glob("*.npy")):
            run_cmd([
                sys.executable, "compute_per_subject_metrics.py",
                "--meta", str(meta_file),
                "--pred-dir", str(pred_dir),
                "--scheme", "loso",
                "--out", str(out_dir / f"per_subject_metrics_250_{feat_label}_loso.csv"),
            ], desc=f"Per-subject metrics ({feat_label})")

    # If full-81 results exist
    if OUT_FULL.exists():
        full_meta = FEATURES_DIR / "combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
        pred_dir = OUT_FULL / "predictions"
        if pred_dir.exists() and list(pred_dir.glob("*.npy")):
            run_cmd([
                sys.executable, "compute_per_subject_metrics.py",
                "--meta", str(full_meta),
                "--pred-dir", str(pred_dir),
                "--scheme", "loso",
                "--out", str(OUT_FULL / "per_subject_metrics_250_full-81_loso.csv"),
            ], desc="Per-subject metrics (full-81)")

    # Run the existing analysis scripts on original results
    # (these still reference results_loso_light/ and results_cnn_loso/)
    log("  Running existing analysis scripts on original/canonical results...")
    run_cmd([sys.executable, "run_all_analyses.py"],
            desc="Statistical analyses (Wilcoxon, CIs, correlations)")

    run_cmd([sys.executable, "merge_all_gaps.py"],
            desc="Merge generalization gaps across models")

    run_cmd([sys.executable, "plot_results_report_figs.py"],
            desc="Generate report figures")

    log("")
    log("=" * 60)
    log("  PHASE 7 COMPLETE — NEXT STEPS:")
    log("  1. Check Phase 6 comparison (report_figs/loso_feature_comparison.csv)")
    log("  2. Determine winning feature set")
    log("  3. Copy winning results to results_loso_light/ to update canonical results")
    log("  4. Re-run: python run_all_analyses.py")
    log("=" * 60)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--skip-cnn", action="store_true", help="Skip CNN LOSO (Phase 3)")
    ap.add_argument("--skip-extract", action="store_true", help="Skip feature extraction (Phase 1-2)")
    ap.add_argument("--only-compare", action="store_true", help="Only run comparison + analysis (Phase 6-7)")
    ap.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                     help="Run only a specific phase")
    args = ap.parse_args()

    log("\n" + "=" * 60)
    log("  FULL PIPELINE — STARTED")
    log(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    if args.phase:
        phases = {
            1: phase1_extract,
            2: phase2_merge,
            3: phase3_cnn,
            4: phase4_svm,
            5: phase5_rf,
            6: phase6_compare,
            7: phase7_analysis,
        }
        phases[args.phase]()
        return

    if args.only_compare:
        phase6_compare()
        phase7_analysis()
        return

    if not args.skip_extract:
        phase1_extract()
        phase2_merge()

    if not args.skip_cnn:
        phase3_cnn()

    phase4_svm()
    phase5_rf()
    phase6_compare()
    phase7_analysis()

    log("\n" + "=" * 60)
    log("  FULL PIPELINE — COMPLETED")
    log(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("  Check pipeline_log.txt for full run log.")
    log("=" * 60)


if __name__ == "__main__":
    main()
