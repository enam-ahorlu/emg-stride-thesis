#!/usr/bin/env python3
"""
Rerun LOSO with all fixes applied:
  1. Feature extraction with frequency-domain features (MNF, MDF, spectral power)
  2. StandardScaler always-on in classical LOSO pipeline (leak-free)
  3. Stop-start method for memory management

Usage:
  python rerun_loso_with_fixes.py --step extract    # Step 1: extract features
  python rerun_loso_with_fixes.py --step loso-svm   # Step 2: run SVM LOSO (stop-start)
  python rerun_loso_with_fixes.py --step loso-rf    # Step 3: run RF LOSO (stop-start)
  python rerun_loso_with_fixes.py --step loso-cnn   # Step 4: run CNN LOSO (stop-start)
  python rerun_loso_with_fixes.py --step all        # Run everything in sequence

The stop-start method:
  For each LOSO model, we run a batch of N subjects, then kill and restart
  the process to free memory. The --resume flag in train_classical_loso.py
  and train_cnn_loso.py handles picking up from where we left off.
"""

import argparse
import subprocess
import sys
import os
import time

PROJ = os.path.dirname(os.path.abspath(__file__))

# --- Config ---
NPZ_250 = os.path.join(PROJ, "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz")
META_250 = os.path.join(PROJ, "windows_WAK_UPS_DNS_STDUP_v1_meta_w250_ov50_conf60_AorR.csv")
FEATURES_DIR = os.path.join(PROJ, "features_out")

# Output dirs for the fixed LOSO runs (separate from original results)
LOSO_OUT_CLASSICAL = os.path.join(PROJ, "results_loso_fixed")
LOSO_OUT_CNN = os.path.join(PROJ, "results_cnn_loso_fixed")

# Stop-start: run this many subjects before restarting to free memory
BATCH_SIZE_CLASSICAL = 10  # SVM/RF: restart every 10 subjects
BATCH_SIZE_CNN = 5         # CNN: restart every 5 subjects (heavier memory)

TOTAL_SUBJECTS = 40


def run_cmd(cmd, desc=""):
    """Run a command, print output, return exit code."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=PROJ)
    return result.returncode


def step_extract():
    """Extract features with frequency-domain features + wavelet."""
    print("\n[STEP] Feature extraction with frequency-domain features")

    # With wavelet + frequency (full extended set)
    rc = run_cmd([
        sys.executable, "extract_features.py",
        "--npz", NPZ_250,
        "--meta", META_250,
        "--out-dir", FEATURES_DIR,
        "--prefix", "freq",
        "--use", "raw",
        "--freq",
        "--fs", "2000.0",
        # wavelet is on by default; will fail if pywt not installed
    ], desc="Extracting features (base + WAMP + wavelet + MNF + MDF + spectral_power)")

    if rc != 0:
        print("[WARN] Extraction failed (likely missing pywt). Retrying without wavelet...")
        rc = run_cmd([
            sys.executable, "extract_features.py",
            "--npz", NPZ_250,
            "--meta", META_250,
            "--out-dir", FEATURES_DIR,
            "--prefix", "freq",
            "--use", "raw",
            "--freq",
            "--no-wavelet",
            "--fs", "2000.0",
        ], desc="Extracting features (base + WAMP + MNF + MDF + spectral_power, no wavelet)")

    return rc


def step_loso_classical(model_name):
    """
    Run classical LOSO with stop-start method.
    model_name: "SVM" or "RF"
    """
    print(f"\n[STEP] Classical LOSO for {model_name} (stop-start method)")

    features_file = os.path.join(FEATURES_DIR,
        "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz")
    meta_file = os.path.join(FEATURES_DIR,
        "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv")

    if not os.path.exists(features_file):
        # Fall back to base features if extended not available
        features_file = os.path.join(FEATURES_DIR,
            "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_base.npz")
        print(f"[INFO] Extended features not found, using base: {features_file}")

    os.makedirs(LOSO_OUT_CLASSICAL, exist_ok=True)

    batch_num = 0
    while True:
        batch_num += 1
        print(f"\n--- {model_name} batch {batch_num} (subjects resume automatically) ---")

        rc = run_cmd([
            sys.executable, "train_classical_loso.py",
            "--features", features_file,
            "--meta", meta_file,
            "--out", LOSO_OUT_CLASSICAL,
            "--models", model_name,
            # StandardScaler is now always-on (no --no-scale flag)
            "--inner-splits", "5",
            "--save-preds",
            "--cv-scheme", "loso",
            "--resume",
            "--flush-preds",
        ], desc=f"{model_name} LOSO batch {batch_num}")

        if rc == 0:
            print(f"[DONE] {model_name} LOSO completed successfully!")
            break
        else:
            print(f"[WARN] {model_name} batch {batch_num} exited with code {rc}.")
            print("  If this was a memory issue, it will resume from the next subject.")
            print("  Waiting 5 seconds before next batch...")
            time.sleep(5)

            # Safety: check if all subjects are done
            import pandas as pd
            ckpt_dir = os.path.join(LOSO_OUT_CLASSICAL, "checkpoints")
            ckpt_glob = os.path.join(ckpt_dir, f"*__{model_name}_subjectwise_ckpt.csv")
            import glob
            ckpts = glob.glob(ckpt_glob)
            if ckpts:
                df = pd.read_csv(ckpts[0])
                done = set(df["heldout_subject"].astype(int).tolist())
                if len(done) >= TOTAL_SUBJECTS:
                    print(f"[DONE] All {TOTAL_SUBJECTS} subjects completed for {model_name}!")
                    break
                print(f"  Completed {len(done)}/{TOTAL_SUBJECTS} subjects so far.")
            else:
                print("  No checkpoint found yet.")

            if batch_num > 20:
                print("[ABORT] Too many batches. Something is wrong.")
                return 1

    return 0


def step_loso_cnn():
    """Run CNN LOSO with stop-start method."""
    print("\n[STEP] CNN LOSO (stop-start method)")

    os.makedirs(LOSO_OUT_CNN, exist_ok=True)

    batch_num = 0
    while True:
        batch_num += 1
        print(f"\n--- CNN batch {batch_num} (subjects resume automatically) ---")

        rc = run_cmd([
            sys.executable, "train_cnn_loso.py",
            "--npz", NPZ_250,
            "--meta", META_250,
            "--xkey", "X_env",
            "--label-col", "movement",
            "--out", LOSO_OUT_CNN,
            "--epochs", "25",
            "--batch", "512",
            "--lr", "1e-3",
            "--val-frac", "0.15",
            "--seed", "42",
            "--resume",
            "--num-workers", "0",  # reduce memory overhead from workers
        ], desc=f"CNN LOSO batch {batch_num}")

        if rc == 0:
            print("[DONE] CNN LOSO completed successfully!")
            break
        else:
            print(f"[WARN] CNN batch {batch_num} exited with code {rc}.")
            time.sleep(5)

            import pandas as pd
            metrics_csv = os.path.join(LOSO_OUT_CNN, "per_subject_metrics_cnn_loso.csv")
            if os.path.exists(metrics_csv):
                df = pd.read_csv(metrics_csv)
                done = set(df["subject"].astype(int).tolist())
                if len(done) >= TOTAL_SUBJECTS:
                    print(f"[DONE] All {TOTAL_SUBJECTS} subjects completed for CNN!")
                    break
                print(f"  Completed {len(done)}/{TOTAL_SUBJECTS} subjects so far.")

            if batch_num > 20:
                print("[ABORT] Too many batches. Something is wrong.")
                return 1

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True,
                     choices=["extract", "loso-svm", "loso-rf", "loso-cnn", "all"],
                     help="Which step to run")
    args = ap.parse_args()

    if args.step == "extract":
        step_extract()
    elif args.step == "loso-svm":
        step_loso_classical("SVM")
    elif args.step == "loso-rf":
        step_loso_classical("RF")
    elif args.step == "loso-cnn":
        step_loso_cnn()
    elif args.step == "all":
        print("\n" + "="*60)
        print("  RUNNING ALL STEPS IN SEQUENCE")
        print("="*60)
        step_extract()
        step_loso_classical("SVM")
        step_loso_classical("RF")
        step_loso_cnn()
        print("\n[ALL DONE] All steps completed.")


if __name__ == "__main__":
    main()
