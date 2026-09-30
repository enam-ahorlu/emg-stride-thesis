#!/usr/bin/env python3
"""
src/kc23_c3_merge_proba.py
========================
KC-C3 data-prep step for src/ensemble_v2_combine.py (docs/plans/EXPERIMENT_PLAN_KC23_CLASSICAL.md
KC-C3). src/ensemble_v2_combine.py's --proba-dir must point at ONE directory
holding every ensemble member's {MODEL}_sub{K:02d}.npz -- but the KC-C3 tuned
SVM (SVM-X, results/kc23_c3_svm_per_subject/proba/) and the published
RF/CNN/RESNET_SE (results/ensemble_v2/proba/) live in different directories.
This copies (never moves/deletes) both sets' npz files into one merged
directory: the KC-C3-tuned SVM in place of the published SVM, RF/CNN/RESNET_SE
unchanged from the published run -- so the ensemble comparison tests whether a
better-tuned SVM changes which combiner/subset wins, holding every other
member fixed.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

PUBLISHED_MODELS = ["RF", "CNN", "RESNET_SE"]


def merge(new_svm_dir: Path, published_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    svm_files = sorted(new_svm_dir.glob("SVM_sub*.npz"))
    if not svm_files:
        raise FileNotFoundError(f"no SVM_sub*.npz found in {new_svm_dir}")
    for f in svm_files:
        shutil.copy(f, out_dir / f.name)
    counts["SVM"] = len(svm_files)
    for model in PUBLISHED_MODELS:
        files = sorted(published_dir.glob(f"{model}_sub*.npz"))
        if not files:
            raise FileNotFoundError(f"no {model}_sub*.npz found in {published_dir}")
        for f in files:
            shutil.copy(f, out_dir / f.name)
        counts[model] = len(files)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-svm-dir", default="results/kc23_c3_svm_per_subject/proba")
    # proba_aug_chandrop holds the augmented ResNet-SE behind the 0.8579 headline; results/ensemble_v2/proba holds the
    # UN-augmented one (soft 0.8134). The earlier default here merged the wrong one (conformance pass, 26 Sept 2026).
    ap.add_argument("--published-dir", default="results/ensemble_v2/proba_aug_chandrop")
    ap.add_argument("--out", default="results/kc23_c3_ensemble_proba")
    args = ap.parse_args()
    counts = merge(Path(args.new_svm_dir), Path(args.published_dir), Path(args.out))
    print(f"[c3-merge-proba] merged into {args.out}: {counts}")
    if len(set(counts.values())) != 1:
        print(f"[c3-merge-proba] WARNING: subject counts differ across models: {counts}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
