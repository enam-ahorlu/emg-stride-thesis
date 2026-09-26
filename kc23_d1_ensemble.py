#!/usr/bin/env python3
"""
kc23_d1_ensemble.py
=====================
KC-D1 per-seed ensemble (plan D1.3: "for each R2 realization, recompute the soft vote with the
(deterministic) SVM probabilities using ensemble_v2_combine.py, and record its F1"; C13 and the
descriptive C13b of decision D-6a). 26 September 2026.

For one seed it builds a merged probability directory in the layout ensemble_v2_combine.py needs
({MODEL}_sub{K:02d}.npz, 40 subjects, MODEL in SVM, RF, CNN, RESNET_SE):
  SVM, RF, CNN   copied from results_ensemble_v2/proba_aug_chandrop, the probabilities behind the
                 published 85.8% headline. The SVM is deterministic at fixed seed, so it is the same
                 for every realization.
  RESNET_SE      this realization's ResNet-SE+CD, from results_kc23_d1_r2_s<seed>/proba/
                 RESNET_SE_CD_sub<K>.npz (the R2 arm, --save-proba, --model-tag RESNET_SE_CD).
It then runs ensemble_v2_combine.py itself, unchanged, on that directory, and keeps the two columns C13 and
C13b use: 'SVM+RESNET_SE [soft]' and 'SVM+RESNET_SE [stacking]' (a logistic-regression meta-learner fit on the
other 39 subjects). Row alignment is asserted here as well: every subject's y_true must match across models.

Output (in --out): ensemble_s<seed>.csv with columns subject, soft, stacking (40 rows).
Any missing file, misaligned subject or failed combine exits 1 and writes nothing.

NOTE: results_ensemble_v2/proba holds the UN-augmented ResNet-SE (its soft vote is 0.8134); the 0.8579 headline
uses proba_aug_chandrop. Reading the wrong directory is exactly the mistake this script exists to avoid.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PUBLISHED_DIR = "results_ensemble_v2/proba_aug_chandrop"
COPIED_MODELS = ["SVM", "RF", "CNN"]
SOFT_COL, STACK_COL = "SVM+RESNET_SE [soft]", "SVM+RESNET_SE [stacking]"
N_SUBJECTS = 40


def build_merged(root: Path, seed: int, merged: Path) -> None:
    pub = root / PUBLISHED_DIR
    r2 = root / f"results_kc23_d1_r2_s{seed}" / "proba"
    merged.mkdir(parents=True, exist_ok=True)
    for model in COPIED_MODELS + ["RESNET_SE"]:
        files = sorted(pub.glob(f"{model}_sub*.npz"))
        if len(files) != N_SUBJECTS:
            raise FileNotFoundError(f"{pub}: {len(files)} {model}_sub*.npz files, expected {N_SUBJECTS}")
        if model != "RESNET_SE":
            for f in files:
                shutil.copy(f, merged / f.name)
    subs = [int(f.stem.split("_sub")[1]) for f in sorted(pub.glob("SVM_sub*.npz"))]
    for s in subs:
        src = r2 / f"RESNET_SE_CD_sub{s:02d}.npz"
        if not src.exists():
            raise FileNotFoundError(f"missing input: {src}")
        # context managers: an open NpzFile keeps the file locked on Windows, which makes the temporary directory's
        # cleanup raise PermissionError after an error and hide the real one
        with np.load(src) as z, np.load(merged / f"SVM_sub{s:02d}.npz") as svm:
            same = z["proba"].shape == svm["proba"].shape and np.array_equal(z["y_true"], svm["y_true"])
            shapes = (z["proba"].shape, svm["proba"].shape)
        if not same:
            raise ValueError(f"subject {s}: the R2 seed-{seed} probabilities are not row-aligned with the SVM's "
                             f"(shape {shapes[0]} against {shapes[1]}, or different y_true)")
        shutil.copy(src, merged / f"RESNET_SE_sub{s:02d}.npz")


def run(root: Path, seed: int, out_dir: Path) -> int:
    try:
        with tempfile.TemporaryDirectory(prefix=f"kc23_d1_ens_s{seed}_") as tmp:
            tmp = Path(tmp)
            merged = tmp / "proba"
            build_merged(root, seed, merged)
            combined = tmp / "combined"
            r = subprocess.run([sys.executable, str(ROOT / "ensemble_v2_combine.py"), "--proba-dir", str(merged),
                                "--out", str(combined)], cwd=root, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                raise RuntimeError(f"ensemble_v2_combine.py exited {r.returncode}: {(r.stderr or r.stdout)[-600:]}")
            sw = combined / "ensemble_v2_subjectwise.csv"
            if not sw.exists():
                raise FileNotFoundError(f"missing output: {sw}")
            df = pd.read_csv(sw)
            for c in (SOFT_COL, STACK_COL):
                if c not in df.columns:
                    raise ValueError(f"{sw.name} has no column {c!r}")
            out = df[["subject", SOFT_COL, STACK_COL]].rename(columns={SOFT_COL: "soft", STACK_COL: "stacking"})
            if len(out) != N_SUBJECTS or out["subject"].duplicated().any() or out[["soft", "stacking"]].isna().any().any():
                raise ValueError(f"the combined table needs {N_SUBJECTS} unique non-NaN subjects, found {len(out)}")
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"[D1-ensemble] FAIL, no output written: {e}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"ensemble_s{seed}.csv", index=False)
    print(f"[D1-ensemble] seed {seed}: soft {out['soft'].mean():.4f}, stacking {out['stacking'].mean():.4f} "
          f"-> {out_dir / f'ensemble_s{seed}.csv'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.root).resolve(), args.seed, Path(args.out)))


if __name__ == "__main__":
    main()
