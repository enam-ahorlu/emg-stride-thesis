#!/usr/bin/env python3
"""
kc23_s3_inventory.py
======================
KC-S3 step 1. EXPERIMENT_PLAN_KC23_DEPLOYMENT.md "KC-S3 ... S3.2 Design,
item 1": inventory the existing active-only results from RUN_MANIFEST.csv and
the S-1 outputs, as a table of family x normalization x present-or-missing.

No outcome letters (the plan states none for KC-S3); this writes
active_only_inventory.csv, which the actual KC-S3 runner reads to decide
which cells still need running.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

FAMILIES = ["LDA", "SVM", "RF", "SVMX", "HGB", "simple", "resnet_se", "resnet_se_cd"]
NORMS = ["global", "per_subject"]

# Known active-only result directories as of this session (extend as new ones
# are found or run). "Aonly" in the tag is this project's convention for the
# active-only (non-STDUP-rest) window/feature set.
KNOWN_AONLY_DIRS = {
    ("SVM", "global"): "results_aonly_global",
    ("RF", "global"): "results_aonly_global",
}


def build_inventory(manifest_path: Path = ROOT / "RUN_MANIFEST.csv") -> pd.DataFrame:
    rows = []
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    aonly_dirs = set()
    if len(manifest):
        aonly_dirs = set(manifest[manifest["dir"].astype(str).str.contains("aonly", case=False, na=False)]["dir"])

    for fam in FAMILIES:
        for norm in NORMS:
            present = False
            source = None
            if (fam, norm) in KNOWN_AONLY_DIRS:
                d = ROOT / KNOWN_AONLY_DIRS[(fam, norm)]
                if d.exists():
                    present = True
                    source = KNOWN_AONLY_DIRS[(fam, norm)]
            if not present:
                for d in aonly_dirs:
                    dl = str(d).lower()
                    if fam.lower() in dl and norm.split("_")[0] in dl:
                        present = (ROOT / d).exists()
                        source = d
                        break
            rows.append({"family": fam, "normalization": norm, "present": present, "source": source})
    return pd.DataFrame(rows)


def run(out_dir: Path) -> int:
    inv = build_inventory()
    out_dir.mkdir(parents=True, exist_ok=True)
    inv.to_csv(out_dir / "active_only_inventory.csv", index=False)
    n_missing = int((~inv["present"]).sum())
    print(f"[S3] inventory: {len(inv) - n_missing}/{len(inv)} cells present, {n_missing} missing")
    print(inv.to_string(index=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()
