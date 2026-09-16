#!/usr/bin/env python3
"""Copies the published Freq-72 SVM/RF subjectwise CSVs (which carry best_params)
into a per-arm directory renamed to match that arm's own feature-file stem, so
train_classical_loso.py --save-proba --reuse-params-dir can find them.
load_best_params_lookup() requires an exact filename match on features_path.stem;
each FILTER arm's regenerated features file has a different stem (the _cbX_ceY
tag), so the published results_loso_freq_persubj/ file can't be pointed at
directly -- same fix already used for AONLY_ENSEMBLE's Phase 1 SVM re-run.
This ONLY renames a copy of the published best_params; it does not re-tune
anything, so "hyperparameters held at the published values" stays true.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
PUB_DIR = ROOT / "results_loso_freq_persubj"
PUB_STEM = "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext"
OUT_ROOT = ROOT / "results_filter_causal"

ARM_STEMS = {
    "A": "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbFalse_ceFalse_features_ext",
    "B": "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbTrue_ceFalse_features_ext",
    "C": "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbFalse_ceTrue_features_ext",
    "D": "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_cbTrue_ceTrue_features_ext",
}

for arm, stem in ARM_STEMS.items():
    dest_dir = OUT_ROOT / f"reuse_params_{arm}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for model in ("SVM", "RF"):
        src = PUB_DIR / f"{PUB_STEM}__{model}_nested_loso_subjectwise.csv"
        dst = dest_dir / f"{stem}__{model}_nested_loso_subjectwise.csv"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        print(f"  [arm {arm}] {model}: {src.name} -> {dst}")
print("done")
