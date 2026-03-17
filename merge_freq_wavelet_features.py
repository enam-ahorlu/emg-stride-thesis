#!/usr/bin/env python3
"""
Merge the existing extended features (with wavelet) and new frequency features
into a single combined feature set.

Existing ext: base(36) + WAMP(9) + wavelet(9) = 54 dims
New freq ext: base(36) + WAMP(9) + MNF(9) + MDF(9) + spectral_power(9) = 72 dims

Combined:     base(36) + WAMP(9) + wavelet(9) + MNF(9) + MDF(9) + spectral_power(9) = 81 dims

Usage:
  python merge_freq_wavelet_features.py

Creates: features_out/combined_*_features_full.npz  (81 dims)
"""
import os
import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
FOUT = os.path.join(PROJ, "features_out")

# 250ms windows
OLD_EXT = os.path.join(FOUT, "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz")
NEW_FREQ = os.path.join(FOUT, "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz")
OUT_COMBINED = os.path.join(FOUT, "combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_full.npz")

# The meta file is the same for both (same windows, just different features)
META = os.path.join(FOUT, "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv")

if not os.path.exists(OLD_EXT):
    print(f"[ERROR] Existing extended features not found: {OLD_EXT}")
    print("  Run extract_features.py with wavelet first.")
    exit(1)

if not os.path.exists(NEW_FREQ):
    print(f"[ERROR] New frequency features not found: {NEW_FREQ}")
    print("  Run extract_features.py with --freq first.")
    exit(1)

# Load both
X_old = np.load(OLD_EXT)["X"]  # (N, 54) = base(36) + WAMP(9) + wavelet(9)
X_new = np.load(NEW_FREQ)["X"]  # (N, 72) = base(36) + WAMP(9) + MNF(9) + MDF(9) + SP(9)

assert X_old.shape[0] == X_new.shape[0], f"Row mismatch: {X_old.shape[0]} vs {X_new.shape[0]}"

print(f"[load] Old extended: {X_old.shape}  (base + WAMP + wavelet)")
print(f"[load] New freq ext: {X_new.shape}  (base + WAMP + MNF + MDF + SP)")

# Old: cols 0-35 = base, 36-44 = WAMP, 45-53 = wavelet
# New: cols 0-35 = base, 36-44 = WAMP, 45-53 = MNF, 54-62 = MDF, 63-71 = SP
# Combined: base(36) + WAMP(9) + wavelet(9) + MNF(9) + MDF(9) + SP(9) = 81

wavelet_cols = X_old[:, 45:54]   # 9 cols of wavelet energy
freq_cols = X_new[:, 45:]         # 27 cols: MNF(9) + MDF(9) + SP(9)
base_wamp = X_new[:, :45]         # 45 cols: base(36) + WAMP(9)

X_combined = np.concatenate([base_wamp, wavelet_cols, freq_cols], axis=1)
print(f"[combined] shape: {X_combined.shape}  (base + WAMP + wavelet + MNF + MDF + SP)")

np.savez_compressed(OUT_COMBINED, X=X_combined.astype(np.float32))
print(f"[save] {OUT_COMBINED}")

# Copy/symlink meta to match combined name (so pipeline can find it consistently)
import shutil
OUT_META = os.path.join(FOUT, "combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv")
if not os.path.exists(OUT_META):
    shutil.copy2(META, OUT_META)
    print(f"[copy] meta → {OUT_META}")
else:
    print(f"[info] Meta already exists: {OUT_META}")

print("[DONE]")
