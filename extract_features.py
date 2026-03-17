
# extract_features.py


from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# Feature implementations
def feat_mav(x: np.ndarray) -> np.ndarray:
    """Mean Absolute Value per channel."""
    return np.mean(np.abs(x), axis=-1)

def feat_rms(x: np.ndarray) -> np.ndarray:
    """Root Mean Square per channel."""
    return np.sqrt(np.mean(x**2, axis=-1))

def feat_wl(x: np.ndarray) -> np.ndarray:
    """Waveform Length per channel."""
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)

def feat_zc(x: np.ndarray, thr: float = 1e-6) -> np.ndarray:
    """
    Zero crossings per channel with a small threshold to avoid noise flicker.
    Counts sign changes where |delta| >= thr.
    """
    # x: (C, T)
    s = np.sign(x)
    s[s == 0] = 1  # treat zeros as positive
    sign_change = (s[..., 1:] * s[..., :-1]) < 0  # (C, T-1)
    delta = np.abs(x[..., 1:] - x[..., :-1]) >= thr
    return np.sum(sign_change & delta, axis=-1).astype(np.float32)

def feat_wamp(x: np.ndarray, thr: float = 1e-6) -> np.ndarray:

    # Willison Amplitude per channel.
    return np.sum(np.abs(np.diff(x, axis=-1)) > thr, axis=-1).astype(np.float32)

def feat_entropy_shannon(x: np.ndarray, n_bins: int = 32, eps: float = 1e-12) -> np.ndarray:
    # Simple Shannon entropy per channel using histogram bins.
    C, T = x.shape
    out = np.zeros(C, dtype=np.float32)
    for c in range(C):
        hist, _ = np.histogram(x[c], bins=n_bins, density=True)
        p = hist / (np.sum(hist) + eps)
        p = p[p > 0]
        out[c] = float(-np.sum(p * np.log(p + eps)))
    return out

def feat_mean_freq(x: np.ndarray, fs: float = 2000.0) -> np.ndarray:
    """Mean Frequency (MNF) per channel via FFT.
    MNF = sum(f_k * |X_k|^2) / sum(|X_k|^2)
    """
    C, T = x.shape
    out = np.zeros(C, dtype=np.float32)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    for c in range(C):
        spectrum = np.abs(np.fft.rfft(x[c])) ** 2
        total_power = np.sum(spectrum)
        if total_power > 1e-12:
            out[c] = float(np.sum(freqs * spectrum) / total_power)
    return out


def feat_median_freq(x: np.ndarray, fs: float = 2000.0) -> np.ndarray:
    """Median Frequency (MDF) per channel via FFT.
    MDF = frequency at which cumulative PSD reaches 50%.
    """
    C, T = x.shape
    out = np.zeros(C, dtype=np.float32)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    for c in range(C):
        spectrum = np.abs(np.fft.rfft(x[c])) ** 2
        total_power = np.sum(spectrum)
        if total_power > 1e-12:
            cumulative = np.cumsum(spectrum)
            idx = np.searchsorted(cumulative, total_power * 0.5)
            idx = min(idx, len(freqs) - 1)
            out[c] = float(freqs[idx])
    return out


def feat_spectral_power(x: np.ndarray, fs: float = 2000.0) -> np.ndarray:
    """Total spectral power per channel (log-scale for numerical stability)."""
    C, T = x.shape
    out = np.zeros(C, dtype=np.float32)
    for c in range(C):
        spectrum = np.abs(np.fft.rfft(x[c])) ** 2
        total = np.sum(spectrum)
        out[c] = float(np.log1p(total))
    return out


def feat_wavelet_energy(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 2
) -> np.ndarray:

    # Wavelet energy per channel using PyWavelets.
    try:
        import pywt
    except ImportError as e:
        raise ImportError(
            "PyWavelets (pywt) is required for wavelet features.\n"
            "Install with: pip install PyWavelets"
        ) from e

    C, T = x.shape
    out = np.zeros(C, dtype=np.float32)
    for c in range(C):
        coeffs = pywt.wavedec(x[c], wavelet=wavelet, level=level)
        cD_L = coeffs[1]
        out[c] = float(np.sum(np.square(cD_L)))
    return out


# Utilities
def flatten_features(per_channel: np.ndarray) -> np.ndarray:

    return per_channel.reshape(-1)

def load_npz_windows(npz_path: Path):
 
     return np.load(npz_path, allow_pickle=False)  # keep pickle OFF long-term

def pick_window_array(npz_obj, preferred_keys: List[str]) -> Tuple[str, np.ndarray]:

    files = list(npz_obj.files)

    # Preferred keys first
    for k in preferred_keys:
        if k in files:
            try:
                arr = npz_obj[k]  
            except ValueError:

                continue
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.dtype != object:
                return k, arr

    # Fallback: any numeric (N,C,T) array
    for k in files:
        try:
            arr = npz_obj[k]
        except ValueError:
            continue
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.dtype != object:
            return k, arr

    raise KeyError(
        f"Could not find a numeric window array (N,C,T) in NPZ. "
        f"Available keys: {files}. "
        "Expected something like X_raw/X/X_env with numeric dtype."
    )



def debug_npz_keys(npz_obj, max_items: int = 60) -> None:

    print("[debug] NPZ keys/shapes/dtypes:")
    files = list(npz_obj.files)[:max_items]
    for k in files:
        try:
            arr = npz_obj[k]
            if isinstance(arr, np.ndarray):
                print(f"  - {k}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"  - {k}: (non-ndarray)")
        except ValueError as e:
            print(f"  - {k}: [SKIPPED] {e}")
    if len(npz_obj.files) > max_items:
        print(f"  ... ({len(npz_obj.files) - max_items} more keys not shown)")


def assert_alignment(X: np.ndarray, meta: pd.DataFrame) -> None:
    if len(meta) != X.shape[0]:
        raise ValueError(
            f"Meta rows ({len(meta)}) do not match windows in NPZ ({X.shape[0]}). "
            "You must use the meta CSV generated from the same preprocessing run."
        )
    

def pick_first_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
    what: str,
    required: bool = True
) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find a {what} column. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def coerce_subject_series(s: pd.Series) -> np.ndarray:

    return s.astype(str).to_numpy()



@dataclass
class FeatureConfig:
    zc_thr: float = 1e-6
    wamp_thr: float = 1e-6
    entropy_bins: int = 32
    use_entropy: bool = False
    use_wavelet: bool = True
    wavelet_name: str = "db4"
    wavelet_level: int = 2
    use_freq: bool = False       # frequency-domain features (MNF, MDF, spectral power)
    sampling_rate: float = 2000.0  # Hz — needed for frequency features


# Core extraction
def extract_one_window_features(
    x_ct: np.ndarray,
    cfg: FeatureConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_ct: (C, T) for one window
    Returns:
      base_vec, ext_vec
    """
    # Base features per channel
    mav = feat_mav(x_ct)
    rms = feat_rms(x_ct)
    wl  = feat_wl(x_ct)
    zc  = feat_zc(x_ct, thr=cfg.zc_thr)

    base = np.concatenate([mav, rms, wl, zc], axis=0)  

    ext_parts = [base]

    # Extended features
    wamp = feat_wamp(x_ct, thr=cfg.wamp_thr)
    ext_parts.append(wamp)

    if cfg.use_wavelet:
        we = feat_wavelet_energy(x_ct, wavelet=cfg.wavelet_name, level=cfg.wavelet_level)
        ext_parts.append(we)

    if cfg.use_entropy:
        ent = feat_entropy_shannon(x_ct, n_bins=cfg.entropy_bins)
        ext_parts.append(ent)

    if cfg.use_freq:
        mnf = feat_mean_freq(x_ct, fs=cfg.sampling_rate)
        mdf = feat_median_freq(x_ct, fs=cfg.sampling_rate)
        sp  = feat_spectral_power(x_ct, fs=cfg.sampling_rate)
        ext_parts.extend([mnf, mdf, sp])

    ext = np.concatenate(ext_parts, axis=0)
    return base.astype(np.float32), ext.astype(np.float32)


def extract_features(
    X: np.ndarray,
    cfg: FeatureConfig,
    max_windows: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:

    N = X.shape[0]
    if max_windows is not None:
        N = min(N, int(max_windows))

    # Pre-allocate lists 
    base_list = []
    ext_list = []

    for i in range(N):
        x_ct = X[i]
        if x_ct.ndim != 2:
            raise ValueError(f"Expected per-window array shape (C,T). Got {x_ct.shape} at i={i}")

        b, e = extract_one_window_features(x_ct, cfg)
        base_list.append(b)
        ext_list.append(e)

        if (i + 1) % 5000 == 0:
            print(f"[extract] processed {i+1}/{N} windows")

    X_base = np.stack(base_list, axis=0)
    X_ext  = np.stack(ext_list, axis=0)
    return X_base, X_ext



# Main
def main():
    parser = argparse.ArgumentParser(description="Extract EMG features from preprocessed windows NPZ + meta CSV.")
    parser.add_argument("--npz", type=str, required=True, help="Path to preprocessed windows .npz")
    parser.add_argument("--meta", type=str, required=True, help="Path to preprocessed meta CSV (1 row per window)")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--prefix", type=str, default="", help="Optional prefix for output filenames")
    parser.add_argument("--use", type=str, default="raw", choices=["raw", "env"],
                        help="Which window signal to use: raw or env (depends on NPZ keys).")
    parser.add_argument("--max-windows", type=int, default=0, help="If >0, limit number of windows for quick tests.")
    parser.add_argument("--zc-thr", type=float, default=1e-6)
    parser.add_argument("--wamp-thr", type=float, default=1e-6)
    parser.add_argument("--entropy", action="store_true", help="Include Shannon entropy feature (lightweight).")
    parser.add_argument("--entropy-bins", type=int, default=32)
    parser.add_argument("--no-wavelet", action="store_true", help="Disable wavelet-energy feature.")
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=2)
    parser.add_argument("--freq", action="store_true", help="Include frequency-domain features (MNF, MDF, spectral power).")
    parser.add_argument("--fs", type=float, default=2000.0, help="Sampling rate in Hz (for frequency features).")


    args = parser.parse_args()

    npz_path = Path(args.npz)
    meta_path = Path(args.meta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames 
    stem = (args.prefix + "_" if args.prefix else "") + npz_path.stem
    base_npz = out_dir / f"{stem}_features_base.npz"
    ext_npz  = out_dir / f"{stem}_features_ext.npz"
    meta_csv = out_dir / f"{stem}_features_meta.csv"
    cfg_json = out_dir / f"{stem}_features_cfg.json"


    meta = pd.read_csv(meta_path)
    print("[debug] Meta columns:", list(meta.columns))


    npz_obj = load_npz_windows(npz_path)
    debug_npz_keys(npz_obj)

    preferred_raw = ["X_raw", "X", "windows", "emg_windows"]
    preferred_env = ["X_env", "env_windows", "envelope_windows"]

    if args.use == "env":
        key, X = pick_window_array(npz_obj, preferred_env + preferred_raw)
    else:
        key, X = pick_window_array(npz_obj, preferred_raw + preferred_env)

    print(f"[load] Using NPZ key='{key}' with shape={X.shape} (expected N,C,T)")

    if X.ndim == 3:

        if X.shape[1] > X.shape[2]:
            X = np.transpose(X, (0, 2, 1))
            print(f"[fix] Transposed windows to (N,C,T). New shape={X.shape}")

    assert_alignment(X, meta)


    cfg = FeatureConfig(
        zc_thr=float(args.zc_thr),
        wamp_thr=float(args.wamp_thr),
        entropy_bins=int(args.entropy_bins),
        use_entropy=bool(args.entropy),
        use_wavelet=(not args.no_wavelet),
        wavelet_name=str(args.wavelet),
        wavelet_level=int(args.wavelet_level),
        use_freq=bool(args.freq),
        sampling_rate=float(args.fs),
    )

    max_windows = None if args.max_windows <= 0 else int(args.max_windows)

    X_base, X_ext = extract_features(X, cfg, max_windows=max_windows)


    meta_out = meta.iloc[: X_base.shape[0]].copy()


    if "movement" in meta_out.columns:
        label_col = "movement"
    else:
        label_col = pick_first_existing_column(
            meta_out,
            candidates=["mode_label", "label", "y", "class", "movement_label", "status_mode"],
            what="label",
        )

    subj_col = pick_first_existing_column(
        meta_out,
        candidates=["subject", "subject_id", "subj", "subj_id", "sid"],
        what="subject",
        required=False,   
    )


    y_arr = meta_out[label_col].to_numpy()

    subj_arr = None
    if subj_col is not None:
        subj_arr = coerce_subject_series(meta_out[subj_col])


    # Save NUMERIC ONLY 
    np.savez_compressed(base_npz, X=X_base.astype(np.float32))
    np.savez_compressed(ext_npz,  X=X_ext.astype(np.float32))


    # Ensure saved NPZ contains only numeric arrays
    tmp = np.load(base_npz, allow_pickle=False)
    assert list(tmp.files) == ["X"], f"Unexpected keys in {base_npz}: {tmp.files}"
    assert np.issubdtype(tmp["X"].dtype, np.number), f"Non-numeric dtype saved: {tmp['X'].dtype}"





 

    # Stable string labels for reproducibility
    meta_out["y_str"] = meta_out[label_col].astype(str)

    # Numeric labels
    classes_sorted = sorted(meta_out["y_str"].unique().tolist())
    label_to_int = {lab: i for i, lab in enumerate(classes_sorted)}
    meta_out["y_int"] = meta_out["y_str"].map(label_to_int).astype(int)


    subj_to_int = None
    if subj_col is not None:
        meta_out["subject_str"] = meta_out[subj_col].astype(str)
        subj_sorted = sorted(meta_out["subject_str"].unique().tolist())
        subj_to_int = {s: i for i, s in enumerate(subj_sorted)}
        meta_out["subject_int"] = meta_out["subject_str"].map(subj_to_int).astype(int)


    label_map_path = meta_csv.with_name(meta_csv.stem.replace("features_meta", "features_label_map") + ".json")
    payload = {"label_to_int": label_to_int, "int_to_label": {v: k for k, v in label_to_int.items()}}
    if subj_to_int is not None:
        payload["subject_to_int"] = subj_to_int
        payload["int_to_subject"] = {v: k for k, v in subj_to_int.items()}

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[save] Label map      : {label_map_path}")




    meta_out.to_csv(meta_csv, index=False)

    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(f"[save] Base features   : {base_npz}  shape={X_base.shape}")
    print(f"[save] Extended feats : {ext_npz}   shape={X_ext.shape}")
    print(f"[save] Meta copy       : {meta_csv}")
    print(f"[save] Feature config  : {cfg_json}")

    # Sanity checks
    if np.any(~np.isfinite(X_base)):
        print("[warn] Non-finite values found in BASE features.")
    if np.any(~np.isfinite(X_ext)):
        print("[warn] Non-finite values found in EXT features.")

    # Print class distribution
    counts = meta_out[label_col].value_counts()
    print("[info] Label distribution (first run):")
    print(counts.head(20))


if __name__ == "__main__":
    main()
