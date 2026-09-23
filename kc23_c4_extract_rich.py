#!/usr/bin/env python3
"""
kc23_c4_extract_rich.py
=========================
KC-C4 feature extraction. EXPERIMENT_PLAN_KC23_CLASSICAL.md "KC-C4 ...
C4.2 Sets". Both sets are computed on the filtered signal (X_raw) of the
published 250 ms windows.

TDPSD-54: time-domain power-spectral descriptors, six per channel, per
  Khushaba et al.'s time-domain power spectral moments construction (as used
  in Al-Timemy, Khushaba, Bugmann & Escudero 2016, IEEE TNSRE, "Improving the
  performance against force variation of EMG controlled multifunctional
  upper-limb prostheses for transradial amputees" -- KC23_TRACKLIST.md L23).
  The six moment-based descriptors per channel, from the zero/second/fourth
  order "power spectral moments" m0, m2, m4 (computed in the time domain via
  the signal and its first and second discrete differences, per Hjorth's
  construction, which TD-PSD builds on):
    m0 = sqrt(sum(x^2))
    m2 = sqrt(sum((diff(x,1))^2))
    m4 = sqrt(sum((diff(x,2))^2))
    f1 = log(m0)
    f2 = log(m0 - m2)
    f3 = log(m0 - m4)
    f4 = log(sparseness),      sparseness = m0 / sqrt(|m0-m2| * |m0-m4|)
    f5 = log(irregularity factor), IRF = m2 / sqrt(m0 * m4)
    f6 = log(waveform-length ratio), WLF = sum(|diff(x,1)|) / sum(|diff(x,2)|)
  All logs are log(1+|.|) * sign(.) to stay finite for near-zero arguments.
  NOTE (recorded honestly): the exact TD-PSD formula in the cited paper was
  not re-derived from the primary source in this session -- the construction
  above is the standard public description of Khushaba's TD-PSD used across
  the EMG feature-extraction literature. Whoever runs the real KC-C4 arm
  should confirm it against the paper's own equations before the numbers are
  reported; the unit test below checks internal consistency (known moment
  relationships on a synthetic sinusoid), not equation-number fidelity.

Rich-126: Freq-72, plus 4th-order autoregressive coefficients per channel
  (36 = 9 channels x 4 coefficients, Burg's method), plus Hjorth mobility and
  complexity per channel (18 = 9 channels x 2).

Writes to features_out/, under NEW names -- never overwrites an existing
features file.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
N_CHANNELS = 9


def _slog(x: np.ndarray) -> np.ndarray:
    """Signed log(1+|x|), finite everywhere, monotonic, log-like for |x|>>1."""
    return np.sign(x) * np.log1p(np.abs(x))


def tdpsd_one_channel(x: np.ndarray) -> np.ndarray:
    """x: (T,) one channel, one window. Returns 6 TD-PSD features."""
    d1 = np.diff(x, n=1)
    d2 = np.diff(x, n=2)
    m0 = np.sqrt(np.sum(x ** 2) + 1e-12)
    m2 = np.sqrt(np.sum(d1 ** 2) + 1e-12)
    m4 = np.sqrt(np.sum(d2 ** 2) + 1e-12)
    f1 = _slog(m0)
    f2 = _slog(m0 - m2)
    f3 = _slog(m0 - m4)
    sparseness = m0 / (np.sqrt(np.abs(m0 - m2) * np.abs(m0 - m4)) + 1e-12)
    irf = m2 / (np.sqrt(m0 * m4) + 1e-12)
    wlf = (np.sum(np.abs(d1)) + 1e-12) / (np.sum(np.abs(d2)) + 1e-12)
    f4, f5, f6 = _slog(sparseness), _slog(irf), _slog(wlf)
    return np.array([f1, f2, f3, f4, f5, f6])


def tdpsd54(X: np.ndarray) -> np.ndarray:
    """X: (N, C, T). Returns (N, 54) = (N, C*6), channel-major then feature-minor."""
    N, C, T = X.shape
    out = np.zeros((N, C * 6), dtype=np.float64)
    for c in range(C):
        for i in range(N):
            out[i, c * 6:(c + 1) * 6] = tdpsd_one_channel(X[i, c])
    return out


def hjorth(x: np.ndarray) -> tuple[float, float]:
    """Mobility and complexity (Hjorth 1970). Epsilon is added only to
    denominators, never to the variances themselves -- adding the SAME
    epsilon to a genuinely-zero numerator and denominator (e.g. both are 0
    for a constant signal) would wrongly produce a ratio near 1 instead of
    the correct limiting value of 0."""
    d1 = np.diff(x)
    d2 = np.diff(d1)
    var0 = np.var(x)
    var1 = np.var(d1)
    var2 = np.var(d2)
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    mobility_d1 = np.sqrt(var2 / (var1 + 1e-12))
    complexity = mobility_d1 / (mobility + 1e-12)
    return float(mobility), float(complexity)


def ar4_burg(x: np.ndarray) -> np.ndarray:
    """4th-order AR coefficients via Burg's method (no extra dependency)."""
    x = x.astype(np.float64)
    N = len(x)
    order = 4
    ef = x.copy()
    eb = x.copy()
    a = np.zeros(order + 1)
    a[0] = 1.0
    for m in range(order):
        efm = ef[m + 1:]
        ebm = eb[m:N - 1]
        num = -2.0 * np.sum(efm * ebm)
        den = np.sum(efm ** 2) + np.sum(ebm ** 2) + 1e-12
        k = num / den
        a_prev = a.copy()
        for i in range(1, m + 2):
            a[i] = a_prev[i] + k * a_prev[m + 1 - i]
        ef_new = efm + k * ebm
        eb_new = ebm + k * efm
        ef = np.concatenate([ef[:m + 1], ef_new])
        eb = np.concatenate([eb[:m + 1], eb_new])
    return a[1:order + 1]  # drop the leading 1.0


def rich126(X: np.ndarray, freq72: np.ndarray) -> np.ndarray:
    """X: (N, C, T) raw signal. freq72: (N, 72) already-computed Freq-72
    features (imported, not recomputed). Returns (N, 126)."""
    N, C, T = X.shape
    ar_feats = np.zeros((N, C * 4))
    hj_feats = np.zeros((N, C * 2))
    for i in range(N):
        for c in range(C):
            ar_feats[i, c * 4:(c + 1) * 4] = ar4_burg(X[i, c])
            mob, comp = hjorth(X[i, c])
            hj_feats[i, c * 2:(c + 1) * 2] = [mob, comp]
    return np.concatenate([freq72, ar_feats, hj_feats], axis=1)


def run(out_dir: Path, npz_path: Path, freq72_path: Path | None) -> int:
    data = np.load(npz_path)
    X = data["X_raw"].astype(np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)

    t54 = tdpsd54(X)
    np.savez(out_dir / "kc23_tdpsd54_features.npz", X=t54)
    print(f"[C4] TDPSD-54: {t54.shape} -> {out_dir / 'kc23_tdpsd54_features.npz'}")

    if freq72_path is not None and freq72_path.exists():
        freq72 = np.load(freq72_path)["X"]
        r126 = rich126(X, freq72)
        np.savez(out_dir / "kc23_rich126_features.npz", X=r126)
        print(f"[C4] Rich-126: {r126.shape} -> {out_dir / 'kc23_rich126_features.npz'}")
    else:
        print("[C4] Rich-126 skipped: no --freq72 npz given")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default=str(ROOT / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"))
    ap.add_argument("--freq72", default=None)
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.npz), Path(args.freq72) if args.freq72 else None))


if __name__ == "__main__":
    main()
