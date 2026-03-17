# validate_npz.py
# Validation for:
#   windows_WAK_UPS_DNS_STDUP_v1.npz
#   windows_WAK_UPS_DNS_STDUP_v1_meta.csv 
from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Defaults for "one-click" run
# The script auto-selects the newest windows_*.npz if these exact paths don't exist,
# so the defaults below are intentionally kept as fallback names.  The actual
# preprocessed files follow the naming pattern:
#   windows_WAK_UPS_DNS_STDUP_v1_w<W>_ov50_conf60_AorR.npz
#   windows_WAK_UPS_DNS_STDUP_v1_meta_w<W>_ov50_conf60_AorR.csv
# e.g. w250 → windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_NPZ = PROJECT_ROOT / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz"
DEFAULT_META = PROJECT_ROOT / "windows_WAK_UPS_DNS_STDUP_v1_meta_w250_ov50_conf60_AorR.csv"

def _pick_latest_windows_npz(project_root: Path) -> Path | None:
    candidates = sorted(project_root.glob("windows_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _default_meta_for(npz_path: Path) -> Path | None:
    # Convention A: <npz_stem>_meta.csv
    a = npz_path.with_name(f"{npz_path.stem}_meta.csv")
    if a.exists():
        return a

    # Convention B: <base>_meta_<tag>.csv
    stem = npz_path.stem
    if "_w" in stem:
        base, tag = stem.split("_w", 1)
        b = npz_path.with_name(f"{base}_meta_w{tag}.csv")
        if b.exists():
            return b

    return None



# Helpers
def load_npz(npz_path: Path) -> dict:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    # keep pickle OFF
    data = np.load(npz_path, allow_pickle=False)
    keys = set(data.files)

    # only require numeric window arrays
    required = {"X_raw", "X_env"}
    missing = required - keys
    if missing:
        raise KeyError(f"NPZ missing keys {missing}. Found keys: {sorted(keys)}")

    # only materialize numeric arrays
    out = {"X_raw": data["X_raw"], "X_env": data["X_env"]}
    return out



def safe_read_meta_csv(meta_path: Path | None) -> pd.DataFrame | None:
    if meta_path is None:
        print("[WARN] No meta CSV provided / found. Continuing without meta.")
        return None
    if not meta_path.exists():
        print(f"[WARN] Meta CSV not found: {meta_path}. Continuing without meta.")
        return None

    try:
        df = pd.read_csv(meta_path)
        print(f"[INFO] Loaded meta CSV: {meta_path} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"[WARN] Failed to read meta CSV: {meta_path} ({e}). Continuing without meta.")
        return None



def summarize_npz(npz: dict, meta_df: pd.DataFrame | None = None) -> None:
    X_raw = npz["X_raw"]
    X_env = npz["X_env"]

    print("\n==== NPZ SUMMARY ====")
    print(f"X_raw: shape={X_raw.shape}, dtype={X_raw.dtype}")
    print(f"X_env: shape={X_env.shape}, dtype={X_env.dtype}")

    # consistency checks
    n = X_raw.shape[0]
    if X_env.shape[0] != n:
        raise ValueError(f"Inconsistent N: X_raw={n}, X_env={X_env.shape[0]}")

    if X_raw.ndim != 3 or X_env.ndim != 3:
        raise ValueError("Expected X_raw and X_env to be 3D: (N, C, T)")

    # Enforce canonical NCT
    if X_raw.shape[1] <= 1 or X_raw.shape[2] <= 1:
        raise ValueError(f"Suspicious X_raw shape for NCT: {X_raw.shape}")


    print(f"\nN windows: {n}")
    print(f"# channels: {X_raw.shape[1]}")          # C
    print(f"Window length (samples): {X_raw.shape[2]}")  # T


    # label counts from meta CSV (if present)
    if meta_df is not None and "movement" in meta_df.columns:
        print("\nClass counts (meta_df['movement']):")
        print(meta_df["movement"].value_counts(dropna=False).to_string())
    else:
        print("\n[INFO] No meta CSV (or no 'movement' col) → skipping class counts.")



def summarize_meta_df(meta_df: pd.DataFrame) -> None:
    print("\n==== META CSV SUMMARY ====")
    print("Columns:", list(meta_df.columns))
    print("\nFirst 5 rows:")
    print(meta_df.head(5).to_string(index=False))

    # Common columns expected from preprocessing pipeline
    cols = set(meta_df.columns)

    if "movement" in cols:
        print("\nWindows per movement:")
        print(meta_df["movement"].value_counts(dropna=False).to_string())

    if "subject" in cols:
        print("\nWindows per subject (top 10):")
        print(meta_df["subject"].value_counts().head(10).to_string())

    if "status_mode" in cols:
        print("\nStatus_mode distribution (overall):")
        print(meta_df["status_mode"].value_counts(dropna=False).to_string())

    if "status_mode" in cols and "movement" in cols:
        print("\nStatus_mode by movement (crosstab):")
        ct = pd.crosstab(meta_df["movement"], meta_df["status_mode"], dropna=False)
        print(ct.to_string())


def pick_indices(
    n: int,
    meta_df: pd.DataFrame | None,
    movement: str | None,
    subject: int | None,
    k: int,
    seed: int = 7
) -> list[int]:
    rng = random.Random(seed)

    if meta_df is None:
        return [rng.randrange(0, n) for _ in range(min(k, n))]

    df = meta_df.copy()

    if movement is not None and "movement" in df.columns:
        df = df[df["movement"].astype(str).str.upper() == movement.upper()]

    if subject is not None and "subject" in df.columns:
        df = df[df["subject"].astype(int) == int(subject)]

    if len(df) == 0:
        print("\n[WARN] No rows matched the requested filters. Falling back to random picks from full dataset.")
        return [rng.randrange(0, n) for _ in range(min(k, n))]

    # meta_df row index is assumed to align with NPZ window index 
    candidates = df.index.to_list()
    rng.shuffle(candidates)
    return candidates[: min(k, len(candidates))]


def plot_windows(
    X_raw: np.ndarray,
    X_env: np.ndarray,
    meta_df: pd.DataFrame | None,
    indices: list[int],
    channels_to_plot: int = 3
) -> None:
    # NCT: (N, C, T)
    n_channels = X_raw.shape[1]
    c = min(channels_to_plot, n_channels)

    for idx in indices:
        raw = X_raw[idx]  # (C, T)
        env = X_env[idx]  # (C, T)

        label = None
        if meta_df is not None:
            if "movement" in meta_df.columns:
                try:
                    label = str(meta_df.loc[idx, "movement"])
                except Exception:
                    label = None

        title_bits = [f"idx={idx}"]
        if label is not None:
            title_bits.append(f"y={label}")

        if meta_df is not None:
            row = meta_df.loc[idx]
            if "subject" in meta_df.columns:
                try:
                    title_bits.append(f"Sub{int(row['subject']):02d}")
                except Exception:
                    pass
            if "movement" in meta_df.columns:
                title_bits.append(str(row["movement"]))
            if "status_mode" in meta_df.columns:
                title_bits.append(f"status_mode={row['status_mode']}")
            if "confidence" in meta_df.columns:
                try:
                    title_bits.append(f"conf={float(row['confidence']):.2f}")
                except Exception:
                    pass


        fig = plt.figure(figsize=(11, 6))
        fig.suptitle(" | ".join(title_bits), fontsize=12)

        # Raw
        ax1 = fig.add_subplot(2, 1, 1)
        for ch in range(c):
            ax1.plot(raw[ch, :], linewidth=1.0, label=f"raw ch{ch+1}")
        ax1.set_ylabel("Raw (filtered)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="upper right")

        # Envelope
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        for ch in range(c):
            ax2.plot(env[ch, :], linewidth=1.0, label=f"env ch{ch+1}")
        ax2.set_xlabel("Samples (within window)")
        ax2.set_ylabel("Envelope")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="upper right")

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()


# Main
def main():
    parser = argparse.ArgumentParser(description="Validate SIAT-LLMD NPZ windows (one-click defaults).")
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ), help="Path to NPZ file")
    parser.add_argument("--meta", type=str, default=str(DEFAULT_META), help="Path to meta CSV (optional)")
    parser.add_argument("--movement", type=str, default=None, help="Filter plots by movement, e.g. WAK")
    parser.add_argument("--subject", type=int, default=None, help="Filter plots by subject, e.g. 12")
    parser.add_argument("--n-plots", type=int, default=6, help="How many windows to plot")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for picking windows")
    parser.add_argument("--channels", type=int, default=3, help="How many channels to plot per window")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    meta_path = Path(args.meta)

    if not npz_path.exists():
        picked = _pick_latest_windows_npz(PROJECT_ROOT)
        if picked is None:
            raise FileNotFoundError(
                f"NPZ not found: {npz_path}\n"
                f"Also couldn't find any windows_*.npz in: {PROJECT_ROOT}"
            )
        print(f"[INFO] Default NPZ not found. Auto-selected latest: {picked.name}")
        npz_path = picked

    # Try to pair meta CSV with the selected NPZ if meta doesn't exist
    if not meta_path.exists():
        paired = _default_meta_for(npz_path)
        if paired is not None:
            meta_path = paired
            print(f"[INFO] Using paired meta CSV: {meta_path.name}")
        else:
            print(f"[WARN] Meta CSV not found: {meta_path} (and no paired meta found). Continuing without meta.")
            meta_path = None


    npz = load_npz(npz_path)


    meta_df = safe_read_meta_csv(meta_path)
    summarize_npz(npz, meta_df)

    if meta_df is None:
        print(f"\n[INFO] Meta CSV not found (ok): {meta_path}")
        print("       Plot selection will be random only (no movement/subject filtering).")
    else:
        summarize_meta_df(meta_df)

        if len(meta_df) != npz["X_raw"].shape[0]:
            print(
                f"\n[WARN] Meta CSV rows ({len(meta_df)}) != NPZ windows ({npz['X_raw'].shape[0]}). "
                "Filtering by movement/subject may be unreliable."
            )

    indices = pick_indices(
        n=npz["X_raw"].shape[0],
        meta_df=meta_df,
        movement=args.movement,
        subject=args.subject,
        k=args.n_plots,
        seed=args.seed,
    )

    print("\nPlotting window indices:", indices)

    y_plot = None
    if meta_df is not None:
        if "movement" in meta_df.columns:
            y_plot = meta_df["movement"].to_numpy()
        elif "mode_label" in meta_df.columns:
            y_plot = meta_df["mode_label"].to_numpy()

    plot_windows(
        X_raw=npz["X_raw"],
        X_env=npz["X_env"],
        meta_df=meta_df,
        indices=indices,
        channels_to_plot=args.channels,
    )


if __name__ == "__main__":
    main()
