
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


@dataclass
class PreprocessConfig:
    # Dataset location
    base_dir: Path

    # Filtering
    bandpass_low: float = 20.0
    bandpass_high: float = 450.0
    bandpass_order: int = 4

    # Envelope
    envelope_ms: float = 50.0

    # Windowing
    win_ms: float = 150.0
    overlap: float = 0.5

    # Label handling
    drop_none_labels: bool = True
    drop_wak_unlabelled_zero: bool = True  
    min_label_conf: float = 0.60           # confidence threshold for majority label


    keep_only_active_stdup: bool = False   # if True: STDUP windows must be majority 'A'


    align_tol: Optional[float] = None


MOVEMENTS_DEFAULT = ["WAK", "UPS", "DNS", "STDUP"]



def build_paths(base_dir: Path, subject: int, movement: str) -> Tuple[Path, Path]:

    movement = movement.upper()
    sub = f"Sub{subject:02d}"
    data_path = base_dir / sub / "Data" / f"{sub}_{movement}_Data.csv"
    label_path = base_dir / sub / "Labels" / f"{sub}_{movement}_Label.csv"
    return data_path, label_path


def load_emg_and_labels(base_dir: Path, subject: int, movement: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_path, label_path = build_paths(base_dir, subject, movement)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    data_df = pd.read_csv(data_path)
    label_df = pd.read_csv(label_path)

    # Strip whitespace in headers (common source of "Status not found")
    data_df.columns = [c.strip() for c in data_df.columns]
    label_df.columns = [c.strip() for c in label_df.columns]
    return data_df, label_df


def find_status_column(label_df: pd.DataFrame) -> str:

    cols = list(label_df.columns)
    if "Status" in cols:
        return "Status"

    for c in cols:
        if c.lower().strip() == "status":
            return c
    raise KeyError(f"No Status column found. Available columns: {cols}")


def estimate_fs(time_array: np.ndarray) -> float:

    t = np.asarray(time_array, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot estimate fs: Time column has insufficient finite values.")
    med_dt = float(np.median(dt))
    if med_dt <= 0:
        raise ValueError(f"Cannot estimate fs: non-positive median dt={med_dt}")
    return 1.0 / med_dt


def align_labels_to_data_by_time(
    data_time: pd.Series,
    label_df: pd.DataFrame,
    status_col: str,
    tol: float
) -> pd.Series:

    d = pd.DataFrame({"Time": pd.to_numeric(data_time, errors="coerce")})
    l = pd.DataFrame({
        "Time": pd.to_numeric(label_df["Time"], errors="coerce"),
        "Status": label_df[status_col]
    })

    d = d.dropna(subset=["Time"]).copy()
    l = l.dropna(subset=["Time"]).copy()

    d = d.sort_values("Time")
    l = l.sort_values("Time")

    merged = pd.merge_asof(
        d,
        l,
        on="Time",
        direction="nearest",
        tolerance=tol
    )

    # merged index is subset if data_time had NaNs, rebuild full aligned series
    aligned = pd.Series(index=pd.RangeIndex(len(data_time)), dtype=object)

    # positions where data_time is finite
    finite_mask = pd.to_numeric(data_time, errors="coerce").notna().to_numpy()
    aligned.loc[np.where(finite_mask)[0]] = merged["Status"].to_numpy()

    return aligned


def get_emg_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("sEMG")]


def load_aligned_trial(subject: int, movement: str, cfg: PreprocessConfig) -> Tuple[pd.DataFrame, float]:
    """
    Loads Data + Label, aligns Status onto data time.
    Returns (df: Time + EMG + Status, fs_est)
    """
    movement = movement.upper()
    data_df, label_df = load_emg_and_labels(cfg.base_dir, subject, movement)

    if "Time" not in data_df.columns:
        raise KeyError("Data file missing 'Time' column.")
    if "Time" not in label_df.columns:
        raise KeyError("Label file missing 'Time' column.")

    status_col = find_status_column(label_df)

    fs = estimate_fs(data_df["Time"].to_numpy())

    # default tolerance: half a sample 
    if cfg.align_tol is None:
        med_dt = 1.0 / fs
        tol = 0.5 * med_dt
    else:
        tol = float(cfg.align_tol)

    aligned_status = align_labels_to_data_by_time(data_df["Time"], label_df, status_col, tol=tol)

    out = data_df.copy()
    out["Status"] = aligned_status.to_numpy()

    emg_cols = get_emg_cols(out)
    if not emg_cols:
        raise ValueError("No sEMG columns found in data file.")

    return out[["Time"] + emg_cols + ["Status"]], fs



# SIGNAL PROCESSING
def design_bandpass(fs: float, low: float, high: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    lo = low / nyq
    hi = high / nyq
    if not (0 < lo < hi < 1):
        raise ValueError(f"Invalid bandpass edges after normalization: low={lo}, high={hi}, fs={fs}")
    b, a = butter(order, [lo, hi], btype="bandpass")
    return b, a


def apply_bandpass(df: pd.DataFrame, fs: float, cfg: PreprocessConfig) -> pd.DataFrame:
    b, a = design_bandpass(fs, cfg.bandpass_low, cfg.bandpass_high, cfg.bandpass_order)
    out = df.copy()
    emg_cols = get_emg_cols(out)

    for col in emg_cols:
        x = out[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError(f"Non-finite values in {col}. Clean input before filtering.")
        out[col] = filtfilt(b, a, x)

    return out


def rectify_and_envelope(df: pd.DataFrame, fs: float, cfg: PreprocessConfig) -> pd.DataFrame:
    out = df.copy()
    emg_cols = get_emg_cols(out)

    win_samples = int(round(cfg.envelope_ms * fs / 1000.0))
    win_samples = max(win_samples, 1)

    for col in emg_cols:
        rect = np.abs(out[col].to_numpy(dtype=float))
        out[col] = pd.Series(rect).rolling(win_samples, center=True, min_periods=1).mean().to_numpy()

    return out



# WINDOWING + LABELING
def majority_label_with_conf(status_series: pd.Series) -> Tuple[Optional[object], float]:

    s = status_series.dropna()
    if s.empty:
        return None, 0.0
    mode = s.mode()
    if mode.empty:
        return None, 0.0
    m = mode.iloc[0]
    conf = float((s == m).mean())
    return m, conf

def normalize_status_mode(x: object) -> Optional[str]:

    if x is None:
        return None

    # handle numpy/pandas NaN
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
    except Exception:
        pass

    # If already a string, clean it
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        # keep R/A
        if s.upper() in {"R", "A"}:
            return s.upper()
        # try to normalize numeric strings like "2.0"
        try:
            v = float(s)
            if np.isfinite(v) and abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return s
        except Exception:
            return s

    # Numeric types
    if isinstance(x, (int, np.integer)):
        return str(int(x))

    if isinstance(x, (float, np.floating)):
        v = float(x)
        if not np.isfinite(v):
            return None
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return str(v)

    # Fallback
    return str(x).strip() or None



def window_trial(
    df_raw_filt: pd.DataFrame,
    df_env: pd.DataFrame,
    fs: float,
    subject: int,
    movement: str,
    cfg: PreprocessConfig
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:

    movement = movement.upper()
    emg_cols = get_emg_cols(df_raw_filt)

    win_samples = int(round(cfg.win_ms * fs / 1000.0))
    win_samples = max(win_samples, 1)

    overlap = float(cfg.overlap)
    overlap = min(max(overlap, 0.0), 0.99)
    step = int(round(win_samples * (1.0 - overlap)))
    step = max(step, 1)

    t = df_raw_filt["Time"].to_numpy(dtype=float)

    X_raw_list = []
    X_env_list = []
    meta_rows: List[Dict] = []

    for start in range(0, len(df_raw_filt) - win_samples + 1, step):
        end = start + win_samples

        win_raw = df_raw_filt.iloc[start:end]
        win_env = df_env.iloc[start:end]

        mode_label, conf = majority_label_with_conf(win_raw["Status"])
        status_mode = normalize_status_mode(mode_label)



        if cfg.drop_none_labels:
            if mode_label is None or (isinstance(mode_label, float) and np.isnan(mode_label)):
                continue


        if cfg.min_label_conf is not None and cfg.min_label_conf > 0:
            if conf < float(cfg.min_label_conf):
                continue

        if cfg.drop_wak_unlabelled_zero and movement == "WAK":
            if status_mode == "0":
                continue


        if cfg.keep_only_active_stdup and movement == "STDUP":
            if status_mode != "A":
                continue


        X_raw_list.append(win_raw[emg_cols].to_numpy(dtype=np.float32))
        X_env_list.append(win_env[emg_cols].to_numpy(dtype=np.float32))

        meta_rows.append({
            "subject": int(subject),
            "movement": movement,
            "status_mode": status_mode,
            "confidence": float(conf),
            "fs": float(fs),
            "t_start": float(t[start]),
            "t_end": float(t[end - 1]),
            "win_samples": int(win_samples),
            "n_channels": int(len(emg_cols)),
        })

    if not X_raw_list:
        X_raw = np.empty((0, win_samples, len(emg_cols)), dtype=np.float32)
        X_env = np.empty((0, win_samples, len(emg_cols)), dtype=np.float32)
    else:
        X_raw = np.stack(X_raw_list, axis=0)
        X_env = np.stack(X_env_list, axis=0)

    return X_raw, X_env, meta_rows


def process_trial(subject: int, movement: str, cfg: PreprocessConfig) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[str]]:

    df, fs = load_aligned_trial(subject, movement, cfg)
    df_raw_filt = apply_bandpass(df, fs, cfg)
    df_env = rectify_and_envelope(df_raw_filt, fs, cfg)

    X_raw, X_env, meta_rows = window_trial(
        df_raw_filt=df_raw_filt,
        df_env=df_env,
        fs=fs,
        subject=subject,
        movement=movement,
        cfg=cfg
    )

    return X_raw, X_env, meta_rows, get_emg_cols(df_raw_filt)


def build_full_dataset(
    subjects: List[int],
    movements: List[str],
    out_npz: Path,
    out_meta_csv: Path,
    cfg: PreprocessConfig
) -> None:

    X_raw_all = []
    X_env_all = []
    meta_all: List[Dict] = []
    channel_names: Optional[List[str]] = None

    for subj in subjects:
        for mov in movements:
            mov = mov.upper()
            try:
                X_raw, X_env, meta_rows, emg_cols = process_trial(subj, mov, cfg)

                if channel_names is None:
                    channel_names = emg_cols
                else:
                    if emg_cols != channel_names:
                        raise ValueError(
                            f"Channel mismatch for Sub{subj:02d} {mov}. "
                            f"Expected {channel_names}, got {emg_cols}"
                        )

                if X_raw.shape[0] == 0:
                    print(f"Sub{subj:02d} {mov}: 0 windows (after cleaning)")
                    continue

                X_raw_all.append(X_raw)
                X_env_all.append(X_env)


                meta_all.extend(meta_rows)
                print(f"Sub{subj:02d} {mov}: {X_raw.shape[0]} windows")

            except FileNotFoundError as e:
                print(f"[Missing] Sub{subj:02d} {mov}: {e}")
            except Exception as e:
                print(f"[Error]   Sub{subj:02d} {mov}: {e}")

    if not X_raw_all:
        raise RuntimeError("No windows were produced. Check paths, labels, thresholds, and toggles.")

    X_raw_arr = np.concatenate(X_raw_all, axis=0)
    X_env_arr = np.concatenate(X_env_all, axis=0)




    def _ensure_nct(X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected 3D windows array. Got {X.shape}")

        if X.shape[1] > X.shape[2]:
            return np.transpose(X, (0, 2, 1))
        return X

    X_raw_arr = _ensure_nct(X_raw_arr).astype(np.float32, copy=False)
    X_env_arr = _ensure_nct(X_env_arr).astype(np.float32, copy=False)

    print(f"[save] X_raw shape (N,C,T): {X_raw_arr.shape}")
    print(f"[save] X_env shape (N,C,T): {X_env_arr.shape}")




    np.savez_compressed(out_npz, X_raw=X_raw_arr, X_env=X_env_arr)


    # Save run configuration / metadata as JSON 
    out_cfg_json = out_npz.with_suffix("").with_name(f"{out_npz.stem}_cfg.json")

    cfg_payload = {
        "axis_order": "NCT",
        "base_dir": str(cfg.base_dir),
        "bandpass_low": float(cfg.bandpass_low),
        "bandpass_high": float(cfg.bandpass_high),
        "bandpass_order": int(cfg.bandpass_order),
        "envelope_ms": float(cfg.envelope_ms),
        "win_ms": float(cfg.win_ms),
        "overlap": float(cfg.overlap),
        "drop_none_labels": bool(cfg.drop_none_labels),
        "drop_wak_unlabelled_zero": bool(cfg.drop_wak_unlabelled_zero),
        "min_label_conf": float(cfg.min_label_conf),
        "keep_only_active_stdup": bool(cfg.keep_only_active_stdup),
        "align_tol": None if cfg.align_tol is None else float(cfg.align_tol),
        "channels": list(channel_names) if channel_names else [],
        "movements_included": [m.upper() for m in movements],
        "subjects_included": [int(s) for s in subjects],
    }

    import json
    with open(out_cfg_json, "w", encoding="utf-8") as f:
        json.dump(cfg_payload, f, indent=2)

    print(f"CFG JSON: {out_cfg_json}")



    # Save metadata CSV
    meta_df = pd.DataFrame(meta_all)
    meta_df.to_csv(out_meta_csv, index=False)


    if len(meta_df) != X_raw_arr.shape[0]:
        raise ValueError(
            f"Meta rows ({len(meta_df)}) != NPZ windows ({X_raw_arr.shape[0]}). "
            "This will break feature extraction/training."
        )



    summary = {
        "total_windows": int(len(meta_df)),
        "n_subjects": int(meta_df["subject"].nunique()) if "subject" in meta_df else None,
        "n_movements": int(meta_df["movement"].nunique()) if "movement" in meta_df else None,
        "win_ms": int(cfg.win_ms),
        "overlap": float(cfg.overlap),
        "min_label_conf": float(cfg.min_label_conf),
        "keep_only_active_stdup": bool(cfg.keep_only_active_stdup),
    }

    by_movement = meta_df.groupby("movement").size().sort_values(ascending=False)
    by_subject = meta_df.groupby("subject").size().sort_values(ascending=False)

    # Save alongside meta CSV
    out_summary_csv = out_meta_csv.with_name(f"{out_meta_csv.stem}_summary.csv")
    out_summary_json = out_meta_csv.with_name(f"{out_meta_csv.stem}_summary.json")

    # CSV: two small tables stacked
    summary_rows = pd.DataFrame(list(summary.items()), columns=["key", "value"])
    with open(out_summary_csv, "w", encoding="utf-8") as f:
        summary_rows.to_csv(f, index=False)
        f.write("\n# windows_by_movement\n")
        by_movement.to_csv(f, header=["count"])
        f.write("\n\n# windows_by_subject\n")
        by_subject.to_csv(f, header=["count"])


    import json
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "windows_by_movement": by_movement.to_dict(),
                "windows_by_subject": by_subject.to_dict(),
            },
            f,
            indent=2
        )

    print(f"SUMMARY CSV : {out_summary_csv}")
    print(f"SUMMARY JSON: {out_summary_json}")


    print("\n==== SAVED ====")
    print(f"NPZ : {out_npz}")
    print(f"META: {out_meta_csv}")
    print(f"X_raw shape: {X_raw_arr.shape} (N, n_channels, win_len)")
    print(f"X_env shape: {X_env_arr.shape} (N, n_channels, win_len)")




# CLI
def parse_subjects(s: str) -> List[int]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Preprocess + window SIAT-LLMD EMG and save NPZ + CSV meta")
    DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "SIAT_LLMD20230404"

    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DEFAULT_BASE_DIR),
        help="Path to SIAT_LLMD dataset root (default: ./SIAT_LLMD20230404)"
    )

    parser.add_argument("--subjects", type=str, default="1-40", help="e.g. '1-40' or '1,2,5,10'")
    parser.add_argument("--movements", type=str, default=",".join(MOVEMENTS_DEFAULT), help="Comma list, e.g. WAK,UPS,DNS,STDUP")

    parser.add_argument("--out-npz", type=str, default="windows_WAK_UPS_DNS_STDUP_v1.npz")
    parser.add_argument("--out-meta", type=str, default="windows_WAK_UPS_DNS_STDUP_v1_meta.csv")

    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional suffix tag appended to output filenames (before extension). "
             "Example: run1_w150 or w250_conf60"
    )
    parser.add_argument(
        "--auto-tag",
        action="store_true",
        help="If set, automatically creates a tag from key params (win_ms, overlap, min_conf, active_stdup)."
    )


    parser.add_argument("--low", type=float, default=20.0)
    parser.add_argument("--high", type=float, default=450.0)
    parser.add_argument("--order", type=int, default=4)

    parser.add_argument("--envelope-ms", type=float, default=50.0)
    parser.add_argument("--win-ms", type=float, default=150.0)

    parser.add_argument(
        "--ablation",
        action="store_true",
        help="If set, runs preprocessing for both 150ms and 250ms window lengths in one click."
    )
    parser.add_argument(
        "--ablation-win-ms",
        type=str,
        default="150,250",
        help="Comma-separated window lengths to run when --ablation is set. Default: 150,250"
    )


    parser.add_argument("--overlap", type=float, default=0.5)

    parser.add_argument("--keep-wak-zero", action="store_true", help="Keep WAK status==0 windows (default drops them)")
    parser.add_argument("--keep-none-labels", action="store_true", help="Keep windows with None/NaN labels (default drops them)")
    parser.add_argument("--min-conf", type=float, default=0.60, help="Min majority-label confidence per window (Decision 4)")

    parser.add_argument("--keep-only-active-stdup", action="store_true", help="(Optional) For STDUP, keep only windows with majority 'A'")

    parser.add_argument("--align-tol", type=float, default=None, help="Optional alignment tolerance in seconds")

    args = parser.parse_args()

    def _parse_ablation_list(s: str) -> list[float]:
        vals = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            vals.append(float(part))
        if not vals:
            raise ValueError("No window lengths provided for ablation.")
        return vals


    def _auto_tag(win_ms: float) -> str:
        ov_pct = int(round(float(args.overlap) * 100))
        conf_pct = int(round(float(args.min_conf) * 100))
        stdup = "Aonly" if args.keep_only_active_stdup else "AorR"
        return f"w{int(win_ms)}_ov{ov_pct}_conf{conf_pct}_{stdup}"

    tag = args.run_tag.strip()
    if args.auto_tag and not args.ablation:
        tag = _auto_tag(args.win_ms) if not tag else f"{tag}_{_auto_tag(args.win_ms)}"


    def _append_tag(path_str: str, tag_str: str) -> str:
        if not tag_str:
            return path_str
        p = Path(path_str)
        return str(p.with_name(f"{p.stem}_{tag_str}{p.suffix}"))

    out_npz = _append_tag(args.out_npz, tag)
    out_meta = _append_tag(args.out_meta, tag)


    cfg = PreprocessConfig(
        base_dir=Path(args.base_dir),
        bandpass_low=args.low,
        bandpass_high=args.high,
        bandpass_order=args.order,
        envelope_ms=args.envelope_ms,
        win_ms=args.win_ms,
        overlap=args.overlap,
        drop_none_labels=not args.keep_none_labels,
        drop_wak_unlabelled_zero=not args.keep_wak_zero,
        min_label_conf=float(args.min_conf),
        keep_only_active_stdup=bool(args.keep_only_active_stdup),
        align_tol=args.align_tol,
    )

    subjects = parse_subjects(args.subjects)
    movements = [m.strip().upper() for m in args.movements.split(",") if m.strip()]

    # If ablation mode, run multiple window lengths in one click
    if args.ablation:
        win_list = _parse_ablation_list(args.ablation_win_ms)

        for w in win_list:
            cfg_run = replace(cfg, win_ms=float(w))

            tag_run = _auto_tag(cfg_run.win_ms)
            out_npz_run = _append_tag(args.out_npz, tag_run)
            out_meta_run = _append_tag(args.out_meta, tag_run)

            print(f"\n=== RUN: {tag_run} ===")
            print(f"Window length: {cfg_run.win_ms} ms | Overlap: {cfg_run.overlap} | Min conf: {cfg_run.min_label_conf}")
            build_full_dataset(
                subjects=subjects,
                movements=movements,
                out_npz=Path(out_npz_run),
                out_meta_csv=Path(out_meta_run),
                cfg=cfg_run
            )

    else:
        build_full_dataset(
            subjects=subjects,
            movements=movements,
            out_npz=Path(out_npz),
            out_meta_csv=Path(out_meta),
            cfg=cfg
        )




if __name__ == "__main__":
    main()
