from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# --------- CONFIG ----------
PROJECT_ROOT = Path(__file__).parent
DATASET_ROOT = PROJECT_ROOT / "SIAT_LLMD20230404"

SUBJECTS = range(1, 41)               # 1..40
MOVEMENTS = ["WAK", "UPS", "DNS", "STDUP"]     # your thesis classes

OUT_DIR = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)


# --------- HELPERS ----------
def build_paths(subject: int, movement: str) -> tuple[Path, Path] | None:
    """Return (data_path, label_path) for CSV or XLSX. None if missing."""
    sub = f"Sub{subject:02d}"
    mov = movement.upper()

    data_candidates = [
        DATASET_ROOT / sub / "Data" / f"{sub}_{mov}_Data.csv",
        DATASET_ROOT / sub / "Data" / f"{sub}_{mov}_Data.xlsx",
    ]
    label_candidates = [
        DATASET_ROOT / sub / "Labels" / f"{sub}_{mov}_Label.csv",
        DATASET_ROOT / sub / "Labels" / f"{sub}_{mov}_Label.xlsx",
    ]

    data_path = next((p for p in data_candidates if p.exists()), None)
    label_path = next((p for p in label_candidates if p.exists()), None)

    if data_path is None or label_path is None:
        return None
    return data_path, label_path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def find_status_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "status":
            return c
    # substring fallback
    cand = [c for c in df.columns if "status" in str(c).lower()]
    if cand:
        return cand[0]
    # common label format: Time, Status, Group (3 cols)
    if len(df.columns) == 3:
        return df.columns[1]
    return None


def time_checks(t: pd.Series) -> dict:
    t = pd.to_numeric(t, errors="coerce")
    missing = int(t.isna().sum())
    t = t.dropna().to_numpy()

    if len(t) < 3:
        return {
            "time_len": int(len(t)),
            "time_missing": missing,
            "time_monotonic": False,
            "dt_median": np.nan,
            "dt_min": np.nan,
            "dt_max": np.nan,
        }

    diffs = np.diff(t)
    monotonic = bool(np.all(diffs > 0))
    return {
        "time_len": int(len(t)),
        "time_missing": missing,
        "time_monotonic": monotonic,
        "dt_median": float(np.median(diffs)),
        "dt_min": float(np.min(diffs)),
        "dt_max": float(np.max(diffs)),
    }


def summarize_status(status_series: pd.Series) -> tuple[str, str, int]:
    """Return (format_tag, unique_values_str, n_unique)"""
    s = status_series.copy()
    # keep NaN but normalize strings
    unique = []
    for v in s.to_list():
        if pd.isna(v):
            continue
        unique.append(str(v).strip())
    uniq = sorted(set(unique))

    if len(uniq) == 0:
        return ("EMPTY", "", 0)

    if all(u in {"R", "A"} for u in uniq):
        tag = "RA"
    elif all(u.replace(".", "", 1).isdigit() for u in uniq):
        tag = "NUM"
    else:
        tag = "MIXED"

    return (tag, ",".join(uniq[:30]) + ("..." if len(uniq) > 30 else ""), len(uniq))

def first_active_time_std_up(label_df: pd.DataFrame) -> float:
    """Return first time where Status == 'A' for STDUP, else NaN."""
    if "Time" not in label_df.columns:
        return np.nan
    status_col = find_status_column(label_df)
    if status_col is None:
        return np.nan

    t = pd.to_numeric(label_df["Time"], errors="coerce")
    s = label_df[status_col].astype(str).str.strip().str.upper()

    mask_a = (s == "A") & (~t.isna())
    if not mask_a.any():
        return np.nan
    return float(t[mask_a].iloc[0])



# --------- MAIN ----------
rows = []

for subj in SUBJECTS:
    for mov in MOVEMENTS:
        paths = build_paths(subj, mov)
        if paths is None:
            rows.append({
                "subject": subj, "movement": mov,
                "data_exists": False, "label_exists": False,
            })
            continue

        data_path, label_path = paths
        data_df = read_table(data_path)
        label_df = read_table(label_path)

        # Basic existence + columns
        data_has_time = "Time" in data_df.columns
        label_has_time = "Time" in label_df.columns

        status_col = find_status_column(label_df)
        label_has_status = status_col is not None

        # Time checks
        data_time_stats = time_checks(data_df["Time"]) if data_has_time else {}
        label_time_stats = time_checks(label_df["Time"]) if label_has_time else {}

        # Status summary
        if label_has_status:
            tag, uniq_str, nuniq = summarize_status(label_df[status_col])
        else:
            tag, uniq_str, nuniq = ("NO_STATUS_COL", "", 0)

        a_time = np.nan
        if mov.upper() == "STDUP":
            a_time = first_active_time_std_up(label_df)
        

        # Effective sample rate estimate (Hz) from median dt
        dt = data_time_stats.get("dt_median", np.nan)
        fs_est = (1.0 / dt) if (isinstance(dt, float) and dt > 0) else np.nan

        rows.append({
            "subject": subj,
            "movement": mov,
            "data_exists": True,
            "label_exists": True,
            "data_file": str(data_path.name),
            "label_file": str(label_path.name),
            "data_has_time": data_has_time,
            "label_has_time": label_has_time,
            "label_status_col": str(status_col) if status_col is not None else "",
            "status_format": tag,
            "status_unique_n": nuniq,
            "status_unique_vals": uniq_str,
            "data_time_len": data_time_stats.get("time_len", np.nan),
            "data_time_missing": data_time_stats.get("time_missing", np.nan),
            "data_time_monotonic": data_time_stats.get("time_monotonic", np.nan),
            "data_dt_median": data_time_stats.get("dt_median", np.nan),
            "data_fs_est_hz": fs_est,
            "label_time_len": label_time_stats.get("time_len", np.nan),
            "label_time_missing": label_time_stats.get("time_missing", np.nan),
            "label_time_monotonic": label_time_stats.get("time_monotonic", np.nan),
            "stdup_first_A_time": a_time,
            "stdup_suggested_tmax": (min(a_time + 0.6, 6.0) if np.isfinite(a_time) else np.nan),

        })

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "dataset_audit_WAK_UPS_DNS.csv"
df.to_csv(csv_path, index=False)

# Markdown summary (quick human-readable)
md_path = OUT_DIR / "dataset_audit_WAK_UPS_DNS.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# SIAT-LLMD Audit (WAK/UPS/DNS)\n\n")
    f.write(f"- Subjects: {len(list(SUBJECTS))}\n")
    f.write(f"- Movements: {', '.join(MOVEMENTS)}\n\n")

    missing = df[(df["data_exists"] == False) | (df["label_exists"] == False)]
    f.write(f"## Missing files\n\nCount: {len(missing)}\n\n")
    if len(missing) > 0:
        f.write(missing[["subject", "movement", "data_exists", "label_exists"]].to_markdown(index=False))
        f.write("\n\n")

    f.write("## Status formats by movement\n\n")
    f.write(df.groupby(["movement", "status_format"]).size().reset_index(name="count").to_markdown(index=False))
    f.write("\n\n")

    f.write("## Sample rate estimate (median dt)\n\n")
    fs_summary = df.groupby("movement")["data_fs_est_hz"].describe().reset_index()
    f.write(fs_summary.to_markdown(index=False))
    f.write("\n")
    f.write("\n## STDUP: first 'A' (active) time summary\n\n")
    st = df[df["movement"] == "STDUP"][["subject", "stdup_first_A_time", "stdup_suggested_tmax"]].dropna()
    if len(st) == 0:
        f.write("No STDUP 'A' transitions detected.\n")
    else:
        f.write(st.describe().to_markdown())
        f.write("\n")


print("Wrote:", csv_path)
print("Wrote:", md_path)
