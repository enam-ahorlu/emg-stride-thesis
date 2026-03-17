"""
Quick EMG + gait-phase visualisation for SIAT-LLMD dataset.

- Loads one subject + movement (default: Sub01 WAK)
- Plots all sEMG channels in a vertical stack
- Background is colour-coded by gait phase using the label file

"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def find_status_column(label_df: pd.DataFrame) -> str:
    """
    Try to find the column that represents 'Status' in the label file.
    match case-insensitively and ignore leading/trailing spaces.
    Raises a clear error if nothing reasonable is found.
    """
    # Exact match ignoring case/whitespace
    for col in label_df.columns:
        if col.strip().lower() == "status":
            return col

    # Fallback: any column that contains 'status' as a substring
    candidates = [c for c in label_df.columns if "status" in c.lower()]
    if candidates:
        return candidates[0]


    if len(label_df.columns) == 3:
        return label_df.columns[1]

    raise KeyError(
        f"No 'Status' column found. Available columns: {list(label_df.columns)}"
    )



# 1. CONFIG

# Folder that contains "SIAT_LLMD20230404"
PROJECT_ROOT = Path(__file__).parent
DATASET_ROOT = PROJECT_ROOT / "SIAT_LLMD20230404"


DEFAULT_SUBJECT = 27          
DEFAULT_MOVEMENT = "LUGF"    
DEFAULT_TIME_WINDOW = (0.0, 2.2)   


# PLOT OUTPUT CONFIG

PLOTS_ROOT = PROJECT_ROOT / "plots" / "raw"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)



# 2. WAK-SPECIFIC GAIT PHASE INFO

# Mapping follows the author’s Load_ForGait script
GAIT_PHASES_WAK: Dict[str, str] = {
    "0": "Unlabelled / other",
    "1": "HS–MSF",
    "2": "MSF–MSE",
    "3": "MSE–TO",
    "4": "TO–MWF",
    "5": "MWF–HS",
}

GAIT_COLORS_WAK: Dict[str, str] = {
    "1": "#ffe6e6",  # pale red
    "2": "#e6ffe6",  # pale green
    "3": "#e6e6ff",  # pale blue
    "4": "#ffe6cc",  # pale orange
    "5": "#f2f2f2",  # pale grey
    
}

# Palette for generic movements (R/A, 1–3, etc.)
GENERIC_PALETTE = [
    "#fde0ef",
    "#e0f3db",
    "#fee8c8",
    "#d9d9d9",
    "#ccebc5",
    "#fbb4ae",
    "#b3cde3",
]



def shorten_muscle_name(col_name: str) -> str:
    """
    Abbreviate long muscle names for cleaner y-axis labels.
    """
    # Common abbreviations
    replacements = {
        "tensor fascia lata": "TFL",
        "rectus femoris": "RFem",
        "vastus medialis": "VMed",
        "semimembranosus": "SMem",
        "upper tibialis anterior": "TAup",
        "lower tibialis anterior": "TAlow",
        "lateral gastrocnemius": "GClat",
        "medial gastrocnemius": "GCmed",
        "soleus": "Sol"
    }
    
    # Remove sEMG: prefix if present
    if col_name.startswith("sEMG:"):
        base_name = col_name[5:].strip()
    else:
        base_name = col_name
    
    # Replace with abbreviation if found
    for full, abbrev in replacements.items():
        if full.lower() in base_name.lower():
            return f"sEMG\n{abbrev}"
    
    # Fallback: keep original but truncate
    if len(base_name) > 10:
        return f"sEMG\n{base_name[:8]}.."
    return f"sEMG\n{base_name}"





# 3. IO HELPERS



def build_paths(subject: int, movement: str) -> Tuple[Path, Path]:
    """
    Construct data + label paths for a given subject & movement.

    Prefers CSV, but will fall back to XLSX if CSV is not found.
    This keeps things robust if some trials are stored as Excel.
    """
    sub_str = f"Sub{subject:02d}"
    mov_str = movement.upper()

    # Candidate paths
    data_csv = DATASET_ROOT / sub_str / "Data" / f"{sub_str}_{mov_str}_Data.csv"
    data_xlsx = DATASET_ROOT / sub_str / "Data" / f"{sub_str}_{mov_str}_Data.xlsx"

    label_csv = DATASET_ROOT / sub_str / "Labels" / f"{sub_str}_{mov_str}_Label.csv"
    label_xlsx = DATASET_ROOT / sub_str / "Labels" / f"{sub_str}_{mov_str}_Label.xlsx"

    # Resolve data path
    if data_csv.exists():
        data_path = data_csv
    elif data_xlsx.exists():
        data_path = data_xlsx
    else:
        raise FileNotFoundError(f"No data file found for {sub_str} {mov_str}")

    # Resolve label path
    if label_csv.exists():
        label_path = label_csv
    elif label_xlsx.exists():
        label_path = label_xlsx
    else:
        raise FileNotFoundError(f"No label file found for {sub_str} {mov_str}")

    return data_path, label_path


def align_labels_to_data_by_time(data: pd.DataFrame, labels: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """
    Align labels to data using nearest-time matching.
    Assumes both have a 'Time' column. Returns a labels dataframe
    with the same row count as data.
    """
    data_sorted = data.sort_values("Time").reset_index(drop=True)
    labels_sorted = labels.sort_values("Time").reset_index(drop=True)

    # merge_asof aligns each data time to the nearest label time
    aligned = pd.merge_asof(
        data_sorted[["Time"]],
        labels_sorted,
        on="Time",
        direction="nearest",
        tolerance=tol,
    )

    # if tolerance too strict, you can increase it (e.g., 1/FS)
    return aligned



def load_emg_and_labels(subject: int, movement: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_path, label_path = build_paths(subject, movement)

    # Read data
    if data_path.suffix.lower() == ".csv":
        data = pd.read_csv(data_path)
    else:
        data = pd.read_excel(data_path)

    # Read labels
    if label_path.suffix.lower() == ".csv":
        labels = pd.read_csv(label_path)
    else:
        labels = pd.read_excel(label_path)


    # Basic sanity checks
    if "Time" not in data.columns or "Time" not in labels.columns:
        raise ValueError("Both data and label files must contain a 'Time' column.")

    if len(data) != len(labels) or not np.allclose(data["Time"], labels["Time"]):
        print("WARNING: Time columns differ, aligning by nearest time.")
        labels = align_labels_to_data_by_time(data, labels, tol=1e-4)  # adjust tol as needed

    return data, labels




# 4. STATUS NORMALISATION & COLOUR MAPPING

def get_status_and_colours(
    movement: str,
    status_series: pd.Series,
) -> Tuple[np.ndarray, Dict[str, str], Dict[str, str]]:
    """
    Normalise 'Status' values to string codes and build:
      - status_norm: np.ndarray of str/None, same length as series
      - label_map:  code -> human-readable label
      - color_map:  code -> colour hex

    Handles:
      - WAK: numeric codes 0–5 (with fixed labels & colours)
      - Other movements: e.g. 'R', 'A', '1', '2', '3', ...
    """
    movement = movement.upper()

    # Convert to python objects for easier inspection
    raw = status_series.to_list()


    # WAK: use fixed mapping
    if movement == "WAK":
        status_norm = []
        used_codes: set[str] = set()

        for s in raw:
            if pd.isna(s):
                status_norm.append(None)
                continue

            # WAK labels are numbers in the file
            try:
                code = str(int(s))
            except (TypeError, ValueError):
                code = str(s)

            status_norm.append(code)
            used_codes.add(code)

        # Only keep phases that actually appear
        label_map = {
            c: GAIT_PHASES_WAK.get(c, f"Phase {c}")
            for c in sorted(used_codes)
        }
        color_map = {
            c: GAIT_COLORS_WAK.get(c, "#eeeeee")
            for c in sorted(used_codes)
        }
        return np.array(status_norm, dtype=object), label_map, color_map


    # NON-WAK: generic handling (R/A, 1–3, etc.)
    status_norm = []
    uniques_ordered: list[str] = []  # preserve first-seen order

    for s in raw:
        if pd.isna(s):
            status_norm.append(None)
            continue

        code = str(s).strip()
        status_norm.append(code)
        if code not in uniques_ordered:
            uniques_ordered.append(code)

    # Human-readable labels
    label_map: Dict[str, str] = {}
    for code in uniques_ordered:
        up = code.upper()
        if up == "R":
            label_map[code] = "Rest (R)"
        elif up == "A":
            label_map[code] = "Active (A)"
        elif code.isdigit():
            label_map[code] = f"Phase {code}"
        else:
            label_map[code] = f"Status {code}"

    # Colours from generic palette
    color_map = {
        code: GENERIC_PALETTE[i % len(GENERIC_PALETTE)]
        for i, code in enumerate(uniques_ordered)
    }

    return np.array(status_norm, dtype=object), label_map, color_map



# 5. SHADING FUNCTION
def shade_gait_phases(
    ax: plt.Axes,
    time: np.ndarray,
    status_codes: np.ndarray,
    color_map: Dict[str, str],
    alpha: float = 0.62,
) -> None:
    """
    Draw coloured vertical spans for each contiguous gait phase.
    """

    time = np.asarray(time)
    assert time.ndim == 1
    assert len(time) == len(status_codes)

    current = status_codes[0]
    seg_start = time[0]

    def draw_segment(start_t, end_t, code):
        if code is None:
            return
        color = color_map.get(code)
        if color is None:
            return
        ax.axvspan(start_t, end_t, color=color, alpha=alpha, zorder=-1)

    # Walk along the vector and break when status changes
    for t, s in zip(time[1:], status_codes[1:]):
        if s != current:
            draw_segment(seg_start, t, current)
            seg_start = t
            current = s

    # Close the last one
    draw_segment(seg_start, time[-1], current)


# Auto time window for STDUP

def auto_time_window_for_movement(
    movement: str,
    label_df: pd.DataFrame,
    default=(0.0, 2.2),
    post_active_margin: float = 0.6,
    max_tmax: float = 6.0,
):
    """
    Returns a sensible (tmin, tmax) for plotting.
    For STDUP: extend until shortly after the first 'A' appears (rest->active),
    because the transition time varies across subjects.
    """
    mov = str(movement).upper()

    # Default for most movements
    if mov != "STDUP":
        return default

    # STDUP: try to detect first 'A'
    if "Time" not in label_df.columns:
        return default

    status_col = find_status_column(label_df)
    if status_col is None:
        return default

    t = pd.to_numeric(label_df["Time"], errors="coerce")
    s = label_df[status_col].astype(str).str.strip().str.upper()

    # Find first time where status == 'A'
    mask_a = (s == "A") & (~t.isna())
    if not mask_a.any():
        # no active phase detected in the window -> fallback
        return default

    first_a_time = float(t[mask_a].iloc[0])
    tmin = default[0]
    tmax = min(first_a_time + post_active_margin, max_tmax)

    # Ensure not shorter than default tmax
    tmax = max(tmax, default[1])
    return (tmin, tmax)




# 6. MAIN PLOTTING FUNCTION
def plot_emg_with_gait(
    data_df,
    label_df,
    subject,
    movement,
    time_window=None,
    save=True,
    dpi=300,
    show=False,
    out_dir=None,
) -> None:
    """Plot all sEMG channels for one subject+movement with gait shading."""
    movement = movement.upper()
    # Decide time window (AUTO for STDUP, default for others)
    if time_window is None:
        time_window = auto_time_window_for_movement(
            movement=movement,
            label_df=label_df,
            default=DEFAULT_TIME_WINDOW,
            post_active_margin=0.6,
            max_tmax=6.0,
        )
    
    # Time window mask
    t = data_df["Time"].to_numpy()
    if time_window is not None:
        t_min, t_max = time_window
        mask = (t >= t_min) & (t <= t_max)
    else:
        mask = np.ones_like(t, dtype=bool)

    t_win = t[mask]

    # EMG columns
    emg_cols = [c for c in data_df.columns if c.startswith("sEMG")]
    emg_win = data_df.loc[mask, emg_cols]

    # Status + colour info
    status_col = find_status_column(label_df)
    status_raw = label_df.loc[mask, status_col]
    status_codes, label_map, color_map = get_status_and_colours(movement, status_raw)

    n_ch = len(emg_cols)
    duration = float(time_window[1] - time_window[0])

    # width scales with duration 
    base_w = 16.0
    scale = min(max(duration / 2.2, 1.0), 1.8)
    fig_w = base_w * scale

    # WAK gets wider figure to accommodate longer labels
    if movement.upper() == "WAK":
        fig_w = 18.0  # Even wider for WAK

    
    row_h = 1.25  
    fig_h = max(10.0, n_ch * row_h)

    
    if movement.upper() == "WAK":
        fig_h += 2.0

    fig, axes = plt.subplots(
        nrows=n_ch,
        ncols=1,
        sharex=True,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )

    if n_ch == 1:
        axes = [axes]

    # PLOTTING LOOP 
    for i, (ax, col) in enumerate(zip(axes, emg_cols)):
        # Plot the EMG signal
        ax.plot(t_win, emg_win[col].to_numpy(), linewidth=0.7)
        
        # Add gait phase shading
        shade_gait_phases(ax, t_win, status_codes, color_map)
        
        # Set y-label with shortened muscle names
        short_label = shorten_muscle_name(col)
        ax.set_ylabel(short_label, rotation=0, ha="right", va="center", fontsize=9)
        
        # Add grid
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        
        # Only bottom axis gets x-label
        if i == n_ch - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])
    # END PLOTTING LOOP

    # Title
    fig.suptitle(
        f"Subject {subject:02d} – {movement} – raw sEMG with gait phases",
        y=0.995,
        fontsize=13,
    )

    # Legend outside the plot on the right
    handles = [
        Patch(facecolor=color_map[code], edgecolor="none", alpha=0.52, label=label)
        for code, label in label_map.items()
    ]


    axes[0].legend(
        handles=handles,
        title="Gait phases (Status)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )


    if movement.upper() == "WAK":
        fig.tight_layout(rect=[0.08, 0.02, 0.85, 0.93])  
    else:
        fig.tight_layout(rect=[0.02, 0.02, 0.80, 0.93])


    # Save figure
    if save:
        tmin, tmax = time_window if time_window else (None, None)

        # default base folder
        base = PLOTS_ROOT if out_dir is None else Path(out_dir)

        # subject subfolder
        sub_folder = base / f"Sub{subject:02d}"
        sub_folder.mkdir(parents=True, exist_ok=True)

        if tmin is not None and tmax is not None:
            fname = f"{movement}_t{tmin:.2f}-{tmax:.2f}s_raw_emg.png"
        else:
            fname = f"{movement}_raw_emg.png"

        save_path = sub_folder / fname
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[Saved] {save_path}")

    if show:
        plt.show()

    plt.close(fig)




# 7. SCRIPT ENTRY POINT
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot SIAT-LLMD sEMG with gait-phase shading."
    )
    parser.add_argument(
        "-s", "--subject", type=int, default=DEFAULT_SUBJECT,
        help="Subject number (1–40). Default: 1",
    )
    parser.add_argument(
        "-m", "--movement", type=str, default=DEFAULT_MOVEMENT,
        help="Movement code, e.g. WAK, KLCL, ... Default: WAK",
    )
    parser.add_argument(
        "--tmin", type=float, default=DEFAULT_TIME_WINDOW[0],
        help="Start time (s) for window",
    )
    parser.add_argument(
        "--tmax", type=float, default=DEFAULT_TIME_WINDOW[1],
        help="End time (s) for window",
    )

    args = parser.parse_args()

    data_df, label_df = load_emg_and_labels(args.subject, args.movement)
    plot_emg_with_gait(
        data_df,
        label_df,
        subject=args.subject,
        movement=args.movement,
        time_window=(args.tmin, args.tmax),
    )
