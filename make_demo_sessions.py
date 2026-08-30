#!/usr/bin/env python3
"""
make_demo_sessions.py
=====================
Builds the MyoLens demonstration data from the three held-out SIAT-LLMD
subjects — the same three excluded from the deployment model, so the demo is
an honest out-of-sample demonstration rather than a replay of training data.

For each subject:
  demo_SubXX_calibration.csv.gz   labelled, N non-contiguous blocks per task
  demo_SubXX_session.csv.gz       unlabelled multi-task session, bouts interleaved
  demo_SubXX_truth.csv            bout boundaries + true task (OUR validation only)

THE RECORDINGS ARE SHORT. Measured on Sub10: DNS ≈ 20,056 samples (10.4 s at
1920 Hz), WAK similar, UPS ≈ 13 s, STDUP ≈ 36 s. Bout and block lengths are
therefore derived from what each trial actually contains rather than assumed;
asking for a 15-second bout fails outright on DNS.

Calibration and session draw from DISJOINT halves of each recording — the first
`--cal-frac` for calibration, the remainder for the session. A calibration
capture that overlapped the assessment would make the demo look better than the
system is.

HONESTY NOTE, which belongs in the user manual too: these sessions are
*assembled* from real trials, not recorded as continuous multi-task sessions.
Each task was recorded separately, so a session is real signal from a real
participant, re-sequenced into a plausible clinical order. Not synthetic, not
one continuous take. Say both halves of that.

Why interleave rather than concatenate: four concatenated blocks give four bouts
and a segmentation nobody needs to review. Interleaving produces real transitions
and puts DNS next to WAK repeatedly, which is where the confusion lives.

Licence: SIAT-LLMD is CC0 1.0 Universal. Cite Wei et al. (2023) Scientific Data
10:358 regardless — academic practice, and examination Rule 6.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["DNS", "STDUP", "UPS", "WAK"]

# Relative appetite for bouts per task. STDUP and WAK recur through a real
# assessment; DNS and UPS are the constrained ones anyway.
BOUT_WEIGHT = {"STDUP": 3, "WAK": 3, "UPS": 2, "DNS": 2}


def load_trial(base: Path, subject: int, movement: str):
    sub = f"Sub{subject:02d}"
    data_p = base / sub / "Data" / f"{sub}_{movement}_Data.csv"
    label_p = base / sub / "Labels" / f"{sub}_{movement}_Label.csv"
    if not data_p.exists():
        raise SystemExit(f"FATAL: missing {data_p}")
    df = pd.read_csv(data_p)
    df.columns = [c.strip() for c in df.columns]
    lab = None
    if label_p.exists():
        lab = pd.read_csv(label_p)
        lab.columns = [c.strip() for c in lab.columns]
    return df, lab


def emg_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("sEMG")]
    if len(cols) != 9:
        raise SystemExit(f"FATAL: expected 9 sEMG columns, found {len(cols)}")
    return cols


def usable_span(df: pd.DataFrame, lab: pd.DataFrame | None, margin: float = 0.05):
    """Usable index range: the whole trial minus a margin at each end.

    Each SIAT-LLMD file is a single-movement trial, so the recording *is* that
    movement apart from brief idle at the head and tail. An earlier version took
    the longest contiguous run of non-null Status and it was badly wrong: the
    Status semantics differ per movement — DNS/UPS use '1'/'2'/'3', STDUP uses
    'R'/'A', and **WAK uses floating-point gait-phase labels with NaN gaps
    between gait events**. Those gaps are a label-alignment artefact of
    merge_asof, not inactivity, so requiring contiguity collapsed WAK from
    ~20,000 samples to ~1,900 and starved it of calibration windows.

    Keeping the whole trial also matches the thesis, whose windowing config sets
    keep_only_active_stdup = false — it retains STDUP's 'R' spans too.

    Status is still reported, for the record.
    """
    n = len(df)
    lo, hi = int(margin * n), int((1.0 - margin) * n)
    if lab is None or "Status" not in lab.columns or "Time" not in lab.columns:
        return lo, hi, "whole trial ±margin (no Status column)"

    merged = pd.merge_asof(
        pd.DataFrame({"Time": pd.to_numeric(df["Time"], errors="coerce")}).sort_values("Time"),
        pd.DataFrame({"Time": pd.to_numeric(lab["Time"], errors="coerce"),
                      "Status": lab["Status"]}).dropna(subset=["Time"]).sort_values("Time"),
        on="Time", direction="nearest", tolerance=0.05,
    )
    vals = pd.Series(merged["Status"].astype(str).str.strip()).value_counts().head(4).to_dict()
    return lo, hi, f"whole trial ±{margin:.0%}; Status {vals}"


def even_slices(lo: int, hi: int, count: int, length: int):
    """`count` non-overlapping slices of `length`, spread across [lo, hi)."""
    span = hi - lo
    if count < 1 or length < 1 or span < length:
        return []
    count = min(count, span // length)
    if count < 1:
        return []
    starts = np.linspace(lo, hi - length, count).round().astype(int)
    # de-overlap defensively
    out, last_end = [], -1
    for s in starts:
        s = max(int(s), last_end)
        if s + length > hi:
            break
        out.append((s, s + length)); last_end = s + length
    return out


def write_csv(path: Path, time_s, block, cols, label_col=None, precision=6):
    out = pd.DataFrame({"Time": np.round(time_s, 6)})
    for j, c in enumerate(cols):
        out[c] = np.round(block[:, j], precision)
    if label_col is not None:
        out["label"] = label_col
    out.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)
    return path.stat().st_size


def main():
    ap = argparse.ArgumentParser(description="Build MyoLens demo data from held-out subjects.")
    ap.add_argument("--siat", default="SIAT_LLMD20230404")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default="10,13,22")
    ap.add_argument("--fs", type=float, default=1920.0)
    ap.add_argument("--cal-frac", type=float, default=0.40, help="leading fraction reserved for calibration")
    ap.add_argument("--cal-blocks", type=int, default=3)
    ap.add_argument("--cal-seconds", type=float, default=1.5, help="per block; ~34 windows/class at 3 blocks")
    ap.add_argument("--bout-seconds", type=float, default=3.0)
    args = ap.parse_args()

    base = Path(args.siat).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / args.fs
    WIN, STEP = 480, 240

    for subject in [int(s) for s in args.subjects.split(",")]:
        print(f"\n=== Sub{subject:02d} ===")
        cal_slices, bout_pool, cols = {}, {}, None

        for mv in LABELS:
            df, lab = load_trial(base, subject, mv)
            cols = cols or emg_cols(df)
            lo, hi, why = usable_span(df, lab)
            span = hi - lo
            split = lo + int(args.cal_frac * span)

            cal_len = int(round(args.cal_seconds * args.fs))
            # shrink the block if the calibration half cannot hold N of them
            cal_len = min(cal_len, max(WIN, (split - lo) // args.cal_blocks))
            cal = even_slices(lo, split, args.cal_blocks, cal_len)

            bout_len = int(round(args.bout_seconds * args.fs))
            bout_len = min(bout_len, max(WIN, hi - split))
            bouts = even_slices(split, hi, BOUT_WEIGHT[mv], bout_len)

            wins_per_class = args.cal_blocks * ((cal_len - WIN) // STEP + 1) if cal_len >= WIN else 0
            print(f"  {mv:<6} {len(df):>7,} samples ({len(df)*dt:>5.1f}s)  usable {span:>7,}  "
                  f"cal {len(cal)}x{cal_len/args.fs:.2f}s (~{wins_per_class} win)  "
                  f"bouts {len(bouts)}x{bout_len/args.fs:.2f}s   [{why}]")

            if wins_per_class < 20:
                print(f"    ! only ~{wins_per_class} calibration windows for {mv} — "
                      f"below the 20/class requirement")
            if not bouts:
                raise SystemExit(f"FATAL: no session bout fits for {mv}")

            arr = df[cols].to_numpy(dtype=np.float64)
            cal_slices[mv] = [arr[a:b] for a, b in cal]
            bout_pool[mv] = [arr[a:b] for a, b in bouts]

        # ---- session: round-robin interleave, so tasks alternate ----
        order, pools = [], {k: list(v) for k, v in bout_pool.items()}
        while any(pools.values()):
            for task in ["STDUP", "WAK", "UPS", "WAK", "DNS", "STDUP"]:
                if pools.get(task):
                    order.append((task, pools[task].pop(0)))

        blocks, truth, cursor = [], [], 0
        for task, blk in order:
            blocks.append(blk)
            truth.append({"task": task,
                          "start_sample": cursor, "end_sample": cursor + len(blk),
                          "start_s": round(cursor * dt, 4),
                          "end_s": round((cursor + len(blk)) * dt, 4)})
            cursor += len(blk)

        session = np.vstack(blocks)
        p = out / f"demo_Sub{subject:02d}_session.csv.gz"
        size = write_csv(p, np.arange(len(session)) * dt, session, cols)
        print(f"  -> {p.name}  {len(session):,} samples ({len(session)*dt:.1f}s)  "
              f"{size/1e6:.2f} MB gz  {len(truth)} bouts")
        pd.DataFrame(truth).to_csv(out / f"demo_Sub{subject:02d}_truth.csv", index=False)

        # ---- calibration: labelled ----
        cb, cl = [], []
        for task in LABELS:
            for blk in cal_slices[task]:
                cb.append(blk); cl.append(np.repeat(task, len(blk)))
        cal_arr = np.vstack(cb)
        pc = out / f"demo_Sub{subject:02d}_calibration.csv.gz"
        size = write_csv(pc, np.arange(len(cal_arr)) * dt, cal_arr, cols,
                         label_col=np.concatenate(cl))
        print(f"  -> {pc.name}  {len(cal_arr):,} samples ({len(cal_arr)*dt:.1f}s)  {size/1e6:.2f} MB gz")

    print(f"\nDone. Demo data in {out}")
    print("Reminder: sessions are assembled from separately-recorded real trials — "
          "state that in the user manual.")


if __name__ == "__main__":
    main()
