# adapt_external_dataset.py
# ---------------------------------------------------------------------------
# Action 2.1 — Adapter: convert the ENABL3S download into this project's exact
# windowed-NPZ + meta schema, so extract_features.py and the train_*_loso.py
# scripts run UNCHANGED on the new data.
#
# >>> PRE-FILLED FOR THE ACTUAL ENABL3S DOWNLOAD (figshare 5362627). <<<
# Verified against the real files, NOT the MATLAB code:
#   - Data are CSV, not .mat:  <root>/AB###/Raw/AB###_Circuit_###_raw.csv
#   - Each raw CSV has 49 columns: 30 IMU, then 14 EMG (cols 31-44), 4 goniometer,
#     and a per-sample integer "Mode" label (col 49). No MATLAB/PVD step needed.
#   - EMG columns (in order): Right_TA,MG,SOL,BF,ST,VL,RF, Left_TA,MG,SOL,BF,ST,VL,RF
#   - Mode codes (confirmed from "Subject Trigger Channel Feature Information.xlsx"):
#       0=Sitting 1=LevelWalking 2=RampAscent 3=RampDescent 4=StairAscent
#       5=StairDescent 6=Standing
#   - Sampling rate FS = 1000 Hz (EMG native rate; Hu et al. 2018).
#
# Preprocessing mirrors preprocess_emg.py: band-pass 20-450 Hz (4th-order
# Butterworth, zero-phase), rectify + 50 ms envelope, 250 ms windows, 50%
# overlap, 60% label-purity. (At FS=1000, 450 Hz < Nyquist 500, so the band is
# applied exactly as in the thesis.)
#
# Output (to --out):
#   windows_<tag>.npz       keys X_raw, X_env   (N, C, T)   [C=14, T=250 by default]
#   windows_<tag>_meta.csv  cols: subject, movement, t_start, fs, win_samples, n_channels
# Then:  python extract_features.py --windows <out>/windows_<tag>.npz ...
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ----- preprocessing params (identical to preprocess_emg.py) ----------------
BP_LOW, BP_HIGH, BP_ORDER = 20.0, 450.0, 4
ENV_MS = 50.0
WIN_MS = 250.0
OVERLAP = 0.5
MIN_PURITY = 0.60
FS = 1000.0                # ENABL3S combined raw CSV sample rate (EMG native, Hu et al. 2018)

# ----- ENABL3S raw CSV layout ----------------------------------------------
EMG_COLS = ["Right_TA", "Right_MG", "Right_SOL", "Right_BF", "Right_ST", "Right_VL", "Right_RF",
            "Left_TA", "Left_MG", "Left_SOL", "Left_BF", "Left_ST", "Left_VL", "Left_RF"]
MODE_COL = "Mode"
MODE_SIT, MODE_LW, MODE_RA, MODE_RD, MODE_SA, MODE_SD, MODE_STAND = 0, 1, 2, 3, 4, 5, 6
STS_HALFWIN_S = 1.0        # seconds either side of a sit->stand transition -> STDUP

# (2) mode short-code -> thesis class. Ramps (RA/RD) are dropped (no SIAT match).
LABEL_MAP = {"LW": "WAK", "SA": "UPS", "SD": "DNS", "STS": "STDUP"}

# (3) Optional: restrict to a subset of the 14 EMG channels (e.g. one leg's 7,
#     to mimic SIAT's single-leg montage). None = use all 14.
#     Right leg = indices 0..6, Left leg = 7..13.
CHANNEL_SUBSET = list(range(0, 7))   # RIGHT leg's 7 muscles (TA,MG,SOL,BF,ST,VL,RF) —
                                     # mirrors SIAT's single-leg, low-channel montage for a
                                     # like-for-like replication. Set to None to use all 14.


def _build_labels(mode: np.ndarray, fs: float) -> np.ndarray:
    n = len(mode)
    lab = np.array(["OTHER"] * n, dtype=object)
    lab[mode == MODE_LW] = "LW"
    lab[mode == MODE_SA] = "SA"
    lab[mode == MODE_SD] = "SD"
    half = int(round(STS_HALFWIN_S * fs))
    trans = np.where((mode[:-1] == MODE_SIT) & (mode[1:] == MODE_STAND))[0]
    for t in trans:
        lab[max(0, t - half): min(n, t + half)] = "STS"
    return lab


def load_subject_trials(root: Path, skip_sids: set | None = None):
    """Yield (subject_id:int, emg[T,C], labels[T] (str codes), fs:float) per circuit CSV.

    skip_sids: subject ids to skip entirely (not even globbed/read) -- used by
    --resume so already-checkpointed subjects cost no I/O on a re-run.
    """
    skip_sids = skip_sids or set()
    subj_dirs = sorted([d for d in root.glob("AB*") if d.is_dir()])
    if not subj_dirs:
        raise SystemExit(f"No AB### subject folders under {root}.")
    for sd in subj_dirs:
        m = re.search(r"AB(\d+)", sd.name)
        if not m:
            continue
        sid = int(m.group(1))
        if sid in skip_sids:
            print(f"[resume] skip AB{sid:03d} (already checkpointed)")
            continue
        raw_dir = sd / "Raw"
        csvs = sorted(raw_dir.glob("*_raw.csv")) if raw_dir.exists() else sorted(sd.glob("**/*_raw.csv"))
        if not csvs:
            print(f"[warn] no *_raw.csv in {sd}")
            continue
        for csv in csvs:
            df = pd.read_csv(csv)
            missing = [c for c in EMG_COLS + [MODE_COL] if c not in df.columns]
            if missing:
                raise SystemExit(f"{csv.name} missing expected columns {missing}. "
                                 f"Found: {list(df.columns)[:6]}...")
            emg = df[EMG_COLS].to_numpy(dtype=float)
            if CHANNEL_SUBSET is not None:
                emg = emg[:, CHANNEL_SUBSET]
            labels = _build_labels(df[MODE_COL].to_numpy(), FS)
            yield sid, emg, labels, FS


# ---------------------------------------------------------------------------
def bandpass(x, fs):
    lo, hi = BP_LOW / (fs / 2.0), min(BP_HIGH / (fs / 2.0), 0.999)
    b, a = butter(BP_ORDER, [lo, hi], btype="bandpass")
    return filtfilt(b, a, x, axis=0)


def envelope(xf, fs):
    rect = np.abs(xf)
    k = max(1, int(round(ENV_MS * fs / 1000.0)))
    ker = np.ones(k) / k
    return np.stack([np.convolve(rect[:, c], ker, mode="same") for c in range(rect.shape[1])], axis=1)


def majority_label(win):
    vals, counts = np.unique(win.astype(str), return_counts=True)
    i = int(np.argmax(counts))
    return vals[i], counts[i] / counts.sum()


def window_trial(emg_raw, labels, fs, subject, t_offset):
    xf = bandpass(emg_raw, fs)
    env = envelope(xf, fs)
    win = int(round(WIN_MS * fs / 1000.0))
    step = max(1, int(round(win * (1.0 - OVERLAP))))
    Xr, Xe, meta = [], [], []
    for s in range(0, len(emg_raw) - win + 1, step):
        e = s + win
        raw_mode, conf = majority_label(labels[s:e])
        cls = LABEL_MAP.get(str(raw_mode), None)
        if cls is None or conf < MIN_PURITY:
            continue
        Xr.append(xf[s:e].T)
        Xe.append(env[s:e].T)
        meta.append({"subject": int(subject), "movement": cls, "t_start": int(t_offset + s),
                     "fs": float(fs), "win_samples": int(win), "n_channels": xf.shape[1]})
    return Xr, Xe, meta


def _ckpt_paths(ckpt_dir: Path, sid: int):
    return (ckpt_dir / f"sub{sid:03d}_windows.npz", ckpt_dir / f"sub{sid:03d}_meta.csv")


def _flush_subject(ckpt_dir: Path, sid: int, Xr_list, Xe_list, meta_list):
    """Write one subject's accumulated windows to disk (crash-safe checkpoint)."""
    if not Xr_list:
        return
    npz_path, meta_path = _ckpt_paths(ckpt_dir, sid)
    Xr = np.stack(Xr_list).astype(np.float32)
    Xe = np.stack(Xe_list).astype(np.float32)
    np.savez_compressed(npz_path, X_raw=Xr, X_env=Xe)
    pd.DataFrame(meta_list).to_csv(meta_path, index=False)
    print(f"[checkpoint] AB{sid:03d}: {len(Xr_list)} windows -> {npz_path.name}")


def main():
    ap = argparse.ArgumentParser("Adapter: ENABL3S CSV -> thesis windowed-NPZ schema.")
    ap.add_argument("--root", required=True, help="ENABL3S folder (contains AB###/Raw/*_raw.csv), e.g. 5362627")
    ap.add_argument("--out", default="features_out_ext")
    ap.add_argument("--tag", default="ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60")
    ap.add_argument("--resume", action="store_true",
                    help="Resume: skip subjects already checkpointed in --out/checkpoints; "
                         "merge everything found on disk into the final NPZ at the end.")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)

    done_sids = set()
    if args.resume:
        for f in ckpt_dir.glob("sub*_windows.npz"):
            m = re.search(r"sub(\d+)_windows\.npz", f.name)
            if m:
                done_sids.add(int(m.group(1)))
        if done_sids:
            print(f"[resume] {len(done_sids)} subjects already checkpointed, skipping them")

    subj_offset = {}
    cur_sid = None
    cur_Xr, cur_Xe, cur_meta = [], [], []
    for sid, emg, labels, fs in load_subject_trials(Path(args.root), skip_sids=done_sids):
        if cur_sid is not None and sid != cur_sid:
            _flush_subject(ckpt_dir, cur_sid, cur_Xr, cur_Xe, cur_meta)
            cur_Xr, cur_Xe, cur_meta = [], [], []
        cur_sid = sid

        off = subj_offset.get(sid, 0)
        Xr, Xe, meta = window_trial(np.asarray(emg, float), np.asarray(labels), float(fs), sid, off)
        subj_offset[sid] = off + len(emg)
        cur_Xr += Xr; cur_Xe += Xe; cur_meta += meta
        if Xr:
            print(f"[AB{sid:03d}] +{len(Xr)} windows (subject running total {len(cur_Xr)})")
    if cur_sid is not None:
        _flush_subject(ckpt_dir, cur_sid, cur_Xr, cur_Xe, cur_meta)

    # ---- Merge: read every subject's checkpoint FROM DISK (not from this run's
    # in-memory accumulation), so a fully-resumed run still produces the complete
    # final output even if it processed zero subjects fresh this time. ----
    ckpt_files = sorted(ckpt_dir.glob("sub*_windows.npz"))
    if not ckpt_files:
        raise SystemExit("No windows produced — check EMG column names / Mode codes / LABEL_MAP.")
    allXr, allXe, allmeta = [], [], []
    for npz_path in ckpt_files:
        m = re.search(r"sub(\d+)_windows\.npz", npz_path.name)
        sid = int(m.group(1))
        _, meta_path = _ckpt_paths(ckpt_dir, sid)
        d = np.load(npz_path)
        allXr.append(d["X_raw"]); allXe.append(d["X_env"])
        allmeta.append(pd.read_csv(meta_path))

    shapes = {x.shape[1:] for x in allXr}
    if len(shapes) != 1:
        raise SystemExit(f"Inconsistent window shapes across subjects: {shapes}.")
    Xr = np.concatenate(allXr).astype(np.float32)
    Xe = np.concatenate(allXe).astype(np.float32)
    meta = pd.concat(allmeta, ignore_index=True)

    npz_path = out / f"windows_{args.tag}.npz"
    np.savez_compressed(npz_path, X_raw=Xr, X_env=Xe)
    meta.to_csv(out / f"windows_{args.tag}_meta.csv", index=False)
    print(f"\n[done] {Xr.shape[0]} windows, (C,T)={Xr.shape[1:]}, subjects={meta['subject'].nunique()}")
    print("[class counts]\n" + meta["movement"].value_counts().to_string())
    print(f"[save] {npz_path}")
    print(f"[save] {out / f'windows_{args.tag}_meta.csv'}")
    print("\nNext: run extract_features.py on this NPZ, then the train_*_loso.py commands "
          "(EXPERIMENTS_README.md), pointed at the new feature/meta files.")


if __name__ == "__main__":
    main()
