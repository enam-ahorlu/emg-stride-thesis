#!/usr/bin/env python3
"""
kc23_s2_transition_table.py
==============================
KC-S2 F0/S2.2. Builds the ground-truth transition table directly from the raw
per-sample Mode signal (adapt_external_dataset.load_subject_trials, which
already reads it via _build_labels), with the circuit id and per-circuit time
the F0 extension added -- NOT a time-gap heuristic on the windowed data.

A transition is kept only when:
  - both sides are RETAINED modes (LevelWalking=LW, StairAscent=SA,
    StairDescent=SD -- adapt_external_dataset.MODE_LW/SA/SD, raw codes 1/4/5);
  - the change is DIRECT: sample i is one retained mode, sample i+1 is a
    DIFFERENT retained mode. Any change that passes through even one sample of
    a dropped mode (ramps 2/3, standing 6, sitting 0, or the synthetic STS
    window _build_labels overlays near a sit-to-stand event) becomes two
    separate non-qualifying changes (retained->dropped, dropped->retained),
    per S2.2's "no dropped mode in between."
STS/STDUP is not a retained mode for this table -- it is a synthetic label
_build_labels paints onto a fixed window around a raw sit->stand event, not a
raw Mode value, so a published STDUP window has no corresponding "segment"
here (see validate_against_windows).

Output columns: subject, circuit, t_change_s, from, to.

Validation (S2.2): every published ENABL3S window (from the F0 adapter's
--with-circuit-meta output: subject, movement, t_start, fs, win_samples,
circuit, t_start_circuit) that lies WHOLLY inside one label run must carry
that run's class. Disagreements are reported, not silently dropped.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RETAINED = ("LW", "SA", "SD")
LABEL_MAP = {"LW": "WAK", "SA": "UPS", "SD": "DNS"}  # adapt_external_dataset.LABEL_MAP, minus STS


def find_transitions_one_circuit(labels: np.ndarray, fs: float) -> list[tuple[float, str, str]]:
    """labels: adapt_external_dataset._build_labels' per-sample string-code
    array (LW/SA/SD/STS/OTHER). Returns [(t_change_s, from, to), ...] for
    every DIRECT sample-to-sample change between two different retained
    modes."""
    out = []
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        if a in RETAINED and b in RETAINED and a != b:
            out.append(((i + 1) / fs, a, b))
    return out


def label_runs(labels: np.ndarray, fs: float) -> list[tuple[float, float, str]]:
    """Run-length encoding of the per-sample label sequence into
    [(start_s, end_s, code), ...] maximal runs, for the window-agreement
    check (a segment's class is well-defined regardless of whether its
    boundaries qualify as a "direct" transition in find_transitions_one_circuit)."""
    if len(labels) == 0:
        return []
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start / fs, i / fs, labels[start]))
            start = i
    return runs


def build_transition_table(root: Path, skip_sids: set | None = None) -> pd.DataFrame:
    import adapt_external_dataset as aed
    rows = []
    for sid, emg, labels, fs, circuit in aed.load_subject_trials(root, skip_sids=skip_sids):
        for t_change_s, a, b in find_transitions_one_circuit(labels, fs):
            rows.append({"subject": sid, "circuit": circuit, "t_change_s": t_change_s, "from": a, "to": b})
    return pd.DataFrame(rows, columns=["subject", "circuit", "t_change_s", "from", "to"])


def build_runs_table(root: Path, skip_sids: set | None = None) -> pd.DataFrame:
    import adapt_external_dataset as aed
    rows = []
    for sid, emg, labels, fs, circuit in aed.load_subject_trials(root, skip_sids=skip_sids):
        for start_s, end_s, code in label_runs(labels, fs):
            rows.append({"subject": sid, "circuit": circuit, "start_s": start_s, "end_s": end_s, "code": code})
    return pd.DataFrame(rows, columns=["subject", "circuit", "start_s", "end_s", "code"])


def validate_against_windows(runs: pd.DataFrame, windows_meta: pd.DataFrame,
                             win_ms: float = 250.0) -> pd.DataFrame:
    """windows_meta: subject, movement, fs, win_samples, circuit, t_start_circuit
    (the F0 adapter's --with-circuit-meta output). movement is one of
    WAK/UPS/DNS/STDUP; STDUP windows are skipped (no raw-mode segment
    corresponds to the synthetic STS label -- see module docstring), and any
    window this cannot place inside exactly one run is also skipped (not
    silently counted as agreeing). Returns the DISAGREEMENT rows only."""
    disagreements = []
    runs_by_key = {(s, c): g.sort_values("start_s") for (s, c), g in runs.groupby(["subject", "circuit"])}
    for _, w in windows_meta.iterrows():
        if w["movement"] == "STDUP":
            continue
        key = (w["subject"], w["circuit"])
        if key not in runs_by_key:
            continue
        fs = float(w["fs"])
        t0 = float(w["t_start_circuit"]) / fs
        t1 = t0 + float(w["win_samples"]) / fs
        g = runs_by_key[key]
        hit = g[(g["start_s"] <= t0) & (g["end_s"] >= t1)]
        if len(hit) != 1:
            continue  # spans a boundary or no run found -- not a "wholly inside" case
        run_code = hit.iloc[0]["code"]
        expected = LABEL_MAP.get(run_code)
        if expected is None or expected != w["movement"]:
            disagreements.append({"subject": w["subject"], "circuit": w["circuit"],
                                  "t_start_circuit": w["t_start_circuit"], "published_movement": w["movement"],
                                  "run_code": run_code, "expected_movement": expected})
    return pd.DataFrame(disagreements)


def run(out_dir: Path, root: Path, windows_meta_path: Path | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    trans = build_transition_table(root)
    trans.to_csv(out_dir / "s2_transition_table.csv", index=False)
    counts = (trans.groupby(["subject", "from", "to"]).size().rename("n").reset_index())
    counts.to_csv(out_dir / "s2_transition_counts.csv", index=False)
    print(f"[S2] {len(trans)} transitions across {trans['subject'].nunique() if len(trans) else 0} subjects")

    runs = build_runs_table(root)
    runs.to_csv(out_dir / "s2_label_runs.csv", index=False)

    if windows_meta_path is not None and windows_meta_path.exists():
        wm = pd.read_csv(windows_meta_path)
        disagree = validate_against_windows(runs, wm)
        disagree.to_csv(out_dir / "s2_window_agreement_disagreements.csv", index=False)
        print(f"[S2] window-agreement check: {len(disagree)} disagreement(s) "
             f"(0 = every wholly-inside published window matches its segment's class)")
    else:
        print(f"[S2] no windows_meta given/found -- skipping the window-agreement check")

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="ENABL3S raw root (5362627)")
    ap.add_argument("--out", default="results_kc23_s2_transition_table")
    ap.add_argument("--windows-meta", default=None)
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.root), Path(args.windows_meta) if args.windows_meta else None))


if __name__ == "__main__":
    main()
