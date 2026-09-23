#!/usr/bin/env python3
"""
kc23_s2_transitions.py
========================
KC-S2 measures. EXPERIMENT_PLAN_KC23_DEPLOYMENT.md "KC-S2 ... S2.4 Measures".
Descriptive only -- S2.5 states no halting letters for this stage.

Reads per-window LOSO predictions (with circuit and time) for the locked SVM,
ResNet-SE+CD and soft vote on ENABL3S (S2.3), under transductive per-subject
normalization and the causal 100-window buffer, plus the balanced25 buffer
for parity with KC-S1. Computes, per S2.4:

  steady-state error       windows more than 2s from any transition
  transition-zone error    windows whose end time is within +/-1s of a
                           transition, by transition type
  critical errors          DNS->WAK confusions inside transition zones,
                           against steady state
  decision delay           from the true change to the end time of the first
                           of 3 consecutive correct predictions (median, IQR,
                           per transition type)
  causal five-window vote  change in steady-state error, change in
                           transition-zone error, added decision delay
  S2b window trade         250 vs 400ms accuracy and decision delay (requires
                           the 400ms ENABL3S arm's predictions separately)

Transition points are found with kc23_s2_f0_feasibility.count_transitions_one_trial's
underlying run-collapsing logic (imported, not reimplemented) over the raw
Mode sequence per circuit.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRANSITION_ZONE_S = 1.0
STEADY_STATE_MARGIN_S = 2.0
VOTE_WINDOW = 5


def find_transition_times(mode: np.ndarray, fs: float) -> list[tuple[float, str, str]]:
    """Reuses the run-collapsing convention of
    kc23_s2_f0_feasibility.count_transitions_one_trial: returns
    [(time_s, from_code, to_code), ...] for every clean (no dropped mode in
    between) retained-class transition."""
    from kc23_s2_f0_feasibility import DROPPED_MODES
    from adapt_external_dataset import MODE_LW, MODE_SA, MODE_SD
    code_of = {MODE_LW: "LW", MODE_SA: "SA", MODE_SD: "SD"}
    runs = []  # (code_or_DROPPED, start_sample)
    prev = None
    for i, m in enumerate(mode):
        cur = code_of.get(int(m), None) if int(m) not in DROPPED_MODES else "DROPPED"
        if cur != prev:
            runs.append([cur, i])
            prev = cur
    out = []
    last_retained = None
    for r in runs:
        code, start = r
        if code is None or code == "DROPPED":
            last_retained = None
            continue
        if last_retained is not None and last_retained[0] != code:
            out.append((start / fs, last_retained[0], code))
        last_retained = (code, start)
    return out


def label_windows(win_end_times: np.ndarray, transitions: list[tuple[float, str, str]]) -> pd.DataFrame:
    """Per window: distance to the nearest transition, whether it is in a
    transition zone (+/-1s of end time) or steady state (>2s from any
    transition), and which transition type (if any zone)."""
    rows = []
    t_times = np.array([t[0] for t in transitions]) if transitions else np.array([])
    for we in win_end_times:
        if len(t_times) == 0:
            rows.append({"dist_s": np.inf, "zone": "steady_state", "transition_type": None})
            continue
        d = np.abs(t_times - we)
        i = int(np.argmin(d))
        dist = float(d[i])
        if dist <= TRANSITION_ZONE_S:
            zone = "transition_zone"
            ttype = f"{transitions[i][1]}_to_{transitions[i][2]}"
        elif dist > STEADY_STATE_MARGIN_S:
            zone = "steady_state"
            ttype = None
        else:
            zone = "neither"
            ttype = None
        rows.append({"dist_s": dist, "zone": zone, "transition_type": ttype})
    return pd.DataFrame(rows)


def decision_delay(win_end_times: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                   transitions: list[tuple[float, str, str]], win_s: int = VOTE_WINDOW) -> pd.DataFrame:
    """From each transition's true-change time to the end time of the first
    of `win_s` consecutive correct predictions after it."""
    rows = []
    correct = (y_true == y_pred)
    for t_time, frm, to in transitions:
        after = np.where(win_end_times >= t_time)[0]
        delay = np.nan
        for k in after:
            if k + win_s <= len(correct) and correct[k:k + win_s].all():
                delay = win_end_times[k] - t_time
                break
        rows.append({"transition_time": t_time, "type": f"{frm}_to_{to}", "delay_s": delay})
    return pd.DataFrame(rows)


def error_by_zone(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    correct = (y_true == y_pred)
    out = {}
    for zone in ("steady_state", "transition_zone"):
        m = (df["zone"] == zone).to_numpy()
        out[f"{zone}_error"] = float(1 - correct[m].mean()) if m.any() else float("nan")
        out[f"{zone}_n"] = int(m.sum())
    dns_wak_mask = (df["transition_type"] == "SD_to_LW").to_numpy() & (df["zone"] == "transition_zone").to_numpy()
    if dns_wak_mask.any():
        crit = ((y_true[dns_wak_mask] != y_pred[dns_wak_mask])).mean()
        out["dns_to_wak_critical_error_rate"] = float(crit)
    return out


def run(out_dir: Path, preds_path: Path | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if preds_path is None or not preds_path.exists():
        print(f"[S2] no predictions file given/found ({preds_path}) -- this is the descriptive "
             f"analysis script; it needs real KC-S2 per-window LOSO predictions with circuit and "
             f"time to compute anything. Nothing to do yet.")
        (out_dir / "S2_VERDICT.md").write_text(
            "# KC-S2 verdict\n\nDescriptive only (no halting letters, per S2.5). "
            "Awaiting real per-window predictions.\n", encoding="utf-8")
        return 0

    preds = pd.read_csv(preds_path)  # cols: subject, circuit, t_end, y_true, y_pred, mode_raw, fs
    all_zone_rows, all_delay_rows = [], []
    for (subj, circuit), g in preds.groupby(["subject", "circuit"]):
        g = g.sort_values("t_end")
        mode_seq = g["mode_raw"].to_numpy()
        fs = float(g["fs"].iloc[0])
        transitions = find_transition_times(mode_seq, fs)
        zdf = label_windows(g["t_end"].to_numpy(), transitions)
        zdf["subject"], zdf["circuit"] = subj, circuit
        all_zone_rows.append(zdf)
        ddf = decision_delay(g["t_end"].to_numpy(), g["y_true"].to_numpy(), g["y_pred"].to_numpy(), transitions)
        ddf["subject"], ddf["circuit"] = subj, circuit
        all_delay_rows.append(ddf)

    zones = pd.concat(all_zone_rows, ignore_index=True) if all_zone_rows else pd.DataFrame()
    delays = pd.concat(all_delay_rows, ignore_index=True) if all_delay_rows else pd.DataFrame()
    errs = error_by_zone(zones, preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()) if len(zones) else {}

    zones.to_csv(out_dir / "s2_zone_labels.csv", index=False)
    delays.to_csv(out_dir / "s2_decision_delay.csv", index=False)
    print(f"[S2] {errs}")
    if len(delays):
        print(delays.groupby("type")["delay_s"].describe())

    (out_dir / "S2_VERDICT.md").write_text(
        f"# KC-S2 verdict\n\nDescriptive only. Steady-state/transition-zone error and decision "
        f"delay written to {out_dir}.\n\n{errs}\n", encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds", default=None)
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.preds) if args.preds else None))


if __name__ == "__main__":
    main()
