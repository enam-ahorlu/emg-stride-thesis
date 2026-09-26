#!/usr/bin/env python3
"""
kc23_s2_transitions.py
========================
KC-S2 measures. EXPERIMENT_PLAN_KC23_DEPLOYMENT.md "KC-S2 ... S2.4 Measures".
Descriptive only -- S2.5 states no halting letters for this stage.

Rewritten 2026-09-25. The first version (a) never read the ground-truth
transition table, it re-derived "transitions" from a per-window `mode_raw`
column (which the predictions job fills from the window's movement label, not
from the raw Mode signal) and divided a WINDOW index by fs as if it were a
sample index, and (b) grouped by subject and circuit only, so the three
models' rows in one predictions file were pooled. Run on the real
predictions it would have exited 0 with meaningless numbers. It also wrote a
placeholder verdict and exited 0 when its predictions file was missing.

Now: transition times come from results_kc23_s2_transition_table/
s2_transition_table.csv (kc23_s2_transition_table.py, built from the raw
per-sample Mode signal, direct retained-to-retained changes only), and every
measure is computed separately for each model and each normalization
condition. Any missing or malformed input stops the script with a non-zero
exit, no measures file, and a verdict that contains no outcome line.

Inputs (--preds-dir holds one file per condition, from kc23_s2_predictions.py):
  s2_predictions_{transductive,causal100,causal_balanced25}.csv
      model, subject, circuit, t_end, y_true, y_pred, mode_raw, fs
      models SVM, RESNET_SE_CD, soft; t_end is seconds from the circuit start.
  --table   s2_transition_table.csv: subject, circuit, t_change_s, from, to

Measures per (condition, model), per S2.4:
  steady-state error        windows whose end time is > 2 s from any transition
  transition-zone error     windows whose end time is within +/-1 s of one,
                            also by transition type
  critical errors           true DNS predicted WAK, inside transition zones
                            against steady state (rate among true-DNS windows)
  decision delay            from the true change to the end time of the first
                            of 3 consecutive windows predicted as the NEW class
                            (windows must be adjacent, <= 0.25 s apart), search
                            stopping at the next transition; median and IQR by
                            type, with the number of transitions never resolved
  causal five-window vote   plurality of the last 5 window predictions inside a
                            circuit (ties go to the most recent): change in
                            steady-state error, in transition-zone error, and
                            the added decision delay (paired per transition)
The S2b window trade (250 against 400 ms) is computed by kc23_s2b_window_trade.py from the 400 ms predictions; when
its reading file exists (--s2b-dir) the verdict includes it, otherwise it says Reading 2 is not computed.

Reordering, 26 September 2026: the verdict leads with the transductive and balanced25 conditions. The causal-100
collapse is reported as a finding in its own right, not as a reading.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import write_no_outcome_verdict

TRANSITION_ZONE_S = 1.0
STEADY_STATE_MARGIN_S = 2.0
VOTE_WINDOW = 5
N_CONSECUTIVE = 3
MAX_GAP_S = 0.25          # adjacent windows are 0.125 s apart (w250, 50% overlap); allow one skipped step
CONDITIONS = ["transductive", "causal100", "causal_balanced25"]
MODELS = ["SVM", "RESNET_SE_CD", "soft"]
# ENABL3S label codes are alphabetical (kc23_s2_predictions.LABELS = DNS, STDUP, UPS, WAK)
CODE_DNS, CODE_UPS, CODE_WAK = 0, 2, 3
CLASS_OF = {"LW": CODE_WAK, "SA": CODE_UPS, "SD": CODE_DNS}
TYPES = ["LW_to_SA", "SA_to_LW", "LW_to_SD", "SD_to_LW"]
PRED_COLS = {"model", "subject", "circuit", "t_end", "y_true", "y_pred", "mode_raw", "fs"}
TABLE_COLS = {"subject", "circuit", "t_change_s", "from", "to"}
EXPECTED_ROWS = len(CONDITIONS) * len(MODELS)


class InputError(Exception):
    pass


def label_windows(t_end: np.ndarray, transitions: list) -> pd.DataFrame:
    """Per window: distance to the nearest transition, and its zone. The zone
    is transition_zone within +/-1 s (type = that transition's type),
    steady_state beyond 2 s, and neither in between."""
    t_end = np.asarray(t_end, float)
    n = len(t_end)
    if not transitions:
        return pd.DataFrame({"dist_s": np.full(n, np.inf), "zone": "steady_state", "transition_type": None})
    t_times = np.array([t[0] for t in transitions], float)
    t_types = np.array([f"{t[1]}_to_{t[2]}" for t in transitions], dtype=object)
    d = np.abs(t_end[:, None] - t_times[None, :])
    i = d.argmin(axis=1)
    dist = d[np.arange(n), i]
    zone = np.where(dist <= TRANSITION_ZONE_S, "transition_zone",
                    np.where(dist > STEADY_STATE_MARGIN_S, "steady_state", "neither"))
    ttype = np.where(zone == "transition_zone", t_types[i], None)
    return pd.DataFrame({"dist_s": dist, "zone": zone, "transition_type": ttype})


def decision_delay(t_end: np.ndarray, y_pred: np.ndarray, transitions: list,
                   n_consecutive: int = N_CONSECUTIVE, max_gap_s: float | None = None) -> pd.DataFrame:
    """Per transition: delay from the true change to the end time of the first
    of n_consecutive adjacent windows all predicted as the NEW class. Only
    windows ending at or after the change and before the next transition are
    searched; NaN if the new class is never stably reached (unresolved)."""
    max_gap_s = MAX_GAP_S if max_gap_s is None else max_gap_s     # S2b passes 2 x its window step (0.40 s at 400 ms)
    t_end = np.asarray(t_end, float)
    y_pred = np.asarray(y_pred)
    rows = []
    for j, (t_c, frm, to) in enumerate(transitions):
        nxt = transitions[j + 1][0] if j + 1 < len(transitions) else np.inf
        target = CLASS_OF[to]
        idx = np.where((t_end >= t_c) & (t_end < nxt))[0]
        delay = np.nan
        run, start, prev = 0, None, None
        for k in idx:
            if y_pred[k] == target and (run > 0 and t_end[k] - t_end[prev] <= max_gap_s):
                run += 1
            elif y_pred[k] == target:
                run, start = 1, k
            else:
                run = 0
            prev = k
            if run >= n_consecutive:
                delay = float(t_end[start] - t_c)
                break
        rows.append({"transition_time": float(t_c), "type": f"{frm}_to_{to}", "delay_s": delay})
    return pd.DataFrame(rows, columns=["transition_time", "type", "delay_s"])


def causal_vote(y_pred: np.ndarray, n: int = VOTE_WINDOW) -> np.ndarray:
    """Plurality of the last n predictions (fewer at the start of a circuit);
    a tie goes to the class predicted most recently among the tied classes."""
    y_pred = np.asarray(y_pred, int)
    out = np.empty_like(y_pred)
    for i in range(len(y_pred)):
        w = y_pred[max(0, i - n + 1): i + 1]
        counts = np.bincount(w, minlength=int(y_pred.max()) + 1)
        best = np.flatnonzero(counts == counts.max())
        if len(best) == 1:
            out[i] = best[0]
        else:
            out[i] = next(c for c in w[::-1] if c in best)
    return out


def _load_preds(preds_dir: Path) -> dict:
    out = {}
    for cond in CONDITIONS:
        p = preds_dir / f"s2_predictions_{cond}.csv"
        if not p.exists():
            raise InputError(f"required predictions file missing: {p}")
        df = pd.read_csv(p)
        if not PRED_COLS <= set(df.columns):
            raise InputError(f"{p.name} lacks columns {sorted(PRED_COLS - set(df.columns))}")
        if set(df["model"]) != set(MODELS):
            raise InputError(f"{p.name} holds models {sorted(set(df['model']))}, expected {sorted(MODELS)}")
        sizes = df.groupby("model").size()
        if sizes.nunique() != 1 or df[["t_end", "y_true", "y_pred"]].isna().any().any():
            raise InputError(f"{p.name}: models have different window counts {sizes.to_dict()} or NaN values")
        out[cond] = df
    return out


def _load_table(table_path: Path) -> pd.DataFrame:
    if not table_path.exists():
        raise InputError(f"required transition table missing: {table_path}")
    t = pd.read_csv(table_path)
    if not TABLE_COLS <= set(t.columns) or len(t) == 0:
        raise InputError(f"{table_path.name} lacks columns {sorted(TABLE_COLS - set(t.columns))} or is empty")
    bad = set(t["from"]) | set(t["to"])
    if not bad <= set(CLASS_OF):
        raise InputError(f"{table_path.name} holds unknown transition codes {sorted(bad - set(CLASS_OF))}")
    return t


def _rate(mask_num: np.ndarray, mask_den: np.ndarray) -> float:
    d = int(mask_den.sum())
    return float(mask_num[mask_den].mean()) if d else float("nan")


def analyse(cond: str, model: str, df: pd.DataFrame, table: pd.DataFrame, max_gap_s: float | None = None):
    """Returns (measures dict, per-window zone frame, per-transition delay frame)."""
    trans_by = {k: sorted(zip(g["t_change_s"], g["from"], g["to"])) for k, g in table.groupby(["subject", "circuit"])}
    zone_frames, delay_frames = [], []
    for (subj, circ), g in df.groupby(["subject", "circuit"], sort=True):
        g = g.sort_values("t_end", kind="stable")
        trans = trans_by.get((subj, circ), [])
        z = label_windows(g["t_end"].to_numpy(), trans)
        z["subject"], z["circuit"] = subj, circ
        z["y_true"], z["y_pred"] = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        z["y_vote"] = causal_vote(g["y_pred"].to_numpy())
        zone_frames.append(z)
        for label, col in (("raw", "y_pred"), ("vote", "y_vote")):
            d = decision_delay(g["t_end"].to_numpy(), z[col].to_numpy(), trans, max_gap_s=max_gap_s)
            d["subject"], d["circuit"], d["source"] = subj, circ, label
            delay_frames.append(d)
    zones = pd.concat(zone_frames, ignore_index=True)
    delays = pd.concat(delay_frames, ignore_index=True)

    wrong = (zones["y_true"] != zones["y_pred"]).to_numpy()
    wrong_v = (zones["y_true"] != zones["y_vote"]).to_numpy()
    steady = (zones["zone"] == "steady_state").to_numpy()
    zone = (zones["zone"] == "transition_zone").to_numpy()
    true_dns = (zones["y_true"] == CODE_DNS).to_numpy()
    dns_as_wak = (zones["y_pred"] == CODE_WAK).to_numpy()

    def subj_mean(mask, w):
        v = [w[(zones["subject"] == s).to_numpy() & mask].mean() for s in zones["subject"].unique()
             if ((zones["subject"] == s).to_numpy() & mask).any()]
        return float(np.mean(v)) if v else float("nan")

    m = {"condition": cond, "model": model, "n_windows": len(zones), "n_subjects": int(zones["subject"].nunique()),
         "steady_n": int(steady.sum()), "steady_error": _rate(wrong, steady),
         "steady_error_subj_mean": subj_mean(steady, wrong),
         "zone_n": int(zone.sum()), "zone_error": _rate(wrong, zone),
         "zone_error_subj_mean": subj_mean(zone, wrong),
         "dns_as_wak_rate_zone": _rate(dns_as_wak, zone & true_dns), "dns_as_wak_n_true_dns_zone": int((zone & true_dns).sum()),
         "dns_as_wak_rate_steady": _rate(dns_as_wak, steady & true_dns),
         "dns_as_wak_n_true_dns_steady": int((steady & true_dns).sum()),
         "vote_steady_error": _rate(wrong_v, steady), "vote_zone_error": _rate(wrong_v, zone)}
    m["vote_delta_steady_error"] = m["vote_steady_error"] - m["steady_error"]
    m["vote_delta_zone_error"] = m["vote_zone_error"] - m["zone_error"]
    for t in TYPES:
        mt = (zones["transition_type"] == t).to_numpy() & zone
        m[f"zone_error_{t}"] = _rate(wrong, mt)
        m[f"zone_n_{t}"] = int(mt.sum())

    raw = delays[delays["source"] == "raw"].reset_index(drop=True)
    vote = delays[delays["source"] == "vote"].reset_index(drop=True)
    for t in TYPES + ["all"]:
        sel = raw if t == "all" else raw[raw["type"] == t]
        d = sel["delay_s"].dropna()
        m[f"delay_{t}_n_transitions"] = int(len(sel))
        m[f"delay_{t}_n_resolved"] = int(len(d))
        m[f"delay_{t}_median_s"] = float(d.median()) if len(d) else float("nan")
        m[f"delay_{t}_q1_s"] = float(d.quantile(0.25)) if len(d) else float("nan")
        m[f"delay_{t}_q3_s"] = float(d.quantile(0.75)) if len(d) else float("nan")
    dv = vote["delay_s"].dropna()
    m["vote_delay_all_median_s"] = float(dv.median()) if len(dv) else float("nan")
    paired = (vote["delay_s"] - raw["delay_s"]).dropna()   # same transition order in raw and vote
    m["vote_added_delay_paired_median_s"] = float(paired.median()) if len(paired) else float("nan")
    m["vote_added_delay_n_paired"] = int(len(paired))

    zones["condition"], zones["model"] = cond, model
    delays["condition"], delays["model"] = cond, model
    return m, zones, delays


LEAD_CONDITIONS = ["transductive", "causal_balanced25"]      # the readings S2 leads with (decision of 26 Sept 2026)
SEPARATE_CONDITION = "causal100"                              # reported as a finding of its own, not as a reading


def vote_reading(measures: pd.DataFrame, cond: str, model: str = "soft") -> dict:
    """S2.5 reading 1 for one condition: does the five-window vote cut steady-state error AND add under 250 ms median delay?"""
    sel = measures[(measures["condition"] == cond) & (measures["model"] == model)]
    if sel.empty:
        return {"cond": cond, "evaluated": False}
    r = sel.iloc[0]
    added_ms = r["vote_added_delay_paired_median_s"] * 1000.0
    cut = bool(r["vote_delta_steady_error"] < 0)
    under = bool(added_ms < 250.0) if not np.isnan(added_ms) else False
    return {"cond": cond, "evaluated": True, "cut": cut, "under": under, "holds": cut and under,
            "d_steady_pt": r["vote_delta_steady_error"] * 100.0, "d_zone_pt": r["vote_delta_zone_error"] * 100.0,
            "added_ms": added_ms, "n_paired": int(r["vote_added_delay_n_paired"])}


def _reading(measures: pd.DataFrame) -> str:
    """Reading 1 leads with the transductive and balanced25 conditions, on the soft vote; the causal-100 collapse follows
    as its own finding (a buffer taken from the start of a real continuous session does not work)."""
    parts = [vote_reading(measures, c) for c in LEAD_CONDITIONS]
    if not all(p["evaluated"] for p in parts):
        return "Reading 1 not evaluated (a lead-condition soft row is absent)."
    lines = [f"- {p['cond']}: steady-state error change {p['d_steady_pt']:+.2f} pt, transition-zone error change "
             f"{p['d_zone_pt']:+.2f} pt, paired median added delay {p['added_ms']:+.0f} ms ({p['n_paired']} transitions)"
             for p in parts]
    if all(p["holds"] for p in parts):
        verdict = ("the five-window vote cuts steady-state error and adds under 250 ms median delay under both lead conditions, "
                   "so Section 4.4.3's smoothing paragraph can be stated for real transitions, with these numbers")
    elif not any(p["holds"] for p in parts):
        verdict = ("the vote does not both cut steady-state error and add under 250 ms median delay under either lead condition, "
                   "so the limitation stays and gains these numbers")
    else:
        verdict = "the two lead conditions disagree, so neither statement is made without both sets of numbers"
    return "Reading 1 (soft vote; the lead conditions):\n\n" + "\n".join(lines) + f"\n\nSo: {verdict}."


def _causal100_finding(measures: pd.DataFrame) -> str:
    rows = []
    for model in MODELS:
        a = measures[(measures["condition"] == "transductive") & (measures["model"] == model)]
        b = measures[(measures["condition"] == SEPARATE_CONDITION) & (measures["model"] == model)]
        if a.empty or b.empty:
            continue
        rows.append(f"{model}: steady-state error {a.iloc[0]['steady_error']:.3f} transductive, "
                    f"{b.iloc[0]['steady_error']:.3f} causal-100")
    if not rows:
        return "Causal-100 finding not evaluated (rows absent)."
    return ("## Finding, reported on its own: the causal 100-window buffer collapses\n\nA buffer taken from the start of a real "
            "continuous session does not work as a normalization reference: " + "; ".join(rows) + ". This is not a reading "
            "of S2.5; it is the measurement that supports the need for a scripted commissioning step (decision D-6b). The "
            "causal-100 rows stay in the table and in s2_measures.csv, and Reading 1 is not taken from them.")


def _s2b_reading(s2b_dir: Path | None) -> str:
    if s2b_dir is None or not (Path(s2b_dir) / "s2b_reading.txt").exists():
        return ("Reading 2 (S2b, 400 ms against 250 ms): NOT computed here; it needs the 400 ms per-window predictions "
                "with circuit and time (kc23_s2b_window_trade.py).")
    return (Path(s2b_dir) / "s2b_reading.txt").read_text(encoding="utf-8").strip()


def run(out_dir: Path, preds_dir: Path, table_path: Path, s2b_dir: Path | None = None) -> int:
    try:
        preds = _load_preds(preds_dir)
        table = _load_table(table_path)
        pred_subjects = set().union(*(set(d["subject"]) for d in preds.values()))
        if not pred_subjects <= set(table["subject"]):
            raise InputError(f"predictions cover subjects {sorted(pred_subjects - set(table['subject']))} "
                             f"that the transition table lacks")
        rows, zone_parts, delay_parts = [], [], []
        for cond in CONDITIONS:
            for model in MODELS:
                m, z, d = analyse(cond, model, preds[cond][preds[cond]["model"] == model], table)
                rows.append(m); zone_parts.append(z); delay_parts.append(d)
                print(f"[S2] {cond:18s} {model:13s} steady {m['steady_error']:.4f} zone {m['zone_error']:.4f} "
                      f"median delay {m['delay_all_median_s']:.3f}s")
    except InputError as e:
        print(f"[S2] FAIL (no outcome computed): {e}", file=sys.stderr)
        write_no_outcome_verdict(out_dir / "S2_VERDICT.md", "KC-S2 verdict", str(e))
        m = out_dir / "s2_measures.csv"
        if m.exists():
            m.unlink()          # a stale measures file must not satisfy the queue's completeness check
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    measures = pd.DataFrame(rows)
    if len(measures) != EXPECTED_ROWS:
        write_no_outcome_verdict(out_dir / "S2_VERDICT.md", "KC-S2 verdict", f"{len(measures)} measure rows, expected {EXPECTED_ROWS}")
        return 2
    measures.to_csv(out_dir / "s2_measures.csv", index=False)
    pd.concat(zone_parts, ignore_index=True).to_csv(out_dir / "s2_zone_labels.csv", index=False)
    pd.concat(delay_parts, ignore_index=True).to_csv(out_dir / "s2_decision_delay.csv", index=False)

    key = ["condition", "model", "steady_error", "zone_error", "dns_as_wak_rate_zone", "dns_as_wak_rate_steady",
           "delay_all_median_s", "vote_delta_steady_error", "vote_delta_zone_error", "vote_added_delay_paired_median_s"]
    head = "| " + " | ".join(key) + " |\n|" + "---|" * len(key) + "\n"
    order = {c: i for i, c in enumerate(LEAD_CONDITIONS + [SEPARATE_CONDITION])}
    shown = measures.assign(_o=measures["condition"].map(order)).sort_values(["_o"], kind="stable")
    body = "".join("| " + " | ".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in key) + " |\n"
                   for _, r in shown.iterrows())
    (out_dir / "S2_VERDICT.md").write_text(
        "# KC-S2 verdict\n\nDescriptive only (no halting letters, per S2.5). The readings lead with the transductive and "
        "balanced25 conditions; the causal-100 collapse is reported separately below. Measures per condition and model, "
        "windows scored against the ground-truth transition table (rows in that order).\n\n" + head + body +
        f"\n## Readings\n\n{_reading(measures)}\n\n{_causal100_finding(measures)}\n\n{_s2b_reading(s2b_dir)}\n\n"
        f"Full tables: s2_measures.csv, s2_decision_delay.csv, s2_zone_labels.csv.\n", encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--s2b-dir", default=None, help="results_kc23_s2b_window_trade, for Reading 2 (optional)")
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.preds_dir), Path(args.table), Path(args.s2b_dir) if args.s2b_dir else None))


if __name__ == "__main__":
    main()
