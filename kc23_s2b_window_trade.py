#!/usr/bin/env python3
"""
kc23_s2b_window_trade.py
==========================
KC-S2b, the window trade. EXPERIMENT_PLAN_KC23_DEPLOYMENT.md S2.4: "build ENABL3S windows at 400 ms (the same adapter, a new
tag), train the locked ResNet-SE+CD and SVM at 400 ms, and report accuracy and decision delay at 250 against 400 ms. This
measures the latency cost Section 4.4.6 argues from the literature." S2.5 reading 2: "If 400 ms adds at least 100 ms median
delay for less than a 2-point accuracy gain, Section 4.4.6's trade is supported by a lower-limb measurement. Otherwise, it is
reported as measured." Descriptive: no halting letters.

Inputs (both from kc23_s2_predictions.py, which runs the same locked pipeline at --window-ms 250 and 400):
  --preds-250 results_kc23_s2_predictions      s2_predictions_{transductive,causal100,causal_balanced25}.csv
  --preds-400 results_kc23_s2b_predictions     the same three files, at 400 ms
  --table     s2_transition_table.csv           transitions are properties of the raw Mode signal, so both window lengths are
                                                scored against the SAME table
Measures per (condition, model): window accuracy (all windows and steady-state), decision delay median and IQR (250 and 400),
and the PAIRED per-transition difference in delay (400 minus 250; only transitions resolved at both lengths). The 3-consecutive
rule is unchanged; the adjacency allowance is twice the window step (0.25 s at 250 ms, 0.40 s at 400 ms), so it means the same
thing at both lengths.

Reading 2 is taken from the lead conditions (transductive, balanced25), per locked model (SVM, ResNet-SE+CD); the causal-100
rows are reported separately, as in S2. The queue's completeness check is s2b_measures.csv; the verdict contains no outcome
line. When --s2-out is given, the S2 verdict is regenerated with Reading 2 included.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import kc23_s2_transitions as s2
from kc23_stats_common import write_no_outcome_verdict

GAP_250_S = 0.25
GAP_400_S = 0.40
LOCKED_MODELS = ["SVM", "RESNET_SE_CD"]
ADDED_DELAY_MIN_MS = 100.0
ACCURACY_GAIN_MAX_PP = 2.0
EXPECTED_ROWS = len(s2.CONDITIONS) * len(s2.MODELS)


def _accuracy(zones: pd.DataFrame) -> dict:
    ok = (zones["y_true"] == zones["y_pred"]).to_numpy()
    steady = (zones["zone"] == "steady_state").to_numpy()
    return {"accuracy": float(ok.mean()), "steady_accuracy": float(ok[steady].mean()) if steady.any() else float("nan")}


def _paired_delay(d250: pd.DataFrame, d400: pd.DataFrame) -> pd.Series:
    key = ["subject", "circuit", "transition_time", "type"]
    a = d250[d250["source"] == "raw"][key + ["delay_s"]]
    b = d400[d400["source"] == "raw"][key + ["delay_s"]]
    j = a.merge(b, on=key, suffixes=("_250", "_400"))
    return (j["delay_s_400"] - j["delay_s_250"]).dropna()


def compare(preds250: dict, preds400: dict, table: pd.DataFrame):
    rows = []
    for cond in s2.CONDITIONS:
        for model in s2.MODELS:
            m250, z250, d250 = s2.analyse(cond, model, preds250[cond][preds250[cond]["model"] == model], table, GAP_250_S)
            m400, z400, d400 = s2.analyse(cond, model, preds400[cond][preds400[cond]["model"] == model], table, GAP_400_S)
            a250, a400 = _accuracy(z250), _accuracy(z400)
            paired = _paired_delay(d250, d400)
            rows.append({
                "condition": cond, "model": model,
                "accuracy_250": a250["accuracy"], "accuracy_400": a400["accuracy"],
                "accuracy_gain_pp": (a400["accuracy"] - a250["accuracy"]) * 100.0,
                "steady_accuracy_250": a250["steady_accuracy"], "steady_accuracy_400": a400["steady_accuracy"],
                "delay_median_250_s": m250["delay_all_median_s"], "delay_q1_250_s": m250["delay_all_q1_s"],
                "delay_q3_250_s": m250["delay_all_q3_s"],
                "delay_median_400_s": m400["delay_all_median_s"], "delay_q1_400_s": m400["delay_all_q1_s"],
                "delay_q3_400_s": m400["delay_all_q3_s"],
                "n_transitions_250": m250["delay_all_n_transitions"], "n_resolved_250": m250["delay_all_n_resolved"],
                "n_resolved_400": m400["delay_all_n_resolved"],
                "added_delay_paired_median_ms": float(paired.median() * 1000.0) if len(paired) else float("nan"),
                "n_paired": int(len(paired))})
    return pd.DataFrame(rows)


def reading_2(measures: pd.DataFrame) -> str:
    """S2.5 reading 2 per locked model on the lead conditions; a model holds when 400 ms adds >= 100 ms median delay (paired) for
    < 2 pt accuracy gain."""
    lines, holds = [], []
    for cond in s2.LEAD_CONDITIONS:
        for model in LOCKED_MODELS:
            r = measures[(measures["condition"] == cond) & (measures["model"] == model)].iloc[0]
            h = bool(r["added_delay_paired_median_ms"] >= ADDED_DELAY_MIN_MS and r["accuracy_gain_pp"] < ACCURACY_GAIN_MAX_PP)
            holds.append(h)
            lines.append(f"- {cond}, {model}: 400 ms adds {r['added_delay_paired_median_ms']:+.0f} ms paired median delay "
                         f"({int(r['n_paired'])} transitions) for a {r['accuracy_gain_pp']:+.2f} pt accuracy change "
                         f"({r['accuracy_250']:.3f} to {r['accuracy_400']:.3f}): {'meets' if h else 'does not meet'} the reading")
    if all(holds):
        verdict = "400 ms adds at least 100 ms median delay for less than a 2-point accuracy gain in every lead cell, so Section 4.4.6's trade is supported by a lower-limb measurement"
    elif not any(holds):
        verdict = "no lead cell shows at least 100 ms added median delay for less than a 2-point accuracy gain, so the trade is reported as measured"
    else:
        verdict = "the lead cells disagree, so the trade is reported as measured, cell by cell"
    return ("Reading 2 (S2b, 400 ms against 250 ms; the locked SVM and ResNet-SE+CD, lead conditions):\n\n" + "\n".join(lines) +
            f"\n\nSo: {verdict}.")


def run(out_dir: Path, preds_250: Path, preds_400: Path, table_path: Path, s2_out: Path | None = None) -> int:
    try:
        p250 = s2._load_preds(preds_250)
        p400 = s2._load_preds(preds_400)
        table = s2._load_table(table_path)
        for name, p in (("250", p250), ("400", p400)):
            subs = set().union(*(set(d["subject"]) for d in p.values()))
            if not subs <= set(table["subject"]):
                raise s2.InputError(f"{name} ms predictions cover subjects {sorted(subs - set(table['subject']))} the table lacks")
        if set().union(*(set(d["subject"]) for d in p250.values())) != set().union(*(set(d["subject"]) for d in p400.values())):
            raise s2.InputError("the 250 and 400 ms predictions cover different subjects")
        measures = compare(p250, p400, table)
        if len(measures) != EXPECTED_ROWS:
            raise s2.InputError(f"{len(measures)} measure rows, expected {EXPECTED_ROWS}")
    except s2.InputError as e:
        print(f"[S2b] FAIL (no outcome computed): {e}", file=sys.stderr)
        write_no_outcome_verdict(out_dir / "S2B_VERDICT.md", "KC-S2b verdict", str(e))
        for f in ("s2b_measures.csv", "s2b_reading.txt"):
            (out_dir / f).unlink(missing_ok=True)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    measures.to_csv(out_dir / "s2b_measures.csv", index=False)
    text = reading_2(measures)
    (out_dir / "s2b_reading.txt").write_text(text + "\n", encoding="utf-8")
    key = ["condition", "model", "accuracy_250", "accuracy_400", "accuracy_gain_pp", "delay_median_250_s", "delay_median_400_s",
           "added_delay_paired_median_ms", "n_paired"]
    order = {c: i for i, c in enumerate(s2.LEAD_CONDITIONS + [s2.SEPARATE_CONDITION])}
    shown = measures.assign(_o=measures["condition"].map(order)).sort_values("_o", kind="stable")
    head = "| " + " | ".join(key) + " |\n|" + "---|" * len(key) + "\n"
    body = "".join("| " + " | ".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in key) + " |\n"
                   for _, r in shown.iterrows())
    (out_dir / "S2B_VERDICT.md").write_text(
        "# KC-S2b verdict\n\nDescriptive only (no halting letters). Rows lead with the transductive and balanced25 conditions; "
        "the causal-100 rows are reported separately.\n\n" + head + body + f"\n{text}\n", encoding="utf-8")
    if s2_out is not None:
        rc = s2.run(Path(s2_out), preds_250, table_path, out_dir)        # regenerate S2_VERDICT.md with Reading 2 included
        if rc != 0:
            return rc
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results_kc23_s2b_window_trade")
    ap.add_argument("--preds-250", default="results_kc23_s2_predictions")
    ap.add_argument("--preds-400", default="results_kc23_s2b_predictions")
    ap.add_argument("--table", default="results_kc23_s2_transition_table/s2_transition_table.csv")
    ap.add_argument("--s2-out", default=None)
    args = ap.parse_args()
    sys.exit(run(Path(args.out), Path(args.preds_250), Path(args.preds_400), Path(args.table),
                 Path(args.s2_out) if args.s2_out else None))


if __name__ == "__main__":
    main()
