#!/usr/bin/env python3
"""
src/kc23_d6_retry_job_gen.py
==========================
docs/plans/EXPERIMENT_PLAN_KC23_DEEP.md 6.3: "An arm has diverged if the source validation loss at its best epoch exceeds twice that of
the same seed's lambda_max = 0 arm, or any loss is NaN. A diverged arm is retried ONCE with --grad-clip 5.0. If it diverges
again, it is recorded as diverged (a result: training destroyed at that strength) and never dropped silently." Written 26
September 2026, before D6 Stage 1 finishes; run by hand after a family's arms are in.

For every ORIGINAL arm of the chosen family and seeds that diverged by that rule and has neither a retry directory nor a queued
retry row, it appends one GPU row that reruns the same command with --grad-clip 5.0 into <arm directory>__retry (the
instrumented harness, so the retry has its own probes). A retry is never itself retried: only original arms are examined, so an
arm that diverges again stays diverged. The aggregator (src/kc23_d6_aggregate.py) refuses to run the manipulation, outcome or
mechanism check while a diverged arm has no retry directory, and substitutes a completed retry for its arm (judged by the same
rule, flagged retried=True). After the retries finish, release a failed check with
`src/kc23_queue.py --retry d6_manipulation_check` (or its outcome / mechanism sibling).
Idempotent; an incomplete family or unreadable arm exits 2 and appends nothing.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import kc23_d6_aggregate as agg
from kc23_d6_stage2_job_gen import EO, FAMILY_SPECS, ROOT, STAGE1_DEPENDS, append_rows


def retry_rows(diverged: list[tuple], family: str) -> list[dict]:
    spec = FAMILY_SPECS[family]
    rows = []
    for knob, seed, orig_dir in diverged:
        out = orig_dir + agg.RETRY_SUFFIX
        rows.append({"job_id": spec["job_id"](knob, seed) + agg.RETRY_SUFFIX, "stage": "D6", "seed": str(seed),
                     "command": spec["command"](knob, seed, out, extra=f" --grad-clip {agg.GRAD_CLIP_RETRY:g}"),
                     "out_dir": out, "depends_on": ";".join(STAGE1_DEPENDS), "gate_script": "",
                     "expected_outputs": EO[family], "light": ""})
    return rows


def queued_ids(*csvs: Path) -> set[str]:
    ids = set()
    for c in csvs:
        if c.exists():
            with open(c, newline="", encoding="utf-8") as f:
                ids |= {r["job_id"] for r in csv.DictReader(f)}
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--families", default=",".join(agg.KNOWN_FAMILIES))
    ap.add_argument("--seeds", default=",".join(map(str, agg.OUTCOME_SEEDS)),
                    help="the realizations whose arms are all in (42 alone after Stage 1)")
    ap.add_argument("--gpu-csv", default=str(ROOT / "jobs/kc23_jobs_gpu.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root, seeds = Path(args.root), [int(x) for x in args.seeds.split(",") if x.strip()]
    already = queued_ids(Path(args.gpu_csv))
    new_rows = []
    try:
        for fam in [f.strip() for f in args.families.split(",") if f.strip()]:
            if fam not in FAMILY_SPECS:
                raise ValueError(f"unknown family {fam!r}")
            div = [d for d in agg.diverged_arms(root, fam, seeds) if not (root / (d[2] + agg.RETRY_SUFFIX)).exists()]
            rows = [r for r in retry_rows(div, fam) if r["job_id"] not in already]
            for r in rows:
                print(f"[d6-retry-gen] {fam}: {r['out_dir']} diverged; one retry with --grad-clip {agg.GRAD_CLIP_RETRY:g}")
            new_rows += rows
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[d6-retry-gen] cannot decide (an arm is missing or unreadable): {e}", file=sys.stderr)
        return 2
    if not new_rows:
        print("[d6-retry-gen] no diverged arm without a retry: nothing to append")
        return 0
    if args.dry_run:
        print(f"[d6-retry-gen] --dry-run: would append {len(new_rows)} row(s)")
        return 0
    append_rows(Path(args.gpu_csv), new_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
