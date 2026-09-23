#!/usr/bin/env python3
"""Synthetic gate for the kc23_queue.py fail-closed-on-crash test (found live
2026-09-24: kc23_s2_f0_feasibility.py crashed with an uncaught exception,
rc=1, and the queue treated it as a clean pass). Always raises, regardless of
--out, so its exit code is Python's default 1 -- not 0, 10 or 20. Reusable
fixture for future queue tests, like _synthetic_escalate_gate.py."""
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    raise RuntimeError(f"[synthetic-crash-gate] deliberate uncaught exception (--out={args.out})")
