#!/usr/bin/env python3
"""Synthetic gate for the src/kc23_queue.py halt-protocol empirical test (item 6
of the 24 September continuation). Always exits 20 (ESCALATE), regardless of
--out. Temporary: used once by the halt-protocol test, then the test files
that reference it are removed; this file itself stays in tests/kc23/ as a
reusable fixture for future queue tests."""
import argparse
import sys

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(f"[synthetic-gate] always ESCALATE (--out={args.out})")
    sys.exit(20)
