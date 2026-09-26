"""Per-input fail-closed tests (fail-open sweep, level 2, 25 September 2026):
remove ONE required input at a time from a complete synthetic fixture and
require a non-zero exit and a verdict with no outcome line. Covers the scripts
whose inputs are plain files and that had no such test: the S3 inventory.
(S1, D1, D5, D6, S2, C3, C4, C5 and the S3 benchmark carry their own in their test files.)
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_s3_inventory as s3

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")


def test_s3_inventory_missing_manifest_exits_nonzero_and_writes_nothing(tmp_path, monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("run manifest missing")
    monkeypatch.setattr(s3, "build_inventory", missing)
    out = tmp_path / "out"
    assert s3.run(out) == 1
    assert not (out / "active_only_inventory.csv").exists()


def test_s3_build_inventory_raises_on_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        s3.build_inventory(tmp_path / "nope.csv")


def test_s3_build_inventory_raises_on_manifest_without_dir_column(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame({"x": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        s3.build_inventory(p)
