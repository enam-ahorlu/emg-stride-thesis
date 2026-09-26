"""Fail-open sweep (D-6c follow-up, 25 September 2026).

The same defect kept appearing in different stages: a stats, gate, aggregate
or analysis script that, with an input missing, wrote a verdict and exited 0
(S1's never-wired gate, D5's two unproduced CSVs, S2's placeholder verdict).
This file runs EVERY such script from a sandbox in which none of its inputs
exist and requires:

  1. a non-zero exit code;
  2. no file matching *VERDICT*.md that contains a '**...: LETTER**' outcome
     line (the same pattern kc23_queue.check_expected_outputs uses for LETTER);
  3. that the failure is the script's own decision, not an ImportError or a
     missing module in the sandbox (which would pass for the wrong reason).

The script under test is COPIED into an empty sandbox and run there, so
anything it locates relative to its own file (ROOT = Path(__file__).parent)
or to the working directory resolves to nothing. Helper modules still import
from the repository through PYTHONPATH.

Below it, a second group removes one required input at a time from a
synthetic complete fixture, for the scripts whose inputs are plain files.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")

# script -> argv (after the script path), {out} is the output dir under the sandbox
SCRIPTS = {
    "kc23_c1_nested_selection.py": [],
    "kc23_c2_whitening_stats.py": ["--out", "{out}"],
    "kc23_c3_merge_proba.py": ["--out", "{out}"],
    "kc23_c3_tuning_stats.py": ["--out", "{out}"],
    "kc23_c4_extract_rich.py": ["--out", "{out}"],
    "kc23_c4_feature_stats.py": ["--out", "{out}"],
    "kc23_c5_leak_stats.py": ["--out", "{out}"],
    "kc23_c6_ladder_stats.py": ["--out", "{out}"],
    "kc23_d1_aggregate.py": ["--root", ".", "--out", "{out}"],
    "kc23_d1_replicate_stats.py": ["--out", "{out}"],
    "kc23_d2_reliance_stats.py": ["--out", "{out}"],
    "kc23_d3_axis_stats.py": ["--out", "{out}"],
    "kc23_d4_invariance_stats.py": ["--out", "{out}"],
    "kc23_d5_aggregate.py": ["--root", ".", "--out", "{out}"],
    "kc23_d5_replication_stats.py": ["--out", "{out}"],
    "kc23_d6_aggregate.py": ["--root", ".", "--out", "{out}"],
    "kc23_d6_stats.py": ["--out", "{out}"],
    "kc23_s1_scripted_stats.py": ["--out", "{out}"],
    "kc23_s2_f0_feasibility.py": ["--out", "{out}", "--root", "nowhere"],
    "kc23_s2_transition_table.py": ["--root", "nowhere", "--out", "{out}"],
    "kc23_s2_transitions.py": ["--out", "{out}", "--preds-dir", "nowhere", "--table", "nowhere/t.csv"],
    "kc23_s2_predictions.py": ["--root", ".", "--out", "{out}"],
    "kc23_s3_inventory.py": ["--out", "{out}"],
}
# d6_stats dispatches on the out_dir NAME; run it under both names it serves
OUT_NAMES = {"kc23_d6_stats.py": ["results_kc23_d6_sanity_check", "results_kc23_d6_manipulation_check"]}


def _cases():
    for script in SCRIPTS:
        for name in OUT_NAMES.get(script, ["out"]):
            yield pytest.param(script, name, id=f"{script}:{name}")


@pytest.mark.parametrize("script,out_name", list(_cases()))
def test_script_with_every_input_missing_fails_closed(tmp_path, script, out_name):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    shutil.copy(REPO / script, sandbox / script)
    out = sandbox / out_name
    argv = [a.replace("{out}", out_name) for a in SCRIPTS[script]]
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, str(sandbox / script), *argv], cwd=sandbox, env=env,
                       capture_output=True, text=True, timeout=300)
    combined = (r.stdout or "") + (r.stderr or "")
    assert "ModuleNotFoundError" not in combined and "ImportError" not in combined, (
        f"{script} failed on an import in the sandbox, which proves nothing:\n{combined[-1500:]}")
    assert r.returncode != 0, f"{script} exited 0 with every input missing (FAIL-OPEN):\n{combined[-1500:]}"
    for v in sandbox.rglob("*VERDICT*.md"):
        assert not LETTER_RE.search(v.read_text(encoding="utf-8", errors="replace")), (
            f"{script} left a verdict containing an outcome letter with no inputs: {v}")
