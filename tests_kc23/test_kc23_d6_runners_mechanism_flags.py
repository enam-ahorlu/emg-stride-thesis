"""KC-D6 runner flags added 26 September 2026 for the mechanism test and the divergence retry:
--within-class-probe (run_adv_align_loso.py, needs --instrument) and --grad-clip (run_deep_coral_align_loso.py). Both are
additive and inert when absent. The base for 'unchanged' is commit d613a06 (before any KC-D6 runner change)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kc23_d6_runners_instrument import (REPO, _tiny_data, _run, _head_copy, _same_files)


def test_within_class_probe_adds_one_column_and_leaves_everything_else_unchanged(tmp_path):
    _tiny_data(tmp_path)
    args = ["--adv-lambda", "1.0", "--adv-mode", "marginal"]
    a = _run(REPO / "run_adv_align_loso.py", tmp_path, "plain", args + ["--instrument", str(tmp_path / "plain" / "instr")])
    b = _run(REPO / "run_adv_align_loso.py", tmp_path, "wc", args + ["--instrument", str(tmp_path / "wc" / "instr"),
                                                                       "--within-class-probe"])
    _same_files(a, b, ["adv_subjectwise.csv", "alignment_subjectwise.csv", "training_log.csv"])
    for f in ("occlusion.csv", "attenuation.csv", "permutation.csv"):
        assert (a / "instr" / f).read_bytes() == (b / "instr" / f).read_bytes(), f
    pa, pb = pd.read_csv(a / "instr" / "embed_probes.csv"), pd.read_csv(b / "instr" / "embed_probes.csv")
    assert "subject_probe_within_class_bacc" not in pa.columns
    extra = {"subject_probe_within_class_bacc", "n_within_class_probes"}
    assert extra <= set(pb.columns)
    pd.testing.assert_frame_equal(pa, pb.drop(columns=list(extra)))
    assert pb["subject_probe_within_class_bacc"].between(0, 1).all() and (pb["n_within_class_probes"] >= 1).all()


def test_within_class_probe_without_instrument_is_refused(tmp_path):
    import subprocess, os
    _tiny_data(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([sys.executable, str(REPO / "run_adv_align_loso.py"), "--npz", str(tmp_path / "w.npz"), "--meta",
                        str(tmp_path / "w_meta.csv"), "--within-class-probe", "--out", str(tmp_path / "o")],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "needs --instrument" in (r.stderr + r.stdout)


def test_deep_coral_grad_clip_is_recorded_in_the_run_config_and_default_is_unchanged_from_the_base(tmp_path):
    _tiny_data(tmp_path)
    old = _head_copy("run_deep_coral_align_loso.py", tmp_path)
    a = _run(old, tmp_path, "old", ["--coral-lambda", "1.0"])
    b = _run(REPO / "run_deep_coral_align_loso.py", tmp_path, "new", ["--coral-lambda", "1.0"])
    _same_files(a, b, ["deep_coral_subjectwise.csv", "alignment_subjectwise.csv", "training_log.csv"])
    c = _run(REPO / "run_deep_coral_align_loso.py", tmp_path, "clip", ["--coral-lambda", "1.0", "--grad-clip", "5.0"])
    import json
    cfg = json.loads((c / "run_config.json").read_text())
    assert cfg["args"]["grad_clip"] == 5.0
    assert (c / "deep_coral_subjectwise.csv").exists()
