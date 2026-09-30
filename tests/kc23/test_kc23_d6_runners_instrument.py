"""KC-D6 runners: the --instrument flag added on 26 September 2026 (src/run_adv_align_loso.py and
src/run_deep_coral_align_loso.py) is additive and inert when absent, and writes the D0.3 files (occlusion, attenuation,
permutation, embed_probes) when present.

Inertness is asserted the way the house rule asks: the pre-change script (git HEAD, taken before the edit was
committed) and the new script run on the same tiny synthetic fold on CPU and every output file must match byte for
byte. src/run_deep_coral_align_loso.py is the script the D1 R11 rows use, so this is what protects those runs. Skipped
(not passed) if git or the HEAD copy is unavailable."""
import filecmp
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
LABELS = ["DNS", "STDUP", "UPS", "WAK"]
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="-1", PYTHONHASHSEED="0", OMP_NUM_THREADS="1")


def _tiny_data(tmp: Path):
    rng = np.random.default_rng(0)
    subs = list(range(1, 9))
    rows, X = [], []
    for s in subs:
        for ci, lab in enumerate(LABELS):
            for _ in range(14):
                rows.append({"subject": s, "movement": lab})
                X.append(rng.normal(ci * 0.3, 1.0, (9, 64)).astype(np.float32))
    np.savez(tmp / "w.npz", X_env=np.stack(X), X_raw=np.stack(X))
    pd.DataFrame(rows).to_csv(tmp / "w_meta.csv", index=False)


def _run(script: Path, tmp: Path, out: str, extra: list[str]):
    cmd = [PY, str(script), "--npz", str(tmp / "w.npz"), "--meta", str(tmp / "w_meta.csv"), "--arch", "resnet_se",
           "--epochs", "1", "--batch", "16", "--seed", "42", "--heldout", "1", "--out", str(tmp / out), *extra]
    r = subprocess.run(cmd, cwd=tmp, env=ENV, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return tmp / out


# The pre-change scripts are the ones in commit d613a06 (the last commit before --instrument and the D6 normalisation
# check were added). A fixed commit, not HEAD, so the proof does not turn into a skip once the change is committed.
BASE_COMMIT = "d613a06"


def _head_copy(name: str, tmp: Path) -> Path:
    r = subprocess.run(["git", "show", f"{BASE_COMMIT}:{name}"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run(["git", "show", f"{BASE_COMMIT}:06_Code/{name}"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout:
        pytest.skip("git HEAD copy of the pre-change script is unavailable")
    p = tmp / f"old_{name}"
    p.write_text(r.stdout, encoding="utf-8")
    return p


def _same_files(a: Path, b: Path, names):
    for n in names:
        assert (a / n).exists() and (b / n).exists(), n
        assert (a / n).read_bytes() == (b / n).read_bytes(), f"{n} differs"


def test_deep_coral_runner_is_byte_identical_without_the_flag(tmp_path):
    _tiny_data(tmp_path)
    old = _head_copy("src/run_deep_coral_align_loso.py", tmp_path)
    if "--instrument" in old.read_text():
        pytest.skip("HEAD already contains the flag; nothing to compare against")
    a = _run(old, tmp_path, "old_out", ["--coral-lambda", "1.0"])
    b = _run(REPO / "src/run_deep_coral_align_loso.py", tmp_path, "new_out", ["--coral-lambda", "1.0"])
    _same_files(a, b, ["deep_coral_subjectwise.csv", "alignment_subjectwise.csv", "training_log.csv",
                       "alignment_summary.csv"])
    assert not (b / "instr").exists()


def test_deep_coral_runner_with_the_flag_writes_probes_and_leaves_the_main_outputs_unchanged(tmp_path):
    _tiny_data(tmp_path)
    a = _run(REPO / "src/run_deep_coral_align_loso.py", tmp_path, "plain", ["--coral-lambda", "1.0"])
    b = _run(REPO / "src/run_deep_coral_align_loso.py", tmp_path, "instr_out",
             ["--coral-lambda", "1.0", "--instrument", str(tmp_path / "instr_out" / "instr")])
    _same_files(a, b, ["deep_coral_subjectwise.csv", "alignment_subjectwise.csv", "training_log.csv"])
    for f in ("occlusion.csv", "attenuation.csv", "permutation.csv", "embed_probes.csv"):
        assert (b / "instr" / f).exists(), f
    ep = pd.read_csv(b / "instr" / "embed_probes.csv")
    assert list(ep["subject"]) == [1] and ep[["subject_probe_bacc", "held_out_class_silhouette",
                                              "held_out_class_probe_bacc"]].notna().all().all()


def test_adv_runner_with_the_flag_writes_probes_and_leaves_the_main_outputs_unchanged(tmp_path):
    _tiny_data(tmp_path)
    args = ["--adv-lambda", "1.0", "--adv-mode", "marginal"]
    a = _run(REPO / "src/run_adv_align_loso.py", tmp_path, "plain", args)
    b = _run(REPO / "src/run_adv_align_loso.py", tmp_path, "instr_out", args + ["--instrument", str(tmp_path / "instr_out" / "instr")])
    _same_files(a, b, ["adv_subjectwise.csv", "alignment_subjectwise.csv", "training_log.csv"])
    for f in ("occlusion.csv", "attenuation.csv", "permutation.csv", "embed_probes.csv"):
        assert (b / "instr" / f).exists(), f
    assert not (a / "instr").exists()


def test_the_l2_path_gains_only_the_normalisation_check_columns(tmp_path):
    """KC-D6 ruling of 26 Sept: the run now CHECKS that the embedding the CORAL term sees is unit norm. The check is a
    measurement (no RNG draw, no effect on the loss): F1, every alignment measure and every logged loss are unchanged, and
    the only difference is the new column."""
    _tiny_data(tmp_path)
    old = _head_copy("src/run_deep_coral_align_loso.py", tmp_path)
    args = ["--coral-lambda", "10", "--coral-normalize", "l2"]
    a = _run(old, tmp_path, "old_l2", args)
    b = _run(REPO / "src/run_deep_coral_align_loso.py", tmp_path, "new_l2", args)
    _same_files(a, b, ["deep_coral_subjectwise.csv"])
    for name, extra_col in (("alignment_subjectwise.csv", "coral_embed_normdev_max"), ("training_log.csv", "coral_embed_normdev")):
        da, db = pd.read_csv(a / name), pd.read_csv(b / name)
        assert extra_col in db.columns and extra_col not in da.columns
        pd.testing.assert_frame_equal(da, db.drop(columns=[extra_col]))
    assert (pd.read_csv(b / "alignment_subjectwise.csv")["coral_embed_normdev_max"] < 1e-5).all()


def test_the_default_path_writes_no_normalisation_column(tmp_path):
    _tiny_data(tmp_path)
    b = _run(REPO / "src/run_deep_coral_align_loso.py", tmp_path, "plain", ["--coral-lambda", "1.0"])
    assert "coral_embed_normdev_max" not in pd.read_csv(b / "alignment_subjectwise.csv").columns
    assert "coral_embed_normdev" not in pd.read_csv(b / "training_log.csv").columns
