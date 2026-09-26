"""Synthetic tests for kc23_c6_ladder_stats.py: R1/R2."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kc23_c6_ladder_stats import classify_r

RNG = np.random.default_rng(4)
N = 10


def test_r1_replicates():
    f1_rung3 = np.full(N, 0.75) + RNG.normal(0, 0.01, N)
    f1_4lw = f1_rung3 - 0.05  # rung3 beats 4lw for all 10
    letter, d = classify_r(f1_rung3, f1_4lw, probe_rung0=0.9, probe_rung1=0.6, probe_rung3=0.3)
    assert letter == "R1", d


def test_r2_few_wins():
    f1_rung3 = np.array([0.7] * 3 + [0.6] * 7)
    f1_4lw = np.array([0.65] * 3 + [0.65] * 7)  # rung3 beats 4lw only 3/10
    letter, d = classify_r(f1_rung3, f1_4lw, probe_rung0=0.9, probe_rung1=0.6, probe_rung3=0.3)
    assert letter == "R2", d


def test_r2_centering_not_dominant():
    f1_rung3 = np.full(N, 0.75) + RNG.normal(0, 0.01, N)
    f1_4lw = f1_rung3 - 0.05  # beats 4lw 10/10
    # centering only removes a small fraction of the total reduction
    letter, d = classify_r(f1_rung3, f1_4lw, probe_rung0=0.9, probe_rung1=0.85, probe_rung3=0.3)
    assert letter == "R2", d


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


# ---------------------------------------------------------------------------------------------------------------
# 26 September 2026: the geometry producer, and the stats reading the real file (fail closed).
import os
import re
import subprocess
import pandas as pd
import pytest
import kc23_c6_ladder_stats as c6
from kc23_c6_ladder_stats import run as c6_run

LETTER_RE = re.compile(r"\*\*[^*\n]*:\s*[A-Za-z0-9][^*\n]*\*\*")
REPO = Path(__file__).resolve().parent.parent


def _write_ladder(d: Path, probes=None, f1_gap=0.05, rungs=("0", "1", "2", "3", "4", "4lw", "4o")):
    d.mkdir(parents=True, exist_ok=True)
    probes = probes or {"0": 0.90, "1": 0.55, "2": 0.6, "3": 0.30, "4": 0.2, "4lw": 0.25, "4o": 0.3}
    pd.DataFrame([{"rung": r, "name": f"n{r}", "subject_probe_linear": probes[r], "subject_probe_forest": 0.5,
                   "subject_probe_mlp": 0.5} for r in rungs]).to_csv(d / "ladder_geometry.csv", index=False)
    base = np.linspace(0.6, 0.7, N)
    for r, v in (("3", base), ("4lw", base - f1_gap)):
        pd.DataFrame({"subject": range(1, N + 1), "f1_macro": v}).to_csv(d / f"ladder_loso_{r}_SVM_subjectwise.csv", index=False)


def test_c6_stats_end_to_end_r1(tmp_path):
    _write_ladder(tmp_path)
    assert c6_run(tmp_path) == 0
    v = (tmp_path / "C6_VERDICT.md").read_text()
    assert "**Outcome: R1**" in v and "10 of 10" in v


def test_c6_centering_fraction_exactly_half_is_r1(tmp_path):
    _write_ladder(tmp_path, probes={"0": 0.90, "1": 0.60, "2": 0.6, "3": 0.30, "4": 0.2, "4lw": 0.25, "4o": 0.3})
    assert c6_run(tmp_path) == 0 and "**Outcome: R1**" in (tmp_path / "C6_VERDICT.md").read_text()


def test_c6_r2_when_centering_does_not_do_most(tmp_path):
    _write_ladder(tmp_path, probes={"0": 0.90, "1": 0.80, "2": 0.6, "3": 0.30, "4": 0.2, "4lw": 0.25, "4o": 0.3})
    assert c6_run(tmp_path) == 0 and "**Outcome: R2**" in (tmp_path / "C6_VERDICT.md").read_text()


def test_c6_rungs_are_matched_by_id_not_by_row_order(tmp_path):
    _write_ladder(tmp_path, rungs=("4o", "4lw", "4", "3", "2", "1", "0"))     # reversed
    assert c6_run(tmp_path) == 0 and "**Outcome: R1**" in (tmp_path / "C6_VERDICT.md").read_text()


@pytest.mark.parametrize("victim", ["ladder_geometry.csv", "ladder_loso_3_SVM_subjectwise.csv", "ladder_loso_4lw_SVM_subjectwise.csv"])
def test_c6_each_input_missing_fails_closed_with_no_letter(tmp_path, victim):
    _write_ladder(tmp_path)
    (tmp_path / victim).unlink()
    assert c6_run(tmp_path) == 20 and not LETTER_RE.search((tmp_path / "C6_VERDICT.md").read_text())


@pytest.mark.parametrize("missing", ["0", "1", "3"])
def test_c6_a_missing_probe_rung_fails_closed(tmp_path, missing):
    rungs = tuple(r for r in ("0", "1", "2", "3", "4", "4lw", "4o") if r != missing)
    _write_ladder(tmp_path, rungs=rungs)
    assert c6_run(tmp_path) == 20


def test_c6_non_finite_probe_fails_closed(tmp_path):
    _write_ladder(tmp_path, probes={"0": 0.9, "1": float("nan"), "2": 0.6, "3": 0.3, "4": 0.2, "4lw": 0.25, "4o": 0.3})
    assert c6_run(tmp_path) == 20


def test_c6_incomplete_f1_file_fails_closed(tmp_path):
    _write_ladder(tmp_path)
    p = tmp_path / "ladder_loso_3_SVM_subjectwise.csv"
    pd.read_csv(p).iloc[:7].to_csv(p, index=False)
    assert c6_run(tmp_path) == 20


def test_c6_uses_the_linear_pooled_probe_not_another_column(tmp_path):
    _write_ladder(tmp_path)
    g = pd.read_csv(tmp_path / "ladder_geometry.csv").drop(columns=["subject_probe_linear"])
    g.to_csv(tmp_path / "ladder_geometry.csv", index=False)
    assert c6_run(tmp_path) == 20


# --- kc23_c6_geometry.py on a tiny synthetic ENABL3S-like set (10 subjects, 72 features), in a subprocess because the
# published modules read their feature paths from the environment at import time.
def _tiny_enabl3s(tmp: Path):
    rng = np.random.default_rng(0)
    rows, X = [], []
    for s in range(1, 11):
        off = rng.normal(0, 1.0, 72)
        for ci, lab in enumerate(["DNS", "STDUP", "UPS", "WAK"]):
            for _ in range(40):
                rows.append({"subject": s, "movement": lab})
                X.append(off + ci * 0.4 + rng.normal(0, 1.0, 72))
    np.savez(tmp / "feat.npz", X=np.array(X))
    pd.DataFrame(rows).to_csv(tmp / "meta.csv", index=False)


def _geometry(tmp: Path, *extra):
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1")
    return subprocess.run([sys.executable, str(REPO / "kc23_c6_geometry.py"), "--out", str(tmp / "out"),
                           "--feat", str(tmp / "feat.npz"), "--meta", str(tmp / "meta.csv"), *extra],
                          cwd=tmp, env=env, capture_output=True, text=True, timeout=1500)


def test_c6_geometry_end_to_end_and_feeds_the_stats(tmp_path):
    _tiny_enabl3s(tmp_path)
    r = _geometry(tmp_path, "--rungs", "0,3", "--cap", "12", "--cap-mmd", "20", "--n-draws", "1")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    g = pd.read_csv(tmp_path / "out" / "ladder_geometry.csv")
    assert list(g["rung"].astype(str)) == ["0", "3"]
    for col in ("mmd_mean", "wasserstein1_mean", "mmd_removed_pct", "w1_removed_pct", "subject_probe_linear",
                "subject_probe_forest", "subject_probe_mlp", "wm_probe_linear", "sm_probe_linear", "wm_probe_forest",
                "sm_probe_mlp", "silhouette_by_class", "silhouette_within_subject", "chance_floor", "oracle"):
        assert col in g.columns and g[col].notna().all(), col
    assert g["chance_floor"].iloc[0] == pytest.approx(0.1)
    assert g.loc[g["rung"].astype(str) == "0", "mmd_removed_pct"].iloc[0] == pytest.approx(0.0)
    p = g.set_index(g["rung"].astype(str))["subject_probe_linear"]
    assert p["3"] < p["0"]                                   # per-subject z-scoring removes linearly decodable subject identity


def test_c6_geometry_missing_input_or_unknown_rung_fails_and_writes_nothing(tmp_path):
    (tmp_path / "meta.csv").write_text("subject,movement\n1,WAK\n")
    r = _geometry(tmp_path, "--rungs", "0,3")
    assert r.returncode == 1 and not (tmp_path / "out" / "ladder_geometry.csv").exists()
    _tiny_enabl3s(tmp_path)
    r2 = _geometry(tmp_path, "--rungs", "0,9")
    assert r2.returncode == 1 and not (tmp_path / "out" / "ladder_geometry.csv").exists()


def test_c6_geometry_without_rung_zero_fails_closed(tmp_path):
    _tiny_enabl3s(tmp_path)
    r = _geometry(tmp_path, "--rungs", "3", "--cap", "12", "--cap-mmd", "20", "--n-draws", "1")
    assert r.returncode == 1 and not (tmp_path / "out" / "ladder_geometry.csv").exists()
