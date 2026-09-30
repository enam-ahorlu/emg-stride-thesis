"""Synthetic-classical-output tests for src/kc23_c5_cnn_job_gen.py (C5.3's
SimpleEMGCNN B/I-at-the-plateau-guard rows): read_g_star() reads a synthetic
C5_decomposition.csv correctly, build_rows()/append_rows() generate exactly
the right rows, idempotently, and g_star=None (L3) appends nothing."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))
from kc23_c5_cnn_job_gen import read_g_star, build_rows, append_rows, main
import kc23_c5_cnn_job_gen as mod


def _decomposition_csv(tmp_path, g_star):
    p = tmp_path / "C5_decomposition.csv"
    pd.DataFrame([{"g_star": g_star, "total_pp": 5.0, "overlap_pp": 3.0,
                  "autocorr_pp": 1.0, "drift_pp": 1.0}]).to_csv(p, index=False)
    return p


def test_read_g_star_normal_value(tmp_path):
    p = _decomposition_csv(tmp_path, 4)
    assert read_g_star(p) == 4


def test_read_g_star_none_when_no_plateau(tmp_path):
    p = tmp_path / "C5_decomposition.csv"
    pd.DataFrame([{"g_star": None, "note": "no matching I-g or B-g for g_star"}]).to_csv(p, index=False)
    assert read_g_star(p) is None


def test_build_rows_correct_job_ids_and_flags():
    rows = build_rows(4)
    ids = {r["job_id"] for r in rows}
    assert ids == {"c5_simplecnn_sd_b4", "c5_simplecnn_sd_i4"}
    b_row = [r for r in rows if r["job_id"] == "c5_simplecnn_sd_b4"][0]
    i_row = [r for r in rows if r["job_id"] == "c5_simplecnn_sd_i4"][0]
    assert "--scheme blocked" in b_row["command"] and "--guard-windows 4" in b_row["command"]
    assert "--scheme interleaved" in i_row["command"] and "--n-chunks 20" in i_row["command"]
    assert b_row["out_dir"] != i_row["out_dir"]  # no is_complete() collision


def test_append_rows_idempotent(tmp_path):
    gpu_csv = tmp_path / "jobs/kc23_jobs_gpu.csv"
    gpu_csv.parent.mkdir(parents=True, exist_ok=True)
    gpu_csv.write_text("job_id,stage,seed,command,out_dir,depends_on,gate_script,expected_outputs\n"
                      "existing_job,X,,cmd,out,,,\n", encoding="utf-8")
    rows = build_rows(2)
    n1 = append_rows(gpu_csv, rows)
    assert n1 == 2
    n2 = append_rows(gpu_csv, rows)
    assert n2 == 0
    df = pd.read_csv(gpu_csv)
    assert len(df) == 3
    assert df["job_id"].duplicated().sum() == 0


def test_main_no_plateau_appends_nothing(tmp_path, monkeypatch, capsys):
    dcsv = tmp_path / "C5_decomposition.csv"
    pd.DataFrame([{"g_star": None}]).to_csv(dcsv, index=False)
    gpu_csv = tmp_path / "jobs/kc23_jobs_gpu.csv"
    gpu_csv.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["src/kc23_c5_cnn_job_gen.py", "--decomposition-csv", str(dcsv),
                                      "--gpu-csv", str(gpu_csv)])
    rc = main()
    assert rc == 0
    assert not gpu_csv.exists()


def test_main_with_plateau_appends_two_rows(tmp_path, monkeypatch):
    dcsv = _decomposition_csv(tmp_path, 8)
    gpu_csv = tmp_path / "jobs/kc23_jobs_gpu.csv"
    gpu_csv.parent.mkdir(parents=True, exist_ok=True)
    gpu_csv.write_text("job_id,stage,seed,command,out_dir,depends_on,gate_script,expected_outputs\n",
                      encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["src/kc23_c5_cnn_job_gen.py", "--decomposition-csv", str(dcsv),
                                      "--gpu-csv", str(gpu_csv)])
    rc = main()
    assert rc == 0
    df = pd.read_csv(gpu_csv)
    assert set(df["job_id"]) == {"c5_simplecnn_sd_b8", "c5_simplecnn_sd_i8"}
