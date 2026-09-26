"""The by-hand D6 generators: ADV-C rows (kc23_d6_advc_job_gen), ADV-CDAN rows conditional on C-M1 (kc23_d6_cdan_job_gen) and the
one divergence retry with --grad-clip 5.0 (kc23_d6_retry_job_gen)."""
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import kc23_d6_aggregate as agg
import kc23_d6_advc_job_gen as advc
import kc23_d6_cdan_job_gen as cdan
import kc23_d6_retry_job_gen as retry
from test_kc23_d6 import make_arm
from test_kc23_d6_mechanism import ADV_F1, KNOBS, SEEDS, build_adv

REPO = Path(__file__).resolve().parent.parent
FIELDS = ["job_id", "stage", "seed", "command", "out_dir", "depends_on", "gate_script", "expected_outputs", "light"]


def _csv(p: Path, ids=()):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i in ids:
            w.writerow({"job_id": i, "stage": "D6", "command": "x", "out_dir": "o", "depends_on": ""})


def _rows(p: Path):
    return {r["job_id"]: r for r in csv.DictReader(open(p, newline="", encoding="utf-8"))}


def _run(script, *argv):
    return subprocess.run([sys.executable, str(REPO / script), *argv], capture_output=True, text=True)


# ---------------------------------------------------------------- ADV-C
def test_advc_rows_at_the_collapse_and_the_next_value_up_with_the_oracle_flag(tmp_path):
    build_adv(tmp_path)
    gpu, cpu = tmp_path / "gpu.csv", tmp_path / "cpu.csv"
    _csv(gpu); _csv(cpu)
    r = _run("kc23_d6_advc_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu))
    assert r.returncode == 0 and "collapse at 1, next value up 3: 6 ADV-C rows" in r.stdout
    rows = _rows(gpu)
    assert sorted(rows) == sorted(f"d6_advc_l{k}_s{sd}" for k in (1, 3) for sd in SEEDS)
    for jid, x in rows.items():
        c = x["command"]
        assert "--adv-mode classcond --oracle-target-labels" in c and "--within-class-probe" in c and "--instrument" in c
        assert "--batch 256" in c and "--epochs 40" in c and x["expected_outputs"] == "adv_subjectwise.csv|40"
    assert rows["d6_advc_l1_s42"]["out_dir"] == "results_kc23_d6_advc_l1_s42"
    light = _rows(cpu)
    assert set(light) == {"d6_mechanism_check", "d6_secondary_check"}
    assert set(light["d6_mechanism_check"]["depends_on"].split(";")) == set(rows)
    assert light["d6_mechanism_check"]["light"] == "1" and light["d6_mechanism_check"]["expected_outputs"] == "*_VERDICT.md|LETTER"
    assert light["d6_mechanism_check"]["gate_script"] == "kc23_d6_stats.py"


def test_advc_no_collapse_appends_no_advc_row_and_says_so_but_still_queues_the_check(tmp_path):
    build_adv(tmp_path, f1s=[0.80, 0.82, 0.84, 0.85, 0.86, 0.86, 0.855])
    gpu, cpu = tmp_path / "gpu.csv", tmp_path / "cpu.csv"
    _csv(gpu); _csv(cpu)
    r = _run("kc23_d6_advc_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu))
    assert r.returncode == 0 and "NO COLLAPSE" in r.stdout and "ADV-C is not run" in r.stdout
    assert _rows(gpu) == {} and "d6_mechanism_check" in _rows(cpu)


def test_advc_generator_is_idempotent(tmp_path):
    build_adv(tmp_path)
    gpu, cpu = tmp_path / "gpu.csv", tmp_path / "cpu.csv"
    _csv(gpu); _csv(cpu)
    for _ in range(2):
        _run("kc23_d6_advc_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu))
    assert len(_rows(gpu)) == 6 and len(list(csv.DictReader(open(cpu, newline="", encoding="utf-8")))) == 2


def test_advc_generator_cannot_plan_from_an_incomplete_adv_and_appends_nothing(tmp_path):
    build_adv(tmp_path, seeds=[42])
    gpu, cpu = tmp_path / "gpu.csv", tmp_path / "cpu.csv"
    _csv(gpu); _csv(cpu)
    r = _run("kc23_d6_advc_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu))
    assert r.returncode == 2 and _rows(gpu) == {} and _rows(cpu) == {}


def test_advc_generator_refuses_while_an_adv_arm_diverged_without_a_retry(tmp_path):
    for sd in SEEDS:
        for i, k in enumerate(KNOBS):
            make_arm(tmp_path, "adv_marginal", k, sd, f1=ADV_F1[i], dom=0.5, subj=0.5, sil=0.1, cprobe=0.9, within=0.5,
                     loss=0.5 if i < 6 else 1.7)
    gpu, cpu = tmp_path / "gpu.csv", tmp_path / "cpu.csv"
    _csv(gpu); _csv(cpu)
    r = _run("kc23_d6_advc_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu))
    assert r.returncode == 2 and "retry" in r.stderr


# ---------------------------------------------------------------- ADV-CDAN, only after C-M1
def _mech_check(root: Path, line: str, meta=True, collapse=1, nxt=3):
    d = root / "results_kc23_d6_mechanism_check"
    d.mkdir(parents=True, exist_ok=True)
    (d / "D6_VERDICT.md").write_text(f"# KC-D6 verdict\n\n{line}\n", encoding="utf-8")
    if meta:
        pd.DataFrame([{"peak_knob": 0.1, "peak_f1": 0.86, "collapse_knob": collapse, "next_knob": nxt, "not_run": False,
                       "seeds": "42;7;123", "adv_f1_by_knob": ""}]).to_csv(d / "d6_mechanism_meta.csv", index=False)


def test_cdan_rows_only_after_c_m1_without_the_oracle_flag(tmp_path):
    _mech_check(tmp_path, "- **mechanism: C-M1**")
    gpu, cpu = tmp_path / "gpu.csv", tmp_path / "cpu.csv"
    _csv(gpu); _csv(cpu)
    r = _run("kc23_d6_cdan_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu))
    assert r.returncode == 0 and "6 ADV-CDAN rows at lambda_max [1, 3]" in r.stdout
    rows = _rows(gpu)
    assert sorted(rows) == sorted(f"d6_cdan_l{k}_s{sd}" for k in (1, 3) for sd in SEEDS)
    for x in rows.values():
        assert "--adv-mode cdan" in x["command"] and "--oracle-target-labels" not in x["command"]
        assert "--within-class-probe" in x["command"]
    chk = _rows(cpu)["d6_mechanism_cdan_check"]
    assert "results_kc23_d6_mechanism_cdan_check" in chk["command"] and chk["light"] == "1"
    assert set(chk["depends_on"].split(";")) == set(rows)


@pytest.mark.parametrize("line,phrase", [("- **mechanism: C-M2**", "landed C-M2, not C-M1"), ("- **mechanism: C-M3**", "landed C-M3, not C-M1"),
                                         ("- **mechanism: C-NOT-RUN**", "no collapse")])
def test_cdan_is_not_run_unless_advc_landed_c_m1(tmp_path, line, phrase):
    _mech_check(tmp_path, line)
    gpu, cpu = tmp_path / "gpu.csv", tmp_path / "cpu.csv"
    _csv(gpu); _csv(cpu)
    r = _run("kc23_d6_cdan_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu))
    assert r.returncode == 0 and phrase in r.stdout and _rows(gpu) == {} and _rows(cpu) == {}


def test_cdan_cannot_decide_without_a_mechanism_verdict_or_letter(tmp_path):
    gpu, cpu = tmp_path / "gpu.csv", tmp_path / "cpu.csv"
    _csv(gpu); _csv(cpu)
    assert _run("kc23_d6_cdan_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu)).returncode == 2
    d = tmp_path / "results_kc23_d6_mechanism_check"
    d.mkdir()
    (d / "D6_VERDICT.md").write_text("# KC-D6 verdict\n\nNO OUTCOME COMPUTED. x\n", encoding="utf-8")
    assert _run("kc23_d6_cdan_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu)).returncode == 2
    assert _rows(gpu) == {}


def test_cdan_generator_is_idempotent(tmp_path):
    _mech_check(tmp_path, "- **mechanism: C-M1**")
    gpu, cpu = tmp_path / "gpu.csv", tmp_path / "cpu.csv"
    _csv(gpu); _csv(cpu)
    for _ in range(2):
        _run("kc23_d6_cdan_job_gen.py", "--root", str(tmp_path), "--gpu-csv", str(gpu), "--cpu-csv", str(cpu))
    assert len(_rows(gpu)) == 6 and len(_rows(cpu)) == 1


# ---------------------------------------------------------------- the one divergence retry
def _diverging(tmp_path, seeds=(42,)):
    for sd in seeds:
        for i, k in enumerate(KNOBS):
            make_arm(tmp_path, "adv_marginal", k, sd, f1=ADV_F1[i], dom=0.5, subj=0.5, sil=0.1, cprobe=0.9,
                     loss=0.5 if i < 6 else 1.7)


def test_a_diverged_arm_gets_one_retry_row_with_grad_clip_5(tmp_path):
    _diverging(tmp_path)
    gpu = tmp_path / "gpu.csv"
    _csv(gpu)
    r = _run("kc23_d6_retry_job_gen.py", "--root", str(tmp_path), "--families", "adv_marginal", "--seeds", "42", "--gpu-csv", str(gpu))
    assert r.returncode == 0
    rows = _rows(gpu)
    assert list(rows) == ["d6_adv_marginal_l10_s42__retry"]
    x = rows["d6_adv_marginal_l10_s42__retry"]
    assert "--grad-clip 5" in x["command"] and "--out results_kc23_d6_adv_marginal_l10_s42__retry" in x["command"]
    assert "--adv-lambda 10 " in x["command"] and "--instrument results_kc23_d6_adv_marginal_l10_s42__retry/instr" in x["command"]
    assert x["out_dir"] == "results_kc23_d6_adv_marginal_l10_s42__retry" and x["expected_outputs"] == "adv_subjectwise.csv|40"


def test_the_retry_command_is_the_stage_command_plus_only_the_grad_clip(tmp_path):
    from kc23_d6_stage2_job_gen import FAMILY_SPECS
    spec = FAMILY_SPECS["adv_marginal"]
    plain = spec["command"](10, 42, "OUT")
    clipped = spec["command"](10, 42, "OUT", extra=" --grad-clip 5")
    assert clipped == plain.replace("--within-class-probe", "--within-class-probe --grad-clip 5")


def test_sfc_retry_uses_the_deep_coral_runners_grad_clip(tmp_path):
    from kc23_d6_stage2_job_gen import FAMILY_SPECS
    c = FAMILY_SPECS["sfc"]["command"](100, 7, "OUT", extra=" --grad-clip 5")
    assert "run_deep_coral_align_loso.py" in c and "--grad-clip 5" in c and "--coral-normalize l2" in c


def test_a_retry_is_never_retried_and_an_existing_retry_row_is_not_duplicated(tmp_path):
    _diverging(tmp_path)
    make_arm(tmp_path, "adv_marginal", 10, 42, f1=0.3, dom=0.5, subj=0.5, sil=0.1, cprobe=0.9, loss=1.9, retry=True)   # diverged AGAIN
    gpu = tmp_path / "gpu.csv"
    _csv(gpu)
    r = _run("kc23_d6_retry_job_gen.py", "--root", str(tmp_path), "--families", "adv_marginal", "--seeds", "42", "--gpu-csv", str(gpu))
    assert r.returncode == 0 and "nothing to append" in r.stdout and _rows(gpu) == {}


def test_the_retry_generator_is_idempotent(tmp_path):
    _diverging(tmp_path)
    gpu = tmp_path / "gpu.csv"
    _csv(gpu)
    for _ in range(2):
        _run("kc23_d6_retry_job_gen.py", "--root", str(tmp_path), "--families", "adv_marginal", "--seeds", "42", "--gpu-csv", str(gpu))
    assert len(_rows(gpu)) == 1


def test_nothing_is_appended_when_no_arm_diverged(tmp_path):
    for i, k in enumerate(KNOBS):
        make_arm(tmp_path, "adv_marginal", k, 42, f1=ADV_F1[i], dom=0.5, subj=0.5, sil=0.1, cprobe=0.9)
    gpu = tmp_path / "gpu.csv"
    _csv(gpu)
    r = _run("kc23_d6_retry_job_gen.py", "--root", str(tmp_path), "--families", "adv_marginal", "--seeds", "42", "--gpu-csv", str(gpu))
    assert r.returncode == 0 and _rows(gpu) == {}


def test_an_incomplete_family_cannot_be_judged_and_appends_nothing(tmp_path):
    _diverging(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / agg.FAMILY_DIR["adv_marginal"](0.3, 42))
    gpu = tmp_path / "gpu.csv"
    _csv(gpu)
    r = _run("kc23_d6_retry_job_gen.py", "--root", str(tmp_path), "--families", "adv_marginal", "--seeds", "42", "--gpu-csv", str(gpu))
    assert r.returncode == 2 and _rows(gpu) == {}


def test_after_the_retry_completes_the_aggregate_runs(tmp_path):
    _diverging(tmp_path)
    for fam in ("sfc", "advps"):
        for k in agg.FAMILY_KNOBS[fam]:
            make_arm(tmp_path, fam, k, 42, f1=0.8, dom=0.6, subj=0.5, sil=0.1, cprobe=0.9)
    assert agg.run(tmp_path / "o", tmp_path, "manipulation") == 1
    make_arm(tmp_path, "adv_marginal", 10, 42, f1=0.7, dom=0.5, subj=0.5, sil=0.1, cprobe=0.9, loss=0.55, retry=True)
    assert agg.run(tmp_path / "o", tmp_path, "manipulation") == 0
    div = pd.read_csv(tmp_path / "o" / "d6_arm_divergence.csv")
    assert bool(div[div["knob"] == 10]["retried"].iloc[0]) and not bool(div[div["knob"] == 10]["diverged"].iloc[0])
