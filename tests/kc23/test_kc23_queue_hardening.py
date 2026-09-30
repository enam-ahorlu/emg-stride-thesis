"""The runner exited with 0xC000013A (STATUS_CONTROL_C_EXIT: a console window closed, or Ctrl+C) and its running job died with it.
Hardening of 26 September 2026: jobs start in their own hidden console and process group, the task runs the runner under
pythonw.exe (no console) with --redirect-output. These tests close a real console window and check who survives."""
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import kc23_queue as q

HARNESS = Path(__file__).resolve().parent / "hardening_harness.py"
WIN = os.name == "nt"
pytestmark = pytest.mark.skipif(not WIN, reason="console-window behaviour is Windows-specific")
WM_CLOSE = 0x0010


def _wait_state(path: Path, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists() and path.read_text().strip().endswith("}"):
            return json.loads(path.read_text())
        time.sleep(0.3)
    raise AssertionError("harness did not start")


def _descendants(pid):
    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return []


def _job_python(shell_pid):
    """The long-sleeping python process the job's cmd.exe started."""
    for _ in range(40):
        for c in _descendants(shell_pid):
            try:
                if "time.sleep" in " ".join(c.cmdline()):
                    return c
            except psutil.Error:
                pass
        time.sleep(0.25)
    raise AssertionError("job process not found")


def _alive(p: psutil.Process) -> bool:
    try:
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _kill_tree(*procs):
    for p in procs:
        try:
            for c in p.children(recursive=True):
                c.kill()
            p.kill()
        except psutil.Error:
            pass


def _run_in_console(tmp_path, extra):
    """Run the harness in a classic conhost console window (conhost.exe python harness), so closing it is a real console close."""
    state = tmp_path / "state.json"
    cmd = ["conhost.exe", sys.executable, str(HARNESS), "--state", str(state), "--workdir", str(tmp_path), *extra]
    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=tmp_path)
    return _wait_state(state)


def _close_window(hwnd: int):
    ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)


def _wait_dead(p: psutil.Process, seconds=20) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if not _alive(p):
            return True
        time.sleep(0.5)
    return False


def test_control_closing_the_console_kills_a_job_that_shares_it(tmp_path):
    """The old behaviour, reproduced: the job started in the runner's console and process group dies with the window."""
    st = _run_in_console(tmp_path, ["--old-flags"])
    runner = psutil.Process(st["runner_pid"])
    job = _job_python(st["shell_pid"])
    try:
        assert st["console_hwnd"] and _alive(runner) and _alive(job)
        _close_window(st["console_hwnd"])
        assert _wait_dead(runner) and _wait_dead(job), "the control should have lost both processes"
    finally:
        _kill_tree(runner, job)


def test_a_job_started_by_the_hardened_launcher_survives_its_runners_console_being_closed(tmp_path):
    st = _run_in_console(tmp_path, [])
    runner = psutil.Process(st["runner_pid"])
    job = _job_python(st["shell_pid"])
    try:
        assert st["console_hwnd"] and _alive(runner) and _alive(job)
        _close_window(st["console_hwnd"])
        assert _wait_dead(runner), "a console-attached runner still ends when ITS console is closed"
        time.sleep(3)
        assert _alive(job), "the job must not share that console"
    finally:
        _kill_tree(job)
        _kill_tree(runner)


def test_a_console_less_runner_and_its_job_survive_closing_any_terminal(tmp_path):
    """The scheduled-task configuration: pythonw.exe, --redirect-output. A terminal tailing the runner's log is closed."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    assert pythonw.exists()
    state = tmp_path / "state.json"
    subprocess.Popen([str(pythonw), str(HARNESS), "--state", str(state), "--workdir", str(tmp_path), "--daemon"], cwd=tmp_path,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    st = _wait_state(state)
    runner = psutil.Process(st["runner_pid"])
    job = _job_python(st["shell_pid"])
    tail = tmp_path / "tail.log"
    tail.write_text("x")
    subprocess.Popen(["conhost.exe", "cmd.exe", "/c", f"type \"{tail}\" && timeout /t 60 > nul"],
                     creationflags=subprocess.CREATE_NEW_CONSOLE)
    try:
        assert st["console_hwnd"] == 0, "the runner must have no console"
        assert st["stdout_is_none"] is False and (tmp_path / "queue_stdout.log").read_text().strip().endswith("alive")
        time.sleep(2)
        for p in psutil.process_iter(["name", "cmdline"]):                       # close the terminal(s) tailing the log
            if p.info["name"] == "conhost.exe" and "timeout /t 60" in " ".join(p.info["cmdline"] or []):
                p.kill()
        time.sleep(3)
        assert _alive(runner) and _alive(job)
    finally:
        _kill_tree(job)
        _kill_tree(runner)


def test_the_redirected_runner_ignores_ctrl_c_and_ctrl_break(tmp_path):
    code = ("import sys, signal, time\n"
            f"sys.path.insert(0, r'{REPO / 'src'}')\n"
            "import kc23_queue as q\nfrom pathlib import Path\n"
            f"q.LOG_DIR = Path(r'{tmp_path}')\nq.redirect_output_and_ignore_console_signals()\n"
            "signal.raise_signal(signal.SIGINT)\nsignal.raise_signal(signal.SIGBREAK)\nprint('still here', flush=True)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert "still here" in (tmp_path / "queue_stdout.log").read_text() and r.stdout == ""


def test_without_the_flag_ctrl_c_still_stops_a_runner_started_by_hand():
    code = ("import signal, sys\nsys.path.insert(0, r'%s')\nimport kc23_queue\n"
            "try:\n    signal.raise_signal(signal.SIGINT)\nexcept KeyboardInterrupt:\n    print('interrupted')\n" % (REPO / "src"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert "interrupted" in r.stdout


def test_launch_uses_the_detached_spawn_with_its_own_console_group_and_no_stdin(monkeypatch, tmp_path):
    seen = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kw):
        seen.update(kw, cmd=cmd)
        return FakeProc()

    monkeypatch.setattr(q.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(q, "LOG_DIR", tmp_path)
    monkeypatch.setattr(q, "record_running", lambda job: None)
    job = q.Job({"job_id": "j1", "stage": "S", "seed": "", "command": "echo hi", "out_dir": str(tmp_path / "o"),
                 "depends_on": "", "gate_script": "", "expected_outputs": "", "light": ""})
    q.launch(job)
    assert seen["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP and seen["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert not (seen["creationflags"] & subprocess.CREATE_NEW_CONSOLE) and seen["stdin"] == subprocess.DEVNULL
    assert seen["shell"] is True and seen["stderr"] == subprocess.STDOUT


def test_the_scheduled_task_runs_pythonw_with_redirect_output():
    ps1 = (REPO / "scripts/kc23_queue_task.ps1").read_text(encoding="utf-8")
    assert "pythonw.exe" in ps1 and "--redirect-output" in ps1
    assert "<Command>cmd.exe</Command>" not in ps1 and "queue_stdout.log\" 2>>" not in ps1
