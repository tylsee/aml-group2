import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdccp import gpu_runner
from cdccp.gpu_runner import require_fresh, run_on_gpus

FAST = dict(poll_seconds=0.05, report_seconds=0.2, terminate_grace=5.0)

REPORT_CODE = "import os,sys,time; print(sys.argv[1], 'gpu=' + os.environ['CUDA_VISIBLE_DEVICES'], 'start=%f' % time.time(), flush=True); time.sleep(0.3); print('done', flush=True)"
SLEEP_CODE = "import sys,time; print('sleeping', sys.argv[1], flush=True); time.sleep(60)"
FAIL_CODE = "import sys,time; print('about to fail', flush=True); time.sleep(1.0); sys.exit(3)"
HEARTBEAT_CODE = (
    "import sys,time\n"
    "while True:\n"
    "    open(sys.argv[1], 'w').write(str(time.time()))\n"
    "    time.sleep(0.05)\n"
)


def command(code, *args):
    return [sys.executable, "-c", code, *args]


def p4_threads():
    return [t for t in threading.enumerate() if t.name.startswith("job-gpu")]


def test_success_runs_every_arm_in_order_with_right_gpu(tmp_path):
    jobs = {0: ["a0", "a1"], 1: ["b0", "b1"]}
    logs = tmp_path / "logs"
    result = run_on_gpus(jobs, lambda arm: command(REPORT_CODE, arm), logs, tmp_path, printer=lambda text: None, **FAST)
    assert result is None
    started = {}
    for gpu, arms in jobs.items():
        for arm in arms:
            text = (logs / f"{arm}.log").read_text()
            assert f"{arm} gpu={gpu} " in text and "done" in text
            started[arm] = float(text.split("start=")[1].split()[0])
    assert started["a0"] < started["a1"] and started["b0"] < started["b1"]
    assert p4_threads() == []


def test_tuple_keys_run_two_queues_at_once_on_one_gpu(tmp_path):
    jobs = {(0, 0): ["a", "b"], (0, 1): ["c", "d"]}
    logs = tmp_path / "logs"
    seen_threads = []

    def printer(text):
        seen_threads.extend(t.name for t in p4_threads())

    result = run_on_gpus(jobs, lambda arm: command(REPORT_CODE, arm), logs, tmp_path, printer=printer, **FAST)
    assert result is None
    spans = {}
    for arm in "abcd":
        text = (logs / f"{arm}.log").read_text()
        assert f"{arm} gpu=0 " in text and "done" in text
        spans[arm] = float(text.split("start=")[1].split()[0])
    assert spans["a"] < spans["b"] and spans["c"] < spans["d"]
    assert abs(spans["a"] - spans["c"]) < 0.25
    assert spans["b"] - spans["a"] >= 0.3 and spans["d"] - spans["c"] >= 0.3
    assert set(seen_threads) <= {"job-gpu0-slot0", "job-gpu0-slot1"}
    assert p4_threads() == []


def test_mixed_int_and_tuple_keys_use_their_gpu(tmp_path):
    jobs = {0: ["a"], (1, 0): ["b"], (1, 1): ["c"]}
    logs = tmp_path / "logs"
    run_on_gpus(jobs, lambda arm: command(REPORT_CODE, arm), logs, tmp_path, printer=lambda text: None, **FAST)
    assert "a gpu=0 " in (logs / "a.log").read_text()
    assert "b gpu=1 " in (logs / "b.log").read_text()
    assert "c gpu=1 " in (logs / "c.log").read_text()


def test_thread_names_stay_unique_for_int_and_tuple_keys(tmp_path, monkeypatch):
    names = []
    real_thread = threading.Thread

    def spy(*args, **kwargs):
        names.append(kwargs["name"])
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(gpu_runner.threading, "Thread", spy)
    jobs = {0: ["a"], 1: ["b"], (2, 0): ["c"], (2, 1): ["d"]}
    run_on_gpus(jobs, lambda arm: command(REPORT_CODE, arm), tmp_path / "logs", tmp_path, printer=lambda text: None, **FAST)
    assert sorted(names) == ["job-gpu0", "job-gpu1", "job-gpu2-slot0", "job-gpu2-slot1"]
    assert len(set(names)) == len(names)


def test_failure_in_one_slot_stops_the_other_slot_on_the_same_gpu(tmp_path):
    jobs = {(0, 0): ["bad", "later0"], (0, 1): ["sleeper", "later1"]}
    logs = tmp_path / "logs"
    begin = time.monotonic()

    def make(arm):
        return command(FAIL_CODE if arm == "bad" else SLEEP_CODE, arm)

    with pytest.raises(RuntimeError) as err:
        run_on_gpus(jobs, make, logs, tmp_path, printer=lambda text: None, **FAST)
    assert time.monotonic() - begin < 15
    message = str(err.value)
    assert "bad" in message and "exit code 3" in message
    assert "sleeper" in message and "stopped" in message
    assert not (logs / "later0.log").exists() and not (logs / "later1.log").exists()
    assert p4_threads() == []


def test_failing_job_stops_other_gpu_and_later_arms(tmp_path):
    jobs = {0: ["bad", "later0"], 1: ["sleeper", "later1"]}
    logs = tmp_path / "logs"
    begin = time.monotonic()

    def make(arm):
        return command(FAIL_CODE if arm == "bad" else SLEEP_CODE, arm)

    with pytest.raises(RuntimeError) as err:
        run_on_gpus(jobs, make, logs, tmp_path, printer=lambda text: None, **FAST)
    assert time.monotonic() - begin < 15
    message = str(err.value)
    assert "bad" in message and "exit code 3" in message
    assert "sleeper" in message and "stopped" in message
    assert not (logs / "later0.log").exists() and not (logs / "later1.log").exists()
    assert p4_threads() == []


def test_failure_prints_reason_and_log_tail(tmp_path):
    lines = []
    with pytest.raises(RuntimeError):
        run_on_gpus({0: ["bad"]}, lambda arm: command(FAIL_CODE), tmp_path / "logs", tmp_path, printer=lines.append, **FAST)
    text = "\n".join(lines)
    assert "FAILED bad: exit code 3" in text and "about to fail" in text


def test_make_command_error_is_recorded_and_others_stop(tmp_path):
    jobs = {0: ["boom", "later0"], 1: ["sleeper"]}
    logs = tmp_path / "logs"

    def make(arm):
        if arm == "boom":
            raise ValueError("cannot build command")
        return command(SLEEP_CODE, arm)

    begin = time.monotonic()
    with pytest.raises(RuntimeError) as err:
        run_on_gpus(jobs, make, logs, tmp_path, printer=lambda text: None, **FAST)
    assert time.monotonic() - begin < 15
    message = str(err.value)
    assert "boom" in message and "ValueError" in message and "cannot build command" in message
    assert not (logs / "later0.log").exists()
    assert p4_threads() == []


def test_popen_error_is_recorded(tmp_path):
    def make(arm):
        return [str(tmp_path / "no_such_program.exe")]

    with pytest.raises(RuntimeError) as err:
        run_on_gpus({0: ["a"], 1: ["b"]}, make, tmp_path / "logs", tmp_path, printer=lambda text: None, **FAST)
    assert "a:" in str(err.value) and ("FileNotFoundError" in str(err.value) or "OSError" in str(err.value))
    assert p4_threads() == []


def test_budget_exceeded_stops_everything(tmp_path):
    jobs = {0: ["s0", "s2"], 1: ["s1"]}
    logs = tmp_path / "logs"
    begin = time.monotonic()
    with pytest.raises(RuntimeError) as err:
        run_on_gpus(jobs, lambda arm: command(SLEEP_CODE, arm), logs, tmp_path, budget_hours=1.0 / 3600,
                    printer=lambda text: None, **FAST)
    assert time.monotonic() - begin < 15
    message = str(err.value)
    assert "budget" in message and "s0" in message and "s1" in message
    assert not (logs / "s2.log").exists()
    assert p4_threads() == []


def test_keyboard_interrupt_propagates_and_children_are_gone(tmp_path):
    beats = {"h0": tmp_path / "h0.txt", "h1": tmp_path / "h1.txt"}
    logs = tmp_path / "logs"

    def printer(text):
        if all(path.exists() and path.read_text() for path in beats.values()):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_on_gpus({0: ["h0", "next0"], 1: ["h1"]}, lambda arm: command(HEARTBEAT_CODE, str(beats.get(arm, tmp_path / arm))),
                    logs, tmp_path, printer=printer, **FAST)
    assert p4_threads() == []
    before = {name: path.stat().st_mtime for name, path in beats.items()}
    time.sleep(0.5)
    after = {name: path.stat().st_mtime for name, path in beats.items()}
    assert before == after
    assert not (logs / "next0.log").exists()


class FakeProc:
    instances = []

    def __init__(self, command, **kwargs):
        self.calls = []
        self.killed = False
        FakeProc.instances.append(self)

    def poll(self):
        return -9 if self.killed else None

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")
        self.killed = True

    def wait(self, timeout=None):
        if self.killed:
            return -9
        self.calls.append(f"wait({timeout})")
        time.sleep(min(timeout or 0.05, 0.05))
        raise subprocess.TimeoutExpired("fake", timeout)


def test_child_that_ignores_terminate_is_killed_after_grace(tmp_path, monkeypatch):
    FakeProc.instances = []
    monkeypatch.setattr(gpu_runner.subprocess, "Popen", FakeProc)
    with pytest.raises(RuntimeError) as err:
        run_on_gpus({0: ["stubborn"]}, lambda arm: ["unused"], tmp_path / "logs", tmp_path, budget_hours=0.2 / 3600,
                    poll_seconds=0.05, report_seconds=0.2, terminate_grace=0.1, printer=lambda text: None)
    assert "budget" in str(err.value)
    (proc,) = FakeProc.instances
    assert "terminate" in proc.calls and "kill" in proc.calls
    assert proc.calls.index("terminate") < proc.calls.index("kill")
    assert "wait(0.1)" in proc.calls
    assert p4_threads() == []


def test_require_fresh_accepts_missing_folders(tmp_path):
    require_fresh(tmp_path, ["a", "b"])
    require_fresh(tmp_path / "does_not_exist_yet", ["a"])


def test_require_fresh_names_every_existing_folder_even_empty(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "best.pt").write_text("x")
    with pytest.raises(FileExistsError) as err:
        require_fresh(tmp_path, ["a", "b", "c"])
    message = str(err.value)
    assert "a" in message.split("for: ")[1].split(" (")[0].split(", ")
    assert "c" in message.split("for: ")[1].split(" (")[0].split(", ")
    assert "b" not in message.split("for: ")[1].split(" (")[0].split(", ")
    assert "new Kaggle version" in message
