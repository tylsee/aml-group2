import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path


def require_fresh(out_dir, arms):
    existing = [arm for arm in arms if (Path(out_dir) / arm).exists()]
    if existing:
        raise FileExistsError(
            f"output folders already exist for: {', '.join(existing)} (in {out_dir}). "
            "Start a new Kaggle version or use a fresh output folder so old results are not mixed with new ones."
        )


def _last_line(path):
    path = Path(path)
    if not path.exists():
        return "(no log yet)"
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 4096))
        text = handle.read().decode("utf-8", errors="replace")
    lines = [line for line in text.replace("\r", "\n").split("\n") if line.strip()]
    return lines[-1] if lines else "(empty log)"


def _tail(path, count):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=count)]


def _terminate(proc, grace):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class _Run:
    def __init__(self, jobs_per_gpu, make_command, log_dir, cwd, poll_seconds, terminate_grace):
        self.jobs_per_gpu = jobs_per_gpu
        self.make_command = make_command
        self.log_dir = Path(log_dir)
        self.cwd = cwd
        self.poll = poll_seconds
        self.grace = terminate_grace
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.procs = {}
        self.reasons = {}
        self.run_reasons = []

    def fail(self, arm, reason):
        with self.lock:
            self.reasons.setdefault(arm, reason)
        self.stop.set()

    def log_path(self, arm):
        return self.log_dir / f"{arm}.log"

    def worker(self, gpu, arms):
        for arm in arms:
            if self.stop.is_set():
                with self.lock:
                    self.reasons.setdefault(arm, "not started (run stopped)")
                continue
            try:
                self.run_arm(gpu, arm)
            except Exception as exc:
                self.fail(arm, f"{type(exc).__name__}: {exc}")

    def run_arm(self, gpu, arm):
        command = self.make_command(arm)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with open(self.log_path(arm), "wb") as log:
            with self.lock:
                if self.stop.is_set():
                    self.reasons.setdefault(arm, "not started (run stopped)")
                    return
                proc = subprocess.Popen(command, cwd=self.cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
                self.procs[arm] = proc
            try:
                while True:
                    try:
                        code = proc.wait(timeout=self.poll)
                        break
                    except subprocess.TimeoutExpired:
                        if self.stop.is_set():
                            _terminate(proc, self.grace)
                            self.fail(arm, "stopped (another job failed, budget exceeded or interrupted)")
                            return
                if code != 0:
                    if self.stop.is_set():
                        self.fail(arm, f"stopped (another job failed, budget exceeded or interrupted); exit code {code}")
                    else:
                        self.fail(arm, f"exit code {code}")
            finally:
                _terminate(proc, self.grace)

    def terminate_all(self):
        with self.lock:
            procs = list(self.procs.values())
        for proc in procs:
            _terminate(proc, self.grace)


def _gpu_of(key):
    return key[0] if isinstance(key, tuple) else key


def _thread_name(key):
    if isinstance(key, tuple):
        gpu, slot = key
        return f"job-gpu{gpu}-slot{slot}"
    return f"job-gpu{key}"


def run_on_gpus(jobs_per_gpu, make_command, log_dir, cwd, budget_hours=None, poll_seconds=5.0, report_seconds=120.0,
                terminate_grace=30.0, printer=print):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run = _Run(jobs_per_gpu, make_command, log_dir, cwd, poll_seconds, terminate_grace)
    threads = [threading.Thread(target=run.worker, args=(_gpu_of(key), list(arms)), name=_thread_name(key), daemon=True)
               for key, arms in jobs_per_gpu.items()]
    started = time.monotonic()
    next_report = started + report_seconds
    all_arms = [arm for arms in jobs_per_gpu.values() for arm in arms]
    try:
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            time.sleep(run.poll)
            now = time.monotonic()
            if budget_hours is not None and now - started > budget_hours * 3600 and not run.stop.is_set():
                run.run_reasons.append(f"budget of {budget_hours} hours exceeded")
                run.stop.set()
            if now >= next_report:
                next_report = now + report_seconds
                stamp = time.strftime("%H:%M")
                for arm in all_arms:
                    if run.log_path(arm).exists():
                        printer(f"[{stamp}] {arm}: {_last_line(run.log_path(arm))}")
    finally:
        run.stop.set()
        run.terminate_all()
        for thread in threads:
            if thread.is_alive():
                thread.join()
        run.terminate_all()

    problems = [(arm, run.reasons[arm]) for arm in all_arms if arm in run.reasons]
    problems += [("run", reason) for reason in run.run_reasons]
    if not problems:
        return None
    for arm, reason in problems:
        printer(f"FAILED {arm}: {reason}")
        for line in _tail(run.log_path(arm), 30):
            printer(f"    {line}")
    raise RuntimeError("Training stopped: " + "; ".join(f"{arm}: {reason}" for arm, reason in problems))
