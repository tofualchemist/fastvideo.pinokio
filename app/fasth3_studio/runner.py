"""Parent side of the runner protocol.

A runner is a child Python process that owns the model runtime (MLX today,
CUDA later). The parent sends JSON commands on stdin and reads JSON events on
stdout, one per line. Killing the child is how a running generation is
cancelled, so the child is kept stateless between jobs.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

Event = dict

STEP_RE = re.compile(r"H3 denoise step (\d+)/(\d+).*? in ([\d.]+)s")
PHASE_MARKERS = (
    ("Loaded prompt embeddings", "condition"),
    ("Prepared ", "condition"),
    ("Geometry:", "condition"),
    ("Loaded MLX H3 DiT", "denoise"),
    ("Decoding video", "decode"),
    ("Generation complete", "mux"),
)


def progress_from_log_line(line: str) -> Event | None:
    """Derive a progress event from a runtime log line (works whatever the child's logging setup)."""
    match = STEP_RE.search(line)
    if match:
        return {"event": "progress", "phase": "denoise", "step": int(match.group(1)), "steps": int(match.group(2)),
                "step_s": float(match.group(3)), "message": line.split("] ", 1)[-1][:200]}
    for needle, phase in PHASE_MARKERS:
        if needle in line:
            return {"event": "progress", "phase": phase, "message": line.split("] ", 1)[-1][:200]}
    return None


class RunnerError(Exception):
    pass


class Runner:

    def __init__(self, script: Path, python: str | None = None, env: dict | None = None,
                 on_event: Callable[[Event], None] | None = None, log_path: Path | None = None):
        self.script = Path(script)
        self.python = python or sys.executable
        self.env = {**os.environ, **(env or {})}
        self.on_event = on_event or (lambda e: None)
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._events: queue.Queue[Event] = queue.Queue()
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------------------------

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.alive():
            return
        self.proc = subprocess.Popen([self.python, "-u", str(self.script)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                                     env=self.env, cwd=str(self.script.parent))
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._wait_for("ready", timeout=600)

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:  # noqa: BLE001
            pass

    # -- commands -------------------------------------------------------------------------

    def generate(self, job_id: str, load: dict, params: dict) -> dict:
        with self._lock:
            self.start()
            self._send({"cmd": "generate", "job_id": job_id, "load": load, "params": params})
            while True:
                event = self._events.get()
                if event.get("event") == "done" and event.get("job_id") == job_id:
                    return event.get("result", {})
                if event.get("event") == "error":
                    raise RunnerError(event.get("error", "runner error"))
                if event.get("event") == "exit":
                    raise RunnerError(event.get("error", "runner exited unexpectedly"))

    def ping(self) -> bool:
        if not self.alive():
            return False
        with self._lock:
            self._send({"cmd": "ping"})
            try:
                self._wait_for("pong", timeout=30)
                return True
            except RunnerError:
                return False

    # -- internals ------------------------------------------------------------------------

    def _send(self, message: dict) -> None:
        if not self.alive():
            raise RunnerError("runner is not running")
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, name: str, timeout: float) -> Event:
        while True:
            try:
                event = self._events.get(timeout=timeout)
            except queue.Empty as exc:
                raise RunnerError(f"timed out waiting for runner '{name}'") from exc
            if event.get("event") == name:
                return event
            if event.get("event") in ("error", "exit"):
                raise RunnerError(event.get("error", "runner failed"))

    def _read_stdout(self) -> None:
        proc = self.proc
        assert proc and proc.stdout
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"event": "log", "message": line}
            self.on_event(event)
            if event.get("event") in ("ready", "pong", "done", "error", "exit"):
                self._events.put(event)
        code = proc.wait()
        if self.proc is proc or self.proc is None:
            event = {"event": "exit", "code": code,
                     "error": f"runner exited with code {code}" if code else "runner exited"}
            self.on_event(event)
            self._events.put(event)

    def _read_stderr(self) -> None:
        proc = self.proc
        assert proc and proc.stderr
        log = open(self.log_path, "a", encoding="utf-8") if self.log_path else None
        try:
            for line in proc.stderr:
                if log:
                    log.write(line)
                    log.flush()
                line = line.rstrip()
                self.on_event({"event": "log", "message": line})
                progress = progress_from_log_line(line)
                if progress:
                    self.on_event(progress)
        finally:
            if log:
                log.close()
