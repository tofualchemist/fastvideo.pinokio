"""Job queue: one worker thread turns queued turns into clips, installing models on demand."""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from . import registry
from .models import Cancelled, InstallError, ModelManager
from .costmodel import CostStore, estimate, sample_from_result
from .projects import ProjectStore
from .runner import Runner, RunnerError

FRAMES_FOR_SECONDS = {5: 124, 10: 243, 15: 362}
CANVAS = {"480p": (832, 480), "768p": (1344, 768)}


class EventBus:

    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


class JobQueue:

    def __init__(self, store: ProjectStore, models: ModelManager, bus: EventBus, costs: CostStore, data_dir: Path,
                 runner_factory: Callable[[Callable[[dict], None]], Runner], settings: Callable[[], dict]):
        self.store = store
        self.models = models
        self.bus = bus
        self.costs = costs
        self.data_dir = Path(data_dir)
        self.settings = settings
        self._runner_factory = runner_factory
        self.runner: Runner | None = None
        self.queue: list[tuple[str, str]] = []
        self.current: tuple[str, str] | None = None
        self.cancel_flag = threading.Event()
        self.cond = threading.Condition()
        self.log_lines: list[str] = []
        self.thread = threading.Thread(target=self._loop, daemon=True, name="fasth3-jobs")

    # -- public ---------------------------------------------------------------------------

    def start(self) -> None:
        # Turns that were active when the server last stopped cannot be resumed.
        for pid, tid in self.store.active_turns():
            self.store.update_turn(pid, tid, status="failed", error="Interrupted by a server restart.",
                                   finished_at=time.time())
            stale = self.store.clip_path(pid, tid).with_suffix(".preview.jpg")
            if stale.exists():
                stale.unlink()
        self.thread.start()

    def enqueue(self, pid: str, tid: str) -> None:
        with self.cond:
            self.queue.append((pid, tid))
            self.cond.notify()
        self._publish_queue()

    def cancel(self, pid: str, tid: str) -> bool:
        with self.cond:
            if (pid, tid) in self.queue:
                self.queue.remove((pid, tid))
                if pid != "__install__":
                    self.store.update_turn(pid, tid, status="cancelled", finished_at=time.time())
                    self._publish_turn(pid, tid)
                self._publish_queue()
                return True
            if self.current == (pid, tid):
                self.cancel_flag.set()
                if self.runner is not None:
                    self.runner.stop()
                return True
        return False

    def view(self) -> dict:
        with self.cond:
            current = self.current
            view = {"current": list(current) if current else None,
                    "queued": [list(x) for x in self.queue],
                    "runner_alive": bool(self.runner and self.runner.alive())}
        if current and current[0] != "__install__":
            try:
                turn = self.store.get_turn(*current)
                p = turn.get("progress", {})
                view["active"] = {"project_id": current[0], "project_name": self.store.get(current[0])["name"],
                                  "turn_id": current[1], "turn_index": turn.get("index"), "status": turn["status"],
                                  "phase": p.get("phase"), "step": p.get("step"), "steps": p.get("steps"),
                                  "eta_s": p.get("eta_s"), "message": (p.get("message") or "")[:120]}
            except KeyError:
                pass
        return view

    # -- worker ---------------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            with self.cond:
                while not self.queue:
                    self.cond.wait()
                self.current = self.queue.pop(0)
                self.cancel_flag.clear()
            pid, tid = self.current
            self._publish_queue()
            try:
                if pid == "__install__":
                    self._install_only(tid)
                else:
                    self._run(pid, tid)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                if pid != "__install__":
                    try:
                        self.store.update_turn(pid, tid, status="failed", error=f"{type(exc).__name__}: {exc}",
                                               finished_at=time.time())
                        self._publish_turn(pid, tid)
                    except KeyError:
                        pass
            finally:
                with self.cond:
                    self.current = None
                self._publish_queue()

    def enqueue_install(self, model_id: str) -> None:
        self.enqueue("__install__", model_id)

    def _install_only(self, model_id: str) -> None:
        def progress(phase: str, fraction: float | None, message: str) -> None:
            self.bus.publish({"type": "install", "model": model_id, "phase": phase, "fraction": fraction,
                              "message": message})
        try:
            self.models.ensure_model(model_id, progress, self.cancel_flag)
            self.bus.publish({"type": "install", "model": model_id, "phase": "done", "fraction": 1.0,
                              "message": "Installed"})
        except Cancelled:
            self.bus.publish({"type": "install", "model": model_id, "phase": "cancelled", "message": "Cancelled"})
        except InstallError as exc:
            self.bus.publish({"type": "install", "model": model_id, "phase": "failed", "message": str(exc)})
        self.bus.publish({"type": "models"})

    def _progress(self, pid: str, tid: str, progress: dict) -> None:
        turn = self.store.get_turn(pid, tid)
        merged = {**turn.get("progress", {}), **progress, "updated": time.time()}
        merged["eta_s"] = self._eta(turn, merged)
        self.store.update_turn(pid, tid, progress=merged)
        self._publish_turn(pid, tid)

    def _eta(self, turn: dict, progress: dict) -> float | None:
        phase = progress.get("phase")
        if phase in ("download", "convert"):
            return None
        try:
            return self._eta_seconds(turn, progress, phase)
        except Exception:  # noqa: BLE001  - an estimate must never take the runner's reader thread down
            return None

    def _eta_seconds(self, turn: dict, progress: dict, phase: str | None) -> float | None:
        settings = self.settings()
        spec = registry.MODELS[turn["model"]]
        est = estimate(self.costs, spec, quality=turn.get("quality", "final"), frames=turn["num_frames"],
                       width=turn["width"], height=turn["height"], refs=len(turn.get("refs") or []),
                       ref_clips=sum(1 for r in turn.get("refs") or [] if r.get("media_type") == "video"),
                       fast=bool(turn.get("fast")), fast_spatial=bool(turn.get("fast_spatial")),
                       decoder=turn.get("decoder"), cooldown_s=float(settings.get("cooldown_s") or 0.0),
                       breakdown=True)
        if est["total_s"] is None:
            return None
        est = est["phases"]
        steps = int(progress.get("steps") or turn.get("num_steps") or est.get("steps") or 0)
        step = int(progress.get("step") or 0)
        step_s = progress.get("step_s") or est["step_s"]
        est["decode_s"] = est["decode_s"] + est.get("rife_s", 0.0)
        remaining = 0.0
        order = ["load", "condition", "denoise", "decode", "mux"]
        if phase not in order:
            phase = "load"
        idx = order.index(phase)
        if idx <= 0:
            remaining += est["load_s"]
        if idx <= 1:
            remaining += est["condition_s"]
        if idx <= 2:
            remaining += max(0, steps - step) * (step_s + est.get("cooldown_s", 0.0))
        if idx <= 3:
            remaining += est["decode_s"]
        return round(remaining)

    def _run(self, pid: str, tid: str) -> None:
        turn = self.store.get_turn(pid, tid)
        if turn["status"] != "queued":
            return
        spec = registry.MODELS[turn["model"]]
        self.store.update_turn(pid, tid, status="preparing", started_at=time.time(),
                               progress={"phase": "prepare", "message": "Preparing", "fraction": None})
        self._publish_turn(pid, tid)

        def install_progress(phase: str, fraction: float | None, message: str) -> None:
            self._progress(pid, tid, {"phase": phase, "fraction": fraction, "message": message})

        try:
            self.models.ensure_model(spec.id, install_progress, self.cancel_flag)
        except Cancelled:
            self.store.update_turn(pid, tid, status="cancelled", finished_at=time.time())
            self._publish_turn(pid, tid)
            return
        except InstallError as exc:
            self.store.update_turn(pid, tid, status="failed", error=str(exc), finished_at=time.time())
            self._publish_turn(pid, tid)
            return
        self.bus.publish({"type": "models"})

        settings = self.settings()
        decoder = turn.get("decoder") or ("fast" if turn.get("quality") == "draft" else "quality")
        clip = self.store.clip_path(pid, tid)
        clip.parent.mkdir(parents=True, exist_ok=True)
        if spec.engine == "wan":
            load = {**self.models.paths_for(spec.id), "fastvideo_dir": str(self.models.fastvideo_dir),
                    "prompt_cache_dir": str(self.data_dir / "prompt_cache_wan")}
            params = {
                "prompt": turn["prompt"], "output_path": str(clip), "width": turn["width"], "height": turn["height"],
                "num_frames": turn["num_frames"], "seed": turn["seed"], "num_steps": turn.get("num_steps") or spec.steps,
                "fps": spec.fps, "fast": bool(turn.get("fast")), "fast_spatial": bool(turn.get("fast_spatial")),
                # The full Wan VAE costs ~6 min per clip on MPS; TAEHV is the lab's default. Opt in per turn.
                "decode_backend": "wan-vae" if turn.get("wan_vae") else "taehv",
            }
            return self._execute(pid, tid, turn, spec, load, params, clip)
        load = {
            **self.models.paths_for(spec.id, decoder=decoder),
            "prompt_cache_dir": str(self.data_dir / "prompt_cache"),
            "vae_dtype": settings.get("vae_dtype", "bf16"),
            "metal_wired_limit_gib": self._wired_limit(spec, turn, settings),
            "mlx_cache_limit_gib": settings.get("mlx_cache_limit_gib", 8),
            "quant_gemm": settings.get("quant_gemm", "auto"),
            "resident": bool(settings.get("resident", False)),
            "preview": bool(settings.get("previews", True)),
            "attention_chunk_mib": settings.get("attention_chunk_mib", 192),
        }
        refs = []
        for ref in turn.get("refs", []):
            refs.append({"source": str(self.store.dir(pid) / ref["path"]), "media_type": ref["media_type"]})
        params = {
            "prompt": turn["prompt"],
            "output_path": str(clip),
            "width": turn["width"],
            "height": turn["height"],
            "num_frames": turn["num_frames"],
            "seed": turn["seed"],
            "num_steps": turn.get("num_steps") or spec.steps,
            "fast": bool(turn.get("fast")),
            "fast_spatial": bool(turn.get("fast_spatial")),
            "vsa": spec.vsa if spec.vsa else False,
            "vsa_impl": turn.get("vsa_impl") or settings.get("vsa_impl") or spec.default_vsa_impl,
            "inter_step_cooldown_s": float(settings.get("cooldown_s", 0.0)),
            "references": refs or None,
            "reference_max_pixels": turn.get("reference_max_pixels"),
        }
        if spec.vsa_sparsity is not None:
            params["vsa_sparsity"] = spec.vsa_sparsity
        if spec.vsa_tile_size is not None:
            params["vsa_tile_size"] = spec.vsa_tile_size
        return self._execute(pid, tid, turn, spec, load, params, clip)


    def _wired_limit(self, spec, turn: dict, settings: dict):
        """An explicit value in Settings wins; otherwise the runner applies FastVideo's recipe (two thirds of RAM).

        A per-job estimate (weights + scaled activations + headroom, 28-34 GiB on a 64 GB Mac) was tried in
        Oct 2026 and reverted: it is not what the lab validated, and it brought no measurable benefit. The GPU
        watchdog failures seen at the time occur at the recipe value too, so they are not caused by the limit.
        """
        explicit = settings.get("metal_wired_limit_gib", "auto")
        if explicit not in (None, "", "auto"):
            return explicit
        return "auto"

    def _execute(self, pid: str, tid: str, turn: dict, spec, load: dict, params: dict, clip: Path) -> None:
        wired = load.get("metal_wired_limit_gib")
        note = f"Starting runner · pinning {wired} GiB" if isinstance(wired, (int, float)) else "Starting runner"
        self.store.update_turn(pid, tid, status="running",
                               progress={"phase": "load", "message": note, "fraction": None,
                                         "steps": params["num_steps"], "step": 0})
        self._progress(pid, tid, {})
        if self.runner is None or not self.runner.alive():
            self.runner = self._runner_factory(self._on_runner_event)
        self._active_turn = (pid, tid)
        self._job_meta = {}
        try:
            result = self.runner.generate(tid, load, params)
        except RunnerError as exc:
            preview_file = clip.with_suffix(".preview.jpg")
            if preview_file.exists():
                preview_file.unlink()
            if self.cancel_flag.is_set():
                self.store.update_turn(pid, tid, status="cancelled", finished_at=time.time())
            else:
                message = str(exc)
                if "code -15" in message or "code -9" in message:
                    message = "The model runner was stopped (server restart). Use Retry to run this turn again."
                self.store.update_turn(pid, tid, status="failed", error=message, finished_at=time.time())
            self._publish_turn(pid, tid)
            return
        finally:
            self._active_turn = None
        if not clip.exists():
            self.store.update_turn(pid, tid, status="failed", error="Runner finished without writing a clip.",
                                   finished_at=time.time())
            self._publish_turn(pid, tid)
            return
        sample = sample_from_result(turn, spec, result, self._job_meta, self.settings())
        if sample:
            self.costs.record(sample)
        preview_file = clip.with_suffix(".preview.jpg")
        if preview_file.exists():
            preview_file.unlink()
        self.store.update_turn(pid, tid, status="done", finished_at=time.time(), seen=False,
                               result={"clip": self.store.rel(pid, clip), "wall_s": result.get("wall_s"),
                                       "timings_s": result.get("timings_s", {}),
                                       "peak_memory_gib": result.get("peak_memory_gib", {}),
                                       "step_times_s": result.get("step_times_s", [])},
                               progress={"phase": "done", "fraction": 1.0})
        self._publish_turn(pid, tid)
        self.bus.publish({"type": "notify", "project_id": pid, "turn_id": tid,
                          "title": "Clip ready", "body": turn["prompt"][:80]})

    _active_turn: tuple[str, str] | None = None
    _job_meta: dict = {}

    def _on_runner_event(self, event: dict) -> None:
        kind = event.get("event")
        if kind == "log":
            line = event.get("message", "")
            self.log_lines.append(line)
            self.log_lines = self.log_lines[-400:]
            self.bus.publish({"type": "log", "line": line})
        elif kind == "meta" and self._active_turn:
            self._job_meta.update({k: v for k, v in event.items() if k != "event"})
        elif kind == "preview" and self._active_turn:
            pid, tid = self._active_turn
            try:
                import base64
                path = self.store.clip_path(pid, tid).with_suffix(".preview.jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(event.get("jpeg_b64", "")))
                self._progress(pid, tid, {"preview": self.store.rel(pid, path), "preview_step": event.get("step"),
                                          "preview_cost_s": event.get("cost_s")})
            except (KeyError, OSError, ValueError):
                pass
        elif kind == "progress" and self._active_turn:
            pid, tid = self._active_turn
            progress = {k: v for k, v in event.items() if k in ("phase", "step", "steps", "step_s", "message")}
            steps = progress.get("steps")
            step = progress.get("step")
            if steps and step is not None:
                progress["fraction"] = step / steps
            try:
                self._progress(pid, tid, progress)
            except KeyError:
                pass

    def _publish_turn(self, pid: str, tid: str) -> None:
        try:
            turn = self.store.get_turn(pid, tid)
        except KeyError:
            return
        self.bus.publish({"type": "turn", "project_id": pid, "turn": turn})
        if self.current == (pid, tid):
            p = turn.get("progress", {})
            try:
                name = self.store.get(pid)["name"]
            except KeyError:
                name = pid
            self.bus.publish({"type": "active", "project_id": pid, "project_name": name, "turn_id": tid,
                              "turn_index": turn.get("index"), "status": turn["status"], "phase": p.get("phase"),
                              "step": p.get("step"), "steps": p.get("steps"), "eta_s": p.get("eta_s"),
                              "message": (p.get("message") or "")[:120]})

    def _publish_queue(self) -> None:
        self.bus.publish({"type": "queue", **self.view()})
