"""FastVideo MLX server: static UI + JSON API + SSE, one job worker, one model runner."""
from __future__ import annotations

import asyncio
import json
import os
import random
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fasth3_studio import registry  # noqa: E402
from fasth3_studio.costmodel import CostStore, estimate  # noqa: E402
from fasth3_studio.jobs import CANVAS, FRAMES_FOR_SECONDS, EventBus, JobQueue  # noqa: E402
from fasth3_studio.models import ModelManager  # noqa: E402
from fasth3_studio.projects import ACTIVE, ProjectStore  # noqa: E402
from fasth3_studio.runner import Runner  # noqa: E402


def env_path(name: str, default: str) -> Path:
    return (HERE / os.environ.get(name, default)).resolve()


MODELS_DIR = env_path("FASTH3_STUDIO_MODELS", "../models")
DATA_DIR = env_path("FASTH3_STUDIO_DATA", "../data")
FASTVIDEO_DIR = env_path("FASTH3_STUDIO_FASTVIDEO", "../FastVideo")
FAKE = os.environ.get("FASTH3_STUDIO_FAKE", "") not in ("", "0", "false")
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "logs").mkdir(exist_ok=True)

SETTINGS_PATH = DATA_DIR / "settings.json"
DEFAULT_SETTINGS = {
    "default_model": registry.DEFAULT_MODEL,
    "canvas": "480p",
    "vae_dtype": "bf16",
    "metal_wired_limit_gib": "auto",
    "mlx_cache_limit_gib": 8,
    "quant_gemm": "auto",
    "resident": False,
    "vsa_impl": None,
    "cooldown_s": 0,
    "notifications": True,
    "previews": True,
    "attention_chunk_mib": 48,
    "mlx_ops_per_buffer": 1,
}


def load_settings() -> dict:
    try:
        return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_PATH.read_text())}
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)


def save_settings(values: dict) -> dict:
    merged = {**load_settings(), **{k: v for k, v in values.items() if k in DEFAULT_SETTINGS}}
    SETTINGS_PATH.write_text(json.dumps(merged, indent=1))
    return merged


bus = EventBus()
store = ProjectStore(DATA_DIR)
models = ModelManager(MODELS_DIR, FASTVIDEO_DIR, fake=FAKE)
costs = CostStore(DATA_DIR / ("stats-fake.json" if FAKE else "stats.json"))   # fake runs must not calibrate real ones


def runner_factory(on_event):
    script = HERE / "fasth3_studio" / ("runner_fake.py" if FAKE else "runner_mlx.py")
    env = {"PYTHONPATH": str(FASTVIDEO_DIR), "FASTVIDEO_MLX_DQ_GEMM": "1",
           # macOS kills a GPU command buffer that runs long enough to starve the display
           # (kIOGPUCommandBufferCallbackErrorImpactingInteractivity). Long clips on slower chips hit that
           # inside one attention dispatch. This Apple driver switch relaxes the timeout for this process;
           # llama.cpp's Metal backend sets the same variable unconditionally for the same reason.
           "AGX_RELAX_CDM_CTXSTORE_TIMEOUT": "1"}
    ops = load_settings().get("mlx_ops_per_buffer")
    if ops:
        # Smaller command buffers keep each GPU submission short enough for macOS's watchdog.
        env["MLX_MAX_OPS_PER_BUFFER"] = str(int(ops))
    return Runner(script, on_event=on_event, log_path=DATA_DIR / "logs" / "runner.log", env=env)


jobs = JobQueue(store, models, bus, costs, DATA_DIR, runner_factory, load_settings)


def benchmark_if_needed() -> None:
    """The cost model needs one machine constant before the first run; install.js measures it, this is the fallback."""
    if FAKE or costs.has_benchmark() or costs.samples():
        return

    def run() -> None:
        try:
            import bench
            costs.set_benchmark(bench.measure_tflops(), bench.chip_name())
            bus.publish({"type": "models"})
        except Exception as exc:  # noqa: BLE001
            print(f"GPU benchmark skipped: {exc}")

    threading.Thread(target=run, daemon=True, name="gpu-bench").start()
app = FastAPI(title="FastVideo MLX")


# -- helpers --------------------------------------------------------------------------------


def project_or_404(pid: str) -> dict:
    try:
        return store.get(pid)
    except KeyError:
        raise HTTPException(404, "project not found")


def turn_or_404(pid: str, tid: str) -> dict:
    try:
        return store.get_turn(pid, tid)
    except KeyError:
        raise HTTPException(404, "turn not found")


def media_type_for(name: str) -> str:
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext in ("png", "jpg", "jpeg", "webp", "bmp", "gif"):
        return "image"
    if ext in ("mp4", "mov", "webm", "mkv", "m4v"):
        return "video"
    if ext in ("wav", "mp3", "m4a", "aac", "flac", "ogg"):
        return "audio"
    raise HTTPException(400, f"unsupported reference file type: .{ext}")


# -- API: state -----------------------------------------------------------------------------


@app.get("/api/state")
def api_state():
    return {
        "models": models.status(),
        "queue": jobs.view(),
        "settings": load_settings(),
        "projects": store.list(),
        "fake": FAKE,
        "fastvideo_commit": registry.FASTVIDEO_COMMIT,
    }


@app.get("/api/settings")
def api_settings():
    return load_settings()


@app.put("/api/settings")
async def api_put_settings(request: Request):
    body = await request.json()
    merged = save_settings(body)
    bus.publish({"type": "settings", "settings": merged})
    return merged


@app.get("/api/estimate")
def api_estimate(model: str, quality: str = "draft", refs: int = 0, ref_clips: int = 0, duration: int = 5,
                 canvas: str = "480p", fast: int = 1, fast_spatial: int = 0):
    if model not in registry.MODELS:
        raise HTTPException(404, "unknown model")
    spec = registry.MODELS[model]
    width, height = spec.canvases.get(canvas, next(iter(spec.canvases.values())))
    frames = spec.frames_for_seconds.get(int(duration), next(iter(spec.frames_for_seconds.values())))
    est = estimate(costs, spec, quality=quality, frames=frames, width=width, height=height, refs=refs,
                   ref_clips=ref_clips, fast=bool(fast and quality == "draft" and spec.supports_fast_modes and not refs),
                   fast_spatial=bool(fast_spatial and quality == "draft" and spec.supports_fast_modes),
                   cooldown_s=float(load_settings().get("cooldown_s") or 0.0))
    return {**est, "steps": spec.steps}


@app.get("/api/picker")
def api_picker(quality: str = "draft", need: str = "any"):
    from fasth3_studio import picker
    return picker.build(models, costs, quality=quality, need=need if need in ("any", "audio", "silent", "refs") else "any",
                        cooldown_s=float(load_settings().get("cooldown_s") or 0.0))


@app.get("/api/log")
def api_log():
    return {"lines": jobs.log_lines[-200:]}


# -- API: models ----------------------------------------------------------------------------


@app.get("/api/models")
def api_models():
    return models.status()


@app.post("/api/models/{model_id}/install")
def api_install(model_id: str):
    if model_id not in registry.MODELS:
        raise HTTPException(404, "unknown model")
    jobs.enqueue_install(model_id)
    return {"ok": True}


@app.delete("/api/models/{model_id}")
def api_remove_model(model_id: str):
    if model_id not in registry.MODELS:
        raise HTTPException(404, "unknown model")
    view = jobs.view()
    if view["current"] or view["queued"]:
        raise HTTPException(409, "wait for the queue to finish before removing models")
    models.remove_model(model_id)
    bus.publish({"type": "models"})
    return {"ok": True}


# -- API: projects --------------------------------------------------------------------------


@app.get("/api/projects")
def api_projects():
    return store.list()


@app.post("/api/projects")
async def api_create_project(request: Request):
    body = await request.json()
    project = store.create(body.get("name") or "Untitled")
    bus.publish({"type": "projects"})
    return project


@app.get("/api/projects/{pid}")
def api_project(pid: str):
    return project_or_404(pid)


@app.patch("/api/projects/{pid}")
async def api_rename_project(pid: str, request: Request):
    body = await request.json()
    project_or_404(pid)
    project = store.rename(pid, body.get("name", ""))
    bus.publish({"type": "projects"})
    return project


@app.delete("/api/projects/{pid}")
def api_delete_project(pid: str):
    project = project_or_404(pid)
    for turn in project["turns"]:
        if turn["status"] in ACTIVE:
            jobs.cancel(pid, turn["id"])
    store.delete(pid)
    bus.publish({"type": "projects"})
    return {"ok": True}


@app.post("/api/projects/{pid}/seen")
def api_seen(pid: str):
    project_or_404(pid)
    store.mark_seen(pid)
    bus.publish({"type": "projects"})
    return {"ok": True}


@app.post("/api/projects/{pid}/uploads")
async def api_upload(pid: str, file: UploadFile = File(...)):
    project_or_404(pid)
    data = await file.read()
    if len(data) > 2_000_000_000:
        raise HTTPException(413, "file too large")
    media_type = media_type_for(file.filename or "")
    rel = store.save_upload(pid, file.filename or "upload", data)
    return {"path": rel, "media_type": media_type, "name": file.filename}


# -- API: turns -----------------------------------------------------------------------------


@app.post("/api/projects/{pid}/turns")
async def api_create_turn(pid: str, request: Request):
    project_or_404(pid)
    body = await request.json()
    model_id = body.get("model") or load_settings()["default_model"]
    if model_id not in registry.MODELS:
        raise HTTPException(400, "unknown model")
    spec = registry.MODELS[model_id]
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")
    quality = body.get("quality") if body.get("quality") in ("draft", "final") else "draft"
    duration = int(body.get("duration_s") or 5)
    if duration not in spec.durations_s:
        duration = spec.durations_s[0]
    canvas = body.get("canvas") if body.get("canvas") in spec.canvases else next(iter(spec.canvases))
    width, height = spec.canvases[canvas]
    seed = body.get("seed")
    if seed is None or seed == "":
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed) & 0xFFFFFFFF

    refs = []
    for ref in body.get("refs") or []:
        if not spec.supports_references:
            break
        media_type = ref.get("media_type")
        if media_type not in ("image", "video", "audio"):
            raise HTTPException(400, "reference media_type must be image, video or audio")
        if ref.get("from_turn"):
            src = turn_or_404(pid, ref["from_turn"])
            if src["status"] != "done":
                raise HTTPException(400, "reference turn has no clip")
            refs.append({"media_type": media_type, "path": src["result"]["clip"], "from_turn": src["id"],
                         "name": f"turn {src.get('index', '?')}"})
        elif ref.get("path"):
            path = (store.dir(pid) / ref["path"]).resolve()
            if not str(path).startswith(str(store.dir(pid).resolve())) or not path.exists():
                raise HTTPException(400, "reference file not found")
            refs.append({"media_type": media_type, "path": ref["path"], "name": ref.get("name") or path.name})
    if spec.supports_references and len([r for r in refs if r["media_type"] == "video"]) > 3:
        raise HTTPException(400, "at most 3 video references")

    draft = quality == "draft"
    turn = {
        "kind": body.get("kind") if body.get("kind") in ("generate", "reroll", "finalize") else "generate",
        "parent_id": body.get("parent_id"),
        "model": model_id,
        "prompt": prompt,
        "seed": seed,
        "quality": quality,
        "duration_s": duration,
        "canvas": canvas,
        "width": width,
        "height": height,
        "num_frames": spec.frames_for_seconds[duration],
        "fps": spec.fps,
        "engine": spec.engine,
        "num_steps": spec.steps,
        "decoder": "fast" if draft else "quality",
        "fast": bool(draft and spec.supports_fast_modes and body.get("fast", True)),
        "fast_spatial": bool(draft and spec.supports_fast_modes and body.get("fast_spatial", False)),
        "reference_max_pixels": ("match" if (draft and refs) else None),
        "vsa_impl": body.get("vsa_impl") or None,
        "wan_vae": bool(body.get("wan_vae")) and spec.engine == "wan",
        "refs": refs,
    }
    turn = store.add_turn(pid, turn)
    bus.publish({"type": "turn", "project_id": pid, "turn": turn})
    bus.publish({"type": "projects"})
    jobs.enqueue(pid, turn["id"])
    return turn


@app.post("/api/projects/{pid}/turns/{tid}/cancel")
def api_cancel(pid: str, tid: str):
    turn = turn_or_404(pid, tid)
    if turn["status"] not in ACTIVE:
        raise HTTPException(409, "turn is not active")
    jobs.cancel(pid, tid)
    return {"ok": True}


@app.delete("/api/projects/{pid}/turns/{tid}")
def api_delete_turn(pid: str, tid: str):
    turn = turn_or_404(pid, tid)
    if turn["status"] in ACTIVE:
        jobs.cancel(pid, tid)
    store.delete_turn(pid, tid)
    bus.publish({"type": "turn_deleted", "project_id": pid, "turn_id": tid})
    bus.publish({"type": "projects"})
    return {"ok": True}


# -- SSE ------------------------------------------------------------------------------------


@app.get("/api/events")
async def api_events(request: Request):
    q = bus.subscribe()

    async def stream():
        try:
            yield "retry: 2000\n\n"
            loop = asyncio.get_event_loop()
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await loop.run_in_executor(None, q.get, True, 15)
                except Exception:  # noqa: BLE001  (queue.Empty)
                    yield 'data: {"type": "ping"}\n\n'
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# -- static ---------------------------------------------------------------------------------

app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@app.on_event("startup")
async def announce():
    port = app.state.port
    print(f"FastVideo MLX {'(fake runner) ' if FAKE else ''}running at http://127.0.0.1:{port}", flush=True)


def main() -> None:
    port = int(os.environ.get("FASTH3_STUDIO_PORT") or free_port())
    app.state.port = port
    jobs.start()
    benchmark_if_needed()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
