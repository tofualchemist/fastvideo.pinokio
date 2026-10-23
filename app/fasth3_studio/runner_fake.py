"""Fake runner for UI development: same protocol, produces a test clip with ffmpeg in seconds."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback

_PROTO = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def emit(event: dict) -> None:
    _PROTO.write(json.dumps(event) + "\n")
    _PROTO.flush()


def generate(job_id: str, load: dict, params: dict) -> dict:
    steps = int(params.get("num_steps", 4))
    step_s = float(os.environ.get("FASTH3_FAKE_STEP_S", "1.0"))
    w, h, n = int(params.get("width", 832)), int(params.get("height", 480)), int(params.get("num_frames", 124))
    seed = int(params.get("seed", 0))
    emit({"event": "progress", "job_id": job_id, "phase": "load", "message": "Fake: preparing"})
    time.sleep(step_s * 0.5)
    emit({"event": "progress", "job_id": job_id, "phase": "condition", "message": "Fake: encoding prompt"})
    time.sleep(step_s * 0.5)
    times = []
    for i in range(steps):
        time.sleep(step_s)
        times.append(step_s)
        emit({"event": "progress", "job_id": job_id, "phase": "denoise", "step": i + 1, "steps": steps,
              "step_s": step_s, "message": f"H3 denoise step {i + 1}/{steps} in {step_s:.2f}s"})
    emit({"event": "progress", "job_id": job_id, "phase": "decode", "message": "Fake: decoding"})
    hue = (seed * 37) % 360
    duration = n / 24
    out = params["output_path"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    freq = 220 + (seed % 12) * 40
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=24,hue=h={hue}",
           "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=32000",
           "-t", f"{duration:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out]
    subprocess.run(cmd, check=True)
    emit({"event": "progress", "job_id": job_id, "phase": "mux", "message": "Fake: muxing"})
    return {"video_path": out, "wall_s": round(step_s * (steps + 1), 2),
            "timings_s": {"condition_s": step_s * 0.5, "denoise_s": step_s * steps, "video_decode_s": 0.2},
            "peak_memory_gib": {"denoise_gib": 19.5}, "step_times_s": times, "vsa": {}}


def main() -> None:
    emit({"event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        command = json.loads(line)
        if command.get("cmd") == "ping":
            emit({"event": "pong"})
        elif command.get("cmd") == "generate":
            job_id = command.get("job_id")
            try:
                emit({"event": "done", "job_id": job_id, "result": generate(job_id, command["load"], command["params"])})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                emit({"event": "error", "job_id": job_id, "error": f"{type(exc).__name__}: {exc}"})
        elif command.get("cmd") == "quit":
            break


if __name__ == "__main__":
    main()
