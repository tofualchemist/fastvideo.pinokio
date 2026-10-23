"""MLX runner: child process that drives FastVideo's MiniMaxH3MLXPipeline.

Protocol: one JSON object per line. stdin commands:
  {"cmd": "ping"}
  {"cmd": "generate", "job_id", "load": {...paths/settings...}, "params": {...generate kwargs...}}
stdout events:
  {"event": "ready"} {"event": "pong"} {"event": "log", "message"}
  {"event": "progress", "job_id", "phase", "step", "steps", "step_s", "message"}
  {"event": "done", "job_id", "result"} {"event": "error", "job_id", "error"}

stdout is reserved for the protocol: everything the libraries print is sent to
stderr, which the parent appends to a log file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Reserve fd 1 for protocol messages and push all other stdout output to stderr.
_PROTO = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

STEP_RE = re.compile(r"H3 denoise step (\d+)/(\d+).*? in ([\d.]+)s")
PHASES = (
    ("Loaded prompt embeddings", "condition"),
    ("Prepared", "condition"),
    ("Loaded MLX H3 DiT", "denoise"),
    ("Geometry:", "condition"),
    ("decode", "decode"),
    ("Generation complete", "mux"),
)


def emit(event: dict) -> None:
    _PROTO.write(json.dumps(event) + "\n")
    _PROTO.flush()


class ProtocolLogHandler(logging.Handler):
    """Turns FastVideo log lines into progress events."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.job_id: str | None = None

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        emit({"event": "log", "message": message[:400]})
        if self.job_id is None:
            return
        match = STEP_RE.search(message)
        if match:
            emit({"event": "progress", "job_id": self.job_id, "phase": "denoise", "step": int(match.group(1)),
                  "steps": int(match.group(2)), "step_s": float(match.group(3)), "message": message[:200]})
            return
        for needle, phase in PHASES:
            if needle in message:
                emit({"event": "progress", "job_id": self.job_id, "phase": phase, "message": message[:200]})
                return


def auto_wired_limit_gib() -> float:
    try:
        import subprocess
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 2**30
    except Exception:  # noqa: BLE001
        mem = 32.0
    return min(64.0, round(mem * 0.66, 1))


def attach_everywhere(handler: logging.Handler) -> None:
    """FastVideo loggers are configured with propagate=False, so hook each one directly."""
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and name.startswith("fastvideo") and handler not in logger.handlers:
            logger.addHandler(handler)
            if logger.level > logging.INFO or logger.level == logging.NOTSET:
                logger.setLevel(logging.INFO)


def apply_attention_chunk_cap(mib: float | None) -> None:
    """Cap the reference VSA gather budget so single attention kernels stay short.

    FastVideo sizes the chunk at 1/128 of unified memory (512 MB on a 64 GB Mac). On slower GPUs a
    chunk that large can run longer than macOS allows for one command buffer and the step dies with
    'Impacting Interactivity'. Smaller chunks mean more, shorter kernels; results are identical.
    """
    from fastvideo.mlx_runtime import minimax_h3_vsa as vsa

    original = getattr(vsa, "_fasth3_studio_original_budget", None) or vsa._reference_gather_target_bytes
    vsa._fasth3_studio_original_budget = original
    if not mib or float(mib) <= 0:
        vsa._reference_gather_target_bytes = original
        return
    cap = int(float(mib) * 2**20)
    vsa._reference_gather_target_bytes = lambda: min(original(), cap)
    emit({"event": "log", "message": f"attention chunk budget capped at {mib} MiB"})


def run_generate(job_id: str, load: dict, params: dict, handler: ProtocolLogHandler) -> dict:
    from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline
    from fastvideo.mlx_runtime.minimax_h3_reference import MiniMaxH3Reference

    attach_everywhere(handler)
    apply_attention_chunk_cap(load.get("attention_chunk_mib"))

    wired = load.get("metal_wired_limit_gib")
    if wired in (None, "auto"):
        wired = auto_wired_limit_gib()
    elif wired in ("off", 0, "0"):
        wired = None
    cache_limit = load.get("mlx_cache_limit_gib", 8)
    emit({"event": "progress", "job_id": job_id, "phase": "load", "message": "Preparing pipeline"})
    pipeline = MiniMaxH3MLXPipeline(
        model_root=Path(load["model_root"]),
        mlx_dit_checkpoint=Path(load["dit"]),
        vae_dtype=load.get("vae_dtype", "bf16"),
        vae_decoder_path=load.get("vae_decoder"),
        vae_encoder_path=load.get("vae_encoder"),
        conditioner_dir=load.get("conditioner_dir"),
        conditioner_mode="nvfp4",
        prompt_cache_dir=load.get("prompt_cache_dir"),
        metal_wired_limit_gib=wired,
        mlx_cache_limit_gib=cache_limit if cache_limit is not None and cache_limit >= 0 else None,
        quant_gemm=load.get("quant_gemm", "auto"),
        resident=bool(load.get("resident", False)),
    )
    hook = None
    if load.get("preview", True):
        try:
            from preview import PreviewHook  # same folder as this script
        except ImportError:
            from fasth3_studio.preview import PreviewHook
        hook = PreviewHook(pipeline, emit, job_id, dtype=("fp32" if load.get("vae_dtype") == "fp32" else "fp16"))
        hook.install()
    try:
        references = None
        if params.get("references"):
            references = [MiniMaxH3Reference(source=str(r["source"]), media_type=r["media_type"])
                          for r in params["references"]]
        kwargs = dict(
            output_path=params["output_path"],
            height=int(params.get("height", 480)),
            width=int(params.get("width", 832)),
            num_frames=int(params.get("num_frames", 124)),
            seed=int(params.get("seed", 0)),
            num_steps=int(params["num_steps"]),
            inter_step_cooldown_s=float(params.get("inter_step_cooldown_s", 0.0)),
        )
        if references is not None:
            kwargs["references"] = references
            rmp = params.get("reference_max_pixels")
            if rmp == "match":
                kwargs["reference_max_pixels"] = kwargs["width"] * kwargs["height"]
            elif rmp:
                kwargs["reference_max_pixels"] = int(rmp)
        else:
            if params.get("fast"):
                kwargs["fast"] = True
            if params.get("fast_spatial"):
                kwargs["fast_spatial"] = True
        # VSA: explicit False opts out of a VSA contract; None lets the snapshot contract decide.
        if params.get("vsa") is False:
            kwargs["vsa"] = False
        else:
            if params.get("vsa_sparsity") is not None:
                kwargs["vsa_sparsity"] = float(params["vsa_sparsity"])
            if params.get("vsa_tile_size") is not None:
                kwargs["vsa_tile_size"] = int(params["vsa_tile_size"])
        if params.get("vsa_impl"):
            kwargs["vsa_impl"] = params["vsa_impl"]
        started = time.perf_counter()
        result = pipeline.generate(params["prompt"], **kwargs)
        return {
            "video_path": result.video_path,
            "wall_s": round(time.perf_counter() - started, 2),
            "timings_s": {k: round(float(v), 2) for k, v in result.timings.items()},
            "peak_memory_gib": {k: round(float(v), 2) for k, v in result.peak_memory_gib.items()},
            "step_times_s": [round(float(s), 2) for s in getattr(pipeline, "last_denoise_step_times", [])],
            "vsa": result.vsa,
        }
    finally:
        if hook is not None:
            hook.uninstall()
        pipeline.close()


# -- FastMetal (Wan family): drive the lab's supported script as a child process ---------------------

WAN_STEP_RE = re.compile(r"denoise step (\d+)/(\d+) complete")
WAN_MARKERS = (
    ("[fast] generating", "condition"),
    ("[wan22 fast] generating", "condition"),
    ("Prompt encode time", "load"),
    ("[5B] prompt encoded", "load"),
    ("MLX DiT load time", "denoise"),
    ("[5B] DiT loaded", "denoise"),
    ("[5B] denoise", "decode"),
    ("Decode/export time", "mux"),
    ("[5B] decoded", "mux"),
    ("Post-decode", "mux"),
    ("Output written to", "mux"),
)
_active_child: subprocess.Popen | None = None


def _terminate(signum, frame):  # noqa: ARG001
    child = _active_child
    if child is not None and child.poll() is None:
        try:
            child.terminate()
            child.wait(timeout=5)
        except Exception:  # noqa: BLE001
            child.kill()
    sys.exit(0)


signal.signal(signal.SIGTERM, _terminate)
signal.signal(signal.SIGINT, _terminate)


def run_generate_wan(job_id: str, load: dict, params: dict) -> dict:
    global _active_child
    fastvideo_dir = Path(load["fastvideo_dir"])
    variant = load.get("script", "wan21")
    script_name = "mlx_wan22_generate.py" if variant == "wan22" else "mlx_wan_prompt_to_video.py"
    script = fastvideo_dir / "examples" / "inference" / "basic" / script_name
    if not script.exists():
        raise FileNotFoundError(f"FastVideo checkout is missing {script}")
    root = load["model_root"]
    output = Path(params["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = output.with_suffix(".metrics.json")
    if variant == "wan22":
        roots = ["--mlx-checkpoint", root, "--text-encoder-root", root, "--vae-root", str(Path(root) / "vae")]
    else:
        roots = ["--model-root", root, "--mlx-checkpoint", root]
    cmd = [sys.executable, "-u", str(script), *roots,
           "--prompt", params["prompt"],
           "--output-path", str(output),
           "--height", str(int(params.get("height", 480))), "--width", str(int(params.get("width", 832))),
           "--num-frames", str(int(params.get("num_frames", 81))),
           "--seed", str(int(params.get("seed", 0))),
           "--fps", str(int(params.get("fps", 16))),
           "--decode-backend", params.get("decode_backend", "taehv"),
           "--metrics-json", str(metrics)]
    if params.get("fast"):
        cmd.append("--fast")
    if params.get("fast_spatial"):
        cmd.append("--fast-spatial")
    emit({"event": "progress", "job_id": job_id, "phase": "condition", "message": "Encoding prompt (UMT5)"})
    started = time.perf_counter()
    env = {**os.environ, "PYTHONPATH": str(fastvideo_dir), "PYTHONUNBUFFERED": "1"}
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             cwd=str(fastvideo_dir), env=env)
    _active_child = child
    tail: list[str] = []
    try:
        assert child.stdout is not None
        for line in child.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            tail[:] = tail[-40:]
            emit({"event": "log", "message": line[:400]})
            match = WAN_STEP_RE.search(line)
            if match:
                emit({"event": "progress", "job_id": job_id, "phase": "denoise", "step": int(match.group(1)),
                      "steps": int(match.group(2)), "message": line[:200]})
                continue
            for needle, phase in WAN_MARKERS:
                if needle in line:
                    emit({"event": "progress", "job_id": job_id, "phase": phase, "message": line[:200]})
                    break
        code = child.wait()
    finally:
        _active_child = None
    if code != 0:
        raise RuntimeError(f"FastMetal script exited with code {code}:\n" + "\n".join(tail[-8:]))
    if not output.exists():
        raise RuntimeError("FastMetal script finished without writing a clip")
    data: dict = {}
    if metrics.exists():
        try:
            data = json.loads(metrics.read_text())
        finally:
            metrics.unlink(missing_ok=True)
    steps = max(1, int(params.get("num_steps", 3)))
    denoise = float(data.get("mlx_denoise_s", data.get("denoise_s", 0.0)) or 0.0)
    peak_gib = float(data.get("peak_gib") or 0.0) or float(data.get("mlx_denoise_peak_bytes", 0)) / 2**30
    return {
        "video_path": str(output),
        "wall_s": round(float(data.get("total_s", data.get("wall_total_s", time.perf_counter() - started))), 2),
        "timings_s": {"condition_s": round(float(data.get("prompt_encode_s", 0.0)), 2),
                      "dit_load_s": round(float(data.get("mlx_dit_load_s", data.get("dit_load_s", 0.0))), 2),
                      "denoise_s": round(denoise, 2),
                      "video_decode_s": round(float(data.get("vae_decode_export_s", data.get("decode_s", 0.0))), 2),
                      "rife_s": round(float(data.get("rife_interpolate_s", data.get("rife_s", 0.0)) or 0.0), 2)},
        "peak_memory_gib": {"denoise_gib": round(peak_gib, 2),
                            "load_gib": round(float(data.get("mlx_load_peak_bytes", 0)) / 2**30, 2)},
        "step_times_s": [round(denoise / steps, 2)] * steps if denoise else [],
        "vsa": {},
    }


def main() -> None:
    handler = ProtocolLogHandler()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    for name in ("fastvideo", "fastvideo.mlx_runtime"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if handler not in logger.handlers:
            logger.addHandler(handler)
    try:
        import mlx.core  # noqa: F401
        import fastvideo.mlx_runtime  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        emit({"event": "error", "error": f"MLX runtime import failed: {exc}"})
        sys.exit(1)
    emit({"event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = command.get("cmd")
        if cmd == "ping":
            emit({"event": "pong"})
        elif cmd == "generate":
            job_id = command.get("job_id")
            handler.job_id = job_id
            try:
                if command["load"].get("engine") == "wan":
                    result = run_generate_wan(job_id, command["load"], command["params"])
                else:
                    result = run_generate(job_id, command["load"], command["params"], handler)
                emit({"event": "done", "job_id": job_id, "result": result})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                emit({"event": "error", "job_id": job_id, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                handler.job_id = None
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
