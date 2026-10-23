"""Render-time cost model.

A denoise step costs  seconds = s * (F_linear + R * c * F_attention).  The two FLOP counts come from the
model's architecture and the number of tokens in the sequence (linear projections grow with the token
count, attention with its square). R is how much more a FLOP of attention costs than a FLOP of matmul on
this GPU and runtime, and c is the attention kernel's cost for each model relative to dense attention
(the sparse models use a gather-based kernel that is not cheaper than dense at these sizes). s is one
machine constant, seconds per matmul FLOP, fitted from every run this machine has done whatever the clip
length, canvas or reference count, because the token count is recorded with each run; R is refitted too
once runs at different sequence lengths exist. Before any run, s comes from a GPU micro-benchmark.
Encode, load, decode and interpolation are recorded per model and scaled by what they depend on.
There are no per-length, per-canvas or per-chip multipliers anywhere.

Calibration (M1 Max, Oct 2026, nine FastH3 runs from 9k to 24k tokens): the projections ran at the
micro-benchmark's bf16 matmul rate (7.9 vs 7.7 TFLOPS), attention at 1/5.1 of it; V2's kernel cost 1.34x
dense, Trim's 1.6x, Omni's 0.76x.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import threading
import time
from pathlib import Path

# Architecture per model id. attn = the model's attention kernel cost relative to dense attention
# (measured; V1 is dense). spatial = pixels per token side (VAE spatial compression x patch size).
# text = tokens that ride along in the DiT sequence (H3 puts the prompt in the sequence; Wan uses
# cross-attention).
ARCH = {
    "fasth3-v1":      dict(hidden=5376, ffn=14336, layers=50, attn=1.00, spatial=32, text=100, engine="h3"),
    "fasth3-v2":      dict(hidden=5376, ffn=14336, layers=50, attn=1.34, spatial=32, text=100, engine="h3"),
    "fasth3-trim":    dict(hidden=5376, ffn=14336, layers=42, attn=1.60, spatial=32, text=100, engine="h3"),
    "fasth3-omni":    dict(hidden=5376, ffn=14336, layers=50, attn=0.76, spatial=32, text=100, engine="h3"),
    "fastmetal-1.3b": dict(hidden=1536, ffn=8960,  layers=30, attn=1.00, spatial=16, text=0,   engine="wan"),
    "fastmetal-5b":   dict(hidden=3072, ffn=14336, layers=30, attn=1.00, spatial=32, text=0,   engine="wan"),
    "fastmetal-14b":  dict(hidden=5120, ffn=13824, layers=40, attn=1.00, spatial=16, text=0,   engine="wan"),
}

# Priors used until this machine's own runs replace them. BENCH_EFFICIENCY: fraction of the
# micro-benchmark's matmul rate the engine's projections reach (H3 measured at 1.03; Wan assumed the
# same for its bf16 matmuls). ATTENTION_COST: R, the cost of an attention FLOP relative to a matmul FLOP
# (H3 measured; Wan's MLX attention has not been fitted on its own yet, so it starts between that and 1).
BENCH_EFFICIENCY = {"h3": 1.0, "wan": 1.0}
ATTENTION_COST = {"h3": 5.1, "wan": 3.0}

H3_FRAMES_PER_CHUNK, H3_LATENTS_PER_CHUNK = 17, 5
H3_AUDIO_LATENTS_PER_SECOND, H3_AUDIO_CHANNELS, H3_FPS = 40, 2, 24
VISION_TOKENS_PER_REFERENCE = 350   # Qwen3-VL tokens Omni adds to the prompt per reference (one image: 349)

# Log lines the runtime prints that carry the shape a run actually used.
GEOMETRY_RE = re.compile(r"Geometry: output=(\d+)x(\d+)x(\d+) model=(\d+)x(\d+)x(\d+)")
ROWS_RE = re.compile(r"affine dequant\+GEMM engaged \(rows=(\d+)")
REFS_RE = re.compile(r"Encoded (\d+) reference\(s\): (\d+) video condition rows, (\d+|no) audio condition rows")
WAN_FAST_RE = re.compile(r"\[(?:wan22 )?fast\] generating (\d+) frames")


def meta_from_log_line(line: str) -> dict | None:
    """Shape facts from a runtime log line: the frames and canvas the DiT saw, its sequence length."""
    m = GEOMETRY_RE.search(line)
    if m:
        return {"frames": int(m.group(3)), "model_width": int(m.group(4)), "model_height": int(m.group(5)),
                "frames_denoised": int(m.group(6))}
    m = ROWS_RE.search(line)
    if m:
        return {"tokens": int(m.group(1))}
    m = REFS_RE.search(line)
    if m:
        audio = 0 if m.group(3) == "no" else int(m.group(3))
        return {"ref_rows": int(m.group(2)) + audio, "ref_count": int(m.group(1))}
    m = WAN_FAST_RE.search(line)
    if m:
        return {"frames_denoised": int(m.group(1))}
    return None


# -- geometry ---------------------------------------------------------------------------------


def h3_align_frames(frames: int) -> int:
    while frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        frames += 1
    return frames


def h3_latent_frames(frames: int) -> int:
    frames = h3_align_frames(frames)
    return (frames - H3_LATENTS_PER_CHUNK) // H3_FRAMES_PER_CHUNK * H3_LATENTS_PER_CHUNK + 2


def h3_fast_source_frames(frames: int, factor: int = 2) -> int:
    """Mirrors plan_fast_temporal: the smallest H3-aligned clip RIFE can expand to the target."""
    frames = h3_align_frames(frames)
    return h3_align_frames((frames - 1 + factor - 1) // factor + 1)


def wan_fast_source_frames(frames: int, factor: int = 2, temporal: int = 4) -> int:
    """Mirrors aligned_keyframe_count."""
    intervals = (frames - 1 + factor - 1) // factor
    return (intervals + temporal - 1) // temporal * temporal + 1


def frames_denoised_for(model_id: str, frames: int, fast: bool) -> int:
    if not fast:
        return frames
    return h3_fast_source_frames(frames) if ARCH[model_id]["engine"] == "h3" else wan_fast_source_frames(frames)


def rows_per_frame(model_id: str, width: int, height: int) -> int:
    s = ARCH[model_id]["spatial"]
    return (width // s) * (height // s)


def tokens_for(model_id: str, frames_denoised: int, width: int, height: int, *, audio_frames: int | None = None,
               ref_rows: int = 0, ref_count: int = 0) -> int:
    """Sequence length the DiT processes per step."""
    a = ARCH[model_id]
    per_frame = rows_per_frame(model_id, width, height)
    if a["engine"] == "h3":
        video = h3_latent_frames(frames_denoised) * per_frame
        audio_latents = round((audio_frames or frames_denoised) / H3_FPS * H3_AUDIO_LATENTS_PER_SECOND)
        audio = audio_latents * H3_AUDIO_CHANNELS
        return int(video + audio + a["text"] + ref_rows + ref_count * VISION_TOKENS_PER_REFERENCE)
    latent_frames = (frames_denoised - 1) // 4 + 1
    return int(latent_frames * per_frame)


def flops_per_step(model_id: str, tokens: int) -> tuple[float, float]:
    """(linear FLOPs, dense-equivalent attention FLOPs x this model's kernel cost) for one denoise step."""
    a = ARCH[model_id]
    linear = 2.0 * (4 * a["hidden"] ** 2 + 2 * a["hidden"] * a["ffn"]) * a["layers"] * tokens
    attention = 4.0 * a["hidden"] * a["layers"] * tokens * tokens * a["attn"]
    return linear, attention


def step_seconds(model_id: str, tokens: int, s: float, r: float) -> float:
    linear, attention = flops_per_step(model_id, tokens)
    return s * (linear + r * attention)


def fit(points: list[tuple[str, int, float]], r_prior: float) -> tuple[float, float]:
    """Fit (s, R) to (model, tokens, median step seconds) points in log space.

    s has a closed form for a given R; R is searched on a log grid, and kept at the prior unless there are at
    least four points spanning sequence lengths different enough (30 %) to tell linear from quadratic cost."""
    def s_for(r: float) -> tuple[float, float]:
        # Median in log space: one run slowed by other work on the Mac does not move the estimate.
        logs = [math.log(t / (flops_per_step(m, tok)[0] + r * flops_per_step(m, tok)[1])) for m, tok, t in points]
        med = statistics.median(logs)
        return math.exp(med), sum(abs(x - med) for x in logs)

    tokens = [tok for _, tok, _ in points]
    if len(points) < 4 or max(tokens) < 1.3 * min(tokens):
        return s_for(r_prior)[0], r_prior
    best = (None, None, float("inf"))
    for i in range(0, 41):
        r = r_prior / 3 * (9 ** (i / 40))      # prior/3 .. prior*3: a few noisy runs cannot send R anywhere
        s, err = s_for(r)
        if err < best[2]:
            best = (s, r, err)
    return best[0], best[1]


# -- store ------------------------------------------------------------------------------------


class CostStore:
    """Samples from real runs plus the machine benchmark, persisted in one JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        try:
            data = json.loads(self.path.read_text()) if self.path.exists() else {}
        except json.JSONDecodeError:
            data = {}
        # Only these two keys are the current format; anything else is from an older launcher version.
        self.data = {"samples": list(data.get("samples") or []), "machine": dict(data.get("machine") or {})}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))

    def record(self, sample: dict) -> None:
        with self.lock:
            self.data["samples"].append({**sample, "ts": time.time()})
            self.data["samples"] = self.data["samples"][-400:]
            self._save()

    def set_benchmark(self, tflops: float, chip: str) -> None:
        with self.lock:
            self.data["machine"] = {"tflops": round(float(tflops), 2), "chip": chip, "ts": time.time()}
            self._save()

    def has_benchmark(self) -> bool:
        with self.lock:
            return bool(self.data["machine"].get("tflops"))

    def samples(self, model_id: str | None = None) -> list[dict]:
        with self.lock:
            return [s for s in self.data["samples"] if model_id is None or s.get("model") == model_id]

    def peak_for(self, model_id: str) -> float | None:
        peaks = [s.get("peak_gib") or 0 for s in self.samples(model_id)]
        return max(peaks) if peaks else None

    # -- fitted quantities ----------------------------------------------------------------

    def rate(self, model_id: str) -> tuple[float | None, float, str]:
        """(s, R, basis): basis 'model' (this model ran here), 'engine' (a sibling ran), 'machine' (anything
        ran), 'benchmark' (micro-benchmark only) or 'none'."""
        engine = ARCH[model_id]["engine"]
        points: dict[str, list[tuple[str, int, float]]] = {}
        for smp in self.samples():
            steps = smp.get("step_s") or []
            if len(steps) < 2 or not smp.get("tokens") or smp.get("model") not in ARCH:
                continue
            # Median step: the first includes warm-up, late ones thermal throttling.
            median = statistics.median(steps)
            if median <= 0:
                continue
            points.setdefault(ARCH[smp["model"]]["engine"], []).append((smp["model"], int(smp["tokens"]), median))
        if engine in points:
            s, r = fit(points[engine], ATTENTION_COST[engine])
            basis = "model" if any(m == model_id for m, _, _ in points[engine]) else "engine"
            return s, r, basis
        with self.lock:
            tflops = self.data["machine"].get("tflops")
        if points:
            # Another engine ran: carry its matmul rate over through the efficiency priors.
            other = next(iter(points))
            s_other, _ = fit(points[other], ATTENTION_COST[other])
            return s_other * BENCH_EFFICIENCY[other] / BENCH_EFFICIENCY[engine], ATTENTION_COST[engine], "machine"
        if tflops:
            return 1.0 / (tflops * 1e12 * BENCH_EFFICIENCY[engine]), ATTENTION_COST[engine], "benchmark"
        return None, ATTENTION_COST[engine], "none"

    def phase_stats(self, model_id: str, decoder: str | None = None) -> dict:
        """Medians of the non-step phases for this model, normalized by what they scale with."""
        samples = self.samples(model_id)
        if decoder:
            narrowed = [s for s in samples if s.get("decoder") == decoder]
            samples = narrowed or samples

        def med(key, norm=lambda s: 1.0):
            vals = [s[key] / norm(s) for s in samples if s.get(key) is not None and norm(s)]
            return statistics.median(vals) if vals else None

        return {
            "load_s": med("load_s"),
            "condition_s": med("condition_s"),
            "decode_s_per_mpix_frame": med("decode_s", lambda s: s["frames"] * s["width"] * s["height"] / 1e6),
            "decoder": samples[0].get("decoder") if samples else None,   # which decoder the decode median is from
            "rife_s_per_frame": med("rife_s", lambda s: s["frames"] if s.get("rife_s") else 0),
            "n": len(samples),
        }

    def exact(self, model_id: str, **shape) -> float | None:
        """Median wall time of runs with exactly this shape, if any."""
        walls = [s["wall_s"] for s in self.samples(model_id)
                 if s.get("wall_s") and all(s.get(k) == v for k, v in shape.items())]
        return statistics.median(walls) if walls else None


# Phases nobody has measured on this machine yet. Seconds on the M1 Max reference, scaled by the
# fitted matmul rate relative to that machine, so a faster Mac gets proportionally shorter phases.
REFERENCE_SECONDS_PER_FLOP = {"h3": 1 / 7.9e12, "wan": 1 / 7.9e12}
# decode: seconds per megapixel-frame with the draft decoder (H3 8-layer NVFP4 decoder 1.0, Wan TAEHV 0.6);
# the 36-layer H3 decoder measured 7.3x the 8-layer one.
DEFAULT_PHASES = {
    "h3": dict(load_s=5.0, condition_s=16.0, decode_s_per_mpix_frame=1.0, rife_s_per_frame=0.15,
               quality_decode_factor=7.3),
    "wan": dict(load_s=5.0, condition_s=18.0, decode_s_per_mpix_frame=0.6, rife_s_per_frame=0.1,
                quality_decode_factor=1.0),
}


def estimate(store: CostStore, spec, *, quality: str, frames: int, width: int, height: int,
             refs: int = 0, ref_clips: int = 0, fast: bool | None = None, fast_spatial: bool = False,
             decoder: str | None = None, cooldown_s: float = 0.0, breakdown: bool = False) -> dict:
    """{'total_s', 'basis', 'phases'} for a job shape. basis 'measured' when this exact shape has run here
    (then phases is empty, unless breakdown=True asks for the modelled phases regardless)."""
    engine = ARCH[spec.id]["engine"]
    if fast is None:
        fast = quality == "draft" and spec.supports_fast_modes and not refs
    model_w, model_h = (width // 2, height // 2) if fast_spatial else (width, height)
    shape = dict(quality=quality, frames=frames, width=width, height=height, refs=bool(refs), fast=bool(fast),
                 fast_spatial=bool(fast_spatial), decoder=decoder or ("fast" if quality == "draft" else "quality"),
                 cooldown_s=float(cooldown_s))
    exact = None if breakdown else store.exact(spec.id, **shape)
    if exact is not None:
        return {"total_s": round(exact), "basis": "measured", "phases": {}}
    s, r, basis = store.rate(spec.id)
    if s is None:
        return {"total_s": None, "basis": "none", "phases": {}}
    frames_denoised = frames_denoised_for(spec.id, frames, fast)
    per_frame = rows_per_frame(spec.id, model_w, model_h)
    ref_rows = (refs - ref_clips) * per_frame + ref_clips * h3_latent_frames(frames) * per_frame if engine == "h3" else 0
    tokens = tokens_for(spec.id, frames_denoised, model_w, model_h, audio_frames=frames, ref_rows=ref_rows,
                        ref_count=refs if engine == "h3" else 0)
    step_s = step_seconds(spec.id, tokens, s, r)
    steps = spec.steps
    wanted = decoder or ("fast" if quality == "draft" else "quality")
    ph = store.phase_stats(spec.id, wanted)
    d = DEFAULT_PHASES[engine]
    speed = min(1.0, s / REFERENCE_SECONDS_PER_FLOP[engine])   # <1 on machines faster than the reference
    load = ph["load_s"] if ph["load_s"] is not None else d["load_s"] * speed
    cond = ph["condition_s"] if ph["condition_s"] is not None else d["condition_s"] * speed
    if ph["decode_s_per_mpix_frame"] is not None:
        decode = ph["decode_s_per_mpix_frame"] * frames * width * height / 1e6
        if ph["decoder"] and ph["decoder"] != wanted:   # the other decoder ran here, not this one
            decode *= d["quality_decode_factor"] if wanted == "quality" else 1.0 / d["quality_decode_factor"]
    else:
        decode = d["decode_s_per_mpix_frame"] * speed * frames * width * height / 1e6
        if wanted == "quality":
            decode *= d["quality_decode_factor"]
    rife_rate = ph["rife_s_per_frame"] if ph["rife_s_per_frame"] is not None else d["rife_s_per_frame"] * speed
    rife = rife_rate * frames if frames_denoised < frames else 0.0
    total = load + cond + steps * step_s + max(0, steps - 1) * cooldown_s + decode + rife
    return {
        "total_s": round(total),
        "basis": basis,
        "phases": {"load_s": round(load, 1), "condition_s": round(cond, 1), "step_s": round(step_s, 1), "steps": steps,
                   "cooldown_s": cooldown_s, "decode_s": round(decode, 1), "rife_s": round(rife, 1), "tokens": tokens,
                   "frames_denoised": frames_denoised},
    }


def sample_from_result(turn: dict, spec, result: dict, meta: dict, settings: dict) -> dict | None:
    """Build one stored sample from a finished run, its runtime log facts and the settings it ran with."""
    timings = result.get("timings_s", {})
    steps = [float(x) for x in (result.get("step_times_s") or [])]
    if not steps:
        return None
    fast = bool(turn.get("fast"))
    frames = int(turn["num_frames"])
    frames_denoised = int(meta.get("frames_denoised") or frames_denoised_for(spec.id, frames, fast))
    model_w = int(meta.get("model_width") or (turn["width"] // 2 if turn.get("fast_spatial") else turn["width"]))
    model_h = int(meta.get("model_height") or (turn["height"] // 2 if turn.get("fast_spatial") else turn["height"]))
    computed = tokens_for(spec.id, frames_denoised, model_w, model_h, audio_frames=frames,
                          ref_rows=int(meta.get("ref_rows") or 0), ref_count=int(meta.get("ref_count") or 0))
    logged = int(meta.get("tokens") or 0)
    # The runtime logs the row count of its first wide GEMM; take it only when it is clearly the DiT sequence.
    tokens = logged if logged and abs(logged - computed) <= 0.2 * computed else computed
    cooldown = float(settings.get("cooldown_s") or 0.0)
    denoise = float(timings.get("denoise_s") or 0.0)
    if timings.get("dit_load_s") is not None:
        load = float(timings["dit_load_s"])
    else:
        load = denoise - sum(steps) - cooldown * max(0, len(steps) - 1) if denoise else None
    decode = sum(float(timings.get(k) or 0.0) for k in ("video_decode_s", "audio_decode_s", "mux_s", "spatial_upsample_s"))
    peaks = result.get("peak_memory_gib") or {}
    return {
        "model": spec.id, "quality": turn.get("quality", "final"), "refs": bool(turn.get("refs")),
        "decoder": turn.get("decoder") or ("fast" if turn.get("quality") == "draft" else "quality"),
        "fast": fast, "fast_spatial": bool(turn.get("fast_spatial")),
        "frames": frames, "frames_denoised": frames_denoised, "width": int(turn["width"]), "height": int(turn["height"]),
        "model_width": model_w, "model_height": model_h, "tokens": tokens,
        "step_s": [round(x, 2) for x in steps],
        "condition_s": float(timings["condition_s"]) if "condition_s" in timings else None,
        "load_s": round(max(0.0, load), 2) if load is not None else None,
        "decode_s": round(decode, 2) if decode else None,
        "rife_s": float(timings["rife_s"]) if timings.get("rife_s") else None,
        "wall_s": float(result.get("wall_s") or 0.0) or None,
        "peak_gib": max((float(v) for v in peaks.values() if v), default=None),
        "cooldown_s": cooldown,
        "attention_chunk_mib": settings.get("attention_chunk_mib"),
        "ops_per_buffer": settings.get("mlx_ops_per_buffer"),
        "previews": bool(settings.get("previews", True)),
    }
