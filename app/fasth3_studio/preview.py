"""Per-step previews for the MLX runner.

After every DiT forward, the clean-image estimate ``x0 = x_t + sigma * v`` of the
target video rows is unpatchified and decoded with the tiny TAEH3 decoder, and a
small JPEG of the middle frame (or a strip, if ``frame_count`` > 1) is sent to the parent. The hook
wraps the DiT class method for the duration of one generation and never touches
the FastVideo checkout. Any failure disables previews for the rest of the job;
generation itself is unaffected.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from collections.abc import Callable

import numpy as np

log = logging.getLogger("fasth3_studio.preview")


class PreviewHook:

    def __init__(self, pipeline, emit: Callable[[dict], None], job_id: str, *, frame_count: int = 1,
                 frame_width: int = 416, dtype: str = "fp16", jpeg_quality: int = 72):
        self.pipeline = pipeline
        self.emit = emit
        self.job_id = job_id
        self.frame_count = max(1, frame_count)
        self.frame_width = frame_width
        self.dtype = dtype
        self.jpeg_quality = jpeg_quality
        self.enabled = True
        self._installed = False

    # -- install / uninstall ----------------------------------------------------------------

    def install(self) -> None:
        from fastvideo.mlx_runtime.minimax_h3 import MLXMiniMaxH3DiT

        original = getattr(MLXMiniMaxH3DiT, "_fasth3_studio_original_forward", None) \
            or MLXMiniMaxH3DiT.forward_with_cache
        MLXMiniMaxH3DiT._fasth3_studio_original_forward = original
        hook = self

        def forward_with_cache(dit_self, video_rows, audio_rows, text_rows, *, layout, step_timesteps,
                               row_timestep_inverse, step_index=0, **kwargs):
            out = original(dit_self, video_rows, audio_rows, text_rows, layout=layout, step_timesteps=step_timesteps,
                           row_timestep_inverse=row_timestep_inverse, step_index=step_index, **kwargs)
            if hook.enabled:
                try:
                    hook.after_step(video_rows, out[0], layout, step_timesteps, row_timestep_inverse, step_index)
                except Exception as exc:  # noqa: BLE001 - previews must never break generation
                    hook.enabled = False
                    log.warning("Previews disabled for this job: %s: %s", type(exc).__name__, exc)
            return out

        MLXMiniMaxH3DiT.forward_with_cache = forward_with_cache
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        from fastvideo.mlx_runtime.minimax_h3 import MLXMiniMaxH3DiT

        original = getattr(MLXMiniMaxH3DiT, "_fasth3_studio_original_forward", None)
        if original is not None:
            MLXMiniMaxH3DiT.forward_with_cache = original
        self._installed = False

    # -- the preview itself -----------------------------------------------------------------

    def after_step(self, video_rows, video_velocity, layout, step_timesteps, row_timestep_inverse,
                   step_index: int) -> None:
        import mlx.core as mx
        from fastvideo.mlx_runtime.minimax_h3 import unpatchify_video_tokens
        from fastvideo.mlx_runtime.minimax_h3_taeh3 import decode_latents_taeh3_mlx

        started = time.perf_counter()
        n_cond = int(layout.num_condition_video_rows)
        first_target_row = int(np.asarray(layout.video_indices)[n_cond])
        timestep = float(np.asarray(step_timesteps)[int(np.asarray(row_timestep_inverse)[first_target_row])])
        sigma = 1.0 - timestep
        x0 = video_rows[n_cond:].astype(mx.float32) + sigma * video_velocity[n_cond:].astype(mx.float32)
        mx.eval(x0)
        rows = np.asarray(x0, dtype=np.float32)
        latents = unpatchify_video_tokens(rows, int(layout.num_video_latent_frames), int(layout.latent_height),
                                          int(layout.latent_width), int(self.pipeline._dit_in_channels),
                                          tuple(self.pipeline._dit_patch_size))
        pixels = decode_latents_taeh3_mlx(latents, dtype=self.dtype, chunk_size=5)  # (1, T, H, W, 3) in [0, 1]
        total = int(pixels.shape[1])
        if self.frame_count <= 1 or total <= 1:
            picks = [total // 2]
        else:
            picks = sorted({0, total // 2, total - 1})[:self.frame_count]
        strip = self._strip([pixels[0, i] for i in picks])
        del x0, rows, latents, pixels
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        cost = time.perf_counter() - started
        self.emit({"event": "preview", "job_id": self.job_id, "step": int(step_index) + 1, "sigma": round(sigma, 4),
                   "cost_s": round(cost, 2), "jpeg_b64": strip})

    def _strip(self, frames: list[np.ndarray]) -> str:
        from PIL import Image

        images = []
        for frame in frames:
            arr = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            image = Image.fromarray(arr)
            height = max(1, round(image.height * self.frame_width / image.width))
            images.append(image.resize((self.frame_width, height), Image.BILINEAR))
        width = sum(i.width for i in images)
        height = max(i.height for i in images)
        canvas = Image.new("RGB", (width, height))
        x = 0
        for image in images:
            canvas.paste(image, (x, 0))
            x += image.width
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
