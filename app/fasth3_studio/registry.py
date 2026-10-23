"""Static registry of FastH3 models and the shared components they need.

Everything here is data: Hugging Face repos pinned to revisions, the files each
model needs, how it is built (plain download vs. local INT6 conversion), and the
capabilities the UI should expose for it. The ModelManager turns these entries
into folders under models/; runners read the resulting layout.

Layout produced under <models_dir>:

  components/
    text_encoder/        NVFP4 Qwen3-VL encoder in MLX layout (exported locally once)
    tokenizer/ processor/ audio_vae/
    vae/                 FastH3 NVFP4 decoders (36L quality, 8L fast) + LightVAE encoder + config.json
  <model_id>/
    dit/                 mlx_h3_dit.safetensors + mlx_h3_dit.json (INT6)
    root/                fastvideo_inference.json, scheduler/, audio_scheduler/, transformer*/config.json
                         + symlinks to ../../components/* so it is a complete H3 model root
"""
from __future__ import annotations

from dataclasses import dataclass, field

FASTVIDEO_COMMIT = "fa2d1324c60050cd65209c6aaeac470e7fb412f5"

# -- shared components ---------------------------------------------------------------------

QAD_REPO = "FastVideo/FastH3-Omni-NVFP4-QAD"
QAD_REV = "f01e6a23ac6066aa9eb622cf94b6259442e70ec2"
DECODER_REPO = "FastVideo/FastH3-NVFP4-VAE-Decoder"
DECODER_REV = "bd0c248f2a0f57ca643b3df2ac1f375fa55101fb"
LIGHTVAE_REPO = "lightx2v/LightVAE"
LIGHTVAE_REV = "4925ea978147b9de492d7a5fd4f9c6d1854ca1cc"

DECODER_FILES = {
    "quality": "fasth3-quality-36L-decoder-nvfp4.safetensors",
    "fast": "fasth3-fast-8L-decoder-nvfp4.safetensors",
}
ENCODER_FILE = "lightvae-lite-h3-encoder.safetensors"

# Approximate sizes in GB, for the UI only.
COMPONENTS_DOWNLOAD_GB = 19.3   # 16.5 encoder shards + 0.6 audio vae + 1.4 + 0.33 decoders + 0.41 lightvae
COMPONENTS_DISK_GB = 18.2       # exported encoder cache replaces the shards


@dataclass(frozen=True)
class Download:
    repo: str
    revision: str
    patterns: tuple[str, ...]
    dest: str                 # relative to the model (or components) folder
    strip_prefix: str = ""    # remove this leading path from each downloaded file
    rename: tuple[tuple[str, str], ...] = ()   # (downloaded name -> final name) after stripping


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    short: str
    family: str                 # "t2av" / "ref2va" (FastH3, MLX pipeline) or "wan" (FastMetal, script runner)
    build: str                  # "download" (ready-made weights) or "convert" (BF16 -> INT6 locally)
    dit: Download               # where the DiT (or its BF16 source) comes from
    contract: Download | None   # fastvideo_inference.json + scheduler configs (+ transformer config); H3 only
    transformer_dir: str        # "transformer" or "transformer_ref" (name inside the base snapshot)
    steps: int
    vsa: bool
    vsa_sparsity: float | None
    vsa_tile_size: int | None
    supports_references: bool
    supports_fast_modes: bool   # --fast / --fast-spatial
    durations_s: tuple[int, ...]
    download_gb: float
    disk_gb: float
    description: str
    default_vsa_impl: str = "reference"
    audio: bool = True
    fps: int = 24
    frames_for_seconds: dict = field(default_factory=lambda: {5: 124, 10: 243, 15: 362})
    canvases: dict = field(default_factory=lambda: {"480p": (832, 480), "768p": (1344, 768)})
    peak_gib: float = 20.0      # for the picker; replaced by measurements once available
    # Reference end-to-end seconds for a 5 s 480p clip on the lab's M4 Max (36 GB): draft and final.
    ref_s: dict = field(default_factory=lambda: {"draft": 500.0, "final": 925.0})
    output: str = "Video with synced audio · 480p · 5 s at 24 fps"
    extra: dict = field(default_factory=dict)

    @property
    def engine(self) -> str:
        return "wan" if self.family == "wan" else "h3"

    @property
    def needs_conversion(self) -> bool:
        return self.build == "convert"


MODELS: dict[str, ModelSpec] = {}


def _add(spec: ModelSpec) -> ModelSpec:
    MODELS[spec.id] = spec
    return spec


V1_BASE = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
V1_BASE_REV = "f58c2eb023afcad8379de99bebe14c43b23b49b8"
# Note: the Hub repos named *-MLX-INT6/INT8/INT4 for V1 hold only the manifest (the weights file 404s,
# checked Oct 22 2026), so V1 is converted locally like V2.

_add(ModelSpec(
    id="fasth3-v1",
    name="FastH3 V1",
    short="V1 · 4 step",
    family="t2av",
    build="convert",
    dit=Download(V1_BASE, V1_BASE_REV, ("transformer/*",), "src", strip_prefix="transformer/"),
    contract=Download(V1_BASE, V1_BASE_REV,
                      ("fastvideo_inference.json", "scheduler/*", "audio_scheduler/*", "transformer/config.json"),
                      "root"),
    transformer_dir="transformer",
    steps=4,
    vsa=False,
    vsa_sparsity=None,
    vsa_tile_size=None,
    supports_references=False,
    supports_fast_modes=True,
    durations_s=(5, 10, 15),
    download_gb=70.1,
    disk_gb=17.0,
    description="Fastest. Four-step text to video with audio, dense attention. Downloads the 70 GB BF16 "
                "transformer and converts it to INT6 on this Mac (about 17 GB kept).",
    ref_s={"draft": 248.0, "final": 465.0},
    output="Video with synced audio · 480p · 5 s at 24 fps · 4 steps",
))

V2_BASE = "FastVideo/FastVideo-FastH3-8-Step-V2"
V2_BASE_REV = "3da2ddfe1954d9cda4c05b643dc0f26007a655c5"

_add(ModelSpec(
    id="fasth3-v2",
    name="FastH3 V2",
    short="V2 · 8 step",
    family="t2av",
    build="convert",
    dit=Download(V2_BASE, V2_BASE_REV, ("transformer/*",), "src", strip_prefix="transformer/"),
    contract=Download(V2_BASE, V2_BASE_REV,
                      ("fastvideo_inference.json", "scheduler/*", "audio_scheduler/*", "transformer/config.json"),
                      "root"),
    transformer_dir="transformer",
    steps=8,
    vsa=True,
    vsa_sparsity=0.8,
    vsa_tile_size=64,
    supports_references=False,
    supports_fast_modes=True,
    durations_s=(5, 10, 15),
    download_gb=70.1,
    disk_gb=17.0,
    description="Higher quality. Eight-step text to video with audio, sparse attention. Downloads the 70 GB BF16 "
                "transformer and converts it to INT6 on this Mac (about 17 GB kept).",
    ref_s={"draft": 560.0, "final": 1000.0},
    output="Video with synced audio · 480p · 5 s at 24 fps · 8 steps",
))

TRIM_MLX = "FastVideo/FastVideo-FastH3-Trim-8-Step-MLX-INT6"
TRIM_MLX_REV = "dc6ab6cde207c283708a106000f5bcd98d3cf39b"

_add(ModelSpec(
    id="fasth3-trim",
    name="FastH3 Trim",
    short="Trim · 8 step",
    family="t2av",
    build="download",
    dit=Download(TRIM_MLX, TRIM_MLX_REV, ("transformer/*",), "dit", strip_prefix="transformer/"),
    contract=Download(TRIM_MLX, TRIM_MLX_REV,
                      ("fastvideo_inference.json", "scheduler/*", "audio_scheduler/*"), "root"),
    transformer_dir="transformer",
    steps=8,
    vsa=True,
    vsa_sparsity=0.8,
    vsa_tile_size=64,
    supports_references=False,
    supports_fast_modes=True,
    durations_s=(5, 10, 15),
    download_gb=15.4,
    disk_gb=15.4,
    description="Quickest to set up: the only ready-made INT6 download (15 GB). Experimental pruned model, "
                "eight steps with audio. Quality below V2.",
    ref_s={"draft": 500.0, "final": 925.0},
    output="Video with synced audio · 480p · 5 s at 24 fps · pruned H3",
))

OMNI_BASE = "FastVideo/FastVideo-FastH3-Omni-8-Step-V1"
OMNI_BASE_REV = "53e0d26a1691aa473b6262d8196a8feb2c297df0"

_add(ModelSpec(
    id="fasth3-omni",
    name="FastH3 Omni",
    short="Omni · refs",
    family="ref2va",
    build="convert",
    dit=Download(OMNI_BASE, OMNI_BASE_REV, ("transformer_ref/*",), "src", strip_prefix="transformer_ref/"),
    contract=Download(OMNI_BASE, OMNI_BASE_REV,
                      ("fastvideo_inference.json", "scheduler/*", "audio_scheduler/*",
                       "transformer_ref/config.json"), "root"),
    transformer_dir="transformer_ref",
    steps=8,
    vsa=True,
    vsa_sparsity=0.9,
    vsa_tile_size=128,
    supports_references=True,
    supports_fast_modes=False,
    durations_s=(5, 10, 15),
    download_gb=70.1,
    disk_gb=17.0,
    description="Reference to video with audio: prompt plus ordered image, video and voice references. "
                "Downloads the 70 GB BF16 transformer and converts it to INT6 on this Mac (about 17 GB kept).",
    default_vsa_impl="auto",
    ref_s={"draft": 560.0, "final": 2300.0},
    output="Reference image / video / voice → video with audio · 5, 10 or 15 s",
))

# -- FastMetal (Wan 2.1 based, silent video, 3-step DMD, INT8 MLX DiT) ------------------------------
# 1.3B and 14B share byte-identical text_encoder/, tokenizer/ and vae/ files, kept once in components/wan.

FASTMETAL_13B = "FastVideo/FastMetal-1.3B-QAD"
FASTMETAL_13B_REV = "main"
FASTMETAL_14B = "FastVideo/FastMetal-14B-QAD"
FASTMETAL_14B_REV = "main"
WAN_CANVASES = {"480p": (832, 480)}
WAN_CANVASES_HD = {"480p": (832, 480), "720p": (1280, 704)}
WAN_FRAMES = {5: 81}

_add(ModelSpec(
    id="fastmetal-1.3b",
    name="FastMetal 1.3B",
    short="1.3B · silent",
    family="wan",
    build="download",
    dit=Download(FASTMETAL_13B, FASTMETAL_13B_REV,
                 ("mlx_dit.json", "mlx_dit.safetensors", "model_index.json", "scheduler/*"), "root"),
    contract=None,
    transformer_dir="",
    steps=3,
    vsa=False,
    vsa_sparsity=None,
    vsa_tile_size=None,
    supports_references=False,
    supports_fast_modes=True,
    durations_s=(5,),
    download_gb=1.5,
    disk_gb=1.5,
    description="Quick silent video. Wan 2.1 1.3B distilled to 3 steps, 480p, 5 s at 16 fps. Minutes per clip.",
    audio=False,
    fps=16,
    frames_for_seconds=WAN_FRAMES,
    canvases=WAN_CANVASES,
    peak_gib=4.0,
    ref_s={"draft": 45.0, "final": 110.0},
    output="Silent video · 480p · 5 s at 16 fps",
))

_add(ModelSpec(
    id="fastmetal-14b",
    name="FastMetal 14B",
    short="14B · silent",
    family="wan",
    build="download",
    dit=Download(FASTMETAL_14B, FASTMETAL_14B_REV,
                 ("mlx_dit.json", "mlx_dit.safetensors", "model_index.json", "scheduler/*"), "root"),
    contract=None,
    transformer_dir="",
    steps=3,
    vsa=False,
    vsa_sparsity=None,
    vsa_tile_size=None,
    supports_references=False,
    supports_fast_modes=True,
    durations_s=(5,),
    download_gb=15.2,
    disk_gb=15.2,
    description="Best silent video quality in the Wan family. 14B distilled to 3 steps, 480p or 720p, 5 s at 16 fps.",
    audio=False,
    fps=16,
    frames_for_seconds=WAN_FRAMES,
    canvases=WAN_CANVASES_HD,
    peak_gib=22.0,
    ref_s={"draft": 211.0, "final": 602.0},
    output="Silent video · 480p or 720p · 5 s at 16 fps · best Wan quality",
))

FASTMETAL_5B = "FastVideo/FastMetal-5B-QAD"
FASTMETAL_5B_REV = "main"

_add(ModelSpec(
    id="fastmetal-5b",
    name="FastMetal 5B",
    short="5B · silent · 720p",
    family="wan",
    build="download",
    dit=Download(FASTMETAL_5B, FASTMETAL_5B_REV,
                 ("mlx_dit.json", "mlx_dit.safetensors", "model_index.json", "scheduler/*",
                  "text_encoder/*", "tokenizer/*", "vae/*"), "root"),
    contract=None,
    transformer_dir="",
    steps=3,
    vsa=False,
    vsa_sparsity=None,
    vsa_tile_size=None,
    supports_references=False,
    supports_fast_modes=True,
    durations_s=(5,),
    download_gb=19.5,
    disk_gb=19.5,
    description="Silent video at 720p. Wan 2.2 5B distilled to 3 steps, 5 s at 24 fps. Its own text encoder "
                "and VAE (not shared with the other FastMetal models).",
    audio=False,
    fps=24,
    frames_for_seconds={5: 121},
    canvases={"480p": (832, 448), "720p": (1280, 704)},
    peak_gib=9.5,
    ref_s={"draft": 47.0, "final": 151.0},
    output="Silent video · 480p or 720p · 5 s at 24 fps",
    extra={"script": "wan22", "self_contained": True},
))

WAN_COMPONENTS_DOWNLOAD_GB = 11.9   # UMT5 encoder 11.36 + VAE 0.51 + tokenizer

DEFAULT_MODEL = "fasth3-trim"


def component_downloads() -> list[Download]:
    return [
        Download(QAD_REPO, QAD_REV, ("tokenizer/*", "processor/*", "audio_vae/*"), "."),
        Download(QAD_REPO, QAD_REV, ("text_encoder/*",), "text_encoder_src", strip_prefix="text_encoder/"),
        Download(DECODER_REPO, DECODER_REV, ("quality-36L/decoder-nvfp4.safetensors",), "vae",
                 strip_prefix="quality-36L/", rename=(("decoder-nvfp4.safetensors", DECODER_FILES["quality"]),)),
        Download(DECODER_REPO, DECODER_REV, ("fast-8L/decoder-nvfp4.safetensors",), "vae",
                 strip_prefix="fast-8L/", rename=(("decoder-nvfp4.safetensors", DECODER_FILES["fast"]),)),
        Download(LIGHTVAE_REPO, LIGHTVAE_REV, (ENCODER_FILE,), "vae"),
    ]


def wan_component_downloads() -> list[Download]:
    return [Download(FASTMETAL_13B, FASTMETAL_13B_REV, ("text_encoder/*", "tokenizer/*", "vae/*"), ".")]


def model_json(spec: ModelSpec) -> dict:
    return {
        "id": spec.id,
        "name": spec.name,
        "short": spec.short,
        "family": spec.family,
        "engine": spec.engine,
        "audio": spec.audio,
        "fps": spec.fps,
        "canvases": list(spec.canvases.keys()),
        "peak_gib": spec.peak_gib,
        "ref_s": dict(spec.ref_s),
        "output": spec.output,
        "build": spec.build,
        "steps": spec.steps,
        "vsa": spec.vsa,
        "supports_references": spec.supports_references,
        "supports_fast_modes": spec.supports_fast_modes,
        "durations_s": list(spec.durations_s),
        "download_gb": spec.download_gb,
        "disk_gb": spec.disk_gb,
        "description": spec.description,
    }
