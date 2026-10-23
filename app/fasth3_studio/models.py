"""Model manager: downloads, converts and composes FastH3 models on demand.

All work is synchronous and reports through a progress callback; the job queue
runs it on its worker thread so downloads never race with generation.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import registry
from .registry import Download, ModelSpec

Progress = Callable[[str, float | None, str], None]   # (phase, fraction 0..1 or None, message)


class Cancelled(Exception):
    pass


class InstallError(Exception):
    pass


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _rm(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        (length, ) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def machine_memory_gib() -> float:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 2**30
    except Exception:  # noqa: BLE001
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 2**30


def free_disk_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


class ModelManager:

    def __init__(self, models_dir: Path, fastvideo_dir: Path, python: str | None = None, fake: bool = False):
        self.models_dir = Path(models_dir)
        self.components_dir = self.models_dir / "components"
        self.fastvideo_dir = Path(fastvideo_dir)
        self.python = python or sys.executable
        self.fake = fake
        self.lock = threading.Lock()
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # -- status ---------------------------------------------------------------------------

    def model_dir(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def components_installed(self) -> bool:
        return (self.components_dir / ".complete").exists()

    @property
    def wan_components_dir(self) -> Path:
        return self.components_dir / "wan"

    def wan_components_installed(self) -> bool:
        return (self.wan_components_dir / ".complete").exists()

    def model_installed(self, model_id: str) -> bool:
        spec = registry.MODELS[model_id]
        if spec.extra.get("self_contained"):
            shared = True
        else:
            shared = self.wan_components_installed() if spec.engine == "wan" else self.components_installed()
        return (self.model_dir(model_id) / ".complete").exists() and shared

    def status(self) -> dict:
        models = {}
        for spec in registry.MODELS.values():
            d = self.model_dir(spec.id)
            models[spec.id] = {
                **registry.model_json(spec),
                "installed": self.model_installed(spec.id),
                "partial": d.exists() and not (d / ".complete").exists(),
                "size_gb": round(_dir_bytes(d) / 1e9, 1) if d.exists() else 0.0,
            }
        return {
            "wan_components": {"installed": self.wan_components_installed(),
                               "download_gb": registry.WAN_COMPONENTS_DOWNLOAD_GB,
                               "size_gb": round(_dir_bytes(self.wan_components_dir) / 1e9, 1)},
            "components": {
                "installed": self.components_installed(),
                "download_gb": registry.COMPONENTS_DOWNLOAD_GB,
                "disk_gb": registry.COMPONENTS_DISK_GB,
                "size_gb": round(_dir_bytes(self.components_dir) / 1e9, 1),
            },
            "models": models,
            "models_dir": str(self.models_dir),
            "free_disk_gb": round(free_disk_gb(self.models_dir), 1),
            "memory_gib": round(machine_memory_gib(), 1),
        }

    def paths_for(self, model_id: str, decoder: str = "quality") -> dict:
        spec = registry.MODELS[model_id]
        d = self.model_dir(model_id)
        if spec.engine == "wan":
            return {"model_id": model_id, "engine": "wan", "model_root": str(d / "root"), "family": spec.family,
                    "steps": spec.steps, "fps": spec.fps, "script": spec.extra.get("script", "wan21")}
        return {
            "engine": "h3",
            "model_id": model_id,
            "model_root": str(d / "root"),
            "dit": str(d / "dit"),
            "vae_decoder": str(self.components_dir / "vae" / registry.DECODER_FILES[decoder]),
            "vae_encoder": str(self.components_dir / "vae" / registry.ENCODER_FILE),
            "conditioner_dir": str(self.components_dir / "text_encoder"),
            "steps": spec.steps,
            "vsa": spec.vsa,
            "vsa_sparsity": spec.vsa_sparsity,
            "vsa_tile_size": spec.vsa_tile_size,
            "vsa_impl": spec.default_vsa_impl,
            "family": spec.family,
        }

    def remove_model(self, model_id: str) -> None:
        _rm(self.model_dir(model_id))

    def remove_components(self) -> None:
        _rm(self.components_dir)

    # -- install --------------------------------------------------------------------------

    def ensure_model(self, model_id: str, progress: Progress, cancel: threading.Event) -> None:
        spec = registry.MODELS[model_id]
        with self.lock:
            if self.fake:
                self._fake_install(spec, progress, cancel)
                return
            if spec.engine == "wan":
                self._ensure_wan_model(spec, progress, cancel)
                return
            self.ensure_components(progress, cancel)
            if (self.model_dir(model_id) / ".complete").exists():
                self._compose_root(spec)
                return
            d = self.model_dir(model_id)
            d.mkdir(parents=True, exist_ok=True)
            needed = spec.download_gb + (spec.disk_gb if spec.needs_conversion else 0) + 5
            if free_disk_gb(self.models_dir) < needed:
                raise InstallError(f"Not enough free disk for {spec.name}: need about {needed:.0f} GB free.")
            self._download(spec.contract, d, progress, cancel, label=f"{spec.name}: contract")
            if spec.build == "download":
                self._download(spec.dit, d, progress, cancel, label=f"{spec.name}: INT6 weights")
            else:
                src_dir = d / "root" / spec.transformer_dir
                dl = Download(spec.dit.repo, spec.dit.revision, spec.dit.patterns, str(src_dir.relative_to(d)),
                              strip_prefix=spec.dit.strip_prefix)
                if not (d / "dit" / "mlx_h3_dit.safetensors").exists():
                    self._download(dl, d, progress, cancel, label=f"{spec.name}: BF16 transformer (70 GB)")
                    self._convert(spec, src_dir, d, progress, cancel)
                for shard in src_dir.glob("*.safetensors"):
                    shard.unlink()
                for extra in src_dir.glob("*.safetensors.index.json"):
                    extra.unlink()
            self._compose_root(spec)
            (d / ".complete").write_text(json.dumps({"model": spec.id, "fastvideo": registry.FASTVIDEO_COMMIT,
                                                      "time": time.time()}))
            progress("ready", 1.0, f"{spec.name} installed")

    def ensure_components(self, progress: Progress, cancel: threading.Event) -> None:
        if self.components_installed():
            return
        c = self.components_dir
        c.mkdir(parents=True, exist_ok=True)
        if free_disk_gb(self.models_dir) < registry.COMPONENTS_DOWNLOAD_GB + registry.COMPONENTS_DISK_GB:
            raise InstallError("Not enough free disk for the shared H3 components (about 40 GB needed during setup).")
        downloads = registry.component_downloads()
        encoder_done = (c / "text_encoder" / "model.safetensors").exists() and \
            (c / "text_encoder" / "mlx_h3_nvfp4_encoder.json").exists()
        for dl in downloads:
            if dl.dest == "text_encoder_src" and encoder_done:
                continue
            self._download(dl, c, progress, cancel, label="Shared components")
        if not encoder_done:
            self._export_encoder(c, progress, cancel)
        _rm(c / "text_encoder_src")
        self._write_vae_config(c)
        (c / ".complete").write_text(json.dumps({"time": time.time()}))

    # -- FastMetal (Wan) ------------------------------------------------------------------

    def ensure_wan_components(self, progress: Progress, cancel: threading.Event) -> None:
        if self.wan_components_installed():
            return
        c = self.wan_components_dir
        c.mkdir(parents=True, exist_ok=True)
        if free_disk_gb(self.models_dir) < registry.WAN_COMPONENTS_DOWNLOAD_GB + 3:
            raise InstallError("Not enough free disk for the shared Wan components (about 12 GB).")
        for dl in registry.wan_component_downloads():
            self._download(dl, c, progress, cancel, label="Shared Wan components")
        (c / ".complete").write_text(json.dumps({"time": time.time()}))

    def _ensure_wan_model(self, spec: ModelSpec, progress: Progress, cancel: threading.Event) -> None:
        self_contained = bool(spec.extra.get("self_contained"))
        if not self_contained:
            self.ensure_wan_components(progress, cancel)
        d = self.model_dir(spec.id)
        root = d / "root"
        if not (d / ".complete").exists():
            if free_disk_gb(self.models_dir) < spec.download_gb + 3:
                raise InstallError(f"Not enough free disk for {spec.name}: need about {spec.download_gb + 3:.0f} GB.")
            self._download(spec.dit, d, progress, cancel, label=f"{spec.name}: INT8 weights")
        root.mkdir(parents=True, exist_ok=True)
        for name in (() if self_contained else ("text_encoder", "tokenizer", "vae")):
            link = root / name
            target = self.wan_components_dir / name
            if link.is_symlink() or link.exists():
                if link.is_symlink() and os.path.realpath(link) == os.path.realpath(target):
                    continue
                _rm(link)
            link.symlink_to(os.path.relpath(target, root), target_is_directory=True)
        if not (d / ".complete").exists():
            (d / ".complete").write_text(json.dumps({"model": spec.id, "time": time.time()}))
        progress("ready", 1.0, f"{spec.name} installed")

    # -- steps ----------------------------------------------------------------------------

    def _download(self, dl: Download, base: Path, progress: Progress, cancel: threading.Event, label: str) -> None:
        from huggingface_hub import HfApi, snapshot_download
        from fnmatch import fnmatch

        dest = (base / dl.dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        # Expected size from the Hub so we can show a fraction without hooking tqdm.
        info = HfApi().model_info(dl.repo, revision=dl.revision, files_metadata=True)
        wanted = [s for s in info.siblings if any(fnmatch(s.rfilename, p) for p in dl.patterns)]
        total = sum((s.size or 0) for s in wanted) or 1
        # Already there (resume / re-run)?
        present = 0
        renames = dict(dl.rename)
        for s in wanted:
            rel = s.rfilename[len(dl.strip_prefix):] if dl.strip_prefix and s.rfilename.startswith(dl.strip_prefix) \
                else s.rfilename
            rel = renames.get(rel, rel)
            p = dest / rel
            if p.exists() and (s.size is None or p.stat().st_size == s.size):
                present += s.size or 0
        if present >= total and wanted:
            return
        progress("download", 0.0, f"{label}: downloading {total / 1e9:.1f} GB from {dl.repo}")
        staging = dest if not dl.strip_prefix else (base / ".staging" / dl.dest.replace("/", "_")).resolve()
        staging.mkdir(parents=True, exist_ok=True)
        error: list[BaseException] = []

        def work():
            try:
                snapshot_download(dl.repo, revision=dl.revision, allow_patterns=list(dl.patterns),
                                  local_dir=str(staging), max_workers=4)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        while thread.is_alive():
            if cancel.is_set():
                raise Cancelled()
            done = _dir_bytes(staging)
            progress("download", min(done / total, 0.999),
                     f"{label}: {done / 1e9:.1f} / {total / 1e9:.1f} GB")
            thread.join(timeout=1.0)
        if error:
            raise InstallError(f"Download failed from {dl.repo}: {error[0]}")
        if dl.strip_prefix:
            src_root = staging / dl.strip_prefix.rstrip("/")
            for item in src_root.iterdir():
                target = dest / renames.get(item.name, item.name)
                _rm(target)
                shutil.move(str(item), str(target))
            _rm(staging)
        else:
            for old_name, new_name in renames.items():
                if (dest / old_name).exists():
                    _rm(dest / new_name)
                    shutil.move(str(dest / old_name), str(dest / new_name))
        _rm(dest / ".cache")
        progress("download", 1.0, f"{label}: download complete")

    def _run_logged(self, cmd: list[str], progress: Progress, cancel: threading.Event, phase: str, label: str,
                    env: dict | None = None, cwd: Path | None = None) -> None:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env={**os.environ, **(env or {})}, cwd=str(cwd) if cwd else None)
        tail: list[str] = []
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if cancel.is_set():
                proc.kill()
                raise Cancelled()
            line = line.rstrip()
            if line:
                tail.append(line)
                tail[:] = tail[-30:]
                progress(phase, None, f"{label}: {line[-160:]}")
        if proc.returncode != 0:
            raise InstallError(f"{label} failed (exit {proc.returncode}):\n" + "\n".join(tail[-12:]))

    def _convert(self, spec: ModelSpec, src_dir: Path, d: Path, progress: Progress, cancel: threading.Event) -> None:
        if machine_memory_gib() < 24:
            raise InstallError("INT6 conversion needs at least 24 GB of unified memory.")
        build = d / "dit_build"
        _rm(build)
        script = self.fastvideo_dir / "scripts" / "checkpoint_conversion" / "convert_minimax_h3_mlx.py"
        if not script.exists():
            raise InstallError(f"FastVideo checkout is missing {script}")
        progress("convert", None, f"{spec.name}: converting BF16 transformer to INT6 (this takes a while)")
        cmd = [self.python, str(script), "--model-root", str(src_dir), "--out", str(build), "--formats", "int6"]
        if spec.vsa:
            cmd.append("--include-vsa")
        self._run_logged(cmd, progress, cancel, "convert", f"{spec.name}: INT6 conversion",
                         env={"PYTHONPATH": str(self.fastvideo_dir)}, cwd=self.fastvideo_dir)
        out = build / "int6"
        if not (out / "mlx_h3_dit.safetensors").exists():
            raise InstallError("Conversion finished but produced no INT6 checkpoint.")
        _rm(d / "dit")
        shutil.move(str(out), str(d / "dit"))
        _rm(build)

    def _export_encoder(self, c: Path, progress: Progress, cancel: threading.Event) -> None:
        src = c / "text_encoder_src"
        out = c / "text_encoder"
        _rm(out)
        progress("convert", None, "Shared components: packing the NVFP4 text encoder for MLX")
        code = ("import sys\nfrom fastvideo.mlx_runtime.minimax_h3_conditioner import export_mlx_h3_nvfp4_encoder\n"
                "export_mlx_h3_nvfp4_encoder(sys.argv[1], sys.argv[2])\n")
        self._run_logged([self.python, "-c", code, str(src), str(out)], progress, cancel, "convert",
                         "Text encoder export", env={"PYTHONPATH": str(self.fastvideo_dir)})
        manifest = out / "mlx_h3_nvfp4_encoder.json"
        data = json.loads(manifest.read_text())
        data["source_dir"] = f"hf://{registry.QAD_REPO}/text_encoder"
        manifest.write_text(json.dumps(data, indent=2) + "\n")

    def _write_vae_config(self, c: Path) -> None:
        enc = c / "vae" / registry.ENCODER_FILE
        meta = safetensors_header(enc).get("__metadata__") or {}
        config = json.loads(meta["architecture"])["teacher_config"]
        (c / "vae" / "config.json").write_text(json.dumps(config, indent=2))

    def _compose_root(self, spec: ModelSpec) -> None:
        root = self.model_dir(spec.id) / "root"
        root.mkdir(parents=True, exist_ok=True)
        for name in ("text_encoder", "tokenizer", "processor", "audio_vae", "vae"):
            link = root / name
            target = self.components_dir / name
            if link.is_symlink() or link.exists():
                if link.is_symlink() and os.path.realpath(link) == os.path.realpath(target):
                    continue
                _rm(link)
            link.symlink_to(os.path.relpath(target, root), target_is_directory=True)

    # -- fake mode (UI development without weights) --------------------------------------

    def _fake_install(self, spec: ModelSpec, progress: Progress, cancel: threading.Event) -> None:
        d = self.model_dir(spec.id)
        if (d / ".complete").exists():
            return
        d.mkdir(parents=True, exist_ok=True)
        (self.components_dir).mkdir(parents=True, exist_ok=True)
        (self.components_dir / ".complete").write_text("{}")
        steps = 20
        for i in range(steps):
            if cancel.is_set():
                raise Cancelled()
            time.sleep(0.15)
            progress("download", (i + 1) / steps, f"{spec.name}: fake download {i + 1}/{steps}")
        if spec.needs_conversion:
            for i in range(5):
                time.sleep(0.2)
                progress("convert", None, f"{spec.name}: fake conversion step {i + 1}")
        (d / ".complete").write_text("{}")
