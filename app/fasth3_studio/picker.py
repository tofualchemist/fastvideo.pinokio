"""Model picker data: will it run here, how long will it take, what does it cost.

Times come from the cost model: the machine's fitted seconds-per-FLOP applied to each model's
architecture at the picker's reference shape (5 s, 480p, text only). A model that has run here in
exactly that shape shows its measured time.
"""
from __future__ import annotations

import subprocess

from . import registry
from .costmodel import CostStore, estimate
from .models import ModelManager


def chip_name() -> str:
    try:
        raw = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "Apple Silicon"
    return raw or "Apple Silicon"


BASIS_NOTE = {
    "measured": "measured on this Mac",
    "model": "from this model's runs on this Mac",
    "engine": "from runs of a related model on this Mac",
    "machine": "from runs of other models on this Mac",
    "benchmark": "from this Mac's GPU benchmark",
    "none": "no timing data yet",
}


def build(models: ModelManager, costs: CostStore, *, quality: str, need: str, cooldown_s: float = 0.0) -> dict:
    status = models.status()
    memory = status["memory_gib"]
    free_disk = status["free_disk_gb"]

    rows = []
    bases = set()
    for spec in registry.MODELS.values():
        info = status["models"][spec.id]
        width, height = next(iter(spec.canvases.values()))
        frames = spec.frames_for_seconds[min(spec.frames_for_seconds)]
        est, basis = {}, {}
        for q in ("draft", "final"):
            e = estimate(costs, spec, quality=q, frames=frames, width=width, height=height, cooldown_s=cooldown_s)
            est[q] = e["total_s"]
            basis[q] = e["basis"]
            bases.add(e["basis"])
        peak = max(float(spec.peak_gib), float(costs.peak_for(spec.id) or 0.0))
        if peak > memory * 0.9:
            fit, reason = "no", f"needs about {peak:.0f} GiB, this Mac has {memory:.0f}"
        elif est["draft"] is not None and est["draft"] > 600:
            fit, reason = "slow", "GPU at 100 % for the whole render"
        else:
            fit, reason = "well", ""
        needs_disk = spec.download_gb + (spec.disk_gb if spec.needs_conversion else 0) + 5
        if not info["installed"] and free_disk < needs_disk:
            fit, reason = "no", f"needs about {needs_disk:.0f} GB free on disk, {free_disk:.0f} available"
        rows.append({
            **info,
            "fit": fit,
            "fit_reason": reason,
            "est_s": est,
            "basis": basis,
            "measured": {q: basis[q] == "measured" for q in basis},
            "peak_gib": round(peak, 1),
            "convert": spec.needs_conversion,
        })

    # Recommendation: the fastest draft that satisfies the need.
    def ok(row: dict) -> bool:
        if row["fit"] == "no":
            return False
        if need == "refs":
            return row["supports_references"]
        if need == "audio":
            return row["audio"]
        return True

    candidates = [r for r in rows if ok(r)]
    candidates.sort(key=lambda r: r["est_s"]["draft"] if r["est_s"]["draft"] is not None else float("inf"))
    recommended = candidates[0]["id"] if candidates else None
    tag = {"refs": "Needed for references", "audio": "Fastest with audio"}.get(need, "Fastest for drafts")
    # One line for the picker header: the strongest basis in use, measured runs aside.
    order = ["model", "engine", "machine", "benchmark", "none"]
    note = next((BASIS_NOTE[b] for b in order if b in bases), BASIS_NOTE["measured"])
    return {
        "machine": {"chip": chip_name(), "memory_gib": memory, "free_disk_gb": free_disk,
                    "benchmark_tflops": costs.data["machine"].get("tflops"), "note": f"estimates {note}",
                    "runs": len(costs.samples())},
        "quality": quality,
        "need": need,
        "recommended": recommended,
        "recommended_tag": tag,
        "models": rows,
    }
