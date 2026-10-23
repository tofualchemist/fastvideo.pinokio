"""Model picker data: will it run here, how long will it take, what does it cost.

Time estimates start from the lab's M4 Max reference timings scaled by a chip factor, and
switch to measured numbers as soon as this machine has run a model. Measured runs also
calibrate the chip factor for the models that have not run yet.
"""
from __future__ import annotations

import re
import statistics
import subprocess

from . import registry
from .jobs import Stats
from .models import ModelManager

# GPU throughput relative to the lab's M4 Max (1.0) for this kind of workload. Rough, and
# replaced by calibration from real runs on the machine.
CHIP_FACTORS = {
    "M1": 0.22, "M1 Pro": 0.3, "M1 Max": 0.45, "M1 Ultra": 0.8,
    "M2": 0.28, "M2 Pro": 0.36, "M2 Max": 0.55, "M2 Ultra": 1.0,
    "M3": 0.32, "M3 Pro": 0.42, "M3 Max": 0.7, "M3 Ultra": 1.3,
    "M4": 0.4, "M4 Pro": 0.55, "M4 Max": 1.0,
    "M5": 0.6, "M5 Pro": 1.0, "M5 Max": 2.0, "M5 Ultra": 3.0,
}


def chip_name() -> str:
    try:
        raw = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "Apple Silicon"
    return raw or "Apple Silicon"


def table_factor(chip: str) -> float | None:
    match = re.search(r"M(\d)(?:\s+(Pro|Max|Ultra))?", chip)
    if not match:
        return None
    key = f"M{match.group(1)}" + (f" {match.group(2)}" if match.group(2) else "")
    return CHIP_FACTORS.get(key)


def measured_total(stats: Stats, model_id: str, quality: str, refs: bool) -> float | None:
    """End-to-end seconds learned on this machine for a 5 s 480p clip of (model, quality), or None."""
    turn = {"model": model_id, "quality": quality, "refs": [1] if refs else [], "duration_s": 5, "canvas": "480p"}
    if not stats.has_measurement(turn):
        return None
    est = stats.estimate(turn)
    steps = registry.MODELS[model_id].steps
    return est["load_s"] + est["condition_s"] + est["step_s"] * steps + est["decode_s"]


def build(models: ModelManager, stats: Stats, *, quality: str, need: str) -> dict:
    status = models.status()
    chip = chip_name()
    memory = status["memory_gib"]
    free_disk = status["free_disk_gb"]
    factor = table_factor(chip)
    factor_source = "table" if factor else "unknown"
    # Calibrate from every model that has run here.
    ratios = []
    for spec in registry.MODELS.values():
        for q in ("draft", "final"):
            measured = measured_total(stats, spec.id, q, refs=False)
            ref = spec.ref_s.get(q)
            if measured and ref:
                ratios.append(ref / measured)
    if ratios:
        factor = statistics.median(ratios)
        factor_source = "calibrated"
    factor = factor or 0.5

    rows = []
    for spec in registry.MODELS.values():
        info = status["models"][spec.id]
        est = {}
        measured = {}
        for q in ("draft", "final"):
            m = measured_total(stats, spec.id, q, refs=False)
            if m:
                est[q] = round(m)
                measured[q] = True
            else:
                est[q] = round(spec.ref_s.get(q, 0) / factor)
                measured[q] = False
        if spec.peak_gib > memory * 0.9:
            fit, reason = "no", f"needs about {spec.peak_gib:.0f} GiB, this Mac has {memory:.0f}"
        elif est["draft"] > 600:
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
            "measured": measured,
            "convert": spec.needs_conversion,
        })

    # Recommendation: the fastest draft that satisfies the need, preferring installed models.
    def ok(row: dict) -> bool:
        if row["fit"] == "no":
            return False
        if need == "refs":
            return row["supports_references"]
        if need == "audio":
            return row["audio"]
        return True

    candidates = [r for r in rows if ok(r)]
    candidates.sort(key=lambda r: r["est_s"]["draft"])   # fastest fit for the need, installed or not
    recommended = candidates[0]["id"] if candidates else None
    tag = {"refs": "Needed for references", "audio": "Fastest with audio"}.get(need, "Fastest for drafts")
    return {
        "machine": {"chip": chip, "memory_gib": memory, "free_disk_gb": free_disk, "factor": round(factor, 2),
                    "factor_source": factor_source},
        "quality": quality,
        "need": need,
        "recommended": recommended,
        "recommended_tag": tag,
        "models": rows,
    }
