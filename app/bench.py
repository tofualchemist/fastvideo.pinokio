"""GPU micro-benchmark (about 15 s): bf16 matmul throughput in MLX, written to data/stats.json.

Gives the cost model a machine constant before any real run. Replaced by fitted values afterwards.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def chip_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "Apple Silicon"


def measure_tflops(seconds: float = 12.0) -> float:
    import mlx.core as mx
    n, k, m = 4096, 5376, 14336        # one FastH3 FFN projection on a 4k-token chunk
    a = mx.random.normal((n, k)).astype(mx.bfloat16)
    w = mx.random.normal((k, m)).astype(mx.bfloat16)
    mx.eval(a, w)
    # warm up
    for _ in range(3):
        mx.eval(a @ w)
    flops_per = 2.0 * n * k * m
    rates = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        mx.eval(a @ w)      # MLX is lazy: each product must be evaluated to be computed
        rates.append(flops_per / (time.perf_counter() - t0) / 1e12)
    rates.sort()
    return rates[len(rates) // 2]


def main() -> None:
    sys.path.insert(0, str(HERE))
    from fasth3_studio.costmodel import CostStore
    data_dir = (HERE / os.environ.get("FASTH3_STUDIO_DATA", "../data")).resolve()
    store = CostStore(data_dir / "stats.json")
    tflops = measure_tflops()
    store.set_benchmark(tflops, chip_name())
    print(f"GPU benchmark: {tflops:.1f} TFLOPS bf16 matmul on {chip_name()} -> {store.path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # Never fail the install over the benchmark: the first real run supplies the constant instead.
        print(f"GPU benchmark skipped: {exc}")
