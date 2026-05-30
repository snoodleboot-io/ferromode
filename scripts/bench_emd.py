#!/usr/bin/env python3
"""
Benchmark EMD decomposition time across signal sizes using the CLI binary.
Referenced from webpage/docs.html.

Note: ferromode_run only exposes EMD. Ensemble methods (EEMD, CEEMDAN, ICEEMDAN)
and VMD are available via the Rust / Python / WASM APIs but not the CLI binary.

Noise: 0.01 × max amplitude of the zero-mean signal.
Boundary: ExtremasMirror (default — Huang 1998, matches PyEMD).

Usage:
    python scripts/bench_emd.py

Requires the release binary:
    cargo build --release -p ferromode_run
"""
import subprocess
import tempfile
import os
import time

import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "..", "target", "release", "ferromode_run")
RUNS   = 5
SIZES  = [500, 1000, 5000, 10000]

print(f"{'Signal length':>14}  {'Median ms':>10}  {'Min ms':>8}  {'Max ms':>8}")
print("-" * 46)

for n in SIZES:
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 4.0, n)
    # Clean multi-frequency signal, zero mean
    sig = (np.sin(2 * np.pi * 32 * t)
         + np.sin(2 * np.pi * 8 * t)
         + np.sin(2 * np.pi * 2 * t))
    # Noise: 0.01 × max amplitude of the zero-mean signal
    noise_std = 0.01 * np.max(np.abs(sig))
    sig = sig + noise_std * rng.standard_normal(n)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        inp = f.name
        for v in sig:
            f.write(f"{v}\n")
    out = inp.replace(".csv", "_out.csv")

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        subprocess.run(
            [BINARY, "--input", inp, "--output", out,
             "--max-imfs", "6", "--sd-only", "--sd-thr", "0.2"],
            capture_output=True,
        )
        times.append((time.perf_counter() - t0) * 1000)
        if os.path.exists(out):
            os.unlink(out)

    os.unlink(inp)
    times.sort()
    median = times[RUNS // 2]
    print(f"{n:>14,}  {median:>10.1f}  {times[0]:>8.1f}  {times[-1]:>8.1f}")
