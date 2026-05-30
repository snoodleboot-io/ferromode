#!/usr/bin/env python3
"""
EMD Gut Check — PyEMD vs Ferromode side-by-side on MIT-BIH record 100.

Downloads record 100 (360 Hz, 10 s = 3600 samples, MLII lead),
decomposes with both PyEMD and the ferromode Rust binary,
then produces a side-by-side SVG showing the first 6 IMFs + residual.

Usage:
    python gut_check_compare.py
    python gut_check_compare.py --duration 5   # shorter segment
"""

import argparse
import csv
import math
import os
import subprocess
import sys

import numpy as np
import wfdb
from PyEMD import EMD


FS         = 360
RECORD     = "100"
DB         = "mitdb"
HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(HERE, "data", "mitdb")
SIGNAL_CSV = os.path.join(DATA_DIR, "signal.csv")
FERRO_CSV  = os.path.join(DATA_DIR, "ferromode_imfs.csv")
BINARY     = os.path.join(HERE, "..", "target", "release", "ferromode_run")
OUT_SVG    = os.path.join(HERE, "gut_check_compare.svg")
MAX_IMFS   = 6   # display this many IMFs (+ residual)


# ── Colours ──────────────────────────────────────────────────────────────────
IRON       = "#0d0d0d"
IRON_MID   = "#161616"
IRON_LIGHT = "#1e1e1e"
BORDER     = "#2a2826"
BORDER_HOT = "#3d3835"
RUST       = "#e0521a"
PATINA     = "#3d9e96"
BRONZE     = "#c4973d"
TEXT       = "#f0ede8"
TEXT_MUTED = "#9a9490"
TEXT_DIM   = "#5c5854"
IMF_COLORS = [RUST, "#ff6b2b", PATINA, BRONZE, "#52c4bb", "#e8c87d", "#c47a52"]
MONO       = "'Space Mono','SF Mono',monospace"


# ── Helpers ───────────────────────────────────────────────────────────────────

def dominant_freq(arr: np.ndarray) -> float:
    n    = len(arr)
    fft  = np.abs(np.fft.rfft(arr * np.hanning(n)))
    freq = np.fft.rfftfreq(n, d=1.0 / FS)
    return float(freq[np.argmax(fft)])


def energy_pct(arr: np.ndarray, signal: np.ndarray) -> float:
    tot = np.sum(signal ** 2)
    return float(np.sum(arr ** 2) / tot * 100) if tot > 0 else 0.0


def normalised_rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    """RMS of (a-b) normalised by RMS of a.  0 = identical."""
    denom = np.sqrt(np.mean(a ** 2))
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom) if denom > 0 else 0.0


# ── Data acquisition ──────────────────────────────────────────────────────────

def load_mitdb(n_samples: int) -> np.ndarray:
    os.makedirs(DATA_DIR, exist_ok=True)
    rec_path = os.path.join(DATA_DIR, RECORD)
    if not os.path.exists(rec_path + ".dat"):
        print("Downloading MIT-BIH record 100…")
        wfdb.dl_database(DB, dl_dir=DATA_DIR, records=[RECORD])
    rec    = wfdb.rdrecord(rec_path, sampfrom=0, sampto=n_samples, channels=[0])
    signal = rec.p_signal[:, 0].astype(float)
    signal -= signal.mean()
    return signal


# ── PyEMD decomposition ───────────────────────────────────────────────────────

def run_pyemd(signal: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    emd = EMD()
    emd.MAX_ITERATION = 1000
    all_imfs = emd.emd(signal, max_imf=MAX_IMFS + 2)
    imfs     = [all_imfs[i] for i in range(min(MAX_IMFS, len(all_imfs) - 1))]
    residual = all_imfs[-1]
    return imfs, residual


# ── Ferromode decomposition ───────────────────────────────────────────────────

def run_ferromode(signal: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    # Write signal to CSV
    with open(SIGNAL_CSV, "w") as f:
        for v in signal:
            f.write(f"{v:.10f}\n")

    binary = os.path.normpath(BINARY)
    if not os.path.exists(binary):
        print(f"ERROR: ferromode_run binary not found at {binary}")
        print("  Run: cargo build --release -p ferromode_run")
        sys.exit(1)

    result = subprocess.run(
        [binary, "--input", SIGNAL_CSV, "--output", FERRO_CSV,
         "--max-imfs", str(MAX_IMFS + 2), "--sd-only", "--sd-thr", "0.2"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("ferromode_run failed:")
        print(result.stderr)
        sys.exit(1)
    print(result.stderr.strip())

    # Read output CSV (header: residue, imf1, imf2, ...)
    imf_cols: dict[str, list[float]] = {}
    with open(FERRO_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                imf_cols.setdefault(k, []).append(float(v))

    residual = np.array(imf_cols["residue"])
    imfs = []
    for i in range(1, MAX_IMFS + 1):
        key = f"imf{i}"
        if key in imf_cols:
            imfs.append(np.array(imf_cols[key]))

    return imfs, residual


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(signal, py_imfs, py_res, fe_imfs, fe_res):
    n = min(len(py_imfs), len(fe_imfs))
    print(f"\n{'':>4}  {'PyEMD dom.Hz':>13}  {'Ferro dom.Hz':>13}  "
          f"{'PyEMD E%':>9}  {'Ferro E%':>9}  {'NRMS diff':>10}")
    print("─" * 70)
    for i in range(n):
        pf = dominant_freq(py_imfs[i])
        ff = dominant_freq(fe_imfs[i])
        pe = energy_pct(py_imfs[i], signal)
        fe = energy_pct(fe_imfs[i], signal)
        nd = normalised_rms_diff(py_imfs[i], fe_imfs[i])
        print(f"IMF{i+1:>2}  {pf:>13.1f}  {ff:>13.1f}  {pe:>8.1f}%  {fe:>8.1f}%  {nd:>10.4f}")

    pf = dominant_freq(py_res)
    ff = dominant_freq(fe_res)
    pe = energy_pct(py_res, signal)
    fe_e = energy_pct(fe_res, signal)
    nd = normalised_rms_diff(py_res, fe_res)
    print(f"{'Res':>5}  {pf:>13.1f}  {ff:>13.1f}  {pe:>8.1f}%  {fe_e:>8.1f}%  {nd:>10.4f}")


# ── SVG side-by-side plot ─────────────────────────────────────────────────────

def save_svg(signal, py_imfs, py_res, fe_imfs, fe_res):
    row_labels = ["Signal"] + [f"IMF {i+1}" for i in range(MAX_IMFS)] + ["Residual"]
    py_rows    = [signal] + py_imfs[:MAX_IMFS] + [py_res]
    fe_rows    = [signal] + fe_imfs[:MAX_IMFS] + [fe_res]
    n_rows     = len(row_labels)

    W, ROW_H   = 1400, 100
    ML, MR     = 60, 20
    MT, MB     = 52, 30
    GAP        = 10   # gap between the two panels
    PANEL_W    = (W - ML - MR - GAP) // 2
    H          = MT + n_rows * ROW_H + MB
    N          = len(signal)

    lines = [
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{IRON}"/>',
        # Panel headers
        f'<text x="{ML + PANEL_W//2}" y="22" font-family="{MONO}" font-size="11" '
        f'font-weight="700" fill="{RUST}" text-anchor="middle">PyEMD  ·  SD-threshold stop  ·  mirror  ·  cubic</text>',
        f'<text x="{ML + PANEL_W + GAP + PANEL_W//2}" y="22" font-family="{MONO}" font-size="11" '
        f'font-weight="700" fill="{PATINA}" text-anchor="middle">Ferromode (Rust)  ·  SD-threshold stop  ·  extrema mirror  ·  not-a-knot</text>',
        f'<text x="{W//2}" y="38" font-family="{MONO}" font-size="9" fill="{TEXT_DIM}" '
        f'text-anchor="middle">MIT-BIH record 100 · MLII · {N} samples @ {FS} Hz</text>',
    ]

    def draw_panel(rows, x_off, panel_color):
        for ri, (row, label) in enumerate(zip(rows, row_labels)):
            yt  = MT + ri * ROW_H
            bg  = IRON_MID if ri % 2 == 0 else IRON_LIGHT
            col = panel_color if ri == 0 else IMF_COLORS[ri % len(IMF_COLORS)]
            rmin, rmax = row.min(), row.max()
            rspan = rmax - rmin if rmax != rmin else 1.0
            lo, hi = rmin - rspan * 0.12, rmax + rspan * 0.12

            def sy(v):
                it = yt + ROW_H * 0.08
                ib = yt + ROW_H * 0.92
                f  = (v - lo) / (hi - lo) if hi != lo else 0.5
                return ib - f * (ib - it)

            lines.append(f'<rect x="{x_off}" y="{yt}" width="{PANEL_W}" height="{ROW_H}" fill="{bg}"/>')

            if lo <= 0 <= hi:
                zy = sy(0)
                lines.append(
                    f'<line x1="{x_off}" y1="{zy:.1f}" x2="{x_off+PANEL_W}" y2="{zy:.1f}" '
                    f'stroke="{BORDER_HOT}" stroke-width="0.6" stroke-dasharray="4,3"/>')

            lines.append(
                f'<line x1="{x_off}" y1="{yt}" x2="{x_off}" y2="{yt+ROW_H}" '
                f'stroke="{BORDER_HOT}" stroke-width="1"/>')

            lines.append(
                f'<text x="{x_off+5}" y="{yt+13}" font-family="{MONO}" font-size="9" '
                f'font-weight="600" fill="{col}">{label}</text>')

            df = dominant_freq(row)
            lines.append(
                f'<text x="{x_off+PANEL_W-4}" y="{yt+13}" font-family="{MONO}" font-size="8" '
                f'fill="{TEXT_DIM}" text-anchor="end">{df:.1f} Hz</text>')

            step = max(1, N // 1500)
            pts  = [(x_off + (i / max(N-1, 1)) * PANEL_W, sy(v))
                    for i, v in enumerate(row[::step])]
            pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            lw = 1.3 if ri == 0 or ri == n_rows - 1 else 0.9
            lines.append(
                f'<polyline points="{pts_str}" stroke="{col}" stroke-width="{lw}" '
                f'fill="none" stroke-linejoin="round"/>')

            lines.append(
                f'<line x1="{x_off}" y1="{yt+ROW_H}" x2="{x_off+PANEL_W}" y2="{yt+ROW_H}" '
                f'stroke="{BORDER}" stroke-width="0.7"/>')

    draw_panel(py_rows, ML,                       RUST)
    draw_panel(fe_rows, ML + PANEL_W + GAP,       PATINA)

    # Vertical divider
    xd = ML + PANEL_W + GAP // 2
    lines.append(
        f'<line x1="{xd}" y1="{MT}" x2="{xd}" y2="{MT + n_rows * ROW_H}" '
        f'stroke="{BORDER_HOT}" stroke-width="1" stroke-dasharray="6,4"/>')

    lines.append("</svg>")
    with open(OUT_SVG, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSVG saved: {OUT_SVG}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=int, default=10)
    args = parser.parse_args()

    n = FS * args.duration
    signal = load_mitdb(n)
    print(f"Signal: {len(signal)} samples, std={signal.std():.4f} mV")

    print("\nRunning PyEMD…")
    py_imfs, py_res = run_pyemd(signal)
    print(f"  → {len(py_imfs)} IMFs extracted")

    print("\nRunning Ferromode (Rust)…")
    fe_imfs, fe_res = run_ferromode(signal)
    print(f"  → {len(fe_imfs)} IMFs extracted")

    print_comparison(signal, py_imfs, py_res, fe_imfs, fe_res)
    save_svg(signal, py_imfs, py_res, fe_imfs, fe_res)


if __name__ == "__main__":
    main()
