#!/usr/bin/env python3
"""
Generate dark-themed SVG visualizations of EMD decompositions.

Creates: Original signal + 3–4 IMFs + Residual for each sample dataset.
Each row is independently scaled with Y-axis tick marks and value labels.

Usage (from the webpage/ directory):
    python generate_visualizations.py

Output: images/{ecg,seismic,speech,finance}_decomposition.svg
"""

import csv
import math
import os
import sys


# ── Palette (matches webpage/css/style.css) ─────────────────────────────────
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

# Line colour per row: [signal, IMF1, IMF2, IMF3, residual]
ROW_COLORS = [RUST, "#ff6b2b", PATINA, BRONZE, "#52c4bb"]

FONT_STACK = "'Space Grotesk', 'Inter', -apple-system, sans-serif"
MONO_STACK = "'Space Mono', 'SF Mono', 'Monaco', monospace"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv(filename: str) -> list[float]:
    """Load second numeric column, skipping comment lines (starting with #) and header."""
    data = []
    try:
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # Skip comment rows and the column-header row
                first = row[0].strip()
                if first.startswith("#") or not first[0].lstrip("-").replace(".", "", 1).isdigit():
                    continue
                # Prefer second column (signal), fall back to first
                for cell in (row[1], row[0]) if len(row) > 1 else (row[0],):
                    try:
                        data.append(float(cell))
                        break
                    except ValueError:
                        continue
    except Exception as e:
        print(f"  Error loading {filename}: {e}", file=sys.stderr)
    return data


# ── Decomposition simulation ─────────────────────────────────────────────────
# Uses a cascade of exponential moving averages to produce genuinely different
# frequency bands — each IMF will oscillate visibly at a distinct time scale.

def _ema(signal: list[float], alpha: float) -> list[float]:
    """Exponential moving average."""
    out = [signal[0]]
    for v in signal[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out

def _subtract(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]

def _add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]

def simulate_decomposition(signal: list[float]) -> tuple[list[list[float]], list[float]]:
    """
    Produce 3 pseudo-IMFs and a residual using EMA cascade.
    Each IMF is genuinely oscillatory at a distinct scale.
    Residual = long-term trend.
    """
    if not signal:
        return [], []

    n = len(signal)

    # Smooth at three scales — fine, medium, coarse
    alpha_fine   = min(0.30, 6.0 / n) if n > 20 else 0.30
    alpha_medium = min(0.08, 1.5 / n) if n > 20 else 0.08
    alpha_coarse = min(0.02, 0.4 / n) if n > 20 else 0.02

    trend_fine   = _ema(signal, alpha_fine)
    trend_medium = _ema(signal, alpha_medium)
    trend_coarse = _ema(signal, alpha_coarse)

    # IMF1 = signal minus fine-scale smooth (highest freq)
    imf1 = _subtract(signal, trend_fine)

    # IMF2 = fine smooth minus medium smooth (mid freq)
    imf2 = _subtract(trend_fine, trend_medium)

    # IMF3 = medium smooth minus coarse smooth (low freq)
    imf3 = _subtract(trend_medium, trend_coarse)

    # Residual = coarse trend
    residual = trend_coarse

    return [imf1, imf2, imf3], residual


# ── SVG helpers ──────────────────────────────────────────────────────────────

def _fmt(v: float, decimals: int = 3) -> str:
    """Format float to limited decimals, strip trailing zeros."""
    s = f"{v:.{decimals}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

def _tick_values(lo: float, hi: float, n_ticks: int = 5) -> list[float]:
    """Nice round tick values spanning [lo, hi]."""
    span = hi - lo
    if span == 0:
        return [lo]
    raw_step = span / (n_ticks - 1)
    mag = 10 ** math.floor(math.log10(abs(raw_step)))
    nice_steps = [mag * m for m in (1, 2, 2.5, 5, 10)]
    step = min(nice_steps, key=lambda s: abs(s - raw_step))
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step * 0.01:
        if lo - step * 0.01 <= t <= hi + step * 0.01:
            ticks.append(round(t, 10))
        t += step
        t = round(t, 10)
    return ticks or [lo, hi]

def polyline(points: list[tuple[float, float]], color: str, width: float) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" stroke="{color}" stroke-width="{width}" fill="none" stroke-linejoin="round" stroke-linecap="round"/>'

def text_el(x: float, y: float, content: str, **attrs) -> str:
    attr_str = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_str}>{content}</text>'


# ── Main SVG builder ─────────────────────────────────────────────────────────

def create_svg(
    signal: list[float],
    imfs: list[list[float]],
    residual: list[float],
    title: str,
    filename: str,
    y_unit: str = "",
) -> None:
    if not signal or not imfs or not residual:
        print(f"  ⚠ Skipping {filename} — empty data", file=sys.stderr)
        return

    rows_data = [signal] + imfs + [residual]
    row_labels = ["Signal"] + [f"IMF {i+1}" for i in range(len(imfs))] + ["Residual"]
    n_rows = len(rows_data)

    # ── Layout ────────────────────────────────────────────────────────────────
    W           = 1100
    ROW_H       = 130
    MARGIN_TOP  = 56
    MARGIN_BOT  = 40
    MARGIN_LEFT = 78      # space for Y-axis labels
    MARGIN_RIGHT= 28
    PLOT_W      = W - MARGIN_LEFT - MARGIN_RIGHT
    H           = MARGIN_TOP + n_rows * ROW_H + MARGIN_BOT
    N           = len(signal)

    lines = [
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}">',
        # Background
        f'<rect width="{W}" height="{H}" fill="{IRON}"/>',
        # Title
        f'<text x="{W//2}" y="32" '
        f'font-family="{FONT_STACK}" font-size="14" font-weight="700" '
        f'fill="{TEXT}" text-anchor="middle" letter-spacing="-0.3">{title}</text>',
    ]

    for row_idx, row_data in enumerate(rows_data):
        row_y_top = MARGIN_TOP + row_idx * ROW_H
        row_center = row_y_top + ROW_H / 2
        label     = row_labels[row_idx]
        color     = ROW_COLORS[row_idx % len(ROW_COLORS)]

        # ── Per-row stats ──────────────────────────────────────────────────────
        rmin = min(row_data)
        rmax = max(row_data)
        rspan = rmax - rmin if rmax != rmin else 1.0
        pad = rspan * 0.12
        lo = rmin - pad
        hi = rmax + pad
        half = (hi - lo) / 2

        def to_svg_y(v: float) -> float:
            # Map [lo, hi] → [row_y_top + ROW_H*0.1, row_y_top + ROW_H*0.9]
            inner_top = row_y_top + ROW_H * 0.1
            inner_bot = row_y_top + ROW_H * 0.9
            frac = (v - lo) / (hi - lo) if (hi != lo) else 0.5
            return inner_bot - frac * (inner_bot - inner_top)

        # Row background (alternating)
        bg = IRON_MID if row_idx % 2 == 0 else IRON_LIGHT
        lines.append(
            f'<rect x="{MARGIN_LEFT}" y="{row_y_top}" '
            f'width="{PLOT_W}" height="{ROW_H}" fill="{bg}"/>'
        )

        # ── Zero line ──────────────────────────────────────────────────────────
        if lo <= 0 <= hi:
            zy = to_svg_y(0)
            lines.append(
                f'<line x1="{MARGIN_LEFT}" y1="{zy:.2f}" '
                f'x2="{MARGIN_LEFT + PLOT_W}" y2="{zy:.2f}" '
                f'stroke="{BORDER_HOT}" stroke-width="0.8" stroke-dasharray="4,4"/>'
            )

        # ── Y-axis ticks ───────────────────────────────────────────────────────
        ticks = _tick_values(lo, hi, n_ticks=5)
        for tv in ticks:
            ty = to_svg_y(tv)
            # Tick line
            lines.append(
                f'<line x1="{MARGIN_LEFT - 5}" y1="{ty:.2f}" '
                f'x2="{MARGIN_LEFT}" y2="{ty:.2f}" '
                f'stroke="{BORDER_HOT}" stroke-width="1"/>'
            )
            # Horizontal grid line (very subtle)
            lines.append(
                f'<line x1="{MARGIN_LEFT}" y1="{ty:.2f}" '
                f'x2="{MARGIN_LEFT + PLOT_W}" y2="{ty:.2f}" '
                f'stroke="{BORDER}" stroke-width="0.5" opacity="0.6"/>'
            )
            # Tick label — right-align against the axis
            decimals = 4 if abs(tv) < 0.1 and tv != 0 else (2 if abs(tv) < 10 else 1)
            tick_label = _fmt(tv, decimals) + (f" {y_unit}" if y_unit and tv == ticks[-1] else "")
            lines.append(
                f'<text x="{MARGIN_LEFT - 8:.2f}" y="{ty + 4:.2f}" '
                f'font-family="{MONO_STACK}" font-size="9" fill="{TEXT_DIM}" '
                f'text-anchor="end">{tick_label}</text>'
            )

        # ── Left axis border ───────────────────────────────────────────────────
        lines.append(
            f'<line x1="{MARGIN_LEFT}" y1="{row_y_top}" '
            f'x2="{MARGIN_LEFT}" y2="{row_y_top + ROW_H}" '
            f'stroke="{BORDER_HOT}" stroke-width="1"/>'
        )

        # ── Row label (top-left inside the panel) ──────────────────────────────
        lines.append(
            f'<text x="{MARGIN_LEFT + 8}" y="{row_y_top + 16}" '
            f'font-family="{FONT_STACK}" font-size="10.5" font-weight="600" '
            f'fill="{color}" letter-spacing="0.02em">{label}</text>'
        )

        # ── Signal path ────────────────────────────────────────────────────────
        pts = [
            (MARGIN_LEFT + (i / max(N - 1, 1)) * PLOT_W, to_svg_y(v))
            for i, v in enumerate(row_data)
        ]
        lw = 1.5 if row_idx == 0 or row_idx == n_rows - 1 else 1.2
        lines.append(polyline(pts, color, lw))

        # ── Row separator ──────────────────────────────────────────────────────
        sep_y = row_y_top + ROW_H
        lines.append(
            f'<line x1="0" y1="{sep_y}" x2="{W}" y2="{sep_y}" '
            f'stroke="{BORDER}" stroke-width="1"/>'
        )

    # ── X-axis labels (sample indices) ────────────────────────────────────────
    n_xticks = 6
    for i in range(n_xticks):
        frac = i / (n_xticks - 1)
        xi = MARGIN_LEFT + frac * PLOT_W
        idx = int(frac * (N - 1))
        bot_y = MARGIN_TOP + n_rows * ROW_H
        lines.append(
            f'<line x1="{xi:.2f}" y1="{bot_y}" x2="{xi:.2f}" y2="{bot_y + 5}" '
            f'stroke="{BORDER_HOT}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{xi:.2f}" y="{bot_y + 18}" '
            f'font-family="{MONO_STACK}" font-size="9" fill="{TEXT_DIM}" '
            f'text-anchor="middle">n={idx}</text>'
        )

    lines.append("</svg>")

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as f:
        f.write("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    examples = [
        (
            "data/ecg_sample.csv",
            "ECG Decomposition · 250 Hz · 5 seconds",
            "images/ecg_decomposition.svg",
            "mV",
        ),
        (
            "data/seismic_sample.csv",
            "Seismic Decomposition · 100 Hz · 10 seconds",
            "images/seismic_decomposition.svg",
            "m/s²",
        ),
        (
            "data/speech_sample.csv",
            "Speech Decomposition · 16 kHz · 125 ms",
            "images/speech_decomposition.svg",
            "amp",
        ),
        (
            "data/finance_sample.csv",
            "Financial Decomposition · S&amp;P 500 · Daily Returns",
            "images/finance_decomposition.svg",
            "",
        ),
    ]

    print("Generating Ferromode IMF visualizations (dark theme)…")
    for data_file, title, output_file, unit in examples:
        print(f"  {data_file} → {output_file}")
        signal = load_csv(data_file)
        if signal:
            imfs, residual = simulate_decomposition(signal)
            create_svg(signal, imfs, residual, title, output_file, y_unit=unit)
            print(f"    ✓ {len(signal)} samples, {len(imfs)} IMFs")
        else:
            print(f"    ✗ No data loaded")

    print("Done.")


if __name__ == "__main__":
    main()
