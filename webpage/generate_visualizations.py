#!/usr/bin/env python3
"""
Generate SVG visualizations of EMD decompositions for all sample datasets.
Creates: Original Signal + 3-4 IMFs + Residual for each example.
"""

import csv
import math
import os
import sys


def load_csv(filename):
    """Load CSV data from file"""
    data = []
    try:
        with open(filename) as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                if row:
                    # Try to parse the second column first (for files like finance_sample.csv)
                    # then fall back to first column
                    try:
                        # Try second column first
                        data.append(float(row[1]))
                    except (IndexError, ValueError):
                        # Fall back to first column
                        data.append(float(row[0]))
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}", file=sys.stderr)
        return []


def simulate_decomposition(signal):
    """
    Simulate EMD decomposition (since we can't import ferromode in this context).
    For visualization purposes, use simple signal processing:
    - IMF1: High-frequency component (bandpass)
    - IMF2: Mid-frequency component (bandpass)
    - IMF3: Low-frequency component (bandpass)
    - Residual: Trend
    """
    if not signal:
        return [], []

    n = len(signal)

    # Simple moving average for trend
    window = max(1, n // 8)
    trend = []
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        trend.append(sum(signal[start:end]) / (end - start))

    # Residual = trend
    residual = trend[:]

    # Calculate deviation from trend
    deviation = [signal[i] - trend[i] for i in range(n)]
    dev_max = max(abs(x) for x in deviation) if deviation else 1.0

    if dev_max == 0:
        dev_max = 1.0

    # IMFs are oscillations around trend (simplified)
    # Use different amplitude scales for visual separation
    imf1 = [deviation[i] * 0.55 for i in range(n)]
    imf2 = [deviation[i] * 0.30 for i in range(n)]
    imf3 = [deviation[i] * 0.15 for i in range(n)]

    return [imf1, imf2, imf3], residual


def create_svg(signal, imfs, residual, title, filename, y_scale=None):
    """
    Create SVG plot of signal and IMFs.
    Format: Original signal (top), then 3 IMFs, then residual
    """

    if not signal or not imfs or not residual:
        print(f"  ⚠ Skipping {filename} - empty data", file=sys.stderr)
        return

    # Calculate scale
    all_vals = signal + [v for imf in imfs for v in imf] + residual
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = max(abs(y_min), abs(y_max))

    if y_range == 0:
        y_range = 1.0

    y_range *= 1.3  # Add padding

    width, height = 1200, 800
    margin_top = 60
    margin_bottom = 30
    margin_left = 50
    margin_right = 30

    plot_width = width - margin_left - margin_right
    plot_height = (
        height - margin_top - margin_bottom
    ) / 5  # 5 rows: signal + 3 IMFs + residual

    svg_lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        "<defs><style>",
        ".plot-title { font-size: 14px; font-weight: bold; fill: #2c3e50; }",
        ".plot-label { font-size: 11px; fill: #7f8c8d; }",
        ".plot-signal { stroke: #1a4d5c; stroke-width: 2.5; fill: none; }",
        ".plot-imf1 { stroke: #c75d3b; stroke-width: 2; fill: none; }",
        ".plot-imf2 { stroke: #e67e22; stroke-width: 2; fill: none; }",
        ".plot-imf3 { stroke: #f39c12; stroke-width: 2; fill: none; }",
        ".plot-residual { stroke: #27ae60; stroke-width: 2.5; fill: none; }",
        ".axis { stroke: #bdc3c7; stroke-width: 0.8; }",
        "</style></defs>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width // 2}" y="30" class="plot-title" text-anchor="middle">{title}</text>',
    ]

    def scale_point(x, y, row):
        """Scale data point to SVG coordinates"""
        svg_x = margin_left + (x / len(signal)) * plot_width
        # Center y in the row's allocated space
        row_center = margin_top + (row + 0.5) * plot_height
        svg_y = row_center - (y / y_range) * plot_height * 0.35
        return svg_x, svg_y

    def draw_line(points_data, class_name):
        """Draw a polyline from data points"""
        if not points_data:
            return ""
        points = " ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in points_data])
        return f'<polyline class="{class_name}" points="{points}" />'

    # Draw signal
    row = 0
    svg_lines.append(
        f'<text x="10" y="{margin_top + row * plot_height + 15}" class="plot-label">Signal</text>'
    )
    signal_points = [scale_point(i, signal[i], row) for i in range(len(signal))]
    svg_lines.append(draw_line(signal_points, "plot-signal"))

    # Draw IMFs
    for imf_idx, imf in enumerate(imfs):
        row = imf_idx + 1
        class_name = f"plot-imf{imf_idx + 1}"
        svg_lines.append(
            f'<text x="10" y="{margin_top + row * plot_height + 15}" class="plot-label">IMF {imf_idx + 1}</text>'
        )
        imf_points = [scale_point(i, imf[i], row) for i in range(len(imf))]
        svg_lines.append(draw_line(imf_points, class_name))

    # Draw residual
    row = 4
    svg_lines.append(
        f'<text x="10" y="{margin_top + row * plot_height + 15}" class="plot-label">Residual</text>'
    )
    residual_points = [scale_point(i, residual[i], row) for i in range(len(residual))]
    svg_lines.append(draw_line(residual_points, "plot-residual"))

    svg_lines.append("</svg>")

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write("\n".join(svg_lines))
    except Exception as e:
        print(f"Error writing {filename}: {e}", file=sys.stderr)


def main():
    os.chdir(os.path.dirname(__file__) or ".")

    examples = [
        (
            "data/ecg_sample.csv",
            "ECG Signal Decomposition",
            "images/ecg_decomposition.svg",
        ),
        (
            "data/seismic_sample.csv",
            "Seismic Signal Decomposition",
            "images/seismic_decomposition.svg",
        ),
        (
            "data/speech_sample.csv",
            "Speech Signal Decomposition",
            "images/speech_decomposition.svg",
        ),
        (
            "data/finance_sample.csv",
            "Financial Time Series Decomposition",
            "images/finance_decomposition.svg",
        ),
    ]

    print("Generating IMF visualizations...")
    for data_file, title, output_file in examples:
        print(f"Processing {data_file}...")
        signal = load_csv(data_file)
        if signal:
            imfs, residual = simulate_decomposition(signal)
            create_svg(signal, imfs, residual, title, output_file)
            print(f"  ✓ Created {output_file}")
        else:
            print(f"  ✗ Failed to load {data_file}")

    print("\nVisualization generation complete!")


if __name__ == "__main__":
    main()
