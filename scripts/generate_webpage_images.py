#!/usr/bin/env python3
"""
Generate all SVG images for the ferromode webpage examples.
Runs the real ferromode binary on each signal and renders
decomposition visualisations in the ferromode dark-metal style.

Usage:
    python scripts/generate_webpage_images.py
"""

import csv
import os
import subprocess
import tempfile

import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "..", "target", "release", "ferromode_run")
OUTDIR = os.path.join(HERE, "..", "webpage", "images")

# ── Palette ──────────────────────────────────────────────────────────────────
IRON      = "#0d0d0d"
IRON_MID  = "#161616"
IRON_DARK = "#111111"
BORDER    = "#3d3835"
RUST      = "#e0521a"
PATINA    = "#3d9e96"
BRONZE    = "#c4973d"
TEXT      = "#f0ede8"
TEXT_MUT  = "#9a9490"
TEXT_DIM  = "#5c5854"
WHITE     = "#f0ede8"

FONT_SANS = "'Space Grotesk', 'Inter', -apple-system, sans-serif"
FONT_MONO = "'Space Mono', 'SF Mono', 'Monaco', monospace"

# IMF colours: signal, imf1, imf2, imf3, imf4, residual
ROW_COLORS = [RUST, "#ff8c42", PATINA, BRONZE, "#52c4bb", "#a0d4d0"]


# ── ferromode runner ──────────────────────────────────────────────────────────

def run_ferromode(signal, max_imfs=4, boundary="extrema"):
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        sig_path = f.name
        for v in signal:
            f.write(f"{v}\n")
    out_path = sig_path.replace(".csv", "_out.csv")
    try:
        r = subprocess.run(
            [BINARY,
             "--input",    sig_path,
             "--output",   out_path,
             "--max-imfs", str(max_imfs + 2),
             "--sd-only",
             "--sd-thr",   "0.2",
             "--boundary", boundary],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ferromode error: {r.stderr[:300]}")
        rows    = list(csv.DictReader(open(out_path)))
        n_imfs  = sum(1 for k in rows[0] if k.startswith("imf"))
        n_out   = min(n_imfs, max_imfs)
        imfs    = [np.array([float(row[f"imf{i}"]) for row in rows]) for i in range(1, n_out + 1)]
        residue = np.array([float(row["residue"]) for row in rows])
        # pad to max_imfs
        while len(imfs) < max_imfs:
            imfs.append(np.zeros(len(signal)))
        return imfs, residue
    finally:
        os.unlink(sig_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


# ── SVG helpers ───────────────────────────────────────────────────────────────

def _polyline(x_pix, y_arr, y_top, y_bot, color, lw=1.4):
    """Render a 1-D array as an SVG polyline clipped to [y_top, y_bot]."""
    n      = len(y_arr)
    if n == 0:
        return ""
    lo, hi = y_arr.min(), y_arr.max()
    span   = hi - lo if hi != lo else 1.0
    h      = y_bot - y_top
    pts    = []
    for i in range(n):
        px = x_pix[0] + (x_pix[1] - x_pix[0]) * i / max(n - 1, 1)
        py = y_bot - (y_arr[i] - lo) / span * h * 0.82 - h * 0.09
        pts.append(f"{px:.2f},{py:.2f}")
    return (f'<polyline points="{" ".join(pts)}" '
            f'stroke="{color}" stroke-width="{lw}" fill="none" '
            f'stroke-linejoin="round" stroke-linecap="round"/>')


def _zero_line(x0, x1, y_top, y_bot, color=BORDER):
    y = (y_top + y_bot) / 2
    return (f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="0.6" stroke-dasharray="4,3"/>')


def _row_box(x0, y_top, w, h, label, color, is_signal=False):
    lines = []
    lines.append(f'<rect x="{x0}" y="{y_top}" width="{w}" height="{h}" fill="{IRON_MID}"/>')
    lines.append(f'<rect x="{x0}" y="{y_top}" width="{w}" height="{h}" '
                 f'fill="none" stroke="{BORDER}" stroke-width="0.6"/>')
    fs    = 9.5 if is_signal else 9
    lines.append(f'<text x="{x0 + 7}" y="{y_top + 13}" '
                 f'font-family="{FONT_SANS}" font-size="{fs}" font-weight="600" '
                 f'fill="{color}" letter-spacing="0.02em">{label}</text>')
    return "\n".join(lines)


def make_decomp_svg(signal, imfs, residue, title, subtitle,
                    x_label, x_vals, outpath, n_rows=None):
    """
    Render signal + IMFs + residue as stacked rows in one SVG.
    x_vals: the physical x-axis values (time, year, …) — used for x tick labels only.
    """
    rows   = [signal] + list(imfs) + [residue]
    labels = ["Signal"] + [f"IMF {i+1}" for i in range(len(imfs))] + ["Residual"]
    colors = ROW_COLORS[:len(rows)]
    if n_rows is None:
        n_rows = len(rows)
    rows   = rows[:n_rows]
    labels = labels[:n_rows]
    colors = colors[:n_rows]

    W      = 1100
    LEFT   = 82
    RIGHT  = W - 16
    PLOT_W = RIGHT - LEFT

    TOP_PAD    = 52
    TITLE_H    = 28
    ROW_H      = 110
    ROW_GAP    = 6
    BOTTOM_PAD = 36

    H = TOP_PAD + TITLE_H + n_rows * (ROW_H + ROW_GAP) + BOTTOM_PAD

    parts = []
    parts.append(f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
                 f'viewBox="0 0 {W} {H}">')
    parts.append(f'<rect width="{W}" height="{H}" fill="{IRON}"/>')

    # Title
    parts.append(f'<text x="{W//2}" y="32" font-family="{FONT_SANS}" font-size="14" '
                 f'font-weight="700" fill="{TEXT}" text-anchor="middle" '
                 f'letter-spacing="-0.3">{title}</text>')
    parts.append(f'<text x="{W//2}" y="48" font-family="{FONT_SANS}" font-size="10" '
                 f'font-weight="400" fill="{TEXT_MUT}" text-anchor="middle">{subtitle}</text>')

    y_cursor = TOP_PAD + TITLE_H + 4

    for ri, (arr, label, color) in enumerate(zip(rows, labels, colors)):
        y_top = y_cursor
        y_bot = y_top + ROW_H

        parts.append(_row_box(LEFT, y_top, PLOT_W, ROW_H, label, color,
                              is_signal=(ri == 0)))
        parts.append(_zero_line(LEFT, RIGHT, y_top, y_bot))
        parts.append(_polyline([LEFT, RIGHT], arr, y_top, y_bot, color,
                                lw=(1.6 if ri == 0 else 1.3)))

        # y-axis label (min/max)
        lo, hi = arr.min(), arr.max()
        parts.append(f'<text x="{LEFT - 4}" y="{y_bot - 3}" font-family="{FONT_MONO}" '
                     f'font-size="8" fill="{TEXT_DIM}" text-anchor="end">{lo:.2g}</text>')
        parts.append(f'<text x="{LEFT - 4}" y="{y_top + 11}" font-family="{FONT_MONO}" '
                     f'font-size="8" fill="{TEXT_DIM}" text-anchor="end">{hi:.2g}</text>')

        # left border tick
        parts.append(f'<line x1="{LEFT - 3}" y1="{y_top}" x2="{LEFT}" y2="{y_top}" '
                     f'stroke="{BORDER}" stroke-width="1"/>')
        parts.append(f'<line x1="{LEFT - 3}" y1="{y_bot}" x2="{LEFT}" y2="{y_bot}" '
                     f'stroke="{BORDER}" stroke-width="1"/>')

        y_cursor += ROW_H + ROW_GAP

    # x-axis ticks (5 ticks)
    n_ticks = 5
    for ti in range(n_ticks + 1):
        frac = ti / n_ticks
        px   = LEFT + frac * PLOT_W
        xv   = x_vals[0] + frac * (x_vals[-1] - x_vals[0])
        py   = y_cursor + 2
        parts.append(f'<line x1="{px:.1f}" y1="{py}" x2="{px:.1f}" y2="{py + 5}" '
                     f'stroke="{BORDER}" stroke-width="1"/>')
        fmt  = ".0f" if abs(xv) >= 10 else ".2f"
        parts.append(f'<text x="{px:.1f}" y="{py + 16}" font-family="{FONT_MONO}" '
                     f'font-size="9" fill="{TEXT_DIM}" text-anchor="middle">'
                     f'{xv:{fmt}}</text>')

    # x-axis label
    parts.append(f'<text x="{W//2}" y="{H - 5}" font-family="{FONT_SANS}" '
                 f'font-size="10" fill="{TEXT_MUT}" text-anchor="middle">{x_label}</text>')

    parts.append("</svg>")
    svg = "\n".join(parts)
    with open(outpath, "w") as f:
        f.write(svg)
    print(f"  Saved: {outpath}")


# ── Signal generators ─────────────────────────────────────────────────────────

def make_ecg(n=1000, fs=250.0):
    """Synthetic ECG: 5 normal beats + 1 PVC, baseline wander, 50 Hz artifact."""
    np.random.seed(7)
    t     = np.arange(n) / fs
    dt    = 1.0 / fs

    def qrs_pulse(t, t0, amp=1.0, pvc=False):
        w  = 0.05 if not pvc else 0.09
        s  = amp * np.exp(-((t - t0) ** 2) / (2 * (w * 0.4) ** 2))
        # P wave
        s += 0.15 * np.exp(-((t - (t0 - 0.10)) ** 2) / (2 * 0.015 ** 2))
        # T wave
        tw = 0.07 if not pvc else 0.12
        s += (0.3 if not pvc else 0.5) * np.exp(-((t - (t0 + 0.20)) ** 2) / (2 * tw ** 2))
        return s

    beat_times = [0.4, 1.0, 1.6, 2.2, 2.9, 3.5]  # beat 4 (idx 3) is PVC
    sig = np.zeros(n)
    for i, bt in enumerate(beat_times):
        pvc = (i == 3)
        sig += qrs_pulse(t, bt, amp=(1.6 if pvc else 1.0), pvc=pvc)

    # baseline wander
    sig += 0.15 * np.sin(2 * np.pi * 0.3 * t)
    # 50 Hz artifact
    sig += 0.08 * np.sin(2 * np.pi * 50.0 * t)
    # EMG noise
    sig += 0.04 * np.random.randn(n)
    return t, sig


def make_seismic(n=2000, fs=100.0):
    """Synthetic seismic: P @5s, S @8s, surface wave @12s."""
    np.random.seed(13)
    t   = np.arange(n) / fs
    sig = np.zeros(n)

    def arrival(t, t0, freq, amp, decay):
        x = t - t0
        env = amp * np.where(x >= 0, np.exp(-decay * x), 0.0)
        return env * np.sin(2 * np.pi * freq * x) * np.where(x >= 0, 1.0, 0.0)

    sig += arrival(t, 5.0, 9.0,  0.4, 2.5)   # P-wave
    sig += arrival(t, 8.0, 3.5,  1.2, 0.8)   # S-wave
    sig += arrival(t, 12.0, 0.4, 2.5, 0.15)  # surface wave
    sig += 0.05 * np.random.randn(n)          # pre-event noise
    return t, sig


def make_speech(n=2000, fs=16000.0):
    """Synthetic VCV /a-t-a/: voiced vowel, closure, burst, vowel release."""
    np.random.seed(19)
    t   = np.arange(n) / fs
    sig = np.zeros(n)

    # First vowel: 0–45 ms  (F0=127, F1=768, F2=1333 Hz)
    vowel_mask1 = t < 0.045
    for f in [127.0, 768.0, 1333.0, 2534.0]:
        amp = 0.5 if f == 127 else (0.3 if f == 768 else 0.15)
        sig += amp * np.sin(2 * np.pi * f * t) * vowel_mask1

    # Closure: 45–65 ms — near silence
    closure = (t >= 0.045) & (t < 0.065)
    sig += 0.01 * np.random.randn(n) * closure

    # Burst: 65–70 ms
    burst = (t >= 0.065) & (t < 0.070)
    sig += 0.8 * np.random.randn(n) * burst

    # Second vowel: 70–125 ms
    vowel_mask2 = t >= 0.070
    for f in [127.0, 768.0, 1333.0, 2534.0]:
        amp = 0.5 if f == 127 else (0.3 if f == 768 else 0.15)
        fade = np.clip((t - 0.070) / 0.015, 0.0, 1.0)  # 15 ms onset ramp
        sig += amp * np.sin(2 * np.pi * f * t) * vowel_mask2 * fade

    sig += 0.02 * np.random.randn(n)
    return t, sig


def make_finance(n=502):
    """GARCH(1,1) synthetic equity: bull → crash → recovery → new highs."""
    np.random.seed(42)
    # GARCH(1,1) parameters calibrated to S&P 500 daily vol
    omega, alpha1, beta1 = 1e-6, 0.09, 0.90
    h   = np.zeros(n)
    ret = np.zeros(n)
    h[0] = omega / (1 - alpha1 - beta1)

    for i in range(1, n):
        eps   = np.random.randn()
        ret[i] = np.sqrt(h[i - 1]) * eps
        h[i]  = omega + alpha1 * ret[i - 1] ** 2 + beta1 * h[i - 1]

    # Bull → crash → recovery regime
    mu = np.zeros(n)
    mu[:200]     =  0.0005   # bull
    mu[200:222]  = -0.0160   # crash (-34% over 22 days)
    mu[222:350]  =  0.0020   # volatile recovery
    mu[350:]     =  0.0006   # new highs

    log_ret = mu + ret
    prices  = 3000.0 * np.exp(np.cumsum(log_ret))
    return np.arange(n, dtype=float), prices - prices.mean()


def load_wolf():
    path = os.path.join(HERE, "data", "sunspots_annual.txt")
    years, values = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                yr = float(parts[0]); val = float(parts[1])
            except ValueError:
                continue
            if int(yr) < 1700 or int(yr) > 1987:
                continue
            years.append(yr)
            values.append(max(val, 0.0))
    years  = np.array(years)
    values = np.array(values)
    return years, values - values.mean()


# ── Main ──────────────────────────────────────────────────────────────────────

os.makedirs(OUTDIR, exist_ok=True)

print("\n── ECG ──────────────────────────────────")
t_ecg, sig_ecg = make_ecg()
imfs_ecg, res_ecg = run_ferromode(sig_ecg, max_imfs=4)
make_decomp_svg(
    sig_ecg, imfs_ecg[:3], res_ecg,
    title="ECG Decomposition · 250 Hz · 1000 samples · 1 PVC event",
    subtitle="Signal: 5 normal beats + PVC (beat 4) + 0.3 Hz baseline wander + 50 Hz artifact",
    x_label="Time (s)", x_vals=t_ecg,
    outpath=os.path.join(OUTDIR, "ecg_decomposition.svg"),
    n_rows=5,
)

print("\n── Seismic ──────────────────────────────────")
t_seis, sig_seis = make_seismic()
imfs_seis, res_seis = run_ferromode(sig_seis, max_imfs=4)
make_decomp_svg(
    sig_seis, imfs_seis[:3], res_seis,
    title="Seismic Decomposition · 100 Hz · 2000 samples · P + S + surface waves",
    subtitle="P-wave @5 s · S-wave @8 s · Rayleigh/Love surface train @12 s",
    x_label="Time (s)", x_vals=t_seis,
    outpath=os.path.join(OUTDIR, "seismic_decomposition.svg"),
    n_rows=5,
)

print("\n── Speech ──────────────────────────────────")
t_sp, sig_sp = make_speech()
imfs_sp, res_sp = run_ferromode(sig_sp, max_imfs=4)
make_decomp_svg(
    sig_sp, imfs_sp[:3], res_sp,
    title="Speech Decomposition · 16 kHz · 2000 samples · /ɑ-t-ɑ/ VCV token",
    subtitle="Voiced /ɑ/ (0–45 ms) · stop closure · burst @65 ms · vowel release (70–125 ms)",
    x_label="Time (ms)", x_vals=t_sp * 1000,
    outpath=os.path.join(OUTDIR, "speech_decomposition.svg"),
    n_rows=5,
)

print("\n── Finance ──────────────────────────────────")
days, sig_fin = make_finance()
imfs_fin, res_fin = run_ferromode(sig_fin, max_imfs=4)
make_decomp_svg(
    sig_fin, imfs_fin[:3], res_fin,
    title="Financial Decomposition · S&P 500 (synthetic) · 502 daily samples · crash @day 200",
    subtitle="Bull (d0–200) · −34% crash (d200–222) · recovery · new highs · GARCH(1,1)",
    x_label="Trading day", x_vals=days,
    outpath=os.path.join(OUTDIR, "finance_decomposition.svg"),
    n_rows=5,
)

print("\n── Sunspots ──────────────────────────────────")
years_w, sig_w = load_wolf()
imfs_sun, res_sun = run_ferromode(sig_w, max_imfs=4, boundary="periodic")
make_decomp_svg(
    sig_w, imfs_sun[:3], res_sun,
    title="Wolf Annual Sunspots 1700–1987 · 288 observations",
    subtitle="Zhao-Huang periodic boundary · ~11-year Schwabe cycle visible in IMF 1",
    x_label="Year", x_vals=years_w,
    outpath=os.path.join(OUTDIR, "sunspots_decomposition.svg"),
    n_rows=5,
)

print("\nDone.")
