#!/usr/bin/env python3
"""
Compare all ferromode boundary conditions + PyEMD across four datasets:
  1. Medium article: sin(2π·1·t) + sin(2π·4·t)  @  1000 Hz, 10 s
  2. Synthetic 4-sinusoid: 32+16+8+2 Hz  @  256 Hz, 4 s
  3. Wolf annual sunspots 1700–1987 (288 obs)
  4. SILSO monthly sunspots 1749–2026 (3327 obs)

Per dataset, produces:
  boundary_compare_<name>.png       — full signal, edge regions shaded
  boundary_compare_<name>_zoom.png  — tight zoom on start/end to expose artifacts
"""

import csv
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PyEMD import EMD as PyEMD

# ── Palette ───────────────────────────────────────────────────────────────────
IRON       = "#0d0d0d"
IRON_MID   = "#161616"
BORDER     = "#3d3835"
RUST       = "#e0521a"
PATINA     = "#3d9e96"
BRONZE     = "#c4973d"
TEXT       = "#f0ede8"
TEXT_MUTED = "#9a9490"
TEXT_DIM   = "#5c5854"
IMF_COLORS = [RUST, "#ff6b2b", PATINA, BRONZE, "#52c4bb", "#e8c87d"]

HERE   = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "..", "target", "release", "ferromode_run")

BOUNDARIES = [
    ("pyemd",    "PyEMD\n(reference)",        "#aaaaaa"),
    ("remd",     "R EMD\n(reference)",         "#d4a0e0"),
    ("extrema",  "ExtremasMirror\n(default)",  PATINA),
    ("mirror",   "MirrorEven",                 "#52c4bb"),
    ("periodic", "Zhao-Huang\nMirror+Circular",  BRONZE),
    ("slope",    "Slope",                      "#e8c87d"),
    ("ar",       "AR Model",                   "#c47a52"),
    ("waveform", "WaveformMatching",            "#8ecfc9"),
]

MAX_IMFS = 4


# ── Runners ───────────────────────────────────────────────────────────────────

R_SCRIPT = os.path.join(HERE, "run_r_emd.R")

def run_remd(signal, max_imfs=MAX_IMFS):
    """Run R EMD package on a signal, return (imfs, residue)."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        sig_path = f.name
        for v in signal:
            f.write(f"{v}\n")
    out_path = sig_path.replace(".csv", "_out.csv")
    try:
        r = subprocess.run(
            ["Rscript", R_SCRIPT, sig_path, out_path, str(max_imfs + 2)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            print(f"  [remd] error: {r.stderr[-200:]}")
            return [np.zeros_like(signal)] * max_imfs, signal
        rows = list(csv.DictReader(open(out_path)))
        imfs = [np.array([float(row[f"imf{i}"]) for row in rows])
                for i in range(1, max_imfs + 1) if f"imf{i}" in rows[0]]
        res  = np.array([float(row["residue"]) for row in rows])
        while len(imfs) < max_imfs:
            imfs.append(np.zeros_like(signal))
        return imfs, res
    finally:
        os.unlink(sig_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def run_pyemd(signal, max_imfs=MAX_IMFS):
    emd = PyEMD()
    emd.MAX_ITERATION = 1000
    all_imfs = emd.emd(signal, max_imf=max_imfs + 2)
    imfs    = list(all_imfs[:-1][:max_imfs])
    residue = all_imfs[-1]
    while len(imfs) < max_imfs:
        imfs.append(np.zeros_like(signal))
    return imfs, residue


def run_ferromode(signal, boundary_flag, max_imfs=MAX_IMFS):
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        sig_path = f.name
        for v in signal:
            f.write(f"{v}\n")
    out_path = sig_path.replace(".csv", "_out.csv")
    try:
        r = subprocess.run(
            [BINARY, "--input", sig_path, "--output", out_path,
             "--max-imfs", str(max_imfs + 2), "--sd-only", "--sd-thr", "0.2",
             "--boundary", boundary_flag],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            print(f"    [{boundary_flag}] error: {r.stderr[:200]}")
            return [np.zeros_like(signal)] * max_imfs, signal
        rows = list(csv.DictReader(open(out_path)))
        imfs = [np.array([float(row[f"imf{i}"]) for row in rows])
                for i in range(1, max_imfs + 1) if f"imf{i}" in rows[0]]
        res  = np.array([float(row["residue"]) for row in rows])
        while len(imfs) < max_imfs:
            imfs.append(np.zeros_like(signal))
        return imfs, res
    finally:
        os.unlink(sig_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def dominant_freq(arr, fs):
    n    = len(arr)
    fft  = np.abs(np.fft.rfft(arr * np.hanning(n)))
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    return float(freq[np.argmax(fft)])


# ── Grid plotter ──────────────────────────────────────────────────────────────

def make_grid(x, results, fs, zoom_n, title, xlabel, outfile):
    """
    rows = Signal + IMFs 1–MAX_IMFS + Residual
    cols = boundary strategies (BOUNDARIES order)
    zoom_n: number of samples to show at each end in zoom mode (None = full)
    """
    row_labels = ["Signal"] + [f"IMF {i+1}" for i in range(MAX_IMFS)] + ["Residual"]
    n_rows = len(row_labels)
    n_cols = len(BOUNDARIES)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 2.0 * n_rows),
                             facecolor=IRON, sharey=False)
    fig.suptitle(title, color=TEXT, fontsize=10, fontweight="bold")

    sig = results["_signal"]

    for ci, (flag, label, col) in enumerate(BOUNDARIES):
        imfs, res = results[flag]
        data_rows = [sig] + list(imfs) + [res]

        for ri in range(n_rows):
            ax   = axes[ri, ci]
            arr  = data_rows[ri]
            last = (ri == n_rows - 1)

            ax.set_facecolor(IRON_MID)
            ax.axhline(0, color=BORDER, linewidth=0.4, linestyle="--")

            xi = x[:len(arr)]

            if zoom_n is None:
                ax.plot(xi, arr, color=col, linewidth=0.6, alpha=0.9)
                shade = len(xi) // 20   # shade outermost 5%
                ax.axvspan(xi[0],       xi[min(shade, len(xi)-1)],  alpha=0.12, color=RUST, lw=0)
                ax.axvspan(xi[max(-shade-1, -len(xi))], xi[-1],     alpha=0.12, color=RUST, lw=0)
            else:
                zn   = min(zoom_n, len(arr) // 3)
                xs   = xi[:zn]
                xe   = xi[-zn:]
                gap  = np.array([(xs[-1] + xe[0]) / 2])
                xz   = np.concatenate([xs, gap, xe])
                az   = np.concatenate([arr[:zn], [np.nan], arr[-zn:]])
                ax.plot(xz, az, color=col, linewidth=0.8, alpha=0.9)
                ax.axvline(gap[0], color=BORDER, linewidth=0.8,
                           linestyle=":", alpha=0.6)

            ax.tick_params(axis="both", colors=TEXT_DIM, labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2, prune="both"))
            ax.tick_params(labelbottom=last, labelleft=(ci == 0))
            if last:
                ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=6)

            if ci == 0:
                df  = dominant_freq(arr, fs) if np.any(arr != 0) else 0.0
                lbl = row_labels[ri]
                ax.set_ylabel(
                    f"{lbl}\n{df:.3g}" if ri > 0 else lbl,
                    color=TEXT_MUTED, fontsize=7, labelpad=3,
                )

            if ri == 0:
                ax.set_title(label, color=col, fontsize=8,
                             fontweight="bold", pad=5, loc="center")

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.2, w_pad=0.3)
    fig.patch.set_facecolor(IRON)
    plt.savefig(outfile, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outfile}")


def run_and_plot(name, x, sig, fs, zoom_n, full_title, zoom_title, xlabel):
    print(f"\n── {name} ──────────────────────────────────────")
    results = {"_signal": sig}
    for flag, label, _ in BOUNDARIES:
        if flag == "pyemd":
            imfs, res = run_pyemd(sig)
        elif flag == "remd":
            imfs, res = run_remd(sig)
        else:
            imfs, res = run_ferromode(sig, flag)
        results[flag] = (imfs, res)
        dom = [f"{dominant_freq(m, fs):.3g}" for m in imfs if np.any(m != 0)]
        print(f"  {flag:<12} → dom: {dom}")

    make_grid(x, results, fs, zoom_n=None,
              title=full_title, xlabel=xlabel,
              outfile=os.path.join(HERE, f"boundary_compare_{name}.png"))

    make_grid(x, results, fs, zoom_n=zoom_n,
              title=zoom_title, xlabel=xlabel,
              outfile=os.path.join(HERE, f"boundary_compare_{name}_zoom.png"))


# ── Dataset loaders ───────────────────────────────────────────────────────────

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
    return np.array(years), np.array(values)


def load_silso():
    path = os.path.join(HERE, "data", "sunspots_monthly.txt")
    years, values = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                yr = float(parts[2]); val = float(parts[3])
            except ValueError:
                continue
            years.append(yr)
            values.append(max(val, 0.0))
    return np.array(years), np.array(values)


# ── Main ──────────────────────────────────────────────────────────────────────

# 1. Medium article: 1 Hz + 4 Hz
FS1 = 1000.0
t1  = np.arange(0, 10.0, 1.0 / FS1)
s1  = np.sin(2 * np.pi * 1 * t1) + np.sin(2 * np.pi * 4 * t1)
run_and_plot(
    "medium",
    x=t1, sig=s1, fs=FS1,
    zoom_n=50,  # 50 ms at each end
    full_title="Medium article: sin(2π·1t)+sin(2π·4t)  @  1000 Hz 10 s  ·  all boundary conditions",
    zoom_title="Medium — edge zoom (first/last 50 ms)  ·  left of gap = start  ·  right = end",
    xlabel="Time (s)",
)

# 2. Synthetic 4-sinusoid: 32+16+8+2 Hz
FS2 = 256.0
t2  = np.arange(0, 4.0, 1.0 / FS2)
s2  = sum(np.sin(2 * np.pi * f * t2) for f in [32, 16, 8, 2])
run_and_plot(
    "synthetic",
    x=t2, sig=s2, fs=FS2,
    zoom_n=int(0.05 * FS2),  # 50 ms at each end
    full_title="Synthetic: 32+16+8+2 Hz  @  256 Hz 4 s  ·  all boundary conditions",
    zoom_title="Synthetic — edge zoom (first/last 50 ms)",
    xlabel="Time (s)",
)

# 3. Wolf annual sunspots
years_w, sig_w = load_wolf()
sig_w_dm = sig_w - sig_w.mean()
run_and_plot(
    "wolf",
    x=years_w, sig=sig_w_dm, fs=1.0,
    zoom_n=15,  # 15 years at each end
    full_title="Wolf annual sunspots 1700–1987  ·  all boundary conditions",
    zoom_title="Wolf — edge zoom (first/last 15 years)",
    xlabel="Year",
)

# 4. SILSO monthly sunspots
years_s, sig_s = load_silso()
sig_s_dm = sig_s - sig_s.mean()
run_and_plot(
    "silso",
    x=years_s, sig=sig_s_dm, fs=12.0,
    zoom_n=int(2 * 12),  # 2 years at each end
    full_title="SILSO monthly sunspots 1749–2026  ·  all boundary conditions",
    zoom_title="SILSO — edge zoom (first/last 2 years)",
    xlabel="Year",
)

print("\nDone.")
