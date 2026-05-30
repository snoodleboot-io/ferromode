#!/usr/bin/env python3
"""
Paper validation: compare ferromode EMD output against published examples.

1. SILSO monthly sunspot means, 1749–present
   Reference: arXiv:1812.11526
   Output: paper_validation_sunspots.png

2. Wolf annual sunspot series, 1700–1987 (288 obs)
   Reference: arXiv:1812.11526 (same paper, Wolf series Figure 2)
   Output: paper_validation_wolf.png

3. Synthetic 4-sinusoid signal
   s(t) = sin(2π·32t) + sin(2π·16t) + sin(2π·8t) + sin(2π·2t)
   Reference: arXiv:2106.15319 — Serial-EMD
   Output: paper_validation_synthetic.png

Run from the scripts/ directory.
"""

import csv
import math
import os
import subprocess
import tempfile

from PyEMD import EMD as PyEMD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
IRON       = "#0d0d0d"
IRON_MID   = "#161616"
IRON_LIGHT = "#1e1e1e"
BORDER     = "#3d3835"
RUST       = "#e0521a"
PATINA     = "#3d9e96"
BRONZE     = "#c4973d"
TEXT       = "#f0ede8"
TEXT_MUTED = "#9a9490"
TEXT_DIM   = "#5c5854"
IMF_COLORS = [RUST, "#ff6b2b", PATINA, BRONZE, "#52c4bb", "#e8c87d", "#c47a52", "#8ecfc9"]

HERE    = os.path.dirname(os.path.abspath(__file__))
BINARY  = os.path.join(HERE, "..", "target", "release", "ferromode_run")


def run_pyemd(signal: np.ndarray, max_imfs: int = 8) -> tuple[list[np.ndarray], np.ndarray]:
    """Run PyEMD on a signal, return (imfs, residue)."""
    emd = PyEMD()
    emd.MAX_ITERATION = 1000
    all_imfs = emd.emd(signal, max_imf=max_imfs + 2)
    imfs    = list(all_imfs[:-1][:max_imfs])
    residue = all_imfs[-1]
    return imfs, residue


def run_ferromode(signal: np.ndarray, max_imfs: int = 8) -> tuple[list[np.ndarray], np.ndarray]:
    """Run ferromode_run binary on a signal array, return (imfs, residue)."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        sig_path = f.name
        for v in signal:
            f.write(f"{v}\n")

    out_path = sig_path.replace(".csv", "_out.csv")

    try:
        result = subprocess.run(
            [BINARY, "--input", sig_path, "--output", out_path,
             "--max-imfs", str(max_imfs + 2), "--sd-only", "--sd-thr", "0.2"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print("ferromode_run stderr:", result.stderr[:500])
            return [], signal

        rows = list(csv.DictReader(open(out_path)))
        imfs = []
        for i in range(1, max_imfs + 1):
            key = f"imf{i}"
            if key in rows[0]:
                imfs.append(np.array([float(r[key]) for r in rows]))
        residue = np.array([float(r["residue"]) for r in rows])
        return imfs, residue
    finally:
        os.unlink(sig_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def dominant_freq(arr: np.ndarray, fs: float) -> float:
    n   = len(arr)
    fft = np.abs(np.fft.rfft(arr * np.hanning(n)))
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    return float(freq[np.argmax(fft)])


def plot_sidebyside(
    x: np.ndarray,
    sig: np.ndarray,
    py_imfs: list[np.ndarray],
    py_res: np.ndarray,
    fe_imfs: list[np.ndarray],
    fe_res: np.ndarray,
    fs: float,
    title: str,
    xlabel: str,
    out_path: str,
):
    """Generic side-by-side PyEMD vs Ferromode IMF comparison plot."""
    n_imfs = max(len(py_imfs), len(fe_imfs))
    rows   = ["Signal"] + [f"IMF {i+1}" for i in range(n_imfs)] + ["Residual"]
    n      = len(rows)

    def pad(imfs, res, length):
        """Pad shorter imf list with zeros so both columns have same row count."""
        padded = list(imfs) + [np.zeros(length)] * (n_imfs - len(imfs))
        return padded, res

    py_padded, py_r = pad(py_imfs, py_res, len(sig))
    fe_padded, fe_r = pad(fe_imfs, fe_res, len(sig))

    py_rows = [sig] + py_padded + [py_r]
    fe_rows = [sig] + fe_padded + [fe_r]

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.0 * n), facecolor=IRON, sharey=False)
    fig.suptitle(title, color=TEXT, fontsize=11, fontweight="bold")

    for ri, (label, py_row, fe_row) in enumerate(zip(rows, py_rows, fe_rows)):
        last = (ri == n - 1)
        col  = IMF_COLORS[ri % len(IMF_COLORS)]

        py_len = len(py_row)
        fe_len = len(fe_row)
        min_len = min(py_len, fe_len)

        nrms = float(
            np.sqrt(np.mean((py_row[:min_len] - fe_row[:min_len]) ** 2)) /
            (np.sqrt(np.mean(py_row[:min_len] ** 2)) + 1e-12)
        )

        df_py = dominant_freq(py_row, fs)
        df_fe = dominant_freq(fe_row, fs)

        # PyEMD left
        ax = axes[ri, 0]
        ax.plot(x[:py_len], py_row, color=RUST, linewidth=0.7, alpha=0.9)
        style_ax(ax, f"{label}  ·  {df_py:.3g} {'Hz' if fs > 2 else 'cyc/yr'}", last, xlabel)
        if ri == 0:
            ax.set_title("PyEMD", color=RUST, fontsize=10, fontweight="bold", pad=6, loc="center")

        # Ferromode right
        ax = axes[ri, 1]
        ax.plot(x[:fe_len], fe_row, color=PATINA, linewidth=0.7, alpha=0.9)
        style_ax(ax, f"{label}  ·  {df_fe:.3g} {'Hz' if fs > 2 else 'cyc/yr'}  ·  NRMS {nrms:.3f}", last, xlabel)
        if ri == 0:
            ax.set_title("Ferromode (Rust)", color=PATINA, fontsize=10,
                         fontweight="bold", pad=6, loc="center")

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3, w_pad=1.0)
    fig.patch.set_facecolor(IRON)
    plt.savefig(out_path, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def style_ax(ax, title: str, last: bool = False, xlabel: str = "Sample index"):
    ax.set_facecolor(IRON_MID)
    ax.set_title(title, color=TEXT_MUTED, fontsize=8, pad=3, loc="left",
                 fontfamily="monospace")
    ax.tick_params(axis="both", colors=TEXT_DIM, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
    ax.tick_params(labelbottom=last)
    if last:
        ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=7)
    ax.axhline(0, color=BORDER, linewidth=0.5, linestyle="--")


# ── Case 1: Sunspot data ──────────────────────────────────────────────────────

def load_sunspots() -> tuple[np.ndarray, np.ndarray]:
    """Return (decimal_year, monthly_mean) from SILSO data file."""
    path = os.path.join(HERE, "data", "sunspots_monthly.txt")
    years, values = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                decimal_year = float(parts[2])
                monthly_mean = float(parts[3])
            except ValueError:
                continue
            if monthly_mean < 0:   # -1 = missing
                monthly_mean = 0.0
            years.append(decimal_year)
            values.append(monthly_mean)
    return np.array(years), np.array(values)


def plot_sunspots():
    print("── Sunspot validation ──────────────────────────")
    years, sig = load_sunspots()
    print(f"  Loaded {len(sig)} monthly values, {years[0]:.2f}–{years[-1]:.2f}")

    # Remove mean (EMD works on zero-mean signals)
    sig_dm = sig - sig.mean()

    print("  Running PyEMD…")
    py_imfs, py_res = run_pyemd(sig_dm, max_imfs=8)
    print(f"  PyEMD returned {len(py_imfs)} IMFs")

    print("  Running Ferromode…")
    imfs, residue = run_ferromode(sig_dm, max_imfs=8)
    print(f"  ferromode returned {len(imfs)} IMFs")

    # Sampling rate for sunspot data: 12 samples/year
    FS_SUN = 12.0

    plot_sidebyside(
        years, sig_dm, py_imfs, py_res, imfs, residue,
        fs=FS_SUN,
        title="SILSO Monthly Sunspots 1749–2026  ·  PyEMD vs Ferromode",
        xlabel="Year",
        out_path=os.path.join(HERE, "paper_validation_sunspots_compare.png"),
    )

    # ── Time-domain stacked plot ──
    rows   = ["Signal"] + [f"IMF {i+1}" for i in range(len(imfs))] + ["Residual"]
    data   = [sig_dm] + imfs + [residue]
    n      = len(rows)
    colors = [TEXT] + [IMF_COLORS[i % len(IMF_COLORS)] for i in range(len(imfs))] + [BRONZE]

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0 * n), facecolor=IRON)
    fig.suptitle(
        "SILSO Monthly Sunspot Numbers 1749–2026  ·  Ferromode EMD",
        color=TEXT, fontsize=12, fontweight="bold",
    )

    # Sampling rate for sunspot data: 12 samples/year
    FS_SUN = 12.0

    for ri, (label, arr, col) in enumerate(zip(rows, data, colors)):
        ax   = axes[ri]
        last = (ri == n - 1)
        df   = dominant_freq(arr, FS_SUN)
        period_yr = 1.0 / df if df > 0 else float("inf")
        period_str = f"~{period_yr:.1f} yr" if period_yr < 1000 else "DC"
        ax.plot(years, arr, color=col, linewidth=0.7, alpha=0.9)
        style_ax(ax, f"{label}  ·  dom {df:.3f} cyc/yr ({period_str})", last,
                 xlabel="Year")

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3)
    fig.patch.set_facecolor(IRON)

    out = os.path.join(HERE, "paper_validation_sunspots.png")
    plt.savefig(out, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── PSD plot ──
    from scipy.signal import welch
    nperseg = min(512, len(sig_dm) // 4)

    fig2, axes2 = plt.subplots(n, 1, figsize=(10, 1.8 * n), facecolor=IRON)
    fig2.suptitle(
        "SILSO Sunspot EMD  ·  PSD (Welch)",
        color=TEXT, fontsize=12, fontweight="bold",
    )

    for ri, (label, arr, col) in enumerate(zip(rows, data, colors)):
        ax   = axes2[ri]
        last = (ri == n - 1)
        f, p = welch(arr, fs=FS_SUN, nperseg=nperseg)
        p_db = 10 * np.log10(np.maximum(p, 1e-20))
        dom  = f[np.argmax(p)]
        ax.set_facecolor(IRON_MID)
        ax.fill_between(f, p_db, p_db.min(), alpha=0.25, color=col)
        ax.plot(f, p_db, color=col, linewidth=0.8)
        ax.axvline(dom, color=col, linewidth=0.7, linestyle="--", alpha=0.7)
        period_yr = 1.0 / dom if dom > 0 else float("inf")
        period_str = f"~{period_yr:.1f} yr" if period_yr < 1000 else "DC"
        style_ax(ax, f"{label}  ·  peak {dom:.3f} cyc/yr ({period_str})", last,
                 xlabel="Frequency (cycles/year)")
        ax.set_ylabel("dB/Hz", color=TEXT_DIM, fontsize=6, labelpad=2)

        # Mark the known ~11-year solar cycle on signal row
        if ri == 0:
            ax.axvline(1/11.0, color=BRONZE, linewidth=0.9, linestyle=":",
                       alpha=0.8, label="11-yr cycle")
            ax.legend(fontsize=6, loc="upper right", framealpha=0.2,
                      labelcolor=TEXT_MUTED, facecolor=IRON_MID, edgecolor=BORDER)

        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.tick_params(axis="both", colors=TEXT_DIM, labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3)
    fig2.patch.set_facecolor(IRON)

    out2 = os.path.join(HERE, "paper_validation_sunspots_psd.png")
    plt.savefig(out2, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")


# ── Case 2: Wolf annual sunspot series ───────────────────────────────────────

def load_wolf_annual() -> tuple[np.ndarray, np.ndarray]:
    """Return (year, annual_mean) from SILSO annual file, clipped to 1700–1987."""
    path = os.path.join(HERE, "data", "sunspots_annual.txt")
    years, values = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                decimal_year = float(parts[0])
                annual_mean  = float(parts[1])
            except ValueError:
                continue
            year = int(decimal_year)
            if year < 1700 or year > 1987:
                continue
            if annual_mean < 0:
                annual_mean = 0.0
            years.append(decimal_year)
            values.append(annual_mean)
    return np.array(years), np.array(values)


def plot_wolf():
    print("── Wolf annual sunspot validation ──────────────")
    years, sig = load_wolf_annual()
    print(f"  Loaded {len(sig)} annual values, {years[0]:.1f}–{years[-1]:.1f}")

    sig_dm = sig - sig.mean()

    print("  Running PyEMD…")
    py_imfs, py_res = run_pyemd(sig_dm, max_imfs=8)
    print(f"  PyEMD returned {len(py_imfs)} IMFs")

    print("  Running Ferromode…")
    imfs, residue = run_ferromode(sig_dm, max_imfs=8)
    print(f"  ferromode returned {len(imfs)} IMFs")

    FS_WOLF = 1.0  # 1 sample per year

    plot_sidebyside(
        years, sig_dm, py_imfs, py_res, imfs, residue,
        fs=FS_WOLF,
        title="Wolf Annual Sunspots 1700–1987  ·  PyEMD vs Ferromode  (arXiv:1812.11526)",
        xlabel="Year",
        out_path=os.path.join(HERE, "paper_validation_wolf_compare.png"),
    )

    rows   = ["Signal"] + [f"IMF {i+1}" for i in range(len(imfs))] + ["Residual"]
    data   = [sig_dm] + imfs + [residue]
    n      = len(rows)
    colors = [TEXT] + [IMF_COLORS[i % len(IMF_COLORS)] for i in range(len(imfs))] + [BRONZE]

    # ── Time-domain ──
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0 * n), facecolor=IRON)
    fig.suptitle(
        "Wolf Annual Sunspot Numbers 1700–1987  ·  Ferromode EMD  ·  (arXiv:1812.11526 Fig 2)",
        color=TEXT, fontsize=11, fontweight="bold",
    )

    for ri, (label, arr, col) in enumerate(zip(rows, data, colors)):
        ax   = axes[ri]
        last = (ri == n - 1)
        df   = dominant_freq(arr, FS_WOLF)
        period_yr = 1.0 / df if df > 1e-6 else float("inf")
        period_str = f"~{period_yr:.1f} yr" if period_yr < 500 else "trend"
        ax.plot(years, arr, color=col, linewidth=0.8, alpha=0.9)
        style_ax(ax, f"{label}  ·  dom {df:.4f} cyc/yr ({period_str})", last,
                 xlabel="Year")

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3)
    fig.patch.set_facecolor(IRON)

    out = os.path.join(HERE, "paper_validation_wolf.png")
    plt.savefig(out, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── PSD ──
    from scipy.signal import welch
    nperseg = min(64, len(sig_dm) // 4)

    fig2, axes2 = plt.subplots(n, 1, figsize=(10, 1.8 * n), facecolor=IRON)
    fig2.suptitle(
        "Wolf Annual Sunspot EMD  ·  PSD (Welch)",
        color=TEXT, fontsize=11, fontweight="bold",
    )

    for ri, (label, arr, col) in enumerate(zip(rows, data, colors)):
        ax   = axes2[ri]
        last = (ri == n - 1)
        f, p = welch(arr, fs=FS_WOLF, nperseg=nperseg)
        p_db = 10 * np.log10(np.maximum(p, 1e-20))
        dom  = f[np.argmax(p)]
        ax.set_facecolor(IRON_MID)
        ax.fill_between(f, p_db, p_db.min(), alpha=0.25, color=col)
        ax.plot(f, p_db, color=col, linewidth=0.8)
        ax.axvline(dom, color=col, linewidth=0.7, linestyle="--", alpha=0.7)
        period_yr = 1.0 / dom if dom > 1e-6 else float("inf")
        period_str = f"~{period_yr:.1f} yr" if period_yr < 500 else "trend"
        style_ax(ax, f"{label}  ·  peak {dom:.4f} cyc/yr ({period_str})", last,
                 xlabel="Frequency (cycles/year)")
        ax.set_ylabel("dB/Hz", color=TEXT_DIM, fontsize=6, labelpad=2)

        if ri == 0:
            ax.axvline(1/11.0, color=BRONZE, linewidth=0.9, linestyle=":",
                       alpha=0.8, label="~11 yr")
            ax.axvline(1/22.0, color=PATINA, linewidth=0.9, linestyle=":",
                       alpha=0.8, label="~22 yr (Hale)")
            ax.legend(fontsize=6, loc="upper right", framealpha=0.2,
                      labelcolor=TEXT_MUTED, facecolor=IRON_MID, edgecolor=BORDER)

        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.tick_params(axis="both", colors=TEXT_DIM, labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3)
    fig2.patch.set_facecolor(IRON)

    out2 = os.path.join(HERE, "paper_validation_wolf_psd.png")
    plt.savefig(out2, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # ── Summary table ──
    print("\n  Known solar cycles in Wolf series:")
    print(f"  {'IMF':<10}  {'period (yr)':>12}  {'known cycle':>14}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*14}")
    known = [(11.0, "Schwabe"), (22.0, "Hale"), (87.0, "Gleissberg"), (210.0, "de Vries")]
    for i, arr in enumerate(imfs):
        df = dominant_freq(arr, FS_WOLF)
        period = 1.0 / df if df > 1e-6 else float("inf")
        match = next(((n) for p, n in known if abs(period - p) / p < 0.25), "—")
        print(f"  IMF {i+1:<6}  {period:>12.1f}  {match:>14}")


# ── Case 3: Synthetic 4-sinusoid signal ──────────────────────────────────────

def plot_synthetic():
    print("── Synthetic 4-sinusoid validation ────────────")

    # arXiv:2106.15319 parameters: 32, 16, 8, 2 Hz; fs=256 Hz; duration=4 s
    FS    = 256.0
    T_SEC = 4.0
    N     = int(FS * T_SEC)
    t     = np.arange(N) / FS
    FREQS = [32.0, 16.0, 8.0, 2.0]

    sig = sum(np.sin(2 * np.pi * f * t) for f in FREQS)
    print(f"  Synthetic signal: {N} samples @ {FS} Hz, {T_SEC} s")
    print(f"  Component freqs: {FREQS} Hz")

    print("  Running PyEMD…")
    py_imfs, py_res = run_pyemd(sig, max_imfs=6)
    print(f"  PyEMD returned {len(py_imfs)} IMFs")

    print("  Running Ferromode…")
    imfs, residue = run_ferromode(sig, max_imfs=6)
    print(f"  ferromode returned {len(imfs)} IMFs")

    plot_sidebyside(
        t, sig, py_imfs, py_res, imfs, residue,
        fs=FS,
        title="Synthetic 4-sinusoid  ·  PyEMD vs Ferromode  (arXiv:2106.15319)",
        xlabel="Time (s)",
        out_path=os.path.join(HERE, "paper_validation_synthetic_compare.png"),
    )

    # ── Time-domain plot ──
    rows   = ["Signal"] + [f"IMF {i+1}" for i in range(len(imfs))] + ["Residual"]
    data   = [sig] + imfs + [residue]
    n      = len(rows)
    colors = [TEXT] + [IMF_COLORS[i % len(IMF_COLORS)] for i in range(len(imfs))] + [BRONZE]

    fig, axes = plt.subplots(n, 1, figsize=(13, 2.0 * n), facecolor=IRON)
    fig.suptitle(
        "Synthetic  ·  sin(2π·32t) + sin(2π·16t) + sin(2π·8t) + sin(2π·2t)  ·  Ferromode EMD",
        color=TEXT, fontsize=11, fontweight="bold",
    )

    for ri, (label, arr, col) in enumerate(zip(rows, data, colors)):
        ax   = axes[ri]
        last = (ri == n - 1)
        df   = dominant_freq(arr, FS)
        ax.plot(t, arr, color=col, linewidth=0.7, alpha=0.9)
        style_ax(ax, f"{label}  ·  dom {df:.1f} Hz", last, xlabel="Time (s)")

        # Mark expected frequencies on signal row
        if ri == 0:
            for ef in FREQS:
                ax.axvline(0, alpha=0)   # just to size axes, annotate below

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3)
    fig.patch.set_facecolor(IRON)

    out = os.path.join(HERE, "paper_validation_synthetic.png")
    plt.savefig(out, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── PSD plot ──
    from scipy.signal import welch
    nperseg = min(512, N // 4)

    fig2, axes2 = plt.subplots(n, 1, figsize=(10, 1.8 * n), facecolor=IRON)
    fig2.suptitle(
        "Synthetic 4-sinusoid EMD  ·  PSD (Welch)  ·  Expected: 2, 8, 16, 32 Hz",
        color=TEXT, fontsize=11, fontweight="bold",
    )

    for ri, (label, arr, col) in enumerate(zip(rows, data, colors)):
        ax   = axes2[ri]
        last = (ri == n - 1)
        f, p = welch(arr, fs=FS, nperseg=nperseg)
        p_db = 10 * np.log10(np.maximum(p, 1e-20))
        dom  = f[np.argmax(p)]
        ax.set_facecolor(IRON_MID)
        ax.fill_between(f, p_db, p_db.min(), alpha=0.25, color=col)
        ax.plot(f, p_db, color=col, linewidth=0.8)
        ax.axvline(dom, color=col, linewidth=0.7, linestyle="--", alpha=0.7)
        style_ax(ax, f"{label}  ·  peak {dom:.1f} Hz", last,
                 xlabel="Frequency (Hz)")
        ax.set_ylabel("dB/Hz", color=TEXT_DIM, fontsize=6, labelpad=2)

        # Mark expected component frequencies
        if ri == 0:
            for ef in FREQS:
                ax.axvline(ef, color=BRONZE, linewidth=0.8, linestyle=":",
                           alpha=0.6)

        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.tick_params(axis="both", colors=TEXT_DIM, labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3)
    fig2.patch.set_facecolor(IRON)

    out2 = os.path.join(HERE, "paper_validation_synthetic_psd.png")
    plt.savefig(out2, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # ── Summary table ──
    print("\n  IMF dominant frequencies vs expected:")
    print(f"  {'IMF':<10}  {'dom Hz':>8}  {'expected Hz':>12}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*12}")
    for i, arr in enumerate(imfs):
        df = dominant_freq(arr, FS)
        # closest expected
        closest = min(FREQS, key=lambda f: abs(f - df))
        match = "✓" if abs(df - closest) / closest < 0.15 else "✗"
        print(f"  IMF {i+1:<6}  {df:>8.1f}  {closest:>12.1f}  {match}")


# ── Case 4: Medium article — 1 Hz + 4 Hz toy signal ─────────────────────────

def plot_medium_article():
    print("── Medium article: 1 Hz + 4 Hz toy signal ─────")

    # Article: https://medium.com/data-science/decomposing-signal-using-empirical-mode-decomposition-algorithm-explanation-for-dummy-93a93304c541
    # Signal: sin(2π·1·t) + sin(2π·4·t) — extended to 10 s at 1000 Hz
    FS     = 1000.0
    res    = 1.0 / FS
    t      = np.arange(0, 10.0, res)   # 10000 samples
    sig    = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 4 * t)
    print(f"  Signal: {len(t)} samples @ {FS} Hz, {t[-1]+res:.1f} s")
    print(f"  Components: 1 Hz + 4 Hz")

    print("  Running PyEMD…")
    py_imfs, py_res = run_pyemd(sig, max_imfs=4)
    print(f"  PyEMD returned {len(py_imfs)} IMFs: {[f'{dominant_freq(m, FS):.1f} Hz' for m in py_imfs]}")

    print("  Running Ferromode…")
    fe_imfs, fe_res = run_ferromode(sig, max_imfs=4)
    print(f"  Ferromode returned {len(fe_imfs)} IMFs: {[f'{dominant_freq(m, FS):.1f} Hz' for m in fe_imfs]}")

    plot_sidebyside(
        t, sig, py_imfs, py_res, fe_imfs, fe_res,
        fs=FS,
        title="Medium article: sin(2π·1·t) + sin(2π·4·t)  ·  PyEMD vs Ferromode",
        xlabel="Time (s)",
        out_path=os.path.join(HERE, "paper_validation_medium_compare.png"),
    )

    # ── individual stacked Ferromode plot (mirrors the article figure) ──
    rows   = ["Signal"] + [f"IMF {i+1}" for i in range(len(fe_imfs))] + ["Residual"]
    data   = [sig] + fe_imfs + [fe_res]
    n      = len(rows)
    colors = [TEXT] + [IMF_COLORS[i % len(IMF_COLORS)] for i in range(len(fe_imfs))] + [BRONZE]

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), facecolor=IRON)
    fig.suptitle(
        "Medium article signal  ·  Ferromode EMD  ·  sin(2π·1·t) + sin(2π·4·t)",
        color=TEXT, fontsize=11, fontweight="bold",
    )

    for ri, (label, arr, col) in enumerate(zip(rows, data, colors)):
        ax   = axes[ri]
        last = (ri == n - 1)
        df   = dominant_freq(arr, FS)
        ax.plot(t[:len(arr)], arr, color=col, linewidth=1.0, alpha=0.9)
        style_ax(ax, f"{label}  ·  {df:.1f} Hz", last, xlabel="Time (s)")

        # mark expected component frequencies
        if ri == 0:
            ax.text(0.02, 0.85, "1 Hz + 4 Hz", transform=ax.transAxes,
                    color=TEXT_MUTED, fontsize=7, fontfamily="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3)
    fig.patch.set_facecolor(IRON)
    out = os.path.join(HERE, "paper_validation_medium_ferromode.png")
    plt.savefig(out, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    plot_sunspots()
    print()
    plot_wolf()
    print()
    plot_synthetic()
    print()
    plot_medium_article()
    print("\nDone.")


if __name__ == "__main__":
    main()
