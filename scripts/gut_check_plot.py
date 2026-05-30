#!/usr/bin/env python3
"""
Generate three PNG comparison plots:
  1. gut_check_sidebyside.png  — side-by-side panel per IMF (PyEMD left, Ferromode right)
  2. gut_check_overlay.png     — all IMFs overlaid on the same axes
  3. gut_check_psd.png         — PSD (Welch) for each IMF, PyEMD vs Ferromode side-by-side

Run from the scripts/ directory after gut_check_compare.py has been executed.
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PyEMD import EMD

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
IMF_COLORS = [RUST, "#ff6b2b", PATINA, BRONZE, "#52c4bb", "#e8c87d", "#c47a52"]

FS      = 360
HERE    = os.path.dirname(os.path.abspath(__file__))
SIG_CSV = os.path.join(HERE, "data", "mitdb", "signal.csv")
FE_CSV  = os.path.join(HERE, "data", "mitdb", "ferromode_imfs.csv")
MAX_IMFS = 6


def dominant_freq(arr):
    n    = len(arr)
    fft  = np.abs(np.fft.rfft(arr * np.hanning(n)))
    freq = np.fft.rfftfreq(n, 1.0 / FS)
    return float(freq[np.argmax(fft)])


def energy_pct(arr, signal):
    tot = np.sum(signal ** 2)
    return float(np.sum(arr ** 2) / tot * 100) if tot > 0 else 0.0


def load_data():
    sig = np.loadtxt(SIG_CSV)

    # PyEMD
    emd = EMD()
    emd.MAX_ITERATION = 1000
    all_imfs = emd.emd(sig, max_imf=MAX_IMFS + 2)
    py_imfs  = [all_imfs[i] for i in range(min(MAX_IMFS, len(all_imfs) - 1))]
    py_res   = all_imfs[-1]

    # Ferromode
    rows     = list(csv.DictReader(open(FE_CSV)))
    fe_imfs  = []
    for i in range(1, MAX_IMFS + 1):
        key = f"imf{i}"
        if key in rows[0]:
            fe_imfs.append(np.array([float(r[key]) for r in rows]))
    fe_res = np.array([float(r["residue"]) for r in rows])

    return sig, py_imfs, py_res, fe_imfs, fe_res


def style_ax(ax, title, color, last=False):
    ax.set_facecolor(IRON_MID)
    ax.set_title(title, color=TEXT_MUTED, fontsize=8, pad=3, loc="left",
                 fontfamily="monospace")
    ax.tick_params(axis="both", colors=TEXT_DIM, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
    ax.tick_params(labelbottom=last)
    if last:
        ax.set_xlabel("Sample index", color=TEXT_DIM, fontsize=7)
    ax.axhline(0, color=BORDER, linewidth=0.5, linestyle="--")


# ── Plot 1: side-by-side ──────────────────────────────────────────────────────

def plot_sidebyside(sig, py_imfs, py_res, fe_imfs, fe_res):
    rows   = ["Signal"] + [f"IMF {i+1}" for i in range(MAX_IMFS)] + ["Residual"]
    py_rows = [sig] + py_imfs[:MAX_IMFS] + [py_res]
    fe_rows = [sig] + fe_imfs[:MAX_IMFS] + [fe_res]
    n = len(rows)
    T = np.arange(len(sig))

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.0 * n),
                             facecolor=IRON, sharey=False)
    fig.suptitle(
        "MIT-BIH Record 100  ·  EMD Gut Check  ·  PyEMD vs Ferromode",
        color=TEXT, fontsize=12, fontweight="bold",
    )

    for ri, (label, py_row, fe_row) in enumerate(zip(rows, py_rows, fe_rows)):
        last = (ri == n - 1)
        col  = IMF_COLORS[ri % len(IMF_COLORS)]

        df_py = dominant_freq(py_row)
        df_fe = dominant_freq(fe_row)
        ep_py = energy_pct(py_row, sig)
        ep_fe = energy_pct(fe_row, sig)

        # PyEMD (left column)
        ax = axes[ri, 0]
        ax.plot(T, py_row, color=RUST, linewidth=0.6, alpha=0.9)
        style_ax(ax, f"{label}  ·  {df_py:.1f} Hz  ·  {ep_py:.1f}% energy", RUST, last)
        if ri == 0:
            ax.set_title("PyEMD", color=RUST, fontsize=10, fontweight="bold",
                         pad=6, loc="center")

        # Ferromode (right column)
        ax = axes[ri, 1]
        ax.plot(T, fe_row, color=PATINA, linewidth=0.6, alpha=0.9)
        style_ax(ax, f"{label}  ·  {df_fe:.1f} Hz  ·  {ep_fe:.1f}% energy", PATINA, last)
        if ri == 0:
            ax.set_title("Ferromode (Rust)", color=PATINA, fontsize=10,
                         fontweight="bold", pad=6, loc="center")

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3, w_pad=1.0)
    fig.patch.set_facecolor(IRON)

    out = os.path.join(HERE, "gut_check_sidebyside.png")
    plt.savefig(out, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 2: overlaid ──────────────────────────────────────────────────────────

def plot_overlay(sig, py_imfs, py_res, fe_imfs, fe_res):
    rows    = ["Signal"] + [f"IMF {i+1}" for i in range(MAX_IMFS)] + ["Residual"]
    py_rows = [sig] + py_imfs[:MAX_IMFS] + [py_res]
    fe_rows = [sig] + fe_imfs[:MAX_IMFS] + [fe_res]
    n = len(rows)
    T = np.arange(len(sig))

    fig, axes = plt.subplots(n, 1, figsize=(13, 2.0 * n), facecolor=IRON)
    fig.suptitle(
        "MIT-BIH Record 100  ·  PyEMD (rust) vs Ferromode — overlaid",
        color=TEXT, fontsize=12, fontweight="bold",
    )

    for ri, (label, py_row, fe_row) in enumerate(zip(rows, py_rows, fe_rows)):
        last = (ri == n - 1)
        ax   = axes[ri]

        nrms = float(np.sqrt(np.mean((py_row - fe_row[:len(py_row)])**2)) /
                     (np.sqrt(np.mean(py_row**2)) + 1e-12))

        ax.plot(T, py_row,              color=RUST,   linewidth=0.7, alpha=0.85,
                label=f"PyEMD  {dominant_freq(py_row):.1f} Hz")
        ax.plot(T, fe_row[:len(py_row)], color=PATINA, linewidth=0.7, alpha=0.85,
                linestyle="--", label=f"Ferromode  {dominant_freq(fe_row):.1f} Hz")

        style_ax(ax, f"{label}  ·  NRMS diff = {nrms:.3f}", RUST, last)
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.2,
                  labelcolor=TEXT_MUTED, facecolor=IRON_MID, edgecolor=BORDER)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3)
    fig.patch.set_facecolor(IRON)

    out = os.path.join(HERE, "gut_check_overlay.png")
    plt.savefig(out, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 3: PSD side-by-side ──────────────────────────────────────────────────

def plot_psd(sig, py_imfs, py_res, fe_imfs, fe_res):
    from scipy.signal import welch

    rows    = ["Signal"] + [f"IMF {i+1}" for i in range(MAX_IMFS)] + ["Residual"]
    py_rows = [sig] + py_imfs[:MAX_IMFS] + [py_res]
    fe_rows = [sig] + fe_imfs[:MAX_IMFS] + [fe_res]
    n       = len(rows)

    # Welch parameters — nperseg=512 gives ~0.7 Hz resolution at 360 Hz
    nperseg = min(512, len(sig) // 4)

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.2 * n),
                             facecolor=IRON, sharey=False)
    fig.suptitle(
        "MIT-BIH Record 100  ·  PSD (Welch)  ·  PyEMD vs Ferromode",
        color=TEXT, fontsize=12, fontweight="bold",
    )

    for ri, (label, py_row, fe_row) in enumerate(zip(rows, py_rows, fe_rows)):
        last = (ri == n - 1)

        f_py, p_py = welch(py_row, fs=FS, nperseg=nperseg)
        f_fe, p_fe = welch(fe_row, fs=FS, nperseg=nperseg)

        # Convert to dB, avoid log(0)
        p_py_db = 10 * np.log10(np.maximum(p_py, 1e-20))
        p_fe_db = 10 * np.log10(np.maximum(p_fe, 1e-20))

        dom_py = f_py[np.argmax(p_py)]
        dom_fe = f_fe[np.argmax(p_fe)]

        for col_idx, (ax, f, p_db, dom, color, impl) in enumerate([
            (axes[ri, 0], f_py, p_py_db, dom_py, RUST,   "PyEMD"),
            (axes[ri, 1], f_fe, p_fe_db, dom_fe, PATINA, "Ferromode"),
        ]):
            ax.set_facecolor(IRON_MID)
            ax.fill_between(f, p_db, p_db.min(), alpha=0.25, color=color)
            ax.plot(f, p_db, color=color, linewidth=0.8)
            ax.axvline(dom, color=color, linewidth=0.7, linestyle="--", alpha=0.7)

            ax.set_title(
                f"{label}  ·  peak {dom:.1f} Hz",
                color=TEXT_MUTED, fontsize=8, pad=3, loc="left",
                fontfamily="monospace",
            )
            if ri == 0:
                ax.set_title(impl, color=color, fontsize=10,
                             fontweight="bold", pad=6, loc="center")

            ax.tick_params(axis="both", colors=TEXT_DIM, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
            ax.tick_params(labelbottom=last)
            if last:
                ax.set_xlabel("Frequency (Hz)", color=TEXT_DIM, fontsize=7)
            ax.set_ylabel("dB/Hz", color=TEXT_DIM, fontsize=6, labelpad=2)

            # Shade frequency band labels on signal row only
            if ri == 0:
                for lo, hi, band_label, alpha in [
                    (0,   1,  "baseline", 0.08),
                    (1,   10, "P/T waves", 0.08),
                    (10,  45, "QRS",       0.08),
                    (45,  180,"noise",     0.06),
                ]:
                    ax.axvspan(lo, hi, alpha=alpha, color=TEXT_DIM)
                    mid = (lo + hi) / 2
                    ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else p_db.min(),
                            band_label, color=TEXT_DIM, fontsize=5.5,
                            ha="center", va="bottom", rotation=90)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.3, w_pad=1.0)
    fig.patch.set_facecolor(IRON)

    out = os.path.join(HERE, "gut_check_psd.png")
    plt.savefig(out, dpi=150, facecolor=IRON, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data…")
    sig, py_imfs, py_res, fe_imfs, fe_res = load_data()
    print(f"Signal: {len(sig)} samples  PyEMD: {len(py_imfs)} IMFs  Ferromode: {len(fe_imfs)} IMFs")

    print("Generating side-by-side plot…")
    plot_sidebyside(sig, py_imfs, py_res, fe_imfs, fe_res)

    print("Generating overlay plot…")
    plot_overlay(sig, py_imfs, py_res, fe_imfs, fe_res)

    print("Generating PSD plot…")
    plot_psd(sig, py_imfs, py_res, fe_imfs, fe_res)

    print("Done.")


if __name__ == "__main__":
    main()
