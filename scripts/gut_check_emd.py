#!/usr/bin/env python3
"""
EMD Gut Check — MIT-BIH Arrhythmia Database, Record 100

Replicates the reference analysis from:
  Blanco-Velasco et al. (2008) "ECG signal denoising and baseline wander
  correction based on the empirical mode decomposition."
  Computers in Biology and Medicine 38(1):1-13.
  https://doi.org/10.1016/j.compbiomed.2007.06.003

Expected results (Table 1 / Figure 2 of Blanco-Velasco 2008):
  IMF1      ~80–180 Hz   high-frequency EMG/electrode noise
  IMF2–3    ~10–45 Hz    QRS complex energy
  IMF4–6    ~1–10 Hz     P and T waves
  IMF7+     <1 Hz        baseline wander / respiratory drift
  Total IMFs: 8–10 for a 360 Hz signal over ~10 s

Usage:
    python gut_check_emd.py          # downloads record 100 automatically
    python gut_check_emd.py --plot   # also saves a comparison SVG
"""

import argparse
import math
import os
import sys

import numpy as np
import wfdb
from PyEMD import EMD


# ── Config ───────────────────────────────────────────────────────────────────
RECORD      = "100"
PHYSIONET_DB = "mitdb"
FS          = 360           # MIT-BIH sample rate (Hz)
DURATION_S  = 10            # seconds to analyse
N_SAMPLES   = FS * DURATION_S
CHANNEL     = 0             # MLII lead


def dominant_frequency(imf: np.ndarray, fs: float) -> float:
    """Return the frequency (Hz) of the largest FFT magnitude bin."""
    n   = len(imf)
    fft = np.abs(np.fft.rfft(imf * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return float(freqs[np.argmax(fft)])


def energy_fraction(imf: np.ndarray, signal: np.ndarray) -> float:
    """Fraction of total signal energy carried by this IMF."""
    total = np.sum(signal ** 2)
    return float(np.sum(imf ** 2) / total) if total > 0 else 0.0


def band_label(freq_hz: float) -> str:
    if freq_hz >= 80:
        return "noise / EMG"
    if freq_hz >= 10:
        return "QRS complex"
    if freq_hz >= 1:
        return "P/T waves"
    return "baseline wander"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", action="store_true",
                        help="Save SVG comparison plot to gut_check_output.svg")
    parser.add_argument("--duration", type=int, default=DURATION_S,
                        help=f"Seconds to analyse (default: {DURATION_S})")
    args = parser.parse_args()

    n = FS * args.duration

    # ── 1. Download / load MIT-BIH record 100 ────────────────────────────────
    cache_dir = os.path.join(os.path.dirname(__file__), "data", "mitdb")
    os.makedirs(cache_dir, exist_ok=True)

    rec_path = os.path.join(cache_dir, RECORD)
    if not os.path.exists(rec_path + ".dat"):
        print(f"Downloading MIT-BIH record {RECORD} from PhysioNet…")
        wfdb.dl_database(PHYSIONET_DB, dl_dir=cache_dir, records=[RECORD])
    else:
        print(f"Using cached record at {cache_dir}/")

    record = wfdb.rdrecord(rec_path, sampfrom=0, sampto=n, channels=[CHANNEL])
    raw    = record.p_signal[:, 0].astype(float)

    # Remove mean (baseline shift)
    signal = raw - np.mean(raw)
    print(f"\nLoaded {len(signal)} samples at {FS} Hz  "
          f"({len(signal)/FS:.1f} s)  "
          f"lead: {record.sig_name[0]}")
    print(f"Signal  min={signal.min():.4f}  max={signal.max():.4f}  "
          f"std={signal.std():.4f} mV")

    # ── 2. Run EMD ────────────────────────────────────────────────────────────
    print("\nRunning EMD (PyEMD)…  this may take ~30 s for 3600 samples")
    emd        = EMD()
    emd.MAX_ITERATION = 1000
    imfs_all   = emd.emd(signal, max_imf=12)

    # PyEMD returns IMFs stacked; last row is residual
    imfs     = imfs_all[:-1]
    residual = imfs_all[-1]

    print(f"\nExtracted {len(imfs)} IMFs + residual\n")

    # ── 3. Per-IMF analysis ───────────────────────────────────────────────────
    print(f"{'IMF':>4}  {'Dom. Freq (Hz)':>16}  {'Energy %':>9}  {'Band':>20}  {'Blanco-Velasco match?'}")
    print("─" * 80)

    # Expected band assignments from Blanco-Velasco Table 1
    blanco_bands = {
        1: "noise / EMG",
        2: "QRS complex", 3: "QRS complex",
        4: "P/T waves", 5: "P/T waves", 6: "P/T waves",
    }

    for i, imf in enumerate(imfs, start=1):
        df    = dominant_frequency(imf, FS)
        ef    = energy_fraction(imf, signal) * 100
        blabel = band_label(df)
        expected = blanco_bands.get(i, "baseline wander")
        match = "✓" if blabel == expected else f"? (expected {expected})"
        print(f"{i:>4}  {df:>16.1f}  {ef:>8.1f}%  {blabel:>20}  {match}")

    # Residual
    df_r  = dominant_frequency(residual, FS)
    ef_r  = energy_fraction(residual, signal) * 100
    print(f"{'Res':>4}  {df_r:>16.1f}  {ef_r:>8.1f}%  {'baseline wander':>20}")

    # Reconstruction check
    reconstructed = np.sum(imfs_all, axis=0)
    max_err = np.max(np.abs(signal - reconstructed[:len(signal)]))
    print(f"\nReconstruction max error: {max_err:.2e} mV  "
          f"({'PASS' if max_err < 1e-6 else 'FAIL — check tolerance'})")

    # ── 4. Optional SVG plot ──────────────────────────────────────────────────
    if args.plot:
        save_svg(signal, imfs, residual, FS)


# ── Minimal SVG output (no matplotlib dependency) ────────────────────────────

IRON       = "#0d0d0d"
IRON_MID   = "#161616"
IRON_LIGHT = "#1e1e1e"
BORDER     = "#2a2826"
BORDER_HOT = "#3d3835"
RUST       = "#e0521a"
PATINA     = "#3d9e96"
BRONZE     = "#c4973d"
TEXT       = "#f0ede8"
TEXT_DIM   = "#5c5854"
ROW_COLORS = [RUST, "#ff6b2b", PATINA, BRONZE, "#52c4bb",
              "#e8c87d", "#c47a52", "#6bbdb8", "#9a9490", "#e0521a", PATINA]
MONO = "'Space Mono','SF Mono',monospace"


def _polyline(pts, color, width=1.2):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polyline points="{p}" stroke="{color}" stroke-width="{width}" fill="none" stroke-linejoin="round"/>'


def save_svg(signal, imfs, residual, fs):
    rows   = [signal] + list(imfs[:6]) + [residual]   # cap at 6 IMFs for readability
    labels = ["Signal"] + [f"IMF {i+1}" for i in range(len(rows) - 2)] + ["Residual"]

    W, ROW_H = 1100, 110
    ML, MR, MT, MB = 68, 24, 48, 32
    PW = W - ML - MR
    H  = MT + len(rows) * ROW_H + MB
    N  = len(signal)

    out = [
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{IRON}"/>',
        f'<text x="{W//2}" y="30" font-family="{MONO}" font-size="13" font-weight="700" '
        f'fill="{TEXT}" text-anchor="middle">MIT-BIH Record 100 · EMD Gut Check · Blanco-Velasco 2008</text>',
    ]

    for ri, (row, label) in enumerate(zip(rows, labels)):
        yt   = MT + ri * ROW_H
        bg   = IRON_MID if ri % 2 == 0 else IRON_LIGHT
        col  = ROW_COLORS[ri % len(ROW_COLORS)]
        rmin, rmax = row.min(), row.max()
        rspan = rmax - rmin if rmax != rmin else 1.0
        pad   = rspan * 0.12
        lo, hi = rmin - pad, rmax + pad

        def sy(v):
            it = yt + ROW_H * 0.08
            ib = yt + ROW_H * 0.92
            f  = (v - lo) / (hi - lo) if hi != lo else 0.5
            return ib - f * (ib - it)

        out.append(f'<rect x="{ML}" y="{yt}" width="{PW}" height="{ROW_H}" fill="{bg}"/>')

        if lo <= 0 <= hi:
            zy = sy(0)
            out.append(f'<line x1="{ML}" y1="{zy:.1f}" x2="{ML+PW}" y2="{zy:.1f}" '
                       f'stroke="{BORDER_HOT}" stroke-width="0.7" stroke-dasharray="4,3"/>')

        out.append(f'<line x1="{ML}" y1="{yt}" x2="{ML}" y2="{yt+ROW_H}" '
                   f'stroke="{BORDER_HOT}" stroke-width="1"/>')
        out.append(f'<text x="{ML+7}" y="{yt+14}" font-family="{MONO}" font-size="9.5" '
                   f'font-weight="600" fill="{col}">{label}</text>')

        # Dominant frequency annotation
        df = dominant_frequency(row, fs)
        out.append(f'<text x="{ML+PW-4}" y="{yt+14}" font-family="{MONO}" font-size="8.5" '
                   f'fill="{TEXT_DIM}" text-anchor="end">{df:.1f} Hz · {band_label(df)}</text>')

        step = max(1, N // 2000)
        pts  = [(ML + (i / max(N-1, 1)) * PW, sy(v)) for i, v in enumerate(row[::step])]
        lw   = 1.4 if ri == 0 or ri == len(rows) - 1 else 1.0
        out.append(_polyline(pts, col, lw))
        out.append(f'<line x1="0" y1="{yt+ROW_H}" x2="{W}" y2="{yt+ROW_H}" '
                   f'stroke="{BORDER}" stroke-width="0.8"/>')

    out.append("</svg>")
    path = os.path.join(os.path.dirname(__file__), "gut_check_output.svg")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"\nSVG saved: {path}")


if __name__ == "__main__":
    main()
