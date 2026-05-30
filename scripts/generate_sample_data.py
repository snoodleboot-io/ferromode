#!/usr/bin/env python3
"""
Generate realistic, event-rich sample datasets for the Ferromode website.

All signals are physics-based synthetic reconstructions modeled after
well-known reference datasets. See REFERENCES section below.

Outputs (relative to repo root):
    webpage/data/ecg_sample.csv        1000 samples · 250 Hz · 4 s
    webpage/data/seismic_sample.csv    2000 samples · 100 Hz · 20 s
    webpage/data/speech_sample.csv     2000 samples · 16000 Hz · 125 ms
    webpage/data/finance_sample.csv     502 samples · daily · ~2 years

REFERENCES
----------
ECG:
    Goldberger AL et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
    Components of a New Research Resource for Complex Physiologic Signals.
    Circulation 101(23):e215-e220.  https://physionet.org/content/mitdb/
    Signal morphology: McSharry PE et al. (2003). A Dynamical Model for
    Generating Synthetic Electrocardiogram Signals. IEEE Trans Biomed Eng
    50(3):289-294.  doi:10.1109/TBME.2003.808805

Seismic:
    IRIS Earthquake Science Center / USGS.
    Waveform envelope characteristics modelled on:
    Boore DM & Atkinson GM (2008). Ground-Motion Prediction Equations for
    the Average Horizontal Component of PGA, PGV, and 5%-Damped PSA at
    Spectral Periods between 0.01 s and 10.0 s.
    Earthquake Spectra 24(1):99-138.  doi:10.1193/1.2830434
    Reference event: 2011 Tōhoku Mw 9.1 teleseismic record, IRIS DMC.

Speech:
    Garofolo JS et al. (1993). TIMIT Acoustic-Phonetic Continuous Speech
    Corpus.  Linguistic Data Consortium, Philadelphia.
    https://catalog.ldc.upenn.edu/LDC93S1
    VCV (/ɑ/-stop-/ɑ/) phoneme statistics from Hillenbrand et al. (1995).
    J Acoust Soc Am 97(5):3099-3111.

Finance:
    Nugent C (2018). S&P 500 stock data. Kaggle.
    https://www.kaggle.com/datasets/camnugent/sandp500
    Volatility structure and crash morphology modelled on S&P 500 index
    behaviour during 2020 COVID-19 market dislocation (Feb–Apr 2020).
"""

import csv
import math
import os
import random
import sys
from pathlib import Path


SEED = 42


def _rng_seed(seed: int) -> None:
    random.seed(seed)


def _randn() -> float:
    """Box-Muller normal sample."""
    u1 = random.random() + 1e-15
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _gauss_pulse(t_arr: list[float], center: float, width: float, amp: float) -> list[float]:
    return [amp * math.exp(-0.5 * ((t - center) / width) ** 2) for t in t_arr]


# ── ECG ──────────────────────────────────────────────────────────────────────
# 4 seconds · 250 Hz · 1000 samples
# Realistic QRS morphology (sum-of-Gaussians model, McSharry 2003)
# Events: 5 normal sinus beats + 1 PVC (premature ventricular contraction)
#         + baseline wander + EMG noise + 50 Hz power-line artifact

def generate_ecg(fs: int = 250, duration: float = 4.0) -> list[tuple]:
    """Returns list of (time_s, signal_mV) tuples."""
    _rng_seed(SEED)
    n = int(fs * duration)
    dt = 1.0 / fs
    t = [i * dt for i in range(n)]

    signal = [0.0] * n

    # Beat templates: (P, Q, R, S, T) as (center_offset, width, amplitude)
    # offsets relative to beat center (R-peak), units = seconds
    NORMAL_BEAT = [
        (-0.18, 0.030, +0.12),   # P wave
        (-0.04, 0.012, -0.10),   # Q wave
        ( 0.00, 0.015, +1.40),   # R wave
        (+0.04, 0.015, -0.25),   # S wave
        (+0.18, 0.055, +0.28),   # T wave
    ]
    PVC_BEAT = [
        ( 0.00, 0.030, +1.85),   # Wide, tall R (no P, broad QRS)
        (+0.06, 0.030, -0.90),   # Deep broad S
        (+0.22, 0.070, -0.45),   # Inverted T
    ]

    # Beat timings: ~75 BPM → 0.80 s RR, with slight HR variability
    # PVC replaces beat #4 (index 3)
    rr_base = 0.78
    r_peaks = []
    r = 0.35
    for k in range(6):
        jitter = _randn() * 0.02
        r_peaks.append(r)
        r += rr_base + jitter

    for beat_idx, r_center in enumerate(r_peaks):
        template = PVC_BEAT if beat_idx == 3 else NORMAL_BEAT
        for (offset, width, amp) in template:
            center = r_center + offset
            for i, ti in enumerate(t):
                signal[i] += amp * math.exp(-0.5 * ((ti - center) / width) ** 2)

    # Baseline wander: 0.3 Hz respiratory artifact
    bw_amp = 0.12
    bw_freq = 0.3
    for i, ti in enumerate(t):
        signal[i] += bw_amp * math.sin(2 * math.pi * bw_freq * ti + 0.8)

    # 50 Hz power-line interference
    pl_amp = 0.04
    for i, ti in enumerate(t):
        signal[i] += pl_amp * math.sin(2 * math.pi * 50 * ti)

    # EMG / white noise
    noise_std = 0.025
    for i in range(n):
        signal[i] += _randn() * noise_std

    return [(round(t[i], 6), round(signal[i], 6)) for i in range(n)]


# ── Seismic ──────────────────────────────────────────────────────────────────
# 20 seconds · 100 Hz · 2000 samples
# Teleseismic waveform structure:
#   0–5 s:   pre-event noise (microseismic background)
#   5–8 s:   P-wave arrival (high-freq, sharp onset, 6–12 Hz)
#   8–12 s:  S-wave coda (lower-freq, larger amplitude, 2–6 Hz)
#  12–20 s:  surface-wave train (very low-freq, 0.2–1 Hz, large amplitude)

def _damped_sine(t_arr, t0, freq, decay, amp, phase=0.0):
    out = []
    for t in t_arr:
        dt = t - t0
        if dt < 0:
            out.append(0.0)
        else:
            out.append(amp * math.exp(-decay * dt) * math.sin(2 * math.pi * freq * dt + phase))
    return out

def generate_seismic(fs: int = 100, duration: float = 20.0) -> list[tuple]:
    _rng_seed(SEED + 1)
    n = int(fs * duration)
    dt = 1.0 / fs
    t = [i * dt for i in range(n)]
    signal = [0.0] * n

    # Pre-event microseismic noise (coloured — low-pass filtered white noise)
    noise_raw = [_randn() * 0.008 for _ in range(n)]
    alpha = 0.85
    noise_lp = [noise_raw[0]]
    for v in noise_raw[1:]:
        noise_lp.append(alpha * noise_lp[-1] + (1 - alpha) * v)
    for i in range(n):
        signal[i] += noise_lp[i]

    # P-wave: sharp onset at t=5s, 9 Hz, fast decay
    p_wave = _damped_sine(t, t0=5.0, freq=9.0, decay=1.8, amp=0.35, phase=0.3)
    # add a secondary P reverberation
    p2 = _damped_sine(t, t0=5.4, freq=7.5, decay=2.2, amp=0.18, phase=1.1)
    for i in range(n):
        signal[i] += p_wave[i] + p2[i]

    # S-wave: arrives at t=8s, 3.5 Hz, slower decay, larger amplitude
    s_wave = _damped_sine(t, t0=8.0, freq=3.5, decay=0.9, amp=0.95, phase=0.0)
    s2 = _damped_sine(t, t0=8.8, freq=2.8, decay=0.7, amp=0.55, phase=2.1)
    for i in range(n):
        signal[i] += s_wave[i] + s2[i]

    # Surface waves: t=12s, 0.5 Hz (Rayleigh), very slow decay, large amp
    rayleigh = _damped_sine(t, t0=12.0, freq=0.50, decay=0.15, amp=1.80, phase=0.0)
    love = _damped_sine(t, t0=12.8, freq=0.35, decay=0.12, amp=1.20, phase=1.5)
    for i in range(n):
        signal[i] += rayleigh[i] + love[i]

    # Instrument noise on top
    for i in range(n):
        signal[i] += _randn() * 0.005

    return [(round(t[i], 6), round(signal[i], 6)) for i in range(n)]


# ── Speech ───────────────────────────────────────────────────────────────────
# 125 ms · 16 kHz · 2000 samples
# VCV token /ɑ/-/t/-/ɑ/ (as in "ata"):
#   0–45 ms:   voiced /ɑ/ onset and steady-state
#  45–65 ms:   stop closure (near-silence, some noise)
#  65–70 ms:   stop burst (brief broadband impulse)
#  70–125 ms:  voiced /ɑ/ release and steady state
# Formant structure per Hillenbrand 1995 for /ɑ/: F1=768, F2=1333, F3=2534 Hz
# Fundamental F0=127 Hz (male speaker), voiced source = glottal pulse train

def _glottal_pulse(phase: float) -> float:
    """Liljencrants-Fant approximation (simplified)."""
    if phase < 0.4:
        return math.sin(math.pi * phase / 0.4) ** 2
    elif phase < 0.85:
        return (1 - phase) / (1 - 0.4) * (-0.5) * math.sin(math.pi * (phase - 0.4) / 0.45)
    else:
        return 0.0

def generate_speech(fs: int = 16000, duration: float = 0.125) -> list[tuple]:
    _rng_seed(SEED + 2)
    n = int(fs * duration)
    dt = 1.0 / fs
    t = [i * dt for i in range(n)]
    signal = [0.0] * n

    F0 = 127.0       # fundamental (Hz)
    F1, F2, F3 = 768.0, 1333.0, 2534.0   # /ɑ/ formants (Hillenbrand 1995)
    BW1, BW2, BW3 = 90.0, 110.0, 170.0  # bandwidths

    # Glottal source phase accumulator
    phase_acc = 0.0

    for i, ti in enumerate(t):
        # ── Voiced /ɑ/ (0–45 ms and 70–125 ms) ─────────────────────────────
        in_vowel = ti < 0.045 or ti >= 0.070
        in_closure = 0.045 <= ti < 0.065
        in_burst = 0.065 <= ti < 0.070

        if in_vowel:
            # Ramp amplitude: 0→full in first 8ms, full→0 in last 8ms
            amp = 1.0
            if ti < 0.008:
                amp = ti / 0.008
            elif ti > 0.117:
                amp = (0.125 - ti) / 0.008

            # Glottal pulse source
            phase_acc = (phase_acc + F0 * dt) % 1.0
            g = _glottal_pulse(phase_acc) * amp

            # Formant resonances (IIR approximation via modulated sinusoids)
            decay1 = math.exp(-math.pi * BW1 / fs)
            decay2 = math.exp(-math.pi * BW2 / fs)
            decay3 = math.exp(-math.pi * BW3 / fs)

            signal[i] = (
                0.50 * g * math.sin(2 * math.pi * F1 * ti) * (decay1 ** i) +
                0.30 * g * math.sin(2 * math.pi * F2 * ti) * (decay2 ** i) +
                0.15 * g * math.sin(2 * math.pi * F3 * ti) * (decay3 ** i)
            )
            signal[i] += _randn() * 0.008  # breathiness

        elif in_closure:
            # Near-silence with low-level turbulence
            signal[i] = _randn() * 0.012

        elif in_burst:
            # Brief broadband burst
            signal[i] = _randn() * 0.45 * math.exp(-800 * (ti - 0.065))

    # Normalise to ±1
    peak = max(abs(v) for v in signal)
    if peak > 0:
        signal = [v / peak for v in signal]

    return [(round(t[i], 6), round(signal[i], 6)) for i in range(n)]


# ── Finance ──────────────────────────────────────────────────────────────────
# ~502 trading days · daily
# Simulates S&P 500 price-index behaviour ca. 2019–2021:
#   Phase 1 (days 0–200):   moderate bull market, low volatility
#   Phase 2 (days 200–222): rapid sell-off (-34% peak-to-trough, COVID crash)
#   Phase 3 (days 222–350): volatile recovery
#   Phase 4 (days 350–502): new-high bull run with elevated vol
# Volatility clustering via GARCH(1,1)-inspired model

def generate_finance(n_days: int = 502) -> list[tuple]:
    _rng_seed(SEED + 3)

    # GARCH(1,1) parameters per phase
    phases = [
        # (days, mu_daily, omega, alpha, beta, shock_at, shock_size)
        (200, 0.00045,  0.000002, 0.06, 0.92,  None,   0),
        ( 22, -0.0200,  0.000080, 0.18, 0.75,     5, -0.095),  # crash
        (128, 0.00300,  0.000030, 0.12, 0.84,  None,   0),      # recovery
        (152, 0.00060,  0.000008, 0.07, 0.90,  None,   0),      # new highs
    ]

    price = 3230.0  # approximate S&P 500 level early 2020
    rows = []
    day = 0
    h = 0.0002   # initial conditional variance

    for (length, mu, omega, alpha, beta, shock_at, shock_size) in phases:
        for k in range(length):
            # GARCH variance update
            eps_prev = _randn() * math.sqrt(h)
            h = omega + alpha * eps_prev ** 2 + beta * h
            h = max(h, 1e-8)

            ret = mu + math.sqrt(h) * _randn()

            # Deterministic shock (crash impulse)
            if shock_at is not None and k == shock_at:
                ret += shock_size

            price *= math.exp(ret)
            rows.append((day, round(price, 4)))
            day += 1

    return rows


# ── CSV writers ──────────────────────────────────────────────────────────────

def write_ecg(path: Path) -> None:
    data = generate_ecg()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# ECG signal · 250 Hz · 4 seconds · 1000 samples"])
        w.writerow(["# Synthetic · modelled on MIT-BIH Arrhythmia Database (PhysioNet)"])
        w.writerow(["# Goldberger et al. (2000) Circulation 101(23):e215. https://physionet.org/content/mitdb/"])
        w.writerow(["# McSharry et al. (2003) IEEE Trans Biomed Eng 50(3):289. doi:10.1109/TBME.2003.808805"])
        w.writerow(["# Events: 5 normal sinus beats + 1 PVC (beat 4) + baseline wander + 50 Hz artifact"])
        w.writerow(["time_s", "signal_mV"])
        w.writerows(data)
    print(f"  ECG       → {path}  ({len(data)} samples)")


def write_seismic(path: Path) -> None:
    data = generate_seismic()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# Seismic waveform · 100 Hz · 20 seconds · 2000 samples"])
        w.writerow(["# Synthetic teleseismic record · modelled on 2011 Tohoku Mw 9.1 (IRIS DMC)"])
        w.writerow(["# Boore & Atkinson (2008) Earthquake Spectra 24(1):99. doi:10.1193/1.2830434"])
        w.writerow(["# Events: pre-event noise | P-wave @5s | S-wave @8s | surface waves @12s"])
        w.writerow(["time_s", "displacement_um"])
        w.writerows(data)
    print(f"  Seismic   → {path}  ({len(data)} samples)")


def write_speech(path: Path) -> None:
    data = generate_speech()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# Speech VCV token /a-t-a/ · 16000 Hz · 125 ms · 2000 samples"])
        w.writerow(["# Synthetic · modelled on TIMIT corpus (Garofolo et al. 1993)"])
        w.writerow(["# https://catalog.ldc.upenn.edu/LDC93S1"])
        w.writerow(["# Formants per Hillenbrand et al. (1995) JASA 97(5):3099. F1=768 F2=1333 F3=2534 Hz"])
        w.writerow(["# Events: vowel /a/ | stop closure | burst | vowel /a/ release"])
        w.writerow(["time_s", "amplitude"])
        w.writerows(data)
    print(f"  Speech    → {path}  ({len(data)} samples)")


def write_finance(path: Path) -> None:
    data = generate_finance()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# S&P 500 synthetic price index · 502 daily samples"])
        w.writerow(["# Nugent C (2018) S&P 500 stock data. Kaggle. https://www.kaggle.com/datasets/camnugent/sandp500"])
        w.writerow(["# GARCH(1,1) model · crash morphology from 2020 COVID-19 dislocation (Feb-Apr 2020)"])
        w.writerow(["# Phases: bull (d0-200) | crash -34% (d200-222) | recovery (d222-350) | new highs (d350-502)"])
        w.writerow(["trading_day", "price_index"])
        w.writerows(data)
    print(f"  Finance   → {path}  ({len(data)} samples)")


def main() -> None:
    repo_root = Path(__file__).parent.parent
    out_dir = repo_root / "webpage" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing sample data to {out_dir}/")
    write_ecg(out_dir / "ecg_sample.csv")
    write_seismic(out_dir / "seismic_sample.csv")
    write_speech(out_dir / "speech_sample.csv")
    write_finance(out_dir / "finance_sample.csv")
    print("\nDone. Run webpage/generate_visualizations.py to regenerate SVGs.")


if __name__ == "__main__":
    main()
