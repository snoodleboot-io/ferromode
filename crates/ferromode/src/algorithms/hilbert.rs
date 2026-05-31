#![warn(missing_docs)]

//! Hilbert transform for post-processing EMD results.
//!
//! This module provides FFT-based Hilbert transform functionality for computing
//! analytic signals, instantaneous attributes, and the Hilbert-Huang spectrum.
//!
//! The Hilbert transform is applied to each IMF obtained from EMD to extract:
//! - Instantaneous amplitude (envelope)
//! - Instantaneous phase
//! - Instantaneous frequency
//! - Marginal spectrum (time-integrated energy distribution)
//!
//! # Mathematical Background
//!
//! The Hilbert transform of a real signal x(t) is defined as:
//!
//! ```text
//! H{x}(t) = (1/π) · P.V. ∫ x(τ)/(t-τ) dτ
//! ```
//!
//! The analytic signal is then:
//!
//! ```text
//! z(t) = x(t) + i·H{x}(t) = A(t)·exp(i·φ(t))
//! ```
//!
//! where:
//! - A(t) = |z(t)| is the instantaneous amplitude
//! - φ(t) = arg(z(t)) is the instantaneous phase
//! - f(t) = (1/2π) · dφ/dt is the instantaneous frequency

use num_complex::Complex64;
use rustfft::FftPlanner;

use crate::error::EmdError;
use crate::types::HilbertResult;

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Next power of 2 helper
// ---------------------------------------------------------------------------

/// Compute the next power of 2 greater than or equal to n.
fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// Core Hilbert transform
// ---------------------------------------------------------------------------

/// Compute the FFT-based Hilbert transform of a real signal.
///
/// The algorithm:
/// 1. Zero-pad signal to next power of 2
/// 2. Compute FFT
/// 3. Zero out negative frequencies (multiply by 2 for positive, keep DC/Nyquist)
/// 4. Compute IFFT to get analytic signal
/// 5. Return the imaginary part (Hilbert transform)
///
/// # Arguments
/// * `signal` — Input real-valued signal
///
/// # Returns
/// The Hilbert transform H{x}(t) as a vector of complex numbers where the
/// real part is the original signal and the imaginary part is the transform.
///
/// # Panics
/// Panics if the signal is empty.
pub fn hilbert_transform(signal: &[f64]) -> Vec<Complex64> {
    assert!(!signal.is_empty(), "signal must not be empty");

    let n = signal.len();
    let n_fft = next_pow2(n);

    // Prepare FFT input: zero-pad to next power of 2
    let mut fft_input: Vec<Complex64> = Vec::with_capacity(n_fft);
    for &val in signal {
        fft_input.push(Complex64::new(val, 0.0));
    }
    fft_input.resize(n_fft, Complex64::ZERO);

    // Forward FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    fft.process(&mut fft_input);

    // Zero out negative frequencies to create analytic signal
    // DC component (index 0) stays as-is
    // Positive frequencies (1..n/2) are multiplied by 2
    // Nyquist (n/2) stays as-is
    // Negative frequencies (n/2+1..n-1) are zeroed
    fft_input[0] = Complex64::new(fft_input[0].re, 0.0);
    for i in 1..(n_fft / 2) {
        fft_input[i] = Complex64::new(2.0 * fft_input[i].re, 2.0 * fft_input[i].im);
    }
    if n_fft > 1 {
        fft_input[n_fft / 2] = Complex64::new(fft_input[n_fft / 2].re, 0.0);
    }
    for i in (n_fft / 2 + 1)..n_fft {
        fft_input[i] = Complex64::ZERO;
    }

    // Inverse FFT
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft.process(&mut fft_input);

    // Normalize and truncate back to original length
    let norm = 1.0 / n_fft as f64;
    for val in &mut fft_input {
        *val = Complex64::new(val.re * norm, val.im * norm);
    }
    fft_input.truncate(n);

    fft_input
}

/// Compute the analytic signal z(t) = x(t) + i·H{x}(t).
///
/// This is a convenience wrapper around [`hilbert_transform`] that returns
/// the full analytic signal.
///
/// # Arguments
/// * `signal` — Input real-valued signal
///
/// # Returns
/// Analytic signal as complex vector where real part is x(t) and
/// imaginary part is H{x}(t).
pub fn analytic_signal(signal: &[f64]) -> Vec<Complex64> {
    hilbert_transform(signal)
}

/// Compute instantaneous amplitude A(t) = |z(t)|.
///
/// # Arguments
/// * `analytic` — Analytic signal (output of [`analytic_signal`])
///
/// # Returns
/// Vector of instantaneous amplitudes (envelope).
pub fn instantaneous_amplitude(analytic: &[Complex64]) -> Vec<f64> {
    analytic.iter().map(|z| z.norm()).collect()
}

/// Compute instantaneous phase φ(t) = arctan(H{x}/x).
///
/// # Arguments
/// * `analytic` — Analytic signal (output of [`analytic_signal`])
///
/// # Returns
/// Vector of instantaneous phases in radians, unwrapped to be continuous.
pub fn instantaneous_phase(analytic: &[Complex64]) -> Vec<f64> {
    let mut phase: Vec<f64> = analytic.iter().map(|z| z.arg()).collect();

    // Unwrap phase to avoid 2π jumps
    for i in 1..phase.len() {
        let diff = phase[i] - phase[i - 1];
        if diff > PI {
            phase[i] -= 2.0 * PI;
        } else if diff < -PI {
            phase[i] += 2.0 * PI;
        }
    }

    phase
}

/// Compute instantaneous frequency f(t) = (1/2π) · dφ/dt.
///
/// # Arguments
/// * `phase` — Unwrapped instantaneous phase (output of [`instantaneous_phase`])
/// * `sample_rate` — Sampling rate of the original signal in Hz
///
/// # Returns
/// Vector of instantaneous frequencies in Hz. The first element is set to
/// the second element's value (forward difference at start).
pub fn instantaneous_frequency(phase: &[f64], sample_rate: f64) -> Vec<f64> {
    if phase.is_empty() {
        return Vec::new();
    }
    if phase.len() == 1 {
        return vec![0.0];
    }

    let dt = 1.0 / sample_rate;
    let mut freq = Vec::with_capacity(phase.len());

    // First sample: use forward difference
    freq.push((phase[1] - phase[0]) / (2.0 * PI * dt));

    // Central differences for interior points
    for i in 1..(phase.len() - 1) {
        let df = (phase[i + 1] - phase[i - 1]) / (2.0 * PI * 2.0 * dt);
        freq.push(df);
    }

    // Last sample: use backward difference
    let last = (phase[phase.len() - 1] - phase[phase.len() - 2]) / (2.0 * PI * dt);
    freq.push(last);

    freq
}

/// Process all IMFs through the Hilbert transform pipeline.
///
/// This is the main entry point for Hilbert-Huang analysis. It computes
/// the instantaneous amplitude, frequency, and marginal spectrum for
/// each IMF in the collection.
///
/// # Arguments
/// * `imfs` — Collection of IMFs from EMD decomposition
/// * `sample_rate` — Sampling rate in Hz
///
/// # Returns
/// A [`HilbertResult`] containing instantaneous amplitude and frequency
/// for each IMF, plus the marginal spectrum.
///
/// # Errors
/// Returns [`EmdError::EmptySignal`] if no IMFs are provided.
/// Returns [`EmdError::InvalidSampleRate`] if sample_rate <= 0.
pub fn hilbert_imf(imfs: &[Vec<f64>], sample_rate: f64) -> Result<HilbertResult, EmdError> {
    if imfs.is_empty() {
        return Err(EmdError::EmptySignal);
    }
    if sample_rate <= 0.0 {
        return Err(EmdError::InvalidSampleRate);
    }

    let n_imfs = imfs.len();
    let mut inst_amp = Vec::with_capacity(n_imfs);
    let mut inst_freq = Vec::with_capacity(n_imfs);

    for imf in imfs {
        if imf.is_empty() {
            return Err(EmdError::EmptySignal);
        }

        let z = analytic_signal(imf);
        let amp = instantaneous_amplitude(&z);
        let phase = instantaneous_phase(&z);
        let freq = instantaneous_frequency(&phase, sample_rate);

        inst_amp.push(amp);
        inst_freq.push(freq);
    }

    // Compute marginal spectrum
    let marginal = marginal_spectrum(&inst_amp, &inst_freq);

    Ok(HilbertResult::new(inst_amp, inst_freq, marginal))
}

/// Compute the marginal spectrum (Hilbert spectrum integrated over time).
///
/// The marginal spectrum represents the total energy contribution at each
/// frequency, analogous to the Fourier power spectrum but derived from
/// the Hilbert-Huang transform.
///
/// # Arguments
/// * `instantaneous_amplitude` — Instantaneous amplitude for each IMF
/// * `instantaneous_frequency` — Instantaneous frequency for each IMF
///
/// # Returns
/// Marginal spectrum as a vector of energy values. The length is determined
/// by the maximum number of frequency bins needed.
pub fn marginal_spectrum(
    instantaneous_amplitude: &[Vec<f64>],
    instantaneous_frequency: &[Vec<f64>],
) -> Vec<f64> {
    if instantaneous_amplitude.is_empty() || instantaneous_frequency.is_empty() {
        return Vec::new();
    }

    // Use 1024 frequency bins for the marginal spectrum
    let n_bins = 1024;
    let mut spectrum = vec![0.0f64; n_bins];

    let n_imfs = instantaneous_amplitude.len();

    for imf_idx in 0..n_imfs {
        let amp = &instantaneous_amplitude[imf_idx];
        let freq = &instantaneous_frequency[imf_idx];
        let n = amp.len().min(freq.len());

        for i in 0..n {
            let f = freq[i];
            if f.is_finite() && f > 0.0 {
                // Map frequency to bin index (logarithmic scale)
                let log_f = f.log10();
                let bin = ((log_f + 3.0) / 6.0 * n_bins as f64) as isize;
                if bin >= 0 && bin < n_bins as isize {
                    // Energy contribution: A²(t) at this frequency
                    spectrum[bin as usize] += amp[i] * amp[i];
                }
            }
        }
    }

    spectrum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // =========================================================================
    // T-097: FFT-based Hilbert transform tests
    // =========================================================================

    #[test]
    fn test_hilbert_transform_cosine_gives_sine() {
        // Hilbert transform of cos(ωt) should be sin(ωt) (90° phase shift)
        let n = 256;
        let freq = 10.0;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / n as f64).cos()).collect();

        let analytic = hilbert_transform(&signal);

        // The imaginary part should approximate sin(ωt)
        for i in 0..n {
            let expected = (2.0 * PI * freq * i as f64 / n as f64).sin();
            let actual = analytic[i].im;
            let error = (actual - expected).abs();
            assert!(
                error < 0.1,
                "Hilbert of cos should give sin: at index {} expected {:.4}, got {:.4}, error {:.4}",
                i,
                expected,
                actual,
                error
            );
        }
    }

    #[test]
    fn test_hilbert_transform_real_part_preserves_signal() {
        // Real part of analytic signal should be the original signal
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 5.0 * i as f64 / n as f64).cos()
                    + 0.5 * (2.0 * PI * 15.0 * i as f64 / n as f64).cos()
            })
            .collect();

        let analytic = hilbert_transform(&signal);

        for i in 0..n {
            let error = (analytic[i].re - signal[i]).abs();
            assert!(
                error < 0.01,
                "Real part should match original: at index {} expected {:.4}, got {:.4}",
                i,
                signal[i],
                analytic[i].re
            );
        }
    }

    #[test]
    #[should_panic(expected = "signal must not be empty")]
    fn test_hilbert_transform_empty_signal_panics() {
        let empty: Vec<f64> = vec![];
        hilbert_transform(&empty);
    }

    #[test]
    fn test_hilbert_transform_single_sample() {
        let signal = vec![1.0];
        let analytic = hilbert_transform(&signal);
        assert_eq!(analytic.len(), 1);
        // Single sample: DC only, no imaginary part
        assert!((analytic[0].im).abs() < 1e-10);
    }

    // =========================================================================
    // T-098: Analytic signal tests
    // =========================================================================

    #[test]
    fn test_analytic_signal_returns_same_as_hilbert_transform() {
        let n = 64;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin()).collect();

        let z1 = hilbert_transform(&signal);
        let z2 = analytic_signal(&signal);

        assert_eq!(z1.len(), z2.len());
        for (a, b) in z1.iter().zip(z2.iter()) {
            assert!((a.re - b.re).abs() < 1e-10);
            assert!((a.im - b.im).abs() < 1e-10);
        }
    }

    // =========================================================================
    // T-099: Instantaneous amplitude tests
    // =========================================================================

    #[test]
    fn test_instantaneous_amplitude_pure_tone_constant() {
        // For a pure tone A·cos(ωt), the envelope should be constant = |A|
        let n = 256;
        let amplitude = 3.0;
        let freq = 8.0;
        let signal: Vec<f64> =
            (0..n).map(|i| amplitude * (2.0 * PI * freq * i as f64 / n as f64).cos()).collect();

        let z = analytic_signal(&signal);
        let env = instantaneous_amplitude(&z);

        // Envelope should be approximately constant
        let mean_env: f64 = env.iter().sum::<f64>() / env.len() as f64;
        let variance: f64 =
            env.iter().map(|&e| (e - mean_env).powi(2)).sum::<f64>() / env.len() as f64;

        assert!(
            variance < 0.01,
            "Envelope of pure tone should be constant, variance = {}",
            variance
        );
        assert!(
            (mean_env - amplitude).abs() < 0.1,
            "Mean envelope should match amplitude: expected {}, got {}",
            amplitude,
            mean_env
        );
    }

    #[test]
    fn test_instantaneous_amplitude_am_signal_tracks_envelope() {
        // AM signal: A(t)·cos(ωt) where A(t) = 1 + 0.5·cos(ω_m·t)
        let n = 1024;
        let carrier_freq = 50.0;
        let mod_freq = 2.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                let envelope = 1.0 + 0.5 * (2.0 * PI * mod_freq * t / n as f64).cos();
                envelope * (2.0 * PI * carrier_freq * t / n as f64).cos()
            })
            .collect();

        let z = analytic_signal(&signal);
        let env = instantaneous_amplitude(&z);

        // Expected envelope
        let expected_env: Vec<f64> =
            (0..n).map(|i| 1.0 + 0.5 * (2.0 * PI * mod_freq * i as f64 / n as f64).cos()).collect();

        // Check that envelope tracks the modulation (allowing for edge effects)
        let start = n / 4;
        let end = 3 * n / 4;
        let mut max_error = 0.0f64;
        for i in start..end {
            let error = (env[i] - expected_env[i]).abs();
            if error > max_error {
                max_error = error;
            }
        }

        assert!(max_error < 0.15, "AM envelope should track modulation: max error = {}", max_error);
    }

    #[test]
    fn test_instantaneous_amplitude_non_negative() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 7.0 * i as f64 / n as f64).sin()
                    + 0.3 * (2.0 * PI * 21.0 * i as f64 / n as f64).sin()
            })
            .collect();

        let z = analytic_signal(&signal);
        let env = instantaneous_amplitude(&z);

        for (i, &e) in env.iter().enumerate() {
            assert!(e >= 0.0, "Instantaneous amplitude must be non-negative: index {} = {}", i, e);
        }
    }

    // =========================================================================
    // T-100: Instantaneous phase tests
    // =========================================================================

    #[test]
    fn test_instantaneous_phase_pure_tone_linear() {
        // For pure tone cos(ωt), phase should be linear: φ(t) = ωt
        let n = 256;
        let freq = 8.0;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / n as f64).cos()).collect();

        let z = analytic_signal(&signal);
        let phase = instantaneous_phase(&z);

        // Phase should increase approximately linearly
        // Check phase differences are approximately constant
        let mut diffs = Vec::new();
        for i in 1..phase.len() {
            diffs.push(phase[i] - phase[i - 1]);
        }

        let mean_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let variance: f64 =
            diffs.iter().map(|&d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;

        assert!(
            variance < 0.001,
            "Phase of pure tone should increase linearly, variance of diffs = {}",
            variance
        );
    }

    #[test]
    fn test_instantaneous_phase_unwrapped() {
        // Phase should be unwrapped (no 2π jumps)
        let n = 512;
        let freq = 20.0;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / n as f64).cos()).collect();

        let z = analytic_signal(&signal);
        let phase = instantaneous_phase(&z);

        // Check no jumps larger than π
        for i in 1..phase.len() {
            let jump = (phase[i] - phase[i - 1]).abs();
            assert!(jump < PI, "Phase should be unwrapped: jump of {} at index {}", jump, i);
        }
    }

    // =========================================================================
    // T-101: Instantaneous frequency tests
    // =========================================================================

    #[test]
    fn test_instantaneous_frequency_pure_tone_constant() {
        // For pure tone at frequency f, instantaneous frequency should be constant = f.
        // sample_rate = n gives exactly integer periods, avoiding spectral leakage.
        let n = 1024;
        let sample_rate = n as f64; // 1024 Hz → 50 exact periods
        let freq_hz = 50.0; // 50 Hz tone
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq_hz * i as f64 / sample_rate).cos()).collect();

        let z = analytic_signal(&signal);
        let phase = instantaneous_phase(&z);
        let inst_freq = instantaneous_frequency(&phase, sample_rate);

        // Check frequency is approximately constant (ignore edges)
        let start = n / 8;
        let end = 7 * n / 8;
        let mean_freq: f64 = inst_freq[start..end].iter().sum::<f64>() / (end - start) as f64;

        assert!(
            (mean_freq - freq_hz).abs() < 2.0,
            "Instantaneous frequency of pure tone should match: expected {}, got {}",
            freq_hz,
            mean_freq
        );

        // Check variance is small
        let variance: f64 =
            inst_freq[start..end].iter().map(|&f| (f - mean_freq).powi(2)).sum::<f64>()
                / (end - start) as f64;

        assert!(
            variance < 1.0,
            "Frequency should be approximately constant, variance = {}",
            variance
        );
    }

    #[test]
    fn test_instantaneous_frequency_chirp_signal_linear() {
        // Linear chirp: frequency increases linearly over time.
        // sample_rate = n avoids non-integer-period leakage.
        let n = 1024;
        let sample_rate = n as f64;
        let f0 = 10.0; // Start frequency
        let f1 = 100.0; // End frequency

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let t_end = n as f64 / sample_rate;
                // Instantaneous frequency: f(t) = f0 + (f1 - f0) * t / t_end
                // Phase: φ(t) = 2π * (f0 * t + (f1-f0)/(2*t_end) * t²)
                let phase = 2.0 * PI * (f0 * t + (f1 - f0) / (2.0 * t_end) * t * t);
                phase.cos()
            })
            .collect();

        let z = analytic_signal(&signal);
        let phase = instantaneous_phase(&z);
        let inst_freq = instantaneous_frequency(&phase, sample_rate);

        // Check that frequency increases (compare first quarter to last quarter)
        let q1_end = n / 4;
        let q3_start = 3 * n / 4;

        let freq_first_quarter: f64 = inst_freq[..q1_end].iter().sum::<f64>() / q1_end as f64;
        let freq_last_quarter: f64 =
            inst_freq[q3_start..].iter().sum::<f64>() / (n - q3_start) as f64;

        assert!(
            freq_last_quarter > freq_first_quarter,
            "Chirp frequency should increase: first quarter avg = {}, last quarter avg = {}",
            freq_first_quarter,
            freq_last_quarter
        );

        // Check approximate values
        assert!(
            (freq_first_quarter - f0).abs() < 10.0,
            "First quarter frequency should be near {}: got {}",
            f0,
            freq_first_quarter
        );
        assert!(
            (freq_last_quarter - f1).abs() < 15.0,
            "Last quarter frequency should be near {}: got {}",
            f1,
            freq_last_quarter
        );
    }

    #[test]
    fn test_instantaneous_frequency_empty_phase() {
        let freq = instantaneous_frequency(&[], 1000.0);
        assert!(freq.is_empty());
    }

    #[test]
    fn test_instantaneous_frequency_single_sample() {
        let freq = instantaneous_frequency(&[0.0], 1000.0);
        assert_eq!(freq, vec![0.0]);
    }

    // =========================================================================
    // T-102: Validation against known analytic signals
    // =========================================================================

    #[test]
    fn test_hilbert_known_signal_pure_tone_comprehensive() {
        // Comprehensive test: pure tone should give constant amplitude and frequency.
        // sample_rate = n gives exactly integer periods, avoiding spectral leakage.
        let n = 1024;
        let sample_rate = n as f64; // 1024 Hz → 25 exact periods
        let amplitude = 2.0;
        let freq_hz = 25.0;

        let signal: Vec<f64> = (0..n)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f64 / sample_rate).cos())
            .collect();

        let z = analytic_signal(&signal);
        let amp = instantaneous_amplitude(&z);
        let phase = instantaneous_phase(&z);
        let freq = instantaneous_frequency(&phase, sample_rate);

        // Amplitude test
        let mean_amp: f64 = amp[n / 4..3 * n / 4].iter().sum::<f64>() / (n / 2) as f64;
        assert!(
            (mean_amp - amplitude).abs() < 0.1,
            "Amplitude should be {}: got {}",
            amplitude,
            mean_amp
        );

        // Frequency test
        let mean_freq: f64 = freq[n / 4..3 * n / 4].iter().sum::<f64>() / (n / 2) as f64;
        assert!(
            (mean_freq - freq_hz).abs() < 2.0,
            "Frequency should be {}: got {}",
            freq_hz,
            mean_freq
        );
    }

    #[test]
    fn test_hilbert_known_signal_am_signal() {
        // AM signal validation
        let n = 2048;
        let sample_rate = 1000.0;
        let carrier = 100.0;
        let mod_freq = 5.0;
        let mod_depth = 0.5;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let envelope = 1.0 + mod_depth * (2.0 * PI * mod_freq * t).cos();
                envelope * (2.0 * PI * carrier * t).cos()
            })
            .collect();

        let z = analytic_signal(&signal);
        let amp = instantaneous_amplitude(&z);

        // Check envelope matches expected modulation
        let expected_envelope: Vec<f64> = (0..n)
            .map(|i| 1.0 + mod_depth * (2.0 * PI * mod_freq * i as f64 / sample_rate).cos())
            .collect();

        // Compare in the middle region (avoid edge effects)
        let start = n / 4;
        let end = 3 * n / 4;
        let mut max_error = 0.0f64;
        for i in start..end {
            let error = (amp[i] - expected_envelope[i]).abs();
            if error > max_error {
                max_error = error;
            }
        }

        assert!(max_error < 0.2, "AM envelope should track modulation: max error = {}", max_error);
    }

    #[test]
    fn test_hilbert_known_signal_hilbert_of_cosine_is_sine() {
        // Validate that Hilbert transform of cosine gives sine (90° phase shift)
        let n = 256;
        let freq = 16.0;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / n as f64).cos()).collect();

        let z = hilbert_transform(&signal);

        // Check that imaginary part approximates sin(ωt)
        let mut total_error = 0.0f64;
        for i in 0..n {
            let expected = (2.0 * PI * freq * i as f64 / n as f64).sin();
            let actual = z[i].im;
            total_error += (actual - expected).abs();
        }
        let mean_error = total_error / n as f64;

        assert!(mean_error < 0.05, "Hilbert of cos should give sin: mean error = {}", mean_error);
    }

    // =========================================================================
    // hilbert_imf tests
    // =========================================================================

    #[test]
    fn test_hilbert_imf_empty_imfs_returns_error() {
        let result = hilbert_imf(&[], 1000.0);
        assert!(matches!(result.unwrap_err(), EmdError::EmptySignal));
    }

    #[test]
    fn test_hilbert_imf_invalid_sample_rate_returns_error() {
        let imfs = vec![vec![1.0, 2.0, 3.0]];
        assert!(matches!(hilbert_imf(&imfs, 0.0).unwrap_err(), EmdError::InvalidSampleRate));
        assert!(matches!(hilbert_imf(&imfs, -100.0).unwrap_err(), EmdError::InvalidSampleRate));
    }

    #[test]
    fn test_hilbert_imf_empty_imf_returns_error() {
        let imfs = vec![vec![]];
        assert!(matches!(hilbert_imf(&imfs, 1000.0).unwrap_err(), EmdError::EmptySignal));
    }

    #[test]
    fn test_hilbert_imf_single_imf() {
        let n = 256;
        let sample_rate = 1000.0;
        let freq = 50.0;
        let imf: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).cos()).collect();

        let result = hilbert_imf(&[imf], sample_rate);
        assert!(result.is_ok());

        let hilbert = result.unwrap();
        assert_eq!(hilbert.instantaneous_amplitude.len(), 1);
        assert_eq!(hilbert.instantaneous_frequency.len(), 1);
        assert_eq!(hilbert.instantaneous_amplitude[0].len(), n);
        assert_eq!(hilbert.instantaneous_frequency[0].len(), n);
        assert!(!hilbert.marginal_spectrum.is_empty());
    }

    #[test]
    fn test_hilbert_imf_multiple_imfs() {
        let n = 512;
        let sample_rate = 1000.0;
        let imf1: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * 50.0 * i as f64 / sample_rate).cos()).collect();
        let imf2: Vec<f64> =
            (0..n).map(|i| 0.5 * (2.0 * PI * 150.0 * i as f64 / sample_rate).cos()).collect();

        let result = hilbert_imf(&[imf1, imf2], sample_rate);
        assert!(result.is_ok());

        let hilbert = result.unwrap();
        assert_eq!(hilbert.instantaneous_amplitude.len(), 2);
        assert_eq!(hilbert.instantaneous_frequency.len(), 2);
    }

    // =========================================================================
    // marginal_spectrum tests
    // =========================================================================

    #[test]
    fn test_marginal_spectrum_empty_input() {
        let result = marginal_spectrum(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_marginal_spectrum_single_imf() {
        let n = 256;
        let sample_rate = 1000.0;
        let freq = 50.0;
        let imf: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).cos()).collect();

        let z = analytic_signal(&imf);
        let amp = instantaneous_amplitude(&z);
        let phase = instantaneous_phase(&z);
        let freq_vec = instantaneous_frequency(&phase, sample_rate);

        let spectrum = marginal_spectrum(&[amp], &[freq_vec]);
        assert_eq!(spectrum.len(), 1024);

        // Should have non-zero energy
        let total_energy: f64 = spectrum.iter().sum();
        assert!(total_energy > 0.0, "Marginal spectrum should have positive energy");
    }

    #[test]
    fn test_marginal_spectrum_multiple_imfs() {
        let n = 512;
        let sample_rate = 1000.0;

        let imf1: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * 30.0 * i as f64 / sample_rate).cos()).collect();
        let imf2: Vec<f64> =
            (0..n).map(|i| 0.5 * (2.0 * PI * 100.0 * i as f64 / sample_rate).cos()).collect();

        let z1 = analytic_signal(&imf1);
        let amp1 = instantaneous_amplitude(&z1);
        let phase1 = instantaneous_phase(&z1);
        let freq1 = instantaneous_frequency(&phase1, sample_rate);

        let z2 = analytic_signal(&imf2);
        let amp2 = instantaneous_amplitude(&z2);
        let phase2 = instantaneous_phase(&z2);
        let freq2 = instantaneous_frequency(&phase2, sample_rate);

        let spectrum = marginal_spectrum(&[amp1, amp2], &[freq1, freq2]);
        assert_eq!(spectrum.len(), 1024);

        let total_energy: f64 = spectrum.iter().sum();
        assert!(total_energy > 0.0);
    }

    // =========================================================================
    // next_pow2 tests
    // =========================================================================

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(4), 4);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(7), 8);
        assert_eq!(next_pow2(8), 8);
        assert_eq!(next_pow2(9), 16);
        assert_eq!(next_pow2(100), 128);
        assert_eq!(next_pow2(1024), 1024);
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_hilbert_transform_two_samples() {
        let signal = vec![1.0, -1.0];
        let analytic = hilbert_transform(&signal);
        assert_eq!(analytic.len(), 2);
        // Real parts should be close to original
        assert!((analytic[0].re - 1.0).abs() < 0.1);
        assert!((analytic[1].re - (-1.0)).abs() < 0.1);
    }

    #[test]
    fn test_hilbert_transform_four_samples() {
        let signal = vec![1.0, 0.0, -1.0, 0.0];
        let analytic = hilbert_transform(&signal);
        assert_eq!(analytic.len(), 4);
    }

    #[test]
    fn test_instantaneous_frequency_constant_phase() {
        // Constant phase → zero frequency
        let phase = vec![0.0, 0.0, 0.0, 0.0];
        let freq = instantaneous_frequency(&phase, 1000.0);
        assert_eq!(freq.len(), 4);
        for &f in &freq {
            assert!(f.abs() < 1e-10, "Constant phase should give zero frequency, got {}", f);
        }
    }

    #[test]
    fn test_instantaneous_frequency_linear_phase() {
        // Linear phase increase → constant frequency
        let phase: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let freq = instantaneous_frequency(&phase, 1000.0);
        assert_eq!(freq.len(), 100);

        // Frequency should be approximately constant
        let mean_freq: f64 = freq.iter().sum::<f64>() / freq.len() as f64;
        let expected_freq = 0.1 * 1000.0 / (2.0 * PI); // dφ/dt / 2π

        assert!(
            (mean_freq - expected_freq).abs() < 1.0,
            "Linear phase should give constant frequency: expected ~{}, got {}",
            expected_freq,
            mean_freq
        );
    }
}
