use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

use crate::error::EmdError;
use crate::types::HilbertResult;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

fn pad_to_power_of_two(signal: &[f64]) -> (Vec<Complex64>, usize) {
    let n = signal.len();
    let padded_len = next_power_of_two(n);
    let mut buffer: Vec<Complex64> = signal
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .chain(std::iter::repeat(Complex64::new(0.0, 0.0)))
        .take(padded_len)
        .collect();
    (buffer, n)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the FFT-based Hilbert transform of a real signal.
///
/// Returns the analytic signal `z(t) = x(t) + i·H{x}(t)` where `H{x}` is the
/// Hilbert transform of `x`.
///
/// # Algorithm
/// 1. FFT of the input signal
/// 2. Zero negative frequencies
/// 3. Double positive frequencies (keep DC and Nyquist as-is)
/// 4. IFFT to get analytic signal
pub fn hilbert_transform(signal: &[f64]) -> Vec<Complex64> {
    if signal.is_empty() {
        return Vec::new();
    }

    let (mut buffer, original_len) = pad_to_power_of_two(signal);
    let n = buffer.len();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // Zero negative frequencies and double positive frequencies to create analytic signal
    // DC component (index 0): keep real part, zero imaginary
    // Positive frequencies (1..n/2): double magnitude
    // Nyquist (index n/2): keep real part, zero imaginary
    // Negative frequencies (n/2+1..n): set to zero
    let half = n / 2;

    // DC component: zero imaginary part
    buffer[0] = Complex64::new(buffer[0].re, 0.0);

    // Positive frequencies: double to account for zeroed negative frequencies
    for i in 1..half {
        buffer[i] *= 2.0;
    }

    // Nyquist component: zero imaginary part (if it exists)
    if n > 1 {
        buffer[half] = Complex64::new(buffer[half].re, 0.0);
    }

    // Negative frequencies: zero all of them
    for i in (half + 1)..n {
        buffer[i] = Complex64::new(0.0, 0.0);
    }

    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buffer);

    // Normalize IFFT output (rustfft doesn't normalize)
    let scale = 1.0 / n as f64;
    for c in &mut buffer {
        *c *= scale;
    }

    // Truncate to original length
    buffer.truncate(original_len);
    buffer
}

/// Compute the analytic signal `z(t) = x(t) + i·H{x}(t)`.
///
/// This is equivalent to `hilbert_transform` — the Hilbert transform produces
/// the analytic signal directly.
pub fn analytic_signal(signal: &[f64]) -> Vec<Complex64> {
    hilbert_transform(signal)
}

/// Compute instantaneous amplitude (envelope) from the analytic signal.
///
/// `A(t) = |z(t)| = sqrt(re² + im²)`
pub fn instantaneous_amplitude(analytic: &[Complex64]) -> Vec<f64> {
    analytic.iter().map(|z| z.norm()).collect()
}

/// Compute instantaneous phase from the analytic signal.
///
/// Computes `φ(t) = arg(z)` and unwraps phase discontinuities to produce
/// a continuous phase signal (no 2π jumps).
///
/// The unwrapping process removes phase wrapping artifacts by detecting
/// discontinuities larger than π and adjusting by ±2π to maintain continuity.
pub fn instantaneous_phase(analytic: &[Complex64]) -> Vec<f64> {
    let mut phase: Vec<f64> = analytic.iter().map(|z| z.arg()).collect();

    // Unwrap phase to avoid 2π jumps
    // This ensures the phase is continuous across the entire signal
    for i in 1..phase.len() {
        let diff = phase[i] - phase[i - 1];
        // If jump is > π or < -π, adjust by ±2π
        if diff > PI {
            phase[i] -= 2.0 * PI;
        } else if diff < -PI {
            phase[i] += 2.0 * PI;
        }
    }

    phase
}

/// Compute instantaneous frequency from the phase.
///
/// `f(t) = (1/2π) · dφ/dt`
///
/// The phase derivative is computed using forward differences with the first
/// sample repeated at the end to maintain length.
pub fn instantaneous_frequency(phase: &[f64], sample_rate: f64) -> Vec<f64> {
    if phase.len() <= 1 {
        return vec![0.0; phase.len()];
    }

    let dt = 1.0 / sample_rate;
    let mut freq = Vec::with_capacity(phase.len());

    for i in 0..phase.len() - 1 {
        let mut dphi = phase[i + 1] - phase[i];
        // Unwrap phase discontinuities
        if dphi > PI {
            dphi -= 2.0 * PI;
        } else if dphi < -PI {
            dphi += 2.0 * PI;
        }
        freq.push(dphi / (2.0 * PI * dt));
    }

    // Last sample: repeat previous frequency
    if freq.len() < phase.len() {
        freq.push(*freq.last().unwrap_or(&0.0));
    }

    freq
}

/// Process all IMFs through the Hilbert transform and return a `HilbertResult`.
///
/// For each IMF:
/// 1. Compute analytic signal
/// 2. Extract instantaneous amplitude
/// 3. Extract instantaneous frequency
/// 4. Compute marginal spectrum
pub fn hilbert_imf(imfs: &[Vec<f64>], sample_rate: f64) -> Result<HilbertResult, EmdError> {
    if imfs.is_empty() {
        return Err(EmdError::EmptySignal);
    }

    let mut inst_amplitudes = Vec::with_capacity(imfs.len());
    let mut inst_frequencies = Vec::with_capacity(imfs.len());

    for imf in imfs {
        if imf.is_empty() {
            return Err(EmdError::EmptySignal);
        }
        let analytic = hilbert_transform(imf);
        let amp = instantaneous_amplitude(&analytic);
        let phase = instantaneous_phase(&analytic);
        let freq = instantaneous_frequency(&phase, sample_rate);
        inst_amplitudes.push(amp);
        inst_frequencies.push(freq);
    }

    let marginal = marginal_spectrum(&inst_amplitudes, &inst_frequencies);

    Ok(HilbertResult::new(inst_amplitudes, inst_frequencies, marginal))
}

/// Compute the marginal (Hilbert) spectrum.
///
/// The marginal spectrum is the time-integrated energy distribution across
/// frequencies: `M(f) = ∫ A²(t, f) dt`
///
/// This bins the instantaneous amplitude squared by instantaneous frequency
/// to produce a frequency-domain energy distribution.
pub fn marginal_spectrum(inst_amplitudes: &[Vec<f64>], inst_frequencies: &[Vec<f64>]) -> Vec<f64> {
    if inst_amplitudes.is_empty() || inst_frequencies.is_empty() {
        return Vec::new();
    }

    // Determine frequency range and bin count
    let mut min_freq = f64::INFINITY;
    let mut max_freq = f64::NEG_INFINITY;

    for freqs in inst_frequencies {
        for &f in freqs {
            if f.is_finite() {
                min_freq = min_freq.min(f);
                max_freq = max_freq.max(f);
            }
        }
    }

    if !min_freq.is_finite() || !max_freq.is_finite() {
        return Vec::new();
    }

    // Clamp to non-negative frequencies only
    min_freq = min_freq.max(0.0);
    if max_freq <= min_freq {
        return Vec::new();
    }

    let n_bins = 256;
    let freq_range = max_freq - min_freq;
    let bin_width = freq_range / n_bins as f64;
    let mut spectrum = vec![0.0f64; n_bins];

    for (amps, freqs) in inst_amplitudes.iter().zip(inst_frequencies.iter()) {
        for (&a, &f) in amps.iter().zip(freqs.iter()) {
            if f.is_finite() && f >= min_freq && a.is_finite() {
                let bin = ((f - min_freq) / bin_width).floor() as usize;
                if bin < n_bins {
                    spectrum[bin] += a * a;
                }
            }
        }
    }

    // Normalize by number of time samples
    let n_samples = inst_amplitudes[0].len() as f64;
    if n_samples > 0.0 {
        for s in &mut spectrum {
            *s /= n_samples;
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

    // =========================================================================
    // T-102: Validate against known analytic signals
    // =========================================================================

    // --- Pure tone tests ---

    #[test]
    fn test_hilbert_pure_tone_constant_amplitude() {
        // Pure cosine: A(t) should be constant (= 1.0)
        let n = 256;
        let freq = 10.0;
        let sample_rate = 256.0;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).cos()).collect();

        let analytic = hilbert_transform(&signal);
        let amp = instantaneous_amplitude(&analytic);

        // Amplitude should be approximately constant (near 1.0)
        // Allow some edge effects from padding
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;
        for &a in &amp[mid_start..mid_end] {
            assert!((a - 1.0).abs() < 0.05, "amplitude should be ~1.0, got {} at position", a);
        }
    }

    #[test]
    fn test_hilbert_pure_tone_frequency() {
        // Pure cosine: instantaneous frequency should match the input frequency
        let n = 256;
        let freq = 10.0;
        let sample_rate = 256.0;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).cos()).collect();

        let analytic = hilbert_transform(&signal);
        let phase = instantaneous_phase(&analytic);
        let inst_freq = instantaneous_frequency(&phase, sample_rate);

        // Frequency should be approximately constant near the input frequency
        // Allow some edge effects
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;
        let avg_freq: f64 =
            inst_freq[mid_start..mid_end].iter().sum::<f64>() / (mid_end - mid_start) as f64;

        assert!(
            (avg_freq - freq).abs() < 1.0,
            "average instantaneous frequency should be ~{} Hz, got {} Hz",
            freq,
            avg_freq
        );
    }

    // --- AM signal tests ---

    #[test]
    fn test_hilbert_am_signal_envelope() {
        // AM signal: carrier at 50 Hz, modulated by 2 Hz sine
        // A(t) should follow the modulation envelope
        let n = 512;
        let carrier_freq = 50.0;
        let mod_freq = 2.0;
        let sample_rate = 512.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let envelope = 1.0 + 0.5 * (2.0 * PI * mod_freq * t).sin();
                envelope * (2.0 * PI * carrier_freq * t).cos()
            })
            .collect();

        let analytic = hilbert_transform(&signal);
        let amp = instantaneous_amplitude(&analytic);

        // The envelope should track the modulation
        // Check at a few points where we know the envelope value
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;

        // Envelope oscillates between 0.5 and 1.5
        let max_amp = amp[mid_start..mid_end].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_amp = amp[mid_start..mid_end].iter().cloned().fold(f64::INFINITY, f64::min);

        assert!(max_amp > 1.3, "max amplitude should be > 1.3, got {}", max_amp);
        assert!(min_amp < 0.7, "min amplitude should be < 0.7, got {}", min_amp);
    }

    // --- Phase shift tests ---

    #[test]
    fn test_hilbert_cosine_gives_sine_phase_shift() {
        // Hilbert transform of cos(ωt) should give sin(ωt)
        // i.e., a 90° phase shift
        let n = 256;
        let freq = 5.0;
        let sample_rate = 256.0;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).cos()).collect();

        let analytic = hilbert_transform(&signal);

        // The imaginary part should approximate sin(ωt)
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;
        for i in mid_start..mid_end {
            let t = i as f64 / sample_rate;
            let expected_sin = (2.0 * PI * freq * t).sin();
            let actual_im = analytic[i].im;
            assert!(
                (actual_im - expected_sin).abs() < 0.1,
                "imaginary part should approximate sin, expected {}, got {} at t={}",
                expected_sin,
                actual_im,
                t
            );
        }
    }

    // --- Edge cases ---

    #[test]
    fn test_hilbert_empty_signal() {
        let result = hilbert_transform(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_hilbert_single_sample() {
        let result = hilbert_transform(&[1.0]);
        assert_eq!(result.len(), 1);
        // Single sample: DC component only
        assert!((result[0].re - 1.0).abs() < 1e-10);
        assert!(result[0].im.abs() < 1e-10);
    }

    #[test]
    fn test_hilbert_two_samples() {
        let result = hilbert_transform(&[1.0, -1.0]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_analytic_signal_equals_hilbert_transform() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 / 64.0 * 2.0 * PI).cos()).collect();
        let hilbert = hilbert_transform(&signal);
        let analytic = analytic_signal(&signal);
        assert_eq!(hilbert.len(), analytic.len());
        for (h, a) in hilbert.iter().zip(analytic.iter()) {
            assert!((h.re - a.re).abs() < 1e-10);
            assert!((h.im - a.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_instantaneous_amplitude_computation() {
        let analytic =
            vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 5.0), Complex64::new(1.0, 0.0)];
        let amp = instantaneous_amplitude(&analytic);
        assert!((amp[0] - 5.0).abs() < 1e-10);
        assert!((amp[1] - 5.0).abs() < 1e-10);
        assert!((amp[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_instantaneous_phase_computation() {
        let analytic = vec![
            Complex64::new(1.0, 0.0),  // phase = 0
            Complex64::new(0.0, 1.0),  // phase = π/2
            Complex64::new(-1.0, 0.0), // phase = π
            Complex64::new(0.0, -1.0), // phase = -π/2
        ];
        let phase = instantaneous_phase(&analytic);
        assert!(phase[0].abs() < 1e-10);
        assert!((phase[1] - PI / 2.0).abs() < 1e-10);
        assert!((phase[2] - PI).abs() < 1e-10);
        assert!((phase[3] + PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_instantaneous_frequency_constant_phase() {
        // Constant phase → zero frequency
        let phase = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let freq = instantaneous_frequency(&phase, 100.0);
        for &f in &freq {
            assert!(f.abs() < 1e-10, "constant phase should give zero frequency, got {}", f);
        }
    }

    #[test]
    fn test_instantaneous_frequency_linear_phase() {
        // Linear phase increase → constant positive frequency
        let sample_rate = 100.0;
        let expected_freq = 5.0; // Hz
        let dphi_per_sample = 2.0 * PI * expected_freq / sample_rate;
        let phase: Vec<f64> = (0..50).map(|i| i as f64 * dphi_per_sample).collect();
        let freq = instantaneous_frequency(&phase, sample_rate);

        for &f in &freq {
            assert!(
                (f - expected_freq).abs() < 0.1,
                "frequency should be ~{} Hz, got {} Hz",
                expected_freq,
                f
            );
        }
    }

    #[test]
    fn test_instantaneous_frequency_empty() {
        let freq = instantaneous_frequency(&[], 100.0);
        assert!(freq.is_empty());
    }

    #[test]
    fn test_instantaneous_frequency_single_sample() {
        let freq = instantaneous_frequency(&[0.5], 100.0);
        assert_eq!(freq.len(), 1);
        assert_eq!(freq[0], 0.0);
    }

    // --- hilbert_imf tests ---

    #[test]
    fn test_hilbert_imf_single_imf() {
        let n = 128;
        let freq = 10.0;
        let sample_rate = 128.0;
        let imf: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).cos()).collect();

        let result = hilbert_imf(&[imf], sample_rate).unwrap();
        assert_eq!(result.instantaneous_amplitude.len(), 1);
        assert_eq!(result.instantaneous_frequency.len(), 1);
        assert_eq!(result.instantaneous_amplitude[0].len(), n);
        assert_eq!(result.instantaneous_frequency[0].len(), n);
        assert!(!result.marginal_spectrum.is_empty());
    }

    #[test]
    fn test_hilbert_imf_multiple_imfs() {
        let n = 128;
        let sample_rate = 128.0;
        let imf1: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * 10.0 * i as f64 / sample_rate).cos()).collect();
        let imf2: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * 25.0 * i as f64 / sample_rate).cos()).collect();

        let result = hilbert_imf(&[imf1, imf2], sample_rate).unwrap();
        assert_eq!(result.instantaneous_amplitude.len(), 2);
        assert_eq!(result.instantaneous_frequency.len(), 2);
        assert!(!result.marginal_spectrum.is_empty());
    }

    #[test]
    fn test_hilbert_imf_empty_imfs() {
        let result = hilbert_imf(&[], 100.0);
        assert!(matches!(result.unwrap_err(), EmdError::EmptySignal));
    }

    #[test]
    fn test_hilbert_imf_empty_imf() {
        let result = hilbert_imf(&[vec![]], 100.0);
        assert!(matches!(result.unwrap_err(), EmdError::EmptySignal));
    }

    // --- marginal_spectrum tests ---

    #[test]
    fn test_marginal_spectrum_empty() {
        let result = marginal_spectrum(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_marginal_spectrum_single_imf() {
        let amps = vec![vec![1.0, 1.0, 1.0, 1.0]];
        let freqs = vec![vec![10.0, 10.0, 10.0, 10.0]];
        let result = marginal_spectrum(&amps, &freqs);
        assert!(!result.is_empty());
        // All energy should be concentrated around the 10 Hz bin
        let total_energy: f64 = result.iter().sum();
        assert!(total_energy > 0.0);
    }

    #[test]
    fn test_marginal_spectrum_multiple_imfs() {
        let n = 64;
        let amps = vec![vec![1.0; n], vec![0.5; n]];
        let freqs = vec![vec![10.0; n], vec![25.0; n]];
        let result = marginal_spectrum(&amps, &freqs);
        assert!(!result.is_empty());
        let total_energy: f64 = result.iter().sum();
        assert!(total_energy > 0.0);
    }

    // --- Power of two padding tests ---

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(7), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(255), 256);
        assert_eq!(next_power_of_two(256), 256);
    }

    // --- Reconstruction energy test ---

    #[test]
    fn test_hilbert_preserves_energy() {
        // Parseval's theorem: energy in time domain ≈ energy in analytic signal
        let n = 256;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64 * 20.0 * PI).cos()).collect();

        let time_energy: f64 = signal.iter().map(|v| v * v).sum();
        let analytic = hilbert_transform(&signal);
        let analytic_energy: f64 = analytic.iter().map(|z| z.norm_sqr()).sum();

        // Analytic signal has roughly the same energy (within tolerance for padding)
        let ratio = analytic_energy / time_energy;
        assert!((ratio - 1.0).abs() < 0.1, "energy ratio should be ~1.0, got {}", ratio);
    }
}
