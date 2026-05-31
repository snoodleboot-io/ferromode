// ---------------------------------------------------------------------------
// Hilbert marginal spectrum
// ---------------------------------------------------------------------------

/// Compute the Hilbert marginal spectrum.
///
/// Integrates the instantaneous energy (amplitude squared) across time,
/// binned by instantaneous frequency. The result is a frequency-domain
/// energy distribution.
///
/// # Arguments
/// * `instantaneous_amplitude` — per-IMF instantaneous amplitude time series
/// * `instantaneous_frequency` — per-IMF instantaneous frequency time series (Hz)
/// * `sample_rate` — sampling rate of the original signal (Hz)
/// * `n_freq_bins` — number of frequency bins in the output spectrum
///
/// # Returns
/// A vector of length `n_freq_bins` where each element is the time-integrated
/// energy at that frequency bin.
pub fn marginal_spectrum(
    instantaneous_amplitude: &[Vec<f64>],
    instantaneous_frequency: &[Vec<f64>],
    sample_rate: f64,
    n_freq_bins: usize,
) -> Vec<f64> {
    if instantaneous_amplitude.is_empty()
        || instantaneous_frequency.is_empty()
        || n_freq_bins == 0
        || sample_rate <= 0.0
    {
        return vec![0.0; n_freq_bins];
    }

    let nyquist = sample_rate / 2.0;
    let bin_width = nyquist / n_freq_bins as f64;
    let mut spectrum = vec![0.0f64; n_freq_bins];

    for (amps, freqs) in instantaneous_amplitude.iter().zip(instantaneous_frequency.iter()) {
        for (&a, &f) in amps.iter().zip(freqs.iter()) {
            if f.is_finite() && f >= 0.0 && f <= nyquist && a.is_finite() {
                let bin = (f / bin_width).floor() as usize;
                if bin < n_freq_bins {
                    spectrum[bin] += a * a;
                }
            }
        }
    }

    // Normalize by number of time samples
    let n_samples = instantaneous_amplitude[0].len() as f64;
    if n_samples > 0.0 {
        for s in &mut spectrum {
            *s /= n_samples;
        }
    }

    spectrum
}

// ---------------------------------------------------------------------------
// IMF orthogonality index
// ---------------------------------------------------------------------------

/// Compute the orthogonality index between IMFs.
///
/// The orthogonality index measures how orthogonal the IMFs are to each other.
/// A value near 0 indicates well-separated (orthogonal) components, while
/// larger values indicate mode mixing.
///
/// Formula: OI = Σ_i≠j |Σ_t IMF_i(t)·IMF_j(t)| / Σ_t Σ_k IMF_k(t)²
///
/// # Arguments
/// * `imfs` — the intrinsic mode functions
///
/// # Returns
/// The orthogonality index (non-negative, 0 means perfectly orthogonal).
pub fn orthogonality_index(imfs: &[Vec<f64>]) -> f64 {
    if imfs.len() < 2 {
        return 0.0;
    }

    let n = imfs[0].len();
    if n == 0 {
        return 0.0;
    }

    // Compute total energy: Σ_t Σ_k IMF_k(t)²
    let total_energy: f64 = imfs.iter().flat_map(|imf| imf.iter()).map(|v| v * v).sum();

    if total_energy == 0.0 {
        return 0.0;
    }

    // Compute cross-terms: Σ_i≠j |Σ_t IMF_i(t)·IMF_j(t)|
    let mut cross_sum = 0.0f64;
    for i in 0..imfs.len() {
        for j in (i + 1)..imfs.len() {
            let dot_product: f64 = imfs[i].iter().zip(imfs[j].iter()).map(|(a, b)| a * b).sum();
            cross_sum += dot_product.abs();
        }
    }

    cross_sum / total_energy
}

// ---------------------------------------------------------------------------
// Degree of stationarity
// ---------------------------------------------------------------------------

/// Compute the degree of stationarity of the signal based on instantaneous
/// frequency variations across time windows.
///
/// A perfectly stationary signal has constant frequency content over time,
/// yielding a degree of stationarity near 0. Non-stationary signals have
/// time-varying frequency content, yielding larger values.
///
/// # Arguments
/// * `instantaneous_frequency` — per-IMF instantaneous frequency time series (Hz)
/// * `sample_rate` — sampling rate of the original signal (Hz)
///
/// # Returns
/// A non-negative value: 0 means perfectly stationary, larger values indicate
/// more non-stationary behavior.
pub fn degree_of_stationarity(instantaneous_frequency: &[Vec<f64>], sample_rate: f64) -> f64 {
    if instantaneous_frequency.is_empty() || sample_rate <= 0.0 {
        return 0.0;
    }

    let n_imfs = instantaneous_frequency.len();
    let n_samples = instantaneous_frequency[0].len();
    if n_samples < 2 {
        return 0.0;
    }

    // Compute mean instantaneous frequency for each IMF
    let mut mean_freqs = Vec::with_capacity(n_imfs);
    for freqs in instantaneous_frequency {
        let valid: Vec<f64> =
            freqs.iter().filter(|&&f| f.is_finite() && f >= 0.0).copied().collect();
        if valid.is_empty() {
            mean_freqs.push(0.0);
        } else {
            mean_freqs.push(valid.iter().sum::<f64>() / valid.len() as f64);
        }
    }

    // Compute variance of instantaneous frequency around the mean for each IMF
    // Degree of stationarity = average normalized variance across IMFs
    let mut total_stationarity = 0.0f64;
    let mut imf_count = 0usize;

    for (imf_idx, freqs) in instantaneous_frequency.iter().enumerate() {
        let mean = mean_freqs[imf_idx];
        if mean == 0.0 {
            continue;
        }

        let valid_count = freqs.iter().filter(|&&f| f.is_finite() && f >= 0.0).count();

        if valid_count < 2 {
            continue;
        }

        let variance: f64 = freqs
            .iter()
            .filter(|&&f| f.is_finite() && f >= 0.0)
            .map(|&f| {
                let diff = f - mean;
                diff * diff
            })
            .sum::<f64>()
            / valid_count as f64;

        // Normalize by mean squared to get coefficient of variation squared
        total_stationarity += variance / (mean * mean);
        imf_count += 1;
    }

    if imf_count == 0 {
        return 0.0;
    }

    total_stationarity / imf_count as f64
}

// ---------------------------------------------------------------------------
// IMF energy ratios
// ---------------------------------------------------------------------------

/// Compute the energy ratio of each IMF as a fraction of total signal energy.
///
/// # Arguments
/// * `imfs` — the intrinsic mode functions
///
/// # Returns
/// A vector of the same length as `imfs`, where each element is the fraction
/// of total energy contributed by that IMF. The values sum to 1.0 (or 0.0
/// if all IMFs are zero).
pub fn imf_energy_ratios(imfs: &[Vec<f64>]) -> Vec<f64> {
    if imfs.is_empty() {
        return Vec::new();
    }

    let energies: Vec<f64> =
        imfs.iter().map(|imf| imf.iter().map(|v| v * v).sum::<f64>()).collect();

    let total_energy: f64 = energies.iter().sum();

    if total_energy == 0.0 {
        return vec![0.0; imfs.len()];
    }

    energies.iter().map(|&e| e / total_energy).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // T-103: Marginal spectrum tests
    // =========================================================================

    #[test]
    fn test_marginal_spectrum_pure_tone_peak_at_correct_frequency() {
        // Pure tone at 10 Hz, sample rate 256 Hz
        // Marginal spectrum should peak at the 10 Hz bin
        let n = 256;
        let freq = 10.0;
        let sample_rate = 256.0;
        let n_freq_bins = 128;

        // Generate a pure cosine and compute its Hilbert transform properties
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).cos()).collect();

        // Use Hilbert transform to get instantaneous amplitude and frequency
        let analytic = crate::hilbert::hilbert_transform(&signal);
        let inst_amp = crate::hilbert::instantaneous_amplitude(&analytic);
        let phase = crate::hilbert::instantaneous_phase(&analytic);
        let inst_freq = crate::hilbert::instantaneous_frequency(&phase, sample_rate);

        let spectrum = marginal_spectrum(&[inst_amp], &[inst_freq], sample_rate, n_freq_bins);

        // Find the peak frequency bin
        let (peak_bin, _) =
            spectrum.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();

        let bin_width = (sample_rate / 2.0) / n_freq_bins as f64;
        let peak_freq = (peak_bin as f64 + 0.5) * bin_width;

        // Peak should be within one bin width of the true frequency
        assert!(
            (peak_freq - freq).abs() < 2.0 * bin_width,
            "peak frequency should be near {} Hz, got {} Hz (bin {})",
            freq,
            peak_freq,
            peak_bin
        );
    }

    #[test]
    fn test_marginal_spectrum_empty_input() {
        let result = marginal_spectrum(&[], &[], 100.0, 64);
        assert_eq!(result.len(), 64);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_marginal_spectrum_zero_bins() {
        let amps = vec![vec![1.0, 1.0]];
        let freqs = vec![vec![10.0, 10.0]];
        let result = marginal_spectrum(&amps, &freqs, 100.0, 0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_marginal_spectrum_negative_sample_rate() {
        let amps = vec![vec![1.0, 1.0]];
        let freqs = vec![vec![10.0, 10.0]];
        let result = marginal_spectrum(&amps, &freqs, -100.0, 64);
        assert_eq!(result.len(), 64);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_marginal_spectrum_multiple_tones() {
        // Two tones at different frequencies should produce two peaks
        let n = 512;
        let freq1 = 10.0;
        let freq2 = 40.0;
        let sample_rate = 256.0;
        let n_freq_bins = 128;

        let signal1: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq1 * i as f64 / sample_rate).cos()).collect();
        let signal2: Vec<f64> =
            (0..n).map(|i| (2.0 * PI * freq2 * i as f64 / sample_rate).cos()).collect();

        let analytic1 = crate::hilbert::hilbert_transform(&signal1);
        let amp1 = crate::hilbert::instantaneous_amplitude(&analytic1);
        let phase1 = crate::hilbert::instantaneous_phase(&analytic1);
        let freqs1 = crate::hilbert::instantaneous_frequency(&phase1, sample_rate);

        let analytic2 = crate::hilbert::hilbert_transform(&signal2);
        let amp2 = crate::hilbert::instantaneous_amplitude(&analytic2);
        let phase2 = crate::hilbert::instantaneous_phase(&analytic2);
        let freqs2 = crate::hilbert::instantaneous_frequency(&phase2, sample_rate);

        let spectrum =
            marginal_spectrum(&[amp1, amp2], &[freqs1, freqs2], sample_rate, n_freq_bins);

        let bin_width = (sample_rate / 2.0) / n_freq_bins as f64;

        // Find peaks
        let mut peaks: Vec<(usize, f64)> = Vec::new();
        for i in 1..spectrum.len() - 1 {
            if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.0 {
                peaks.push((i, spectrum[i]));
            }
        }

        // Should have at least 2 peaks
        assert!(peaks.len() >= 2, "should have at least 2 peaks, found {}", peaks.len());
    }

    // =========================================================================
    // T-104: Orthogonality index tests
    // =========================================================================

    #[test]
    fn test_orthogonality_index_orthogonal_signals_near_zero() {
        // Two signals with non-overlapping support are perfectly orthogonal
        let imf1 = vec![1.0, 1.0, 0.0, 0.0, 0.0];
        let imf2 = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let result = orthogonality_index(&[imf1, imf2]);
        assert!(result < 1e-10, "orthogonal signals should have OI near 0, got {}", result);
    }

    #[test]
    fn test_orthogonality_index_identical_signals() {
        // Identical signals: OI = |Σ x²| / (Σ x² + Σ x²) = 0.5
        let imf = vec![1.0, 2.0, 3.0, 4.0];
        let result = orthogonality_index(&[imf.clone(), imf.clone()]);
        assert!(
            (result - 0.5).abs() < 1e-10,
            "identical signals should have OI = 0.5, got {}",
            result
        );
    }

    #[test]
    fn test_orthogonality_index_single_imf() {
        let imf = vec![1.0, 2.0, 3.0];
        let result = orthogonality_index(&[imf]);
        assert_eq!(result, 0.0, "single IMF should have OI = 0");
    }

    #[test]
    fn test_orthogonality_index_empty_imfs() {
        let result = orthogonality_index(&[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_orthogonality_index_sine_cosine() {
        // Sine and cosine at the same frequency are approximately orthogonal
        // over an integer number of periods
        let n = 256;
        let imf1: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();
        let imf2: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).cos()).collect();

        let result = orthogonality_index(&[imf1, imf2]);
        assert!(
            result < 0.01,
            "sine/cosine over full period should be nearly orthogonal, got {}",
            result
        );
    }

    #[test]
    fn test_orthogonality_index_multiple_imfs() {
        // Three IMFs: two orthogonal, one overlapping
        let imf1 = vec![1.0, 0.0, 0.0, 0.0];
        let imf2 = vec![0.0, 1.0, 0.0, 0.0];
        let imf3 = vec![1.0, 1.0, 0.0, 0.0];
        let result = orthogonality_index(&[imf1, imf2, imf3]);
        // Cross terms: |imf1·imf3| + |imf2·imf3| = |1| + |1| = 2
        // Total energy: 1 + 1 + 2 = 4
        // OI = 2/4 = 0.5
        assert!((result - 0.5).abs() < 1e-10, "expected OI = 0.5, got {}", result);
    }

    // =========================================================================
    // T-105: Degree of stationarity tests
    // =========================================================================

    #[test]
    fn test_degree_of_stationarity_constant_frequency() {
        // Constant frequency → perfectly stationary → near 0
        let n = 100;
        let inst_freq = vec![10.0; n];
        let result = degree_of_stationarity(&[inst_freq], 100.0);
        assert!(
            result < 1e-10,
            "constant frequency should give stationarity near 0, got {}",
            result
        );
    }

    #[test]
    fn test_degree_of_stationarity_non_stationary_signal() {
        // Linearly increasing frequency → non-stationary → positive value
        let n = 100;
        let inst_freq: Vec<f64> = (0..n).map(|i| 5.0 + 20.0 * i as f64 / n as f64).collect();
        let result = degree_of_stationarity(&[inst_freq], 100.0);
        assert!(
            result > 0.0,
            "non-stationary signal should have positive stationarity, got {}",
            result
        );
    }

    #[test]
    fn test_degree_of_stationarity_stationary_vs_non_stationary() {
        // Stationary signal should have lower stationarity index than non-stationary
        let n = 100;
        let stationary: Vec<f64> = vec![10.0; n];
        let non_stationary: Vec<f64> = (0..n).map(|i| 5.0 + 20.0 * i as f64 / n as f64).collect();

        let stat_result = degree_of_stationarity(&[stationary], 100.0);
        let non_stat_result = degree_of_stationarity(&[non_stationary], 100.0);

        assert!(
            stat_result < non_stat_result,
            "stationary ({}) should be < non-stationary ({})",
            stat_result,
            non_stat_result
        );
    }

    #[test]
    fn test_degree_of_stationarity_empty_input() {
        let result = degree_of_stationarity(&[], 100.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_degree_of_stationarity_single_sample() {
        let result = degree_of_stationarity(&[vec![10.0]], 100.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_degree_of_stationarity_multiple_imfs() {
        let n = 100;
        let imf1_freq = vec![10.0; n]; // stationary
        let imf2_freq: Vec<f64> = (0..n).map(|i| 5.0 + 15.0 * i as f64 / n as f64).collect(); // non-stationary

        let result = degree_of_stationarity(&[imf1_freq, imf2_freq], 100.0);
        assert!(result > 0.0);
    }

    // =========================================================================
    // T-106: IMF energy ratio tests
    // =========================================================================

    #[test]
    fn test_imf_energy_ratios_sum_to_one() {
        let imf1 = vec![1.0, 2.0, 3.0];
        let imf2 = vec![4.0, 5.0, 6.0];
        let imf3 = vec![7.0, 8.0, 9.0];

        let ratios = imf_energy_ratios(&[imf1, imf2, imf3]);

        assert_eq!(ratios.len(), 3);
        let sum: f64 = ratios.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "energy ratios should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_imf_energy_ratios_single_imf() {
        let imf = vec![1.0, 2.0, 3.0];
        let ratios = imf_energy_ratios(&[imf]);
        assert_eq!(ratios.len(), 1);
        assert!((ratios[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_imf_energy_ratios_equal_energy() {
        let imf1 = vec![1.0, 0.0, 0.0];
        let imf2 = vec![0.0, 1.0, 0.0];
        let ratios = imf_energy_ratios(&[imf1, imf2]);
        assert_eq!(ratios.len(), 2);
        assert!((ratios[0] - 0.5).abs() < 1e-10);
        assert!((ratios[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_imf_energy_ratios_empty_imfs() {
        let ratios = imf_energy_ratios(&[]);
        assert!(ratios.is_empty());
    }

    #[test]
    fn test_imf_energy_ratios_all_zero() {
        let imf1 = vec![0.0, 0.0, 0.0];
        let imf2 = vec![0.0, 0.0, 0.0];
        let ratios = imf_energy_ratios(&[imf1, imf2]);
        assert_eq!(ratios.len(), 2);
        assert!((ratios[0] - 0.0).abs() < 1e-10);
        assert!((ratios[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_imf_energy_ratios_known_values() {
        // imf1 energy = 1 + 4 = 5
        // imf2 energy = 9 + 16 = 25
        // total = 30
        // ratios = [5/30, 25/30] = [1/6, 5/6]
        let imf1 = vec![1.0, 2.0];
        let imf2 = vec![3.0, 4.0];
        let ratios = imf_energy_ratios(&[imf1, imf2]);
        assert!((ratios[0] - 1.0 / 6.0).abs() < 1e-10);
        assert!((ratios[1] - 5.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_imf_energy_ratios_many_imfs() {
        let n = 50;
        let imfs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 + 1.0; 10]).collect();
        let ratios = imf_energy_ratios(&imfs);
        assert_eq!(ratios.len(), n);
        let sum: f64 = ratios.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
