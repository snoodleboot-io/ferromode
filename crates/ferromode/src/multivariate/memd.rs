//! Multivariate Empirical Mode Decomposition (MEMD).
//!
//! Implements the MEMD algorithm from Rehman & Mandic (2010),
//! "Multivariate Empirical Mode Decomposition".
//!
//! MEMD extends EMD to n-variate signals by:
//! 1. Projecting the multivariate signal onto direction vectors on the n-sphere
//! 2. Finding extrema of each 1D projection
//! 3. Interpolating multivariate envelopes via cubic splines
//! 4. Averaging envelopes across all directions to compute the local mean
//! 5. Sifting until the multivariate stopping criterion is met

use crate::error::EmdError;
use crate::extrema::detect_extrema;
use crate::multivariate::direction_sampling::{generate_directions, DirectionConfig};
use crate::sifting::SiftingConfig;
use crate::spline::cubic::CubicSpline;
use crate::spline::Spline;
use crate::types::{AlgorithmType, DecompositionResult, ImfCollection};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use web_time::Instant;

/// Configuration for MEMD decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemdConfig {
    /// Direction sampling configuration.
    pub direction_config: DirectionConfig,
    /// Maximum number of IMFs to extract (0 = auto).
    pub max_imfs: usize,
    /// Sifting configuration reused from the sifting module.
    pub sifting_config: SiftingConfig,
}

impl MemdConfig {
    /// Create a new MEMD configuration with defaults.
    #[must_use]
    pub fn new(direction_config: DirectionConfig, sifting_config: SiftingConfig) -> Self {
        Self { direction_config, max_imfs: 0, sifting_config }
    }

    /// Set the maximum number of IMFs to extract.
    #[must_use]
    pub fn with_max_imfs(mut self, max_imfs: usize) -> Self {
        self.max_imfs = max_imfs;
        self
    }
}

/// Multivariate stopping criterion state.
struct MultivariateStoppingState {
    /// Number of consecutive siftings where extrema count was stable.
    consecutive_stable: usize,
    /// Previous total extrema count across all projections.
    prev_total_extrema: usize,
    /// Required consecutive stable iterations (from s_number).
    s_number: usize,
    /// Maximum sifting iterations.
    max_iterations: usize,
    /// SD threshold for convergence.
    sd_threshold: f64,
    /// Previous candidate for SD computation.
    prev_candidate: Option<Vec<Vec<f64>>>,
}

impl MultivariateStoppingState {
    fn new(config: &SiftingConfig) -> Self {
        Self {
            consecutive_stable: 0,
            prev_total_extrema: 0,
            s_number: config.s_number,
            max_iterations: config.max_sifting_iterations,
            sd_threshold: config.sd_threshold,
            prev_candidate: None,
        }
    }

    fn should_stop(
        &mut self,
        candidate: &[Vec<f64>],
        directions: &[Vec<f64>],
        iteration: usize,
    ) -> bool {
        if iteration >= self.max_iterations {
            return true;
        }

        // Count total extrema across all projections
        let total_extrema = count_total_extrema(candidate, directions);

        // Check extrema stability
        let extrema_stable = if self.prev_total_extrema > 0 {
            let diff = (total_extrema as isize - self.prev_total_extrema as isize).abs();
            diff <= 1
        } else {
            false
        };

        if extrema_stable {
            self.consecutive_stable += 1;
        } else {
            self.consecutive_stable = 0;
        }
        self.prev_total_extrema = total_extrema;

        // Check SD criterion
        let sd_met = if let Some(ref prev) = self.prev_candidate {
            let sd = compute_multivariate_sd(prev, candidate);
            sd < self.sd_threshold
        } else {
            false
        };

        self.prev_candidate = Some(candidate.iter().map(|ch| ch.clone()).collect());

        self.consecutive_stable >= self.s_number || sd_met
    }
}

/// Count total extrema across all projections of a multivariate signal.
fn count_total_extrema(signal: &[Vec<f64>], directions: &[Vec<f64>]) -> usize {
    directions
        .par_iter()
        .map(|dir| {
            let projected = project_signal(signal, dir);
            let extrema = detect_extrema(&projected);
            extrema.maxima.len() + extrema.minima.len()
        })
        .sum()
}

/// Compute multivariate SD across all channels.
fn compute_multivariate_sd(prev: &[Vec<f64>], current: &[Vec<f64>]) -> f64 {
    if prev.is_empty() || current.is_empty() {
        return f64::INFINITY;
    }

    let n_channels = prev.len();
    let mut total_sd = 0.0f64;

    for ch in 0..n_channels {
        let numerator: f64 =
            prev[ch].iter().zip(current[ch].iter()).map(|(&p, &c)| (p - c).powi(2)).sum();
        let denominator: f64 = prev[ch].iter().map(|&x| x.powi(2)).sum();
        if denominator > 0.0 {
            total_sd += numerator / denominator;
        }
    }

    total_sd / n_channels as f64
}

/// Project a multivariate signal onto a direction vector.
///
/// For each sample index i, computes the dot product of signal[:, i] with dir.
fn project_signal(signal: &[Vec<f64>], direction: &[f64]) -> Vec<f64> {
    let n_samples = signal[0].len();
    let n_channels = signal.len();

    (0..n_samples)
        .map(|i| {
            let mut dot = 0.0f64;
            for ch in 0..n_channels {
                dot += signal[ch][i] * direction[ch];
            }
            dot
        })
        .collect()
}

/// Compute the envelope for each channel given extrema indices from a projection.
///
/// For each channel, interpolates the envelope using cubic splines through the
/// extrema indices and values of that specific channel.
fn compute_channel_envelopes(
    signal: &[Vec<f64>],
    maxima_indices: &[usize],
    minima_indices: &[usize],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), EmdError> {
    let n_channels = signal.len();
    let n_samples = signal[0].len();

    let mut upper_envelopes = vec![vec![0.0f64; n_samples]; n_channels];
    let mut lower_envelopes = vec![vec![0.0f64; n_samples]; n_channels];

    for ch in 0..n_channels {
        // Upper envelope: interpolate through maxima
        if maxima_indices.len() >= 2 {
            let max_x: Vec<f64> = maxima_indices.iter().map(|&i| i as f64).collect();
            let max_y: Vec<f64> = maxima_indices.iter().map(|&i| signal[ch][i]).collect();
            let spline = CubicSpline::from_knots(&max_x, &max_y)?;
            for i in 0..n_samples {
                upper_envelopes[ch][i] = spline.evaluate(i as f64);
            }
        }

        // Lower envelope: interpolate through minima
        if minima_indices.len() >= 2 {
            let min_x: Vec<f64> = minima_indices.iter().map(|&i| i as f64).collect();
            let min_y: Vec<f64> = minima_indices.iter().map(|&i| signal[ch][i]).collect();
            let spline = CubicSpline::from_knots(&min_x, &min_y)?;
            for i in 0..n_samples {
                lower_envelopes[ch][i] = spline.evaluate(i as f64);
            }
        }
    }

    Ok((upper_envelopes, lower_envelopes))
}

/// Compute the multivariate local mean by averaging envelopes over all directions.
fn compute_local_mean(
    signal: &[Vec<f64>],
    directions: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, EmdError> {
    let n_channels = signal.len();
    let n_samples = signal[0].len();
    let n_directions = directions.len();

    let mut mean = vec![vec![0.0f64; n_samples]; n_channels];

    for dir in directions {
        let projected = project_signal(signal, dir);
        let extrema = detect_extrema(&projected);

        let (upper, lower) = compute_channel_envelopes(signal, &extrema.maxima, &extrema.minima)?;

        for ch in 0..n_channels {
            for i in 0..n_samples {
                mean[ch][i] += (upper[ch][i] + lower[ch][i]) / 2.0;
            }
        }
    }

    // Average over all directions
    for ch in 0..n_channels {
        for i in 0..n_samples {
            mean[ch][i] /= n_directions as f64;
        }
    }

    Ok(mean)
}

/// Perform one multivariate sifting iteration.
fn sift_iteration(signal: &[Vec<f64>], directions: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, EmdError> {
    let mean = compute_local_mean(signal, directions)?;

    let n_channels = signal.len();
    let n_samples = signal[0].len();

    let mut candidate = vec![vec![0.0f64; n_samples]; n_channels];
    for ch in 0..n_channels {
        for i in 0..n_samples {
            candidate[ch][i] = signal[ch][i] - mean[ch][i];
        }
    }

    Ok(candidate)
}

/// Extract one multivariate IMF from the signal.
fn extract_one_imf(
    signal: &[Vec<f64>],
    directions: &[Vec<f64>],
    config: &SiftingConfig,
) -> Result<Vec<Vec<f64>>, EmdError> {
    let mut h = signal.iter().map(|ch| ch.clone()).collect::<Vec<_>>();
    let mut stopping = MultivariateStoppingState::new(config);
    let mut iteration = 0;

    loop {
        let candidate = sift_iteration(&h, directions)?;
        iteration += 1;

        if stopping.should_stop(&candidate, directions, iteration) {
            return Ok(candidate);
        }

        h = candidate;
    }
}

/// Perform Multivariate Empirical Mode Decomposition.
///
/// # Arguments
/// * `signal` - Multivariate signal where each inner Vec is one channel.
///              All channels must have the same length.
/// * `config` - MEMD configuration.
///
/// # Returns
/// Decomposition result containing multivariate IMFs and residue.
///
/// # Errors
/// Returns `EmdError` if the signal is invalid or decomposition fails.
pub fn memd(signal: &[Vec<f64>], config: &MemdConfig) -> Result<DecompositionResult, EmdError> {
    let start = Instant::now();

    // Validate input
    if signal.is_empty() {
        return Err(EmdError::EmptySignal);
    }

    let n_channels = signal.len();
    let n_samples = signal[0].len();

    if n_samples < 3 {
        return Err(EmdError::InsufficientData);
    }

    for ch in signal {
        if ch.len() != n_samples {
            return Err(EmdError::DimensionMismatch);
        }
        for &val in ch {
            if !val.is_finite() {
                return Err(EmdError::InvalidValue);
            }
        }
    }

    if n_channels < 2 {
        return Err(EmdError::InvalidConfig("MEMD requires at least 2 channels".to_string()));
    }

    // Generate direction vectors
    let directions = generate_directions(&config.direction_config, n_channels);

    let mut all_imfs: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut residue = signal.iter().map(|ch| ch.clone()).collect::<Vec<_>>();
    let mut total_siftings = 0usize;

    // Determine max IMFs
    let max_imfs = if config.max_imfs > 0 {
        config.max_imfs
    } else {
        // Auto: roughly log2(n_samples) - 1
        (n_samples as f64).log2().floor() as usize - 1
    };

    let max_imfs = max_imfs.max(1);

    for _imf_idx in 0..max_imfs {
        // Check if residue is monotonic or has insufficient extrema
        let total_extrema: usize = residue
            .iter()
            .map(|ch| {
                let extrema = detect_extrema(ch);
                extrema.maxima.len() + extrema.minima.len()
            })
            .sum();

        if total_extrema < 2 {
            break;
        }

        // Extract one IMF
        match extract_one_imf(&residue, &directions, &config.sifting_config) {
            Ok(imf) => {
                let imf_energy: f64 = imf.iter().flat_map(|ch| ch.iter()).map(|&x| x.powi(2)).sum();

                // Check if IMF has meaningful energy
                let residue_energy: f64 =
                    residue.iter().flat_map(|ch| ch.iter()).map(|&x| x.powi(2)).sum();

                if imf_energy < 1e-15 * residue_energy.max(1e-30) {
                    break;
                }

                // Update residue
                for ch in 0..n_channels {
                    for i in 0..n_samples {
                        residue[ch][i] -= imf[ch][i];
                    }
                }

                all_imfs.push(imf);
                total_siftings += 1;
            }
            Err(EmdError::ConvergenceFailed { .. }) => {
                // If sifting didn't converge, still use the current residue as an IMF
                // This can happen for the last few IMFs
                if !residue.iter().all(|ch| ch.iter().all(|&x| x.abs() < 1e-15)) {
                    all_imfs.push(residue.clone());
                    residue = vec![vec![0.0f64; n_samples]; n_channels];
                }
                break;
            }
            Err(e) => return Err(e),
        }
    }

    // Flatten IMFs for ImfCollection: interleave channels
    // Each IMF in ImfCollection is one channel's worth of data
    // For multivariate, we store each channel's IMFs separately
    let mut flat_imfs: Vec<Vec<f64>> = Vec::new();
    for ch in 0..n_channels {
        for imf in &all_imfs {
            flat_imfs.push(imf[ch].clone());
        }
    }

    // Flatten residue
    let flat_residue: Vec<f64> = residue.iter().flat_map(|ch| ch.clone()).collect();

    let imf_collection = ImfCollection::new(flat_imfs, flat_residue);

    let elapsed = start.elapsed();

    Ok(DecompositionResult::new(
        AlgorithmType::MEMD,
        imf_collection,
        elapsed,
        total_siftings,
        format!(
            "{{\"directions\": {}, \"n_channels\": {}, \"max_imfs\": {}}}",
            config.direction_config.num_directions, n_channels, config.max_imfs
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-6;

    // ─── Helper functions ───────────────────────────────────────────────

    fn generate_sine_wave(freq: f64, n_samples: usize, sample_rate: f64) -> Vec<f64> {
        (0..n_samples).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin()).collect()
    }

    fn generate_hexavariate_signal(
        freq1: f64,
        freq2: f64,
        n_samples: usize,
        sample_rate: f64,
    ) -> Vec<Vec<f64>> {
        let mut channels = Vec::with_capacity(6);
        for ch in 0..6 {
            let phase = ch as f64 * PI / 6.0;
            let ch_signal: Vec<f64> = (0..n_samples)
                .map(|i| {
                    (2.0 * PI * freq1 * i as f64 / sample_rate + phase).sin()
                        + 0.5 * (2.0 * PI * freq2 * i as f64 / sample_rate + phase * 2.0).sin()
                })
                .collect();
            channels.push(ch_signal);
        }
        channels
    }

    fn compute_reconstruction_error(original: &[Vec<f64>], result: &DecompositionResult) -> f64 {
        let n_channels = original.len();
        let n_samples = original[0].len();
        let n_imfs_per_channel = result.imfs.n_imfs() / n_channels;

        let mut max_error = 0.0f64;

        for ch in 0..n_channels {
            for i in 0..n_samples {
                let mut reconstructed = 0.0f64;
                // Sum all IMFs for this channel
                for imf_idx in 0..n_imfs_per_channel {
                    let flat_idx = ch * n_imfs_per_channel + imf_idx;
                    if flat_idx < result.imfs.imfs.len() {
                        reconstructed += result.imfs.imfs[flat_idx][i];
                    }
                }
                // Add residue
                let residue_start = ch * n_samples;
                if residue_start + i < result.imfs.residue.len() {
                    reconstructed += result.imfs.residue[residue_start + i];
                }

                let error = (original[ch][i] - reconstructed).abs();
                max_error = max_error.max(error);
            }
        }

        max_error
    }

    // ─── T-089: Projection tests ────────────────────────────────────────

    #[test]
    fn test_project_signal_bivariate() {
        let signal = vec![vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 0.0]];
        let direction = vec![1.0, 0.0];
        let projected = project_signal(&signal, &direction);
        assert_eq!(projected, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_project_signal_equal_weights() {
        let signal = vec![vec![1.0, 2.0, 3.0], vec![1.0, 0.0, 1.0]];
        let direction = vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
        let projected = project_signal(&signal, &direction);
        let expected = vec![
            (1.0 + 1.0) / 2.0_f64.sqrt(),
            (2.0 + 0.0) / 2.0_f64.sqrt(),
            (3.0 + 1.0) / 2.0_f64.sqrt(),
        ];
        for (a, b) in projected.iter().zip(expected.iter()) {
            assert!((a - b).abs() < TOLERANCE);
        }
    }

    #[test]
    fn test_project_signal_single_direction() {
        let signal = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let direction = vec![1.0, 0.0, 0.0];
        let projected = project_signal(&signal, &direction);
        assert_eq!(projected, vec![1.0, 2.0]);
    }

    // ─── T-090: Envelope interpolation tests ────────────────────────────

    #[test]
    fn test_compute_channel_envelopes_bivariate() {
        let signal = vec![vec![0.0, 1.0, 0.0, -1.0, 0.0], vec![0.0, 0.5, 0.0, -0.5, 0.0]];
        let maxima = vec![1];
        let minima = vec![3];

        // With only 1 max and 1 min (insufficient for spline), the function
        // gracefully returns zero envelopes rather than an error.
        let result = compute_channel_envelopes(&signal, &maxima, &minima);
        assert!(result.is_ok());
        let (upper, lower) = result.unwrap();
        assert_eq!(upper.len(), 2);
        assert_eq!(lower.len(), 2);
        // Envelopes should be zero-filled (no spline computed)
        assert!(upper[0].iter().all(|&v| v == 0.0));
        assert!(lower[0].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_compute_channel_envelopes_sufficient_extrema() {
        // Use 3 full periods so each channel has >= 2 maxima and >= 2 minima.
        let n = 120;
        let signal: Vec<Vec<f64>> = (0..2)
            .map(|ch| {
                (0..n)
                    .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64 + ch as f64 * 0.5).sin())
                    .collect()
            })
            .collect();

        // Find extrema of first channel
        let extrema = detect_extrema(&signal[0]);
        assert!(extrema.maxima.len() >= 2);
        assert!(extrema.minima.len() >= 2);

        let result = compute_channel_envelopes(&signal, &extrema.maxima, &extrema.minima);
        assert!(result.is_ok());

        let (upper, lower) = result.unwrap();
        assert_eq!(upper.len(), 2);
        assert_eq!(lower.len(), 2);
        assert_eq!(upper[0].len(), n);
        assert_eq!(lower[0].len(), n);

        // Upper envelope should be >= signal at maxima
        for &idx in &extrema.maxima {
            for ch in 0..2 {
                assert!(
                    upper[ch][idx] >= signal[ch][idx] - TOLERANCE,
                    "upper envelope at max should >= signal"
                );
            }
        }
    }

    // ─── T-091: Local mean computation tests ────────────────────────────

    #[test]
    fn test_compute_local_mean_bivariate() {
        let n = 100;
        let signal: Vec<Vec<f64>> = (0..2)
            .map(|ch| {
                (0..n).map(|i| (2.0 * PI * i as f64 / n as f64 + ch as f64 * 0.5).sin()).collect()
            })
            .collect();

        let config = DirectionConfig::new(8).with_seed(42);
        let directions = generate_directions(&config, 2);

        let result = compute_local_mean(&signal, &directions);
        assert!(result.is_ok());

        let mean = result.unwrap();
        assert_eq!(mean.len(), 2);
        assert_eq!(mean[0].len(), n);

        // For a pure sine wave, the local mean should be small (near zero)
        let mean_abs: f64 = mean.iter().flat_map(|ch| ch.iter()).map(|&x| x.abs()).sum::<f64>()
            / (mean.len() * mean[0].len()) as f64;
        assert!(mean_abs < 0.5, "local mean of sine wave should be small, got {mean_abs}");
    }

    // ─── T-092: Multivariate sifting tests ──────────────────────────────

    #[test]
    fn test_sift_iteration_bivariate() {
        // Use 3 periods so projections have multiple extrema (enough for spline envelopes).
        let n = 120;
        let signal: Vec<Vec<f64>> = (0..2)
            .map(|ch| {
                (0..n)
                    .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64 + ch as f64 * 0.5).sin())
                    .collect()
            })
            .collect();

        let config = DirectionConfig::new(8).with_seed(42);
        let directions = generate_directions(&config, 2);

        let result = sift_iteration(&signal, &directions);
        assert!(result.is_ok());

        let candidate = result.unwrap();
        assert_eq!(candidate.len(), 2);
        assert_eq!(candidate[0].len(), n);

        // Candidate should differ from original
        let diff: f64 = candidate
            .iter()
            .zip(signal.iter())
            .flat_map(|(c, s)| c.iter().zip(s.iter()).map(|(&a, &b)| (a - b).abs()))
            .sum();
        assert!(diff > 0.0, "sifting should change the signal");
    }

    #[test]
    fn test_extract_one_imf_convergence() {
        let n = 100;
        let signal: Vec<Vec<f64>> = (0..2)
            .map(|ch| {
                (0..n).map(|i| (2.0 * PI * i as f64 / n as f64 + ch as f64 * 0.5).sin()).collect()
            })
            .collect();

        let config = DirectionConfig::new(8).with_seed(42);
        let directions = generate_directions(&config, 2);
        let sifting_config = SiftingConfig::default();

        let result = extract_one_imf(&signal, &directions, &sifting_config);
        assert!(result.is_ok(), "extract_one_imf should converge for simple bivariate sine");

        let imf = result.unwrap();
        assert_eq!(imf.len(), 2);
        assert_eq!(imf[0].len(), n);
    }

    #[test]
    fn test_multivariate_stopping_state() {
        let config = SiftingConfig::default();
        let mut state = MultivariateStoppingState::new(&config);

        let n = 50;
        let signal: Vec<Vec<f64>> = (0..2)
            .map(|ch| (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect())
            .collect();
        let directions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        // Should not stop on first iteration
        assert!(!state.should_stop(&signal, &directions, 0));
    }

    // ─── T-093: Mode-alignment and integration tests ────────────────────

    #[test]
    fn test_memd_bivariate_sine() {
        let n = 200;
        let sample_rate = 100.0;
        let signal =
            vec![generate_sine_wave(5.0, n, sample_rate), generate_sine_wave(5.0, n, sample_rate)];

        let dir_config = DirectionConfig::new(16).with_seed(42);
        let sifting_config = SiftingConfig {
            max_sifting_iterations: 50,
            sd_threshold: 0.3,
            s_number: 3,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: crate::boundary::BoundaryConditionType::MirrorEven,
            spline_type: crate::spline::SplineType::Natural,
        };
        let config = MemdConfig::new(dir_config, sifting_config).with_max_imfs(3);

        let result = memd(&signal, &config);
        assert!(result.is_ok(), "MEMD should succeed for bivariate sine: {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.algorithm, AlgorithmType::MEMD);
        assert!(!result.imfs.imfs.is_empty(), "should extract at least one IMF");
    }

    #[test]
    fn test_memd_trivariate_signal() {
        let n = 200;
        let sample_rate = 100.0;
        let signal = vec![
            generate_sine_wave(5.0, n, sample_rate),
            generate_sine_wave(5.0, n, sample_rate),
            generate_sine_wave(5.0, n, sample_rate),
        ];

        let dir_config = DirectionConfig::new(16).with_seed(42);
        let sifting_config = SiftingConfig {
            max_sifting_iterations: 50,
            sd_threshold: 0.3,
            s_number: 3,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: crate::boundary::BoundaryConditionType::MirrorEven,
            spline_type: crate::spline::SplineType::Natural,
        };
        let config = MemdConfig::new(dir_config, sifting_config).with_max_imfs(3);

        let result = memd(&signal, &config);
        assert!(result.is_ok(), "MEMD should succeed for trivariate signal");
    }

    #[test]
    fn test_memd_tetravariate_signal() {
        let n = 200;
        let sample_rate = 100.0;
        let signal: Vec<Vec<f64>> =
            (0..4).map(|_| generate_sine_wave(5.0, n, sample_rate)).collect();

        let dir_config = DirectionConfig::new(16).with_seed(42);
        let sifting_config = SiftingConfig {
            max_sifting_iterations: 50,
            sd_threshold: 0.3,
            s_number: 3,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: crate::boundary::BoundaryConditionType::MirrorEven,
            spline_type: crate::spline::SplineType::Natural,
        };
        let config = MemdConfig::new(dir_config, sifting_config).with_max_imfs(3);

        let result = memd(&signal, &config);
        assert!(result.is_ok(), "MEMD should succeed for tetravariate signal");
    }

    #[test]
    fn test_memd_hexavariate_mode_alignment() {
        let n = 500;
        let sample_rate = 1000.0;
        let freq1 = 5.0;
        let freq2 = 25.0;

        let signal = generate_hexavariate_signal(freq1, freq2, n, sample_rate);

        let dir_config = DirectionConfig::new(32).with_seed(42);
        let sifting_config = SiftingConfig {
            max_sifting_iterations: 80,
            sd_threshold: 0.2,
            s_number: 4,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: crate::boundary::BoundaryConditionType::MirrorEven,
            spline_type: crate::spline::SplineType::Natural,
        };
        let config = MemdConfig::new(dir_config, sifting_config);

        let result = memd(&signal, &config);
        assert!(result.is_ok(), "MEMD should succeed for hexavariate signal with 2 frequencies");

        let result = result.unwrap();
        let n_channels = 6;
        let n_imfs_per_channel = result.imfs.n_imfs() / n_channels;

        assert!(
            n_imfs_per_channel >= 2,
            "should extract at least 2 IMFs (one per frequency), got {n_imfs_per_channel}"
        );

        // Mode alignment: IMFs should align across channels
        // Check that dominant frequencies are consistent across channels for each IMF
        for imf_idx in 0..n_imfs_per_channel.min(2) {
            let mut channel_dominant_freqs = Vec::new();
            for ch in 0..n_channels {
                let flat_idx = ch * n_imfs_per_channel + imf_idx;
                if flat_idx < result.imfs.imfs.len() {
                    let imf = &result.imfs.imfs[flat_idx];
                    let dominant = estimate_dominant_frequency(imf, sample_rate);
                    channel_dominant_freqs.push(dominant);
                }
            }

            // All channels should have similar dominant frequency for this IMF
            if !channel_dominant_freqs.is_empty() {
                let mean_freq: f64 = channel_dominant_freqs.iter().sum::<f64>()
                    / channel_dominant_freqs.len() as f64;
                for &freq in &channel_dominant_freqs {
                    assert!(
                        (freq - mean_freq).abs() < mean_freq * 0.3 + 2.0,
                        "mode alignment: channel freq {freq} deviates too much from mean {mean_freq}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_memd_octavariate_signal() {
        let n = 200;
        let sample_rate = 100.0;
        let signal: Vec<Vec<f64>> = (0..8)
            .map(|ch| {
                let phase = ch as f64 * PI / 8.0;
                (0..n).map(|i| (2.0 * PI * 5.0 * i as f64 / sample_rate + phase).sin()).collect()
            })
            .collect();

        let dir_config = DirectionConfig::new(32).with_seed(42);
        let sifting_config = SiftingConfig {
            max_sifting_iterations: 50,
            sd_threshold: 0.3,
            s_number: 3,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: crate::boundary::BoundaryConditionType::MirrorEven,
            spline_type: crate::spline::SplineType::Natural,
        };
        let config = MemdConfig::new(dir_config, sifting_config).with_max_imfs(3);

        let result = memd(&signal, &config);
        assert!(result.is_ok(), "MEMD should succeed for octavariate signal");
    }

    #[test]
    fn test_memd_reconstruction_validation() {
        let n = 200;
        let sample_rate = 100.0;
        let signal =
            vec![generate_sine_wave(5.0, n, sample_rate), generate_sine_wave(5.0, n, sample_rate)];

        let dir_config = DirectionConfig::new(16).with_seed(42);
        let sifting_config = SiftingConfig {
            max_sifting_iterations: 50,
            sd_threshold: 0.2,
            s_number: 4,
            fixed_iterations: None,
            energy_threshold: 1e-8,
            boundary_condition: crate::boundary::BoundaryConditionType::MirrorEven,
            spline_type: crate::spline::SplineType::Natural,
        };
        let config = MemdConfig::new(dir_config, sifting_config).with_max_imfs(5);

        let result = memd(&signal, &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        let error = compute_reconstruction_error(&signal, &result);

        assert!(error < 0.1, "reconstruction error should be small, got {error}");
    }

    #[test]
    fn test_memd_empty_signal() {
        let signal: Vec<Vec<f64>> = vec![];
        let config = MemdConfig::new(DirectionConfig::new(8), SiftingConfig::default());
        let result = memd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::EmptySignal));
    }

    #[test]
    fn test_memd_insufficient_samples() {
        let signal = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let config = MemdConfig::new(DirectionConfig::new(8), SiftingConfig::default());
        let result = memd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_memd_dimension_mismatch() {
        let signal = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        let config = MemdConfig::new(DirectionConfig::new(8), SiftingConfig::default());
        let result = memd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::DimensionMismatch));
    }

    #[test]
    fn test_memd_invalid_value() {
        let signal = vec![vec![1.0, f64::NAN, 3.0], vec![4.0, 5.0, 6.0]];
        let config = MemdConfig::new(DirectionConfig::new(8), SiftingConfig::default());
        let result = memd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_memd_single_channel_rejected() {
        let signal = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let config = MemdConfig::new(DirectionConfig::new(8), SiftingConfig::default());
        let result = memd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_memd_config_defaults() {
        let dir_config = DirectionConfig::new(16);
        let sifting_config = SiftingConfig::default();
        let config = MemdConfig::new(dir_config, sifting_config);
        assert_eq!(config.max_imfs, 0);
    }

    #[test]
    fn test_memd_config_with_max_imfs() {
        let dir_config = DirectionConfig::new(16);
        let sifting_config = SiftingConfig::default();
        let config = MemdConfig::new(dir_config, sifting_config).with_max_imfs(5);
        assert_eq!(config.max_imfs, 5);
    }

    // ─── Additional utility tests ───────────────────────────────────────

    #[test]
    fn test_compute_multivariate_sd_identical() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let sd = compute_multivariate_sd(&a, &b);
        assert!(sd < TOLERANCE, "SD of identical signals should be ~0, got {sd}");
    }

    #[test]
    fn test_compute_multivariate_sd_different() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![1.1, 2.1, 3.1], vec![4.1, 5.1, 6.1]];
        let sd = compute_multivariate_sd(&a, &b);
        assert!(sd > 0.0, "SD of different signals should be > 0");
        assert!(sd < 1.0, "SD should be small for small differences");
    }

    #[test]
    fn test_count_total_extrema_bivariate() {
        let n = 100;
        let signal: Vec<Vec<f64>> = (0..2)
            .map(|ch| (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect())
            .collect();
        let directions = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let total = count_total_extrema(&signal, &directions);
        assert!(total > 0, "should find extrema in sine wave projections");
    }

    /// Estimate dominant frequency via zero-crossing count.
    fn estimate_dominant_frequency(signal: &[f64], sample_rate: f64) -> f64 {
        let mut crossings = 0usize;
        for i in 1..signal.len() {
            if (signal[i - 1] >= 0.0 && signal[i] < 0.0)
                || (signal[i - 1] < 0.0 && signal[i] >= 0.0)
            {
                crossings += 1;
            }
        }
        // Each cycle has 2 zero crossings
        let n_cycles = crossings as f64 / 2.0;
        let duration = signal.len() as f64 / sample_rate;
        if duration > 0.0 {
            n_cycles / duration
        } else {
            0.0
        }
    }
}
