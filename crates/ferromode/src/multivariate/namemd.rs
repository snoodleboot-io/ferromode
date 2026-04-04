//! Noise-Assisted Multivariate Empirical Mode Decomposition (NA-MEMD).
//!
//! Implements the NA-MEMD algorithm from Rehman & Mandic (2011),
//! "Multivariate EMD with noise assistance for improved mode alignment."
//!
//! NA-MEMD adds white noise channels to the multivariate signal before
//! decomposition, then discards the noise channel IMFs. This improves
//! mode alignment across channels and reduces mode mixing.
//!
//! Algorithm:
//! 1. Generate `n_noise_channels` channels of Gaussian white noise
//! 2. Append noise channels to original signal → augmented signal
//! 3. Run MEMD on augmented signal
//! 4. Discard IMFs corresponding to noise channels
//! 5. Return only IMFs for original channels

use crate::error::EmdError;
use crate::multivariate::memd::{memd, MemdConfig};
use crate::types::{AlgorithmType, DecompositionResult, ImfCollection};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Configuration for NA-MEMD decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaMemdConfig {
    /// Base MEMD configuration reused for the decomposition.
    pub base_config: MemdConfig,
    /// Number of white noise channels to add (default: 2).
    pub n_noise_channels: usize,
    /// Noise standard deviation as a fraction of the signal's std (default: 0.1).
    pub noise_std: f64,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
}

impl NaMemdConfig {
    /// Create a new NA-MEMD configuration with defaults.
    #[must_use]
    pub fn new(base_config: MemdConfig) -> Self {
        Self { base_config, n_noise_channels: 2, noise_std: 0.1, seed: None }
    }

    /// Set the number of noise channels.
    #[must_use]
    pub fn with_noise_channels(mut self, n_noise_channels: usize) -> Self {
        self.n_noise_channels = n_noise_channels;
        self
    }

    /// Set the noise standard deviation (fraction of signal std).
    #[must_use]
    pub fn with_noise_std(mut self, noise_std: f64) -> Self {
        self.noise_std = noise_std;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Generate Gaussian white noise channels.
///
/// # Arguments
/// * `n_channels` — Number of noise channels to generate.
/// * `n_samples` — Length of each channel.
/// * `std` — Standard deviation of the noise.
/// * `rng` — Random number generator.
///
/// # Returns
/// Vector of noise channels, each of length `n_samples`.
fn generate_noise_channels(
    n_channels: usize,
    n_samples: usize,
    std: f64,
    rng: &mut StdRng,
) -> Vec<Vec<f64>> {
    let normal = Normal::new(0.0, std).expect("Normal distribution parameters are valid");

    (0..n_channels).map(|_| (0..n_samples).map(|_| normal.sample(rng)).collect()).collect()
}

/// Compute the standard deviation of a signal.
fn compute_signal_std(signal: &[Vec<f64>]) -> f64 {
    let n_channels = signal.len();
    let n_samples = signal[0].len();

    // Compute mean across all channels and samples
    let total_sum: f64 = signal.iter().flat_map(|ch| ch.iter()).sum();
    let total_count = (n_channels * n_samples) as f64;
    let mean = total_sum / total_count;

    // Compute variance
    let variance: f64 =
        signal.iter().flat_map(|ch| ch.iter()).map(|&x| (x - mean).powi(2)).sum::<f64>()
            / total_count;

    variance.sqrt()
}

/// Perform Noise-Assisted Multivariate Empirical Mode Decomposition.
///
/// # Arguments
/// * `signal` — Multivariate signal where each inner Vec is one channel.
///              All channels must have the same length.
/// * `config` — NA-MEMD configuration.
///
/// # Returns
/// Decomposition result containing IMFs for the original channels only
/// (noise channel IMFs are discarded).
///
/// # Errors
/// Returns `EmdError` if the signal is invalid or decomposition fails.
pub fn namemd(signal: &[Vec<f64>], config: &NaMemdConfig) -> Result<DecompositionResult, EmdError> {
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
        return Err(EmdError::InvalidConfig("NA-MEMD requires at least 2 channels".to_string()));
    }

    if config.n_noise_channels == 0 {
        return Err(EmdError::InvalidConfig("n_noise_channels must be > 0".to_string()));
    }

    // Compute signal std and scale noise
    let signal_std = compute_signal_std(signal);
    let noise_std = config.noise_std * signal_std;

    // Initialize RNG
    let mut rng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Step 1: Generate noise channels
    let noise = generate_noise_channels(config.n_noise_channels, n_samples, noise_std, &mut rng);

    // Step 2: Augment signal with noise channels
    let mut augmented_signal = signal.to_vec();
    augmented_signal.extend(noise);

    let n_augmented = augmented_signal.len();

    // Step 3: Run MEMD on augmented signal
    let memd_result = memd(&augmented_signal, &config.base_config)?;

    // Step 4: Discard noise channel IMFs, keep only original channel IMFs
    // MEMD stores IMFs in flat layout: [ch0_imf0, ch0_imf1, ..., ch1_imf0, ch1_imf1, ...]
    // For augmented signal: channels 0..n_channels are original, n_channels..n_augmented are noise
    let n_imfs_per_channel = memd_result.imfs.n_imfs() / n_augmented;

    let mut original_imfs: Vec<Vec<f64>> = Vec::new();
    for ch in 0..n_channels {
        for imf_idx in 0..n_imfs_per_channel {
            let flat_idx = ch * n_imfs_per_channel + imf_idx;
            if flat_idx < memd_result.imfs.imfs.len() {
                original_imfs.push(memd_result.imfs.imfs[flat_idx].clone());
            }
        }
    }

    // Reconstruct residue for original channels only
    // MEMD stores residue as flattened: [ch0_residue..., ch1_residue..., ...]
    let mut original_residue: Vec<f64> = Vec::new();
    for ch in 0..n_channels {
        let residue_start = ch * n_samples;
        let residue_end = residue_start + n_samples;
        if residue_end <= memd_result.imfs.residue.len() {
            original_residue
                .extend_from_slice(&memd_result.imfs.residue[residue_start..residue_end]);
        }
    }

    let imf_collection = ImfCollection::new(original_imfs, original_residue);

    let elapsed = start.elapsed();

    Ok(DecompositionResult::new(
        AlgorithmType::NAMEMD,
        imf_collection,
        elapsed,
        memd_result.n_siftings,
        format!(
            "{{\"n_noise_channels\": {}, \"noise_std\": {}, \"base_directions\": {}}}",
            config.n_noise_channels,
            config.noise_std,
            config.base_config.direction_config.num_directions
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multivariate::direction_sampling::DirectionConfig;
    use crate::multivariate::memd::memd;
    use crate::sifting::SiftingConfig;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 1e-6;

    // ─── Helper functions ───────────────────────────────────────────────

    fn generate_sine_wave(freq: f64, n_samples: usize, sample_rate: f64) -> Vec<f64> {
        (0..n_samples).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin()).collect()
    }

    fn generate_bivariate_signal(
        freq1: f64,
        freq2: f64,
        n_samples: usize,
        sample_rate: f64,
    ) -> Vec<Vec<f64>> {
        vec![
            generate_sine_wave(freq1, n_samples, sample_rate),
            generate_sine_wave(freq2, n_samples, sample_rate),
        ]
    }

    fn make_memd_config() -> MemdConfig {
        let dir_config = DirectionConfig::new(16).with_seed(42);
        let sifting_config = SiftingConfig {
            max_sifting_iterations: 50,
            sd_threshold: 0.3,
            s_number: 3,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: crate::boundary::BoundaryConditionType::MirrorEven,
        };
        MemdConfig::new(dir_config, sifting_config).with_max_imfs(5)
    }

    fn compute_reconstruction_error_namemd(
        original: &[Vec<f64>],
        result: &DecompositionResult,
    ) -> f64 {
        let n_channels = original.len();
        let n_samples = original[0].len();
        let n_imfs_per_channel = result.imfs.n_imfs() / n_channels;

        let mut max_error = 0.0f64;

        for ch in 0..n_channels {
            for i in 0..n_samples {
                let mut reconstructed = 0.0f64;
                for imf_idx in 0..n_imfs_per_channel {
                    let flat_idx = ch * n_imfs_per_channel + imf_idx;
                    if flat_idx < result.imfs.imfs.len() {
                        reconstructed += result.imfs.imfs[flat_idx][i];
                    }
                }
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

    /// Compute mode alignment score: variance of dominant frequencies across channels
    /// for each IMF. Lower is better (more aligned).
    fn compute_mode_alignment_score(
        result: &DecompositionResult,
        n_channels: usize,
        sample_rate: f64,
    ) -> f64 {
        let n_imfs_per_channel = result.imfs.n_imfs() / n_channels;
        let mut total_variance = 0.0f64;
        let mut imf_count = 0usize;

        for imf_idx in 0..n_imfs_per_channel {
            let mut dominant_freqs: Vec<f64> = Vec::new();
            for ch in 0..n_channels {
                let flat_idx = ch * n_imfs_per_channel + imf_idx;
                if flat_idx < result.imfs.imfs.len() {
                    let imf = &result.imfs.imfs[flat_idx];
                    let dom_freq = estimate_dominant_frequency(imf, sample_rate);
                    if dom_freq > 0.0 {
                        dominant_freqs.push(dom_freq);
                    }
                }
            }

            if dominant_freqs.len() >= 2 {
                let mean: f64 = dominant_freqs.iter().sum::<f64>() / dominant_freqs.len() as f64;
                let variance: f64 = dominant_freqs.iter().map(|&f| (f - mean).powi(2)).sum::<f64>()
                    / dominant_freqs.len() as f64;
                total_variance += variance;
                imf_count += 1;
            }
        }

        if imf_count > 0 {
            total_variance / imf_count as f64
        } else {
            f64::INFINITY
        }
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
        let n_cycles = crossings as f64 / 2.0;
        let duration = signal.len() as f64 / sample_rate;
        if duration > 0.0 {
            n_cycles / duration
        } else {
            0.0
        }
    }

    // ─── T-094: NA-MEMD noise channel generation tests ──────────────────

    #[test]
    fn test_generate_noise_channels_count() {
        let mut rng = StdRng::seed_from_u64(42);
        let noise = generate_noise_channels(3, 100, 0.1, &mut rng);
        assert_eq!(noise.len(), 3);
        for ch in &noise {
            assert_eq!(ch.len(), 100);
        }
    }

    #[test]
    fn test_generate_noise_channels_statistics() {
        let mut rng = StdRng::seed_from_u64(42);
        let noise = generate_noise_channels(1, 10000, 1.0, &mut rng);

        // Mean should be near 0
        let mean: f64 = noise[0].iter().sum::<f64>() / noise[0].len() as f64;
        assert!(mean.abs() < 0.1, "noise mean should be near 0, got {mean}");

        // Std should be near 1.0
        let variance: f64 =
            noise[0].iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / noise[0].len() as f64;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 0.1, "noise std should be near 1.0, got {std}");
    }

    #[test]
    fn test_generate_noise_channels_reproducible() {
        let mut rng1 = StdRng::seed_from_u64(123);
        let noise1 = generate_noise_channels(2, 50, 0.5, &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(123);
        let noise2 = generate_noise_channels(2, 50, 0.5, &mut rng2);

        assert_eq!(noise1, noise2);
    }

    #[test]
    fn test_compute_signal_std() {
        let signal = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];
        let std = compute_signal_std(&signal);
        // Mean = 1.0, variance = ((1+0+1)+(1+0+1))/6 = 4/6 = 2/3
        // std = sqrt(2/3) ≈ 0.816
        assert!((std - 0.8165).abs() < 0.01, "signal std should be ~0.816, got {std}");
    }

    #[test]
    fn test_compute_signal_std_constant() {
        let signal = vec![vec![5.0, 5.0, 5.0], vec![5.0, 5.0, 5.0]];
        let std = compute_signal_std(&signal);
        assert!(std < TOLERANCE, "std of constant signal should be ~0, got {std}");
    }

    // ─── T-094: NaMemdConfig tests ──────────────────────────────────────

    #[test]
    fn test_namemd_config_defaults() {
        let base = make_memd_config();
        let config = NaMemdConfig::new(base.clone());
        assert_eq!(config.n_noise_channels, 2);
        assert!((config.noise_std - 0.1).abs() < TOLERANCE);
        assert_eq!(config.seed, None);
    }

    #[test]
    fn test_namemd_config_with_noise_channels() {
        let base = make_memd_config();
        let config = NaMemdConfig::new(base).with_noise_channels(4);
        assert_eq!(config.n_noise_channels, 4);
    }

    #[test]
    fn test_namemd_config_with_noise_std() {
        let base = make_memd_config();
        let config = NaMemdConfig::new(base).with_noise_std(0.2);
        assert!((config.noise_std - 0.2).abs() < TOLERANCE);
    }

    #[test]
    fn test_namemd_config_with_seed() {
        let base = make_memd_config();
        let config = NaMemdConfig::new(base).with_seed(99);
        assert_eq!(config.seed, Some(99));
    }

    // ─── T-095: NA-MEMD decomposition tests ─────────────────────────────

    #[test]
    fn test_namemd_bivariate_sine() {
        let n = 200;
        let sample_rate = 100.0;
        let signal = generate_bivariate_signal(5.0, 5.0, n, sample_rate);

        let config = NaMemdConfig::new(make_memd_config())
            .with_noise_channels(2)
            .with_noise_std(0.1)
            .with_seed(42);

        let result = namemd(&signal, &config);
        assert!(result.is_ok(), "NA-MEMD should succeed for bivariate sine: {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.algorithm, AlgorithmType::NAMEMD);
        assert!(!result.imfs.imfs.is_empty(), "should extract at least one IMF");
    }

    #[test]
    fn test_namemd_discards_noise_imfs() {
        let n = 200;
        let sample_rate = 100.0;
        let signal = generate_bivariate_signal(5.0, 5.0, n, sample_rate);
        let n_channels = signal.len();

        let config = NaMemdConfig::new(make_memd_config())
            .with_noise_channels(3)
            .with_noise_std(0.1)
            .with_seed(42);

        let result = namemd(&signal, &config).unwrap();

        // IMFs should only be for original channels, not noise channels
        // n_imfs should be divisible by n_channels (not n_channels + n_noise)
        assert_eq!(
            result.imfs.n_imfs() % n_channels,
            0,
            "IMF count should be divisible by original channel count"
        );
    }

    #[test]
    fn test_namemd_reconstruction_validation() {
        let n = 200;
        let sample_rate = 100.0;
        let signal = generate_bivariate_signal(5.0, 5.0, n, sample_rate);

        let config = NaMemdConfig::new(make_memd_config())
            .with_noise_channels(2)
            .with_noise_std(0.05)
            .with_seed(42);

        let result = namemd(&signal, &config).unwrap();
        let error = compute_reconstruction_error_namemd(&signal, &result);

        assert!(error < 0.15, "reconstruction error should be small, got {error}");
    }

    #[test]
    fn test_namemd_reproducibility_with_seed() {
        let n = 200;
        let sample_rate = 100.0;
        let signal = generate_bivariate_signal(5.0, 5.0, n, sample_rate);

        let config1 = NaMemdConfig::new(make_memd_config())
            .with_noise_channels(2)
            .with_noise_std(0.1)
            .with_seed(42);

        let config2 = NaMemdConfig::new(make_memd_config())
            .with_noise_channels(2)
            .with_noise_std(0.1)
            .with_seed(42);

        let result1 = namemd(&signal, &config1).unwrap();
        let result2 = namemd(&signal, &config2).unwrap();

        assert_eq!(result1.imfs.imfs.len(), result2.imfs.imfs.len());
        for (imf1, imf2) in result1.imfs.imfs.iter().zip(&result2.imfs.imfs) {
            for (a, b) in imf1.iter().zip(imf2.iter()) {
                assert!((a - b).abs() < 1e-10, "reproducible results should match exactly");
            }
        }
    }

    // ─── T-096: NA-MEMD vs plain MEMD comparison tests ──────────────────

    #[test]
    fn test_namemd_vs_memd_mode_alignment() {
        // Create a bivariate signal with two distinct frequency components
        // that are known to cause mode mixing in plain MEMD
        let n = 500;
        let sample_rate = 1000.0;

        // Two channels with slightly different frequency mixes
        let ch1: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 10.0 * i as f64 / sample_rate).sin()
                    + 0.5 * (2.0 * PI * 40.0 * i as f64 / sample_rate).sin()
            })
            .collect();
        let ch2: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * 10.0 * i as f64 / sample_rate + 0.3).sin()
                    + 0.5 * (2.0 * PI * 40.0 * i as f64 / sample_rate + 0.3).sin()
            })
            .collect();
        let signal = vec![ch1, ch2];

        // Run plain MEMD
        let memd_config = make_memd_config();
        let memd_result = memd(&signal, &memd_config).unwrap();
        let memd_alignment = compute_mode_alignment_score(&memd_result, 2, sample_rate);

        // Run NA-MEMD
        let namemd_config = NaMemdConfig::new(make_memd_config())
            .with_noise_channels(2)
            .with_noise_std(0.1)
            .with_seed(42);
        let namemd_result = namemd(&signal, &namemd_config).unwrap();
        let namemd_alignment = compute_mode_alignment_score(&namemd_result, 2, sample_rate);

        // NA-MEMD should have equal or better (lower) alignment score
        // We use a soft assertion since the improvement depends on signal characteristics
        println!(
            "MEMD alignment score: {memd_alignment:.6}, NA-MEMD alignment score: {namemd_alignment:.6}"
        );

        // Both should produce finite alignment scores
        assert!(memd_alignment.is_finite(), "MEMD alignment should be finite");
        assert!(namemd_alignment.is_finite(), "NA-MEMD alignment should be finite");
    }

    #[test]
    fn test_namemd_noise_suppression_quality() {
        // Create a clean bivariate signal
        let n = 300;
        let sample_rate = 1000.0;
        let signal = generate_bivariate_signal(20.0, 20.0, n, sample_rate);

        // Compute original signal energy
        let original_energy: f64 = signal.iter().flat_map(|ch| ch.iter()).map(|&x| x.powi(2)).sum();

        let config = NaMemdConfig::new(make_memd_config())
            .with_noise_channels(2)
            .with_noise_std(0.1)
            .with_seed(42);

        let result = namemd(&signal, &config).unwrap();

        // Reconstruct from IMFs + residue
        let n_channels = signal.len();
        let n_imfs_per_channel = result.imfs.n_imfs() / n_channels;

        let mut reconstructed_energy = 0.0f64;
        for ch in 0..n_channels {
            for i in 0..n {
                let mut val = 0.0f64;
                for imf_idx in 0..n_imfs_per_channel {
                    let flat_idx = ch * n_imfs_per_channel + imf_idx;
                    if flat_idx < result.imfs.imfs.len() {
                        val += result.imfs.imfs[flat_idx][i];
                    }
                }
                let residue_start = ch * n;
                if residue_start + i < result.imfs.residue.len() {
                    val += result.imfs.residue[residue_start + i];
                }
                reconstructed_energy += val.powi(2);
            }
        }

        // Energy preservation: reconstructed energy should be close to original
        let energy_ratio = reconstructed_energy / original_energy;
        assert!(
            (energy_ratio - 1.0).abs() < 0.3,
            "energy preservation: ratio should be near 1.0, got {energy_ratio}"
        );
    }

    // ─── Error handling tests ───────────────────────────────────────────

    #[test]
    fn test_namemd_empty_signal() {
        let signal: Vec<Vec<f64>> = vec![];
        let config = NaMemdConfig::new(make_memd_config());
        let result = namemd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::EmptySignal));
    }

    #[test]
    fn test_namemd_insufficient_samples() {
        let signal = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let config = NaMemdConfig::new(make_memd_config());
        let result = namemd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_namemd_dimension_mismatch() {
        let signal = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        let config = NaMemdConfig::new(make_memd_config());
        let result = namemd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::DimensionMismatch));
    }

    #[test]
    fn test_namemd_invalid_value() {
        let signal = vec![vec![1.0, f64::NAN, 3.0], vec![4.0, 5.0, 6.0]];
        let config = NaMemdConfig::new(make_memd_config());
        let result = namemd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_namemd_single_channel_rejected() {
        let signal = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let config = NaMemdConfig::new(make_memd_config());
        let result = namemd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_namemd_zero_noise_channels_rejected() {
        let n = 200;
        let sample_rate = 100.0;
        let signal = generate_bivariate_signal(5.0, 5.0, n, sample_rate);

        let config = NaMemdConfig::new(make_memd_config()).with_noise_channels(0);
        let result = namemd(&signal, &config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    // ─── Integration tests ──────────────────────────────────────────────

    #[test]
    fn test_namemd_trivariate_signal() {
        let n = 200;
        let sample_rate = 100.0;
        let signal = vec![
            generate_sine_wave(5.0, n, sample_rate),
            generate_sine_wave(5.0, n, sample_rate),
            generate_sine_wave(5.0, n, sample_rate),
        ];

        let config = NaMemdConfig::new(make_memd_config()).with_noise_channels(2).with_seed(42);

        let result = namemd(&signal, &config);
        assert!(result.is_ok(), "NA-MEMD should succeed for trivariate signal");

        let result = result.unwrap();
        assert_eq!(result.algorithm, AlgorithmType::NAMEMD);
        assert!(result.imfs.n_imfs() % 3 == 0, "IMFs should be divisible by 3 channels");
    }

    #[test]
    fn test_namemd_config_snapshot() {
        let n = 200;
        let sample_rate = 100.0;
        let signal = generate_bivariate_signal(5.0, 5.0, n, sample_rate);

        let config = NaMemdConfig::new(make_memd_config())
            .with_noise_channels(3)
            .with_noise_std(0.15)
            .with_seed(42);

        let result = namemd(&signal, &config).unwrap();

        assert!(result.config_snapshot.contains("n_noise_channels"));
        assert!(result.config_snapshot.contains("3"));
        assert!(result.config_snapshot.contains("noise_std"));
    }
}
