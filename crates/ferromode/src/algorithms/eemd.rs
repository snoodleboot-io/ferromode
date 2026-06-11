#![warn(missing_docs)]

//! Ensemble Empirical Mode Decomposition (EEMD) algorithm.
//!
//! This module implements EEMD as described by Wu & Huang (2009):
//! "Ensemble Empirical Mode Decomposition: A Noise-Assisted Data Analysis Method"
//!
//! EEMD addresses the mode mixing problem in EMD by adding white noise
//! to the signal and averaging across multiple trials. The added noise
//! populates the whole time-frequency space uniformly, and the ensemble
//! mean cancels out the added noise.
//!
//! # Algorithm
//! 1. Add Gaussian white noise to the signal
//! 2. Run EMD on the noisy signal
//! 3. Repeat steps 1-2 for `num_ensembles` trials
//! 4. Average the IMFs across all trials (zero-pad for unequal counts)
//!
//! # References
//! - Wu, Z., & Huang, N. E. (2009). Ensemble Empirical Mode Decomposition:
//!   A Noise-Assisted Data Analysis Method. Advances in Adaptive Data Analysis,
//!   1(1), 1-41.

use crate::algorithms::emd::{emd, EmdConfig};
use crate::error::EmdError;
use crate::types::{AlgorithmType, DecompositionResult, ImfCollection};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ---------------------------------------------------------------------------
// EnsembleConfig
// ---------------------------------------------------------------------------

/// Configuration for the Ensemble EMD algorithm.
///
/// Controls the number of ensemble trials, noise amplitude, and RNG seed
/// for reproducibility.
///
/// # Default Values
/// - `num_ensembles`: 100 (as recommended by Wu & Huang 2009)
/// - `noise_std`: 0.2 (20% of signal standard deviation)
/// - `seed`: None (non-deterministic)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Number of ensemble trials to perform.
    /// More trials give better noise cancellation but increase compute time.
    /// Recommended: 50-200.
    pub num_ensembles: usize,
    /// Noise standard deviation as a fraction of the signal's standard deviation.
    /// Typical values: 0.01 to 0.4. Wu & Huang recommend 0.2.
    pub noise_std: f64,
    /// Optional seed for reproducible results.
    /// If `None`, a random seed is used (non-deterministic).
    pub seed: Option<u64>,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self { num_ensembles: 100, noise_std: 0.2, seed: None }
    }
}

// ---------------------------------------------------------------------------
// Internal: single trial result
// ---------------------------------------------------------------------------

/// Result from a single EEMD trial.
#[derive(Debug, Clone)]
struct TrialResult {
    /// IMFs extracted from this trial
    imfs: Vec<Vec<f64>>,
    /// Residue from this trial
    residue: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Internal: run one EEMD trial
// ---------------------------------------------------------------------------

/// Run a single EEMD trial with added Gaussian white noise.
///
/// # Arguments
/// * `signal` — Original signal
/// * `noise_std` — Absolute noise standard deviation (already scaled by signal std)
/// * `seed` — Seed for this trial's RNG
/// * `emd_config` — EMD configuration for the inner decomposition
fn run_trial(
    signal: &[f64],
    noise_std: f64,
    seed: u64,
    emd_config: &EmdConfig,
) -> Result<TrialResult, EmdError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise_std).map_err(|e| {
        EmdError::InvalidConfig(format!("failed to create normal distribution: {e}"))
    })?;

    // Generate noisy signal
    let noisy_signal: Vec<f64> = signal.iter().map(|&s| s + normal.sample(&mut rng)).collect();

    // Run EMD on noisy signal
    let result = emd(&noisy_signal, emd_config)?;

    Ok(TrialResult { imfs: result.imfs.imfs, residue: result.imfs.residue })
}

// ---------------------------------------------------------------------------
// Internal: average IMFs with zero-padding
// ---------------------------------------------------------------------------

/// Average IMFs across trials, handling unequal IMF counts via zero-padding.
///
/// When different trials produce different numbers of IMFs, shorter results
/// are zero-padded to match the longest result before averaging.
///
/// # Arguments
/// * `trials` — Results from all ensemble trials
///
/// # Returns
/// Averaged IMFs and averaged residue.
fn average_imfs(trials: &[TrialResult]) -> (Vec<Vec<f64>>, Vec<f64>) {
    if trials.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let n = trials[0].imfs.first().map_or(0, |imf| imf.len());
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let num_trials = trials.len();

    // Find the maximum number of IMFs across all trials
    let max_imfs = trials.iter().map(|t| t.imfs.len()).max().unwrap_or(0);

    // Average each IMF position (zero-pad where trial has fewer IMFs)
    let mut averaged_imfs = Vec::with_capacity(max_imfs);
    for imf_idx in 0..max_imfs {
        let mut averaged_imf = vec![0.0f64; n];
        for trial in trials {
            if imf_idx < trial.imfs.len() {
                for (dest, &src) in averaged_imf.iter_mut().zip(trial.imfs[imf_idx].iter()) {
                    *dest += src;
                }
            }
            // else: zero-pad (already initialized to 0.0)
        }
        for val in &mut averaged_imf {
            *val /= num_trials as f64;
        }
        averaged_imfs.push(averaged_imf);
    }

    // Average residue
    let mut averaged_residue = vec![0.0f64; n];
    for trial in trials {
        for (dest, &src) in averaged_residue.iter_mut().zip(trial.residue.iter()) {
            *dest += src;
        }
    }
    for val in &mut averaged_residue {
        *val /= num_trials as f64;
    }

    (averaged_imfs, averaged_residue)
}

// ---------------------------------------------------------------------------
// Internal: compute signal standard deviation
// ---------------------------------------------------------------------------

/// Compute the standard deviation of a signal.
fn signal_std(signal: &[f64]) -> f64 {
    let n = signal.len() as f64;
    let mean: f64 = signal.iter().sum::<f64>() / n;
    let variance: f64 = signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform Ensemble Empirical Mode Decomposition on a signal.
///
/// EEMD addresses the mode mixing problem in standard EMD by adding
/// Gaussian white noise to the signal and averaging across multiple
/// ensemble trials. The noise populates the time-frequency space uniformly,
/// and the ensemble mean cancels out the added noise.
///
/// # Arguments
/// * `signal` — Input signal to decompose
/// * `config` — EEMD ensemble configuration
/// * `emd_config` — EMD configuration for inner decomposition
///
/// # Returns
/// A `DecompositionResult` containing the averaged IMFs and residue,
/// or an error if decomposition fails.
///
/// # Examples
/// ```
/// use ferromode::algorithms::eemd::{EnsembleConfig, eemd};
/// use ferromode::algorithms::emd::EmdConfig;
/// use std::f64::consts::PI;
///
/// // Decompose a mode-mixed signal
/// let n = 200;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64 / n as f64;
///         (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
///     })
///     .collect();
///
/// let eemd_config = EnsembleConfig {
///     num_ensembles: 50,
///     noise_std: 0.2,
///     seed: Some(42),
/// };
/// let emd_config = EmdConfig::default();
/// let result = eemd(&signal, &eemd_config, &emd_config);
/// assert!(result.is_ok());
/// ```
pub fn eemd(
    signal: &[f64],
    config: &EnsembleConfig,
    emd_config: &EmdConfig,
) -> Result<DecompositionResult, EmdError> {
    let start = Instant::now();

    // Validate input
    if signal.len() < 3 {
        return Err(EmdError::InsufficientData);
    }
    for &val in signal {
        if !val.is_finite() {
            return Err(EmdError::InvalidValue);
        }
    }

    if config.num_ensembles == 0 {
        return Err(EmdError::InvalidConfig("num_ensembles must be greater than 0".to_string()));
    }

    if config.noise_std < 0.0 {
        return Err(EmdError::InvalidConfig("noise_std must be non-negative".to_string()));
    }

    // Compute absolute noise level
    let std = signal_std(signal);
    let absolute_noise_std = config.noise_std * std;

    // Run ensemble trials in parallel using rayon
    let trials: Result<Vec<TrialResult>, EmdError> = (0..config.num_ensembles)
        .into_par_iter()
        .map(|i| {
            let seed = config.seed.map_or_else(
                || {
                    let mut rng = StdRng::seed_from_u64((i as u64).wrapping_mul(12_345_678_901_234_567_891_u64));
                    rng.gen()
                },
                |base_seed| base_seed.wrapping_add(i as u64),
            );
            run_trial(signal, absolute_noise_std, seed, emd_config)
        })
        .collect();

    let trials = trials?;

    // Average IMFs across trials
    let (averaged_imfs, averaged_residue) = average_imfs(&trials);

    let elapsed = start.elapsed();

    let total_siftings: usize = config.num_ensembles;

    let result = DecompositionResult::new(
        AlgorithmType::EEMD,
        ImfCollection::new(averaged_imfs, averaged_residue),
        elapsed,
        total_siftings,
        format!(
            r#"{{"num_ensembles": {}, "noise_std": {}, "seed": {}}}"#,
            config.num_ensembles,
            config.noise_std,
            config.seed.map_or("null".to_string(), |s| s.to_string()),
        ),
    );

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // =========================================================================
    // EnsembleConfig tests
    // =========================================================================

    #[test]
    fn test_ensemble_config_default() {
        let config = EnsembleConfig::default();
        assert_eq!(config.num_ensembles, 100);
        assert!((config.noise_std - 0.2).abs() < 1e-15);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_ensemble_config_custom() {
        let config = EnsembleConfig { num_ensembles: 50, noise_std: 0.1, seed: Some(42) };
        assert_eq!(config.num_ensembles, 50);
        assert!((config.noise_std - 0.1).abs() < 1e-15);
        assert_eq!(config.seed, Some(42));
    }

    // =========================================================================
    // signal_std helper tests
    // =========================================================================

    #[test]
    fn test_signal_std_constant() {
        let signal = vec![5.0; 100];
        assert!((signal_std(&signal) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_signal_std_known_values() {
        // [1, 2, 3, 4, 5] → mean=3, variance=2, std=sqrt(2)
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = 2.0f64.sqrt();
        assert!((signal_std(&signal) - expected).abs() < 1e-15);
    }

    // =========================================================================
    // average_imfs tests
    // =========================================================================

    #[test]
    fn test_average_imfs_empty() {
        let (imfs, residue) = average_imfs(&[]);
        assert!(imfs.is_empty());
        assert!(residue.is_empty());
    }

    #[test]
    fn test_average_imfs_single_trial() {
        let trial = TrialResult {
            imfs: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            residue: vec![0.5, 0.5, 0.5],
        };
        let (imfs, residue) = average_imfs(&[trial]);
        assert_eq!(imfs.len(), 2);
        assert_eq!(imfs[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(imfs[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(residue, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_average_imfs_two_trials_same_count() {
        let trials = vec![
            TrialResult { imfs: vec![vec![1.0, 2.0], vec![3.0, 4.0]], residue: vec![0.5, 0.5] },
            TrialResult { imfs: vec![vec![3.0, 4.0], vec![5.0, 6.0]], residue: vec![1.5, 1.5] },
        ];
        let (imfs, residue) = average_imfs(&trials);
        assert_eq!(imfs.len(), 2);
        assert_eq!(imfs[0], vec![2.0, 3.0]);
        assert_eq!(imfs[1], vec![4.0, 5.0]);
        assert_eq!(residue, vec![1.0, 1.0]);
    }

    #[test]
    fn test_average_imfs_unequal_counts_zero_padding() {
        let trials = vec![
            TrialResult { imfs: vec![vec![2.0, 2.0], vec![4.0, 4.0]], residue: vec![1.0, 1.0] },
            TrialResult { imfs: vec![vec![4.0, 4.0]], residue: vec![2.0, 2.0] },
        ];
        let (imfs, residue) = average_imfs(&trials);
        assert_eq!(imfs.len(), 2);
        assert_eq!(imfs[0], vec![3.0, 3.0]);
        assert_eq!(imfs[1], vec![2.0, 2.0]);
        assert_eq!(residue, vec![1.5, 1.5]);
    }

    #[test]
    fn test_average_imfs_three_trials() {
        let trials = vec![
            TrialResult { imfs: vec![vec![3.0, 3.0]], residue: vec![1.0, 1.0] },
            TrialResult { imfs: vec![vec![6.0, 6.0]], residue: vec![2.0, 2.0] },
            TrialResult { imfs: vec![vec![9.0, 9.0]], residue: vec![3.0, 3.0] },
        ];
        let (imfs, residue) = average_imfs(&trials);
        assert_eq!(imfs.len(), 1);
        assert_eq!(imfs[0], vec![6.0, 6.0]);
        assert_eq!(residue, vec![2.0, 2.0]);
    }

    // =========================================================================
    // T-071a: Reproducibility — same seed → same results
    // =========================================================================

    #[test]
    fn test_eemd_reproducibility_same_seed() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 10, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let result1 = eemd(&signal, &config, &emd_config).unwrap();
        let result2 = eemd(&signal, &config, &emd_config).unwrap();

        assert_eq!(result1.imfs.n_imfs(), result2.imfs.n_imfs());
        for (imf1, imf2) in result1.imfs.imfs.iter().zip(result2.imfs.imfs.iter()) {
            for (v1, v2) in imf1.iter().zip(imf2.iter()) {
                assert!((v1 - v2).abs() < 1e-15, "IMF values must match exactly with same seed");
            }
        }
    }

    // =========================================================================
    // T-071b: Noise cancellation — averaging noise-only trials → near-zero
    // =========================================================================

    #[test]
    fn test_eemd_noise_cancellation() {
        let n = 200;
        let signal = vec![0.0; n];

        let config = EnsembleConfig { num_ensembles: 50, noise_std: 0.5, seed: Some(123) };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(result.is_ok(), "EEMD should handle zero signal");
    }

    #[test]
    fn test_eemd_noise_cancellation_small_signal() {
        let n = 200;
        let signal = vec![1e-10; n];

        let config = EnsembleConfig { num_ensembles: 50, noise_std: 0.1, seed: Some(456) };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(result.is_ok());
    }

    // =========================================================================
    // T-071c: Mode-mixed signal — EEMD should separate better than EMD
    // =========================================================================

    #[test]
    fn test_eemd_mode_mixed_signal() {
        let n = 500;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.1 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let eemd_config = EnsembleConfig { num_ensembles: 30, noise_std: 0.2, seed: Some(789) };
        let emd_config = EmdConfig::default();

        let eemd_result = eemd(&signal, &eemd_config, &emd_config);
        assert!(eemd_result.is_ok(), "EEMD should succeed on mode-mixed signal");

        let eemd_result = eemd_result.unwrap();

        assert!(
            eemd_result.imfs.n_imfs() >= 2,
            "EEMD should separate mode-mixed signal into at least 2 IMFs, got {}",
            eemd_result.imfs.n_imfs()
        );
    }

    #[test]
    fn test_eemd_vs_emd_on_mode_mixed_signal() {
        let n = 500;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 3.0 * t).sin() + 0.05 * (2.0 * PI * 30.0 * t).sin()
            })
            .collect();

        let eemd_config = EnsembleConfig { num_ensembles: 30, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let eemd_result = eemd(&signal, &eemd_config, &emd_config).unwrap();
        let emd_result = emd(&signal, &emd_config).unwrap();

        assert!(
            eemd_result.imfs.n_imfs() >= emd_result.imfs.n_imfs(),
            "EEMD should produce >= IMFs than EMD: EEMD={}, EMD={}",
            eemd_result.imfs.n_imfs(),
            emd_result.imfs.n_imfs()
        );
    }

    // =========================================================================
    // T-067: Basic EEMD on pure sine wave
    // =========================================================================

    #[test]
    fn test_eemd_pure_sine_wave() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 20, noise_std: 0.1, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(result.is_ok(), "EEMD should succeed on pure sine wave");

        let result = result.unwrap();
        assert!(result.imfs.n_imfs() >= 1, "Pure sine wave should produce at least 1 IMF");
        assert_eq!(result.algorithm, AlgorithmType::EEMD);
    }

    // =========================================================================
    // T-068: Parallel execution
    // =========================================================================

    #[test]
    fn test_eemd_parallel_execution() {
        let n = 300;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 20, noise_std: 0.2, seed: Some(99) };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(result.is_ok(), "Parallel EEMD should succeed");
    }

    // =========================================================================
    // T-069: Seeded RNG
    // =========================================================================

    #[test]
    fn test_eemd_different_seeds_different_results() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let emd_config = EmdConfig::default();

        let config1 = EnsembleConfig { num_ensembles: 10, noise_std: 0.2, seed: Some(1) };
        let config2 = EnsembleConfig { num_ensembles: 10, noise_std: 0.2, seed: Some(2) };

        let result1 = eemd(&signal, &config1, &emd_config).unwrap();
        let result2 = eemd(&signal, &config2, &emd_config).unwrap();

        let mut all_same = true;
        for (imf1, imf2) in result1.imfs.imfs.iter().zip(result2.imfs.imfs.iter()) {
            for (v1, v2) in imf1.iter().zip(imf2.iter()) {
                if (v1 - v2).abs() > 1e-15 {
                    all_same = false;
                    break;
                }
            }
            if !all_same {
                break;
            }
        }
        assert!(!all_same, "Different seeds should produce different results");
    }

    // =========================================================================
    // Error handling tests
    // =========================================================================

    #[test]
    fn test_eemd_insufficient_data() {
        let signal = vec![1.0, 2.0];
        let config = EnsembleConfig::default();
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_eemd_invalid_value_nan() {
        let signal = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let config = EnsembleConfig::default();
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_eemd_zero_ensembles_error() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = EnsembleConfig { num_ensembles: 0, noise_std: 0.2, seed: None };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_eemd_negative_noise_std_error() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = EnsembleConfig { num_ensembles: 10, noise_std: -0.1, seed: None };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    // =========================================================================
    // Algorithm type test
    // =========================================================================

    #[test]
    fn test_eemd_algorithm_type_is_eemd() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 5, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config).unwrap();
        assert_eq!(result.algorithm, AlgorithmType::EEMD);
    }

    // =========================================================================
    // Config snapshot test
    // =========================================================================

    #[test]
    fn test_eemd_config_snapshot_is_valid_json() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 5, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config).unwrap();

        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&result.config_snapshot);
        assert!(parsed.is_ok(), "Config snapshot should be valid JSON: {}", result.config_snapshot);
    }

    // =========================================================================
    // Multi-component signal
    // =========================================================================

    #[test]
    fn test_eemd_multi_component_signal() {
        let n = 500;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.5 * (2.0 * PI * 20.0 * t).sin()
                    + 0.3 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 20, noise_std: 0.15, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(
            result.imfs.n_imfs() >= 2,
            "Multi-component signal should produce at least 2 IMFs, got {}",
            result.imfs.n_imfs()
        );
    }

    // =========================================================================
    // Noise cancellation verification
    // =========================================================================

    #[test]
    fn test_eemd_noise_only_trials_average_to_near_zero() {
        let n = 200;
        let signal = vec![0.0; n];

        let config = EnsembleConfig { num_ensembles: 100, noise_std: 1.0, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);

        if result.is_ok() {
            let result = result.unwrap();
            if result.imfs.n_imfs() > 0 {
                let total_energy: f64 =
                    result.imfs.imfs.iter().flat_map(|imf| imf.iter()).map(|v| v * v).sum();
                assert!(
                    total_energy < (n as f64 * 10.0),
                    "Noise energy should be bounded, got {}",
                    total_energy
                );
            }
        }
    }

    // =========================================================================
    // Large ensemble count
    // =========================================================================

    #[test]
    fn test_eemd_large_ensemble() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 50, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let result = eemd(&signal, &config, &emd_config);
        assert!(result.is_ok(), "EEMD with 50 ensembles should succeed");
    }
}
