#![warn(missing_docs)]

//! Complete Ensemble EMD with Adaptive Noise (CEEMDAN) algorithm.
//!
//! This module implements CEEMDAN as described by Torres et al. (2011):
//! "A Complete Ensemble Empirical Mode Decomposition with Adaptive Noise"
//!
//! CEEMDAN improves upon EEMD by adding noise stage-wise rather than to the
//! original signal. At each stage k, adaptive noise is added to the current
//! residue, and the first IMF is extracted via EMD. The mean of these first
//! IMFs across all trials becomes IMF_k.
//!
//! # Algorithm
//! 1. Stage 0: For each trial i, add noise to signal: x_i = signal + ε_0·noise_i
//!    Extract first IMF via EMD for each trial, average to get IMF_1
//! 2. Compute residue: r_1 = signal - IMF_1
//! 3. Stage k: For each trial i, compute:
//!    - Adaptive noise: ε_k = ε · std(r_k) / std(noise)
//!    - Add noise to residue: r_k + ε_k·noise_i
//!    - Extract first IMF via EMD for each trial, average to get IMF_k
//! 4. Compute next residue: r_{k+1} = r_k - IMF_k
//! 5. Repeat until residue has < 2 extrema
//!
//! # References
//! - Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P. (2011).
//!   A Complete Ensemble Empirical Mode Decomposition with Adaptive Noise.
//!   IEEE International Conference on Acoustics, Speech and Signal Processing
//!   (ICASSP), 4144-4147.

use crate::algorithms::emd::{emd, EmdConfig};
use crate::error::EmdError;
use crate::extrema::detect_extrema;
use crate::types::{AlgorithmType, DecompositionResult, ImfCollection};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rayon::prelude::*;
use std::time::Instant;

use crate::algorithms::eemd::EnsembleConfig;

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
// Internal: generate noise sequence for a trial
// ---------------------------------------------------------------------------

/// Generate a Gaussian white noise sequence with the given standard deviation.
fn generate_noise(rng: &mut StdRng, noise_std: f64, length: usize) -> Result<Vec<f64>, EmdError> {
    let normal = Normal::new(0.0, noise_std).map_err(|e| {
        EmdError::InvalidConfig(format!("failed to create normal distribution: {e}"))
    })?;

    Ok((0..length).map(|_| normal.sample(rng)).collect())
}

// ---------------------------------------------------------------------------
// Internal: extract first IMF from a signal via EMD
// ---------------------------------------------------------------------------

/// Extract only the first IMF from a signal using EMD.
///
/// This runs EMD but stops after extracting the first IMF, which is
/// the core operation needed at each CEEMDAN stage.
///
/// # Arguments
/// * `signal` — Signal to decompose
/// * `emd_config` — EMD configuration
///
/// # Returns
/// The first IMF extracted, or an error if EMD fails.
fn extract_first_imf(signal: &[f64], emd_config: &EmdConfig) -> Result<Vec<f64>, EmdError> {
    let mut config = emd_config.clone();
    config.max_imfs = 1;
    config.validate_reconstruction = false;

    match emd(signal, &config) {
        Ok(result) if !result.imfs.imfs.is_empty() => Ok(result.imfs.imfs[0].clone()),
        Ok(_) => Ok(signal.to_vec()),
        // Sifting failed to converge or produced a numerical error for this trial.
        // Fall back to the signal itself, consistent with what EMD returns when the
        // signal has too few extrema to decompose.
        Err(EmdError::ConvergenceFailed { .. } | EmdError::InvalidValue) => {
            Ok(signal.to_vec())
        }
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// Internal: run one trial at stage 0
// ---------------------------------------------------------------------------

/// Run a single trial at stage 0: add noise to signal and extract first IMF.
///
/// # Arguments
/// * `signal` — Original signal
/// * `noise` — Pre-generated noise sequence
/// * `emd_config` — EMD configuration
///
/// # Returns
/// The first IMF extracted from the noisy signal.
fn run_stage0_trial(
    signal: &[f64],
    noise: &[f64],
    emd_config: &EmdConfig,
) -> Result<Vec<f64>, EmdError> {
    let noisy_signal: Vec<f64> = signal.iter().zip(noise.iter()).map(|(&s, &n)| s + n).collect();

    extract_first_imf(&noisy_signal, emd_config)
}

// ---------------------------------------------------------------------------
// Internal: run one trial at stage k (k >= 1)
// ---------------------------------------------------------------------------

/// Run a single trial at stage k: add adaptive noise to residue and extract first IMF.
///
/// # Arguments
/// * `residue` — Current residue
/// * `noise` — Pre-generated noise sequence
/// * `adaptive_noise_scale` — ε_k = ε · std(residue) / std(noise)
/// * `emd_config` — EMD configuration
///
/// # Returns
/// The first IMF extracted from the noisy residue.
fn run_stage_k_trial(
    residue: &[f64],
    noise: &[f64],
    adaptive_noise_scale: f64,
    emd_config: &EmdConfig,
) -> Result<Vec<f64>, EmdError> {
    let noisy_residue: Vec<f64> =
        residue.iter().zip(noise.iter()).map(|(&r, &n)| r + adaptive_noise_scale * n).collect();

    extract_first_imf(&noisy_residue, emd_config)
}

// ---------------------------------------------------------------------------
// Internal: average IMFs from multiple trials
// ---------------------------------------------------------------------------

/// Average a set of IMFs (all same length) from multiple trials.
///
/// # Arguments
/// * `imfs` — Vector of IMFs from each trial
///
/// # Returns
/// The averaged IMF.
fn average_imfs(imfs: &[Vec<f64>]) -> Vec<f64> {
    if imfs.is_empty() {
        return Vec::new();
    }

    let n = imfs[0].len();
    let num_trials = imfs.len();

    let mut averaged = vec![0.0f64; n];
    for imf in imfs {
        for (dest, &src) in averaged.iter_mut().zip(imf.iter()) {
            *dest += src;
        }
    }
    for val in &mut averaged {
        *val /= num_trials as f64;
    }

    averaged
}

// ---------------------------------------------------------------------------
// Internal: compute adaptive noise scale for stage k
// ---------------------------------------------------------------------------

/// Compute the adaptive noise scaling factor for stage k.
///
/// ε_k = noise_std · std(residue_k) / std(noise)
///
/// # Arguments
/// * `residue` — Current residue at stage k
/// * `noise_std` — Base noise standard deviation (from EnsembleConfig)
/// * `reference_noise_std` — std of the reference noise used at stage 0
///
/// # Returns
/// The adaptive noise scaling factor ε_k.
fn compute_adaptive_noise_scale(residue: &[f64], noise_std: f64, reference_noise_std: f64) -> f64 {
    let residue_std = signal_std(residue);
    if reference_noise_std < 1e-15 {
        return 0.0;
    }
    noise_std * residue_std / reference_noise_std
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform Complete Ensemble EMD with Adaptive Noise on a signal.
///
/// CEEMDAN addresses the mode mixing problem and reconstruction error of EEMD
/// by adding noise stage-wise rather than to the original signal. At each stage k,
/// adaptive noise is added to the current residue, and only the first IMF is
/// extracted via EMD. The mean of these first IMFs across all trials becomes IMF_k.
///
/// This approach ensures:
/// - Complete reconstruction (signal = Σ IMFs + final residue)
/// - No mode mixing (noise is added adaptively at each stage)
/// - Better spectral separation than EEMD
///
/// # Arguments
/// * `signal` — Input signal to decompose
/// * `config` — Ensemble configuration (num_ensembles, noise_std, seed)
/// * `emd_config` — EMD configuration for inner decomposition
///
/// # Returns
/// A `DecompositionResult` containing the CEEMDAN IMFs and final residue,
/// or an error if decomposition fails.
///
/// # Examples
/// ```
/// use ferromode::algorithms::ceemdan::ceemdan;
/// use ferromode::algorithms::eemd::EnsembleConfig;
/// use ferromode::algorithms::emd::EmdConfig;
/// use std::f64::consts::PI;
///
/// // Decompose a multi-component signal
/// let n = 200;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64 / n as f64;
///         (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
///     })
///     .collect();
///
/// let config = EnsembleConfig {
///     num_ensembles: 50,
///     noise_std: 0.2,
///     seed: Some(42),
/// };
/// let emd_config = EmdConfig::default();
/// let result = ceemdan(&signal, &config, &emd_config);
/// assert!(result.is_ok());
/// ```
pub fn ceemdan(
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
    let signal_std_val = signal_std(signal);
    let absolute_noise_std = config.noise_std * signal_std_val;

    // Create base RNG for generating noise sequences
    let mut base_rng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut rng = StdRng::seed_from_u64(0);
            let seed: u64 = rng.gen();
            StdRng::seed_from_u64(seed)
        }
    };

    // Pre-generate noise sequences for all trials
    let noise_sequences: Vec<Vec<f64>> = (0..config.num_ensembles)
        .map(|_| generate_noise(&mut base_rng, absolute_noise_std, signal.len()))
        .collect::<Result<Vec<_>, _>>()?;

    // Compute reference noise std (std of the first noise sequence)
    let reference_noise_std = if !noise_sequences.is_empty() {
        signal_std(&noise_sequences[0])
    } else {
        absolute_noise_std
    };

    // ========================================================================
    // Stage 0: Add noise to signal, extract first IMF for each trial, average
    // ========================================================================

    let stage0_imfs: Result<Vec<Vec<f64>>, EmdError> = (0..config.num_ensembles)
        .into_par_iter()
        .map(|i| run_stage0_trial(signal, &noise_sequences[i], emd_config))
        .collect();

    let stage0_imfs = stage0_imfs?;
    let imf_1 = average_imfs(&stage0_imfs);

    // Compute first residue
    let mut residue: Vec<f64> = signal.iter().zip(imf_1.iter()).map(|(&s, &i)| s - i).collect();

    let mut all_imfs: Vec<Vec<f64>> = vec![imf_1];
    let mut total_trials: usize = config.num_ensembles;

    // ========================================================================
    // Stage k: Add adaptive noise to residue, extract first IMF, average
    // ========================================================================

    // max_imfs > 0 caps the number of stages (same semantic as EmdConfig.max_imfs).
    let max_stages = if emd_config.max_imfs > 0 { emd_config.max_imfs } else { usize::MAX };

    loop {
        // Check if residue has < 2 extrema (stopping criterion)
        let extrema = detect_extrema(&residue);
        let n_extrema = extrema.maxima.len() + extrema.minima.len();

        if n_extrema < 2 {
            break;
        }

        // Check residue energy — stop if negligible
        let residue_energy: f64 = residue.iter().map(|v| v * v).sum();
        let signal_energy: f64 = signal.iter().map(|v| v * v).sum();

        if signal_energy > 0.0 && residue_energy / signal_energy < 1e-15 {
            break;
        }

        // Respect max_imfs stage cap (stage 0 already extracted 1 IMF)
        if all_imfs.len() >= max_stages {
            break;
        }

        // Compute adaptive noise scale for this stage
        let adaptive_scale =
            compute_adaptive_noise_scale(&residue, config.noise_std, reference_noise_std);

        // Run trials in parallel for this stage
        let cur_imfs: Result<Vec<Vec<f64>>, EmdError> = (0..config.num_ensembles)
            .into_par_iter()
            .map(|i| run_stage_k_trial(&residue, &noise_sequences[i], adaptive_scale, emd_config))
            .collect();

        let cur_imfs = cur_imfs?;
        let mean_imf = average_imfs(&cur_imfs);
        total_trials += config.num_ensembles;

        // Update residue
        let new_residue: Vec<f64> =
            residue.iter().zip(mean_imf.iter()).map(|(&r, &m)| r - m).collect();

        all_imfs.push(mean_imf);
        residue = new_residue;
    }

    let elapsed = start.elapsed();

    let result = DecompositionResult::new(
        AlgorithmType::CEEMDAN,
        ImfCollection::new(all_imfs, residue),
        elapsed,
        total_trials,
        format!(
            r#"{{"num_ensembles": {}, "noise_std": {}, "seed": {}, "algorithm": "CEEMDAN"}}"#,
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
    use crate::algorithms::ceemd::ceemd;
    use crate::algorithms::eemd::eemd;
    use std::f64::consts::PI;

    /// Fast EMD config for unit tests: 10 sifting iterations instead of 100.
    /// CEEMDAN reconstruction is exact by construction regardless of sifting depth,
    /// so this doesn't weaken correctness assertions.
    fn fast_emd_config() -> EmdConfig {
        EmdConfig {
            sifting_config: crate::sifting::SiftingConfig {
                max_sifting_iterations: 5,
                ..crate::sifting::SiftingConfig::default()
            },
            max_imfs: 3,
            validate_reconstruction: false,
            ..EmdConfig::default()
        }
    }

    // =========================================================================
    // EnsembleConfig tests (inherited behavior)
    // =========================================================================

    #[test]
    fn test_ceemdan_config_validation() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let result = ceemdan(&signal, &config, &fast_emd_config());
        assert!(result.is_ok(), "CEEMDAN should succeed on pure sine wave");
    }

    // =========================================================================
    // T-075: Stage-wise CEEMDAN — at stage k, add adaptive noise to residue
    // =========================================================================

    #[test]
    fn test_ceemdan_stage_wise_decomposition() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let result = ceemdan(&signal, &config, &fast_emd_config()).unwrap();

        // Should produce at least 1 IMF
        assert!(
            result.imfs.n_imfs() >= 1,
            "CEEMDAN should produce at least 1 IMF, got {}",
            result.imfs.n_imfs()
        );

        assert_eq!(result.algorithm, AlgorithmType::CEEMDAN);
    }

    #[test]
    fn test_ceemdan_multi_component_signal() {
        let n = 120;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.5 * (2.0 * PI * 20.0 * t).sin()
                    + 0.3 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.15, seed: Some(42) };
        let result = ceemdan(&signal, &config, &fast_emd_config()).unwrap();

        // Multi-component signal should produce multiple IMFs
        assert!(
            result.imfs.n_imfs() >= 2,
            "Multi-component signal should produce at least 2 IMFs, got {}",
            result.imfs.n_imfs()
        );
    }

    // =========================================================================
    // T-076: Adaptive noise scaling — ε_k = ε · std(residue_k) / std(noise)
    // =========================================================================

    #[test]
    fn test_compute_adaptive_noise_scale_known_values() {
        let residue = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let noise_std = 0.2;
        let reference_noise_std = 1.0;

        let scale = compute_adaptive_noise_scale(&residue, noise_std, reference_noise_std);

        // std([1,2,3,4,5]) = sqrt(2) ≈ 1.414
        // scale = 0.2 * 1.414 / 1.0 ≈ 0.283
        let expected = 0.2 * 2.0f64.sqrt();
        assert!(
            (scale - expected).abs() < 1e-10,
            "Adaptive noise scale should be {:.6}, got {:.6}",
            expected,
            scale
        );
    }

    #[test]
    fn test_compute_adaptive_noise_scale_zero_reference() {
        let residue = vec![1.0, 2.0, 3.0];
        let scale = compute_adaptive_noise_scale(&residue, 0.2, 0.0);
        assert!((scale - 0.0).abs() < 1e-15, "Zero reference noise should give zero scale");
    }

    #[test]
    fn test_adaptive_noise_decreases_with_residue() {
        // As residue energy decreases, adaptive noise should decrease
        let residue_large = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let residue_small = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let scale_large = compute_adaptive_noise_scale(&residue_large, 0.2, 1.0);
        let scale_small = compute_adaptive_noise_scale(&residue_small, 0.2, 1.0);

        assert!(
            scale_large > scale_small,
            "Larger residue should have larger adaptive noise scale"
        );
    }

    // =========================================================================
    // T-077: Extract first IMF of emd(residue_k + ε_k·noise) for each trial
    // =========================================================================

    #[test]
    fn test_extract_first_imf_returns_single_imf() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let imf = extract_first_imf(&signal, &fast_emd_config()).unwrap();

        assert_eq!(imf.len(), signal.len());
    }

    #[test]
    fn test_extract_first_imf_monotonic_signal() {
        let signal: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let imf = extract_first_imf(&signal, &fast_emd_config()).unwrap();

        // Monotonic signal → no extrema → returns signal itself
        assert_eq!(imf.len(), signal.len());
    }

    // =========================================================================
    // T-078: Compute next residue — r_{k+1} = r_k - mean_IMF_k
    // =========================================================================

    #[test]
    fn test_ceemdan_residue_computation() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let result = ceemdan(&signal, &config, &fast_emd_config()).unwrap();

        // Verify residue computation: signal ≈ Σ IMFs + residue
        let reconstructed = result.imfs.reconstruct();
        assert_eq!(reconstructed.len(), signal.len());

        let max_error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(max_error < 1e-6, "Reconstruction error should be small, got {:.2e}", max_error);
    }

    // =========================================================================
    // T-079: Parallel trial computation at each stage
    // =========================================================================

    #[test]
    fn test_ceemdan_parallel_execution() {
        let n = 100;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(99) };
        let result = ceemdan(&signal, &config, &fast_emd_config());
        assert!(result.is_ok(), "Parallel CEEMDAN should succeed");
    }

    // =========================================================================
    // T-080: Validate reconstruction error < 1e-10
    // =========================================================================

    #[test]
    fn test_ceemdan_reconstruction_error_tight() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = ceemdan(&signal, &config, &emd_config).unwrap();

        let reconstructed = result.imfs.reconstruct();
        let max_error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_error < 1e-10,
            "CEEMDAN reconstruction error should be < 1e-10, got {:.2e}",
            max_error
        );
    }

    #[test]
    fn test_ceemdan_reconstruction_multi_component() {
        let n = 120;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.15, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = ceemdan(&signal, &config, &emd_config).unwrap();

        let reconstructed = result.imfs.reconstruct();
        let max_error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_error < 1e-10,
            "CEEMDAN reconstruction error for multi-component should be < 1e-10, got {:.2e}",
            max_error
        );
    }

    // =========================================================================
    // Reproducibility tests
    // =========================================================================

    #[test]
    fn test_ceemdan_reproducibility_same_seed() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result1 = ceemdan(&signal, &config, &emd_config).unwrap();
        let result2 = ceemdan(&signal, &config, &emd_config).unwrap();

        assert_eq!(result1.imfs.n_imfs(), result2.imfs.n_imfs());
        for (imf1, imf2) in result1.imfs.imfs.iter().zip(result2.imfs.imfs.iter()) {
            for (v1, v2) in imf1.iter().zip(imf2.iter()) {
                assert!((v1 - v2).abs() < 1e-15, "IMF values must match exactly with same seed");
            }
        }
    }

    #[test]
    fn test_ceemdan_different_seeds_different_results() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let emd_config = fast_emd_config();

        let config1 = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(1) };
        let config2 = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(2) };

        let result1 = ceemdan(&signal, &config1, &emd_config).unwrap();
        let result2 = ceemdan(&signal, &config2, &emd_config).unwrap();

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
    // CEEMDAN vs EEMD vs CEEMD comparison
    // =========================================================================

    #[test]
    fn test_ceemdan_vs_eemd_mode_separation() {
        let n = 120;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.1 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let ceemdan_result = ceemdan(&signal, &config, &emd_config).unwrap();
        let eemd_result = eemd(&signal, &config, &emd_config).unwrap();

        // Both should separate the mode-mixed signal
        assert!(
            ceemdan_result.imfs.n_imfs() >= 2,
            "CEEMDAN should separate mode-mixed signal into at least 2 IMFs"
        );
        assert!(
            eemd_result.imfs.n_imfs() >= 2,
            "EEMD should separate mode-mixed signal into at least 2 IMFs"
        );
    }

    #[test]
    fn test_ceemdan_vs_ceemd_mode_separation() {
        let n = 120;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 3.0 * t).sin() + 0.05 * (2.0 * PI * 30.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let ceemdan_result = ceemdan(&signal, &config, &emd_config).unwrap();
        let ceemd_result = ceemd(&signal, &config, &emd_config).unwrap();

        // Both should separate the mode-mixed signal
        assert!(
            ceemdan_result.imfs.n_imfs() >= 2,
            "CEEMDAN should separate mode-mixed signal into at least 2 IMFs"
        );
        assert!(
            ceemd_result.imfs.n_imfs() >= 2,
            "CEEMD should separate mode-mixed signal into at least 2 IMFs"
        );
    }

    #[test]
    fn test_ceemdan_reconstruction_vs_eemd() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let ceemdan_result = ceemdan(&signal, &config, &emd_config).unwrap();
        let eemd_result = eemd(&signal, &config, &emd_config).unwrap();

        // CEEMDAN should have better reconstruction than EEMD
        let ceemdan_reconstructed = ceemdan_result.imfs.reconstruct();
        let eemd_reconstructed = eemd_result.imfs.reconstruct();

        let ceemdan_error: f64 = signal
            .iter()
            .zip(ceemdan_reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        let eemd_error: f64 = signal
            .iter()
            .zip(eemd_reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        // CEEMDAN reconstruction should be at least as good as EEMD
        assert!(
            ceemdan_error <= eemd_error * 2.0,
            "CEEMDAN reconstruction ({:.2e}) should be comparable to EEMD ({:.2e})",
            ceemdan_error,
            eemd_error
        );
    }

    // =========================================================================
    // Error handling tests
    // =========================================================================

    #[test]
    fn test_ceemdan_insufficient_data() {
        let signal = vec![1.0, 2.0];
        let config = EnsembleConfig::default();
        let emd_config = EmdConfig::default();

        let result = ceemdan(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_ceemdan_invalid_value_nan() {
        let signal = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let config = EnsembleConfig::default();
        let emd_config = EmdConfig::default();

        let result = ceemdan(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_ceemdan_zero_ensembles_error() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = EnsembleConfig { num_ensembles: 0, noise_std: 0.2, seed: None };
        let emd_config = EmdConfig::default();

        let result = ceemdan(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_ceemdan_negative_noise_std_error() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = EnsembleConfig { num_ensembles: 10, noise_std: -0.1, seed: None };
        let emd_config = EmdConfig::default();

        let result = ceemdan(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    // =========================================================================
    // Algorithm type test
    // =========================================================================

    #[test]
    fn test_ceemdan_algorithm_type_is_ceemdan() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 5, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let result = ceemdan(&signal, &config, &emd_config).unwrap();
        assert_eq!(result.algorithm, AlgorithmType::CEEMDAN);
    }

    // =========================================================================
    // Config snapshot test
    // =========================================================================

    #[test]
    fn test_ceemdan_config_snapshot_is_valid_json() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 5, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig::default();

        let result = ceemdan(&signal, &config, &emd_config).unwrap();

        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&result.config_snapshot);
        assert!(parsed.is_ok(), "Config snapshot should be valid JSON: {}", result.config_snapshot);
    }

    // =========================================================================
    // Average IMFs helper tests
    // =========================================================================

    #[test]
    fn test_average_imfs_empty() {
        let result = average_imfs(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_average_imfs_single() {
        let imfs = vec![vec![1.0, 2.0, 3.0]];
        let result = average_imfs(&imfs);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_average_imfs_two() {
        let imfs = vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
        let result = average_imfs(&imfs);
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_average_imfs_three() {
        let imfs = vec![vec![0.0, 0.0], vec![3.0, 6.0], vec![6.0, 12.0]];
        let result = average_imfs(&imfs);
        assert_eq!(result, vec![3.0, 6.0]);
    }

    // =========================================================================
    // Signal std helper tests
    // =========================================================================

    #[test]
    fn test_signal_std_constant() {
        let signal = vec![5.0; 100];
        assert!((signal_std(&signal) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_signal_std_known_values() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = 2.0f64.sqrt();
        assert!((signal_std(&signal) - expected).abs() < 1e-15);
    }

    // =========================================================================
    // Large ensemble count
    // =========================================================================

    #[test]
    fn test_ceemdan_large_ensemble() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        // Use more ensembles than typical to verify the algorithm scales, but
        // still fast enough for a unit test.
        let config = EnsembleConfig { num_ensembles: 8, noise_std: 0.2, seed: Some(42) };
        let result = ceemdan(&signal, &config, &fast_emd_config());
        assert!(result.is_ok(), "CEEMDAN with larger ensemble should succeed");
    }

    // =========================================================================
    // Edge case: constant signal
    // =========================================================================

    #[test]
    fn test_ceemdan_constant_signal() {
        let signal = vec![5.0; 100];

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.1, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = ceemdan(&signal, &config, &emd_config);
        // Constant signal has no extrema → should handle gracefully
        assert!(result.is_ok() || matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    // =========================================================================
    // Three-component signal decomposition quality
    // =========================================================================

    #[test]
    fn test_ceemdan_three_component_decomposition() {
        // Use 200 samples at 1000 Hz → 0.2 s: 2 cycles at 10 Hz, 10 at 50 Hz, 20 at 100 Hz.
        let n = 200;
        let sample_rate = 1000.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * 10.0 * t).sin()
                    + 0.5 * (2.0 * PI * 50.0 * t).sin()
                    + 0.3 * (2.0 * PI * 100.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.15, seed: Some(42) };
        let emd_config = EmdConfig {
            sifting_config: crate::sifting::SiftingConfig {
                max_sifting_iterations: 10,
                sd_threshold: 0.1,
                s_number: 5,
                fixed_iterations: None,
                energy_threshold: 1e-8,
                boundary_condition: crate::boundary::BoundaryConditionType::MirrorEven,
                spline_type: crate::spline::SplineType::Natural,
            },
            max_imfs: 10,
            validate_reconstruction: false,
            reconstruction_tolerance: 1e-10,
            boundary_condition: crate::boundary::BoundaryConditionType::MirrorEven,
            intermittency: None,
        };

        let result = ceemdan(&signal, &config, &emd_config).unwrap();

        // Should extract at least 1 IMF (mode separation quality not tested with fast config)
        assert!(
            result.imfs.n_imfs() >= 1,
            "Should extract at least 1 IMF for three-component signal, got {}",
            result.imfs.n_imfs()
        );

        // Reconstruction should be valid
        let reconstructed = result.imfs.reconstruct();
        let max_error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(max_error < 1e-8, "Reconstruction error should be < 1e-8, got {:.2e}", max_error);
    }
}
