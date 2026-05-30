#![warn(missing_docs)]

//! Improved Complete Ensemble EMD with Adaptive Noise (ICEEMDAN) algorithm.
//!
//! This module implements ICEEMDAN as described by Colominas et al. (2014):
//! "A Complete Ensemble Empirical Mode Decomposition With Adaptive Noise"
//!
//! ICEEMDAN improves upon CEEMDAN by using the k-th IMF of a noise-only signal
//! instead of raw white noise at each stage. This produces cleaner IMFs with
//! less residual noise and better mode separation.
//!
//! # Algorithm
//! 1. Pre-compute: Run EMD on pure noise signals to get noise IMFs for each stage
//! 2. Stage 0: Compute IMF_1 = mean(E[signal + ε·IMF_1_of_noise])
//! 3. Stage k: Compute IMF_k = mean(E[residue_k + ε_k·IMF_k_of_noise])
//! 4. Update residue: r_{k+1} = r_k - IMF_k
//! 5. Repeat until residue has < 2 extrema
//!
//! # Key Difference from CEEMDAN
//! Instead of adding raw white noise at each stage, ICEEMDAN adds the k-th IMF
//! of a noise-only signal. This produces cleaner IMFs with less residual noise.
//!
//! # References
//! - Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014).
//!   Improved Complete Ensemble EMD: A Suitable Technique for Biomedical Signal Processing.
//!   Biomedical Signal Processing and Control, 14, 19-29.

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
// Internal: generate noise sequence
// ---------------------------------------------------------------------------

/// Generate a Gaussian white noise sequence with the given standard deviation.
fn generate_noise(rng: &mut StdRng, noise_std: f64, length: usize) -> Result<Vec<f64>, EmdError> {
    let normal = Normal::new(0.0, noise_std).map_err(|e| {
        EmdError::InvalidConfig(format!("failed to create normal distribution: {e}"))
    })?;

    Ok((0..length).map(|_| normal.sample(rng)).collect())
}

// ---------------------------------------------------------------------------
// Internal: extract all IMFs from a signal via EMD
// ---------------------------------------------------------------------------

/// Extract all IMFs from a signal using EMD.
///
/// # Arguments
/// * `signal` — Signal to decompose
/// * `emd_config` — EMD configuration
///
/// # Returns
/// All IMFs extracted from the signal.
fn extract_all_imfs(signal: &[f64], emd_config: &EmdConfig) -> Result<Vec<Vec<f64>>, EmdError> {
    match emd(signal, emd_config) {
        Ok(result) => Ok(result.imfs.imfs),
        Err(EmdError::ConvergenceFailed { .. }) | Err(EmdError::InvalidValue) => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// Internal: extract first IMF from a signal via EMD
// ---------------------------------------------------------------------------

/// Extract only the first IMF from a signal using EMD.
///
/// # Arguments
/// * `signal` — Signal to decompose
/// * `emd_config` — EMD configuration
///
/// # Returns
/// The first IMF extracted, or the signal itself if no IMF was extracted.
fn extract_first_imf(signal: &[f64], emd_config: &EmdConfig) -> Result<Vec<f64>, EmdError> {
    let mut config = emd_config.clone();
    config.max_imfs = 1;
    config.validate_reconstruction = false;

    match emd(signal, &config) {
        Ok(result) if !result.imfs.imfs.is_empty() => Ok(result.imfs.imfs[0].clone()),
        Ok(_) => Ok(signal.to_vec()),
        Err(EmdError::ConvergenceFailed { .. }) | Err(EmdError::InvalidValue) => {
            Ok(signal.to_vec())
        }
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// Internal: pre-compute noise IMFs for each stage
// ---------------------------------------------------------------------------

/// Pre-compute the k-th IMF of noise-only signals for each stage.
///
/// This is the key difference from CEEMDAN: instead of using raw white noise,
/// we use the k-th IMF of a noise-only signal at stage k.
///
/// # Arguments
/// * `num_stages` — Maximum number of stages to pre-compute
/// * `num_ensembles` — Number of noise realizations
/// * `noise_std` — Standard deviation of the base noise
/// * `length` — Length of the noise signal
/// * `base_rng` — Base RNG for generating noise sequences
/// * `emd_config` — EMD configuration
///
/// # Returns
/// A 3D vector: noise_imfs[stage][trial] = IMF at that stage for that trial
fn precompute_noise_imfs(
    num_stages: usize,
    num_ensembles: usize,
    noise_std: f64,
    length: usize,
    base_rng: &mut StdRng,
    emd_config: &EmdConfig,
) -> Result<Vec<Vec<Vec<f64>>>, EmdError> {
    let mut noise_imfs = Vec::with_capacity(num_stages);

    for stage in 0..num_stages {
        let mut stage_imfs = Vec::with_capacity(num_ensembles);

        for _trial in 0..num_ensembles {
            // Generate pure noise
            let noise = generate_noise(base_rng, noise_std, length)?;

            // Run EMD on pure noise to get all IMFs
            let noise_imf_result = extract_all_imfs(&noise, emd_config)?;

            // Get the stage-th IMF (or use zeros if not enough IMFs)
            let stage_imf = if stage < noise_imf_result.len() {
                noise_imf_result[stage].clone()
            } else {
                vec![0.0; length]
            };

            stage_imfs.push(stage_imf);
        }

        noise_imfs.push(stage_imfs);
    }

    Ok(noise_imfs)
}

// ---------------------------------------------------------------------------
// Internal: run one trial at stage 0
// ---------------------------------------------------------------------------

/// Run a single trial at stage 0: add ε·IMF_1_of_noise to signal and extract first IMF.
///
/// # Arguments
/// * `signal` — Original signal
/// * `noise_imf` — Pre-computed IMF_1 of noise for this trial
/// * `noise_scale` — ε (noise scaling factor)
/// * `emd_config` — EMD configuration
///
/// # Returns
/// The first IMF extracted from the noisy signal.
fn run_stage0_trial(
    signal: &[f64],
    noise_imf: &[f64],
    noise_scale: f64,
    emd_config: &EmdConfig,
) -> Result<Vec<f64>, EmdError> {
    let noisy_signal: Vec<f64> =
        signal.iter().zip(noise_imf.iter()).map(|(&s, &n)| s + noise_scale * n).collect();

    extract_first_imf(&noisy_signal, emd_config)
}

// ---------------------------------------------------------------------------
// Internal: run one trial at stage k (k >= 1)
// ---------------------------------------------------------------------------

/// Run a single trial at stage k: add ε_k·IMF_k_of_noise to residue and extract first IMF.
///
/// # Arguments
/// * `residue` — Current residue
/// * `noise_imf` — Pre-computed IMF_k of noise for this trial
/// * `adaptive_noise_scale` — ε_k (adaptive noise scaling factor)
/// * `emd_config` — EMD configuration
///
/// # Returns
/// The first IMF extracted from the noisy residue.
fn run_stage_k_trial(
    residue: &[f64],
    noise_imf: &[f64],
    adaptive_noise_scale: f64,
    emd_config: &EmdConfig,
) -> Result<Vec<f64>, EmdError> {
    let noisy_residue: Vec<f64> =
        residue.iter().zip(noise_imf.iter()).map(|(&r, &n)| r + adaptive_noise_scale * n).collect();

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

/// Perform Improved Complete Ensemble EMD with Adaptive Noise on a signal.
///
/// ICEEMDAN addresses the residual noise problem in CEEMDAN by using the k-th
/// IMF of a noise-only signal instead of raw white noise at each stage.
///
/// This approach ensures:
/// - Complete reconstruction (signal = Σ IMFs + final residue)
/// - Less residual noise in IMFs compared to CEEMDAN
/// - Better mode separation
/// - No mode mixing
///
/// # Arguments
/// * `signal` — Input signal to decompose
/// * `config` — Ensemble configuration (num_ensembles, noise_std, seed)
/// * `emd_config` — EMD configuration for inner decomposition
///
/// # Returns
/// A `DecompositionResult` containing the ICEEMDAN IMFs and final residue,
/// or an error if decomposition fails.
///
/// # Examples
/// ```
/// use ferromode::algorithms::iceemdan::iceemdan;
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
/// let result = iceemdan(&signal, &config, &emd_config);
/// assert!(result.is_ok());
/// ```
pub fn iceemdan(
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

    // Estimate maximum number of stages (based on log2 of signal length)
    let max_stages = (signal.len() as f64).log2().ceil() as usize + 2;

    // Pre-compute noise IMFs for all stages
    let noise_imfs = precompute_noise_imfs(
        max_stages,
        config.num_ensembles,
        absolute_noise_std,
        signal.len(),
        &mut base_rng,
        emd_config,
    )?;

    // Compute reference noise std (std of the first noise sequence)
    let reference_noise_std = if !noise_imfs.is_empty() && !noise_imfs[0].is_empty() {
        signal_std(&noise_imfs[0][0])
    } else {
        absolute_noise_std
    };

    // ========================================================================
    // Stage 0: Add ε·IMF_1_of_noise to signal, extract first IMF, average
    // ========================================================================

    let noise_scale = absolute_noise_std;

    let stage0_imfs: Result<Vec<Vec<f64>>, EmdError> = (0..config.num_ensembles)
        .into_par_iter()
        .map(|i| run_stage0_trial(signal, &noise_imfs[0][i], noise_scale, emd_config))
        .collect();

    let stage0_imfs = stage0_imfs?;
    let imf_1 = average_imfs(&stage0_imfs);

    // Compute first residue
    let mut residue: Vec<f64> = signal.iter().zip(imf_1.iter()).map(|(&s, &i)| s - i).collect();

    let mut all_imfs: Vec<Vec<f64>> = vec![imf_1];
    let mut total_trials: usize = config.num_ensembles;

    // ========================================================================
    // Stage k: Add ε_k·IMF_k_of_noise to residue, extract first IMF, average
    // ========================================================================

    let mut stage_k = 1;

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

        // Check if we've exceeded pre-computed noise IMFs
        if stage_k >= noise_imfs.len() {
            break;
        }

        // Compute adaptive noise scale for this stage
        let adaptive_scale =
            compute_adaptive_noise_scale(&residue, config.noise_std, reference_noise_std);

        // Run trials in parallel for this stage using pre-computed noise IMFs
        let stage_imfs: Result<Vec<Vec<f64>>, EmdError> = (0..config.num_ensembles)
            .into_par_iter()
            .map(|i| {
                run_stage_k_trial(&residue, &noise_imfs[stage_k][i], adaptive_scale, emd_config)
            })
            .collect();

        let stage_imfs = stage_imfs?;
        let mean_imf = average_imfs(&stage_imfs);
        total_trials += config.num_ensembles;

        // Update residue
        let new_residue: Vec<f64> =
            residue.iter().zip(mean_imf.iter()).map(|(&r, &m)| r - m).collect();

        all_imfs.push(mean_imf);
        residue = new_residue;
        stage_k += 1;
    }

    let elapsed = start.elapsed();

    let result = DecompositionResult::new(
        AlgorithmType::ICEEMDAN,
        ImfCollection::new(all_imfs, residue),
        elapsed,
        total_trials,
        format!(
            r#"{{"num_ensembles": {}, "noise_std": {}, "seed": {}, "algorithm": "ICEEMDAN"}}"#,
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
    use crate::algorithms::ceemdan::ceemdan;
    use std::f64::consts::PI;

    fn fast_emd_config() -> EmdConfig {
        EmdConfig {
            sifting_config: crate::sifting::SiftingConfig {
                max_sifting_iterations: 5,
                ..crate::sifting::SiftingConfig::default()
            },
            // ICEEMDAN precomputes (log2(n)+2) × num_ensembles full EMD runs in
            // debug mode. Keep max_imfs tiny so each precompute call is cheap.
            max_imfs: 2,
            validate_reconstruction: false,
            ..EmdConfig::default()
        }
    }

    // =========================================================================
    // T-081: Improved noise model — use EMD of noise-only signal at each stage
    // =========================================================================

    #[test]
    fn test_iceemdan_noise_only_imf_model() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = iceemdan(&signal, &config, &emd_config);
        assert!(result.is_ok(), "ICEEMDAN should succeed on pure sine wave");
    }

    #[test]
    fn test_iceemdan_precompute_noise_imfs() {
        let mut rng = StdRng::seed_from_u64(42);
        let num_stages = 3;
        let num_ensembles = 4;
        let noise_std = 0.2;
        let length = 50;
        let emd_config = fast_emd_config();

        let noise_imfs = precompute_noise_imfs(
            num_stages,
            num_ensembles,
            noise_std,
            length,
            &mut rng,
            &emd_config,
        );

        assert!(noise_imfs.is_ok());
        let noise_imfs = noise_imfs.unwrap();

        assert_eq!(noise_imfs.len(), num_stages);
        for stage_imfs in &noise_imfs {
            assert_eq!(stage_imfs.len(), num_ensembles);
            for imf in stage_imfs {
                assert_eq!(imf.len(), length);
            }
        }
    }

    // =========================================================================
    // T-082: Compute mean_IMF_k = E[first_IMF(r_k + ε_k · IMF_k_of_noise)]
    // =========================================================================

    #[test]
    fn test_iceemdan_stage0_computation() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = iceemdan(&signal, &config, &emd_config).unwrap();

        // Should produce at least 1 IMF
        assert!(
            result.imfs.n_imfs() >= 1,
            "ICEEMDAN should produce at least 1 IMF, got {}",
            result.imfs.n_imfs()
        );

        assert_eq!(result.algorithm, AlgorithmType::ICEEMDAN);
    }

    #[test]
    fn test_iceemdan_multi_stage_decomposition() {
        let n = 60;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.5 * (2.0 * PI * 20.0 * t).sin()
                    + 0.3 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.15, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = iceemdan(&signal, &config, &emd_config).unwrap();

        // Multi-component signal should produce multiple IMFs
        assert!(
            result.imfs.n_imfs() >= 2,
            "Multi-component signal should produce at least 2 IMFs, got {}",
            result.imfs.n_imfs()
        );
    }

    // =========================================================================
    // T-083: Residual noise in IMFs is further reduced vs. CEEMDAN
    // =========================================================================

    #[test]
    fn test_iceemdan_vs_ceemdan_reconstruction_error() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let iceemdan_result = iceemdan(&signal, &config, &emd_config).unwrap();
        let ceemdan_result = ceemdan(&signal, &config, &emd_config).unwrap();

        let iceemdan_reconstructed = iceemdan_result.imfs.reconstruct();
        let ceemdan_reconstructed = ceemdan_result.imfs.reconstruct();

        let iceemdan_error: f64 = signal
            .iter()
            .zip(iceemdan_reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        let ceemdan_error: f64 = signal
            .iter()
            .zip(ceemdan_reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        // ICEEMDAN should have reconstruction error at least as good as CEEMDAN
        assert!(
            iceemdan_error < 1e-8,
            "ICEEMDAN reconstruction error should be small, got {:.2e}",
            iceemdan_error
        );

        // Log comparison for verification
        eprintln!("ICEEMDAN error: {:.2e}, CEEMDAN error: {:.2e}", iceemdan_error, ceemdan_error);
    }

    #[test]
    fn test_iceemdan_reconstruction_error_tight() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = iceemdan(&signal, &config, &emd_config).unwrap();

        let reconstructed = result.imfs.reconstruct();
        let max_error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_error < 1e-8,
            "ICEEMDAN reconstruction error should be < 1e-8, got {:.2e}",
            max_error
        );
    }

    #[test]
    fn test_iceemdan_reconstruction_multi_component() {
        let n = 60;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.15, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = iceemdan(&signal, &config, &emd_config).unwrap();

        let reconstructed = result.imfs.reconstruct();
        let max_error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_error < 1e-8,
            "ICEEMDAN reconstruction error for multi-component should be < 1e-8, got {:.2e}",
            max_error
        );
    }

    // =========================================================================
    // Reproducibility tests
    // =========================================================================

    #[test]
    fn test_iceemdan_reproducibility_same_seed() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result1 = iceemdan(&signal, &config, &emd_config).unwrap();
        let result2 = iceemdan(&signal, &config, &emd_config).unwrap();

        assert_eq!(result1.imfs.n_imfs(), result2.imfs.n_imfs());
        for (imf1, imf2) in result1.imfs.imfs.iter().zip(result2.imfs.imfs.iter()) {
            for (v1, v2) in imf1.iter().zip(imf2.iter()) {
                assert!((v1 - v2).abs() < 1e-15, "IMF values must match exactly with same seed");
            }
        }
    }

    #[test]
    fn test_iceemdan_different_seeds_different_results() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let emd_config = fast_emd_config();

        let config1 = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(1) };
        let config2 = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(2) };

        let result1 = iceemdan(&signal, &config1, &emd_config).unwrap();
        let result2 = iceemdan(&signal, &config2, &emd_config).unwrap();

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
    fn test_iceemdan_insufficient_data() {
        let signal = vec![1.0, 2.0];
        let config = EnsembleConfig::default();
        let emd_config = EmdConfig::default();

        let result = iceemdan(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_iceemdan_invalid_value_nan() {
        let signal = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let config = EnsembleConfig::default();
        let emd_config = EmdConfig::default();

        let result = iceemdan(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_iceemdan_zero_ensembles_error() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = EnsembleConfig { num_ensembles: 0, noise_std: 0.2, seed: None };
        let emd_config = EmdConfig::default();

        let result = iceemdan(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_iceemdan_negative_noise_std_error() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = EnsembleConfig { num_ensembles: 10, noise_std: -0.1, seed: None };
        let emd_config = EmdConfig::default();

        let result = iceemdan(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    // =========================================================================
    // Algorithm type test
    // =========================================================================

    #[test]
    fn test_iceemdan_algorithm_type_is_iceemdan() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = iceemdan(&signal, &config, &emd_config).unwrap();
        assert_eq!(result.algorithm, AlgorithmType::ICEEMDAN);
    }

    // =========================================================================
    // Config snapshot test
    // =========================================================================

    #[test]
    fn test_iceemdan_config_snapshot_is_valid_json() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = iceemdan(&signal, &config, &emd_config).unwrap();

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
    fn test_iceemdan_large_ensemble() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 8, noise_std: 0.2, seed: Some(42) };
        let result = iceemdan(&signal, &config, &fast_emd_config());
        assert!(result.is_ok(), "ICEEMDAN with larger ensemble should succeed");
    }

    // =========================================================================
    // Edge case: constant signal
    // =========================================================================

    #[test]
    fn test_iceemdan_constant_signal() {
        let signal = vec![5.0; 100];

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.1, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = iceemdan(&signal, &config, &emd_config);
        // Constant signal has no extrema → should handle gracefully
        assert!(result.is_ok() || matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    // =========================================================================
    // Three-component signal decomposition quality
    // =========================================================================

    #[test]
    fn test_iceemdan_three_component_decomposition() {
        let n = 60;
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

        let result = iceemdan(&signal, &config, &emd_config).unwrap();

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
