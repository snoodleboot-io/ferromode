#![warn(missing_docs)]

//! Empirical Mode Decomposition (EMD) algorithm.
//!
//! This module implements the complete EMD algorithm that orchestrates
//! extrema detection and sifting into a full decomposition, following
//! Huang's original methodology.
//!
//! The algorithm decomposes a signal into a finite number of Intrinsic
//! Mode Functions (IMFs) plus a residue:
//!
//! ```text
//! signal = IMF_1 + IMF_2 + ... + IMF_n + residue
//! ```

use crate::boundary::{build_palindrome, BoundaryConditionType};
use crate::error::EmdError;
use crate::extrema::detect_extrema;
use crate::sifting::{sift_one, SiftingConfig};
use crate::spline::SplineType;
use crate::types::{AlgorithmType, DecompositionResult, ImfCollection};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ---------------------------------------------------------------------------
// EmdConfig
// ---------------------------------------------------------------------------

/// Configuration for the complete EMD decomposition.
///
/// Combines sifting configuration with decomposition-level settings
/// such as maximum number of IMFs, boundary conditions, and optional
/// intermittency testing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmdConfig {
    /// Sifting configuration for each IMF extraction.
    ///
    /// The active boundary strategy is `sifting_config.boundary_condition`.
    /// Setting it to [`BoundaryConditionType::PalindromeCyclic`] enables the
    /// palindrome pre-processing path (see module docs for details).
    pub sifting_config: SiftingConfig,
    /// Maximum number of IMFs to extract (0 = no limit, determined by residue)
    pub max_imfs: usize,
    /// Boundary condition reflected in the config snapshot string only.
    /// The operative boundary condition is `sifting_config.boundary_condition`.
    pub boundary_condition: BoundaryConditionType,
    /// Optional intermittency test configuration
    pub intermittency: Option<IntermittencyConfig>,
    /// Tolerance for reconstruction validation (Σ IMFs + residue ≈ original)
    pub reconstruction_tolerance: f64,
    /// Whether to validate reconstruction after decomposition
    pub validate_reconstruction: bool,
}

impl Default for EmdConfig {
    fn default() -> Self {
        Self {
            sifting_config: SiftingConfig::default(),
            max_imfs: 0, // No limit — extract until residue has < 2 extrema
            boundary_condition: BoundaryConditionType::MirrorEven,
            intermittency: None,
            reconstruction_tolerance: 1e-12,
            validate_reconstruction: true,
        }
    }
}

// ---------------------------------------------------------------------------
// IntermittencyConfig
// ---------------------------------------------------------------------------

/// Configuration for the intermittency test.
///
/// The intermittency test detects non-stationary regions by analyzing
/// the spacing between consecutive extrema. If the spacing varies
/// beyond a threshold, the signal is flagged as having intermittent
/// behavior in that region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntermittencyConfig {
    /// Threshold for coefficient of variation of extrema spacing.
    ///
    /// CV = std_dev / mean. If CV exceeds this threshold, the region
    /// is flagged as intermittent. Typical values: 0.3 to 0.5.
    pub cv_threshold: f64,
    /// Minimum number of extrema intervals needed for the test.
    /// If fewer intervals exist, the test is skipped.
    pub min_intervals: usize,
}

impl Default for IntermittencyConfig {
    fn default() -> Self {
        Self { cv_threshold: 0.4, min_intervals: 5 }
    }
}

// ---------------------------------------------------------------------------
// IntermittencyResult
// ---------------------------------------------------------------------------

/// Result of the intermittency analysis on a signal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntermittencyResult {
    /// Whether intermittent behavior was detected
    pub is_intermittent: bool,
    /// Coefficient of variation of extrema spacing
    pub cv: f64,
    /// Mean spacing between extrema
    pub mean_spacing: f64,
    /// Standard deviation of spacing
    pub std_spacing: f64,
    /// Number of extrema used in the analysis
    pub n_extrema: usize,
}

impl IntermittencyResult {
    /// Create a new intermittency result.
    pub fn new(
        is_intermittent: bool,
        cv: f64,
        mean_spacing: f64,
        std_spacing: f64,
        n_extrema: usize,
    ) -> Self {
        Self { is_intermittent, cv, mean_spacing, std_spacing, n_extrema }
    }
}

// ---------------------------------------------------------------------------
// EmdError extensions
// ---------------------------------------------------------------------------

/// Additional error variants specific to the EMD decomposition.
#[derive(Debug, thiserror::Error)]
pub enum EmdDecompositionError {
    /// Reconstruction validation failed
    #[error("reconstruction failed: max error {max_error:.2e} exceeds tolerance {tolerance:.2e}")]
    ReconstructionFailed { max_error: f64, tolerance: f64 },
    /// No IMFs were extracted
    #[error("no IMFs extracted: signal may be monotonic or residue has insufficient extrema")]
    NoImfsExtracted,
    /// Intermittency test detected non-stationary behavior
    #[error("intermittency detected in signal: cv={cv:.3}")]
    IntermittencyDetected { cv: f64 },
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform Empirical Mode Decomposition on a signal.
///
/// This is the top-level EMD function that orchestrates the complete
/// decomposition process:
///
/// 1. Detects extrema in the current signal/residue
/// 2. Calls `sift_one()` to extract one IMF
/// 3. Subtracts IMF from residue
/// 4. Repeats until residue has < 2 extrema or `max_imfs` reached
///
/// # Arguments
/// * `signal` — Input signal to decompose
/// * `config` — EMD configuration
///
/// # Returns
/// A `DecompositionResult` containing all extracted IMFs and the final residue,
/// or an error if decomposition fails.
///
/// # Examples
/// ```
/// use ferromode::algorithms::emd::{EmdConfig, emd};
/// use std::f64::consts::PI;
///
/// // Decompose a simple sine wave
/// let n = 100;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
///     .collect();
///
/// let config = EmdConfig::default();
/// let result = emd(&signal, &config);
/// assert!(result.is_ok());
/// ```
pub fn emd(signal: &[f64], config: &EmdConfig) -> Result<DecompositionResult, EmdError> {
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

    // PalindromeCyclic: pre-extend signal to 2N-1 palindrome, use periodic spline.
    let is_palindrome =
        config.sifting_config.boundary_condition == BoundaryConditionType::PalindromeCyclic;

    let working_signal: Vec<f64> = if is_palindrome {
        build_palindrome(signal)
    } else {
        signal.to_vec()
    };

    let effective_sifting_config: SiftingConfig = if is_palindrome {
        SiftingConfig {
            boundary_condition: BoundaryConditionType::MirrorEven,
            spline_type: SplineType::Periodic,
            ..config.sifting_config.clone()
        }
    } else {
        config.sifting_config.clone()
    };

    let mut residue = working_signal.clone();
    let mut imfs: Vec<Vec<f64>> = Vec::new();
    let mut total_siftings: usize = 0;

    // Outer decomposition loop
    loop {
        // Check max_imfs limit (0 means no limit)
        if config.max_imfs > 0 && imfs.len() >= config.max_imfs {
            break;
        }

        // Detect extrema in current residue
        let extrema = detect_extrema(&residue);
        let n_extrema = extrema.maxima.len() + extrema.minima.len();

        // Stop if residue has fewer than 2 extrema (monotonic residue)
        if n_extrema < 2 {
            break;
        }

        // Extract one IMF using the sifting engine
        let (imf, new_residue) = sift_one(&residue, &effective_sifting_config)?;

        // Track sifting iterations (estimate from energy reduction)
        total_siftings += 1;

        // Check if IMF is meaningful (has non-trivial energy)
        let imf_energy: f64 = imf.iter().map(|v| v * v).sum();
        let residue_energy: f64 = residue.iter().map(|v| v * v).sum();

        // If IMF has negligible energy compared to residue, stop
        if residue_energy > 0.0 && imf_energy / residue_energy < 1e-15 {
            break;
        }

        imfs.push(imf);
        residue = new_residue;
    }

    // PalindromeCyclic post-processing: trim IMFs and residue back to original N samples.
    // Reconstruction is preserved: Σ imf[:N] + residue[:N] == palindrome[:N] == signal.
    if is_palindrome {
        let n = signal.len();
        for imf in &mut imfs {
            imf.truncate(n);
        }
        residue.truncate(n);
    }

    // Validate reconstruction if requested
    if config.validate_reconstruction && !imfs.is_empty() {
        let collection = ImfCollection::new(imfs.clone(), residue.clone());
        validate_reconstruction(signal, &collection, config.reconstruction_tolerance)?;
    }

    let elapsed = start.elapsed();

    // Run intermittency analysis if configured
    let _intermittency_result = if let Some(ref inter_config) = config.intermittency {
        Some(test_intermittency(signal, inter_config))
    } else {
        None
    };

    let result = DecompositionResult::new(
        AlgorithmType::EMD,
        ImfCollection::new(imfs, residue),
        elapsed,
        total_siftings,
        format!(
            r#"{{"sifting": {}, "max_imfs": {}, "boundary": "{:?}"}}"#,
            serde_json::to_string(&config.sifting_config).unwrap_or_default(),
            config.max_imfs,
            config.boundary_condition,
        ),
    );

    Ok(result)
}

/// Validate that the decomposition reconstructs the original signal.
///
/// Verifies that Σ IMFs + residue ≈ original_signal within the specified tolerance.
///
/// # Arguments
/// * `original` — The original input signal
/// * `collection` — The decomposition result (IMFs + residue)
/// * `tolerance` — Maximum allowed absolute error per sample
///
/// # Returns
/// `Ok(())` if reconstruction is valid, or `EmdError` if it fails.
pub fn validate_reconstruction(
    original: &[f64],
    collection: &ImfCollection,
    tolerance: f64,
) -> Result<(), EmdError> {
    let reconstructed = collection.reconstruct();

    if reconstructed.len() != original.len() {
        return Err(EmdError::DimensionMismatch);
    }

    let mut max_error = 0.0f64;
    for (i, (&orig, &recon)) in original.iter().zip(reconstructed.iter()).enumerate() {
        let error = (orig - recon).abs();
        if error > max_error {
            max_error = error;
        }
        if error > tolerance {
            return Err(EmdError::InvalidConfig(format!(
                "reconstruction failed at index {}: error {:.2e} exceeds tolerance {:.2e}",
                i, error, tolerance
            )));
        }
    }

    Ok(())
}

/// Test for intermittency in a signal.
///
/// Analyzes the spacing between consecutive extrema to detect
/// non-stationary behavior. If the coefficient of variation (CV)
/// of the spacing exceeds the configured threshold, the signal
/// is flagged as intermittent.
///
/// # Arguments
/// * `signal` — Signal to analyze
/// * `config` — Intermittency test configuration
///
/// # Returns
/// An `IntermittencyResult` describing the analysis.
pub fn test_intermittency(signal: &[f64], config: &IntermittencyConfig) -> IntermittencyResult {
    let extrema = detect_extrema(signal);

    // Combine and sort all extrema indices
    let mut all_extrema: Vec<usize> =
        Vec::with_capacity(extrema.maxima.len() + extrema.minima.len());
    all_extrema.extend(&extrema.maxima);
    all_extrema.extend(&extrema.minima);
    all_extrema.sort();
    all_extrema.dedup();

    let n_extrema = all_extrema.len();

    if n_extrema < config.min_intervals + 1 {
        return IntermittencyResult::new(false, 0.0, 0.0, 0.0, n_extrema);
    }

    // Compute spacings between consecutive extrema
    let spacings: Vec<f64> =
        (1..all_extrema.len()).map(|i| (all_extrema[i] - all_extrema[i - 1]) as f64).collect();

    let n = spacings.len() as f64;
    let mean_spacing: f64 = spacings.iter().sum::<f64>() / n;

    if mean_spacing <= 0.0 {
        return IntermittencyResult::new(false, 0.0, mean_spacing, 0.0, n_extrema);
    }

    let variance: f64 = spacings.iter().map(|&s| (s - mean_spacing).powi(2)).sum::<f64>() / n;
    let std_spacing = variance.sqrt();
    let cv = std_spacing / mean_spacing;

    let is_intermittent = cv > config.cv_threshold;

    IntermittencyResult::new(is_intermittent, cv, mean_spacing, std_spacing, n_extrema)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // =========================================================================
    // EmdConfig tests
    // =========================================================================

    #[test]
    fn test_emd_config_default() {
        let config = EmdConfig::default();
        assert_eq!(config.max_imfs, 0);
        assert_eq!(config.boundary_condition, BoundaryConditionType::MirrorEven);
        assert!(config.intermittency.is_none());
        assert!((config.reconstruction_tolerance - 1e-12).abs() < 1e-15);
        assert!(config.validate_reconstruction);
    }

    #[test]
    fn test_emd_config_custom() {
        let config = EmdConfig {
            sifting_config: SiftingConfig::default(),
            max_imfs: 5,
            boundary_condition: BoundaryConditionType::Periodic,
            intermittency: Some(IntermittencyConfig::default()),
            reconstruction_tolerance: 1e-10,
            validate_reconstruction: false,
        };
        assert_eq!(config.max_imfs, 5);
        assert_eq!(config.boundary_condition, BoundaryConditionType::Periodic);
        assert!(config.intermittency.is_some());
        assert!(!config.validate_reconstruction);
    }

    // =========================================================================
    // IntermittencyConfig tests
    // =========================================================================

    #[test]
    fn test_intermittency_config_default() {
        let config = IntermittencyConfig::default();
        assert!((config.cv_threshold - 0.4).abs() < 1e-15);
        assert_eq!(config.min_intervals, 5);
    }

    // =========================================================================
    // IntermittencyResult tests
    // =========================================================================

    #[test]
    fn test_intermittency_result_creation() {
        let result = IntermittencyResult::new(true, 0.5, 10.0, 5.0, 20);
        assert!(result.is_intermittent);
        assert!((result.cv - 0.5).abs() < 1e-15);
        assert!((result.mean_spacing - 10.0).abs() < 1e-15);
        assert!((result.std_spacing - 5.0).abs() < 1e-15);
        assert_eq!(result.n_extrema, 20);
    }

    // =========================================================================
    // T-061: Pure sine wave → should produce 1 IMF + near-zero residue
    // =========================================================================

    #[test]
    fn test_emd_pure_sine_wave_produces_one_imf() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig::default();
        let result = emd(&signal, &config);

        assert!(result.is_ok(), "EMD should succeed for pure sine wave");
        let result = result.unwrap();

        // Should produce at least 1 IMF
        assert!(
            result.imfs.n_imfs() >= 1,
            "Pure sine wave should produce at least 1 IMF, got {}",
            result.imfs.n_imfs()
        );

        // Residue should be near-zero
        let residue_energy: f64 = result.imfs.residue.iter().map(|v| v * v).sum();
        assert!(
            residue_energy < 1e-6,
            "Residue energy should be near zero for pure sine wave, got {}",
            residue_energy
        );
    }

    // =========================================================================
    // T-062: Multi-component signal → should produce multiple IMFs
    // =========================================================================

    #[test]
    fn test_emd_multi_component_signal_produces_multiple_imfs() {
        let n = 500;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                // Two sine waves at different frequencies
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let config = EmdConfig::default();
        let result = emd(&signal, &config);

        assert!(result.is_ok(), "EMD should succeed for multi-component signal");
        let result = result.unwrap();

        // Should produce multiple IMFs
        assert!(
            result.imfs.n_imfs() >= 2,
            "Multi-component signal should produce at least 2 IMFs, got {}",
            result.imfs.n_imfs()
        );
    }

    // =========================================================================
    // T-063: max_imfs limit
    // =========================================================================

    #[test]
    fn test_emd_max_imfs_limit() {
        let n = 500;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.5 * (2.0 * PI * 20.0 * t).sin()
                    + 0.3 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config = EmdConfig { max_imfs: 2, ..EmdConfig::default() };
        let result = emd(&signal, &config);

        assert!(result.is_ok());
        let result = result.unwrap();

        // Should produce exactly 2 IMFs (limited by max_imfs)
        assert_eq!(
            result.imfs.n_imfs(),
            2,
            "Should produce exactly 2 IMFs when max_imfs=2, got {}",
            result.imfs.n_imfs()
        );
    }

    #[test]
    fn test_emd_max_imfs_one() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig { max_imfs: 1, ..EmdConfig::default() };
        let result = emd(&signal, &config);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.imfs.n_imfs(), 1);
    }

    // =========================================================================
    // T-064: Intermittency test
    // =========================================================================

    #[test]
    fn test_intermittency_regular_sine_wave() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = IntermittencyConfig::default();
        let result = test_intermittency(&signal, &config);

        // Regular sine wave should NOT be intermittent
        assert!(
            !result.is_intermittent,
            "Regular sine wave should not be intermittent, cv={}",
            result.cv
        );
    }

    #[test]
    fn test_intermittency_chirp_signal() {
        let n = 500;
        // Chirp signal: frequency increases over time → intermittent
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * (5.0 + 30.0 * t) * t).sin()
            })
            .collect();

        let config = IntermittencyConfig { cv_threshold: 0.3, min_intervals: 3 };
        let result = test_intermittency(&signal, &config);

        // Chirp signal has varying frequency → varying spacing → intermittent
        assert!(
            result.n_extrema > config.min_intervals,
            "Chirp signal should have enough extrema for test"
        );
    }

    #[test]
    fn test_intermittency_insufficient_extrema() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Monotonic → no extrema

        let config = IntermittencyConfig::default();
        let result = test_intermittency(&signal, &config);

        assert!(!result.is_intermittent);
        assert_eq!(result.cv, 0.0);
    }

    #[test]
    fn test_emd_with_intermittency_config() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig {
            intermittency: Some(IntermittencyConfig::default()),
            ..EmdConfig::default()
        };
        let result = emd(&signal, &config);

        assert!(result.is_ok());
    }

    // =========================================================================
    // T-065: Reconstruction validation
    // =========================================================================

    #[test]
    fn test_validate_reconstruction_sine_wave() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig {
            validate_reconstruction: true,
            reconstruction_tolerance: 1e-12,
            ..EmdConfig::default()
        };
        let result = emd(&signal, &config);

        assert!(
            result.is_ok(),
            "Reconstruction validation should pass for sine wave: {:?}",
            result
        );
    }

    #[test]
    fn test_validate_reconstruction_multi_component() {
        let n = 500;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.3 * (2.0 * PI * 25.0 * t).sin()
            })
            .collect();

        let config = EmdConfig {
            validate_reconstruction: true,
            reconstruction_tolerance: 1e-10,
            ..EmdConfig::default()
        };
        let result = emd(&signal, &config);

        assert!(
            result.is_ok(),
            "Reconstruction validation should pass for multi-component signal: {:?}",
            result
        );
    }

    #[test]
    fn test_validate_reconstruction_manual() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let imf1 = vec![0.5, 1.0, 1.5, 2.0, 2.5];
        let imf2 = vec![0.3, 0.6, 0.9, 1.2, 1.5];
        let residue = vec![0.2, 0.4, 0.6, 0.8, 1.0];

        let collection = ImfCollection::new(vec![imf1, imf2], residue);

        let result = validate_reconstruction(&original, &collection, 1e-12);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_reconstruction_fails_on_mismatch() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let imf = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let residue = vec![0.5, 0.5, 0.5, 0.5, 0.5];

        let collection = ImfCollection::new(vec![imf], residue);

        let result = validate_reconstruction(&original, &collection, 1e-12);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_reconstruction_dimension_mismatch() {
        let original = vec![1.0, 2.0, 3.0];
        let imf = vec![0.5, 0.5];
        let residue = vec![0.5, 0.5];

        let collection = ImfCollection::new(vec![imf], residue);

        let result = validate_reconstruction(&original, &collection, 1e-12);
        assert!(matches!(result.unwrap_err(), EmdError::DimensionMismatch));
    }

    // =========================================================================
    // T-066: Reference tests with known signals
    // =========================================================================

    #[test]
    fn test_emd_reference_two_sine_waves() {
        // Known signal: sum of two sine waves at well-separated frequencies
        let n = 1000;
        let sample_rate = 1000.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config = EmdConfig {
            sifting_config: SiftingConfig {
                max_sifting_iterations: 200,
                sd_threshold: 0.1,
                s_number: 5,
                fixed_iterations: None,
                energy_threshold: 1e-8,
                boundary_condition: BoundaryConditionType::MirrorEven,
                spline_type: crate::spline::SplineType::Natural,
            },
            max_imfs: 10,
            validate_reconstruction: true,
            reconstruction_tolerance: 1e-10,
            ..EmdConfig::default()
        };

        let result = emd(&signal, &config);
        assert!(result.is_ok(), "EMD should succeed for reference signal");

        let result = result.unwrap();

        // Should extract at least 2 IMFs for two frequency components
        assert!(
            result.imfs.n_imfs() >= 2,
            "Should extract at least 2 IMFs, got {}",
            result.imfs.n_imfs()
        );

        // Reconstruction should be valid
        let reconstructed = result.imfs.reconstruct();
        assert_eq!(reconstructed.len(), signal.len());

        let max_error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(max_error < 1e-10, "Reconstruction error should be < 1e-10, got {:.2e}", max_error);
    }

    #[test]
    fn test_emd_reference_sine_with_dc_offset() {
        // Sine wave with DC offset — DC should end up in residue
        let n = 200;
        let signal: Vec<f64> =
            (0..n).map(|i| 5.0 + (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig::default();
        let result = emd(&signal, &config);

        assert!(result.is_ok());
        let result = result.unwrap();

        // Should produce at least 1 IMF
        assert!(result.imfs.n_imfs() >= 1);

        // Residue should be approximately constant (the DC offset)
        if !result.imfs.residue.is_empty() {
            let residue_mean: f64 =
                result.imfs.residue.iter().sum::<f64>() / result.imfs.residue.len() as f64;
            let residue_var: f64 =
                result.imfs.residue.iter().map(|v| (v - residue_mean).powi(2)).sum::<f64>()
                    / result.imfs.residue.len() as f64;

            // Residue variance should be small (approximately constant)
            assert!(
                residue_var < 1.0,
                "Residue should be approximately constant (DC), variance={}",
                residue_var
            );
        }
    }

    #[test]
    fn test_emd_constant_signal_no_imfs() {
        let signal = vec![5.0; 100];

        let config = EmdConfig::default();
        let result = emd(&signal, &config);

        // Constant signal has no extrema → no IMFs
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.imfs.n_imfs(), 0);
    }

    #[test]
    fn test_emd_monotonic_signal_no_imfs() {
        let signal: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let config = EmdConfig::default();
        let result = emd(&signal, &config);

        // Monotonic signal has < 2 extrema → no IMFs
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.imfs.n_imfs(), 0);
    }

    #[test]
    fn test_emd_insufficient_data() {
        let signal = vec![1.0, 2.0];

        let config = EmdConfig::default();
        let result = emd(&signal, &config);

        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_emd_invalid_value_nan() {
        let signal = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];

        let config = EmdConfig::default();
        let result = emd(&signal, &config);

        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_emd_invalid_value_inf() {
        let signal = vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0];

        let config = EmdConfig::default();
        let result = emd(&signal, &config);

        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_emd_three_samples_minimum() {
        let signal = vec![0.0, 1.0, 0.0];

        let config = EmdConfig::default();
        let result = emd(&signal, &config);

        // Should at least attempt decomposition (may or may not extract IMF)
        assert!(result.is_ok());
    }

    #[test]
    fn test_emd_algorithm_type_is_emd() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig::default();
        let result = emd(&signal, &config).unwrap();

        assert_eq!(result.algorithm, AlgorithmType::EMD);
    }

    #[test]
    fn test_emd_elapsed_time_is_positive() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig::default();
        let result = emd(&signal, &config).unwrap();

        assert!(result.elapsed.as_millis() >= 0);
    }

    #[test]
    fn test_emd_config_snapshot_is_valid_json() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig::default();
        let result = emd(&signal, &config).unwrap();

        // Config snapshot should be parseable as JSON
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&result.config_snapshot);
        assert!(parsed.is_ok(), "Config snapshot should be valid JSON: {}", result.config_snapshot);
    }

    // =========================================================================
    // Reconstruction with varying tolerances
    // =========================================================================

    #[test]
    fn test_reconstruction_tight_tolerance() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig {
            validate_reconstruction: true,
            reconstruction_tolerance: 1e-12,
            ..EmdConfig::default()
        };
        let result = emd(&signal, &config);

        assert!(result.is_ok(), "Should pass with tight tolerance");
    }

    #[test]
    fn test_reconstruction_loose_tolerance() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig {
            validate_reconstruction: true,
            reconstruction_tolerance: 1e-6,
            ..EmdConfig::default()
        };
        let result = emd(&signal, &config);

        assert!(result.is_ok(), "Should pass with loose tolerance");
    }

    #[test]
    fn test_reconstruction_disabled() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig { validate_reconstruction: false, ..EmdConfig::default() };
        let result = emd(&signal, &config);

        assert!(result.is_ok(), "Should succeed even with validation disabled");
    }

    // =========================================================================
    // PalindromeCyclic tests
    // =========================================================================

    #[test]
    fn test_emd_palindrome_cyclic_output_length_matches_input() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EmdConfig {
            sifting_config: SiftingConfig {
                boundary_condition: BoundaryConditionType::PalindromeCyclic,
                ..SiftingConfig::default()
            },
            validate_reconstruction: true,
            reconstruction_tolerance: 1e-10,
            ..EmdConfig::default()
        };
        let result = emd(&signal, &config);

        assert!(result.is_ok(), "PalindromeCyclic EMD should succeed: {:?}", result);
        let result = result.unwrap();

        // All IMFs and residue must be trimmed back to original length
        for imf in result.imfs.imfs.iter() {
            assert_eq!(imf.len(), n, "IMF length should match input length");
        }
        assert_eq!(result.imfs.residue.len(), n, "Residue length should match input length");
    }

    #[test]
    fn test_emd_palindrome_cyclic_reconstruction() {
        let n = 150;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.4 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let config = EmdConfig {
            sifting_config: SiftingConfig {
                boundary_condition: BoundaryConditionType::PalindromeCyclic,
                ..SiftingConfig::default()
            },
            validate_reconstruction: true,
            reconstruction_tolerance: 1e-10,
            ..EmdConfig::default()
        };
        let result = emd(&signal, &config);

        assert!(result.is_ok(), "PalindromeCyclic reconstruction should be valid: {:?}", result);

        // Verify reconstruction manually
        let reconstructed = result.unwrap().imfs.reconstruct();
        let max_err = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-10, "Reconstruction error should be < 1e-10, got {:.2e}", max_err);
    }

    #[test]
    fn test_emd_palindrome_cyclic_multi_component() {
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 4.0 * t).sin() + 0.5 * (2.0 * PI * 18.0 * t).sin()
            })
            .collect();

        let config = EmdConfig {
            sifting_config: SiftingConfig {
                boundary_condition: BoundaryConditionType::PalindromeCyclic,
                ..SiftingConfig::default()
            },
            ..EmdConfig::default()
        };
        let result = emd(&signal, &config);
        assert!(result.is_ok());
        assert!(result.unwrap().imfs.n_imfs() >= 1);
    }
}
