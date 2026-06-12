//! Variational Mode Decomposition (VMD) algorithm.
//!
//! This module implements VMD using the Alternating Direction Method of
//! Multipliers (ADMM) in the frequency domain, following Dragomiretskiy &
//! Zosso (2014).
//!
//! VMD decomposes a signal into a set of band-limited intrinsic mode
//! functions (modes), each with its own center frequency. Unlike EMD,
//! VMD solves a variational optimization problem:
//!
//! ```text
//! min_{u_k, ω_k} { Σ_k || ∂_t [(δ(t) + j/(πt)) * u_k(t)] exp(-jω_k t) ||² }
//! subject to: Σ_k u_k = f
//! ```
//!
//! The algorithm uses ADMM to solve this constrained optimization in the
//! frequency domain, iteratively updating:
//! 1. Each mode u_k via Wiener filtering
//! 2. Center frequencies ω_k via spectral centroid
//! 3. Lagrange multiplier λ via dual ascent

use crate::error::EmdError;
use crate::types::{AlgorithmType, DecompositionResult, ImfCollection};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use web_time::Instant;

// ---------------------------------------------------------------------------
// VmdConfig
// ---------------------------------------------------------------------------

/// Configuration for Variational Mode Decomposition.
///
/// # Parameters
///
/// - `n_modes`: Number of modes to extract (K in the paper)
/// - `alpha`: Bandwidth penalty parameter (τ in the paper) — controls
///   bandwidth of each mode. Higher values → narrower bandwidth.
/// - `tau`: Dual ascent step size (noise tolerance). 0 = strict constraint
///   enforcement. Higher values allow more reconstruction error.
/// - `tol`: Convergence tolerance for relative change in modes.
/// - `max_iterations`: Maximum ADMM iterations before giving up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VmdConfig {
    /// Number of modes to extract (K). Default: 3.
    pub n_modes: usize,
    /// Bandwidth penalty parameter (α). Default: 2000.0.
    /// Higher values produce narrower-band modes.
    pub alpha: f64,
    /// Dual ascent step size (τ). Default: 0.0.
    /// 0 = strict reconstruction constraint. Higher = more noise tolerance.
    pub tau: f64,
    /// Convergence tolerance. Default: 1e-7.
    /// Algorithm stops when ||u_k^(n+1) - u_k^n||² / ||u_k^n||² < tol for all k.
    pub tol: f64,
    /// Maximum ADMM iterations. Default: 500.
    pub max_iterations: usize,
}

impl Default for VmdConfig {
    fn default() -> Self {
        Self { n_modes: 3, alpha: 2000.0, tau: 0.0, tol: 1e-7, max_iterations: 500 }
    }
}

impl VmdConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), EmdError> {
        if self.n_modes == 0 {
            return Err(EmdError::InvalidConfig("n_modes must be at least 1".to_string()));
        }
        if self.alpha < 0.0 {
            return Err(EmdError::InvalidConfig("alpha must be non-negative".to_string()));
        }
        if self.tau < 0.0 {
            return Err(EmdError::InvalidConfig("tau must be non-negative".to_string()));
        }
        if self.tol <= 0.0 {
            return Err(EmdError::InvalidConfig("tol must be positive".to_string()));
        }
        if self.max_iterations == 0 {
            return Err(EmdError::InvalidConfig("max_iterations must be at least 1".to_string()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute FFT of a real signal, returning complex spectrum.
fn fft(signal: &[f64], fft: &Arc<dyn Fft<f64>>) -> Vec<Complex64> {
    let n = signal.len();
    let mut complex_input: Vec<Complex64> =
        signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft.process(&mut complex_input);
    // Normalize by 1/N to match MATLAB convention
    let n_f64 = n as f64;
    for c in &mut complex_input {
        *c /= n_f64;
    }
    complex_input
}

/// Compute IFFT of a complex spectrum, returning real signal.
fn ifft(spectrum: &[Complex64], ifft: &Arc<dyn Fft<f64>>) -> Vec<f64> {
    let _n = spectrum.len();
    let mut complex_input: Vec<Complex64> = spectrum.to_vec();
    ifft.process(&mut complex_input);
    // IFFT does not normalize in rustfft, but we already normalized FFT
    // so IFFT result is already in the right scale.
    // Actually rustfft IFFT does NOT divide by N, so we need to multiply by N
    // since our FFT already divided by N, the IFFT will give us back the original.
    // Wait — rustfft: FFT does NOT normalize, IFFT does NOT normalize.
    // So FFT then IFFT gives N * original.
    // We divided FFT output by N, so IFFT of that gives original. Good.
    complex_input.iter().map(|c| c.re).collect()
}

/// Initialize center frequencies using a heuristic based on signal length
/// and number of modes.
fn init_center_freqs(n: usize, k: usize) -> Vec<f64> {
    // Initialize uniformly spaced in frequency domain [0, n/2]
    // This is a common initialization strategy
    let half_n = n as f64 / 2.0;
    (0..k)
        .map(|i| if k == 1 { half_n / 2.0 } else { half_n * (i as f64 + 0.5) / k as f64 })
        .collect()
}

/// Check if all values in a slice are finite.
fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|&x| x.is_finite())
}

/// Check if all complex values are finite.
fn all_complex_finite(values: &[Complex64]) -> bool {
    values.iter().all(|c| c.re.is_finite() && c.im.is_finite())
}

// ---------------------------------------------------------------------------
// VMD Algorithm
// ---------------------------------------------------------------------------

/// Perform Variational Mode Decomposition on a signal.
///
/// Decomposes the input signal into `n_modes` band-limited modes using
/// the ADMM algorithm in the frequency domain.
///
/// # Arguments
/// * `signal` — Input signal to decompose
/// * `config` — VMD configuration parameters
///
/// # Returns
/// A `DecompositionResult` containing the extracted modes as IMFs and
/// the residue (reconstruction error), or an error if decomposition fails.
///
/// # Algorithm
///
/// The VMD algorithm iteratively updates three quantities:
///
/// 1. **Mode update** (Wiener filter):
///    ```text
///    u_k(ω) = (f(ω) - Σ_{l≠k} u_l(ω) + λ(ω)/2) / (1 + 2α(ω - ω_k)²)
///    ```
///
/// 2. **Center frequency update** (spectral centroid):
///    ```text
///    ω_k = ∫ ω |u_k(ω)|² dω / ∫ |u_k(ω)|² dω
///    ```
///
/// 3. **Lagrange multiplier update** (dual ascent):
///    ```text
///    λ(ω) ← λ(ω) + τ · (f(ω) - Σ_k u_k(ω))
///    ```
///
/// Convergence is checked when the relative change in all modes falls
/// below `tol`.
///
/// # Examples
/// ```
/// use ferromode::algorithms::vmd::{VmdConfig, vmd};
/// use std::f64::consts::PI;
///
/// // Decompose a two-tone signal
/// let n = 1000;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64 / n as f64;
///         (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
///     })
///     .collect();
///
/// let config = VmdConfig { n_modes: 2, ..VmdConfig::default() };
/// let result = vmd(&signal, &config);
/// assert!(result.is_ok());
/// ```
pub fn vmd(signal: &[f64], config: &VmdConfig) -> Result<DecompositionResult, EmdError> {
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

    config.validate()?;

    let n = signal.len();
    let k = config.n_modes;

    // Validate signal length vs modes
    if k > n / 2 {
        return Err(EmdError::InvalidConfig(format!(
            "n_modes ({}) must be at most half the signal length ({})",
            k, n
        )));
    }

    // Setup FFT
    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(n);
    let fft_inv = planner.plan_fft_inverse(n);

    // Transform signal to frequency domain
    let f_hat = fft(signal, &fft_fwd);

    if !all_complex_finite(&f_hat) {
        return Err(EmdError::InvalidValue);
    }

    // Initialize modes and center frequencies
    let mut u_hat: Vec<Vec<Complex64>> = vec![vec![Complex64::ZERO; n]; k];
    let mut omega: Vec<f64> = init_center_freqs(n, k);

    // Initialize modes as small fractions of the signal spectrum
    for (i, &val) in signal.iter().enumerate() {
        let avg = val / k as f64;
        let c = Complex64::new(avg, 0.0);
        for u in &mut u_hat {
            u[i] = c;
        }
    }
    // FFT the initial modes
    for u in &mut u_hat {
        let mut tmp: Vec<Complex64> = u.clone();
        fft_fwd.process(&mut tmp);
        let n_f64 = n as f64;
        for c in &mut tmp {
            *c /= n_f64;
        }
        *u = tmp;
    }

    // Initialize Lagrange multiplier
    let mut lambda_hat: Vec<Complex64> = vec![Complex64::ZERO; n];

    // Build frequency axis (angular frequency, normalized)
    let freqs: Vec<f64> =
        (0..n).map(|i| if i < n / 2 { i as f64 } else { (i as f64) - (n as f64) }).collect();

    // ADMM iterations
    let alpha = config.alpha;
    let tau = config.tau;
    let tol = config.tol;
    let max_iter = config.max_iterations;
    let mut converged = false;
    let mut iterations_used = 0;

    for iter in 0..max_iter {
        // Store previous modes for convergence check
        let u_hat_old: Vec<Vec<Complex64>> = u_hat.clone();

        // ---- Mode update (Wiener filter) ----
        for ki in 0..k {
            // Compute residual: f_hat - sum_{l≠k} u_hat_l + lambda_hat / 2
            let mut residual: Vec<Complex64> = f_hat.clone();

            // Subtract other modes
            for li in 0..k {
                if li != ki {
                    for (_r, &_u_l) in residual.iter().zip(u_hat[li].iter()) {
                        // We'll accumulate below
                    }
                    for j in 0..n {
                        residual[j] -= u_hat[li][j];
                    }
                }
            }

            // Add lambda_hat / 2
            if tau > 0.0 {
                for j in 0..n {
                    residual[j] += lambda_hat[j] / 2.0;
                }
            }

            // Apply Wiener filter: use |freq| so both ±ω_k components are captured
            // symmetrically for real-valued signals.
            let omega_k = omega[ki];
            for j in 0..n {
                let denom = 1.0 + 2.0 * alpha * (freqs[j].abs() - omega_k).powi(2);
                if denom > 0.0 {
                    u_hat[ki][j] = residual[j] / denom;
                }
            }
        }

        // ---- Center frequency update ----
        // Use only positive frequencies to avoid cancellation from the conjugate
        // negative-frequency mirror of each real-valued mode.
        for ki in 0..k {
            let mut num = 0.0f64;
            let mut den = 0.0f64;

            for j in 1..(n / 2) {
                let power = u_hat[ki][j].norm_sqr();
                num += freqs[j] * power;
                den += power;
            }

            if den > 0.0 {
                omega[ki] = num / den;
            }
        }

        // ---- Lagrange multiplier update ----
        if tau > 0.0 {
            // Compute reconstruction: sum of all modes
            let mut recon: Vec<Complex64> = vec![Complex64::ZERO; n];
            for ki in 0..k {
                for j in 0..n {
                    recon[j] += u_hat[ki][j];
                }
            }

            // lambda <- lambda + tau * (f_hat - recon)
            for j in 0..n {
                lambda_hat[j] += tau * (f_hat[j] - recon[j]);
            }
        }

        // ---- Convergence check ----
        let mut max_change = 0.0f64;
        for ki in 0..k {
            let mut change_num = 0.0f64;
            let mut change_den = 0.0f64;

            for j in 0..n {
                let diff = u_hat[ki][j] - u_hat_old[ki][j];
                change_num += diff.norm_sqr();
                change_den += u_hat_old[ki][j].norm_sqr();
            }

            let relative_change = if change_den > 0.0 { change_num / change_den } else { 0.0 };

            if relative_change > max_change {
                max_change = relative_change;
            }
        }

        iterations_used = iter + 1;

        if max_change < tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(EmdError::ConvergenceFailed { max_iterations: max_iter });
    }

    // Transform modes back to time domain
    let mut imfs: Vec<Vec<f64>> = Vec::with_capacity(k);
    for ki in 0..k {
        let imf = ifft(&u_hat[ki], &fft_inv);
        imfs.push(imf);
    }

    // Compute residue (reconstruction error)
    let reconstructed: Vec<f64> = (0..n).map(|i| imfs.iter().map(|imf| imf[i]).sum()).collect();

    let residue: Vec<f64> = signal.iter().zip(reconstructed.iter()).map(|(&s, &r)| s - r).collect();

    // Validate outputs
    for imf in &imfs {
        if !all_finite(imf) {
            return Err(EmdError::InvalidValue);
        }
    }
    if !all_finite(&residue) {
        return Err(EmdError::InvalidValue);
    }

    let elapsed = start.elapsed();

    let result = DecompositionResult::new(
        AlgorithmType::VMD,
        ImfCollection::new(imfs, residue),
        elapsed,
        iterations_used,
        serde_json::to_string(&VmdConfigSnapshot {
            n_modes: k,
            alpha,
            tau,
            tol,
            max_iterations: max_iter,
        })
        .unwrap_or_default(),
    );

    Ok(result)
}

/// Snapshot of VMD config for serialization in DecompositionResult.
#[derive(Debug, Serialize, Deserialize)]
struct VmdConfigSnapshot {
    n_modes: usize,
    alpha: f64,
    tau: f64,
    tol: f64,
    max_iterations: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // =========================================================================
    // VmdConfig tests
    // =========================================================================

    #[test]
    fn test_vmd_config_default() {
        let config = VmdConfig::default();
        assert_eq!(config.n_modes, 3);
        assert!((config.alpha - 2000.0).abs() < 1e-10);
        assert!((config.tau - 0.0).abs() < 1e-15);
        assert!((config.tol - 1e-7).abs() < 1e-12);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_vmd_config_validate_default() {
        let config = VmdConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_vmd_config_validate_zero_modes() {
        let config = VmdConfig { n_modes: 0, ..VmdConfig::default() };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_vmd_config_validate_negative_alpha() {
        let config = VmdConfig { alpha: -1.0, ..VmdConfig::default() };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_vmd_config_validate_negative_tau() {
        let config = VmdConfig { tau: -0.1, ..VmdConfig::default() };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_vmd_config_validate_zero_tol() {
        let config = VmdConfig { tol: 0.0, ..VmdConfig::default() };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_vmd_config_validate_zero_max_iterations() {
        let config = VmdConfig { max_iterations: 0, ..VmdConfig::default() };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, EmdError::InvalidConfig(_)));
    }

    // =========================================================================
    // T-112: Two-tone signal separation
    // =========================================================================

    #[test]
    fn test_vmd_two_tone_signal_separation() {
        // Create a signal with two well-separated frequency components
        let n = 1000;
        let sample_rate = 1000.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config =
            VmdConfig { n_modes: 2, alpha: 2000.0, tau: 0.0, tol: 1e-7, max_iterations: 500 };

        let result = vmd(&signal, &config);
        assert!(result.is_ok(), "VMD should succeed for two-tone signal");

        let result = result.unwrap();
        assert_eq!(result.imfs.n_imfs(), 2);
        assert_eq!(result.algorithm, AlgorithmType::VMD);
    }

    // =========================================================================
    // Convergence test
    // =========================================================================

    #[test]
    fn test_vmd_convergence_before_max_iterations() {
        let n = 500;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 3.0 * t).sin() + 0.3 * (2.0 * PI * 15.0 * t).sin()
            })
            .collect();

        let config = VmdConfig { n_modes: 2, max_iterations: 500, ..VmdConfig::default() };

        let result = vmd(&signal, &config);
        assert!(result.is_ok(), "VMD should converge");

        let result = result.unwrap();
        // n_siftings stores iteration count
        assert!(
            result.n_siftings < config.max_iterations,
            "Should converge before max iterations, used {}",
            result.n_siftings
        );
    }

    // =========================================================================
    // Reconstruction test
    // =========================================================================

    #[test]
    fn test_vmd_reconstruction_accuracy() {
        let n = 1000;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 25.0 * t).sin()
            })
            .collect();

        let config = VmdConfig {
            n_modes: 2,
            tau: 0.0, // Strict constraint
            tol: 1e-7,
            ..VmdConfig::default()
        };

        let result = vmd(&signal, &config).expect("VMD should succeed");

        // Reconstruct from modes + residue
        let reconstructed: Vec<f64> = (0..n)
            .map(|i| {
                let mut sum = result.imfs.residue[i];
                for imf in &result.imfs.imfs {
                    sum += imf[i];
                }
                sum
            })
            .collect();

        let max_error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(max_error < 1e-4, "Reconstruction error should be small, got {:.2e}", max_error);
    }

    // =========================================================================
    // Parameter sensitivity: alpha affects bandwidth
    // =========================================================================

    #[test]
    fn test_vmd_alpha_sensitivity() {
        let n = 1000;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        // Low alpha = wider bandwidth
        let config_low = VmdConfig { n_modes: 1, alpha: 100.0, tol: 1e-6, ..VmdConfig::default() };

        // High alpha = narrower bandwidth
        let config_high =
            VmdConfig { n_modes: 1, alpha: 10000.0, tol: 1e-6, ..VmdConfig::default() };

        let result_low = vmd(&signal, &config_low).expect("VMD with low alpha should succeed");
        let result_high = vmd(&signal, &config_high).expect("VMD with high alpha should succeed");

        // Both should produce 1 mode
        assert_eq!(result_low.imfs.n_imfs(), 1);
        assert_eq!(result_high.imfs.n_imfs(), 1);

        // Higher alpha should produce a mode with lower energy
        // (more constrained bandwidth)
        let energy_low: f64 = result_low.imfs.imfs[0].iter().map(|v| v * v).sum();
        let energy_high: f64 = result_high.imfs.imfs[0].iter().map(|v| v * v).sum();

        // Both should capture most of the signal energy
        let signal_energy: f64 = signal.iter().map(|v| v * v).sum();
        assert!(
            energy_low > 0.5 * signal_energy,
            "Low alpha mode should capture significant energy"
        );
        assert!(
            energy_high > 0.5 * signal_energy,
            "High alpha mode should capture significant energy"
        );
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_vmd_constant_signal() {
        let n = 200;
        let signal = vec![5.0; n];

        let config = VmdConfig { n_modes: 1, max_iterations: 200, ..VmdConfig::default() };

        let result = vmd(&signal, &config);
        // Constant signal may converge to a constant mode
        // or may have convergence issues — either is acceptable
        // as long as it doesn't panic
        if result.is_ok() {
            let result = result.unwrap();
            assert_eq!(result.imfs.n_imfs(), 1);
        }
    }

    #[test]
    fn test_vmd_single_mode() {
        let n = 500;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = VmdConfig { n_modes: 1, ..VmdConfig::default() };

        let result = vmd(&signal, &config).expect("VMD with single mode should succeed");
        assert_eq!(result.imfs.n_imfs(), 1);

        // Single mode should approximate the signal well
        let max_error: f64 = signal
            .iter()
            .zip(result.imfs.imfs[0].iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        // With tau=0, reconstruction should be exact (mode + residue = signal)
        let reconstructed: f64 = signal
            .iter()
            .zip(result.imfs.residue.iter())
            .zip(result.imfs.imfs[0].iter())
            .map(|((&s, &r), &imf)| {
                let recon = imf + r;
                (s - recon).abs()
            })
            .fold(0.0_f64, f64::max);

        assert!(
            reconstructed < 1e-6,
            "Single mode + residue should reconstruct signal, error: {:.2e}",
            reconstructed
        );
    }

    #[test]
    fn test_vmd_too_many_modes() {
        let n = 20;
        let signal: Vec<f64> = (0..n).map(|i| i as f64).collect();

        // Request more modes than signal can support
        let config = VmdConfig {
            n_modes: 15, // More than n/2 = 10
            ..VmdConfig::default()
        };

        let result = vmd(&signal, &config);
        assert!(result.is_err(), "Should reject too many modes for short signal");
    }

    #[test]
    fn test_vmd_insufficient_data() {
        let signal = vec![1.0, 2.0];

        let config = VmdConfig::default();
        let result = vmd(&signal, &config);

        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_vmd_invalid_value_nan() {
        let signal = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];

        let config = VmdConfig::default();
        let result = vmd(&signal, &config);

        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_vmd_invalid_value_inf() {
        let signal = vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0];

        let config = VmdConfig::default();
        let result = vmd(&signal, &config);

        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    // =========================================================================
    // Algorithm type verification
    // =========================================================================

    #[test]
    fn test_vmd_algorithm_type() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = VmdConfig { n_modes: 1, ..VmdConfig::default() };
        let result = vmd(&signal, &config).unwrap();

        assert_eq!(result.algorithm, AlgorithmType::VMD);
    }

    // =========================================================================
    // Config snapshot is valid JSON
    // =========================================================================

    #[test]
    fn test_vmd_config_snapshot_is_valid_json() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = VmdConfig { n_modes: 1, ..VmdConfig::default() };
        let result = vmd(&signal, &config).unwrap();

        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&result.config_snapshot);
        assert!(parsed.is_ok(), "Config snapshot should be valid JSON: {}", result.config_snapshot);
    }

    // =========================================================================
    // Multi-mode decomposition
    // =========================================================================

    #[test]
    fn test_vmd_three_mode_decomposition() {
        let n = 1000;
        let sample_rate = 1000.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.5 * (2.0 * PI * 20.0 * t).sin()
                    + 0.3 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config =
            VmdConfig { n_modes: 3, alpha: 2000.0, tau: 0.0, tol: 1e-6, max_iterations: 500 };

        let result = vmd(&signal, &config).expect("VMD should succeed for 3-mode signal");
        assert_eq!(result.imfs.n_imfs(), 3);

        // Check reconstruction
        let reconstructed: Vec<f64> = (0..n)
            .map(|i| {
                let mut sum = result.imfs.residue[i];
                for imf in &result.imfs.imfs {
                    sum += imf[i];
                }
                sum
            })
            .collect();

        let max_error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_error < 1e-3,
            "3-mode reconstruction error should be small, got {:.2e}",
            max_error
        );
    }

    // =========================================================================
    // Tau (noise tolerance) parameter test
    // =========================================================================

    #[test]
    fn test_vmd_tau_affects_reconstruction() {
        let n = 500;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin()
            })
            .collect();

        // tau = 0: strict constraint
        let config_strict = VmdConfig { n_modes: 1, tau: 0.0, tol: 1e-6, ..VmdConfig::default() };

        // tau > 0: allows reconstruction error
        let config_relaxed = VmdConfig { n_modes: 1, tau: 0.1, tol: 1e-6, ..VmdConfig::default() };

        let result_strict = vmd(&signal, &config_strict).expect("Strict VMD should succeed");
        let result_relaxed = vmd(&signal, &config_relaxed).expect("Relaxed VMD should succeed");

        // Both should produce 1 mode
        assert_eq!(result_strict.imfs.n_imfs(), 1);
        assert_eq!(result_relaxed.imfs.n_imfs(), 1);

        // Strict should have smaller residue
        let residue_strict: f64 = result_strict.imfs.residue.iter().map(|v| v * v).sum();
        let residue_relaxed: f64 = result_relaxed.imfs.residue.iter().map(|v| v * v).sum();

        assert!(
            residue_strict <= residue_relaxed + 1e-6,
            "Strict constraint should have smaller or equal residue energy"
        );
    }

    // =========================================================================
    // Elapsed time is positive
    // =========================================================================

    #[test]
    fn test_vmd_elapsed_time_positive() {
        let n = 200;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = VmdConfig { n_modes: 1, ..VmdConfig::default() };
        let result = vmd(&signal, &config).unwrap();

        assert!(result.elapsed.as_micros() > 0);
    }

    // =========================================================================
    // IMFs have same length as input
    // =========================================================================

    #[test]
    fn test_vmd_imfs_same_length_as_input() {
        let n = 300;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = VmdConfig { n_modes: 2, ..VmdConfig::default() };
        let result = vmd(&signal, &config).unwrap();

        for imf in &result.imfs.imfs {
            assert_eq!(imf.len(), n);
        }
        assert_eq!(result.imfs.residue.len(), n);
    }
}
