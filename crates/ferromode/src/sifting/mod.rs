#![warn(missing_docs)]

//! Sifting engine for Empirical Mode Decomposition.
//!
//! This module implements the core sifting algorithm that extracts Intrinsic Mode Functions (IMFs)
//! from signals through iterative envelope interpolation and residue computation.

use crate::boundary::{
    build_envelope_knots, get_strategy, BoundaryCondition, BoundaryConditionType, ExtendedSignal,
    Extrema,
};
use crate::error::EmdError;
use crate::extrema::{detect_extrema, detect_extrema_with_endpoints, Extrema as ExtremaStruct};
use crate::spline::circular::CircularSpline;
use crate::spline::cubic::CubicSpline;
use crate::spline::Spline;
use serde::{Deserialize, Serialize};

/// End-condition strategy for cubic spline envelope interpolation.
///
/// Controls how the spline behaves at the first and last knot. This has a
/// meaningful effect on envelope quality near signal boundaries and therefore
/// on the frequency content of each extracted IMF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplineEndCondition {
    /// Not-a-knot: forces continuity of the third derivative across the first
    /// and last interior knots. Matches `scipy.interpolate.CubicSpline` default
    /// and PyEMD behaviour. Generally the best choice for EMD envelope fitting.
    NotAKnot,
    /// Natural: sets the second derivative to zero at both endpoints. Produces
    /// slightly lower curvature near boundaries; can underestimate envelope
    /// amplitude at signal edges.
    Natural,
    /// Periodic: enforces equal first and second derivatives at both ends.
    /// Use only when the signal is genuinely periodic.
    Periodic,
}

/// Configuration for sifting operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiftingConfig {
    /// Maximum number of sifting iterations allowed
    pub max_sifting_iterations: usize,
    /// SD threshold for stopping criterion
    pub sd_threshold: f64,
    /// S-number for consecutive siftings criterion
    pub s_number: usize,
    /// Fixed number of iterations for stopping
    pub fixed_iterations: Option<usize>,
    /// Energy difference threshold for stopping
    pub energy_threshold: f64,
    /// Boundary condition strategy to use
    pub boundary_condition: BoundaryConditionType,
    /// End-condition for cubic spline envelope interpolation
    pub spline_end_condition: SplineEndCondition,
}

impl Default for SiftingConfig {
    fn default() -> Self {
        Self {
            max_sifting_iterations: 100,
            sd_threshold: 0.2,
            // PyEMD's default f2 check is single-iteration (not accumulated).
            // s_number: 1 reproduces that — stop as soon as one iteration
            // satisfies |#extrema - #zero_crossings| ≤ 1.
            s_number: 1,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: BoundaryConditionType::ExtremasMirror,
            spline_end_condition: SplineEndCondition::NotAKnot,
        }
    }
}

/// Stopping criteria for sifting iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoppingCriterion {
    /// Stop when SD = Σ|h_{k-1} - h_k|² / Σ|h_{k-1}|² < threshold
    SdThreshold,
    /// Stop after S consecutive siftings where #extrema and #zero-crossings differ by ≤ 1
    SNumber,
    /// Stop after fixed number of iterations
    FixedIterations,
    /// Stop when energy difference between iterations falls below threshold
    EnergyDifference,
}

fn build_spline(
    x: &[f64],
    y: &[f64],
    end_cond: SplineEndCondition,
) -> Result<Box<dyn Spline>, EmdError> {
    match end_cond {
        SplineEndCondition::NotAKnot => Ok(Box::new(CubicSpline::not_a_knot_from_knots(x, y)?)),
        SplineEndCondition::Natural  => Ok(Box::new(CubicSpline::from_knots(x, y)?)),
        SplineEndCondition::Periodic => Ok(Box::new(CircularSpline::from_knots(x, y)?)),
    }
}

/// Sifting engine for extracting IMFs from signals.
pub struct SiftingEngine {
    config: SiftingConfig,
    stopping_criteria: Vec<StoppingCriterion>,
    boundary_strategy: Box<dyn BoundaryCondition>,
}

impl SiftingEngine {
    /// Create a new sifting engine with the given configuration.
    pub fn new(config: SiftingConfig, stopping_criteria: Vec<StoppingCriterion>) -> Self {
        let boundary_strategy = get_strategy(config.boundary_condition);
        Self { config, stopping_criteria, boundary_strategy }
    }

    /// Extract a single IMF from the signal using configured stopping criteria.
    ///
    /// # Arguments
    /// * `signal` - Input signal to decompose
    ///
    /// # Returns
    /// Tuple of (extracted_imf, residue_signal)
    pub fn sift_one(&self, signal: &[f64]) -> Result<(Vec<f64>, Vec<f64>), EmdError> {
        if signal.len() < 3 {
            return Err(EmdError::InsufficientData);
        }

        for &val in signal {
            if !val.is_finite() {
                return Err(EmdError::InvalidValue);
            }
        }

        let mut h = signal.to_vec();
        let mut iteration_count = 0;
        let mut consecutive_s_count = 0;
        let mut prev_energy = compute_energy(&h);

        let fixed_limit = self.config.fixed_iterations;

        loop {
            if iteration_count >= self.config.max_sifting_iterations {
                return Err(EmdError::ConvergenceFailed {
                    max_iterations: self.config.max_sifting_iterations,
                });
            }

            if let Some(fixed) = fixed_limit {
                if iteration_count >= fixed {
                    let residue: Vec<f64> =
                        signal.iter().zip(h.iter()).map(|(&s, &imf)| s - imf).collect();
                    return Ok((h, residue));
                }
            }

            // Always use interior-only extrema detection here.
            // For ExtremasMirror, build_envelope_knots handles boundaries by
            // reflecting interior extrema — it must NOT receive pre-injected
            // endpoint extrema or the reflected knots land on top of the
            // boundary, creating sharp spline features.
            let extrema = detect_extrema(&h);
            let num_extrema = extrema.maxima.len() + extrema.minima.len();
            let num_zero_crossings = count_zero_crossings(&h);

            if num_extrema <= 1 {
                let residue: Vec<f64> =
                    signal.iter().zip(h.iter()).map(|(&s, &imf)| s - imf).collect();
                return Ok((h, residue));
            }

            let extrema_for_boundary = Extrema {
                maxima_indices: extrema.maxima.clone(),
                minima_indices: extrema.minima.clone(),
            };

            // Zhao-Huang periodic boundary always uses the periodic (circular)
            // spline — the palindrome extension makes the extrema sequence
            // genuinely closed, so no endpoint derivative assumption is needed.
            let ec = if self.config.boundary_condition == BoundaryConditionType::Periodic {
                SplineEndCondition::Periodic
            } else {
                self.config.spline_end_condition
            };
            let n = h.len();

            let (upper_env, lower_env) =
                if self.config.boundary_condition == BoundaryConditionType::ExtremasMirror {
                    // Full PyEMD-compatible conditional extrema mirroring
                    let (max_idx, max_val, min_idx, min_val) = build_envelope_knots(
                        &h,
                        &extrema.maxima,
                        &extrema.minima,
                        2,
                    );

                    let upper = if max_idx.len() >= 2 {
                        let spl = build_spline(&max_idx, &max_val, ec)?;
                        (0..n).map(|i| spl.evaluate(i as f64)).collect()
                    } else {
                        vec![0.0; n]
                    };
                    let lower = if min_idx.len() >= 2 {
                        let spl = build_spline(&min_idx, &min_val, ec)?;
                        (0..n).map(|i| spl.evaluate(i as f64)).collect()
                    } else {
                        vec![0.0; n]
                    };
                    (upper, lower)
                } else {
                    // Extend the signal with boundary padding, then detect
                    // extrema in the *extended* signal so that the spline
                    // sees the boundary knots added by the strategy.
                    let extended = self.boundary_strategy.extend(&h, &extrema_for_boundary);
                    let ext_extrema = detect_extrema(&extended.values);

                    let max_indices: Vec<f64> =
                        ext_extrema.maxima.iter().map(|&i| i as f64).collect();
                    let max_values: Vec<f64> =
                        ext_extrema.maxima.iter().map(|&i| extended.values[i]).collect();
                    let min_indices: Vec<f64> =
                        ext_extrema.minima.iter().map(|&i| i as f64).collect();
                    let min_values: Vec<f64> =
                        ext_extrema.minima.iter().map(|&i| extended.values[i]).collect();

                    let upper = if max_indices.len() >= 2 {
                        let spl = build_spline(&max_indices, &max_values, ec)?;
                        (extended.original_start..extended.original_end)
                            .map(|i| spl.evaluate(i as f64))
                            .collect()
                    } else {
                        vec![0.0; n]
                    };
                    let lower = if min_indices.len() >= 2 {
                        let spl = build_spline(&min_indices, &min_values, ec)?;
                        (extended.original_start..extended.original_end)
                            .map(|i| spl.evaluate(i as f64))
                            .collect()
                    } else {
                        vec![0.0; n]
                    };
                    (upper, lower)
                };

            let mean_env = compute_mean_envelope(&upper_env, &lower_env);

            let h_prev = h.clone();
            h = h.iter().zip(mean_env.iter()).map(|(&hi, &mi)| hi - mi).collect();

            iteration_count += 1;

            // ── Stopping criteria ─────────────────────────────────────────────
            // PyEMD default: stop when BOTH f1 (envelope converged) AND f2
            // (signal looks like an IMF) are satisfied in the same iteration.
            // Using OR logic lets either criterion fire alone, which terminates
            // sifting too early and causes mode mixing.
            //
            // We evaluate each active criterion, then combine them:
            //   - SdThreshold / EnergyDifference  →  contribute to f1 (convergence)
            //   - SNumber                          →  contributes to f2 (IMF shape)
            //   - FixedIterations                  →  hard stop regardless

            let mut f1_met = false; // convergence gate
            let mut f2_met = false; // IMF-shape gate
            let mut hard_stop = false;

            for criterion in &self.stopping_criteria {
                match criterion {
                    StoppingCriterion::SdThreshold => {
                        // PyEMD envelope-sign guard: f1 only fires when the proto-IMF
                        // looks like a valid IMF — all local maxima ≥ 0 and all local
                        // minima ≤ 0. If this fails none of the convergence tests matter.
                        let maxima_ok = extrema.maxima.iter().all(|&i| h_prev[i] >= 0.0);
                        let minima_ok = extrema.minima.iter().all(|&i| h_prev[i] <= 0.0);

                        if maxima_ok && minima_ok {
                            // PyEMD check_imf: passes if ANY ONE of three tests fires.
                            //
                            // 1. Scaled variance: (h_prev - h)² / amplitude_range
                            let amplitude_range = {
                                let max_h = h_prev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let min_h = h_prev.iter().cloned().fold(f64::INFINITY, f64::min);
                                max_h - min_h
                            };
                            let diff_sq_sum: f64 = h_prev.iter().zip(h.iter())
                                .map(|(&p, &c)| (p - c).powi(2)).sum();
                            let svar = if amplitude_range > 0.0 {
                                diff_sq_sum / amplitude_range
                            } else {
                                0.0
                            };

                            // 2. Standard deviation: Σ((h_prev - h) / h)²  (PyEMD std)
                            let std_val = compute_sd(&h_prev, &h); // = Σ(diff²)/Σ(h_prev²)

                            // 3. Energy ratio: ‖h_prev - h‖² / ‖h_prev‖²
                            let h_prev_energy: f64 = h_prev.iter().map(|&x| x * x).sum();
                            let energy_ratio = if h_prev_energy > 0.0 {
                                diff_sq_sum / h_prev_energy
                            } else {
                                0.0
                            };

                            let h_energy: f64 = h.iter().map(|&x| x * x).sum();
                            if h_energy > 1e-10
                                && (svar < 0.001
                                    || std_val < self.config.sd_threshold
                                    || energy_ratio < self.config.sd_threshold)
                            {
                                f1_met = true;
                            }
                        }
                    }
                    StoppingCriterion::SNumber => {
                        let extrema_equal =
                            (num_extrema as isize - num_zero_crossings as isize).abs() <= 1;
                        if extrema_equal {
                            consecutive_s_count += 1;
                        } else {
                            consecutive_s_count = 0;
                        }
                        if consecutive_s_count >= self.config.s_number {
                            f2_met = true;
                        }
                    }
                    StoppingCriterion::FixedIterations => {
                        if let Some(fixed) = fixed_limit {
                            if iteration_count >= fixed {
                                hard_stop = true;
                            }
                        }
                    }
                    StoppingCriterion::EnergyDifference => {
                        let current_energy = compute_energy(&h);
                        let energy_diff = (current_energy - prev_energy).abs();
                        let energy_ratio =
                            if prev_energy > 0.0 { energy_diff / prev_energy } else { 0.0 };
                        if energy_ratio < self.config.energy_threshold {
                            f1_met = true;
                        }
                        prev_energy = current_energy;
                    }
                }
            }

            // Stop when both gates pass, or on a hard stop.
            // If only one type of criterion is active, treat it as sufficient
            // (e.g. FixedIterations-only, or SdThreshold-only configs).
            let has_f1_criterion = self.stopping_criteria.iter().any(|c| {
                matches!(c, StoppingCriterion::SdThreshold | StoppingCriterion::EnergyDifference)
            });
            let has_f2_criterion = self
                .stopping_criteria
                .iter()
                .any(|c| matches!(c, StoppingCriterion::SNumber));

            let should_stop = hard_stop
                || match (has_f1_criterion, has_f2_criterion) {
                    (true, true)   => f1_met && f2_met,  // PyEMD-style AND
                    (true, false)  => f1_met,
                    (false, true)  => f2_met,
                    (false, false) => false,
                };

            if should_stop {
                let residue: Vec<f64> =
                    signal.iter().zip(h.iter()).map(|(&s, &imf)| s - imf).collect();
                return Ok((h, residue));
            }
        }
    }

    /// Returns the current sifting configuration.
    pub fn config(&self) -> &SiftingConfig {
        &self.config
    }

    /// Returns the stopping criteria in use.
    pub fn stopping_criteria(&self) -> &[StoppingCriterion] {
        &self.stopping_criteria
    }
}

/// Compute the SD (Standard Deviation) stopping criterion.
///
/// SD = Σ|h_{k-1} - h_k|² / Σ|h_{k-1}|²
pub fn compute_sd(h_prev: &[f64], h_current: &[f64]) -> f64 {
    if h_prev.len() != h_current.len() || h_prev.is_empty() {
        return f64::INFINITY;
    }

    let numerator: f64 =
        h_prev.iter().zip(h_current.iter()).map(|(&prev, &curr)| (prev - curr).powi(2)).sum();

    let denominator: f64 = h_prev.iter().map(|&x| x.powi(2)).sum();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Check if signal satisfies S-number criterion.
///
/// Returns true if the last S siftings had equal or differing by 1
/// number of extrema and zero crossings.
pub fn check_s_number(extrema_counts: &[usize], zero_crossing_counts: &[usize], s: usize) -> bool {
    if extrema_counts.len() < s || zero_crossing_counts.len() < s {
        return false;
    }

    let start = extrema_counts.len() - s;
    for i in start..extrema_counts.len() {
        let diff = (extrema_counts[i] as isize - zero_crossing_counts[i] as isize).abs();
        if diff > 1 {
            return false;
        }
    }

    true
}

/// Count zero crossings in a signal.
pub fn count_zero_crossings(signal: &[f64]) -> usize {
    if signal.len() < 2 {
        return 0;
    }

    let mut count = 0;
    for i in 0..signal.len() - 1 {
        if (signal[i] >= 0.0 && signal[i + 1] < 0.0) || (signal[i] < 0.0 && signal[i + 1] >= 0.0) {
            count += 1;
        }
    }
    count
}

/// Compute energy of a signal (sum of squares).
pub fn compute_energy(signal: &[f64]) -> f64 {
    signal.iter().map(|&x| x * x).sum()
}

/// Extract upper and lower envelopes from signal using extrema and boundary conditions.
pub fn extract_envelopes(
    signal: &[f64],
    extrema: &ExtremaStruct,
    boundary_strategy: &dyn BoundaryCondition,
) -> Result<(Vec<f64>, Vec<f64>), EmdError> {
    if extrema.maxima.len() < 2 || extrema.minima.len() < 2 {
        return Err(EmdError::InsufficientData);
    }

    let extrema_for_boundary =
        Extrema { maxima_indices: extrema.maxima.clone(), minima_indices: extrema.minima.clone() };

    let extended = boundary_strategy.extend(signal, &extrema_for_boundary);

    let max_indices: Vec<f64> =
        extrema.maxima.iter().map(|&i| (i + extended.original_start) as f64).collect();
    let max_values: Vec<f64> = extrema.maxima.iter().map(|&i| signal[i]).collect();

    let min_indices: Vec<f64> =
        extrema.minima.iter().map(|&i| (i + extended.original_start) as f64).collect();
    let min_values: Vec<f64> = extrema.minima.iter().map(|&i| signal[i]).collect();

    let upper_spline = CubicSpline::from_knots(&max_indices, &max_values)?;
    let lower_spline = CubicSpline::from_knots(&min_indices, &min_values)?;
    // Note: extract_envelopes is a standalone utility; callers that need a
    // specific end condition should use SiftingEngine directly.

    let upper_env: Vec<f64> = (extended.original_start..extended.original_end)
        .map(|i| upper_spline.evaluate(i as f64))
        .collect();

    let lower_env: Vec<f64> = (extended.original_start..extended.original_end)
        .map(|i| lower_spline.evaluate(i as f64))
        .collect();

    Ok((upper_env, lower_env))
}

/// Compute mean of upper and lower envelopes.
pub fn compute_mean_envelope(upper: &[f64], lower: &[f64]) -> Vec<f64> {
    assert_eq!(upper.len(), lower.len(), "envelopes must have same length");
    upper.iter().zip(lower.iter()).map(|(&u, &l)| (u + l) / 2.0).collect()
}

/// Convenience function to sift one IMF with default configuration.
///
/// # Arguments
/// * `signal` - Input signal
/// * `config` - Sifting configuration
///
/// # Returns
/// Tuple of (extracted_imf, residue_signal)
pub fn sift_one(signal: &[f64], config: &SiftingConfig) -> Result<(Vec<f64>, Vec<f64>), EmdError> {
    // Match PyEMD's two-gate default: stop when BOTH SD convergence (f1) AND
    // IMF shape (f2) are satisfied in the same iteration.
    // EnergyDifference is intentionally excluded here — it has no equivalent
    // in PyEMD or R EMD and causes premature termination when used alongside
    // SdThreshold. It remains available for users who construct SiftingEngine
    // directly and explicitly want energy-based stopping.
    let engine = SiftingEngine::new(
        config.clone(),
        vec![
            StoppingCriterion::SdThreshold,
            StoppingCriterion::SNumber,
        ],
    );
    engine.sift_one(signal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // =========================================================================
    // SiftingConfig tests
    // =========================================================================

    #[test]
    fn test_sifting_config_default() {
        let config = SiftingConfig::default();
        assert_eq!(config.max_sifting_iterations, 100);
        assert_eq!(config.sd_threshold, 0.2);
        assert_eq!(config.s_number, 5);
        assert!(config.fixed_iterations.is_none());
        assert_eq!(config.energy_threshold, 1e-6);
        assert_eq!(config.boundary_condition, BoundaryConditionType::MirrorEven);
    }

    #[test]
    fn test_sifting_config_custom() {
        let config = SiftingConfig {
            max_sifting_iterations: 50,
            sd_threshold: 0.1,
            s_number: 3,
            fixed_iterations: Some(10),
            energy_threshold: 1e-8,
            boundary_condition: BoundaryConditionType::Periodic,
        };
        assert_eq!(config.max_sifting_iterations, 50);
        assert_eq!(config.sd_threshold, 0.1);
        assert_eq!(config.s_number, 3);
        assert_eq!(config.fixed_iterations, Some(10));
        assert_eq!(config.energy_threshold, 1e-8);
        assert_eq!(config.boundary_condition, BoundaryConditionType::Periodic);
    }

    // =========================================================================
    // compute_sd tests — T-056
    // =========================================================================

    #[test]
    fn test_compute_sd_identical_signals() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let sd = compute_sd(&a, &b);
        assert!(sd < 1e-10, "SD of identical signals should be 0, got {}", sd);
    }

    #[test]
    fn test_compute_sd_different_signals() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.1, 3.1, 4.1];
        let sd = compute_sd(&a, &b);
        assert!(sd > 0.0, "SD of different signals should be > 0");
        assert!(sd < 1.0, "SD should be small for small differences");
    }

    #[test]
    fn test_compute_sd_large_difference() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let sd = compute_sd(&a, &b);
        assert!(sd > 1.0, "SD should be large for large differences");
    }

    #[test]
    fn test_compute_sd_zero_signal() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sd = compute_sd(&a, &b);
        assert_eq!(sd, 0.0, "SD with zero previous signal should be 0");
    }

    #[test]
    fn test_compute_sd_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sd = compute_sd(&a, &b);
        assert_eq!(sd, f64::INFINITY);
    }

    #[test]
    fn test_compute_sd_empty_signals() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        let sd = compute_sd(&a, &b);
        assert_eq!(sd, f64::INFINITY);
    }

    #[test]
    fn test_compute_sd_single_element() {
        let a = vec![5.0];
        let b = vec![4.0];
        let sd = compute_sd(&a, &b);
        let expected = (5.0_f64 - 4.0_f64).powi(2) / 5.0_f64.powi(2);
        assert!((sd - expected).abs() < 1e-10);
    }

    // =========================================================================
    // count_zero_crossings tests
    // =========================================================================

    #[test]
    fn test_count_zero_crossings_simple() {
        let signal = vec![1.0, -1.0, 1.0, -1.0];
        assert_eq!(count_zero_crossings(&signal), 3);
    }

    #[test]
    fn test_count_zero_crossings_no_crossings() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(count_zero_crossings(&signal), 0);
    }

    #[test]
    fn test_count_zero_crossings_sine_wave() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();
        let crossings = count_zero_crossings(&signal);
        assert!(
            crossings >= 2,
            "Sine wave should have at least 2 zero crossings, got {}",
            crossings
        );
    }

    #[test]
    fn test_count_zero_crossings_empty() {
        let signal: Vec<f64> = vec![];
        assert_eq!(count_zero_crossings(&signal), 0);
    }

    #[test]
    fn test_count_zero_crossings_single_element() {
        let signal = vec![1.0];
        assert_eq!(count_zero_crossings(&signal), 0);
    }

    #[test]
    fn test_count_zero_crossings_with_zeros() {
        let signal = vec![1.0, 0.0, -1.0];
        assert_eq!(count_zero_crossings(&signal), 1);
    }

    #[test]
    fn test_count_zero_crossings_touching_zero() {
        let signal = vec![1.0, 0.0, 1.0];
        assert_eq!(count_zero_crossings(&signal), 0);
    }

    // =========================================================================
    // compute_energy tests
    // =========================================================================

    #[test]
    fn test_compute_energy_simple() {
        let signal = vec![1.0, 2.0, 3.0];
        let energy = compute_energy(&signal);
        assert!((energy - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_energy_zero_signal() {
        let signal = vec![0.0, 0.0, 0.0];
        let energy = compute_energy(&signal);
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn test_compute_energy_empty() {
        let signal: Vec<f64> = vec![];
        let energy = compute_energy(&signal);
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn test_compute_energy_negative_values() {
        let signal = vec![-1.0, -2.0, -3.0];
        let energy = compute_energy(&signal);
        assert!((energy - 14.0).abs() < 1e-10);
    }

    // =========================================================================
    // check_s_number tests — T-057
    // =========================================================================

    #[test]
    fn test_check_s_number_all_equal() {
        let extrema = vec![5, 5, 5, 5, 5];
        let zero_crossings = vec![5, 5, 5, 5, 5];
        assert!(check_s_number(&extrema, &zero_crossings, 5));
    }

    #[test]
    fn test_check_s_number_differ_by_one() {
        let extrema = vec![5, 6, 5, 6, 5];
        let zero_crossings = vec![5, 5, 6, 5, 6];
        assert!(check_s_number(&extrema, &zero_crossings, 5));
    }

    #[test]
    fn test_check_s_number_one_mismatch() {
        let extrema = vec![5, 5, 5, 5, 8];
        let zero_crossings = vec![5, 5, 5, 5, 5];
        assert!(check_s_number(&extrema, &zero_crossings, 5));
    }

    #[test]
    fn test_check_s_number_last_s_consecutive_pass() {
        let extrema = vec![10, 10, 5, 5, 5, 5, 5];
        let zero_crossings = vec![10, 10, 5, 5, 5, 6, 5];
        assert!(check_s_number(&extrema, &zero_crossings, 5));
    }

    #[test]
    fn test_check_s_number_not_enough_data() {
        let extrema = vec![5, 5];
        let zero_crossings = vec![5, 5];
        assert!(!check_s_number(&extrema, &zero_crossings, 5));
    }

    #[test]
    fn test_check_s_number_difference_too_large() {
        let extrema = vec![5, 5, 5, 5, 5];
        let zero_crossings = vec![5, 5, 5, 5, 10];
        assert!(!check_s_number(&extrema, &zero_crossings, 5));
    }

    #[test]
    fn test_check_s_number_s_one() {
        let extrema = vec![5];
        let zero_crossings = vec![5];
        assert!(check_s_number(&extrema, &zero_crossings, 1));
    }

    #[test]
    fn test_check_s_number_s_one_fail() {
        let extrema = vec![5];
        let zero_crossings = vec![10];
        assert!(!check_s_number(&extrema, &zero_crossings, 1));
    }

    // =========================================================================
    // compute_mean_envelope tests
    // =========================================================================

    #[test]
    fn test_compute_mean_envelope_simple() {
        let upper = vec![2.0, 4.0, 6.0];
        let lower = vec![0.0, 0.0, 0.0];
        let mean = compute_mean_envelope(&upper, &lower);
        assert_eq!(mean, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_compute_mean_envelope_symmetric() {
        let upper = vec![1.0, 2.0, 3.0];
        let lower = vec![-1.0, -2.0, -3.0];
        let mean = compute_mean_envelope(&upper, &lower);
        assert_eq!(mean, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn test_compute_mean_envelope_length_mismatch() {
        let upper = vec![1.0, 2.0];
        let lower = vec![1.0, 2.0, 3.0];
        compute_mean_envelope(&upper, &lower);
    }

    // =========================================================================
    // extract_envelopes tests
    // =========================================================================

    #[test]
    fn test_extract_envelopes_insufficient_extrema() {
        let signal = vec![1.0, 2.0, 3.0];
        let extrema = ExtremaStruct { maxima: vec![2], minima: vec![0] };
        let strategy = get_strategy(BoundaryConditionType::MirrorEven);
        let result = extract_envelopes(&signal, &extrema, strategy.as_ref());
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_envelopes_sine_wave() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();
        let extrema = detect_extrema(&signal);
        let strategy = get_strategy(BoundaryConditionType::MirrorEven);

        assert!(extrema.maxima.len() >= 2, "Sine wave should have multiple maxima");
        assert!(extrema.minima.len() >= 2, "Sine wave should have multiple minima");

        let result = extract_envelopes(&signal, &extrema, strategy.as_ref());
        assert!(result.is_ok());

        let (upper, lower) = result.unwrap();
        assert_eq!(upper.len(), signal.len());
        assert_eq!(lower.len(), signal.len());

        for i in 0..signal.len() {
            assert!(
                upper[i] >= signal[i] - 1e-6,
                "upper envelope should be >= signal at index {}",
                i
            );
            assert!(
                lower[i] <= signal[i] + 1e-6,
                "lower envelope should be <= signal at index {}",
                i
            );
        }
    }

    // =========================================================================
    // SiftingEngine tests — T-055
    // =========================================================================

    #[test]
    fn test_sifting_engine_creation() {
        let config = SiftingConfig::default();
        let criteria = vec![StoppingCriterion::SdThreshold];
        let engine = SiftingEngine::new(config, criteria);
        assert_eq!(engine.stopping_criteria().len(), 1);
    }

    #[test]
    fn test_sifting_engine_config_accessor() {
        let config = SiftingConfig::default();
        let criteria = vec![StoppingCriterion::SdThreshold];
        let engine = SiftingEngine::new(config.clone(), criteria);
        assert_eq!(engine.config().max_sifting_iterations, config.max_sifting_iterations);
    }

    #[test]
    fn test_sift_one_insufficient_data() {
        let config = SiftingConfig::default();
        let engine = SiftingEngine::new(config, vec![StoppingCriterion::SdThreshold]);
        let result = engine.sift_one(&[1.0, 2.0]);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_sift_one_invalid_value() {
        let config = SiftingConfig::default();
        let engine = SiftingEngine::new(config, vec![StoppingCriterion::SdThreshold]);
        let result = engine.sift_one(&[1.0, f64::NAN, 3.0]);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_sift_one_constant_signal_returns_as_imf() {
        let config = SiftingConfig {
            max_sifting_iterations: 50,
            sd_threshold: 0.01,
            s_number: 3,
            fixed_iterations: None,
            energy_threshold: 1e-8,
            boundary_condition: BoundaryConditionType::MirrorEven,
        };
        let engine = SiftingEngine::new(config, vec![StoppingCriterion::FixedIterations]);
        let signal = vec![5.0; 20];
        let result = engine.sift_one(&signal);
        assert!(result.is_ok());
        let (imf, residue) = result.unwrap();
        assert_eq!(imf.len(), signal.len());
        assert_eq!(residue.len(), signal.len());
    }

    #[test]
    fn test_sift_one_sine_wave_convergence() {
        let config = SiftingConfig {
            max_sifting_iterations: 100,
            sd_threshold: 0.2,
            s_number: 5,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: BoundaryConditionType::MirrorEven,
        };
        let engine = SiftingEngine::new(
            config,
            vec![
                StoppingCriterion::SdThreshold,
                StoppingCriterion::SNumber,
                StoppingCriterion::EnergyDifference,
            ],
        );

        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let result = engine.sift_one(&signal);
        assert!(result.is_ok(), "sift_one should succeed for sine wave");

        let (imf, residue) = result.unwrap();
        assert_eq!(imf.len(), signal.len());
        assert_eq!(residue.len(), signal.len());

        let imf_energy = compute_energy(&imf);
        let residue_energy = compute_energy(&residue);
        assert!(imf_energy > 0.0, "IMF should have non-zero energy");
    }

    #[test]
    fn test_sift_one_fixed_iterations_stopping() {
        let fixed = 5;
        let config = SiftingConfig {
            max_sifting_iterations: 100,
            sd_threshold: 0.0001,
            s_number: 50,
            fixed_iterations: Some(fixed),
            energy_threshold: 1e-15,
            boundary_condition: BoundaryConditionType::MirrorEven,
        };
        let engine = SiftingEngine::new(config, vec![StoppingCriterion::FixedIterations]);

        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let result = engine.sift_one(&signal);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sift_one_reconstruction_property() {
        let config = SiftingConfig {
            max_sifting_iterations: 100,
            sd_threshold: 0.1,
            s_number: 3,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: BoundaryConditionType::MirrorEven,
        };
        let engine = SiftingEngine::new(
            config,
            vec![
                StoppingCriterion::SdThreshold,
                StoppingCriterion::SNumber,
                StoppingCriterion::EnergyDifference,
            ],
        );

        let n = 80;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * i as f64 / n as f64).sin()
                    + 0.5 * (4.0 * PI * i as f64 / n as f64).sin()
            })
            .collect();

        let result = engine.sift_one(&signal);
        assert!(result.is_ok());

        let (imf, residue) = result.unwrap();

        let reconstructed: Vec<f64> =
            imf.iter().zip(residue.iter()).map(|(&imf_val, &res_val)| imf_val + res_val).collect();

        for (i, (orig, recon)) in signal.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-6,
                "reconstruction mismatch at index {}: expected {}, got {}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_sift_one_convergence_failure() {
        let config = SiftingConfig {
            max_sifting_iterations: 2,
            sd_threshold: 1e-15,
            s_number: 100,
            fixed_iterations: None,
            energy_threshold: 1e-15,
            boundary_condition: BoundaryConditionType::MirrorEven,
        };
        let engine = SiftingEngine::new(
            config,
            vec![
                StoppingCriterion::SdThreshold,
                StoppingCriterion::SNumber,
                StoppingCriterion::EnergyDifference,
            ],
        );

        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let result = engine.sift_one(&signal);
        assert!(matches!(result.unwrap_err(), EmdError::ConvergenceFailed { .. }));
    }

    // =========================================================================
    // Integration tests — IMF properties
    // =========================================================================

    #[test]
    fn test_imf_property_zero_mean_approximate() {
        let config = SiftingConfig {
            max_sifting_iterations: 100,
            sd_threshold: 0.05,
            s_number: 5,
            fixed_iterations: None,
            energy_threshold: 1e-8,
            boundary_condition: BoundaryConditionType::MirrorEven,
        };
        let engine = SiftingEngine::new(
            config,
            vec![StoppingCriterion::SdThreshold, StoppingCriterion::SNumber],
        );

        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let (imf, _residue) = engine.sift_one(&signal).unwrap();

        let mean: f64 = imf.iter().sum::<f64>() / imf.len() as f64;
        assert!(mean.abs() < 0.1, "IMF should have approximately zero mean, got {}", mean);
    }

    #[test]
    fn test_imf_property_extrema_zero_crossing_balance() {
        let config = SiftingConfig {
            max_sifting_iterations: 100,
            sd_threshold: 0.1,
            s_number: 5,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: BoundaryConditionType::MirrorEven,
        };
        let engine = SiftingEngine::new(
            config,
            vec![StoppingCriterion::SdThreshold, StoppingCriterion::SNumber],
        );

        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let (imf, _residue) = engine.sift_one(&signal).unwrap();

        let extrema = detect_extrema(&imf);
        let num_extrema = extrema.maxima.len() + extrema.minima.len();
        let num_zero_crossings = count_zero_crossings(&imf);

        let diff = (num_extrema as isize - num_zero_crossings as isize).abs();
        assert!(
            diff <= 1,
            "IMF should have extrema and zero crossings differing by at most 1, got diff={}",
            diff
        );
    }

    #[test]
    fn test_imf_property_energy_conservation() {
        let config = SiftingConfig {
            max_sifting_iterations: 100,
            sd_threshold: 0.1,
            s_number: 5,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: BoundaryConditionType::MirrorEven,
        };
        let engine = SiftingEngine::new(
            config,
            vec![StoppingCriterion::SdThreshold, StoppingCriterion::EnergyDifference],
        );

        let n = 80;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * i as f64 / n as f64).sin()
                    + 0.3 * (6.0 * PI * i as f64 / n as f64).sin()
            })
            .collect();

        let original_energy = compute_energy(&signal);
        let (imf, residue) = engine.sift_one(&signal).unwrap();

        let imf_energy = compute_energy(&imf);
        let residue_energy = compute_energy(&residue);
        let total_energy = imf_energy + residue_energy;

        assert!(
            (total_energy - original_energy).abs() / original_energy < 0.1,
            "energy should be approximately conserved: original={}, imf+residue={}",
            original_energy,
            total_energy
        );
    }

    #[test]
    fn test_sift_one_multi_frequency_signal() {
        let config = SiftingConfig {
            max_sifting_iterations: 100,
            sd_threshold: 0.1,
            s_number: 5,
            fixed_iterations: None,
            energy_threshold: 1e-6,
            boundary_condition: BoundaryConditionType::MirrorEven,
        };
        let engine = SiftingEngine::new(
            config,
            vec![
                StoppingCriterion::SdThreshold,
                StoppingCriterion::SNumber,
                StoppingCriterion::EnergyDifference,
            ],
        );

        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let result = engine.sift_one(&signal);
        assert!(result.is_ok());

        let (imf, residue) = result.unwrap();
        assert_eq!(imf.len(), signal.len());
        assert_eq!(residue.len(), signal.len());

        let imf_energy = compute_energy(&imf);
        assert!(imf_energy > 0.0);
    }

    #[test]
    fn test_convenience_sift_one_function() {
        let config = SiftingConfig::default();
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let result = sift_one(&signal, &config);
        assert!(result.is_ok());

        let (imf, residue) = result.unwrap();
        assert_eq!(imf.len(), signal.len());
        assert_eq!(residue.len(), signal.len());
    }
}
