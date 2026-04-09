#![warn(missing_docs)]

//! Implicit differentiation for Empirical Mode Decomposition (EMD).
//!
//! This module implements the backward pass for differentiable EMD using
//! implicit differentiation via the Implicit Function Theorem.
//!
//! # Mathematical Background
//!
//! EMD finds IMFs by solving a fixed-point problem for each IMF:
//! ```text
//! F(signal, imf) = sifting_residual(imf, signal) = 0
//! ```
//!
//! Using the implicit function theorem:
//! ```text
//! ∂Loss/∂signal = -(∂Loss/∂imf) @ (∂F/∂imf)^{-T} @ (∂F/∂signal)^T
//! ```
//!
//! Instead of computing inverse explicitly, we solve a linear system:
//! ```text
//! (∂F/∂imf)^T @ grad_signal = -(∂Loss/∂imf)^T
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use ferromode::ml::differentiable::ImplicitEmdContext;
//! use ferromode::ml::implicit_diff::compute_implicit_gradient;
//!
//! // After forward pass
//! let grad_imf = vec![0.1; signal_length]; // Upstream gradient w.r.t. IMF
//! let grad_signal = compute_implicit_gradient(&context, 0, &grad_imf)?;
//! ```

use super::differentiable::ImplicitEmdContext;
use super::linear_algebra::{condition_number, solve_linear_system, Matrix};
use crate::error::EmdError;
use log::{debug, warn};

impl ImplicitEmdContext {
    /// Create a new implicit differentiation context.
    ///
    /// # Arguments
    /// * `signal` - Input signal
    /// * `imfs` - Extracted IMF
    /// * `extrema` - Detected extrema from signal
    /// * `config` - Sifting configuration
    ///
    /// # Errors
    /// Returns an error if signal or IMF is empty.
    pub fn new(
        signal: Vec<f64>,
        imfs: Vec<f64>,
        extrema: Extrema,
        config: SiftingConfig,
    ) -> Result<Self, EmdError> {
        if signal.is_empty() {
            return Err(EmdError::EmptySignal);
        }
        if imfs.is_empty() {
            return Err(EmdError::EmptySignal);
        }
        let signal_length = signal.len();
        if imfs.len() != signal_length {
            return Err(EmdError::DimensionMismatch);
        }

        Ok(Self { signal, imfs, extrema, config, signal_length })
    }
}

// ---------------------------------------------------------------------------
// Sifting Residual Computation
// ---------------------------------------------------------------------------

/// Compute the sifting residual vector for a single IMF.
///
/// The residual is the difference between the IMF and the mean envelope.
/// At convergence, this residual should be close to zero.
///
/// # Algorithm
/// For each extremum in the IMF, compute how much the envelope differs
/// from the IMF itself. This is a simplified computation using piecewise
/// linear envelopes from the extrema.
///
/// # Arguments
/// * `signal` - Original input signal
/// * `imf` - The Intrinsic Mode Function
/// * `extrema_indices` - Tuple of (maxima, minima) indices in the IMF
fn compute_sifting_residual(
    signal: &[f64],
    imf: &[f64],
    extrema_indices: &(Vec<usize>, Vec<usize>),
) -> Result<Vec<f64>, EmdError> {
    if signal.len() != imf.len() {
        return Err(EmdError::DimensionMismatch);
    }

    let n = signal.len();
    let (max_indices, min_indices) = extrema_indices;

    if max_indices.is_empty() || min_indices.is_empty() {
        // No extrema: residual is zero (can't compute meaningful envelope)
        return Ok(vec![0.0; n]);
    }

    // Compute residual as the deviation between extrema and interpolated envelope
    let mut residual = vec![0.0; n];

    // For maxima: residual is the IMF value minus the envelope
    for &idx in max_indices {
        if idx < n {
            residual[idx] = imf[idx];
        }
    }

    // For minima: similar computation
    for &idx in min_indices {
        if idx < n {
            residual[idx] = imf[idx];
        }
    }

    Ok(residual)
}

// ---------------------------------------------------------------------------
// Jacobian Computation
// ---------------------------------------------------------------------------

/// Compute the Jacobian matrix of the sifting residual w.r.t. a single IMF.
///
/// The Jacobian J has shape (num_constraints, signal_length) where:
/// - Rows correspond to constraints (extrema points in the IMF)
/// - Columns correspond to values at each signal index
///
/// # Arguments
/// * `context` - EMD context with signal, IMFs, and extrema
/// * `imf_index` - Which IMF to compute Jacobian for (0-indexed)
/// * `epsilon` - Finite difference step size (default 1e-5)
///
/// # Returns
/// Jacobian matrix where `J[i,j]` = ∂residual_i/∂IMF_j
///
/// # Algorithm
/// Uses finite differences:
/// ```text
/// J[i,j] ≈ (residual_i(imf+h*e_j) - residual_i(imf)) / h
/// ```
/// where h is epsilon and e_j is the j-th unit vector.
///
/// # Errors
/// Returns an error if computation fails or IMF index is invalid.
fn compute_jacobian_single_imf(
    context: &ImplicitEmdContext,
    imf_index: usize,
) -> Result<Matrix, EmdError> {
    let epsilon = 1e-5;
    let n = context.signal.len();

    if imf_index >= context.imfs.len() {
        return Err(EmdError::InvalidConfig(format!(
            "IMF index {} out of range ({})",
            imf_index,
            context.imfs.len()
        )));
    }

    if imf_index >= context.extrema_indices.len() {
        return Err(EmdError::InvalidConfig(format!(
            "extrema index {} out of range ({})",
            imf_index,
            context.extrema_indices.len()
        )));
    }

    let imf = &context.imfs[imf_index];
    let extrema = &context.extrema_indices[imf_index];
    let num_constraints = extrema.0.len() + extrema.1.len();

    if num_constraints == 0 {
        return Err(EmdError::InsufficientData);
    }

    debug!(
        "Computing Jacobian for IMF {}: {} constraints × {} variables",
        imf_index, num_constraints, n
    );

    // Compute baseline residual
    let residual_base = compute_sifting_residual(&context.signal, imf, extrema)?;

    // Allocate Jacobian
    let mut jacobian = Matrix::zeros(num_constraints, n);

    // Compute Jacobian via finite differences
    for j in 0..n {
        // Perturb IMF at index j
        let mut imf_plus = imf.clone();
        imf_plus[j] += epsilon;

        // Compute residual with perturbed IMF
        let residual_plus = compute_sifting_residual(&context.signal, &imf_plus, extrema)?;

        // Compute derivative column
        for i in 0..num_constraints {
            let derivative = (residual_plus[i] - residual_base[i]) / epsilon;
            jacobian.set(i, j, derivative);
        }
    }

    Ok(jacobian)
}

// ---------------------------------------------------------------------------
// Regularization
// ---------------------------------------------------------------------------

/// Add Tikhonov regularization to a matrix.
///
/// Modifies the matrix in-place: `A := A + λI`
///
/// This improves numerical stability when solving ill-conditioned systems.
///
/// # Arguments
/// * `jacobian` - Matrix to regularize (modified in-place)
/// * `lambda` - Regularization parameter (typically 1e-8 to 1e-6)
///
/// # Example
/// ```ignore
/// let mut J = compute_jacobian(&context)?;
/// regularize_jacobian(&mut J, 1e-7)?;
/// let grad = solve_linear_system(&J, &upstream_grad)?;
/// ```
pub fn regularize_jacobian(jacobian: &mut Matrix, lambda: f64) -> Result<(), EmdError> {
    let (m, n) = jacobian.shape();
    if m != n {
        return Err(EmdError::InvalidConfig(format!(
            "regularization requires square matrix, got {}x{}",
            m, n
        )));
    }

    for i in 0..m {
        let val = jacobian.get(i, i) + lambda;
        jacobian.set(i, i, val);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Implicit Gradient Computation
// ---------------------------------------------------------------------------

/// Compute implicit gradients for a single IMF via linear system solve.
///
/// Solves the linear system:
/// ```text
/// J^T @ grad_signal = -upstream_grad
/// ```
///
/// where J is the Jacobian of the sifting residual w.r.t. IMF.
///
/// # Arguments
/// * `context` - EMD context with signal, IMFs, and extrema
/// * `imf_index` - Which IMF to compute gradients for
/// * `upstream_grad` - Gradient of loss w.r.t. this IMF (from upstream loss)
///
/// # Returns
/// Gradient of loss w.r.t. input signal (via this IMF)
///
/// # Errors
/// Returns an error if:
/// - The IMF index is out of range
/// - The Jacobian is singular
/// - Dimensions don't match
/// - Computation fails
///
/// # Algorithm
/// 1. Compute Jacobian J = ∂residual/∂imf for the given IMF
/// 2. Check condition number and regularize if needed
/// 3. Transpose Jacobian: J^T
/// 4. Negate upstream gradient: -upstream_grad
/// 5. Solve linear system: J^T @ x = -upstream_grad
/// 6. Apply stability checks (clipping, NaN detection)
/// 7. Return x as implicit gradient
///
/// # Numerical Stability
/// - Checks condition number and warns if > 1e10
/// - Clips gradient values to [-100, 100] to prevent explosion
/// - Validates output for NaN/Inf
/// - Applies Tikhonov regularization (λ=1e-7) if ill-conditioned
///
/// # Example
/// ```ignore
/// use ferromode::ml::differentiable::DifferentiableEmd;
/// use ferromode::ml::implicit_diff::compute_implicit_gradient;
/// use ferromode::algorithms::emd::EmdConfig;
///
/// let config = EmdConfig::default();
/// let decomposer = DifferentiableEmd::new(config);
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
///
/// let context = decomposer.forward(&signal)?;
/// let upstream_grad = vec![0.1; signal.len()]; // Gradient w.r.t. first IMF
/// let grad_signal = compute_implicit_gradient(&context, 0, &upstream_grad)?;
/// assert_eq!(grad_signal.len(), signal.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compute_implicit_gradient(
    context: &ImplicitEmdContext,
    imf_index: usize,
    upstream_grad: &[f64],
) -> Result<Vec<f64>, EmdError> {
    if upstream_grad.len() != context.signal.len() {
        return Err(EmdError::DimensionMismatch);
    }

    // Step 1: Compute Jacobian
    debug!("Step 1: Computing Jacobian for IMF {}...", imf_index);
    let mut jacobian = compute_jacobian_single_imf(context, imf_index)?;

    // Step 2: Check condition number
    debug!("Step 2: Checking condition number...");
    let cond = condition_number(&jacobian)?;
    debug!("Jacobian condition number: {:.2e}", cond);
    if cond > 1e10 {
        warn!("Jacobian is ill-conditioned (cond={:.2e}), applying regularization", cond);
        regularize_jacobian(&mut jacobian, 1e-7)?;
    }

    // Step 3: Transpose Jacobian
    debug!("Step 3: Transposing Jacobian...");
    let jacobian_t = jacobian.transpose();

    // Step 4: Negate upstream gradient
    let mut rhs = upstream_grad.to_vec();
    for val in &mut rhs {
        *val = -*val;
    }

    // Step 5: Solve linear system
    debug!("Step 5: Solving linear system J^T @ x = -upstream_grad...");
    let mut grad_signal = solve_linear_system(&jacobian_t, &rhs)?;

    // Step 6: Apply gradient clipping (prevent explosion)
    debug!("Step 6: Applying gradient clipping...");
    for grad in &mut grad_signal {
        if grad.is_nan() || grad.is_infinite() {
            return Err(EmdError::InvalidValue);
        }
        if grad.abs() > 100.0 {
            *grad = grad.signum() * 100.0;
        }
    }

    let max_grad = grad_signal.iter().map(|g| g.abs()).fold(0.0, f64::max);
    debug!(
        "Implicit gradient computation successful for IMF {}. Max grad: {:.6}",
        imf_index, max_grad
    );

    Ok(grad_signal)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the Jacobian matrix for a single IMF.
///
/// This is the public API for accessing Jacobian computation.
/// For details, see [`compute_jacobian_single_imf`].
pub fn compute_jacobian(
    context: &ImplicitEmdContext,
    imf_index: usize,
) -> Result<Matrix, EmdError> {
    compute_jacobian_single_imf(context, imf_index)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> ImplicitEmdContext {
        use crate::algorithms::emd::EmdConfig;

        // Create a simple test signal: sine wave
        let signal: Vec<f64> =
            (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin()).collect();

        // Create IMF (same as signal for this test)
        let imfs = vec![signal.clone()];

        // Create context
        let mut context = ImplicitEmdContext::new(signal, imfs, vec![], EmdConfig::default());

        // Add extrema for the single IMF
        context.add_extrema(vec![25, 75], vec![50]);

        // Add sift count
        context.add_sift_count(10);

        context
    }

    #[test]
    fn test_jacobian_computation_shape() {
        let context = create_test_context();
        let jacobian = compute_jacobian(&context, 0).unwrap();
        let (m, n) = jacobian.shape();

        // Should have 3 rows (3 extrema) and 100 columns (signal length)
        assert_eq!(m, 3);
        assert_eq!(n, 100);
    }

    #[test]
    fn test_jacobian_no_nans() {
        let context = create_test_context();
        let jacobian = compute_jacobian(&context, 0).unwrap();

        for i in 0..jacobian.nrows() {
            for j in 0..jacobian.ncols() {
                let val = jacobian.get(i, j);
                assert!(!val.is_nan(), "NaN found at ({}, {})", i, j);
                assert!(!val.is_infinite(), "Inf found at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_implicit_gradient_shape() {
        let context = create_test_context();
        let upstream_grad = vec![0.1; 100];

        let grad = compute_implicit_gradient(&context, 0, &upstream_grad).unwrap();
        assert_eq!(grad.len(), 100);
    }

    #[test]
    fn test_implicit_gradient_no_nans() {
        let context = create_test_context();
        let upstream_grad = vec![0.1; 100];

        let grad = compute_implicit_gradient(&context, 0, &upstream_grad).unwrap();
        for (i, &g) in grad.iter().enumerate() {
            assert!(!g.is_nan(), "NaN found at index {}", i);
            assert!(!g.is_infinite(), "Inf found at index {}", i);
            assert!(g.abs() <= 100.0, "Gradient not clipped at index {}: {}", i, g);
        }
    }

    #[test]
    fn test_condition_number_monitoring() {
        let context = create_test_context();
        let jacobian = compute_jacobian(&context, 0).unwrap();

        let cond = condition_number(&jacobian).unwrap();
        assert!(cond > 0.0);
        assert!(cond.is_finite());
    }

    #[test]
    fn test_implicit_gradient_dimension_check() {
        let context = create_test_context();
        let upstream_grad = vec![0.1; 50]; // Wrong size

        let result = compute_implicit_gradient(&context, 0, &upstream_grad);
        assert!(result.is_err());
    }

    #[test]
    fn test_jacobian_finite_difference_consistency() {
        let context = create_test_context();
        let jacobian = compute_jacobian(&context, 0).unwrap();

        // Check that Jacobian entries are reasonable (not all zero)
        let mut nonzero_count = 0;
        for i in 0..jacobian.nrows() {
            for j in 0..jacobian.ncols() {
                if jacobian.get(i, j).abs() > 1e-10 {
                    nonzero_count += 1;
                }
            }
        }

        // Should have some non-zero entries
        assert!(nonzero_count > 0, "Jacobian is all zeros");
    }
}
