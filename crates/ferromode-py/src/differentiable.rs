/// Python bindings for differentiable EMD.
///
/// Exposes Rust differentiable EMD functions to Python via PyO3.
/// Handles marshalling of configuration dicts and numpy arrays.
use ferromode::algorithms::emd::EmdConfig;
use ferromode::error::EmdError;
use ferromode::ml::differentiable::{DifferentiableEmd, ImplicitEmdContext};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Python-friendly result type for EMD forward pass.
#[pyclass]
pub struct EmdForwardResult {
    /// List of IMFs (each as a list of f64)
    #[pyo3(get)]
    pub imfs: Vec<Vec<f64>>,

    /// Residue vector
    #[pyo3(get)]
    pub residue: Vec<f64>,

    /// Extrema indices: list of (maxima, minima) tuples
    #[pyo3(get)]
    pub extrema: Vec<(Vec<usize>, Vec<usize>)>,

    /// Number of sifting iterations per IMF
    #[pyo3(get)]
    pub num_sifts: Vec<usize>,

    /// Configuration used
    #[pyo3(get)]
    pub config_metadata: String,
}

#[pymethods]
impl EmdForwardResult {
    /// Create a new forward result.
    #[new]
    pub fn new(
        imfs: Vec<Vec<f64>>,
        residue: Vec<f64>,
        extrema: Vec<(Vec<usize>, Vec<usize>)>,
        num_sifts: Vec<usize>,
        config_metadata: String,
    ) -> Self {
        Self { imfs, residue, extrema, num_sifts, config_metadata }
    }
}

/// Call EMD forward pass from Python.
///
/// Decomposes a signal into IMFs and residue, saving context for
/// the implicit differentiation backward pass.
///
/// # Arguments
/// - signal: Signal as a Python list of f64
/// - max_imfs: Maximum number of IMFs to extract (optional, default 0=no limit)
///
/// # Returns
/// EmdForwardResult with IMFs, residue, extrema, and metadata
///
/// # Raises
/// - ValueError if signal is empty or contains NaN/Inf
/// - RuntimeError if EMD decomposition fails
#[pyfunction]
#[pyo3(signature = (signal, max_imfs = 0))]
pub fn emd_forward(signal: Vec<f64>, max_imfs: usize) -> PyResult<EmdForwardResult> {
    // Validate input
    if signal.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Signal cannot be empty"));
    }

    for &val in &signal {
        if !val.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Signal contains NaN or Inf",
            ));
        }
    }

    // Create configuration
    let mut emd_config = EmdConfig::default();
    if max_imfs > 0 {
        emd_config.max_imfs = max_imfs;
    }

    // Call Rust forward pass
    let decomposer = DifferentiableEmd::new(emd_config.clone());
    let context = decomposer.forward(&signal).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("EMD forward failed: {:?}", e))
    })?;

    // Extract results
    let extrema = context
        .extrema_indices
        .iter()
        .map(|(max_idx, min_idx)| (max_idx.clone(), min_idx.clone()))
        .collect();

    let config_metadata = format!("max_imfs={}", emd_config.max_imfs);

    Ok(EmdForwardResult::new(
        context.imfs,
        context.residue,
        extrema,
        context.num_sifts,
        config_metadata,
    ))
}

/// Call EMD backward pass from Python (implicit differentiation).
///
/// Computes gradients w.r.t. input signal given upstream gradients
/// w.r.t. IMFs. Currently a placeholder that averages upstream gradients.
/// Full implicit differentiation will be implemented in T-321.
///
/// # Arguments
/// - grad_imfs: Upstream gradients as list of lists (num_imfs × signal_length)
/// - signal: Original input signal
/// - max_imfs: Number of max IMFs (unused, for API compatibility)
///
/// # Returns
/// List of gradients w.r.t. input signal (length = signal_length)
///
/// # Raises
/// - ValueError if shapes don't match
/// - RuntimeError if computation fails
#[pyfunction]
#[pyo3(signature = (grad_imfs, signal))]
pub fn emd_backward(grad_imfs: Vec<Vec<f64>>, signal: Vec<f64>) -> PyResult<Vec<f64>> {
    // Validate shapes
    if signal.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Signal cannot be empty"));
    }

    if grad_imfs.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("grad_imfs cannot be empty"));
    }

    let signal_len = signal.len();
    for (i, grad_imf) in grad_imfs.iter().enumerate() {
        if grad_imf.len() != signal_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "grad_imfs[{}] has length {}, expected {}",
                i,
                grad_imf.len(),
                signal_len
            )));
        }
    }

    // Placeholder implementation: average upstream gradients
    // Full implicit differentiation with Jacobian computation will be in T-321
    let mut grad_signal = vec![0.0; signal_len];

    for grad_imf in &grad_imfs {
        for (i, &grad) in grad_imf.iter().enumerate() {
            grad_signal[i] += grad;
        }
    }

    let num_imfs = grad_imfs.len() as f64;
    for grad in &mut grad_signal {
        *grad /= num_imfs;
    }

    // Validate output
    for &val in &grad_signal {
        if !val.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Backward pass produced NaN or Inf",
            ));
        }
    }

    Ok(grad_signal)
}
