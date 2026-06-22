/// Python bindings for differentiable EMD.
///
/// Exposes Rust differentiable EMD functions to Python via PyO3.
/// Handles marshalling of configuration dicts and numpy arrays.
use ferromode::algorithms::emd::EmdConfig;
use ferromode::ml::differentiable::{DifferentiableEmd, ImplicitEmdContext};
use pyo3::prelude::*;

/// Python-friendly result type for the EMD forward pass.
///
/// Retains the full differentiation context so [`EmdForwardResult::backward`]
/// can compute the exact gradient using the same configuration and trace.
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

    /// Saved forward context (signal, trace, config) for the backward pass.
    ctx: ImplicitEmdContext,
}

#[pymethods]
impl EmdForwardResult {
    /// Reconstruction error of the forward decomposition.
    pub fn reconstruction_error(&self) -> f64 {
        self.ctx.reconstruction_error()
    }

    /// Backward pass (exact implicit differentiation).
    ///
    /// Given the upstream gradient w.r.t. each IMF (`grad_imfs`, shape
    /// num_imfs × signal_length), returns the gradient w.r.t. the input signal.
    /// Uses the saved forward context, so it is correct for any configuration.
    ///
    /// # Raises
    /// - ValueError if `grad_imfs` has the wrong shape
    /// - RuntimeError if the gradient computation fails
    pub fn backward(&self, grad_imfs: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let signal_len = self.ctx.signal.len();
        if grad_imfs.len() != self.imfs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "expected {} gradient IMFs, got {}",
                self.imfs.len(),
                grad_imfs.len()
            )));
        }
        for (i, g) in grad_imfs.iter().enumerate() {
            if g.len() != signal_len {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "grad_imfs[{}] has length {}, expected {}",
                    i,
                    g.len(),
                    signal_len
                )));
            }
        }
        self.ctx.backward(&grad_imfs).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "EMD backward failed: {:?}",
                e
            ))
        })
    }
}

/// Call EMD forward pass from Python.
///
/// Decomposes a signal into IMFs and residue, saving context for the implicit
/// differentiation backward pass ([`EmdForwardResult::backward`]).
///
/// # Arguments
/// - signal: Signal as a Python list of f64
/// - max_imfs: Maximum number of IMFs to extract (optional, default 0=no limit)
///
/// # Returns
/// EmdForwardResult with IMFs, residue, extrema, metadata, and saved context
///
/// # Raises
/// - ValueError if signal is empty or contains NaN/Inf
/// - RuntimeError if EMD decomposition fails
#[pyfunction]
#[pyo3(signature = (signal, max_imfs = 0))]
pub fn emd_forward(signal: Vec<f64>, max_imfs: usize) -> PyResult<EmdForwardResult> {
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

    let mut emd_config = EmdConfig::default();
    if max_imfs > 0 {
        emd_config.max_imfs = max_imfs;
    }

    let decomposer = DifferentiableEmd::new(emd_config.clone());
    let context = decomposer.forward(&signal).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("EMD forward failed: {:?}", e))
    })?;

    let extrema = context
        .extrema_indices
        .iter()
        .map(|(max_idx, min_idx)| (max_idx.clone(), min_idx.clone()))
        .collect();
    let config_metadata = format!("max_imfs={}", emd_config.max_imfs);

    Ok(EmdForwardResult {
        imfs: context.imfs.clone(),
        residue: context.residue.clone(),
        extrema,
        num_sifts: context.num_sifts.clone(),
        config_metadata,
        ctx: context,
    })
}
