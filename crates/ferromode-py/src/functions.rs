//! Python function wrappers for all 8 Ferromode algorithms.
//!
//! Pure-wrap contract: Python layer is ONLY marshalling, no algorithm logic.
//! All defaults come from Rust `impl Default`, not computed in Python.
//! GIL is released during ensemble methods.
//! Error messages are forwarded verbatim from Rust EmdError.

use ferromode::algorithms::ceemd::ceemd as ferromode_ceemd;
use ferromode::algorithms::ceemdan::ceemdan as ferromode_ceemdan;
use ferromode::algorithms::eemd::eemd as ferromode_eemd;
use ferromode::algorithms::emd::emd as ferromode_emd;
use ferromode::algorithms::iceemdan::iceemdan as ferromode_iceemdan;
use ferromode::algorithms::vmd::vmd as ferromode_vmd;
use ferromode::multivariate::memd::memd as ferromode_memd;
use ferromode::multivariate::namemd::namemd as ferromode_namemd;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::config::{EmdConfigPy, EnsembleConfigPy, MemdConfigPy, NaMemdConfigPy, VmdConfigPy};
use crate::error::emd_error_to_pyerr;
use crate::types::DecompositionResultPy;

// ---------------------------------------------------------------------------
// EMD
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (signal, config=None))]
pub fn emd(
    _py: Python,
    signal: PyReadonlyArray1<f64>,
    config: Option<&EmdConfigPy>,
) -> PyResult<DecompositionResultPy> {
    let signal = signal.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signal must be contiguous: {}", e))
    })?;
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = ferromode_emd(signal, &config).map_err(emd_error_to_pyerr)?;
    Ok(DecompositionResultPy::from_rust(result))
}

// ---------------------------------------------------------------------------
// EEMD
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (signal, ensemble_config=None, emd_config=None))]
pub fn eemd(
    py: Python,
    signal: PyReadonlyArray1<f64>,
    ensemble_config: Option<&EnsembleConfigPy>,
    emd_config: Option<&EmdConfigPy>,
) -> PyResult<DecompositionResultPy> {
    let signal = signal.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signal must be contiguous: {}", e))
    })?;
    let ensemble_config = ensemble_config.map(|c| c.inner.clone()).unwrap_or_default();
    let emd_config = emd_config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = py
        .allow_threads(|| ferromode_eemd(signal, &ensemble_config, &emd_config))
        .map_err(emd_error_to_pyerr)?;
    Ok(DecompositionResultPy::from_rust(result))
}

// ---------------------------------------------------------------------------
// CEEMD
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (signal, ensemble_config=None, emd_config=None))]
pub fn ceemd(
    py: Python,
    signal: PyReadonlyArray1<f64>,
    ensemble_config: Option<&EnsembleConfigPy>,
    emd_config: Option<&EmdConfigPy>,
) -> PyResult<DecompositionResultPy> {
    let signal = signal.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signal must be contiguous: {}", e))
    })?;
    let ensemble_config = ensemble_config.map(|c| c.inner.clone()).unwrap_or_default();
    let emd_config = emd_config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = py
        .allow_threads(|| ferromode_ceemd(signal, &ensemble_config, &emd_config))
        .map_err(emd_error_to_pyerr)?;
    Ok(DecompositionResultPy::from_rust(result))
}

// ---------------------------------------------------------------------------
// CEEMDAN
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (signal, ensemble_config=None, emd_config=None))]
pub fn ceemdan(
    py: Python,
    signal: PyReadonlyArray1<f64>,
    ensemble_config: Option<&EnsembleConfigPy>,
    emd_config: Option<&EmdConfigPy>,
) -> PyResult<DecompositionResultPy> {
    let signal = signal.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signal must be contiguous: {}", e))
    })?;
    let ensemble_config = ensemble_config.map(|c| c.inner.clone()).unwrap_or_default();
    let emd_config = emd_config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = py
        .allow_threads(|| ferromode_ceemdan(signal, &ensemble_config, &emd_config))
        .map_err(emd_error_to_pyerr)?;
    Ok(DecompositionResultPy::from_rust(result))
}

// ---------------------------------------------------------------------------
// ICEEMDAN
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (signal, ensemble_config=None, emd_config=None))]
pub fn iceemdan(
    py: Python,
    signal: PyReadonlyArray1<f64>,
    ensemble_config: Option<&EnsembleConfigPy>,
    emd_config: Option<&EmdConfigPy>,
) -> PyResult<DecompositionResultPy> {
    let signal = signal.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signal must be contiguous: {}", e))
    })?;
    let ensemble_config = ensemble_config.map(|c| c.inner.clone()).unwrap_or_default();
    let emd_config = emd_config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = py
        .allow_threads(|| ferromode_iceemdan(signal, &ensemble_config, &emd_config))
        .map_err(emd_error_to_pyerr)?;
    Ok(DecompositionResultPy::from_rust(result))
}

// ---------------------------------------------------------------------------
// MEMD
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (signal, config=None))]
pub fn memd(
    _py: Python,
    signal: PyReadonlyArray2<f64>,
    config: Option<&MemdConfigPy>,
) -> PyResult<DecompositionResultPy> {
    let dims = signal.shape();
    if dims.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "signal must be a 2D array (channels, samples)",
        ));
    }
    let n_channels = dims[0];
    let n_samples = dims[1];

    let signal_slice = signal.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signal must be contiguous: {}", e))
    })?;

    let mut channels: Vec<Vec<f64>> = Vec::with_capacity(n_channels);
    for ch in 0..n_channels {
        let start = ch * n_samples;
        let end = start + n_samples;
        channels.push(signal_slice[start..end].to_vec());
    }

    let config = config.map_or_else(
        || {
            use ferromode::multivariate::direction_sampling::DirectionConfig;
            use ferromode::sifting::SiftingConfig;
            ferromode::multivariate::memd::MemdConfig::new(
                DirectionConfig::new(8),
                SiftingConfig::default(),
            )
        },
        |c| c.inner.clone(),
    );

    let result = ferromode_memd(&channels, &config).map_err(emd_error_to_pyerr)?;
    Ok(DecompositionResultPy::from_rust(result))
}

// ---------------------------------------------------------------------------
// NA-MEMD
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (signal, config=None))]
pub fn namemd(
    _py: Python,
    signal: PyReadonlyArray2<f64>,
    config: Option<&NaMemdConfigPy>,
) -> PyResult<DecompositionResultPy> {
    let dims = signal.shape();
    if dims.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "signal must be a 2D array (channels, samples)",
        ));
    }
    let n_channels = dims[0];
    let n_samples = dims[1];

    let signal_slice = signal.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signal must be contiguous: {}", e))
    })?;

    let mut channels: Vec<Vec<f64>> = Vec::with_capacity(n_channels);
    for ch in 0..n_channels {
        let start = ch * n_samples;
        let end = start + n_samples;
        channels.push(signal_slice[start..end].to_vec());
    }

    let config = config.map_or_else(
        || {
            use ferromode::multivariate::direction_sampling::DirectionConfig;
            use ferromode::sifting::SiftingConfig;
            let base = ferromode::multivariate::memd::MemdConfig::new(
                DirectionConfig::new(8),
                SiftingConfig::default(),
            );
            ferromode::multivariate::namemd::NaMemdConfig::new(base)
        },
        |c| c.inner.clone(),
    );

    let result = ferromode_namemd(&channels, &config).map_err(emd_error_to_pyerr)?;
    Ok(DecompositionResultPy::from_rust(result))
}

// ---------------------------------------------------------------------------
// VMD
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (signal, config=None))]
pub fn vmd(
    _py: Python,
    signal: PyReadonlyArray1<f64>,
    config: Option<&VmdConfigPy>,
) -> PyResult<DecompositionResultPy> {
    let signal = signal.as_slice().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("signal must be contiguous: {}", e))
    })?;
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = ferromode_vmd(signal, &config).map_err(emd_error_to_pyerr)?;
    Ok(DecompositionResultPy::from_rust(result))
}
