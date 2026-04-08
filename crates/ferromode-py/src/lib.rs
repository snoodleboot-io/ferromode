//! Python bindings for Ferromode via PyO3.
//!
//! This crate provides pure-wrap Python bindings for all 8 Ferromode algorithms.
//! The Python layer handles ONLY marshalling — all algorithm logic and defaults
//! come from Rust.

pub mod config;
pub mod error;
pub mod functions;
pub mod types;

use pyo3::prelude::*;

use config::*;
use functions::*;
use types::*;

/// Ferromode — high-performance signal decomposition algorithms.
///
/// Provides EMD, EEMD, CEEMD, CEEMDAN, ICEEMDAN, MEMD, NA-MEMD, and VMD
/// with numpy array I/O.
#[pymodule]
fn ferromode_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EmdConfigPy>()?;
    m.add_class::<EnsembleConfigPy>()?;
    m.add_class::<MemdConfigPy>()?;
    m.add_class::<NaMemdConfigPy>()?;
    m.add_class::<VmdConfigPy>()?;
    m.add_class::<BoundaryConditionPy>()?;
    m.add_class::<StoppingCriterionPy>()?;
    m.add_class::<AlgorithmTypePy>()?;
    m.add_class::<ImfCollectionPy>()?;
    m.add_class::<HilbertResultPy>()?;
    m.add_class::<DecompositionResultPy>()?;
    m.add_function(wrap_pyfunction!(emd, m)?)?;
    m.add_function(wrap_pyfunction!(eemd, m)?)?;
    m.add_function(wrap_pyfunction!(ceemd, m)?)?;
    m.add_function(wrap_pyfunction!(ceemdan, m)?)?;
    m.add_function(wrap_pyfunction!(iceemdan, m)?)?;
    m.add_function(wrap_pyfunction!(memd, m)?)?;
    m.add_function(wrap_pyfunction!(namemd, m)?)?;
    m.add_function(wrap_pyfunction!(vmd, m)?)?;
    Ok(())
}
