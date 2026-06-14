//! Python bindings for Ferromode via PyO3.
//!
//! This crate provides pure-wrap Python bindings for all 8 Ferromode algorithms.
//! The Python layer handles ONLY marshalling — all algorithm logic and defaults
//! come from Rust.

// pyo3's `#[pymethods]` macro emits trait impls the `non_local_definitions` lint
// flags; this is inherent to the macro and not fixable in our code.
#![allow(non_local_definitions)]
// The pymodule re-exports every pyclass/pyfunction from the submodules below.
#![allow(clippy::wildcard_imports)]

pub mod config;
pub mod differentiable;
pub mod error;
pub mod functions;
pub mod streaming;
pub mod types;

use pyo3::prelude::*;

use config::*;
use differentiable::*;
use functions::*;
use streaming::*;
use types::*;

/// Ferromode — high-performance signal decomposition algorithms.
///
/// Provides EMD, EEMD, CEEMD, CEEMDAN, ICEEMDAN, MEMD, NA-MEMD, and VMD
/// with numpy array I/O, plus streaming decomposition for real-time analysis.
#[pymodule]
fn ferromode(_py: Python, m: &PyModule) -> PyResult<()> {
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
    // Streaming classes
    m.add_class::<StreamingDecomposer>()?;
    m.add_function(wrap_pyfunction!(emd, m)?)?;
    m.add_function(wrap_pyfunction!(eemd, m)?)?;
    m.add_function(wrap_pyfunction!(ceemd, m)?)?;
    m.add_function(wrap_pyfunction!(ceemdan, m)?)?;
    m.add_function(wrap_pyfunction!(iceemdan, m)?)?;
    m.add_function(wrap_pyfunction!(memd, m)?)?;
    m.add_function(wrap_pyfunction!(namemd, m)?)?;
    m.add_function(wrap_pyfunction!(vmd, m)?)?;
    // Differentiable EMD functions
    m.add_class::<EmdForwardResult>()?;
    m.add_function(wrap_pyfunction!(emd_forward, m)?)?;
    m.add_function(wrap_pyfunction!(emd_backward, m)?)?;
    Ok(())
}
