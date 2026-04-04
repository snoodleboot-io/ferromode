use ferromode::error::EmdError;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;

pub fn emd_error_to_pyerr(err: EmdError) -> PyErr {
    PyErr::new::<PyValueError, _>(err.to_string())
}
