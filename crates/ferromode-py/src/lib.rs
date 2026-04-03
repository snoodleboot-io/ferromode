//! Python bindings for Ferromode via PyO3.

use pyo3::prelude::*;

/// A Python module implemented in Rust using Ferromode.
#[pymodule]
fn ferromode_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(greet, m)?)?;
    Ok(())
}

/// Greet the user — placeholder for future Ferromode Python API.
#[pyfunction]
fn greet(name: &str) -> String {
    format!("Hello from Ferromode, {}!", name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(greet("world"), "Hello from Ferromode, world!");
    }
}
