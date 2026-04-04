use ferromode::types::{DecompositionResult, HilbertResult, ImfCollection};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::config::AlgorithmTypePy;

#[pyclass]
#[derive(Clone)]
pub struct ImfCollectionPy {
    inner: ImfCollection,
}

impl ImfCollectionPy {
    pub fn from_rust(inner: ImfCollection) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl ImfCollectionPy {
    fn imfs(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let n_imfs = self.inner.imfs.len();
        if n_imfs == 0 {
            let arr = PyArray2::zeros(py, [0, 0], false);
            return Ok(arr.to_owned());
        }
        let n_samples = self.inner.imfs[0].len();
        let mut data = vec![0.0f64; n_imfs * n_samples];
        for (i, imf) in self.inner.imfs.iter().enumerate() {
            for (j, &val) in imf.iter().enumerate() {
                data[i * n_samples + j] = val;
            }
        }
        let arr = PyArray2::from_vec2(
            py,
            &self.inner.imfs.iter().map(|imf| imf.clone()).collect::<Vec<_>>(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(arr.to_owned())
    }

    fn residue(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(self.inner.residue.clone().into_pyarray(py).to_owned())
    }

    fn reconstruct(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(self.inner.reconstruct().into_pyarray(py).to_owned())
    }

    fn n_imfs(&self) -> usize {
        self.inner.n_imfs()
    }

    fn orthogonality_index(&self) -> f64 {
        self.inner.orthogonality_index()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct HilbertResultPy {
    inner: HilbertResult,
}

impl HilbertResultPy {
    pub fn from_rust(inner: HilbertResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl HilbertResultPy {
    fn instantaneous_amplitude(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let arr = PyArray2::from_vec2(py, &self.inner.instantaneous_amplitude)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(arr.to_owned())
    }

    fn instantaneous_frequency(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let arr = PyArray2::from_vec2(py, &self.inner.instantaneous_frequency)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(arr.to_owned())
    }

    fn marginal_spectrum(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(self.inner.marginal_spectrum.clone().into_pyarray(py).to_owned())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DecompositionResultPy {
    inner: DecompositionResult,
}

impl DecompositionResultPy {
    pub fn from_rust(inner: DecompositionResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl DecompositionResultPy {
    fn algorithm(&self) -> AlgorithmTypePy {
        self.inner.algorithm.clone().into()
    }

    fn elapsed_ms(&self) -> f64 {
        self.inner.elapsed.as_secs_f64() * 1000.0
    }

    fn n_siftings(&self) -> usize {
        self.inner.n_siftings
    }

    fn config_snapshot(&self) -> String {
        self.inner.config_snapshot.clone()
    }

    fn imfs(&self) -> ImfCollectionPy {
        ImfCollectionPy::from_rust(self.inner.imfs.clone())
    }

    fn reconstruct(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        Ok(self.inner.imfs.reconstruct().into_pyarray(py).to_owned())
    }

    fn hilbert(&self, py: Python, sample_rate: f64) -> PyResult<HilbertResultPy> {
        use ferromode::algorithms::hilbert::hilbert_imf;
        let result = hilbert_imf(&self.inner.imfs.imfs, sample_rate)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(HilbertResultPy::from_rust(result))
    }
}
