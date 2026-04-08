//! Python bindings for streaming decomposition.
//!
//! Exposes the streaming decomposer API with thread-safe chunk processing.

use ferromode::adapters::streaming::{ArModel, StreamingDecomposer as RustStreamingDecomposer};
use ferromode::algorithms::emd::EmdConfig;
use ferromode::boundary::BoundaryConditionType;
use ferromode::types::Signal;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Python-compatible streaming decomposer.
///
/// Processes signal chunks sequentially while maintaining state for
/// reduced boundary effects and improved continuity.
#[pyclass]
pub struct StreamingDecomposer {
    inner: RustStreamingDecomposer,
}

#[pymethods]
impl StreamingDecomposer {
    /// Create new streaming decomposer.
    ///
    /// # Arguments
    /// * `max_imfs` - Maximum IMFs to extract (default 8, range 1-16)
    /// * `chunk_size` - Nominal samples per chunk (default 1024)
    /// * `buffer_size` - Ring buffer capacity in samples (default 4096)
    /// * `boundary_condition` - How to extend boundaries: "mirror", "periodic", "extrapolate" (default "mirror")
    #[new]
    #[pyo3(signature = (max_imfs=8, chunk_size=1024, buffer_size=4096, boundary_condition=None))]
    pub fn new(
        max_imfs: usize,
        chunk_size: usize,
        buffer_size: usize,
        boundary_condition: Option<String>,
    ) -> PyResult<Self> {
        // Parse boundary condition
        let boundary = match boundary_condition.as_deref().unwrap_or("mirror") {
            "mirror" => BoundaryConditionType::MirrorEven,
            "mirror_odd" => BoundaryConditionType::MirrorOdd,
            "periodic" => BoundaryConditionType::Periodic,
            "slope" => BoundaryConditionType::Slope,
            "ar_model" => BoundaryConditionType::ARModel,
            "wave_matching" => BoundaryConditionType::WaveformMatching,
            "characteristic_wave" => BoundaryConditionType::CharacteristicWave,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown boundary condition: {}",
                    other
                )))
            }
        };

        // Create base EMD config
        let mut base_config = EmdConfig::default();
        base_config.max_imfs = max_imfs;
        base_config.boundary_condition = boundary;

        // Create predictor (AR model, order 3)
        let predictor = Box::new(
            ArModel::new(3)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
        );

        // Create decomposer
        let decomposer = RustStreamingDecomposer::new(base_config, predictor, buffer_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(Self { inner: decomposer })
    }

    /// Decompose a chunk of signal data.
    ///
    /// # Arguments
    /// * `chunk` - Numpy array of shape (N,) with signal samples
    ///
    /// # Returns
    /// Dictionary with keys:
    /// * 'imfs': (n_imfs, chunk_size) array of IMFs
    /// * 'residue': (chunk_size,) array of residue
    /// * 'metrics': dict with 'spectral_entropy', 'stationarity'
    pub fn decompose_chunk(
        &mut self,
        py: Python,
        chunk: PyReadonlyArray1<f64>,
    ) -> PyResult<PyObject> {
        let chunk_data = chunk.to_vec()?;

        // Convert to Signal type
        let signal = Signal::from_slice(&chunk_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Release GIL during decomposition
        let result = py
            .allow_threads(|| self.inner.decompose_chunk(&signal))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Build Python dict response
        let dict = pyo3::types::PyDict::new(py);

        // IMFs array - result.imfs is Vec<Vec<f64>>
        if !result.imfs.is_empty() {
            let imfs_array = PyArray2::from_vec2(py, &result.imfs)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            dict.set_item("imfs", imfs_array.to_owned())?;
        } else {
            let empty = PyArray2::<f64>::zeros(py, [0, chunk_data.len()], false);
            dict.set_item("imfs", empty.to_owned())?;
        }

        // Residue array
        let residue_array = result.remainder.clone().into_pyarray(py).to_owned();
        dict.set_item("residue", residue_array)?;

        // Metrics
        let metrics = pyo3::types::PyDict::new(py);
        metrics.set_item("spectral_entropy", result.metrics.spectral_entropy)?;
        metrics.set_item("stationarity_score", result.metrics.stationarity_score)?;
        metrics.set_item("extrema_spacing_cv", result.metrics.extrema_spacing_cv)?;
        dict.set_item("metrics", metrics)?;

        Ok(dict.into())
    }

    /// Reset streaming state (clears history, maintains configuration).
    pub fn reset(&mut self) -> PyResult<()> {
        self.inner.reset();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposer_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let _decomposer = StreamingDecomposer::new(8, 1024, 4096, None).unwrap();
        });
    }

    #[test]
    fn test_decomposer_sine_signal() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut decomposer = StreamingDecomposer::new(8, 256, 2048, None).unwrap();

            // Create simple sine signal
            let mut chunk = vec![0.0; 256];
            for i in 0..256 {
                chunk[i] = ((i as f64 * 0.1).sin());
            }

            let result = decomposer.decompose_chunk(py, PyReadonlyArray1::from(&chunk)).unwrap();
            // Result should be a dict
            assert!(pyo3::types::PyDict::is_type_of(&result.as_ref(py)));
        });
    }
}
