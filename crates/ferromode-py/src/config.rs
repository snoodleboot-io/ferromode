//! Python wrappers for Ferromode configuration types.
//!
//! Each pyclass wraps a Rust config struct and accepts kwargs in #[new],
//! delegating defaults to Rust `impl Default`.

use ferromode::algorithms::eemd::EnsembleConfig;
use ferromode::algorithms::emd::EmdConfig;
use ferromode::algorithms::vmd::VmdConfig;
use ferromode::boundary::BoundaryConditionType;
use ferromode::multivariate::memd::MemdConfig;
use ferromode::multivariate::namemd::NaMemdConfig;
use ferromode::sifting::{SiftingConfig, StoppingCriterion};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// BoundaryConditionPy
// ---------------------------------------------------------------------------

#[pyclass(name = "BoundaryCondition")]
#[derive(Clone)]
pub struct BoundaryConditionPy {
    pub inner: BoundaryConditionType,
}

#[pymethods]
impl BoundaryConditionPy {
    #[new]
    fn new(value: &str) -> PyResult<Self> {
        let inner = match value {
            "mirror_even" => BoundaryConditionType::MirrorEven,
            "mirror_odd" => BoundaryConditionType::MirrorOdd,
            "periodic" => BoundaryConditionType::Periodic,
            "slope" => BoundaryConditionType::Slope,
            "ar_model" => BoundaryConditionType::ARModel,
            "characteristic_wave" => BoundaryConditionType::CharacteristicWave,
            "waveform_matching" => BoundaryConditionType::WaveformMatching,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown boundary condition: {}",
                    other
                )));
            }
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("BoundaryCondition.{:?}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// StoppingCriterionPy
// ---------------------------------------------------------------------------

#[pyclass(name = "StoppingCriterion")]
#[derive(Clone)]
pub struct StoppingCriterionPy {
    pub inner: StoppingCriterion,
}

#[pymethods]
impl StoppingCriterionPy {
    #[new]
    fn new(value: &str) -> PyResult<Self> {
        let inner = match value {
            "sd_threshold" => StoppingCriterion::SdThreshold,
            "s_number" => StoppingCriterion::SNumber,
            "fixed_iterations" => StoppingCriterion::FixedIterations,
            "energy_difference" => StoppingCriterion::EnergyDifference,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown stopping criterion: {}",
                    other
                )));
            }
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("StoppingCriterion.{:?}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// AlgorithmTypePy
// ---------------------------------------------------------------------------

#[pyclass(name = "AlgorithmType")]
#[derive(Clone)]
pub struct AlgorithmTypePy {
    pub inner: ferromode::types::AlgorithmType,
}

#[pymethods]
impl AlgorithmTypePy {
    #[new]
    fn new(value: &str) -> PyResult<Self> {
        use ferromode::types::AlgorithmType;
        let inner = match value {
            "emd" => AlgorithmType::EMD,
            "eemd" => AlgorithmType::EEMD,
            "ceemd" => AlgorithmType::CEEMD,
            "ceemdan" => AlgorithmType::CEEMDAN,
            "iceemdan" => AlgorithmType::ICEEMDAN,
            "memd" => AlgorithmType::MEMD,
            "namemd" => AlgorithmType::NAMEMD,
            "vmd" => AlgorithmType::VMD,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown algorithm type: {}",
                    other
                )));
            }
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("AlgorithmType.{:?}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// EmdConfigPy
// ---------------------------------------------------------------------------

#[pyclass(name = "EmdConfig")]
#[derive(Clone)]
pub struct EmdConfigPy {
    pub inner: EmdConfig,
}

#[pymethods]
impl EmdConfigPy {
    #[new]
    #[pyo3(signature = (
        max_imfs=None,
        boundary_condition=None,
        reconstruction_tolerance=None,
        validate_reconstruction=None,
        sd_threshold=None,
        s_number=None,
        max_sifting_iterations=None,
        energy_threshold=None,
    ))]
    fn new(
        max_imfs: Option<usize>,
        boundary_condition: Option<&BoundaryConditionPy>,
        reconstruction_tolerance: Option<f64>,
        validate_reconstruction: Option<bool>,
        sd_threshold: Option<f64>,
        s_number: Option<usize>,
        max_sifting_iterations: Option<usize>,
        energy_threshold: Option<f64>,
    ) -> Self {
        let mut config = EmdConfig::default();
        if let Some(v) = max_imfs {
            config.max_imfs = v;
        }
        if let Some(bc) = boundary_condition {
            config.boundary_condition = bc.inner.clone();
        }
        if let Some(v) = reconstruction_tolerance {
            config.reconstruction_tolerance = v;
        }
        if let Some(v) = validate_reconstruction {
            config.validate_reconstruction = v;
        }
        if sd_threshold.is_some()
            || s_number.is_some()
            || max_sifting_iterations.is_some()
            || energy_threshold.is_some()
        {
            let mut sc = SiftingConfig::default();
            if let Some(v) = sd_threshold {
                sc.sd_threshold = v;
            }
            if let Some(v) = s_number {
                sc.s_number = v;
            }
            if let Some(v) = max_sifting_iterations {
                sc.max_sifting_iterations = v;
            }
            if let Some(v) = energy_threshold {
                sc.energy_threshold = v;
            }
            sc.boundary_condition = config.boundary_condition.clone();
            config.sifting_config = sc;
        }
        Self { inner: config }
    }
}

// ---------------------------------------------------------------------------
// EnsembleConfigPy
// ---------------------------------------------------------------------------

#[pyclass(name = "EnsembleConfig")]
#[derive(Clone)]
pub struct EnsembleConfigPy {
    pub inner: EnsembleConfig,
}

#[pymethods]
impl EnsembleConfigPy {
    #[new]
    #[pyo3(signature = (num_ensembles=None, noise_std=None, seed=None))]
    fn new(num_ensembles: Option<usize>, noise_std: Option<f64>, seed: Option<u64>) -> Self {
        let mut config = EnsembleConfig::default();
        if let Some(v) = num_ensembles {
            config.num_ensembles = v;
        }
        if let Some(v) = noise_std {
            config.noise_std = v;
        }
        config.seed = seed;
        Self { inner: config }
    }
}

// ---------------------------------------------------------------------------
// MemdConfigPy
// ---------------------------------------------------------------------------

#[pyclass(name = "MemdConfig")]
#[derive(Clone)]
pub struct MemdConfigPy {
    pub inner: MemdConfig,
}

#[pymethods]
impl MemdConfigPy {
    #[new]
    #[pyo3(signature = (max_imfs=None, num_directions=None, sd_threshold=None, s_number=None, max_sifting_iterations=None))]
    fn new(
        max_imfs: Option<usize>,
        num_directions: Option<usize>,
        sd_threshold: Option<f64>,
        s_number: Option<usize>,
        max_sifting_iterations: Option<usize>,
    ) -> Self {
        use ferromode::multivariate::direction_sampling::DirectionConfig;

        let dir_config = if let Some(n) = num_directions {
            DirectionConfig::new(n)
        } else {
            DirectionConfig::default()
        };

        let mut sifting_config = SiftingConfig::default();
        if let Some(v) = sd_threshold {
            sifting_config.sd_threshold = v;
        }
        if let Some(v) = s_number {
            sifting_config.s_number = v;
        }
        if let Some(v) = max_sifting_iterations {
            sifting_config.max_sifting_iterations = v;
        }

        let mut config = MemdConfig::new(dir_config, sifting_config);
        if let Some(v) = max_imfs {
            config = config.with_max_imfs(v);
        }
        Self { inner: config }
    }
}

// ---------------------------------------------------------------------------
// NaMemdConfigPy
// ---------------------------------------------------------------------------

#[pyclass(name = "NaMemdConfig")]
#[derive(Clone)]
pub struct NaMemdConfigPy {
    pub inner: NaMemdConfig,
}

#[pymethods]
impl NaMemdConfigPy {
    #[new]
    #[pyo3(signature = (n_noise_channels=None, noise_std=None, seed=None, max_imfs=None, num_directions=None))]
    fn new(
        n_noise_channels: Option<usize>,
        noise_std: Option<f64>,
        seed: Option<u64>,
        max_imfs: Option<usize>,
        num_directions: Option<usize>,
    ) -> Self {
        use ferromode::multivariate::direction_sampling::DirectionConfig;

        let dir_config = if let Some(n) = num_directions {
            DirectionConfig::new(n)
        } else {
            DirectionConfig::default()
        };
        let sifting_config = SiftingConfig::default();
        let base_config = MemdConfig::new(dir_config, sifting_config);
        let base_config =
            if let Some(v) = max_imfs { base_config.with_max_imfs(v) } else { base_config };

        let mut config = NaMemdConfig::new(base_config);
        if let Some(v) = n_noise_channels {
            config = config.with_noise_channels(v);
        }
        if let Some(v) = noise_std {
            config = config.with_noise_std(v);
        }
        if let Some(v) = seed {
            config = config.with_seed(v);
        }
        Self { inner: config }
    }
}

// ---------------------------------------------------------------------------
// VmdConfigPy
// ---------------------------------------------------------------------------

#[pyclass(name = "VmdConfig")]
#[derive(Clone)]
pub struct VmdConfigPy {
    pub inner: VmdConfig,
}

#[pymethods]
impl VmdConfigPy {
    #[new]
    #[pyo3(signature = (n_modes=None, alpha=None, tau=None, tol=None, max_iterations=None))]
    fn new(
        n_modes: Option<usize>,
        alpha: Option<f64>,
        tau: Option<f64>,
        tol: Option<f64>,
        max_iterations: Option<usize>,
    ) -> Self {
        let mut config = VmdConfig::default();
        if let Some(v) = n_modes {
            config.n_modes = v;
        }
        if let Some(v) = alpha {
            config.alpha = v;
        }
        if let Some(v) = tau {
            config.tau = v;
        }
        if let Some(v) = tol {
            config.tol = v;
        }
        if let Some(v) = max_iterations {
            config.max_iterations = v;
        }
        Self { inner: config }
    }
}
