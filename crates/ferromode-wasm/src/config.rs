use super::types::{WasmBoundaryCondition, WasmSplineType};
use ferromode::algorithms::eemd::EnsembleConfig;
use ferromode::algorithms::emd::{EmdConfig, IntermittencyConfig};
use ferromode::algorithms::vmd::VmdConfig;
use ferromode::multivariate::direction_sampling::DirectionConfig;
use ferromode::multivariate::memd::MemdConfig;
use ferromode::multivariate::namemd::NaMemdConfig;
use ferromode::sifting::SiftingConfig;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmEmdConfig {
    pub sd_threshold: f64,
    pub s_number: usize,
    pub max_sifting_iterations: usize,
    pub max_imfs: usize,
    pub boundary_condition: WasmBoundaryCondition,
    pub validate_reconstruction: bool,
    pub reconstruction_tolerance: f64,
    /// Energy-difference sifting stop threshold.
    pub energy_threshold: f64,
    /// Fixed sifting iteration count; negative means "unset".
    pub fixed_iterations: i64,
    /// Spline type for envelope interpolation.
    pub spline_type: WasmSplineType,
    /// Intermittency CV threshold; negative disables intermittency handling.
    pub intermittency_cv: f64,
    /// Minimum extrema intervals for the intermittency test.
    pub intermittency_min_intervals: usize,
}

#[wasm_bindgen]
impl WasmEmdConfig {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)] // mirrors the full Rust EmdConfig surface
    pub fn new(
        sd_threshold: f64,
        s_number: usize,
        max_sifting_iterations: usize,
        max_imfs: usize,
        boundary_condition: WasmBoundaryCondition,
        validate_reconstruction: bool,
        reconstruction_tolerance: f64,
        energy_threshold: f64,
        fixed_iterations: i64,
        spline_type: WasmSplineType,
        intermittency_cv: f64,
        intermittency_min_intervals: usize,
    ) -> Self {
        Self {
            sd_threshold,
            s_number,
            max_sifting_iterations,
            max_imfs,
            boundary_condition,
            validate_reconstruction,
            reconstruction_tolerance,
            energy_threshold,
            fixed_iterations,
            spline_type,
            intermittency_cv,
            intermittency_min_intervals,
        }
    }

    pub(crate) fn to_rust(&self) -> EmdConfig {
        EmdConfig {
            sifting_config: SiftingConfig {
                sd_threshold: self.sd_threshold,
                s_number: self.s_number,
                max_sifting_iterations: self.max_sifting_iterations,
                fixed_iterations: if self.fixed_iterations < 0 {
                    None
                } else {
                    Some(self.fixed_iterations as usize)
                },
                energy_threshold: self.energy_threshold,
                boundary_condition: self.boundary_condition.into(),
                spline_type: self.spline_type.into(),
            },
            max_imfs: self.max_imfs,
            boundary_condition: self.boundary_condition.into(),
            intermittency: if self.intermittency_cv < 0.0 {
                None
            } else {
                Some(IntermittencyConfig {
                    cv_threshold: self.intermittency_cv,
                    min_intervals: self.intermittency_min_intervals,
                })
            },
            reconstruction_tolerance: self.reconstruction_tolerance,
            validate_reconstruction: self.validate_reconstruction,
        }
    }
}

#[wasm_bindgen]
pub struct WasmEnsembleConfig {
    pub num_ensembles: usize,
    pub noise_std: f64,
    pub seed: Option<u64>,
}

#[wasm_bindgen]
impl WasmEnsembleConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(num_ensembles: usize, noise_std: f64, seed: Option<u64>) -> Self {
        Self { num_ensembles, noise_std, seed }
    }

    pub(crate) fn to_rust(&self) -> EnsembleConfig {
        EnsembleConfig {
            num_ensembles: self.num_ensembles,
            noise_std: self.noise_std,
            seed: self.seed,
        }
    }
}

#[wasm_bindgen]
pub struct WasmVmdConfig {
    pub n_modes: usize,
    pub alpha: f64,
    pub tau: f64,
    pub tol: f64,
    pub max_iterations: usize,
}

#[wasm_bindgen]
impl WasmVmdConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(n_modes: usize, alpha: f64, tau: f64, tol: f64, max_iterations: usize) -> Self {
        Self { n_modes, alpha, tau, tol, max_iterations }
    }

    pub(crate) fn to_rust(&self) -> VmdConfig {
        VmdConfig {
            n_modes: self.n_modes,
            alpha: self.alpha,
            tau: self.tau,
            tol: self.tol,
            max_iterations: self.max_iterations,
        }
    }
}

#[wasm_bindgen]
pub struct WasmMemdConfig {
    pub num_directions: usize,
    pub max_imfs: usize,
    pub sd_threshold: f64,
    pub s_number: usize,
    pub max_sifting_iterations: usize,
}

#[wasm_bindgen]
impl WasmMemdConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        num_directions: usize,
        max_imfs: usize,
        sd_threshold: f64,
        s_number: usize,
        max_sifting_iterations: usize,
    ) -> Self {
        Self { num_directions, max_imfs, sd_threshold, s_number, max_sifting_iterations }
    }

    pub(crate) fn to_rust(&self) -> MemdConfig {
        let dir = DirectionConfig::new(if self.num_directions == 0 { 8 } else { self.num_directions });
        let sifting = SiftingConfig {
            sd_threshold: self.sd_threshold,
            s_number: self.s_number,
            max_sifting_iterations: self.max_sifting_iterations,
            ..Default::default()
        };
        let mut cfg = MemdConfig::new(dir, sifting);
        if self.max_imfs > 0 {
            cfg = cfg.with_max_imfs(self.max_imfs);
        }
        cfg
    }
}

#[wasm_bindgen]
pub struct WasmNaMemdConfig {
    pub num_directions: usize,
    pub max_imfs: usize,
    pub max_sifting_iterations: usize,
    pub n_noise_channels: usize,
    pub noise_std: f64,
    pub seed: Option<u64>,
}

#[wasm_bindgen]
impl WasmNaMemdConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        num_directions: usize,
        max_imfs: usize,
        max_sifting_iterations: usize,
        n_noise_channels: usize,
        noise_std: f64,
        seed: Option<u64>,
    ) -> Self {
        Self { num_directions, max_imfs, max_sifting_iterations, n_noise_channels, noise_std, seed }
    }

    pub(crate) fn to_rust(&self) -> NaMemdConfig {
        let dir = DirectionConfig::new(if self.num_directions == 0 { 8 } else { self.num_directions });
        let sifting = SiftingConfig {
            max_sifting_iterations: self.max_sifting_iterations,
            ..Default::default()
        };
        let mut base = MemdConfig::new(dir, sifting);
        if self.max_imfs > 0 {
            base = base.with_max_imfs(self.max_imfs);
        }
        let mut cfg = NaMemdConfig::new(base);
        if self.n_noise_channels > 0 {
            cfg = cfg.with_noise_channels(self.n_noise_channels);
        }
        cfg = cfg.with_noise_std(self.noise_std);
        if let Some(s) = self.seed {
            cfg = cfg.with_seed(s);
        }
        cfg
    }
}
