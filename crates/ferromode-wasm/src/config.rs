use super::types::WasmBoundaryCondition;
use ferromode::algorithms::eemd::EnsembleConfig;
use ferromode::algorithms::emd::EmdConfig;
use ferromode::algorithms::vmd::VmdConfig;
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
}

#[wasm_bindgen]
impl WasmEmdConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        sd_threshold: f64,
        s_number: usize,
        max_sifting_iterations: usize,
        max_imfs: usize,
        boundary_condition: WasmBoundaryCondition,
        validate_reconstruction: bool,
        reconstruction_tolerance: f64,
    ) -> Self {
        Self {
            sd_threshold,
            s_number,
            max_sifting_iterations,
            max_imfs,
            boundary_condition,
            validate_reconstruction,
            reconstruction_tolerance,
        }
    }

    pub(crate) fn to_rust(&self) -> EmdConfig {
        EmdConfig {
            sifting_config: SiftingConfig {
                sd_threshold: self.sd_threshold,
                s_number: self.s_number,
                max_sifting_iterations: self.max_sifting_iterations,
                fixed_iterations: None,
                energy_threshold: 1e-6,
                boundary_condition: self.boundary_condition.into(),
            },
            max_imfs: self.max_imfs,
            boundary_condition: self.boundary_condition.into(),
            intermittency: None,
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
