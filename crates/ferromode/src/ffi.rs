//! C-compatible FFI layer for foreign language bindings.
//!
//! This module provides a stable C ABI for use by Julia, Python (ctypes),
//! and other languages that can call C functions. The types and functions
//! here are pure marshalling — no algorithm logic.

use crate::algorithms::ceemd::ceemd;
use crate::algorithms::ceemdan::ceemdan;
use crate::algorithms::eemd::{eemd, EnsembleConfig};
use crate::algorithms::emd::{emd, EmdConfig};
use crate::algorithms::iceemdan::iceemdan;
use crate::algorithms::vmd::{vmd, VmdConfig};
use crate::multivariate::memd::{memd, MemdConfig};
use crate::multivariate::namemd::{namemd, NaMemdConfig};
use crate::types::{AlgorithmType, DecompositionResult};
use std::slice;

// ---------------------------------------------------------------------------
// C-compatible config structs
// ---------------------------------------------------------------------------

/// C-compatible EMD configuration.
#[repr(C)]
pub struct CEmdConfig {
    /// Maximum number of IMFs to extract (0 = no limit).
    pub max_imfs: usize,
    /// Sifting stop criterion: standard deviation threshold between successive envelopes.
    pub sd_threshold: f64,
    /// Sifting S-number stop criterion: consecutive siftings satisfying the IMF conditions.
    pub s_number: usize,
    /// Hard limit on sifting iterations per IMF.
    pub max_sifting_iterations: usize,
    /// Boundary condition code: 0=MirrorEven, 1=MirrorOdd, 2=Periodic, 3=Slope,
    /// 4=ARModel, 5=CharacteristicWave, 6=WaveformMatching, 7=PalindromeCyclic.
    pub boundary_condition: i32,
    // -- Extended knobs (appended for ABI stability) -------------------------
    /// Spline type for envelope interpolation: 0=Natural, 1=Periodic, 2=NotAKnot.
    pub spline_type: i32,
    /// Fixed sifting iteration count; negative means `None` (use other criteria).
    pub fixed_iterations: i64,
    /// Energy-difference stop threshold for sifting.
    pub energy_threshold: f64,
    /// Reconstruction tolerance used when `validate_reconstruction` is non-zero.
    pub reconstruction_tolerance: f64,
    /// Non-zero to validate that IMFs+residue reconstruct the signal.
    pub validate_reconstruction: i32,
    /// Intermittency CV threshold; negative disables intermittency handling.
    pub intermittency_cv: f64,
    /// Minimum extrema intervals for the intermittency test (used when cv >= 0).
    pub intermittency_min_intervals: usize,
}

impl Default for CEmdConfig {
    fn default() -> Self {
        Self {
            max_imfs: 0,
            sd_threshold: 0.2,
            s_number: 5,
            max_sifting_iterations: 100,
            boundary_condition: 0,
            spline_type: 0,
            fixed_iterations: -1,
            energy_threshold: 1e-6,
            reconstruction_tolerance: 1e-12,
            validate_reconstruction: 0,
            intermittency_cv: -1.0,
            intermittency_min_intervals: 0,
        }
    }
}

/// C-compatible ensemble configuration (EEMD, CEEMD, CEEMDAN, ICEEMDAN).
#[repr(C)]
pub struct CEnsembleConfig {
    /// Number of ensemble trials to average.
    pub num_ensembles: usize,
    /// Standard deviation of the Gaussian noise added to each trial.
    pub noise_std: f64,
    /// RNG seed value used when `use_seed` is non-zero.
    pub seed: u64,
    /// Non-zero to use `seed` for reproducible noise; zero for random noise.
    pub use_seed: i32,
}

/// C-compatible MEMD configuration.
#[repr(C)]
pub struct CMemdConfig {
    /// Number of projection directions for multivariate envelope estimation.
    pub num_directions: usize,
    /// RNG seed for direction sampling (deterministic projection set).
    pub direction_seed: u64,
    /// Maximum number of IMFs to extract (0 = no limit).
    pub max_imfs: usize,
    /// Sifting stop criterion: standard deviation threshold.
    pub sd_threshold: f64,
    /// Sifting S-number stop criterion.
    pub s_number: usize,
    /// Hard limit on sifting iterations per IMF.
    pub max_sifting_iterations: usize,
}

/// C-compatible NA-MEMD configuration.
#[repr(C)]
pub struct CNaMemdConfig {
    /// Base MEMD settings shared with the standard MEMD algorithm.
    pub base: CMemdConfig,
    /// Number of pure noise channels to inject alongside the data channels.
    pub n_noise_channels: usize,
    /// Standard deviation of the injected noise channels.
    pub noise_std: f64,
    /// RNG seed used when `use_seed` is non-zero.
    pub seed: u64,
    /// Non-zero to use `seed` for reproducible noise channels; zero for random.
    pub use_seed: i32,
}

/// C-compatible VMD configuration.
#[repr(C)]
pub struct CVmdConfig {
    /// Number of modes to extract.
    pub n_modes: usize,
    /// Bandwidth constraint penalty parameter.
    pub alpha: f64,
    /// Noise tolerance (Lagrangian multiplier update step size).
    pub tau: f64,
    /// Convergence tolerance for the dual ascent loop.
    pub tol: f64,
    /// Maximum number of dual ascent iterations.
    pub max_iterations: usize,
}

// ---------------------------------------------------------------------------
// C-compatible result structs
// ---------------------------------------------------------------------------

/// C-compatible IMF collection.
///
/// Memory layout: `imfs` points to an array of `n_imfs` pointers, each
/// pointing to an array of `n_samples` f64 values. The `residue` field
/// points to an array of `n_samples` f64 values.
#[repr(C)]
pub struct CImfCollection {
    /// Pointer to an array of `n_imfs` pointers, each pointing to `n_samples` f64 values.
    pub imfs: *const *const f64,
    /// Number of IMFs in the collection.
    pub n_imfs: usize,
    /// Number of samples in each IMF and in the residue array.
    pub n_samples: usize,
    /// Pointer to an array of `n_samples` f64 values representing the residue.
    pub residue: *const f64,
}

/// C-compatible decomposition result.
#[repr(C)]
pub struct CDecompositionResult {
    /// Pointer to the IMF collection, or null on error.
    pub imfs: *mut CImfCollection,
    /// Algorithm code: 0=EMD, 1=EEMD, 2=CEEMD, 3=CEEMDAN, 4=ICEEMDAN, 5=MEMD, 6=NAMEMD, 7=VMD.
    pub algorithm: i32,
    /// Wall-clock time taken by the decomposition, in milliseconds.
    pub elapsed_ms: f64,
    /// Total sifting iterations performed across all IMFs.
    pub n_siftings: usize,
    /// Pointer to an error code (1 = error), or null on success.
    pub error: *mut i32,
    /// Pointer to a null-terminated error message string, or null on success.
    pub error_msg: *mut std::os::raw::c_char,
}

// ---------------------------------------------------------------------------
// Internal: convert Rust result to C result
// ---------------------------------------------------------------------------

fn result_to_c(
    result: Result<DecompositionResult, crate::error::EmdError>,
) -> *mut CDecompositionResult {
    match result {
        Ok(decomp) => {
            let n_imfs = decomp.imfs.imfs.len();
            let n_samples = if n_imfs > 0 {
                decomp.imfs.imfs[0].len()
            } else if !decomp.imfs.residue.is_empty() {
                decomp.imfs.residue.len()
            } else {
                0
            };

            // Allocate IMF pointers
            let mut imf_ptrs: Vec<*const f64> = Vec::with_capacity(n_imfs);
            for imf in &decomp.imfs.imfs {
                let _ptr = imf.as_ptr();
                // Leak the Vec so the pointer remains valid
                // The caller must call ferromode_free_result to free
                std::mem::forget(imf.clone());
                // We need to allocate on heap and leak
                let boxed: Box<[f64]> = imf.clone().into_boxed_slice();
                imf_ptrs.push(Box::into_raw(boxed) as *const f64);
            }

            // Allocate residue
            let residue_box: Box<[f64]> = decomp.imfs.residue.clone().into_boxed_slice();
            let residue_ptr = Box::into_raw(residue_box) as *const f64;

            // Allocate the collection
            let collection = Box::new(CImfCollection {
                imfs: Box::into_raw(imf_ptrs.into_boxed_slice()) as *const *const f64,
                n_imfs,
                n_samples,
                residue: residue_ptr,
            });
            let collection_ptr = Box::into_raw(collection);

            let algo_code = match decomp.algorithm {
                AlgorithmType::EMD => 0,
                AlgorithmType::EEMD => 1,
                AlgorithmType::CEEMD => 2,
                AlgorithmType::CEEMDAN => 3,
                AlgorithmType::ICEEMDAN => 4,
                AlgorithmType::MEMD => 5,
                AlgorithmType::NAMEMD => 6,
                AlgorithmType::VMD => 7,
            };

            let elapsed_ms = decomp.elapsed.as_secs_f64() * 1000.0;

            let c_result = Box::new(CDecompositionResult {
                imfs: collection_ptr,
                algorithm: algo_code,
                elapsed_ms,
                n_siftings: decomp.n_siftings,
                error: std::ptr::null_mut(),
                error_msg: std::ptr::null_mut(),
            });
            Box::into_raw(c_result)
        }
        Err(e) => {
            let error_code = Box::new(1i32);
            let error_ptr = Box::into_raw(error_code);

            let msg = std::ffi::CString::new(e.to_string()).unwrap_or_default();
            let msg_ptr = msg.into_raw();

            let c_result = Box::new(CDecompositionResult {
                imfs: std::ptr::null_mut(),
                algorithm: -1,
                elapsed_ms: 0.0,
                n_siftings: 0,
                error: error_ptr,
                error_msg: msg_ptr,
            });
            Box::into_raw(c_result)
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: config conversion helpers
// ---------------------------------------------------------------------------

fn c_boundary_to_rust(code: i32) -> crate::boundary::BoundaryConditionType {
    use crate::boundary::BoundaryConditionType as B;
    match code {
        0 => B::MirrorEven,
        1 => B::MirrorOdd,
        2 => B::Periodic,
        3 => B::Slope,
        4 => B::ARModel,
        5 => B::CharacteristicWave,
        6 => B::WaveformMatching,
        7 => B::PalindromeCyclic,
        _ => B::MirrorEven,
    }
}

fn c_spline_to_rust(code: i32) -> crate::spline::SplineType {
    use crate::spline::SplineType as S;
    match code {
        1 => S::Periodic,
        2 => S::NotAKnot,
        _ => S::Natural,
    }
}

/// Build a full Rust `EmdConfig` from the (widened) C config, honoring every knob.
fn c_emd_config_to_rust(config: &CEmdConfig) -> EmdConfig {
    let boundary = c_boundary_to_rust(config.boundary_condition);
    let sifting_config = crate::sifting::SiftingConfig {
        max_sifting_iterations: config.max_sifting_iterations,
        sd_threshold: config.sd_threshold,
        s_number: config.s_number,
        fixed_iterations: if config.fixed_iterations < 0 {
            None
        } else {
            Some(config.fixed_iterations as usize)
        },
        energy_threshold: config.energy_threshold,
        boundary_condition: boundary,
        spline_type: c_spline_to_rust(config.spline_type),
    };
    let intermittency = if config.intermittency_cv < 0.0 {
        None
    } else {
        Some(crate::algorithms::emd::IntermittencyConfig {
            cv_threshold: config.intermittency_cv,
            min_intervals: config.intermittency_min_intervals,
        })
    };
    EmdConfig {
        sifting_config,
        max_imfs: config.max_imfs,
        boundary_condition: boundary,
        intermittency,
        reconstruction_tolerance: config.reconstruction_tolerance,
        validate_reconstruction: config.validate_reconstruction != 0,
    }
}

fn c_ensemble_config_to_rust(config: &CEnsembleConfig) -> EnsembleConfig {
    EnsembleConfig {
        num_ensembles: config.num_ensembles,
        noise_std: config.noise_std,
        seed: if config.use_seed != 0 { Some(config.seed) } else { None },
    }
}

// ---------------------------------------------------------------------------
// FFI functions — univariate algorithms
// ---------------------------------------------------------------------------

/// Perform EMD decomposition.
///
/// # Safety
/// Caller must ensure `signal` points to `n_samples` valid f64 values
/// and `config` points to a valid CEmdConfig. Caller must call
/// `ferromode_free_result` on the returned pointer.
#[no_mangle]
pub unsafe extern "C" fn ferromode_emd(
    signal: *const f64,
    n_samples: usize,
    config: *const CEmdConfig,
) -> *mut CDecompositionResult {
    if signal.is_null() || config.is_null() || n_samples == 0 {
        return result_to_c(Err(crate::error::EmdError::EmptySignal));
    }

    let signal_slice = slice::from_raw_parts(signal, n_samples);
    let rust_config = &*config;

    let emd_config = c_emd_config_to_rust(rust_config);
    let result = emd(signal_slice, &emd_config);
    result_to_c(result)
}

/// Perform EEMD decomposition.
///
/// # Safety
/// Caller must ensure `signal` points to `n_samples` valid f64 values,
/// `ensemble_config` and `emd_config` point to valid structs.
#[no_mangle]
pub unsafe extern "C" fn ferromode_eemd(
    signal: *const f64,
    n_samples: usize,
    ensemble_config: *const CEnsembleConfig,
    emd_config: *const CEmdConfig,
) -> *mut CDecompositionResult {
    if signal.is_null() || ensemble_config.is_null() || emd_config.is_null() || n_samples == 0 {
        return result_to_c(Err(crate::error::EmdError::EmptySignal));
    }

    let signal_slice = slice::from_raw_parts(signal, n_samples);
    let ens_config = c_ensemble_config_to_rust(&*ensemble_config);
    let rust_emd_config = &*emd_config;

    let emd_cfg = c_emd_config_to_rust(rust_emd_config);

    let result = eemd(signal_slice, &ens_config, &emd_cfg);
    result_to_c(result)
}

/// Perform CEEMD decomposition.
///
/// # Safety
/// Caller must ensure `signal` points to `n_samples` valid f64 values,
/// `ensemble_config` and `emd_config` point to valid structs.
#[no_mangle]
pub unsafe extern "C" fn ferromode_ceemd(
    signal: *const f64,
    n_samples: usize,
    ensemble_config: *const CEnsembleConfig,
    emd_config: *const CEmdConfig,
) -> *mut CDecompositionResult {
    if signal.is_null() || ensemble_config.is_null() || emd_config.is_null() || n_samples == 0 {
        return result_to_c(Err(crate::error::EmdError::EmptySignal));
    }

    let signal_slice = slice::from_raw_parts(signal, n_samples);
    let ens_config = c_ensemble_config_to_rust(&*ensemble_config);
    let rust_emd_config = &*emd_config;

    let emd_cfg = c_emd_config_to_rust(rust_emd_config);

    let result = ceemd(signal_slice, &ens_config, &emd_cfg);
    result_to_c(result)
}

/// Perform CEEMDAN decomposition.
///
/// # Safety
/// Caller must ensure `signal` points to `n_samples` valid f64 values,
/// `ensemble_config` and `emd_config` point to valid structs.
#[no_mangle]
pub unsafe extern "C" fn ferromode_ceemdan(
    signal: *const f64,
    n_samples: usize,
    ensemble_config: *const CEnsembleConfig,
    emd_config: *const CEmdConfig,
) -> *mut CDecompositionResult {
    if signal.is_null() || ensemble_config.is_null() || emd_config.is_null() || n_samples == 0 {
        return result_to_c(Err(crate::error::EmdError::EmptySignal));
    }

    let signal_slice = slice::from_raw_parts(signal, n_samples);
    let ens_config = c_ensemble_config_to_rust(&*ensemble_config);
    let rust_emd_config = &*emd_config;

    let emd_cfg = c_emd_config_to_rust(rust_emd_config);

    let result = ceemdan(signal_slice, &ens_config, &emd_cfg);
    result_to_c(result)
}

/// Perform ICEEMDAN decomposition.
///
/// # Safety
/// Caller must ensure `signal` points to `n_samples` valid f64 values,
/// `ensemble_config` and `emd_config` point to valid structs.
#[no_mangle]
pub unsafe extern "C" fn ferromode_iceemdan(
    signal: *const f64,
    n_samples: usize,
    ensemble_config: *const CEnsembleConfig,
    emd_config: *const CEmdConfig,
) -> *mut CDecompositionResult {
    if signal.is_null() || ensemble_config.is_null() || emd_config.is_null() || n_samples == 0 {
        return result_to_c(Err(crate::error::EmdError::EmptySignal));
    }

    let signal_slice = slice::from_raw_parts(signal, n_samples);
    let ens_config = c_ensemble_config_to_rust(&*ensemble_config);
    let rust_emd_config = &*emd_config;

    let emd_cfg = c_emd_config_to_rust(rust_emd_config);

    let result = iceemdan(signal_slice, &ens_config, &emd_cfg);
    result_to_c(result)
}

/// Perform VMD decomposition.
///
/// # Safety
/// Caller must ensure `signal` points to `n_samples` valid f64 values
/// and `config` points to a valid CVmdConfig.
#[no_mangle]
pub unsafe extern "C" fn ferromode_vmd(
    signal: *const f64,
    n_samples: usize,
    config: *const CVmdConfig,
) -> *mut CDecompositionResult {
    if signal.is_null() || config.is_null() || n_samples == 0 {
        return result_to_c(Err(crate::error::EmdError::EmptySignal));
    }

    let signal_slice = slice::from_raw_parts(signal, n_samples);
    let rust_config = &*config;

    let vmd_config = VmdConfig {
        n_modes: rust_config.n_modes,
        alpha: rust_config.alpha,
        tau: rust_config.tau,
        tol: rust_config.tol,
        max_iterations: rust_config.max_iterations,
    };

    let result = vmd(signal_slice, &vmd_config);
    result_to_c(result)
}

// ---------------------------------------------------------------------------
// FFI functions — multivariate algorithms
// ---------------------------------------------------------------------------

/// Perform MEMD decomposition.
///
/// # Safety
/// Caller must ensure `signal` points to `n_channels * n_samples` valid f64
/// values in row-major order (channel-major), and `config` is valid.
#[no_mangle]
pub unsafe extern "C" fn ferromode_memd(
    signal: *const f64,
    n_channels: usize,
    n_samples: usize,
    config: *const CMemdConfig,
) -> *mut CDecompositionResult {
    if signal.is_null() || config.is_null() || n_channels == 0 || n_samples == 0 {
        return result_to_c(Err(crate::error::EmdError::EmptySignal));
    }

    let signal_slice = slice::from_raw_parts(signal, n_channels * n_samples);
    let rust_config = &*config;

    // Convert flat array to Vec<Vec<f64>>
    let channels: Vec<Vec<f64>> = (0..n_channels)
        .map(|ch| {
            let start = ch * n_samples;
            signal_slice[start..start + n_samples].to_vec()
        })
        .collect();

    let dir_config =
        crate::multivariate::direction_sampling::DirectionConfig::new(rust_config.num_directions)
            .with_seed(rust_config.direction_seed);
    let sifting_config = crate::sifting::SiftingConfig {
        sd_threshold: rust_config.sd_threshold,
        s_number: rust_config.s_number,
        max_sifting_iterations: rust_config.max_sifting_iterations,
        ..Default::default()
    };
    let memd_config =
        MemdConfig::new(dir_config, sifting_config).with_max_imfs(rust_config.max_imfs);

    let result = memd(&channels, &memd_config);
    result_to_c(result)
}

/// Perform NA-MEMD decomposition.
///
/// # Safety
/// Caller must ensure `signal` points to `n_channels * n_samples` valid f64
/// values in row-major order, and `config` is valid.
#[no_mangle]
pub unsafe extern "C" fn ferromode_namemd(
    signal: *const f64,
    n_channels: usize,
    n_samples: usize,
    config: *const CNaMemdConfig,
) -> *mut CDecompositionResult {
    if signal.is_null() || config.is_null() || n_channels == 0 || n_samples == 0 {
        return result_to_c(Err(crate::error::EmdError::EmptySignal));
    }

    let signal_slice = slice::from_raw_parts(signal, n_channels * n_samples);
    let rust_config = &*config;

    let channels: Vec<Vec<f64>> = (0..n_channels)
        .map(|ch| {
            let start = ch * n_samples;
            signal_slice[start..start + n_samples].to_vec()
        })
        .collect();

    let dir_config = crate::multivariate::direction_sampling::DirectionConfig::new(
        rust_config.base.num_directions,
    )
    .with_seed(rust_config.base.direction_seed);
    let sifting_config = crate::sifting::SiftingConfig {
        sd_threshold: rust_config.base.sd_threshold,
        s_number: rust_config.base.s_number,
        max_sifting_iterations: rust_config.base.max_sifting_iterations,
        ..Default::default()
    };
    let base_memd_config =
        MemdConfig::new(dir_config, sifting_config).with_max_imfs(rust_config.base.max_imfs);

    let namemd_config = NaMemdConfig::new(base_memd_config)
        .with_noise_channels(rust_config.n_noise_channels)
        .with_noise_std(rust_config.noise_std);
    let namemd_config = if rust_config.use_seed != 0 {
        namemd_config.with_seed(rust_config.seed)
    } else {
        namemd_config
    };

    let result = namemd(&channels, &namemd_config);
    result_to_c(result)
}

// ---------------------------------------------------------------------------
// FFI functions — result accessors and memory management
// ---------------------------------------------------------------------------

/// Get the number of IMFs in a decomposition result.
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
#[no_mangle]
pub unsafe extern "C" fn ferromode_result_n_imfs(result: *const CDecompositionResult) -> usize {
    if result.is_null() {
        return 0;
    }
    let r = &*result;
    if r.imfs.is_null() {
        return 0;
    }
    let collection = &*r.imfs;
    collection.n_imfs
}

/// Get the number of samples per IMF.
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
#[no_mangle]
pub unsafe extern "C" fn ferromode_result_n_samples(result: *const CDecompositionResult) -> usize {
    if result.is_null() {
        return 0;
    }
    let r = &*result;
    if r.imfs.is_null() {
        return 0;
    }
    let collection = &*r.imfs;
    collection.n_samples
}

/// Get a pointer to the data for a specific IMF.
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
/// `imf_index` must be less than `ferromode_result_n_imfs(result)`.
#[no_mangle]
pub unsafe extern "C" fn ferromode_result_imf_ptr(
    result: *const CDecompositionResult,
    imf_index: usize,
) -> *const f64 {
    if result.is_null() {
        return std::ptr::null();
    }
    let r = &*result;
    if r.imfs.is_null() {
        return std::ptr::null();
    }
    let collection = &*r.imfs;
    if imf_index >= collection.n_imfs {
        return std::ptr::null();
    }
    let imf_ptrs = slice::from_raw_parts(collection.imfs, collection.n_imfs);
    imf_ptrs[imf_index]
}

/// Get a pointer to the residue data.
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
#[no_mangle]
pub unsafe extern "C" fn ferromode_result_residue_ptr(
    result: *const CDecompositionResult,
) -> *const f64 {
    if result.is_null() {
        return std::ptr::null();
    }
    let r = &*result;
    if r.imfs.is_null() {
        return std::ptr::null();
    }
    let collection = &*r.imfs;
    collection.residue
}

/// Get the total sifting iteration count from a result.
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
#[no_mangle]
pub unsafe extern "C" fn ferromode_result_n_siftings(result: *const CDecompositionResult) -> usize {
    if result.is_null() {
        return 0;
    }
    (&*result).n_siftings
}

/// Get the algorithm code from a result.
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
#[no_mangle]
pub unsafe extern "C" fn ferromode_result_algorithm(result: *const CDecompositionResult) -> i32 {
    if result.is_null() {
        return -1;
    }
    (&*result).algorithm
}

/// Get the elapsed time in milliseconds.
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
#[no_mangle]
pub unsafe extern "C" fn ferromode_result_elapsed_ms(result: *const CDecompositionResult) -> f64 {
    if result.is_null() {
        return 0.0;
    }
    (&*result).elapsed_ms
}

/// Check if the result has an error.
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
#[no_mangle]
pub unsafe extern "C" fn ferromode_result_has_error(result: *const CDecompositionResult) -> i32 {
    if result.is_null() {
        return 1;
    }
    let r = &*result;
    if !r.error.is_null() {
        1
    } else {
        0
    }
}

/// Get the error message from a result (if any).
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
/// The returned string is valid until `ferromode_free_result` is called.
#[no_mangle]
pub unsafe extern "C" fn ferromode_result_error_msg(
    result: *const CDecompositionResult,
) -> *const std::os::raw::c_char {
    if result.is_null() {
        return std::ptr::null();
    }
    (&*result).error_msg
}

/// Free a decomposition result and all associated memory.
///
/// # Safety
/// `result` must be a valid pointer from a ferromode FFI function.
/// Must not be called more than once on the same pointer.
#[no_mangle]
pub unsafe extern "C" fn ferromode_free_result(result: *mut CDecompositionResult) {
    if result.is_null() {
        return;
    }

    let r = Box::from_raw(result);

    // Free error message if present
    if !r.error_msg.is_null() {
        drop(std::ffi::CString::from_raw(r.error_msg));
    }

    // Free error code if present
    if !r.error.is_null() {
        drop(Box::from_raw(r.error));
    }

    // Free IMF collection if present
    if !r.imfs.is_null() {
        let collection = Box::from_raw(r.imfs);

        // Free each IMF data
        if !collection.imfs.is_null() {
            let imf_ptrs = Box::from_raw(slice::from_raw_parts_mut(
                collection.imfs as *mut *const f64,
                collection.n_imfs,
            ));
            for imf_ptr in imf_ptrs.iter() {
                if !imf_ptr.is_null() {
                    // Reconstruct the Box<[f64]> and drop it
                    drop(Box::from_raw(slice::from_raw_parts_mut(
                        *imf_ptr as *mut f64,
                        collection.n_samples,
                    )));
                }
            }
        }

        // Free residue
        if !collection.residue.is_null() {
            drop(Box::from_raw(slice::from_raw_parts_mut(
                collection.residue as *mut f64,
                collection.n_samples,
            )));
        }
    }
}

/// Copy IMF data into a caller-provided buffer.
///
/// # Safety
/// `result` must be valid, `out` must point to at least `n_samples` f64 values.
#[no_mangle]
pub unsafe extern "C" fn ferromode_copy_imf(
    result: *const CDecompositionResult,
    imf_index: usize,
    out: *mut f64,
    n_samples: usize,
) -> i32 {
    if result.is_null() || out.is_null() {
        return -1;
    }
    let r = &*result;
    if r.imfs.is_null() {
        return -1;
    }
    let collection = &*r.imfs;
    if imf_index >= collection.n_imfs {
        return -1;
    }
    if n_samples != collection.n_samples {
        return -1;
    }

    let imf_ptrs = slice::from_raw_parts(collection.imfs, collection.n_imfs);
    let src = slice::from_raw_parts(imf_ptrs[imf_index], n_samples);
    let dst = slice::from_raw_parts_mut(out, n_samples);
    dst.copy_from_slice(src);
    0
}

/// Copy residue data into a caller-provided buffer.
///
/// # Safety
/// `result` must be valid, `out` must point to at least `n_samples` f64 values.
#[no_mangle]
pub unsafe extern "C" fn ferromode_copy_residue(
    result: *const CDecompositionResult,
    out: *mut f64,
    n_samples: usize,
) -> i32 {
    if result.is_null() || out.is_null() {
        return -1;
    }
    let r = &*result;
    if r.imfs.is_null() {
        return -1;
    }
    let collection = &*r.imfs;
    if n_samples != collection.n_samples {
        return -1;
    }

    let src = slice::from_raw_parts(collection.residue, n_samples);
    let dst = slice::from_raw_parts_mut(out, n_samples);
    dst.copy_from_slice(src);
    0
}

/// Reconstruct signal from IMFs and residue into caller-provided buffer.
///
/// # Safety
/// `result` must be valid, `out` must point to at least `n_samples` f64 values.
#[no_mangle]
pub unsafe extern "C" fn ferromode_reconstruct(
    result: *const CDecompositionResult,
    out: *mut f64,
    n_samples: usize,
) -> i32 {
    if result.is_null() || out.is_null() {
        return -1;
    }
    let r = &*result;
    if r.imfs.is_null() {
        return -1;
    }
    let collection = &*r.imfs;
    if n_samples != collection.n_samples {
        return -1;
    }

    let dst = slice::from_raw_parts_mut(out, n_samples);
    dst.fill(0.0);

    // Sum IMFs
    if collection.n_imfs > 0 && !collection.imfs.is_null() {
        let imf_ptrs = slice::from_raw_parts(collection.imfs, collection.n_imfs);
        for imf_ptr in imf_ptrs {
            let imf = slice::from_raw_parts(*imf_ptr, n_samples);
            for (d, s) in dst.iter_mut().zip(imf.iter()) {
                *d += s;
            }
        }
    }

    // Add residue
    if !collection.residue.is_null() {
        let residue = slice::from_raw_parts(collection.residue, n_samples);
        for (d, s) in dst.iter_mut().zip(residue.iter()) {
            *d += s;
        }
    }

    0
}

// ---------------------------------------------------------------------------
// Hilbert spectral analysis
// ---------------------------------------------------------------------------

/// C-compatible Hilbert spectrum result.
///
/// `instantaneous_amplitude` and `instantaneous_frequency` are flattened
/// row-major `n_imfs × n_samples`. `marginal_spectrum` has `n_freq_bins` values.
#[repr(C)]
pub struct CHilbertResult {
    /// Flattened (n_imfs × n_samples) instantaneous amplitude, or null on error.
    pub instantaneous_amplitude: *const f64,
    /// Flattened (n_imfs × n_samples) instantaneous frequency, or null on error.
    pub instantaneous_frequency: *const f64,
    /// Marginal spectrum (`n_freq_bins` values), or null on error.
    pub marginal_spectrum: *const f64,
    /// Number of IMFs.
    pub n_imfs: usize,
    /// Samples per IMF.
    pub n_samples: usize,
    /// Number of marginal-spectrum frequency bins.
    pub n_freq_bins: usize,
    /// Pointer to an error code (1 = error), or null on success.
    pub error: *mut i32,
    /// Null-terminated error message, or null on success.
    pub error_msg: *mut std::os::raw::c_char,
}

fn hilbert_err(msg: &str) -> *mut CHilbertResult {
    let error = Box::into_raw(Box::new(1i32));
    let error_msg = std::ffi::CString::new(msg).unwrap_or_default().into_raw();
    Box::into_raw(Box::new(CHilbertResult {
        instantaneous_amplitude: std::ptr::null(),
        instantaneous_frequency: std::ptr::null(),
        marginal_spectrum: std::ptr::null(),
        n_imfs: 0,
        n_samples: 0,
        n_freq_bins: 0,
        error,
        error_msg,
    }))
}

/// Compute the Hilbert spectrum for a set of IMFs.
///
/// # Safety
/// `imfs_flat` must point to `n_imfs * n_samples` valid f64 (row-major).
/// Caller must call `ferromode_free_hilbert` on the result.
#[no_mangle]
pub unsafe extern "C" fn ferromode_hilbert(
    imfs_flat: *const f64,
    n_imfs: usize,
    n_samples: usize,
    sample_rate: f64,
) -> *mut CHilbertResult {
    if imfs_flat.is_null() || n_imfs == 0 || n_samples == 0 {
        return hilbert_err("invalid input: null pointer or zero dimension");
    }
    let flat = slice::from_raw_parts(imfs_flat, n_imfs * n_samples);
    let imfs: Vec<Vec<f64>> =
        (0..n_imfs).map(|i| flat[i * n_samples..(i + 1) * n_samples].to_vec()).collect();

    match crate::algorithms::hilbert::hilbert_imf(&imfs, sample_rate) {
        Ok(h) => {
            let amp: Vec<f64> = h.instantaneous_amplitude.iter().flatten().copied().collect();
            let freq: Vec<f64> = h.instantaneous_frequency.iter().flatten().copied().collect();
            let n_freq_bins = h.marginal_spectrum.len();
            let amp_ptr = Box::into_raw(amp.into_boxed_slice()) as *const f64;
            let freq_ptr = Box::into_raw(freq.into_boxed_slice()) as *const f64;
            let marg_ptr =
                Box::into_raw(h.marginal_spectrum.into_boxed_slice()) as *const f64;
            Box::into_raw(Box::new(CHilbertResult {
                instantaneous_amplitude: amp_ptr,
                instantaneous_frequency: freq_ptr,
                marginal_spectrum: marg_ptr,
                n_imfs,
                n_samples,
                n_freq_bins,
                error: std::ptr::null_mut(),
                error_msg: std::ptr::null_mut(),
            }))
        }
        Err(e) => hilbert_err(&e.to_string()),
    }
}

/// Free a `CHilbertResult` returned by `ferromode_hilbert`.
///
/// # Safety
/// `result` must be a pointer returned by `ferromode_hilbert` (or null).
#[no_mangle]
pub unsafe extern "C" fn ferromode_free_hilbert(result: *mut CHilbertResult) {
    if result.is_null() {
        return;
    }
    let r = Box::from_raw(result);
    let nm = r.n_imfs * r.n_samples;
    if !r.instantaneous_amplitude.is_null() {
        drop(Box::from_raw(slice::from_raw_parts_mut(
            r.instantaneous_amplitude as *mut f64,
            nm,
        )));
    }
    if !r.instantaneous_frequency.is_null() {
        drop(Box::from_raw(slice::from_raw_parts_mut(
            r.instantaneous_frequency as *mut f64,
            nm,
        )));
    }
    if !r.marginal_spectrum.is_null() {
        drop(Box::from_raw(slice::from_raw_parts_mut(
            r.marginal_spectrum as *mut f64,
            r.n_freq_bins,
        )));
    }
    if !r.error.is_null() {
        drop(Box::from_raw(r.error));
    }
    if !r.error_msg.is_null() {
        drop(std::ffi::CString::from_raw(r.error_msg));
    }
}

// ---------------------------------------------------------------------------
// Streaming decomposition (opaque handle)
// ---------------------------------------------------------------------------

/// C-compatible per-chunk streaming result.
#[repr(C)]
pub struct CChunkResult {
    /// Array of `n_imfs` pointers, each to `n_samples` f64, or null on error.
    pub imfs: *const *const f64,
    /// Number of IMFs.
    pub n_imfs: usize,
    /// Samples per IMF / residue.
    pub n_samples: usize,
    /// Residue (`n_samples` f64), or null on error.
    pub residue: *const f64,
    /// Spectral entropy metric for the chunk.
    pub spectral_entropy: f64,
    /// Stationarity score metric for the chunk.
    pub stationarity_score: f64,
    /// Coefficient of variation of extrema spacing.
    pub extrema_spacing_cv: f64,
    /// Pointer to an error code (1 = error), or null on success.
    pub error: *mut i32,
    /// Null-terminated error message, or null on success.
    pub error_msg: *mut std::os::raw::c_char,
}

fn chunk_err(msg: &str) -> *mut CChunkResult {
    let error = Box::into_raw(Box::new(1i32));
    let error_msg = std::ffi::CString::new(msg).unwrap_or_default().into_raw();
    Box::into_raw(Box::new(CChunkResult {
        imfs: std::ptr::null(),
        n_imfs: 0,
        n_samples: 0,
        residue: std::ptr::null(),
        spectral_entropy: 0.0,
        stationarity_score: 0.0,
        extrema_spacing_cv: 0.0,
        error,
        error_msg,
    }))
}

/// Create a streaming decomposer handle.
///
/// `ar_order` 0 defaults to 3; `buffer_size` 0 defaults to 4096.
///
/// # Safety
/// `config` must point to a valid `CEmdConfig`. Caller must call
/// `ferromode_streaming_free` on the returned handle.
#[no_mangle]
pub unsafe extern "C" fn ferromode_streaming_new(
    config: *const CEmdConfig,
    buffer_size: usize,
    ar_order: usize,
) -> *mut crate::adapters::streaming::StreamingDecomposer {
    if config.is_null() {
        return std::ptr::null_mut();
    }
    let emd_config = c_emd_config_to_rust(&*config);
    let order = if ar_order == 0 { 3 } else { ar_order };
    let predictor = match crate::adapters::streaming::ArModel::new(order) {
        Ok(m) => Box::new(m),
        Err(_) => return std::ptr::null_mut(),
    };
    let buf = if buffer_size == 0 { 4096 } else { buffer_size };
    match crate::adapters::streaming::StreamingDecomposer::new(emd_config, predictor, buf) {
        Ok(d) => Box::into_raw(Box::new(d)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Decompose one chunk through a streaming handle.
///
/// # Safety
/// `handle` must come from `ferromode_streaming_new`; `chunk` must point to
/// `len` valid f64. Caller must free the result with `ferromode_free_chunk_result`.
#[no_mangle]
pub unsafe extern "C" fn ferromode_streaming_decompose_chunk(
    handle: *mut crate::adapters::streaming::StreamingDecomposer,
    chunk: *const f64,
    len: usize,
) -> *mut CChunkResult {
    if handle.is_null() || chunk.is_null() || len == 0 {
        return chunk_err("invalid input: null handle/chunk or zero length");
    }
    let data = slice::from_raw_parts(chunk, len).to_vec();
    let signal = match crate::types::Signal::from_slice(&data) {
        Ok(s) => s,
        Err(e) => return chunk_err(&e.to_string()),
    };
    let decomposer = &mut *handle;
    match decomposer.decompose_chunk(&signal) {
        Ok(res) => {
            let n_imfs = res.imfs.len();
            let n_samples =
                if n_imfs > 0 { res.imfs[0].len() } else { res.remainder.len() };
            let mut imf_ptrs: Vec<*const f64> = Vec::with_capacity(n_imfs);
            for imf in &res.imfs {
                imf_ptrs.push(Box::into_raw(imf.clone().into_boxed_slice()) as *const f64);
            }
            let residue = Box::into_raw(res.remainder.clone().into_boxed_slice()) as *const f64;
            Box::into_raw(Box::new(CChunkResult {
                imfs: Box::into_raw(imf_ptrs.into_boxed_slice()) as *const *const f64,
                n_imfs,
                n_samples,
                residue,
                spectral_entropy: res.metrics.spectral_entropy,
                stationarity_score: res.metrics.stationarity_score,
                extrema_spacing_cv: res.metrics.extrema_spacing_cv,
                error: std::ptr::null_mut(),
                error_msg: std::ptr::null_mut(),
            }))
        }
        Err(e) => chunk_err(&e.to_string()),
    }
}

/// Reset a streaming handle's state.
///
/// # Safety
/// `handle` must come from `ferromode_streaming_new`.
#[no_mangle]
pub unsafe extern "C" fn ferromode_streaming_reset(
    handle: *mut crate::adapters::streaming::StreamingDecomposer,
) {
    if !handle.is_null() {
        (*handle).reset();
    }
}

/// Free a streaming handle.
///
/// # Safety
/// `handle` must come from `ferromode_streaming_new` (or be null).
#[no_mangle]
pub unsafe extern "C" fn ferromode_streaming_free(
    handle: *mut crate::adapters::streaming::StreamingDecomposer,
) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Free a `CChunkResult`.
///
/// # Safety
/// `result` must come from `ferromode_streaming_decompose_chunk` (or be null).
#[no_mangle]
pub unsafe extern "C" fn ferromode_free_chunk_result(result: *mut CChunkResult) {
    if result.is_null() {
        return;
    }
    let r = Box::from_raw(result);
    if !r.imfs.is_null() {
        let ptrs = Box::from_raw(slice::from_raw_parts_mut(
            r.imfs as *mut *const f64,
            r.n_imfs,
        ));
        for &p in ptrs.iter() {
            if !p.is_null() {
                drop(Box::from_raw(slice::from_raw_parts_mut(p as *mut f64, r.n_samples)));
            }
        }
    }
    if !r.residue.is_null() {
        drop(Box::from_raw(slice::from_raw_parts_mut(r.residue as *mut f64, r.n_samples)));
    }
    if !r.error.is_null() {
        drop(Box::from_raw(r.error));
    }
    if !r.error_msg.is_null() {
        drop(std::ffi::CString::from_raw(r.error_msg));
    }
}

// ---------------------------------------------------------------------------
// Differentiable EMD (opaque forward context + placeholder backward)
// ---------------------------------------------------------------------------

/// Run the differentiable EMD forward pass, returning an opaque context handle.
///
/// # Safety
/// `signal` must point to `len` valid f64 and `config` to a valid `CEmdConfig`.
/// Caller must call `ferromode_diff_free` on the returned handle.
#[no_mangle]
pub unsafe extern "C" fn ferromode_diff_forward(
    signal: *const f64,
    len: usize,
    config: *const CEmdConfig,
) -> *mut crate::ml::differentiable::ImplicitEmdContext {
    if signal.is_null() || config.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    let s = slice::from_raw_parts(signal, len);
    let cfg = c_emd_config_to_rust(&*config);
    let diff = crate::ml::differentiable::DifferentiableEmd::new(cfg);
    match diff.forward(s) {
        Ok(ctx) => Box::into_raw(Box::new(ctx)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Number of IMFs in a forward context.
///
/// # Safety
/// `ctx` must come from `ferromode_diff_forward`.
#[no_mangle]
pub unsafe extern "C" fn ferromode_diff_n_imfs(
    ctx: *const crate::ml::differentiable::ImplicitEmdContext,
) -> usize {
    if ctx.is_null() {
        return 0;
    }
    (*ctx).imfs.len()
}

/// Number of samples per IMF in a forward context.
///
/// # Safety
/// `ctx` must come from `ferromode_diff_forward`.
#[no_mangle]
pub unsafe extern "C" fn ferromode_diff_n_samples(
    ctx: *const crate::ml::differentiable::ImplicitEmdContext,
) -> usize {
    if ctx.is_null() {
        return 0;
    }
    (*ctx).signal.len()
}

/// Pointer to IMF `index` (valid while `ctx` lives).
///
/// # Safety
/// `ctx` must come from `ferromode_diff_forward` and `index` be in range.
#[no_mangle]
pub unsafe extern "C" fn ferromode_diff_imf_ptr(
    ctx: *const crate::ml::differentiable::ImplicitEmdContext,
    index: usize,
) -> *const f64 {
    if ctx.is_null() {
        return std::ptr::null();
    }
    let ctx = &*ctx;
    if index >= ctx.imfs.len() {
        return std::ptr::null();
    }
    ctx.imfs[index].as_ptr()
}

/// Pointer to the residue (valid while `ctx` lives).
///
/// # Safety
/// `ctx` must come from `ferromode_diff_forward`.
#[no_mangle]
pub unsafe extern "C" fn ferromode_diff_residue_ptr(
    ctx: *const crate::ml::differentiable::ImplicitEmdContext,
) -> *const f64 {
    if ctx.is_null() {
        return std::ptr::null();
    }
    (*ctx).residue.as_ptr()
}

/// Sifting iterations recorded for IMF `index`.
///
/// # Safety
/// `ctx` must come from `ferromode_diff_forward`.
#[no_mangle]
pub unsafe extern "C" fn ferromode_diff_num_sifts(
    ctx: *const crate::ml::differentiable::ImplicitEmdContext,
    index: usize,
) -> usize {
    if ctx.is_null() {
        return 0;
    }
    let ctx = &*ctx;
    if index >= ctx.num_sifts.len() {
        return 0;
    }
    ctx.num_sifts[index]
}

/// Maximum absolute reconstruction error of the forward context.
///
/// # Safety
/// `ctx` must come from `ferromode_diff_forward`.
#[no_mangle]
pub unsafe extern "C" fn ferromode_diff_reconstruction_error(
    ctx: *const crate::ml::differentiable::ImplicitEmdContext,
) -> f64 {
    if ctx.is_null() {
        return f64::NAN;
    }
    (*ctx).reconstruction_error()
}

/// Differentiable EMD backward pass (placeholder: averages upstream gradients,
/// matching the Python binding pending full implicit differentiation).
///
/// Writes `len` gradient values into `out`. Returns 0 on success, -1 on error.
///
/// # Safety
/// `grad_imfs_flat` must point to `n_imfs * len` f64 (row-major), `out` to `len`.
#[no_mangle]
pub unsafe extern "C" fn ferromode_diff_backward(
    grad_imfs_flat: *const f64,
    n_imfs: usize,
    len: usize,
    out: *mut f64,
) -> i32 {
    if grad_imfs_flat.is_null() || out.is_null() || n_imfs == 0 || len == 0 {
        return -1;
    }
    let grads = slice::from_raw_parts(grad_imfs_flat, n_imfs * len);
    let out_slice = slice::from_raw_parts_mut(out, len);
    for v in out_slice.iter_mut() {
        *v = 0.0;
    }
    for i in 0..n_imfs {
        for j in 0..len {
            out_slice[j] += grads[i * len + j];
        }
    }
    let n = n_imfs as f64;
    for v in out_slice.iter_mut() {
        *v /= n;
    }
    0
}

/// Free a differentiable forward context.
///
/// # Safety
/// `ctx` must come from `ferromode_diff_forward` (or be null).
#[no_mangle]
pub unsafe extern "C" fn ferromode_diff_free(
    ctx: *mut crate::ml::differentiable::ImplicitEmdContext,
) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_ffi_emd_basic() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = CEmdConfig {
            max_imfs: 0,
            sd_threshold: 0.2,
            s_number: 5,
            max_sifting_iterations: 100,
            boundary_condition: 0,
            ..Default::default()
        };

        unsafe {
            let result = ferromode_emd(signal.as_ptr(), n, &config);
            assert!(!result.is_null());

            let has_error = ferromode_result_has_error(result);
            assert_eq!(has_error, 0, "EMD should not have an error");

            let n_imfs = ferromode_result_n_imfs(result);
            assert!(n_imfs >= 1, "Should produce at least 1 IMF");

            let n_samples = ferromode_result_n_samples(result);
            assert_eq!(n_samples, n);

            let algo = ferromode_result_algorithm(result);
            assert_eq!(algo, 0); // EMD = 0

            let elapsed = ferromode_result_elapsed_ms(result);
            assert!(elapsed >= 0.0);

            // Test IMF access
            let imf0_ptr = ferromode_result_imf_ptr(result, 0);
            assert!(!imf0_ptr.is_null());

            // Test residue access
            let residue_ptr = ferromode_result_residue_ptr(result);
            assert!(!residue_ptr.is_null());

            // Test reconstruction
            let mut reconstructed = vec![0.0f64; n];
            let recon_result = ferromode_reconstruct(result, reconstructed.as_mut_ptr(), n);
            assert_eq!(recon_result, 0);

            // Check reconstruction accuracy
            let max_error: f64 = signal
                .iter()
                .zip(reconstructed.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(max_error < 1e-6, "Reconstruction error should be small: {:.2e}", max_error);

            ferromode_free_result(result);
        }
    }

    #[test]
    fn test_ffi_vmd_basic() {
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let config =
            CVmdConfig { n_modes: 2, alpha: 2000.0, tau: 0.0, tol: 1e-7, max_iterations: 500 };

        unsafe {
            let result = ferromode_vmd(signal.as_ptr(), n, &config);
            assert!(!result.is_null());

            let has_error = ferromode_result_has_error(result);
            assert_eq!(has_error, 0, "VMD should not have an error");

            let n_imfs = ferromode_result_n_imfs(result);
            assert_eq!(n_imfs, 2);

            let algo = ferromode_result_algorithm(result);
            assert_eq!(algo, 7); // VMD = 7

            ferromode_free_result(result);
        }
    }

    #[test]
    fn test_ffi_null_signal_returns_error() {
        let config = CEmdConfig {
            max_imfs: 0,
            sd_threshold: 0.2,
            s_number: 5,
            max_sifting_iterations: 100,
            boundary_condition: 0,
            ..Default::default()
        };

        unsafe {
            let result = ferromode_emd(std::ptr::null(), 100, &config);
            assert!(!result.is_null());
            assert_eq!(ferromode_result_has_error(result), 1);
            ferromode_free_result(result);
        }
    }

    #[test]
    fn test_ffi_copy_imf() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = CEmdConfig {
            max_imfs: 0,
            sd_threshold: 0.2,
            s_number: 5,
            max_sifting_iterations: 100,
            boundary_condition: 0,
            ..Default::default()
        };

        unsafe {
            let result = ferromode_emd(signal.as_ptr(), n, &config);
            assert!(!result.is_null());

            let n_imfs = ferromode_result_n_imfs(result);
            assert!(n_imfs >= 1);

            let mut imf_data = vec![0.0f64; n];
            let copy_result = ferromode_copy_imf(result, 0, imf_data.as_mut_ptr(), n);
            assert_eq!(copy_result, 0);

            // Verify data matches direct pointer access
            let imf0_ptr = ferromode_result_imf_ptr(result, 0);
            let direct = slice::from_raw_parts(imf0_ptr, n);
            for (a, b) in imf_data.iter().zip(direct.iter()) {
                assert!((a - b).abs() < 1e-15);
            }

            ferromode_free_result(result);
        }
    }

    #[test]
    fn test_ffi_copy_residue() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = CEmdConfig {
            max_imfs: 0,
            sd_threshold: 0.2,
            s_number: 5,
            max_sifting_iterations: 100,
            boundary_condition: 0,
            ..Default::default()
        };

        unsafe {
            let result = ferromode_emd(signal.as_ptr(), n, &config);
            assert!(!result.is_null());

            let mut residue_data = vec![0.0f64; n];
            let copy_result = ferromode_copy_residue(result, residue_data.as_mut_ptr(), n);
            assert_eq!(copy_result, 0);

            // Verify data matches direct pointer access
            let residue_ptr = ferromode_result_residue_ptr(result);
            let direct = slice::from_raw_parts(residue_ptr, n);
            for (a, b) in residue_data.iter().zip(direct.iter()) {
                assert!((a - b).abs() < 1e-15);
            }

            ferromode_free_result(result);
        }
    }

    #[test]
    fn test_ffi_eemd_basic() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let ens_config =
            CEnsembleConfig { num_ensembles: 5, noise_std: 0.2, seed: 42, use_seed: 1 };
        let emd_config = CEmdConfig {
            max_imfs: 0,
            sd_threshold: 0.2,
            s_number: 5,
            max_sifting_iterations: 100,
            boundary_condition: 0,
            ..Default::default()
        };

        unsafe {
            let result = ferromode_eemd(signal.as_ptr(), n, &ens_config, &emd_config);
            assert!(!result.is_null());
            assert_eq!(ferromode_result_has_error(result), 0);
            assert!(ferromode_result_n_imfs(result) >= 1);
            assert_eq!(ferromode_result_algorithm(result), 1); // EEMD = 1
            ferromode_free_result(result);
        }
    }

    #[test]
    fn test_ffi_free_result_null_is_safe() {
        unsafe {
            ferromode_free_result(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_ffi_hilbert_basic() {
        let n = 128usize;
        let imf0: Vec<f64> = (0..n).map(|i| (2.0 * PI * 8.0 * i as f64 / n as f64).sin()).collect();
        let imf1: Vec<f64> = (0..n).map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).sin()).collect();
        let mut flat = imf0.clone();
        flat.extend_from_slice(&imf1);
        unsafe {
            let h = ferromode_hilbert(flat.as_ptr(), 2, n, 1.0);
            assert!(!h.is_null());
            assert!((*h).error.is_null(), "hilbert errored");
            assert_eq!((*h).n_imfs, 2);
            assert_eq!((*h).n_samples, n);
            assert!(!(*h).instantaneous_amplitude.is_null());
            assert!(!(*h).marginal_spectrum.is_null());
            ferromode_free_hilbert(h);
            // null-safety
            ferromode_free_hilbert(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_ffi_streaming_basic() {
        let cfg = CEmdConfig { max_imfs: 4, ..Default::default() };
        let chunk: Vec<f64> =
            (0..256).map(|i| (2.0 * PI * 5.0 * i as f64 / 256.0).sin()).collect();
        unsafe {
            let handle = ferromode_streaming_new(&cfg, 2048, 3);
            assert!(!handle.is_null());
            let res = ferromode_streaming_decompose_chunk(handle, chunk.as_ptr(), chunk.len());
            assert!(!res.is_null());
            assert!((*res).error.is_null(), "streaming chunk errored");
            assert!((*res).n_imfs >= 1);
            ferromode_free_chunk_result(res);
            ferromode_streaming_reset(handle);
            ferromode_streaming_free(handle);
            ferromode_streaming_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_ffi_diff_forward_and_backward() {
        let n = 200usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin() + 0.4 * (i as f64 / n as f64))
            .collect();
        let cfg = CEmdConfig { max_imfs: 4, ..Default::default() };
        unsafe {
            let ctx = ferromode_diff_forward(signal.as_ptr(), n, &cfg);
            assert!(!ctx.is_null());
            let n_imfs = ferromode_diff_n_imfs(ctx);
            assert!(n_imfs >= 1);
            assert_eq!(ferromode_diff_n_samples(ctx), n);
            assert!(!ferromode_diff_imf_ptr(ctx, 0).is_null());
            assert!(!ferromode_diff_residue_ptr(ctx).is_null());
            assert!(ferromode_diff_reconstruction_error(ctx).is_finite());

            // backward: averaging placeholder
            let grads: Vec<f64> = vec![1.0; n_imfs * n];
            let mut out = vec![0.0f64; n];
            let rc = ferromode_diff_backward(grads.as_ptr(), n_imfs, n, out.as_mut_ptr());
            assert_eq!(rc, 0);
            assert!((out[0] - 1.0).abs() < 1e-12); // mean of all-ones grads = 1

            ferromode_diff_free(ctx);
            ferromode_diff_free(std::ptr::null_mut());
        }
    }
}
