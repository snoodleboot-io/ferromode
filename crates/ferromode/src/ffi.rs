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
use std::ffi::c_void;
use std::slice;

// ---------------------------------------------------------------------------
// C-compatible config structs
// ---------------------------------------------------------------------------

/// C-compatible EMD configuration.
#[repr(C)]
pub struct CEmdConfig {
    pub max_imfs: usize,
    pub sd_threshold: f64,
    pub s_number: usize,
    pub max_sifting_iterations: usize,
    pub boundary_condition: i32,
}

/// C-compatible ensemble configuration (EEMD, CEEMD, CEEMDAN, ICEEMDAN).
#[repr(C)]
pub struct CEnsembleConfig {
    pub num_ensembles: usize,
    pub noise_std: f64,
    pub seed: u64,
    pub use_seed: i32,
}

/// C-compatible MEMD configuration.
#[repr(C)]
pub struct CMemdConfig {
    pub num_directions: usize,
    pub direction_seed: u64,
    pub max_imfs: usize,
    pub sd_threshold: f64,
    pub s_number: usize,
    pub max_sifting_iterations: usize,
}

/// C-compatible NA-MEMD configuration.
#[repr(C)]
pub struct CNaMemdConfig {
    pub base: CMemdConfig,
    pub n_noise_channels: usize,
    pub noise_std: f64,
    pub seed: u64,
    pub use_seed: i32,
}

/// C-compatible VMD configuration.
#[repr(C)]
pub struct CVmdConfig {
    pub n_modes: usize,
    pub alpha: f64,
    pub tau: f64,
    pub tol: f64,
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
    pub imfs: *const *const f64,
    pub n_imfs: usize,
    pub n_samples: usize,
    pub residue: *const f64,
}

/// C-compatible decomposition result.
#[repr(C)]
pub struct CDecompositionResult {
    pub imfs: *mut CImfCollection,
    pub algorithm: i32,
    pub elapsed_ms: f64,
    pub n_siftings: usize,
    pub error: *mut i32,
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
                let ptr = imf.as_ptr();
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

fn c_emd_config_to_rust(
    config: &CEmdConfig,
    sifting_config: crate::sifting::SiftingConfig,
) -> EmdConfig {
    let boundary = match config.boundary_condition {
        0 => crate::boundary::BoundaryConditionType::MirrorEven,
        1 => crate::boundary::BoundaryConditionType::MirrorOdd,
        2 => crate::boundary::BoundaryConditionType::Periodic,
        3 => crate::boundary::BoundaryConditionType::Slope,
        4 => crate::boundary::BoundaryConditionType::ARModel,
        5 => crate::boundary::BoundaryConditionType::CharacteristicWave,
        6 => crate::boundary::BoundaryConditionType::WaveformMatching,
        _ => crate::boundary::BoundaryConditionType::MirrorEven,
    };

    EmdConfig {
        sifting_config,
        max_imfs: config.max_imfs,
        boundary_condition: boundary,
        intermittency: None,
        reconstruction_tolerance: 1e-12,
        validate_reconstruction: false,
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

    let sifting_config = crate::sifting::SiftingConfig {
        sd_threshold: rust_config.sd_threshold,
        s_number: rust_config.s_number,
        max_sifting_iterations: rust_config.max_sifting_iterations,
        ..Default::default()
    };

    let emd_config = c_emd_config_to_rust(rust_config, sifting_config);
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

    let sifting_config = crate::sifting::SiftingConfig {
        sd_threshold: rust_emd_config.sd_threshold,
        s_number: rust_emd_config.s_number,
        max_sifting_iterations: rust_emd_config.max_sifting_iterations,
        ..Default::default()
    };
    let emd_cfg = c_emd_config_to_rust(rust_emd_config, sifting_config);

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

    let sifting_config = crate::sifting::SiftingConfig {
        sd_threshold: rust_emd_config.sd_threshold,
        s_number: rust_emd_config.s_number,
        max_sifting_iterations: rust_emd_config.max_sifting_iterations,
        ..Default::default()
    };
    let emd_cfg = c_emd_config_to_rust(rust_emd_config, sifting_config);

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

    let sifting_config = crate::sifting::SiftingConfig {
        sd_threshold: rust_emd_config.sd_threshold,
        s_number: rust_emd_config.s_number,
        max_sifting_iterations: rust_emd_config.max_sifting_iterations,
        ..Default::default()
    };
    let emd_cfg = c_emd_config_to_rust(rust_emd_config, sifting_config);

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

    let sifting_config = crate::sifting::SiftingConfig {
        sd_threshold: rust_emd_config.sd_threshold,
        s_number: rust_emd_config.s_number,
        max_sifting_iterations: rust_emd_config.max_sifting_iterations,
        ..Default::default()
    };
    let emd_cfg = c_emd_config_to_rust(rust_emd_config, sifting_config);

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
}
