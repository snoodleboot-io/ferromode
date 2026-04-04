//! R bindings for Ferromode via extendr.
//!
//! Pure wrapper — no algorithm logic, only marshalling between R and Rust.
//!
//! Usage in R:
//! ```r
//! library(ferromodeR)
//! signal <- sin(seq(0, 2*pi, length.out = 200))
//! result <- emd(signal, list(max_imfs = 0, sd_threshold = 0.2, s_number = 5, max_sifting_iterations = 100))
//! result$imfs  # list of IMF vectors
//! result$residue
//! ```

use extendr_api::prelude::*;
use ferromode::algorithms::ceemd::ceemd;
use ferromode::algorithms::ceemdan::ceemdan;
use ferromode::algorithms::eemd::{eemd, EnsembleConfig};
use ferromode::algorithms::emd::{emd, EmdConfig};
use ferromode::algorithms::iceemdan::iceemdan;
use ferromode::algorithms::vmd::{vmd, VmdConfig};
use ferromode::multivariate::memd::{memd, MemdConfig};
use ferromode::multivariate::namemd::{namemd, NaMemdConfig};
use ferromode::types::{AlgorithmType, DecompositionResult};

// ---------------------------------------------------------------------------
// Marshalling
// ---------------------------------------------------------------------------

/// Convert an R numeric vector to a Rust Vec<f64>.
fn r_to_slice(robj: Robj) -> Result<Vec<f64>, String> {
    let doubles = Doubles::try_from(robj).map_err(|e| format!("expected numeric vector: {e}"))?;
    let vec: Vec<f64> = doubles.iter().collect();

    for (i, &v) in vec.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("non-finite value at index {i}"));
        }
    }

    Ok(vec)
}

/// Convert a Rust slice to an R numeric vector.
fn slice_to_r(data: &[f64]) -> Robj {
    r!(data)
}

/// Convert an R numeric matrix (column-major) to Vec<Vec<f64>> (channels).
fn r_matrix_to_vecs(robj: Robj) -> Result<Vec<Vec<f64>>, String> {
    let doubles =
        Doubles::try_from(robj.clone()).map_err(|e| format!("expected numeric matrix: {e}"))?;
    let values: Vec<f64> = doubles.iter().collect();

    // Get dimensions from R object
    let dims = robj.dim().ok_or("expected matrix with dims attribute")?;
    let nrows =
        dims.elt(0).map_err(|_| "invalid nrows")?.as_integer().ok_or("nrows not integer")? as usize;
    let ncols =
        dims.elt(1).map_err(|_| "invalid ncols")?.as_integer().ok_or("ncols not integer")? as usize;

    // R stores matrices column-major: each column is a channel
    let mut channels = Vec::with_capacity(ncols);
    for ch in 0..ncols {
        let mut channel = Vec::with_capacity(nrows);
        for row in 0..nrows {
            channel.push(values[ch * nrows + row]);
        }
        channels.push(channel);
    }

    for (ch_i, ch) in channels.iter().enumerate() {
        for (s_i, &v) in ch.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("non-finite value at channel {ch_i}, sample {s_i}"));
            }
        }
    }

    Ok(channels)
}

/// Convert a DecompositionResult to an R named list with S3 class.
fn result_to_r(result: &DecompositionResult) -> Result<List, String> {
    let n_imfs = result.imfs.imfs.len();
    let n_samples = if n_imfs > 0 {
        result.imfs.imfs[0].len()
    } else if !result.imfs.residue.is_empty() {
        result.imfs.residue.len()
    } else {
        0
    };

    // Build list of IMFs
    let mut imf_list = List::from_values(result.imfs.imfs.iter().map(|imf| slice_to_r(imf)));

    // Build result list
    let algo_name = match result.algorithm {
        AlgorithmType::EMD => "emd",
        AlgorithmType::EEMD => "eemd",
        AlgorithmType::CEEMD => "ceemd",
        AlgorithmType::CEEMDAN => "ceemdan",
        AlgorithmType::ICEEMDAN => "iceemdan",
        AlgorithmType::MEMD => "memd",
        AlgorithmType::NAMEMD => "namemd",
        AlgorithmType::VMD => "vmd",
    };

    let result_list = list!(
        imfs := imf_list,
        residue := slice_to_r(&result.imfs.residue),
        algorithm := algo_name,
        elapsed_ms := result.elapsed.as_secs_f64() * 1000.0,
        n_siftings := result.n_siftings as i32,
        n_imfs := n_imfs as i32,
        n_samples := n_samples as i32,
    );

    Ok(result_list)
}

/// Convert a multivariate DecompositionResult to R, reconstructing channel structure.
fn multivariate_result_to_r(
    result: &DecompositionResult,
    n_channels: usize,
) -> Result<List, String> {
    let n_imfs_total = result.imfs.imfs.len();
    if n_channels == 0 {
        return Err("n_channels must be > 0".to_string());
    }
    let n_imfs_per_channel = n_imfs_total / n_channels;
    let n_samples = if n_imfs_per_channel > 0 {
        result.imfs.imfs[0].len()
    } else if !result.imfs.residue.is_empty() {
        result.imfs.residue.len() / n_channels
    } else {
        0
    };

    // Build per-channel IMF lists
    let mut channel_imfs = List::from_values((0..n_channels).map(|ch| {
        let imfs_for_ch: Vec<Robj> = (0..n_imfs_per_channel)
            .map(|imf_idx| {
                let flat_idx = ch * n_imfs_per_channel + imf_idx;
                if flat_idx < result.imfs.imfs.len() {
                    slice_to_r(&result.imfs.imfs[flat_idx])
                } else {
                    Robj::from(())
                }
            })
            .collect();
        List::from_values(imfs_for_ch)
    }));

    // Build per-channel residues
    let channel_residues: Vec<Robj> = (0..n_channels)
        .map(|ch| {
            let start = ch * n_samples;
            let end = start + n_samples;
            if end <= result.imfs.residue.len() {
                slice_to_r(&result.imfs.residue[start..end])
            } else {
                Robj::from(())
            }
        })
        .collect();
    let residue_list = List::from_values(channel_residues);

    let algo_name = match result.algorithm {
        AlgorithmType::MEMD => "memd",
        AlgorithmType::NAMEMD => "namemd",
        _ => "unknown",
    };

    let result_list = list!(
        imfs := channel_imfs,
        residue := residue_list,
        algorithm := algo_name,
        elapsed_ms := result.elapsed.as_secs_f64() * 1000.0,
        n_siftings := result.n_siftings as i32,
        n_channels := n_channels as i32,
        n_imfs_per_channel := n_imfs_per_channel as i32,
        n_samples := n_samples as i32,
    );

    Ok(result_list)
}

/// Extract a config value from an R list, with a default fallback.
fn get_config_field<T: FromRobj>(config: &List, key: &str, default: T) -> T {
    match config.find(key) {
        Some((_, val)) => T::from_robj(val).unwrap_or(default),
        None => default,
    }
}

// ---------------------------------------------------------------------------
// Univariate algorithm wrappers
// ---------------------------------------------------------------------------

/// Perform Empirical Mode Decomposition on a signal.
///
/// @param signal Numeric vector of signal values.
/// @param config Named list with optional keys:
///   - `max_imfs`: Maximum IMFs to extract (0 = auto). Default: 0.
///   - `sd_threshold`: SD stopping criterion. Default: 0.2.
///   - `s_number`: S-number for stopping. Default: 5.
///   - `max_sifting_iterations`: Max sifting iterations. Default: 100.
///   - `boundary_condition`: 0=MirrorEven, 1=MirrorOdd, 2=Periodic, 3=Slope, 4=ARModel, 5=CharacteristicWave, 6=WaveformMatching. Default: 0.
/// @return Named list with S3 class 'ferromode_result' containing:
///   - `imfs`: List of IMF vectors.
///   - `residue`: Residue vector.
///   - `algorithm`: Algorithm name string.
///   - `elapsed_ms`: Execution time in milliseconds.
///   - `n_siftings`: Number of sifting iterations.
///   - `n_imfs`: Number of IMFs extracted.
///   - `n_samples`: Number of samples per IMF.
/// @export
#[extendr]
fn emd(signal: Robj, config: List) -> Result<List, String> {
    let signal_vec = r_to_slice(signal)?;
    if signal_vec.len() < 3 {
        return Err("signal must have at least 3 samples".to_string());
    }

    let boundary_val: i32 = get_config_field(&config, "boundary_condition", 0i32);
    let boundary = match boundary_val {
        0 => ferromode::boundary::BoundaryConditionType::MirrorEven,
        1 => ferromode::boundary::BoundaryConditionType::MirrorOdd,
        2 => ferromode::boundary::BoundaryConditionType::Periodic,
        3 => ferromode::boundary::BoundaryConditionType::Slope,
        4 => ferromode::boundary::BoundaryConditionType::ARModel,
        5 => ferromode::boundary::BoundaryConditionType::CharacteristicWave,
        6 => ferromode::boundary::BoundaryConditionType::WaveformMatching,
        _ => ferromode::boundary::BoundaryConditionType::MirrorEven,
    };

    let emd_config = EmdConfig {
        sifting_config: ferromode::sifting::SiftingConfig {
            sd_threshold: get_config_field(&config, "sd_threshold", 0.2),
            s_number: get_config_field(&config, "s_number", 5usize),
            max_sifting_iterations: get_config_field(&config, "max_sifting_iterations", 100usize),
            ..Default::default()
        },
        max_imfs: get_config_field(&config, "max_imfs", 0usize),
        boundary_condition: boundary,
        intermittency: None,
        reconstruction_tolerance: 1e-12,
        validate_reconstruction: false,
    };

    let result = emd(&signal_vec, &emd_config).map_err(|e| e.to_string())?;
    result_to_r(&result)
}

/// Perform Ensemble EMD on a signal.
///
/// @param signal Numeric vector of signal values.
/// @param ensemble_config Named list with keys:
///   - `num_ensembles`: Number of ensemble trials. Default: 100.
///   - `noise_std`: Noise standard deviation fraction. Default: 0.2.
///   - `seed`: Optional seed for reproducibility. Default: NULL (random).
/// @param emd_config Optional EMD config list (see `emd()`). Default: standard settings.
/// @return Named list with S3 class 'ferromode_result'.
/// @export
#[extendr]
fn eemd(signal: Robj, ensemble_config: List, emd_config: Option<List>) -> Result<List, String> {
    let signal_vec = r_to_slice(signal)?;

    let ens_cfg = EnsembleConfig {
        num_ensembles: get_config_field(&ensemble_config, "num_ensembles", 100usize),
        noise_std: get_config_field(&ensemble_config, "noise_std", 0.2),
        seed: {
            let seed_robj = ensemble_config.find("seed").map(|(_, v)| v);
            match seed_robj {
                Some(v) if !v.is_null() => {
                    let s: Option<i64> = FromRobj::from_robj(v).ok();
                    s.map(|x| x as u64)
                }
                _ => None,
            }
        },
    };

    let inner_config = emd_config.unwrap_or_else(|| list!());
    let inner_emd = EmdConfig {
        sifting_config: ferromode::sifting::SiftingConfig {
            sd_threshold: get_config_field(&inner_config, "sd_threshold", 0.2),
            s_number: get_config_field(&inner_config, "s_number", 5usize),
            max_sifting_iterations: get_config_field(
                &inner_config,
                "max_sifting_iterations",
                100usize,
            ),
            ..Default::default()
        },
        max_imfs: get_config_field(&inner_config, "max_imfs", 0usize),
        ..Default::default()
    };

    let result = eemd(&signal_vec, &ens_cfg, &inner_emd).map_err(|e| e.to_string())?;
    result_to_r(&result)
}

/// Perform Complementary EMD on a signal.
///
/// @param signal Numeric vector of signal values.
/// @param ensemble_config Named list (same as `eemd()`).
/// @param emd_config Optional EMD config list.
/// @return Named list with S3 class 'ferromode_result'.
/// @export
#[extendr]
fn ceemd(signal: Robj, ensemble_config: List, emd_config: Option<List>) -> Result<List, String> {
    let signal_vec = r_to_slice(signal)?;

    let ens_cfg = EnsembleConfig {
        num_ensembles: get_config_field(&ensemble_config, "num_ensembles", 100usize),
        noise_std: get_config_field(&ensemble_config, "noise_std", 0.2),
        seed: {
            let seed_robj = ensemble_config.find("seed").map(|(_, v)| v);
            match seed_robj {
                Some(v) if !v.is_null() => {
                    let s: Option<i64> = FromRobj::from_robj(v).ok();
                    s.map(|x| x as u64)
                }
                _ => None,
            }
        },
    };

    let inner_config = emd_config.unwrap_or_else(|| list!());
    let inner_emd = EmdConfig {
        sifting_config: ferromode::sifting::SiftingConfig {
            sd_threshold: get_config_field(&inner_config, "sd_threshold", 0.2),
            s_number: get_config_field(&inner_config, "s_number", 5usize),
            max_sifting_iterations: get_config_field(
                &inner_config,
                "max_sifting_iterations",
                100usize,
            ),
            ..Default::default()
        },
        max_imfs: get_config_field(&inner_config, "max_imfs", 0usize),
        ..Default::default()
    };

    let result = ceemd(&signal_vec, &ens_cfg, &inner_emd).map_err(|e| e.to_string())?;
    result_to_r(&result)
}

/// Perform CEEMDAN on a signal.
///
/// @param signal Numeric vector of signal values.
/// @param ensemble_config Named list (same as `eemd()`).
/// @param emd_config Optional EMD config list.
/// @return Named list with S3 class 'ferromode_result'.
/// @export
#[extendr]
fn ceemdan(signal: Robj, ensemble_config: List, emd_config: Option<List>) -> Result<List, String> {
    let signal_vec = r_to_slice(signal)?;

    let ens_cfg = EnsembleConfig {
        num_ensembles: get_config_field(&ensemble_config, "num_ensembles", 100usize),
        noise_std: get_config_field(&ensemble_config, "noise_std", 0.2),
        seed: {
            let seed_robj = ensemble_config.find("seed").map(|(_, v)| v);
            match seed_robj {
                Some(v) if !v.is_null() => {
                    let s: Option<i64> = FromRobj::from_robj(v).ok();
                    s.map(|x| x as u64)
                }
                _ => None,
            }
        },
    };

    let inner_config = emd_config.unwrap_or_else(|| list!());
    let inner_emd = EmdConfig {
        sifting_config: ferromode::sifting::SiftingConfig {
            sd_threshold: get_config_field(&inner_config, "sd_threshold", 0.2),
            s_number: get_config_field(&inner_config, "s_number", 5usize),
            max_sifting_iterations: get_config_field(
                &inner_config,
                "max_sifting_iterations",
                100usize,
            ),
            ..Default::default()
        },
        max_imfs: get_config_field(&inner_config, "max_imfs", 0usize),
        ..Default::default()
    };

    let result = ceemdan(&signal_vec, &ens_cfg, &inner_emd).map_err(|e| e.to_string())?;
    result_to_r(&result)
}

/// Perform ICEEMDAN on a signal.
///
/// @param signal Numeric vector of signal values.
/// @param ensemble_config Named list (same as `eemd()`).
/// @param emd_config Optional EMD config list.
/// @return Named list with S3 class 'ferromode_result'.
/// @export
#[extendr]
fn iceemdan(signal: Robj, ensemble_config: List, emd_config: Option<List>) -> Result<List, String> {
    let signal_vec = r_to_slice(signal)?;

    let ens_cfg = EnsembleConfig {
        num_ensembles: get_config_field(&ensemble_config, "num_ensembles", 100usize),
        noise_std: get_config_field(&ensemble_config, "noise_std", 0.2),
        seed: {
            let seed_robj = ensemble_config.find("seed").map(|(_, v)| v);
            match seed_robj {
                Some(v) if !v.is_null() => {
                    let s: Option<i64> = FromRobj::from_robj(v).ok();
                    s.map(|x| x as u64)
                }
                _ => None,
            }
        },
    };

    let inner_config = emd_config.unwrap_or_else(|| list!());
    let inner_emd = EmdConfig {
        sifting_config: ferromode::sifting::SiftingConfig {
            sd_threshold: get_config_field(&inner_config, "sd_threshold", 0.2),
            s_number: get_config_field(&inner_config, "s_number", 5usize),
            max_sifting_iterations: get_config_field(
                &inner_config,
                "max_sifting_iterations",
                100usize,
            ),
            ..Default::default()
        },
        max_imfs: get_config_field(&inner_config, "max_imfs", 0usize),
        ..Default::default()
    };

    let result = iceemdan(&signal_vec, &ens_cfg, &inner_emd).map_err(|e| e.to_string())?;
    result_to_r(&result)
}

/// Perform VMD on a signal.
///
/// @param signal Numeric vector of signal values.
/// @param config Named list with keys:
///   - `n_modes`: Number of modes to extract. Default: 3.
///   - `alpha`: Bandwidth penalty. Default: 2000.0.
///   - `tau`: Dual ascent step size. Default: 0.0.
///   - `tol`: Convergence tolerance. Default: 1e-7.
///   - `max_iterations`: Max ADMM iterations. Default: 500.
/// @return Named list with S3 class 'ferromode_result'.
/// @export
#[extendr]
fn vmd(signal: Robj, config: List) -> Result<List, String> {
    let signal_vec = r_to_slice(signal)?;

    let vmd_config = VmdConfig {
        n_modes: get_config_field(&config, "n_modes", 3usize),
        alpha: get_config_field(&config, "alpha", 2000.0),
        tau: get_config_field(&config, "tau", 0.0),
        tol: get_config_field(&config, "tol", 1e-7),
        max_iterations: get_config_field(&config, "max_iterations", 500usize),
    };

    let result = vmd(&signal_vec, &vmd_config).map_err(|e| e.to_string())?;
    result_to_r(&result)
}

// ---------------------------------------------------------------------------
// Multivariate algorithm wrappers
// ---------------------------------------------------------------------------

/// Perform MEMD on a multivariate signal.
///
/// @param signal Numeric matrix (n_samples x n_channels). Each column is a channel.
/// @param config Named list with keys:
///   - `num_directions`: Number of direction vectors. Default: 16.
///   - `direction_seed`: Seed for direction generation. Default: 42.
///   - `max_imfs`: Maximum IMFs per channel. Default: 0 (auto).
///   - `sd_threshold`: SD stopping criterion. Default: 0.2.
///   - `s_number`: S-number for stopping. Default: 3.
///   - `max_sifting_iterations`: Max sifting iterations. Default: 50.
/// @return Named list with per-channel structure.
/// @export
#[extendr]
fn memd(signal: Robj, config: List) -> Result<List, String> {
    let channels = r_matrix_to_vecs(signal)?;
    let n_channels = channels.len();
    let n_samples = if n_channels > 0 { channels[0].len() } else { 0 };

    if n_channels < 2 {
        return Err("MEMD requires at least 2 channels".to_string());
    }
    if n_samples < 3 {
        return Err("each channel must have at least 3 samples".to_string());
    }

    let dir_config = ferromode::multivariate::direction_sampling::DirectionConfig::new(
        get_config_field(&config, "num_directions", 16usize),
    )
    .with_seed(get_config_field(&config, "direction_seed", 42u64));

    let sifting_config = ferromode::sifting::SiftingConfig {
        sd_threshold: get_config_field(&config, "sd_threshold", 0.2),
        s_number: get_config_field(&config, "s_number", 3usize),
        max_sifting_iterations: get_config_field(&config, "max_sifting_iterations", 50usize),
        ..Default::default()
    };

    let memd_config = MemdConfig::new(dir_config, sifting_config)
        .with_max_imfs(get_config_field(&config, "max_imfs", 0usize));

    let result = memd(&channels, &memd_config).map_err(|e| e.to_string())?;
    multivariate_result_to_r(&result, n_channels)
}

/// Perform NA-MEMD on a multivariate signal.
///
/// @param signal Numeric matrix (n_samples x n_channels). Each column is a channel.
/// @param config Named list with keys:
///   - `base_*`: Same as MEMD config (num_directions, direction_seed, etc.)
///   - `n_noise_channels`: Number of noise channels. Default: 2.
///   - `noise_std`: Noise std fraction. Default: 0.1.
///   - `seed`: Optional seed. Default: NULL.
/// @return Named list with per-channel structure.
/// @export
#[extendr]
fn namemd(signal: Robj, config: List) -> Result<List, String> {
    let channels = r_matrix_to_vecs(signal)?;
    let n_channels = channels.len();
    let n_samples = if n_channels > 0 { channels[0].len() } else { 0 };

    if n_channels < 2 {
        return Err("NA-MEMD requires at least 2 channels".to_string());
    }
    if n_samples < 3 {
        return Err("each channel must have at least 3 samples".to_string());
    }

    let dir_config = ferromode::multivariate::direction_sampling::DirectionConfig::new(
        get_config_field(&config, "num_directions", 16usize),
    )
    .with_seed(get_config_field(&config, "direction_seed", 42u64));

    let sifting_config = ferromode::sifting::SiftingConfig {
        sd_threshold: get_config_field(&config, "sd_threshold", 0.2),
        s_number: get_config_field(&config, "s_number", 3usize),
        max_sifting_iterations: get_config_field(&config, "max_sifting_iterations", 50usize),
        ..Default::default()
    };

    let base_memd_config = MemdConfig::new(dir_config, sifting_config)
        .with_max_imfs(get_config_field(&config, "max_imfs", 0usize));

    let mut na_config = NaMemdConfig::new(base_memd_config)
        .with_noise_channels(get_config_field(&config, "n_noise_channels", 2usize))
        .with_noise_std(get_config_field(&config, "noise_std", 0.1));

    let seed_robj = config.find("seed").map(|(_, v)| v);
    if let Some(v) = seed_robj {
        if !v.is_null() {
            if let Ok(s) = i64::from_robj(v) {
                na_config = na_config.with_seed(s as u64);
            }
        }
    }

    let result = namemd(&channels, &na_config).map_err(|e| e.to_string())?;
    multivariate_result_to_r(&result, n_channels)
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Return the version of the Ferromode R bindings.
/// @export
#[extendr]
fn ferromode_r_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Reconstruct a signal from IMFs and residue.
///
/// @param result A ferromode result list (from any decomposition function).
/// @return Numeric vector of the reconstructed signal.
/// @export
#[extendr]
fn reconstruct(result: List) -> Result<Robj, String> {
    let imfs_robj = result.find("imfs").map(|(_, v)| v).ok_or("result missing 'imfs' field")?;
    let residue_robj =
        result.find("residue").map(|(_, v)| v).ok_or("result missing 'residue' field")?;

    let imfs_list = List::try_from(imfs_robj).map_err(|_| "'imfs' is not a list")?;
    let residue = Doubles::try_from(residue_robj).map_err(|_| "'residue' is not numeric")?;

    let n_samples = residue.len();
    let mut reconstructed = vec![0.0f64; n_samples];

    for i in 0..n_samples {
        reconstructed[i] = residue.iter().nth(i).unwrap_or(0.0);
    }

    for imf_robj in imfs_list.values() {
        let imf = Doubles::try_from(imf_robj.clone()).map_err(|_| "IMF is not numeric")?;
        for (i, val) in imf.iter().enumerate().take(n_samples) {
            reconstructed[i] += val;
        }
    }

    Ok(slice_to_r(&reconstructed))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

extendr_module! {
    mod ferromode_r;
    fn ferromode_r_version;
    fn emd;
    fn eemd;
    fn ceemd;
    fn ceemdan;
    fn iceemdan;
    fn memd;
    fn namemd;
    fn vmd;
    fn reconstruct;
}
