//! R bindings for Ferromode via extendr.
//!
//! All functions accept a plain R named list as config and return a named list
//! as result. Config fields are optional — missing fields fall back to Rust
//! defaults. Result lists always contain:
//!   - `n_imfs`: integer number of extracted IMFs
//!   - `n_samples`: integer length of the input signal
//!   - `algorithm`: character string name of the algorithm
//!   - `elapsed_ms`: numeric milliseconds taken
//!   - `imfs`: list of numeric vectors (one per IMF)
//!   - `residue`: numeric vector

// extendr's `#[extendr]` / `extendr_module!` macros generate wrappers that bind
// and use underscore-prefixed params, and the R-convention module name
// `ferromodeR` is not snake_case; neither is fixable in our source.
#![allow(clippy::used_underscore_binding)]
#![allow(non_snake_case)]

use extendr_api::prelude::*;
use ferromode::algorithms::ceemd::ceemd as rust_ceemd;
use ferromode::algorithms::ceemdan::ceemdan as rust_ceemdan;
use ferromode::algorithms::emd::{emd as rust_emd, EmdConfig};
use ferromode::algorithms::eemd::{eemd as rust_eemd, EnsembleConfig};
use ferromode::algorithms::iceemdan::iceemdan as rust_iceemdan;
use ferromode::algorithms::vmd::{vmd as rust_vmd, VmdConfig};
use ferromode::types::Signal;

// ---------------------------------------------------------------------------
// Config helpers — extract optional fields from R named list
// ---------------------------------------------------------------------------

fn get_usize(config: &List, key: &str, default: usize) -> usize {
    config
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| v.as_integer())
        .map_or(default, |n| n.max(0) as usize)
}

fn get_f64(config: &List, key: &str, default: f64) -> f64 {
    config
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| v.as_real())
        .unwrap_or(default)
}

fn get_u64_opt(config: &List, key: &str) -> Option<u64> {
    config
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| v.as_integer())
        .map(|n| n as u64)
}

// ---------------------------------------------------------------------------
// Result builder
// ---------------------------------------------------------------------------

fn build_result(
    imfs: &[Vec<f64>],
    residue: &[f64],
    algorithm: &str,
    elapsed_ms: f64,
    n_samples: usize,
) -> Result<List> {
    let imfs_robj: Vec<Robj> = imfs.iter().map(|v| r!(v.as_slice())).collect();
    Ok(list!(
        n_imfs = imfs.len() as i32,
        n_samples = n_samples as i32,
        algorithm = algorithm,
        elapsed_ms = elapsed_ms,
        imfs = List::from_values(imfs_robj),
        residue = r!(residue)
    ))
}

// ---------------------------------------------------------------------------
// EMD
// ---------------------------------------------------------------------------

/// Decompose a signal using Empirical Mode Decomposition.
#[extendr]
fn emd(signal: Vec<f64>, config: List) -> Result<List> {
    let max_imfs = get_usize(&config, "max_imfs", 0);

    let sig = Signal::from_slice(&signal)
        .map_err(|e| Error::Other(format!("Invalid signal: {e}")))?;

    let cfg = EmdConfig { max_imfs, ..Default::default() };

    let start = std::time::Instant::now();
    let result = rust_emd(sig.values(), &cfg)
        .map_err(|e| Error::Other(format!("EMD failed: {e}")))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    build_result(&result.imfs.imfs, &result.imfs.residue, "emd", elapsed_ms, signal.len())
}

// ---------------------------------------------------------------------------
// Ensemble algorithms — shared config parsing
// ---------------------------------------------------------------------------

fn parse_ensemble_config(config: &List) -> EnsembleConfig {
    EnsembleConfig {
        num_ensembles: get_usize(config, "num_ensembles", 100),
        noise_std: get_f64(config, "noise_std", 0.2),
        seed: get_u64_opt(config, "seed"),
    }
}

/// Decompose a signal using Ensemble EMD.
#[extendr]
fn eemd(signal: Vec<f64>, config: List) -> Result<List> {
    let sig = Signal::from_slice(&signal)
        .map_err(|e| Error::Other(format!("Invalid signal: {e}")))?;

    let ens_cfg = parse_ensemble_config(&config);
    let emd_cfg = EmdConfig::default();

    let start = std::time::Instant::now();
    let result = rust_eemd(sig.values(), &ens_cfg, &emd_cfg)
        .map_err(|e| Error::Other(format!("EEMD failed: {e}")))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    build_result(&result.imfs.imfs, &result.imfs.residue, "eemd", elapsed_ms, signal.len())
}

/// Decompose a signal using Complementary EEMD.
#[extendr]
fn ceemd(signal: Vec<f64>, config: List) -> Result<List> {
    let sig = Signal::from_slice(&signal)
        .map_err(|e| Error::Other(format!("Invalid signal: {e}")))?;

    let ens_cfg = parse_ensemble_config(&config);
    let emd_cfg = EmdConfig::default();

    let start = std::time::Instant::now();
    let result = rust_ceemd(sig.values(), &ens_cfg, &emd_cfg)
        .map_err(|e| Error::Other(format!("CEEMD failed: {e}")))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    build_result(&result.imfs.imfs, &result.imfs.residue, "ceemd", elapsed_ms, signal.len())
}

/// Decompose a signal using Complete EEMD with Adaptive Noise.
#[extendr]
fn ceemdan(signal: Vec<f64>, config: List) -> Result<List> {
    let sig = Signal::from_slice(&signal)
        .map_err(|e| Error::Other(format!("Invalid signal: {e}")))?;

    let ens_cfg = parse_ensemble_config(&config);
    let emd_cfg = EmdConfig::default();

    let start = std::time::Instant::now();
    let result = rust_ceemdan(sig.values(), &ens_cfg, &emd_cfg)
        .map_err(|e| Error::Other(format!("CEEMDAN failed: {e}")))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    build_result(&result.imfs.imfs, &result.imfs.residue, "ceemdan", elapsed_ms, signal.len())
}

/// Decompose a signal using Improved CEEMDAN.
#[extendr]
fn iceemdan(signal: Vec<f64>, config: List) -> Result<List> {
    let sig = Signal::from_slice(&signal)
        .map_err(|e| Error::Other(format!("Invalid signal: {e}")))?;

    let ens_cfg = parse_ensemble_config(&config);
    let emd_cfg = EmdConfig::default();

    let start = std::time::Instant::now();
    let result = rust_iceemdan(sig.values(), &ens_cfg, &emd_cfg)
        .map_err(|e| Error::Other(format!("ICEEMDAN failed: {e}")))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    build_result(&result.imfs.imfs, &result.imfs.residue, "iceemdan", elapsed_ms, signal.len())
}

// ---------------------------------------------------------------------------
// VMD
// ---------------------------------------------------------------------------

/// Decompose a signal using Variational Mode Decomposition.
#[extendr]
fn vmd(signal: Vec<f64>, config: List) -> Result<List> {
    let sig = Signal::from_slice(&signal)
        .map_err(|e| Error::Other(format!("Invalid signal: {e}")))?;

    let cfg = VmdConfig {
        n_modes: get_usize(&config, "n_modes", 3),
        alpha: get_f64(&config, "alpha", 2000.0),
        tol: get_f64(&config, "tol", 1e-7),
        max_iterations: get_usize(&config, "max_iterations", 500),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let result = rust_vmd(sig.values(), &cfg)
        .map_err(|e| Error::Other(format!("VMD failed: {e}")))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    build_result(&result.imfs.imfs, &result.imfs.residue, "vmd", elapsed_ms, signal.len())
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Reconstruct a signal by summing all IMFs and the residue.
#[extendr]
fn reconstruct(result: List) -> Result<Vec<f64>> {
    let imfs_robj = result
        .iter()
        .find(|(k, _)| *k == "imfs")
        .map(|(_, v)| v)
        .ok_or_else(|| Error::Other("result must contain 'imfs' field".to_string()))?;

    let residue_robj = result
        .iter()
        .find(|(k, _)| *k == "residue")
        .map(|(_, v)| v)
        .ok_or_else(|| Error::Other("result must contain 'residue' field".to_string()))?;

    let imfs_list = List::try_from(imfs_robj)
        .map_err(|_| Error::Other("'imfs' must be a list".to_string()))?;

    let residue: Vec<f64> = residue_robj
        .as_real_vector()
        .ok_or_else(|| Error::Other("'residue' must be numeric".to_string()))?;

    let n = residue.len();
    let mut out = vec![0.0f64; n];

    for (_, imf_robj) in imfs_list.iter() {
        let imf = imf_robj
            .as_real_vector()
            .ok_or_else(|| Error::Other("each IMF must be numeric".to_string()))?;
        if imf.len() != n {
            return Err(Error::Other("IMF length mismatch".to_string()));
        }
        for (o, v) in out.iter_mut().zip(imf.iter()) {
            *o += v;
        }
    }

    for (o, v) in out.iter_mut().zip(residue.iter()) {
        *o += v;
    }

    Ok(out)
}

/// Return the ferromodeR package version.
#[extendr]
fn ferromode_r_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

extendr_module! {
    mod ferromodeR;
    fn emd;
    fn eemd;
    fn ceemd;
    fn ceemdan;
    fn iceemdan;
    fn vmd;
    fn reconstruct;
    fn ferromode_r_version;
}
