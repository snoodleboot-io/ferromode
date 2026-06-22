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
// extendr 0.9's prelude no longer re-exports the `Result<T>` alias, so import it
// explicitly; otherwise `Result<List>` resolves to std's two-parameter Result.
use extendr_api::Result;
use ferromode::adapters::streaming::{ArModel, StreamingDecomposer};
use ferromode::algorithms::ceemd::ceemd as rust_ceemd;
use ferromode::algorithms::ceemdan::ceemdan as rust_ceemdan;
use ferromode::algorithms::eemd::{eemd as rust_eemd, EnsembleConfig};
use ferromode::algorithms::emd::{emd as rust_emd, EmdConfig, IntermittencyConfig};
use ferromode::algorithms::hilbert::hilbert_imf;
use ferromode::algorithms::iceemdan::iceemdan as rust_iceemdan;
use ferromode::algorithms::vmd::{vmd as rust_vmd, VmdConfig};
use ferromode::boundary::BoundaryConditionType;
use ferromode::ml::differentiable::{DifferentiableEmd, ImplicitEmdContext};
use ferromode::multivariate::direction_sampling::DirectionConfig;
use ferromode::multivariate::memd::{memd as rust_memd, MemdConfig};
use ferromode::multivariate::namemd::{namemd as rust_namemd, NaMemdConfig};
use ferromode::sifting::SiftingConfig;
use ferromode::spline::SplineType;
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

fn get_str(config: &List, key: &str, default: &str) -> String {
    config
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| v.as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| default.to_string())
}

fn get_bool(config: &List, key: &str, default: bool) -> bool {
    config
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| v.as_bool())
        .unwrap_or(default)
}

fn parse_boundary(s: &str) -> BoundaryConditionType {
    match s {
        "mirror_even" | "mirror" => BoundaryConditionType::MirrorEven,
        "mirror_odd" => BoundaryConditionType::MirrorOdd,
        "periodic" => BoundaryConditionType::Periodic,
        "slope" => BoundaryConditionType::Slope,
        "ar_model" => BoundaryConditionType::ARModel,
        "characteristic_wave" => BoundaryConditionType::CharacteristicWave,
        "waveform_matching" => BoundaryConditionType::WaveformMatching,
        "palindrome_cyclic" => BoundaryConditionType::PalindromeCyclic,
        _ => BoundaryConditionType::MirrorEven,
    }
}

fn parse_spline(s: &str) -> SplineType {
    match s {
        "periodic" => SplineType::Periodic,
        "not_a_knot" => SplineType::NotAKnot,
        _ => SplineType::Natural,
    }
}

/// Parse a full `EmdConfig` from an R named list (every knob optional).
fn parse_emd_config(config: &List) -> EmdConfig {
    let boundary = parse_boundary(&get_str(config, "boundary_condition", "mirror_even"));
    let fixed = get_usize(config, "fixed_iterations", 0);
    let sifting = SiftingConfig {
        sd_threshold: get_f64(config, "sd_threshold", 0.2),
        s_number: get_usize(config, "s_number", 5),
        max_sifting_iterations: get_usize(config, "max_sifting_iterations", 100),
        fixed_iterations: if fixed > 0 { Some(fixed) } else { None },
        energy_threshold: get_f64(config, "energy_threshold", 1e-6),
        boundary_condition: boundary,
        spline_type: parse_spline(&get_str(config, "spline_type", "natural")),
    };
    let intermittency_cv = get_f64(config, "intermittency_cv", -1.0);
    let intermittency = if intermittency_cv >= 0.0 {
        Some(IntermittencyConfig {
            cv_threshold: intermittency_cv,
            min_intervals: get_usize(config, "intermittency_min_intervals", 3),
        })
    } else {
        None
    };
    EmdConfig {
        sifting_config: sifting,
        max_imfs: get_usize(config, "max_imfs", 0),
        boundary_condition: boundary,
        intermittency,
        reconstruction_tolerance: get_f64(config, "reconstruction_tolerance", 1e-12),
        validate_reconstruction: get_bool(config, "validate_reconstruction", false),
    }
}

/// Extract a list of numeric channel vectors from an R list.
fn parse_channels(channels: &List) -> Result<Vec<Vec<f64>>> {
    let mut out = Vec::with_capacity(channels.len());
    for (_, v) in channels.iter() {
        let ch = v
            .as_real_vector()
            .ok_or_else(|| Error::Other("each channel must be numeric".to_string()))?;
        out.push(ch);
    }
    if out.is_empty() {
        return Err(Error::Other("must provide at least one channel".to_string()));
    }
    Ok(out)
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
    let sig = Signal::from_slice(&signal)
        .map_err(|e| Error::Other(format!("Invalid signal: {e}")))?;

    let cfg = parse_emd_config(&config);

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
    let emd_cfg = parse_emd_config(&config);

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
    let emd_cfg = parse_emd_config(&config);

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
    let emd_cfg = parse_emd_config(&config);

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
    let emd_cfg = parse_emd_config(&config);

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
        tau: get_f64(&config, "tau", 0.0),
        tol: get_f64(&config, "tol", 1e-7),
        max_iterations: get_usize(&config, "max_iterations", 500),
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
// Multivariate (MEMD / NA-MEMD)
// ---------------------------------------------------------------------------

fn parse_memd_config(config: &List) -> MemdConfig {
    let dir = DirectionConfig::new(get_usize(config, "num_directions", 8));
    let sifting = SiftingConfig {
        sd_threshold: get_f64(config, "sd_threshold", 0.2),
        s_number: get_usize(config, "s_number", 5),
        max_sifting_iterations: get_usize(config, "max_sifting_iterations", 100),
        ..Default::default()
    };
    let mut cfg = MemdConfig::new(dir, sifting);
    let max_imfs = get_usize(config, "max_imfs", 0);
    if max_imfs > 0 {
        cfg = cfg.with_max_imfs(max_imfs);
    }
    cfg
}

/// Decompose multivariate channels (a list of numeric vectors) using MEMD.
#[extendr]
fn memd(channels: List, config: List) -> Result<List> {
    let chans = parse_channels(&channels)?;
    let n_samples = chans.first().map_or(0, |c| c.len());
    let cfg = parse_memd_config(&config);
    let start = std::time::Instant::now();
    let result = rust_memd(&chans, &cfg).map_err(|e| Error::Other(format!("MEMD failed: {e}")))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    build_result(&result.imfs.imfs, &result.imfs.residue, "memd", elapsed_ms, n_samples)
}

/// Decompose multivariate channels using Noise-Assisted MEMD.
#[extendr]
fn namemd(channels: List, config: List) -> Result<List> {
    let chans = parse_channels(&channels)?;
    let n_samples = chans.first().map_or(0, |c| c.len());
    let mut cfg = NaMemdConfig::new(parse_memd_config(&config));
    let n_noise = get_usize(&config, "n_noise_channels", 0);
    if n_noise > 0 {
        cfg = cfg.with_noise_channels(n_noise);
    }
    cfg = cfg.with_noise_std(get_f64(&config, "noise_std", 0.1));
    if let Some(seed) = get_u64_opt(&config, "seed") {
        cfg = cfg.with_seed(seed);
    }
    let start = std::time::Instant::now();
    let result =
        rust_namemd(&chans, &cfg).map_err(|e| Error::Other(format!("NA-MEMD failed: {e}")))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    build_result(&result.imfs.imfs, &result.imfs.residue, "namemd", elapsed_ms, n_samples)
}

// ---------------------------------------------------------------------------
// Hilbert spectral analysis
// ---------------------------------------------------------------------------

/// Compute the Hilbert spectrum for a list of IMF numeric vectors.
#[extendr]
fn hilbert(imfs: List, sample_rate: f64) -> Result<List> {
    let imf_vecs = parse_channels(&imfs)?;
    let result = hilbert_imf(&imf_vecs, sample_rate)
        .map_err(|e| Error::Other(format!("Hilbert failed: {e}")))?;
    let amp: Vec<Robj> =
        result.instantaneous_amplitude.iter().map(|v| r!(v.as_slice())).collect();
    let freq: Vec<Robj> =
        result.instantaneous_frequency.iter().map(|v| r!(v.as_slice())).collect();
    Ok(list!(
        instantaneous_amplitude = List::from_values(amp),
        instantaneous_frequency = List::from_values(freq),
        marginal_spectrum = r!(result.marginal_spectrum.as_slice())
    ))
}

// ---------------------------------------------------------------------------
// Streaming decomposition (external pointer handle)
// ---------------------------------------------------------------------------

/// Create a streaming decomposer handle. `ar_order` <= 0 defaults to 3,
/// `buffer_size` <= 0 to 4096.
#[extendr]
fn streaming_new(
    config: List,
    buffer_size: i32,
    ar_order: i32,
) -> Result<ExternalPtr<StreamingDecomposer>> {
    let order = if ar_order <= 0 { 3 } else { ar_order as usize };
    let predictor = ArModel::new(order).map_err(|e| Error::Other(format!("AR model: {e}")))?;
    let buf = if buffer_size <= 0 { 4096 } else { buffer_size as usize };
    let dec = StreamingDecomposer::new(parse_emd_config(&config), Box::new(predictor), buf)
        .map_err(|e| Error::Other(format!("Streaming init: {e}")))?;
    Ok(ExternalPtr::new(dec))
}

/// Decompose one chunk through a streaming handle.
#[extendr]
fn streaming_decompose_chunk(
    handle: ExternalPtr<StreamingDecomposer>,
    chunk: Vec<f64>,
) -> Result<List> {
    let signal =
        Signal::from_slice(&chunk).map_err(|e| Error::Other(format!("Invalid chunk: {e}")))?;
    let mut handle = handle;
    let res = handle
        .decompose_chunk(&signal)
        .map_err(|e| Error::Other(format!("Chunk failed: {e}")))?;
    let imfs: Vec<Robj> = res.imfs.iter().map(|v| r!(v.as_slice())).collect();
    Ok(list!(
        imfs = List::from_values(imfs),
        residue = r!(res.remainder.as_slice()),
        spectral_entropy = res.metrics.spectral_entropy,
        stationarity_score = res.metrics.stationarity_score,
        extrema_spacing_cv = res.metrics.extrema_spacing_cv
    ))
}

/// Reset a streaming handle's state.
#[extendr]
fn streaming_reset(handle: ExternalPtr<StreamingDecomposer>) -> Result<()> {
    let mut handle = handle;
    handle.reset();
    Ok(())
}

// ---------------------------------------------------------------------------
// Differentiable EMD
// ---------------------------------------------------------------------------

/// Differentiable EMD forward pass.
///
/// Returns imfs/residue/num_sifts/error plus an opaque `handle` (the saved
/// forward context) to pass to `emd_backward`.
#[extendr]
fn emd_forward(signal: Vec<f64>, config: List) -> Result<List> {
    let diff = DifferentiableEmd::new(parse_emd_config(&config));
    let ctx = diff.forward(&signal).map_err(|e| Error::Other(format!("forward failed: {e}")))?;
    let imfs: Vec<Robj> = ctx.imfs.iter().map(|v| r!(v.as_slice())).collect();
    let num_sifts: Vec<i32> = ctx.num_sifts.iter().map(|&n| n as i32).collect();
    let residue = r!(ctx.residue.as_slice());
    let reconstruction_error = ctx.reconstruction_error();
    Ok(list!(
        imfs = List::from_values(imfs),
        residue = residue,
        num_sifts = r!(num_sifts.as_slice()),
        reconstruction_error = reconstruction_error,
        handle = ExternalPtr::new(ctx)
    ))
}

/// Differentiable EMD backward pass (exact implicit differentiation).
///
/// `handle` is the `handle` returned by `emd_forward`; `grad_imfs` is the
/// upstream gradient w.r.t. each IMF. Returns the gradient w.r.t. the signal.
#[extendr]
fn emd_backward(handle: ExternalPtr<ImplicitEmdContext>, grad_imfs: List) -> Result<Vec<f64>> {
    let grads = parse_channels(&grad_imfs)?;
    handle.backward(&grads).map_err(|e| Error::Other(format!("backward failed: {e}")))
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
    fn memd;
    fn namemd;
    fn hilbert;
    fn reconstruct;
    fn streaming_new;
    fn streaming_decompose_chunk;
    fn streaming_reset;
    fn emd_forward;
    fn emd_backward;
    fn ferromode_r_version;
}
