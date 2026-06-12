//! C++ bindings for Ferromode via cxx bridge.
//!
//! This crate provides a cxx-based bridge for calling Ferromode algorithms
//! from C++. The C++ layer is ONLY marshalling — no algorithm logic.

// The `#[cxx::bridge]` macro expands to code that uses APIs newer than the
// workspace MSRV; the lint attributes those spans to this file, so it is not
// actionable in our source.
#![allow(clippy::incompatible_msrv)]

use ferromode::algorithms::ceemd::ceemd;
use ferromode::algorithms::ceemdan::ceemdan;
use ferromode::algorithms::eemd::{eemd, EnsembleConfig};
use ferromode::algorithms::emd::{emd, EmdConfig};
use ferromode::algorithms::hilbert::hilbert_imf;
use ferromode::algorithms::iceemdan::iceemdan;
use ferromode::algorithms::vmd::{vmd, VmdConfig};
use ferromode::error::EmdError;
use ferromode::multivariate::direction_sampling::DirectionConfig;
use ferromode::multivariate::memd::{memd, MemdConfig};
use ferromode::multivariate::namemd::{namemd, NaMemdConfig};
use ferromode::sifting::SiftingConfig;
use ferromode::types::HilbertResult;

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        type CEmdConfig;
        type CEnsembleConfig;
        type CMemdConfig;
        type CNaMemdConfig;
        type CVmdConfig;
        type CImfResult;
        type CHilbertResult;

        fn ferromode_emd_cxx(signal: &[f64], config: &CEmdConfig) -> Result<Box<CImfResult>>;
        fn ferromode_eemd_cxx(
            signal: &[f64],
            config: &CEnsembleConfig,
            emd_config: &CEmdConfig,
        ) -> Result<Box<CImfResult>>;
        fn ferromode_ceemd_cxx(
            signal: &[f64],
            config: &CEnsembleConfig,
            emd_config: &CEmdConfig,
        ) -> Result<Box<CImfResult>>;
        fn ferromode_ceemdan_cxx(
            signal: &[f64],
            config: &CEnsembleConfig,
            emd_config: &CEmdConfig,
        ) -> Result<Box<CImfResult>>;
        fn ferromode_iceemdan_cxx(
            signal: &[f64],
            config: &CEnsembleConfig,
            emd_config: &CEmdConfig,
        ) -> Result<Box<CImfResult>>;
        fn ferromode_memd_cxx(
            channels: &[f64],
            n_channels: usize,
            n_samples: usize,
            config: &CMemdConfig,
        ) -> Result<Box<CImfResult>>;
        fn ferromode_namemd_cxx(
            channels: &[f64],
            n_channels: usize,
            n_samples: usize,
            config: &CNaMemdConfig,
        ) -> Result<Box<CImfResult>>;
        fn ferromode_vmd_cxx(signal: &[f64], config: &CVmdConfig) -> Result<Box<CImfResult>>;
        fn ferromode_hilbert_cxx(
            imfs: &[f64],
            n_imfs: usize,
            n_samples: usize,
            sample_rate: f64,
        ) -> Result<Box<CHilbertResult>>;
    }

    extern "C++" {
        include!("ferromode.hpp");
    }
}

/// C-compatible EMD configuration.
#[repr(C)]
pub struct CEmdConfig {
    pub max_imfs: usize,
    pub sd_threshold: f64,
    pub s_number: usize,
    pub max_sifting_iterations: usize,
    pub boundary_condition: i32,
}

/// C-compatible ensemble configuration.
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

/// C-compatible IMF result.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CImfResult {
    pub imfs_data: *const f64,
    pub n_imfs: usize,
    pub n_samples: usize,
    pub residue: *const f64,
}

/// C-compatible Hilbert spectral analysis result.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CHilbertResult {
    pub instantaneous_amplitude: *const f64,
    pub instantaneous_frequency: *const f64,
    pub marginal_spectrum: *const f64,
    pub n_imfs: usize,
    pub n_samples: usize,
    pub n_freq_bins: usize,
}

fn c_emd_config_to_rust(config: &CEmdConfig) -> EmdConfig {
    let boundary = match config.boundary_condition {
        0 => ferromode::boundary::BoundaryConditionType::MirrorEven,
        1 => ferromode::boundary::BoundaryConditionType::MirrorOdd,
        2 => ferromode::boundary::BoundaryConditionType::Periodic,
        3 => ferromode::boundary::BoundaryConditionType::Slope,
        4 => ferromode::boundary::BoundaryConditionType::ARModel,
        5 => ferromode::boundary::BoundaryConditionType::CharacteristicWave,
        6 => ferromode::boundary::BoundaryConditionType::WaveformMatching,
        7 => ferromode::boundary::BoundaryConditionType::PalindromeCyclic,
        _ => ferromode::boundary::BoundaryConditionType::MirrorEven,
    };

    EmdConfig {
        sifting_config: SiftingConfig {
            sd_threshold: config.sd_threshold,
            s_number: config.s_number,
            max_sifting_iterations: config.max_sifting_iterations,
            ..Default::default()
        },
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

fn c_memd_config_to_rust(config: &CMemdConfig) -> MemdConfig {
    let dir_config = DirectionConfig::new(config.num_directions).with_seed(config.direction_seed);
    let sifting_config = SiftingConfig {
        sd_threshold: config.sd_threshold,
        s_number: config.s_number,
        max_sifting_iterations: config.max_sifting_iterations,
        ..Default::default()
    };
    MemdConfig::new(dir_config, sifting_config).with_max_imfs(config.max_imfs)
}

fn c_namemd_config_to_rust(config: &CNaMemdConfig) -> NaMemdConfig {
    let base = c_memd_config_to_rust(&config.base);
    let mut na_config = NaMemdConfig::new(base)
        .with_noise_channels(config.n_noise_channels)
        .with_noise_std(config.noise_std);
    if config.use_seed != 0 {
        na_config = na_config.with_seed(config.seed);
    }
    na_config
}

fn c_vmd_config_to_rust(config: &CVmdConfig) -> VmdConfig {
    VmdConfig {
        n_modes: config.n_modes,
        alpha: config.alpha,
        tau: config.tau,
        tol: config.tol,
        max_iterations: config.max_iterations,
    }
}

fn result_to_c(result: ferromode::types::DecompositionResult) -> CImfResult {
    let n_imfs = result.imfs.imfs.len();
    let n_samples = if n_imfs > 0 {
        result.imfs.imfs[0].len()
    } else if !result.imfs.residue.is_empty() {
        result.imfs.residue.len()
    } else {
        0
    };

    let mut flat_imfs: Vec<f64> = Vec::with_capacity(n_imfs * n_samples);
    for imf in &result.imfs.imfs {
        flat_imfs.extend_from_slice(imf);
    }

    let imfs_ptr = Box::into_raw(flat_imfs.into_boxed_slice()) as *const f64;
    let residue_ptr = Box::into_raw(result.imfs.residue.clone().into_boxed_slice()) as *const f64;

    CImfResult { imfs_data: imfs_ptr, n_imfs, n_samples, residue: residue_ptr }
}

fn hilbert_to_c(result: HilbertResult) -> CHilbertResult {
    let n_imfs = result.instantaneous_amplitude.len();
    let n_samples = if n_imfs > 0 { result.instantaneous_amplitude[0].len() } else { 0 };
    let n_freq_bins = result.marginal_spectrum.len();

    let mut flat_amp: Vec<f64> = Vec::with_capacity(n_imfs * n_samples);
    let mut flat_freq: Vec<f64> = Vec::with_capacity(n_imfs * n_samples);
    for (amp, freq) in
        result.instantaneous_amplitude.iter().zip(result.instantaneous_frequency.iter())
    {
        flat_amp.extend_from_slice(amp);
        flat_freq.extend_from_slice(freq);
    }

    let amp_ptr = Box::into_raw(flat_amp.into_boxed_slice()) as *const f64;
    let freq_ptr = Box::into_raw(flat_freq.into_boxed_slice()) as *const f64;
    let marginal_ptr = Box::into_raw(result.marginal_spectrum.into_boxed_slice()) as *const f64;

    CHilbertResult {
        instantaneous_amplitude: amp_ptr,
        instantaneous_frequency: freq_ptr,
        marginal_spectrum: marginal_ptr,
        n_imfs,
        n_samples,
        n_freq_bins,
    }
}

fn emd_error_to_string(e: &EmdError) -> String {
    e.to_string()
}

pub fn ferromode_emd_cxx(signal: &[f64], config: &CEmdConfig) -> Result<Box<CImfResult>, String> {
    let rust_config = c_emd_config_to_rust(config);
    emd(signal, &rust_config).map(|r| Box::new(result_to_c(r))).map_err(|e| emd_error_to_string(&e))
}

pub fn ferromode_eemd_cxx(
    signal: &[f64],
    config: &CEnsembleConfig,
    emd_config: &CEmdConfig,
) -> Result<Box<CImfResult>, String> {
    let ens_config = c_ensemble_config_to_rust(config);
    let rust_emd_config = c_emd_config_to_rust(emd_config);
    eemd(signal, &ens_config, &rust_emd_config)
        .map(|r| Box::new(result_to_c(r)))
        .map_err(|e| emd_error_to_string(&e))
}

pub fn ferromode_ceemd_cxx(
    signal: &[f64],
    config: &CEnsembleConfig,
    emd_config: &CEmdConfig,
) -> Result<Box<CImfResult>, String> {
    let ens_config = c_ensemble_config_to_rust(config);
    let rust_emd_config = c_emd_config_to_rust(emd_config);
    ceemd(signal, &ens_config, &rust_emd_config)
        .map(|r| Box::new(result_to_c(r)))
        .map_err(|e| emd_error_to_string(&e))
}

pub fn ferromode_ceemdan_cxx(
    signal: &[f64],
    config: &CEnsembleConfig,
    emd_config: &CEmdConfig,
) -> Result<Box<CImfResult>, String> {
    let ens_config = c_ensemble_config_to_rust(config);
    let rust_emd_config = c_emd_config_to_rust(emd_config);
    ceemdan(signal, &ens_config, &rust_emd_config)
        .map(|r| Box::new(result_to_c(r)))
        .map_err(|e| emd_error_to_string(&e))
}

pub fn ferromode_iceemdan_cxx(
    signal: &[f64],
    config: &CEnsembleConfig,
    emd_config: &CEmdConfig,
) -> Result<Box<CImfResult>, String> {
    let ens_config = c_ensemble_config_to_rust(config);
    let rust_emd_config = c_emd_config_to_rust(emd_config);
    iceemdan(signal, &ens_config, &rust_emd_config)
        .map(|r| Box::new(result_to_c(r)))
        .map_err(|e| emd_error_to_string(&e))
}

pub fn ferromode_memd_cxx(
    channels: &[f64],
    n_channels: usize,
    n_samples: usize,
    config: &CMemdConfig,
) -> Result<Box<CImfResult>, String> {
    let rust_config = c_memd_config_to_rust(config);
    let channel_data: Vec<Vec<f64>> = (0..n_channels)
        .map(|ch| {
            let start = ch * n_samples;
            channels[start..start + n_samples].to_vec()
        })
        .collect();
    memd(&channel_data, &rust_config)
        .map(|r| Box::new(result_to_c(r)))
        .map_err(|e| emd_error_to_string(&e))
}

pub fn ferromode_namemd_cxx(
    channels: &[f64],
    n_channels: usize,
    n_samples: usize,
    config: &CNaMemdConfig,
) -> Result<Box<CImfResult>, String> {
    let rust_config = c_namemd_config_to_rust(config);
    let channel_data: Vec<Vec<f64>> = (0..n_channels)
        .map(|ch| {
            let start = ch * n_samples;
            channels[start..start + n_samples].to_vec()
        })
        .collect();
    namemd(&channel_data, &rust_config)
        .map(|r| Box::new(result_to_c(r)))
        .map_err(|e| emd_error_to_string(&e))
}

pub fn ferromode_vmd_cxx(signal: &[f64], config: &CVmdConfig) -> Result<Box<CImfResult>, String> {
    let rust_config = c_vmd_config_to_rust(config);
    vmd(signal, &rust_config).map(|r| Box::new(result_to_c(r))).map_err(|e| emd_error_to_string(&e))
}

pub fn ferromode_hilbert_cxx(
    imfs: &[f64],
    n_imfs: usize,
    n_samples: usize,
    sample_rate: f64,
) -> Result<Box<CHilbertResult>, String> {
    let imf_data: Vec<Vec<f64>> = (0..n_imfs)
        .map(|i| {
            let start = i * n_samples;
            imfs[start..start + n_samples].to_vec()
        })
        .collect();
    hilbert_imf(&imf_data, sample_rate)
        .map(|r| Box::new(hilbert_to_c(r)))
        .map_err(|e| emd_error_to_string(&e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_emd_config() -> CEmdConfig {
        CEmdConfig {
            max_imfs: 0,
            sd_threshold: 0.2,
            s_number: 5,
            max_sifting_iterations: 100,
            boundary_condition: 0,
        }
    }

    fn make_ensemble_config() -> CEnsembleConfig {
        CEnsembleConfig { num_ensembles: 5, noise_std: 0.2, seed: 42, use_seed: 1 }
    }

    fn make_sine_signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect()
    }

    #[test]
    fn test_cxx_emd_basic() {
        let signal = make_sine_signal(100);
        let config = make_emd_config();
        let result = ferromode_emd_cxx(&signal, &config);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.n_imfs >= 1);
        assert_eq!(result.n_samples, 100);
    }

    #[test]
    fn test_cxx_eemd_basic() {
        let signal = make_sine_signal(100);
        let ens_config = make_ensemble_config();
        let emd_config = make_emd_config();
        let result = ferromode_eemd_cxx(&signal, &ens_config, &emd_config);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.n_imfs >= 1);
        assert_eq!(result.n_samples, 100);
    }

    #[test]
    fn test_cxx_vmd_basic() {
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();
        let config =
            CVmdConfig { n_modes: 2, alpha: 2000.0, tau: 0.0, tol: 1e-7, max_iterations: 500 };
        let result = ferromode_vmd_cxx(&signal, &config);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.n_imfs, 2);
    }

    #[test]
    fn test_cxx_empty_signal_error() {
        let signal: Vec<f64> = vec![];
        let config = make_emd_config();
        let result = ferromode_emd_cxx(&signal, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_cxx_nan_signal_error() {
        let signal = vec![1.0, f64::NAN, 3.0];
        let config = make_emd_config();
        let result = ferromode_emd_cxx(&signal, &config);
        assert!(result.is_err());
    }
}
