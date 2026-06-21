//! Input and output marshalling between MATLAB/Octave and Rust.
//!
//! Pure-wrap contract: ONLY marshalling, no algorithm logic.

use crate::mex_compat::{self, MEX_REAL};
use ferromode::algorithms::eemd::EnsembleConfig;
use ferromode::algorithms::emd::EmdConfig;
use ferromode::multivariate::direction_sampling::DirectionConfig;
use ferromode::multivariate::memd::MemdConfig;
use ferromode::multivariate::namemd::NaMemdConfig;
use ferromode::sifting::SiftingConfig;
use ferromode::types::{AlgorithmType, DecompositionResult, HilbertResult};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Extract a 1D signal from a MATLAB mxArray.
///
/// Validates: non-double, complex, empty, non-finite.
pub fn mx_array_to_signal(prhs: *mut mex_sys::mxArray) -> Result<Vec<f64>, String> {
    if prhs.is_null() {
        return Err("signal is null".to_string());
    }

    unsafe {
        if mex_sys::mxIsEmpty(prhs) != 0 {
            return Err("signal is empty: must contain at least one sample".to_string());
        }

        if mex_sys::mxIsDouble(prhs) == 0 {
            return Err("signal must be double precision".to_string());
        }

        if mex_sys::mxIsComplex(prhs) != 0 {
            return Err("signal must be real (non-complex)".to_string());
        }

        let n = mex_sys::mxGetNumberOfElements(prhs);
        let data_ptr = mex_sys::mxGetPr(prhs);
        if data_ptr.is_null() {
            return Err("signal data pointer is null".to_string());
        }

        let data = std::slice::from_raw_parts(data_ptr, n);

        for (i, &val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!(
                    "signal contains non-finite value (NaN or Inf) at index {}",
                    i
                ));
            }
        }

        Ok(data.to_vec())
    }
}

/// Extract a multivariate signal (channels x samples) from a MATLAB matrix.
///
/// MATLAB stores matrices column-major: each column is a channel.
pub fn mx_array_to_multivariate(prhs: *mut mex_sys::mxArray) -> Result<Vec<Vec<f64>>, String> {
    if prhs.is_null() {
        return Err("signal is null".to_string());
    }

    unsafe {
        if mex_sys::mxIsEmpty(prhs) != 0 {
            return Err("signal is empty: must contain at least one channel".to_string());
        }

        if mex_sys::mxIsDouble(prhs) == 0 {
            return Err("signal must be double precision".to_string());
        }

        if mex_sys::mxIsComplex(prhs) != 0 {
            return Err("signal must be real (non-complex)".to_string());
        }

        let ndims = mex_compat::mxGetNumberOfDimensions(prhs);
        if ndims < 2 {
            return Err("signal must be a 2D matrix (samples x channels)".to_string());
        }

        let dims = mex_compat::mxGetDimensions(prhs);
        let n_rows = *dims;
        let n_cols = *dims.add(1);

        if n_rows == 0 || n_cols == 0 {
            return Err("signal dimensions must be non-zero".to_string());
        }

        let data_ptr = mex_sys::mxGetPr(prhs);
        if data_ptr.is_null() {
            return Err("signal data pointer is null".to_string());
        }

        let data = std::slice::from_raw_parts(data_ptr, n_rows * n_cols);

        // MATLAB is column-major: each column is one channel
        let mut channels: Vec<Vec<f64>> = Vec::with_capacity(n_cols);
        for col in 0..n_cols {
            let mut channel = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let idx = col * n_rows + row;
                let val = data[idx];
                if !val.is_finite() {
                    return Err(format!(
                        "signal contains non-finite value at channel {}, sample {}",
                        col, row
                    ));
                }
                channel.push(val);
            }
            channels.push(channel);
        }

        Ok(channels)
    }
}

/// Extract a string from a MATLAB mxArray (char array).
pub fn mx_array_to_string(prhs: *mut mex_sys::mxArray) -> Result<String, String> {
    if prhs.is_null() {
        return Err("null pointer".to_string());
    }

    unsafe {
        if mex_sys::mxIsChar(prhs) == 0 {
            return Err("expected char array".to_string());
        }

        let c_str = mex_sys::mxArrayToString(prhs);
        if c_str.is_null() {
            return Err("failed to convert mxArray to string".to_string());
        }

        let rust_str = CStr::from_ptr(c_str).to_string_lossy().into_owned();
        mex_sys::mxFree(c_str.cast::<std::ffi::c_void>());

        Ok(rust_str)
    }
}

/// Extract an optional double scalar from a MATLAB mxArray field.
pub fn mx_get_optional_double(
    prhs: *mut mex_sys::mxArray,
    field: &str,
) -> Result<Option<f64>, String> {
    if prhs.is_null() {
        return Ok(None);
    }

    unsafe {
        let field_c = CString::new(field).map_err(|e| e.to_string())?;
        let field_ptr = mex_compat::mxGetField(prhs, 0, field_c.as_ptr());
        if field_ptr.is_null() {
            return Ok(None);
        }

        if mex_sys::mxIsDouble(field_ptr) == 0 {
            return Ok(None);
        }

        let data_ptr = mex_sys::mxGetPr(field_ptr);
        if data_ptr.is_null() {
            return Ok(None);
        }

        Ok(Some(*data_ptr))
    }
}

/// Extract an optional usize from a MATLAB mxArray field.
pub fn mx_get_optional_usize(
    prhs: *mut mex_sys::mxArray,
    field: &str,
) -> Result<Option<usize>, String> {
    if let Some(val) = mx_get_optional_double(prhs, field)? {
        return Ok(Some(val as usize));
    }
    Ok(None)
}

/// Extract an optional string from a MATLAB mxArray field.
pub fn mx_get_optional_string(
    prhs: *mut mex_sys::mxArray,
    field: &str,
) -> Result<Option<String>, String> {
    if prhs.is_null() {
        return Ok(None);
    }

    unsafe {
        let field_c = CString::new(field).map_err(|e| e.to_string())?;
        let field_ptr = mex_compat::mxGetField(prhs, 0, field_c.as_ptr());
        if field_ptr.is_null() {
            return Ok(None);
        }

        if mex_sys::mxIsChar(field_ptr) == 0 {
            return Ok(None);
        }

        let c_str = mex_sys::mxArrayToString(field_ptr);
        if c_str.is_null() {
            return Ok(None);
        }

        let rust_str = CStr::from_ptr(c_str).to_string_lossy().into_owned();
        mex_sys::mxFree(c_str.cast::<std::ffi::c_void>());

        Ok(Some(rust_str))
    }
}

/// Parse EmdConfig from an optional MATLAB struct.
pub fn parse_emd_config(prhs: *mut mex_sys::mxArray) -> Result<EmdConfig, String> {
    let mut config = EmdConfig::default();

    if prhs.is_null() {
        return Ok(config);
    }

    unsafe {
        if mex_sys::mxIsStruct(prhs) == 0 {
            return Ok(config);
        }
    }

    if let Some(val) = mx_get_optional_usize(prhs, "MaxIMFs")? {
        config.max_imfs = val;
    }
    if let Some(val) = mx_get_optional_double(prhs, "SDThreshold")? {
        config.sifting_config.sd_threshold = val;
    }
    if let Some(val) = mx_get_optional_usize(prhs, "SNumber")? {
        config.sifting_config.s_number = val;
    }
    if let Some(val) = mx_get_optional_usize(prhs, "MaxSiftingIterations")? {
        config.sifting_config.max_sifting_iterations = val;
    }
    if let Some(val) = mx_get_optional_string(prhs, "BoundaryCondition")? {
        config.boundary_condition = parse_boundary_condition(&val)?;
        config.sifting_config.boundary_condition = config.boundary_condition;
    }
    if let Some(val) = mx_get_optional_string(prhs, "SplineType")? {
        config.sifting_config.spline_type = parse_spline_type(&val);
    }
    if let Some(val) = mx_get_optional_usize(prhs, "FixedIterations")? {
        config.sifting_config.fixed_iterations = Some(val);
    }
    if let Some(val) = mx_get_optional_double(prhs, "EnergyThreshold")? {
        config.sifting_config.energy_threshold = val;
    }
    if let Some(val) = mx_get_optional_double(prhs, "ReconstructionTolerance")? {
        config.reconstruction_tolerance = val;
    }
    if let Some(val) = mx_get_optional_double(prhs, "ValidateReconstruction")? {
        config.validate_reconstruction = val != 0.0;
    }
    if let Some(cv) = mx_get_optional_double(prhs, "IntermittencyCV")? {
        let min_intervals =
            mx_get_optional_usize(prhs, "IntermittencyMinIntervals")?.unwrap_or(3);
        config.intermittency = Some(ferromode::algorithms::emd::IntermittencyConfig {
            cv_threshold: cv,
            min_intervals,
        });
    }

    Ok(config)
}

/// Parse a spline type name; unknown values fall back to Natural.
fn parse_spline_type(s: &str) -> ferromode::spline::SplineType {
    use ferromode::spline::SplineType;
    match s.to_lowercase().as_str() {
        "periodic" => SplineType::Periodic,
        "notaknot" | "not_a_knot" => SplineType::NotAKnot,
        _ => SplineType::Natural,
    }
}

/// Parse EnsembleConfig from an optional MATLAB struct.
pub fn parse_ensemble_config(prhs: *mut mex_sys::mxArray) -> Result<EnsembleConfig, String> {
    let mut config = EnsembleConfig::default();

    if prhs.is_null() {
        return Ok(config);
    }

    unsafe {
        if mex_sys::mxIsStruct(prhs) == 0 {
            return Ok(config);
        }
    }

    if let Some(val) = mx_get_optional_usize(prhs, "NumEnsembles")? {
        config.num_ensembles = val;
    }
    if let Some(val) = mx_get_optional_double(prhs, "NoiseStd")? {
        config.noise_std = val;
    }
    if let Some(val) = mx_get_optional_double(prhs, "Seed")? {
        config.seed = Some(val as u64);
    }

    Ok(config)
}

/// Parse MemdConfig from an optional MATLAB struct.
pub fn parse_memd_config(prhs: *mut mex_sys::mxArray) -> Result<MemdConfig, String> {
    let dir_config = DirectionConfig::new(8);
    let sifting_config = SiftingConfig::default();
    let mut config = MemdConfig::new(dir_config, sifting_config);

    if prhs.is_null() {
        return Ok(config);
    }

    unsafe {
        if mex_sys::mxIsStruct(prhs) == 0 {
            return Ok(config);
        }
    }

    if let Some(val) = mx_get_optional_usize(prhs, "NumDirections")? {
        config.direction_config = DirectionConfig::new(val);
    }
    if let Some(val) = mx_get_optional_usize(prhs, "MaxIMFs")? {
        config = config.with_max_imfs(val);
    }
    if let Some(val) = mx_get_optional_double(prhs, "SDThreshold")? {
        config.sifting_config.sd_threshold = val;
    }
    if let Some(val) = mx_get_optional_usize(prhs, "SNumber")? {
        config.sifting_config.s_number = val;
    }
    if let Some(val) = mx_get_optional_usize(prhs, "MaxSiftingIterations")? {
        config.sifting_config.max_sifting_iterations = val;
    }

    Ok(config)
}

/// Parse NaMemdConfig from an optional MATLAB struct.
pub fn parse_namemd_config(prhs: *mut mex_sys::mxArray) -> Result<NaMemdConfig, String> {
    let base = parse_memd_config(prhs)?;
    let mut config = NaMemdConfig::new(base);

    if prhs.is_null() {
        return Ok(config);
    }

    if let Some(val) = mx_get_optional_usize(prhs, "NNoiseChannels")? {
        config = config.with_noise_channels(val);
    }
    if let Some(val) = mx_get_optional_double(prhs, "NoiseStd")? {
        config = config.with_noise_std(val);
    }
    if let Some(val) = mx_get_optional_double(prhs, "Seed")? {
        config = config.with_seed(val as u64);
    }

    Ok(config)
}

/// Parse VmdConfig from an optional MATLAB struct.
pub fn parse_vmd_config(
    prhs: *mut mex_sys::mxArray,
) -> Result<ferromode::algorithms::vmd::VmdConfig, String> {
    let mut config = ferromode::algorithms::vmd::VmdConfig::default();

    if prhs.is_null() {
        return Ok(config);
    }

    unsafe {
        if mex_sys::mxIsStruct(prhs) == 0 {
            return Ok(config);
        }
    }

    if let Some(val) = mx_get_optional_usize(prhs, "NModes")? {
        config.n_modes = val;
    }
    if let Some(val) = mx_get_optional_double(prhs, "Alpha")? {
        config.alpha = val;
    }
    if let Some(val) = mx_get_optional_double(prhs, "Tau")? {
        config.tau = val;
    }
    if let Some(val) = mx_get_optional_double(prhs, "Tol")? {
        config.tol = val;
    }
    if let Some(val) = mx_get_optional_usize(prhs, "MaxIterations")? {
        config.max_iterations = val;
    }

    Ok(config)
}

/// Parse boundary condition string to enum.
pub fn parse_boundary_condition(
    s: &str,
) -> Result<ferromode::boundary::BoundaryConditionType, String> {
    use ferromode::boundary::BoundaryConditionType;
    // Accept both underscore ("palindrome_cyclic") and compact ("palindromecyclic")
    // spellings by normalizing away underscores.
    match s.to_lowercase().replace('_', "").as_str() {
        "mirror" | "mirroreven" => Ok(BoundaryConditionType::MirrorEven),
        "mirrorodd" => Ok(BoundaryConditionType::MirrorOdd),
        "periodic" => Ok(BoundaryConditionType::Periodic),
        "slope" => Ok(BoundaryConditionType::Slope),
        "ar" | "armodel" => Ok(BoundaryConditionType::ARModel),
        "characteristic" | "characteristicwave" => Ok(BoundaryConditionType::CharacteristicWave),
        "waveform" | "waveformmatching" => Ok(BoundaryConditionType::WaveformMatching),
        "palindrome" | "palindromecyclic" => Ok(BoundaryConditionType::PalindromeCyclic),
        other => Err(format!("unknown boundary condition: {}", other)),
    }
}

/// Marshal a DecompositionResult to a MATLAB struct mxArray.
pub fn result_to_matlab(result: &DecompositionResult) -> Result<*mut mex_sys::mxArray, String> {
    let n_imfs = result.imfs.n_imfs();
    let n_samples = if n_imfs > 0 {
        result.imfs.imfs[0].len()
    } else if !result.imfs.residue.is_empty() {
        result.imfs.residue.len()
    } else {
        0
    };

    unsafe {
        // Create struct with field names
        let field_names: Vec<CString> = vec![
            CString::new("imfs").unwrap(),
            CString::new("residue").unwrap(),
            CString::new("n_imfs").unwrap(),
            CString::new("algorithm").unwrap(),
            CString::new("elapsed_ms").unwrap(),
            CString::new("n_siftings").unwrap(),
        ];
        let field_ptrs: Vec<*const c_char> = field_names.iter().map(|s| s.as_ptr()).collect();

        let struct_ptr = mex_compat::mxCreateStructMatrix(
            1,
            1,
            field_ptrs.len() as i32,
            field_ptrs.as_ptr(),
        );

        if struct_ptr.is_null() {
            return Err("failed to create struct mxArray".to_string());
        }

        // IMFs matrix: n_imfs x n_samples, stored COLUMN-MAJOR for MATLAB/Octave
        // (element (row=imf, col=sample) lives at col*n_imfs + row).
        let mut imf_data = vec![0.0f64; n_imfs * n_samples];
        for (r, imf) in result.imfs.imfs.iter().enumerate() {
            for (c, &v) in imf.iter().enumerate() {
                imf_data[c * n_imfs + r] = v;
            }
        }
        let imfs_matrix = if n_imfs > 0 && n_samples > 0 {
            let mx = mex_compat::mxCreateDoubleMatrix(n_imfs, n_samples, MEX_REAL);
            if mx.is_null() {
                return Err("failed to create IMFs matrix".to_string());
            }
            let dst = mex_sys::mxGetPr(mx);
            if !dst.is_null() {
                libc::memcpy(
                    dst.cast::<libc::c_void>(),
                    imf_data.as_ptr().cast::<libc::c_void>(),
                    imf_data.len() * 8,
                );
            }
            mx
        } else {
            mex_compat::mxCreateDoubleMatrix(0, 0, MEX_REAL)
        };
        mex_compat::mxSetField(struct_ptr, 0, field_names[0].as_ptr(), imfs_matrix);

        // Residue vector: 1 x n_samples
        let residue_len = result.imfs.residue.len();
        let residue_matrix = if residue_len > 0 {
            let mx = mex_compat::mxCreateDoubleMatrix(1, residue_len, MEX_REAL);
            if mx.is_null() {
                return Err("failed to create residue matrix".to_string());
            }
            let dst = mex_sys::mxGetPr(mx);
            if !dst.is_null() {
                libc::memcpy(
                    dst.cast::<libc::c_void>(),
                    result.imfs.residue.as_ptr().cast::<libc::c_void>(),
                    residue_len * 8,
                );
            }
            mx
        } else {
            mex_compat::mxCreateDoubleMatrix(0, 0, MEX_REAL)
        };
        mex_compat::mxSetField(struct_ptr, 0, field_names[1].as_ptr(), residue_matrix);

        // n_imfs scalar
        let n_imfs_mx = mex_sys::mxCreateDoubleScalar(n_imfs as f64);
        mex_compat::mxSetField(struct_ptr, 0, field_names[2].as_ptr(), n_imfs_mx);

        // algorithm string
        let algo_str = algorithm_to_string(&result.algorithm);
        let algo_c = CString::new(algo_str).unwrap_or_default();
        let algo_mx = mex_sys::mxCreateString(algo_c.as_ptr());
        mex_compat::mxSetField(struct_ptr, 0, field_names[3].as_ptr(), algo_mx);

        // elapsed_ms scalar
        let elapsed_ms = result.elapsed.as_secs_f64() * 1000.0;
        let elapsed_mxarr = mex_sys::mxCreateDoubleScalar(elapsed_ms);
        mex_compat::mxSetField(struct_ptr, 0, field_names[4].as_ptr(), elapsed_mxarr);

        // n_siftings scalar
        let n_sift_mx = mex_sys::mxCreateDoubleScalar(result.n_siftings as f64);
        mex_compat::mxSetField(struct_ptr, 0, field_names[5].as_ptr(), n_sift_mx);

        Ok(struct_ptr)
    }
}

/// Marshal a HilbertResult to a MATLAB struct mxArray.
pub fn hilbert_result_to_matlab(result: &HilbertResult) -> Result<*mut mex_sys::mxArray, String> {
    let n_imfs = result.instantaneous_amplitude.len();
    let n_samples = if n_imfs > 0 { result.instantaneous_amplitude[0].len() } else { 0 };

    unsafe {
        let field_names: Vec<CString> = vec![
            CString::new("instantaneous_amplitude").unwrap(),
            CString::new("instantaneous_frequency").unwrap(),
            CString::new("marginal_spectrum").unwrap(),
        ];
        let field_ptrs: Vec<*const c_char> = field_names.iter().map(|s| s.as_ptr()).collect();

        let struct_ptr = mex_compat::mxCreateStructMatrix(
            1,
            1,
            field_ptrs.len() as i32,
            field_ptrs.as_ptr(),
        );

        if struct_ptr.is_null() {
            return Err("failed to create struct mxArray".to_string());
        }

        // Instantaneous amplitude: n_imfs x n_samples (column-major).
        let mut amp_data = vec![0.0f64; n_imfs * n_samples];
        for (r, a) in result.instantaneous_amplitude.iter().enumerate() {
            for (c, &v) in a.iter().enumerate() {
                amp_data[c * n_imfs + r] = v;
            }
        }
        let amp_mx = if n_imfs > 0 && n_samples > 0 {
            let mx = mex_compat::mxCreateDoubleMatrix(n_imfs, n_samples, MEX_REAL);
            if !mx.is_null() {
                let dst = mex_sys::mxGetPr(mx);
                if !dst.is_null() {
                    libc::memcpy(
                        dst.cast::<libc::c_void>(),
                        amp_data.as_ptr().cast::<libc::c_void>(),
                        amp_data.len() * 8,
                    );
                }
            }
            mx
        } else {
            mex_compat::mxCreateDoubleMatrix(0, 0, MEX_REAL)
        };
        mex_compat::mxSetField(struct_ptr, 0, field_names[0].as_ptr(), amp_mx);

        // Instantaneous frequency: n_imfs x n_samples (column-major).
        let mut freq_data = vec![0.0f64; n_imfs * n_samples];
        for (r, f) in result.instantaneous_frequency.iter().enumerate() {
            for (c, &v) in f.iter().enumerate() {
                freq_data[c * n_imfs + r] = v;
            }
        }
        let freq_mx = if n_imfs > 0 && n_samples > 0 {
            let mx = mex_compat::mxCreateDoubleMatrix(n_imfs, n_samples, MEX_REAL);
            if !mx.is_null() {
                let dst = mex_sys::mxGetPr(mx);
                if !dst.is_null() {
                    libc::memcpy(
                        dst.cast::<libc::c_void>(),
                        freq_data.as_ptr().cast::<libc::c_void>(),
                        freq_data.len() * 8,
                    );
                }
            }
            mx
        } else {
            mex_compat::mxCreateDoubleMatrix(0, 0, MEX_REAL)
        };
        mex_compat::mxSetField(struct_ptr, 0, field_names[1].as_ptr(), freq_mx);

        // Marginal spectrum: 1 x n_bins
        let n_bins = result.marginal_spectrum.len();
        let marginal_mx = if n_bins > 0 {
            let mx = mex_compat::mxCreateDoubleMatrix(1, n_bins, MEX_REAL);
            if !mx.is_null() {
                let dst = mex_sys::mxGetPr(mx);
                if !dst.is_null() {
                    libc::memcpy(
                        dst.cast::<libc::c_void>(),
                        result.marginal_spectrum.as_ptr().cast::<libc::c_void>(),
                        n_bins * 8,
                    );
                }
            }
            mx
        } else {
            mex_compat::mxCreateDoubleMatrix(0, 0, MEX_REAL)
        };
        mex_compat::mxSetField(struct_ptr, 0, field_names[2].as_ptr(), marginal_mx);

        Ok(struct_ptr)
    }
}

/// Convert AlgorithmType to string.
pub fn algorithm_to_string(algo: &AlgorithmType) -> String {
    match algo {
        AlgorithmType::EMD => "EMD".to_string(),
        AlgorithmType::EEMD => "EEMD".to_string(),
        AlgorithmType::CEEMD => "CEEMD".to_string(),
        AlgorithmType::CEEMDAN => "CEEMDAN".to_string(),
        AlgorithmType::ICEEMDAN => "ICEEMDAN".to_string(),
        AlgorithmType::MEMD => "MEMD".to_string(),
        AlgorithmType::NAMEMD => "NA-MEMD".to_string(),
        AlgorithmType::VMD => "VMD".to_string(),
    }
}
