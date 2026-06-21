//! Function wrappers for all 8 Ferromode algorithms.
//!
//! Pure-wrap contract: MEX layer is ONLY marshalling, no algorithm logic.

use ferromode::algorithms::ceemd::ceemd;
use ferromode::algorithms::ceemdan::ceemdan;
use ferromode::algorithms::eemd::eemd;
use ferromode::algorithms::emd::emd;
use ferromode::algorithms::iceemdan::iceemdan;
use ferromode::algorithms::vmd::vmd;
use ferromode::hilbert::hilbert_imf;
use ferromode::multivariate::memd::memd;
use ferromode::multivariate::namemd::namemd;

use crate::marshalling::{
    hilbert_result_to_matlab, mx_array_to_multivariate, mx_array_to_signal, parse_emd_config,
    parse_ensemble_config, parse_memd_config, parse_namemd_config, parse_vmd_config,
    result_to_matlab,
};
use crate::mex_compat::{self, MEX_REAL};

/// Get the mxArray at the given index from the argument slice.
fn get_arg(prhs: &[*mut mex_sys::mxArray], index: usize) -> Option<*mut mex_sys::mxArray> {
    prhs.get(index).copied()
}

/// Check if an mxArray is a struct (non-null and mxIsStruct).
fn is_struct(prhs: *mut mex_sys::mxArray) -> bool {
    if prhs.is_null() {
        return false;
    }
    unsafe { mex_sys::mxIsStruct(prhs) != 0 }
}

// ---------------------------------------------------------------------------
// EMD
// ---------------------------------------------------------------------------

/// Perform Empirical Mode Decomposition.
///
/// MATLAB usage:
///   result = ferromode_mex('emd', signal)
///   result = ferromode_mex('emd', signal, config)
pub fn ferromode_emd(prhs: &[*mut mex_sys::mxArray]) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err("Usage: result = ferromode_mex('emd', signal, [config])".to_string());
    }

    let signal = mx_array_to_signal(prhs[0])?;
    let config_ptr =
        get_arg(prhs, 1).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());
    let config = parse_emd_config(config_ptr)?;

    let result = emd(&signal, &config).map_err(|e| format!("EMD failed: {}", e))?;
    result_to_matlab(&result)
}

// ---------------------------------------------------------------------------
// EEMD
// ---------------------------------------------------------------------------

/// Perform Ensemble Empirical Mode Decomposition.
pub fn ferromode_eemd(prhs: &[*mut mex_sys::mxArray]) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err(
            "Usage: result = ferromode_mex('eemd', signal, [ensemble_config], [emd_config])"
                .to_string(),
        );
    }

    let signal = mx_array_to_signal(prhs[0])?;
    let ens_config_ptr =
        get_arg(prhs, 1).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());
    let emd_config_ptr =
        get_arg(prhs, 2).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());

    let ens_config = parse_ensemble_config(ens_config_ptr)?;
    let emd_config = parse_emd_config(emd_config_ptr)?;

    let result =
        eemd(&signal, &ens_config, &emd_config).map_err(|e| format!("EEMD failed: {}", e))?;
    result_to_matlab(&result)
}

// ---------------------------------------------------------------------------
// CEEMD
// ---------------------------------------------------------------------------

/// Perform Complementary Ensemble Empirical Mode Decomposition.
pub fn ferromode_ceemd(prhs: &[*mut mex_sys::mxArray]) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err(
            "Usage: result = ferromode_mex('ceemd', signal, [ensemble_config], [emd_config])"
                .to_string(),
        );
    }

    let signal = mx_array_to_signal(prhs[0])?;
    let ens_config_ptr =
        get_arg(prhs, 1).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());
    let emd_config_ptr =
        get_arg(prhs, 2).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());

    let ens_config = parse_ensemble_config(ens_config_ptr)?;
    let emd_config = parse_emd_config(emd_config_ptr)?;

    let result =
        ceemd(&signal, &ens_config, &emd_config).map_err(|e| format!("CEEMD failed: {}", e))?;
    result_to_matlab(&result)
}

// ---------------------------------------------------------------------------
// CEEMDAN
// ---------------------------------------------------------------------------

/// Perform Complete Ensemble EMD with Adaptive Noise.
pub fn ferromode_ceemdan(prhs: &[*mut mex_sys::mxArray]) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err(
            "Usage: result = ferromode_mex('ceemdan', signal, [ensemble_config], [emd_config])"
                .to_string(),
        );
    }

    let signal = mx_array_to_signal(prhs[0])?;
    let ens_config_ptr =
        get_arg(prhs, 1).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());
    let emd_config_ptr =
        get_arg(prhs, 2).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());

    let ens_config = parse_ensemble_config(ens_config_ptr)?;
    let emd_config = parse_emd_config(emd_config_ptr)?;

    let result =
        ceemdan(&signal, &ens_config, &emd_config).map_err(|e| format!("CEEMDAN failed: {}", e))?;
    result_to_matlab(&result)
}

// ---------------------------------------------------------------------------
// ICEEMDAN
// ---------------------------------------------------------------------------

/// Perform Improved Complete Ensemble EMD with Adaptive Noise.
pub fn ferromode_iceemdan(prhs: &[*mut mex_sys::mxArray]) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err(
            "Usage: result = ferromode_mex('iceemdan', signal, [ensemble_config], [emd_config])"
                .to_string(),
        );
    }

    let signal = mx_array_to_signal(prhs[0])?;
    let ens_config_ptr =
        get_arg(prhs, 1).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());
    let emd_config_ptr =
        get_arg(prhs, 2).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());

    let ens_config = parse_ensemble_config(ens_config_ptr)?;
    let emd_config = parse_emd_config(emd_config_ptr)?;

    let result = iceemdan(&signal, &ens_config, &emd_config)
        .map_err(|e| format!("ICEEMDAN failed: {}", e))?;
    result_to_matlab(&result)
}

// ---------------------------------------------------------------------------
// MEMD
// ---------------------------------------------------------------------------

/// Perform Multivariate Empirical Mode Decomposition.
pub fn ferromode_memd(prhs: &[*mut mex_sys::mxArray]) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err("Usage: result = ferromode_mex('memd', signal, [config])".to_string());
    }

    let signal = mx_array_to_multivariate(prhs[0])?;
    let config_ptr =
        get_arg(prhs, 1).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());
    let config = parse_memd_config(config_ptr)?;

    let result = memd(&signal, &config).map_err(|e| format!("MEMD failed: {}", e))?;
    result_to_matlab(&result)
}

// ---------------------------------------------------------------------------
// NA-MEMD
// ---------------------------------------------------------------------------

/// Perform Noise-Assisted Multivariate Empirical Mode Decomposition.
pub fn ferromode_namemd(prhs: &[*mut mex_sys::mxArray]) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err("Usage: result = ferromode_mex('namemd', signal, [config])".to_string());
    }

    let signal = mx_array_to_multivariate(prhs[0])?;
    let config_ptr =
        get_arg(prhs, 1).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());
    let config = parse_namemd_config(config_ptr)?;

    let result = namemd(&signal, &config).map_err(|e| format!("NA-MEMD failed: {}", e))?;
    result_to_matlab(&result)
}

// ---------------------------------------------------------------------------
// VMD
// ---------------------------------------------------------------------------

/// Perform Variational Mode Decomposition.
pub fn ferromode_vmd(prhs: &[*mut mex_sys::mxArray]) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err("Usage: result = ferromode_mex('vmd', signal, [config])".to_string());
    }

    let signal = mx_array_to_signal(prhs[0])?;
    let config_ptr =
        get_arg(prhs, 1).filter(|p| !p.is_null() && is_struct(*p)).unwrap_or(std::ptr::null_mut());
    let config = parse_vmd_config(config_ptr)?;

    let result = vmd(&signal, &config).map_err(|e| format!("VMD failed: {}", e))?;
    result_to_matlab(&result)
}

// ---------------------------------------------------------------------------
// Hilbert-Huang Transform
// ---------------------------------------------------------------------------

/// Perform Hilbert-Huang transform on IMFs.
///
/// MATLAB usage:
///   h_result = ferromode_mex('hilbert', imfs, sample_rate)
pub fn ferromode_hilbert(prhs: &[*mut mex_sys::mxArray]) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.len() < 2 {
        return Err("Usage: result = ferromode_mex('hilbert', imfs, sample_rate)".to_string());
    }

    // imfs is an (n_imfs x n_samples) matrix (rows = IMFs), column-major.
    let imfs = unsafe {
        let p = prhs[0];
        if p.is_null() || mex_sys::mxIsDouble(p) == 0 {
            return Err("imfs must be a double matrix".to_string());
        }
        let dims = mex_compat::mxGetDimensions(p);
        let n_imfs = *dims;
        let n_samples = *dims.add(1);
        let data_ptr = mex_sys::mxGetPr(p);
        if data_ptr.is_null() {
            return Err("imfs data pointer is null".to_string());
        }
        let data = std::slice::from_raw_parts(data_ptr, n_imfs * n_samples);
        (0..n_imfs)
            .map(|r| (0..n_samples).map(|c| data[c * n_imfs + r]).collect::<Vec<f64>>())
            .collect::<Vec<_>>()
    };

    let sample_rate_ptr = prhs[1];
    if sample_rate_ptr.is_null() {
        return Err("sample_rate must be a non-empty scalar".to_string());
    }

    unsafe {
        if mex_sys::mxIsDouble(sample_rate_ptr) == 0 {
            return Err("sample_rate must be double".to_string());
        }
        let data_ptr = mex_sys::mxGetPr(sample_rate_ptr);
        if data_ptr.is_null() {
            return Err("sample_rate data pointer is null".to_string());
        }
        let sample_rate = *data_ptr;
        if sample_rate <= 0.0 {
            return Err("sample_rate must be positive".to_string());
        }

        let result = hilbert_imf(&imfs, sample_rate)
            .map_err(|e| format!("Hilbert transform failed: {}", e))?;
        hilbert_result_to_matlab(&result)
    }
}

// ---------------------------------------------------------------------------
// Signal Reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct signal from IMFs and residue.
///
/// MATLAB usage:
///   reconstructed = ferromode_mex('reconstruct', result)
pub fn ferromode_reconstruct(
    prhs: &[*mut mex_sys::mxArray],
) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err("Usage: reconstructed = ferromode_mex('reconstruct', result)".to_string());
    }

    let result_ptr = prhs[0];
    if result_ptr.is_null() {
        return Err("result must be a struct from a ferromode decomposition".to_string());
    }

    unsafe {
        if mex_sys::mxIsStruct(result_ptr) == 0 {
            return Err("result must be a struct from a ferromode decomposition".to_string());
        }

        // Get IMFs field
        let imfs_field = mex_compat::mxGetField(result_ptr, 0, b"imfs\0".as_ptr().cast::<i8>());
        if imfs_field.is_null() {
            return Err("result struct missing 'imfs' field".to_string());
        }

        // Read the imfs matrix directly (n_imfs x n_samples, column-major).
        if mex_sys::mxIsDouble(imfs_field) == 0 {
            return Err("imfs must be double".to_string());
        }
        let imf_dims = mex_compat::mxGetDimensions(imfs_field);
        let imf_n_imfs = *imf_dims;
        let imf_n_samples = *imf_dims.add(1);
        let imf_ptr = mex_sys::mxGetPr(imfs_field);
        if imf_ptr.is_null() {
            return Err("imfs data pointer is null".to_string());
        }
        let imf_data = std::slice::from_raw_parts(imf_ptr, imf_n_imfs * imf_n_samples);

        // Get residue field
        let residue_field =
            mex_compat::mxGetField(result_ptr, 0, b"residue\0".as_ptr().cast::<i8>());
        if residue_field.is_null() {
            return Err("result struct missing 'residue' field".to_string());
        }

        if mex_sys::mxIsDouble(residue_field) == 0 {
            return Err("residue must be double".to_string());
        }
        let residue_ptr = mex_sys::mxGetPr(residue_field);
        if residue_ptr.is_null() {
            return Err("residue data pointer is null".to_string());
        }
        let residue_len = mex_sys::mxGetNumberOfElements(residue_field);
        let residue = std::slice::from_raw_parts(residue_ptr, residue_len);

        let n_samples = if imf_n_samples > 0 { imf_n_samples } else { residue.len() };

        let mut reconstructed = vec![0.0f64; n_samples];

        // Sum all IMFs (rows) per sample (column-major: col*n_imfs + row).
        for c in 0..imf_n_samples.min(n_samples) {
            for r in 0..imf_n_imfs {
                reconstructed[c] += imf_data[c * imf_n_imfs + r];
            }
        }

        for (i, &val) in residue.iter().enumerate().take(n_samples) {
            reconstructed[i] += val;
        }

        let output = mex_compat::mxCreateDoubleMatrix(1, n_samples, MEX_REAL);
        if output.is_null() {
            return Err("failed to create output matrix".to_string());
        }

        let dst = mex_sys::mxGetPr(output);
        if !dst.is_null() {
            libc::memcpy(
                dst.cast::<libc::c_void>(),
                reconstructed.as_ptr().cast::<libc::c_void>(),
                n_samples * 8,
            );
        }

        Ok(output)
    }
}
