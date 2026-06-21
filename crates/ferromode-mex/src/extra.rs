//! Differentiable EMD and streaming decomposition for the MEX binding.
//!
//! Pure-wrap contract: ONLY marshalling, no algorithm logic. Streaming handles
//! are kept in a process-local registry keyed by a small integer id (returned
//! to MATLAB/Octave as a double) so no raw pointers cross the MEX boundary.

use crate::marshalling::{mx_array_to_signal, parse_emd_config};
use crate::mex_compat::{self, MEX_REAL};
use ferromode::adapters::streaming::{ArModel, StreamingDecomposer};
use ferromode::ml::differentiable::DifferentiableEmd;
use ferromode::types::Signal;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Small marshalling helpers (output side)
// ---------------------------------------------------------------------------

unsafe fn row_vector(data: &[f64]) -> *mut mex_sys::mxArray {
    let mx = mex_compat::mxCreateDoubleMatrix(1, data.len(), MEX_REAL);
    if !mx.is_null() && !data.is_empty() {
        let dst = mex_sys::mxGetPr(mx);
        if !dst.is_null() {
            libc::memcpy(
                dst.cast::<libc::c_void>(),
                data.as_ptr().cast::<libc::c_void>(),
                data.len() * 8,
            );
        }
    }
    mx
}

unsafe fn imf_matrix(imfs: &[Vec<f64>]) -> *mut mex_sys::mxArray {
    let n_imfs = imfs.len();
    let n_samples = imfs.first().map_or(0, |v| v.len());
    if n_imfs == 0 || n_samples == 0 {
        return mex_compat::mxCreateDoubleMatrix(0, 0, MEX_REAL);
    }
    // Column-major (n_imfs x n_samples): element (row=imf, col=sample) at
    // col*n_imfs + row.
    let mut flat = vec![0.0f64; n_imfs * n_samples];
    for (r, imf) in imfs.iter().enumerate() {
        for (c, &v) in imf.iter().enumerate() {
            flat[c * n_imfs + r] = v;
        }
    }
    let mx = mex_compat::mxCreateDoubleMatrix(n_imfs, n_samples, MEX_REAL);
    if !mx.is_null() {
        let dst = mex_sys::mxGetPr(mx);
        if !dst.is_null() {
            libc::memcpy(
                dst.cast::<libc::c_void>(),
                flat.as_ptr().cast::<libc::c_void>(),
                flat.len() * 8,
            );
        }
    }
    mx
}

/// Build a 1x1 struct from (field name, value mxArray) pairs.
unsafe fn make_struct(fields: &[(&str, *mut mex_sys::mxArray)]) -> *mut mex_sys::mxArray {
    let names: Vec<CString> = fields.iter().map(|(k, _)| CString::new(*k).unwrap()).collect();
    let name_ptrs: Vec<*const std::os::raw::c_char> = names.iter().map(|s| s.as_ptr()).collect();
    let st = mex_compat::mxCreateStructMatrix(1, 1, name_ptrs.len() as i32, name_ptrs.as_ptr());
    if !st.is_null() {
        for (i, (_, v)) in fields.iter().enumerate() {
            mex_compat::mxSetField(st, 0, names[i].as_ptr(), *v);
        }
    }
    st
}

unsafe fn read_scalar(ptr: *mut mex_sys::mxArray) -> Option<f64> {
    if ptr.is_null() || mex_sys::mxIsDouble(ptr) == 0 {
        return None;
    }
    let p = mex_sys::mxGetPr(ptr);
    if p.is_null() {
        None
    } else {
        Some(*p)
    }
}

// ---------------------------------------------------------------------------
// Differentiable EMD
// ---------------------------------------------------------------------------

/// `result = ferromode_mex('emd_forward', signal, config)`
pub fn ferromode_emd_forward(
    prhs: &[*mut mex_sys::mxArray],
) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.is_empty() {
        return Err("Usage: ferromode_mex('emd_forward', signal, config)".to_string());
    }
    let signal = mx_array_to_signal(prhs[0])?;
    let config = parse_emd_config(prhs.get(1).copied().unwrap_or(std::ptr::null_mut()))?;
    let ctx = DifferentiableEmd::new(config)
        .forward(&signal)
        .map_err(|e| format!("emd_forward failed: {e}"))?;
    let num_sifts: Vec<f64> = ctx.num_sifts.iter().map(|&n| n as f64).collect();
    unsafe {
        Ok(make_struct(&[
            ("imfs", imf_matrix(&ctx.imfs)),
            ("residue", row_vector(&ctx.residue)),
            ("num_sifts", row_vector(&num_sifts)),
            ("reconstruction_error", mex_sys::mxCreateDoubleScalar(ctx.reconstruction_error())),
        ]))
    }
}

/// `grad = ferromode_mex('emd_backward', grad_imfs, signal)`
/// `grad_imfs` is an (n_imfs x n_samples) matrix (one gradient IMF per row, to
/// match the `imfs` field that `emd_forward` returns). Placeholder: averages.
pub fn ferromode_emd_backward(
    prhs: &[*mut mex_sys::mxArray],
) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.len() < 2 {
        return Err("Usage: ferromode_mex('emd_backward', grad_imfs, signal)".to_string());
    }
    let signal = mx_array_to_signal(prhs[1])?;
    let n = signal.len();
    if n == 0 {
        return Err("signal must not be empty".to_string());
    }
    // Read grad_imfs as a 2D matrix and sum its rows (column-major storage).
    let g = prhs[0];
    unsafe {
        if g.is_null() || mex_sys::mxIsDouble(g) == 0 {
            return Err("grad_imfs must be a double matrix".to_string());
        }
        let ndims = mex_compat::mxGetNumberOfDimensions(g);
        if ndims < 2 {
            return Err("grad_imfs must be a 2D matrix".to_string());
        }
        let dims = mex_compat::mxGetDimensions(g);
        let n_rows = *dims; // n_imfs
        let n_cols = *dims.add(1); // n_samples
        if n_rows == 0 || n_cols != n {
            return Err("grad_imfs must be (n_imfs x n_samples) matching the signal".to_string());
        }
        let data_ptr = mex_sys::mxGetPr(g);
        if data_ptr.is_null() {
            return Err("grad_imfs data pointer is null".to_string());
        }
        let data = std::slice::from_raw_parts(data_ptr, n_rows * n_cols);
        let mut out = vec![0.0f64; n];
        for (c, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for r in 0..n_rows {
                acc += data[c * n_rows + r];
            }
            *o = acc / n_rows as f64;
        }
        Ok(row_vector(&out))
    }
}

// ---------------------------------------------------------------------------
// Streaming (process-local registry keyed by integer id)
// ---------------------------------------------------------------------------

fn registry() -> &'static Mutex<HashMap<u64, StreamingDecomposer>> {
    static REG: OnceLock<Mutex<HashMap<u64, StreamingDecomposer>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_id() -> u64 {
    static COUNTER: OnceLock<Mutex<u64>> = OnceLock::new();
    let c = COUNTER.get_or_init(|| Mutex::new(0));
    let mut g = c.lock().unwrap();
    *g += 1;
    *g
}

/// `id = ferromode_mex('streaming_new', config, buffer_size, ar_order)`
pub fn ferromode_streaming_new(
    prhs: &[*mut mex_sys::mxArray],
) -> Result<*mut mex_sys::mxArray, String> {
    let config = parse_emd_config(prhs.first().copied().unwrap_or(std::ptr::null_mut()))?;
    let buffer_size = unsafe { prhs.get(1).copied().and_then(|p| read_scalar(p)) }
        .map_or(4096, |v| if v <= 0.0 { 4096 } else { v as usize });
    let ar_order = unsafe { prhs.get(2).copied().and_then(|p| read_scalar(p)) }
        .map_or(3, |v| if v <= 0.0 { 3 } else { v as usize });
    let predictor = ArModel::new(ar_order).map_err(|e| format!("AR model: {e}"))?;
    let dec = StreamingDecomposer::new(config, Box::new(predictor), buffer_size)
        .map_err(|e| format!("streaming init: {e}"))?;
    let id = next_id();
    registry().lock().unwrap().insert(id, dec);
    unsafe { Ok(mex_sys::mxCreateDoubleScalar(id as f64)) }
}

/// `result = ferromode_mex('streaming_decompose_chunk', id, chunk)`
pub fn ferromode_streaming_decompose_chunk(
    prhs: &[*mut mex_sys::mxArray],
) -> Result<*mut mex_sys::mxArray, String> {
    if prhs.len() < 2 {
        return Err("Usage: ferromode_mex('streaming_decompose_chunk', id, chunk)".to_string());
    }
    let id = unsafe { read_scalar(prhs[0]) }.ok_or("id must be a scalar")? as u64;
    let chunk = mx_array_to_signal(prhs[1])?;
    let signal = Signal::from_slice(&chunk).map_err(|e| format!("invalid chunk: {e}"))?;
    let mut reg = registry().lock().unwrap();
    let dec = reg.get_mut(&id).ok_or("unknown streaming handle id")?;
    let res = dec.decompose_chunk(&signal).map_err(|e| format!("chunk failed: {e}"))?;
    unsafe {
        Ok(make_struct(&[
            ("imfs", imf_matrix(&res.imfs)),
            ("residue", row_vector(&res.remainder)),
            ("spectral_entropy", mex_sys::mxCreateDoubleScalar(res.metrics.spectral_entropy)),
            ("stationarity_score", mex_sys::mxCreateDoubleScalar(res.metrics.stationarity_score)),
            ("extrema_spacing_cv", mex_sys::mxCreateDoubleScalar(res.metrics.extrema_spacing_cv)),
        ]))
    }
}

/// `ferromode_mex('streaming_reset', id)`
pub fn ferromode_streaming_reset(
    prhs: &[*mut mex_sys::mxArray],
) -> Result<*mut mex_sys::mxArray, String> {
    let id = unsafe { read_scalar(prhs.first().copied().unwrap_or(std::ptr::null_mut())) }
        .ok_or("id must be a scalar")? as u64;
    let mut reg = registry().lock().unwrap();
    reg.get_mut(&id).ok_or("unknown streaming handle id")?.reset();
    unsafe { Ok(mex_compat::mxCreateDoubleMatrix(0, 0, MEX_REAL)) }
}

/// `ferromode_mex('streaming_free', id)`
pub fn ferromode_streaming_free(
    prhs: &[*mut mex_sys::mxArray],
) -> Result<*mut mex_sys::mxArray, String> {
    let id = unsafe { read_scalar(prhs.first().copied().unwrap_or(std::ptr::null_mut())) }
        .ok_or("id must be a scalar")? as u64;
    registry().lock().unwrap().remove(&id);
    unsafe { Ok(mex_compat::mxCreateDoubleMatrix(0, 0, MEX_REAL)) }
}
