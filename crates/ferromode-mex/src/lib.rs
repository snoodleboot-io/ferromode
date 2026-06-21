//! MATLAB/Octave MEX bindings for Ferromode.
//!
//! Pure-wrap contract: MEX layer is ONLY marshalling, no algorithm logic.
// MEX API uses MATLAB's camelCase naming and raw-pointer conventions throughout.
#![allow(non_snake_case, clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]

pub mod extra;
pub mod functions;
pub mod marshalling;
pub mod mex_compat;

use std::ffi::CString;

/// Entry point for the MEX function.
///
/// This is the C entry point that MATLAB/Octave calls.
/// Signature: void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
#[no_mangle]
pub unsafe extern "C" fn mexFunction(
    nlhs: std::os::raw::c_int,
    plhs: *mut *mut mex_sys::mxArray,
    nrhs: std::os::raw::c_int,
    prhs: *const *mut mex_sys::mxArray,
) {
    // IMPORTANT: never call mexErrMsgIdAndTxt (a C++ throw / longjmp) while Rust
    // frames are live — it unwinds through Rust and aborts. The closure only
    // *returns* errors; mex_error is invoked at the flat top frame below, after
    // catch_unwind has unwound every Rust frame.
    type Outcome = Result<*mut mex_sys::mxArray, (String, String)>;
    let result: std::thread::Result<Outcome> = std::panic::catch_unwind(|| {
        if nrhs == 0 {
            return Err((
                "ferromode:noInput".to_string(),
                "No input arguments provided. Use 'help ferromode' for usage.".to_string(),
            ));
        }

        let prhs_slice = std::slice::from_raw_parts(prhs, nrhs as usize);
        let func_name = marshalling::mx_array_to_string(prhs_slice[0]).map_err(|e| {
            (
                "ferromode:invalidFunction".to_string(),
                format!("First argument must be a function name string: {e}"),
            )
        })?;

        let args = &prhs_slice[1..];

        let output_ptr = match func_name.as_str() {
            "emd" => functions::ferromode_emd(args),
            "eemd" => functions::ferromode_eemd(args),
            "ceemd" => functions::ferromode_ceemd(args),
            "ceemdan" => functions::ferromode_ceemdan(args),
            "iceemdan" => functions::ferromode_iceemdan(args),
            "memd" => functions::ferromode_memd(args),
            "namemd" => functions::ferromode_namemd(args),
            "vmd" => functions::ferromode_vmd(args),
            "hilbert" => functions::ferromode_hilbert(args),
            "reconstruct" => functions::ferromode_reconstruct(args),
            "emd_forward" => extra::ferromode_emd_forward(args),
            "emd_backward" => extra::ferromode_emd_backward(args),
            "streaming_new" => extra::ferromode_streaming_new(args),
            "streaming_decompose_chunk" => extra::ferromode_streaming_decompose_chunk(args),
            "streaming_reset" => extra::ferromode_streaming_reset(args),
            "streaming_free" => extra::ferromode_streaming_free(args),
            other => Err(format!(
                "Unknown function '{}'. Valid: emd, eemd, ceemd, ceemdan, iceemdan, memd, namemd, vmd, hilbert, reconstruct, emd_forward, emd_backward, streaming_new, streaming_decompose_chunk, streaming_reset, streaming_free",
                other
            )),
        };

        output_ptr.map_err(|e| ("ferromode:algorithmError".to_string(), e))
    });

    // Flat frame: safe to raise the MATLAB/Octave error here.
    match result {
        Ok(Ok(ptr)) => {
            if nlhs > 0 && !plhs.is_null() {
                *plhs = ptr;
            }
        }
        Ok(Err((id, msg))) => mex_error(&id, &msg),
        Err(_) => mex_error("ferromode:panic", "Internal error: Rust panic in MEX function"),
    }
}

/// Raise a MATLAB error message.
fn mex_error(id: &str, msg: &str) {
    let id_c = CString::new(id).unwrap_or_default();
    let msg_c = CString::new(msg).unwrap_or_default();
    unsafe {
        mex_sys::mexErrMsgIdAndTxt(id_c.as_ptr(), msg_c.as_ptr());
    }
}
