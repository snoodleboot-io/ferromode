//! MATLAB/Octave MEX bindings for Ferromode.
//!
//! Pure-wrap contract: MEX layer is ONLY marshalling, no algorithm logic.

pub mod functions;
pub mod marshalling;

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
    let result = std::panic::catch_unwind(|| {
        if nrhs == 0 {
            mex_error(
                "ferromode:noInput",
                "No input arguments provided. Use 'help ferromode' for usage.",
            );
            return;
        }

        let prhs_slice = std::slice::from_raw_parts(prhs, nrhs as usize);
        let func_name = match marshalling::mx_array_to_string(prhs_slice[0]) {
            Ok(s) => s,
            Err(e) => {
                mex_error(
                    "ferromode:invalidFunction",
                    &format!("First argument must be a function name string: {e}"),
                );
                return;
            }
        };

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
            other => {
                mex_error(
                    "ferromode:unknownFunction",
                    &format!(
                        "Unknown function '{}'. Valid: emd, eemd, ceemd, ceemdan, iceemdan, memd, namemd, vmd, hilbert, reconstruct",
                        other
                    ),
                );
                return;
            }
        };

        match output_ptr {
            Ok(ptr) => {
                if nlhs > 0 && !plhs.is_null() {
                    *plhs = ptr;
                }
            }
            Err(e) => {
                mex_error("ferromode:algorithmError", &e);
            }
        }
    });

    if result.is_err() {
        mex_error("ferromode:panic", "Internal error: Rust panic in MEX function");
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
