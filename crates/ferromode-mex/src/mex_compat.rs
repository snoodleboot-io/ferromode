//! MEX API compatibility layer
//!
//! The mex-sys crate uses version-specific suffixes (_700, _730, etc.)
//! This module provides version-agnostic wrappers for common MEX functions.

use mex_sys::{mxArray, mxComplexity};
use std::os::raw::c_char;

// MATLAB R2018a+ exports version-suffixed symbols (mx*_730). Octave exports the
// classic non-suffixed names. Build with FERROMODE_OCTAVE to target Octave
// (build.rs sets `cfg(octave)`); the wrappers below dispatch accordingly.
#[cfg(octave)]
extern "C" {
    #[link_name = "mxCreateDoubleMatrix"]
    fn oct_create_double_matrix(m: usize, n: usize, complexity: mxComplexity) -> *mut mxArray;
    #[link_name = "mxCreateStructArray"]
    fn oct_create_struct_array(
        ndim: usize,
        dims: *const usize,
        nfields: i32,
        fieldnames: *mut *const c_char,
    ) -> *mut mxArray;
    #[link_name = "mxCreateStructMatrix"]
    fn oct_create_struct_matrix(
        m: usize,
        n: usize,
        nfields: i32,
        fieldnames: *mut *const c_char,
    ) -> *mut mxArray;
    #[link_name = "mxGetField"]
    fn oct_get_field(pa: *const mxArray, index: usize, fieldname: *const c_char) -> *mut mxArray;
    #[link_name = "mxSetField"]
    fn oct_set_field(pa: *mut mxArray, index: usize, fieldname: *const c_char, value: *mut mxArray);
    #[link_name = "mxGetDimensions"]
    fn oct_get_dimensions(pa: *const mxArray) -> *const usize;
    #[link_name = "mxGetNumberOfDimensions"]
    fn oct_get_number_of_dimensions(pa: *const mxArray) -> usize;
}

/// Create a double matrix (version-agnostic)
pub unsafe fn mxCreateDoubleMatrix(m: usize, n: usize, complexity: mxComplexity) -> *mut mxArray {
    #[cfg(octave)]
    {
        oct_create_double_matrix(m, n, complexity)
    }
    #[cfg(not(octave))]
    {
        mex_sys::mxCreateDoubleMatrix_730(m, n, complexity)
    }
}

/// Create a struct array (version-agnostic)
pub unsafe fn mxCreateStructArray(
    ndim: usize,
    dims: *const usize,
    nfields: i32,
    fieldnames: *const *const c_char,
) -> *mut mxArray {
    #[cfg(octave)]
    {
        oct_create_struct_array(ndim, dims, nfields, fieldnames as *mut *const c_char)
    }
    #[cfg(not(octave))]
    {
        mex_sys::mxCreateStructArray_730(ndim, dims, nfields, fieldnames as *mut *const c_char)
    }
}

/// Create a struct matrix (version-agnostic)
pub unsafe fn mxCreateStructMatrix(
    m: usize,
    n: usize,
    nfields: i32,
    fieldnames: *const *const c_char,
) -> *mut mxArray {
    #[cfg(octave)]
    {
        oct_create_struct_matrix(m, n, nfields, fieldnames as *mut *const c_char)
    }
    #[cfg(not(octave))]
    {
        mex_sys::mxCreateStructMatrix_730(m, n, nfields, fieldnames as *mut *const c_char)
    }
}

/// Get field (version-agnostic)
pub unsafe fn mxGetField(
    pa: *const mxArray,
    index: usize,
    fieldname: *const c_char,
) -> *mut mxArray {
    #[cfg(octave)]
    {
        oct_get_field(pa, index, fieldname)
    }
    #[cfg(not(octave))]
    {
        mex_sys::mxGetField_730(pa, index, fieldname)
    }
}

/// Set field (version-agnostic)
pub unsafe fn mxSetField(
    pa: *mut mxArray,
    index: usize,
    fieldname: *const c_char,
    value: *mut mxArray,
) {
    #[cfg(octave)]
    {
        oct_set_field(pa, index, fieldname, value);
    }
    #[cfg(not(octave))]
    {
        mex_sys::mxSetField_730(pa, index, fieldname, value);
    }
}

/// Get dimensions (version-agnostic)
pub unsafe fn mxGetDimensions(pa: *const mxArray) -> *const usize {
    #[cfg(octave)]
    {
        oct_get_dimensions(pa)
    }
    #[cfg(not(octave))]
    {
        mex_sys::mxGetDimensions_730(pa)
    }
}

/// Get number of dimensions (version-agnostic)
pub unsafe fn mxGetNumberOfDimensions(pa: *const mxArray) -> usize {
    #[cfg(octave)]
    {
        oct_get_number_of_dimensions(pa)
    }
    #[cfg(not(octave))]
    {
        mex_sys::mxGetNumberOfDimensions_730(pa) as usize
    }
}

/// Get number of elements (version-agnostic)
pub unsafe fn mxGetNumberOfElements(pa: *const mxArray) -> usize {
    mex_sys::mxGetNumberOfElements(pa)
}

/// mxREAL constant (for compatibility)
pub const MEX_REAL: mxComplexity = mxComplexity::mxREAL;
