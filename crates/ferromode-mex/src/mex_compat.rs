//! MEX API compatibility layer
//!
//! The mex-sys crate uses version-specific suffixes (_700, _730, etc.)
//! This module provides version-agnostic wrappers for common MEX functions.

use mex_sys::{mxArray, mxComplexity};
use std::os::raw::c_char;

/// Create a double matrix (version-agnostic)
pub unsafe fn mxCreateDoubleMatrix(m: usize, n: usize, complexity: mxComplexity) -> *mut mxArray {
    mex_sys::mxCreateDoubleMatrix_730(m, n, complexity)
}

/// Create a struct array (version-agnostic)
pub unsafe fn mxCreateStructArray(
    ndim: usize,
    dims: *const usize,
    nfields: i32,
    fieldnames: *const *const c_char,
) -> *mut mxArray {
    mex_sys::mxCreateStructArray_730(ndim, dims, nfields, fieldnames as *mut *const c_char)
}

/// Create a struct matrix (version-agnostic)
pub unsafe fn mxCreateStructMatrix(
    m: usize,
    n: usize,
    nfields: i32,
    fieldnames: *const *const c_char,
) -> *mut mxArray {
    mex_sys::mxCreateStructMatrix_730(m, n, nfields, fieldnames as *mut *const c_char)
}

/// Get field (version-agnostic)
pub unsafe fn mxGetField(
    pa: *const mxArray,
    index: usize,
    fieldname: *const c_char,
) -> *mut mxArray {
    mex_sys::mxGetField_730(pa, index, fieldname)
}

/// Set field (version-agnostic)
pub unsafe fn mxSetField(
    pa: *mut mxArray,
    index: usize,
    fieldname: *const c_char,
    value: *mut mxArray,
) {
    mex_sys::mxSetField_730(pa, index, fieldname, value)
}

/// Get dimensions (version-agnostic)
pub unsafe fn mxGetDimensions(pa: *const mxArray) -> *const usize {
    mex_sys::mxGetDimensions_730(pa)
}

/// Get number of dimensions (version-agnostic)
pub unsafe fn mxGetNumberOfDimensions(pa: *const mxArray) -> usize {
    mex_sys::mxGetNumberOfDimensions_730(pa) as usize
}

/// Get number of elements (version-agnostic)
pub unsafe fn mxGetNumberOfElements(pa: *const mxArray) -> usize {
    mex_sys::mxGetNumberOfElements(pa)
}

/// mxREAL constant (for compatibility)
pub const MEX_REAL: mxComplexity = mxComplexity::mxREAL;
