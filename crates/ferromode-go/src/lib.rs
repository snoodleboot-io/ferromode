//! Go bindings for Ferromode.
//!
//! Thin cdylib wrapper that re-exports the FFI layer from the core ferromode
//! crate. The shared library is loaded by the Go package via cgo.

pub use ferromode::ffi::*;
