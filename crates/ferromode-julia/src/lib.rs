//! Julia bindings for Ferromode.
//!
//! This crate is a thin cdylib wrapper that re-exports the FFI layer
//! from the core ferromode crate. The shared library is loaded by
//! the Julia package via `Libdl`.

// Re-export all FFI symbols from the core crate
pub use ferromode::ffi::*;
