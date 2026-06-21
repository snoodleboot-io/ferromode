//! C/C++ bindings for Ferromode.
//!
//! This crate is a thin cdylib that re-exports the stable C ABI from the core
//! `ferromode` crate (`ferromode::ffi`). The C++ convenience layer lives in
//! `include/ferromode.hpp`, which wraps these `extern "C"` symbols with RAII
//! helpers. All algorithm logic stays in the core crate — this is pure FFI.

pub use ferromode::ffi::*;
