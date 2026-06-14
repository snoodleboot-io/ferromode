#![warn(missing_docs)]

//! Ferromode — core signal processing and mode analysis library.
//!
//! This crate provides the foundational types and algorithms for
//! ferromagnetic mode decomposition and analysis.

/// Adapter implementations for GPU acceleration and ML-based boundary prediction.
pub mod adapters;
/// Core decomposition algorithms (EMD, EEMD, CEEMD, CEEMDAN, ICEEMDAN, VMD).
pub mod algorithms;
/// Entropy and spectral analysis utilities for IMF interpretation.
pub mod analysis;
/// Boundary condition strategies for signal endpoint handling during sifting.
pub mod boundary;
/// Error types returned by ferromode operations.
pub mod error;
/// Extrema detection routines for identifying signal maxima and minima.
pub mod extrema;
/// C-compatible FFI layer for interoperability with Julia, Python ctypes, and C.
pub mod ffi;
/// Hilbert transform and instantaneous frequency/amplitude computation.
pub mod hilbert;
/// Imf quality metrics including orthogonality and energy density.
pub mod metrics;
/// Machine learning utilities used internally by the library.
pub mod ml;
/// Mode mixing detection and mitigation utilities.
pub mod mode_mixing;
/// Multivariate decomposition algorithms (MEMD, NA-MEMD).
pub mod multivariate;
/// Sifting loop configuration and execution for IMF extraction.
pub mod sifting;
/// Cubic spline interpolation for envelope fitting.
pub mod spline;
/// Core public types: Signal, ImfCollection, DecompositionResult, and related enums.
pub mod types;

pub use analysis::{
    permutation_entropy, permutation_entropy_normalized, sample_entropy, spectral_entropy,
    spectral_entropy_normalized, EntropyAnalysis,
};
