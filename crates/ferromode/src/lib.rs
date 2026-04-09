#![warn(missing_docs)]

//! Ferromode — core signal processing and mode analysis library.
//!
//! This crate provides the foundational types and algorithms for
//! ferromagnetic mode decomposition and analysis.

pub mod adapters;
pub mod algorithms;
pub mod boundary;
pub mod error;
pub mod extrema;
pub mod ffi;
pub mod hilbert;
pub mod metrics;
pub mod ml;
pub mod multivariate;
pub mod sifting;
pub mod spline;
pub mod types;
pub mod mode_mixing;
