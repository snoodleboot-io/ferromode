//! Analysis module for signal complexity metrics.
//!
//! This module provides entropy-based analysis functions for measuring
//! signal complexity and disorder. It includes spectral, permutation,
//! and sample entropy implementations.

pub mod entropy;
pub mod time_frequency;

pub use entropy::{
    permutation_entropy, permutation_entropy_normalized, sample_entropy, spectral_entropy,
    spectral_entropy_normalized, EntropyAnalysis, EntropyMetric,
};

pub use time_frequency::complexity_score;
