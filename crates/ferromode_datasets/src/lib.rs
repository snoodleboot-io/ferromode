//! # Ferromode Datasets - Reference Signal Library
//!
//! Comprehensive signal library for EMD validation and benchmarking.
//! - 500+ synthetic signals with ground truth
//! - 500+ real signals from multiple domains
//! - Standardized metadata for all signals

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Signal metadata and loading functions
pub mod metadata;

// Re-export key types
pub use metadata::{MetadataCollection, SignalMetadata, SignalSource};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_works() {
        assert!(!crate::VERSION.is_empty());
    }
}
