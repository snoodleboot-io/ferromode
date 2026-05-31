//! Boundary condition strategies for EMD envelope interpolation.
//!
//! Each strategy extends a signal beyond its endpoints so that cubic spline
//! envelope interpolation does not extrapolate blindly past the first and last
//! extrema. The strategy is selected via [`BoundaryConditionType`] and plugged
//! into [`crate::sifting::SiftingConfig`].
//!
//! ## Choosing a strategy
//!
//! | Strategy | Best for |
//! |---|---|
//! | [`BoundaryConditionType::MirrorEven`] | General use; zero-mean or periodic signals (default) |
//! | [`BoundaryConditionType::PalindromeCyclic`] | Signals with trends or nonzero endpoints; most stable |
//! | [`BoundaryConditionType::Periodic`] | Signals known to be periodic (e.g. circular data) |
//! | [`BoundaryConditionType::ARModel`] | Stochastic signals with known AR structure |
//! | [`BoundaryConditionType::CharacteristicWave`] | Signals with clearly repeating waveform shape |
//! | [`BoundaryConditionType::Slope`] | Simple linear extrapolation |
//! | [`BoundaryConditionType::WaveformMatching`] | Quasi-periodic signals, matched interior segment |
//!
//! ## PalindromeCyclic
//!
//! [`BoundaryConditionType::PalindromeCyclic`] is the most stable option for
//! non-periodic signals. It pre-extends the signal to a `2N-1` palindrome
//! **before** the sifting loop and uses a cyclic (periodic) cubic spline for
//! all envelope interpolation. This is handled transparently by [`crate::algorithms::emd::emd`]
//! — set `sifting_config.boundary_condition = BoundaryConditionType::PalindromeCyclic`
//! and everything else is automatic. Output IMFs and residue are trimmed back
//! to the original `N` samples and exact reconstruction is preserved.
//!
//! The utility function [`build_palindrome`] is also exported for direct use.

use serde::{Deserialize, Serialize};

/// Autoregressive model boundary extension using Yule-Walker coefficient estimation.
pub mod ar_model;
/// Characteristic wave boundary extension using the waveform shape near each endpoint.
pub mod characteristic_wave;
/// Mirror (symmetric) boundary extension, both even and odd variants.
pub mod mirror;
/// Palindrome-cyclic boundary extension for stable end-effect suppression.
pub mod palindrome_cyclic;
/// Periodic (tiling) boundary extension for signals known to be cyclic.
pub mod periodic;
/// Slope-based linear extrapolation boundary extension.
pub mod slope;
/// Waveform matching boundary extension using cross-correlation to find similar interior segments.
pub mod waveform_matching;

pub use ar_model::{ARModel, ARModelConfig};
pub use characteristic_wave::{CharacteristicWave, CharacteristicWaveConfig};
pub use mirror::{Mirror, MirrorConfig, MirrorVariant};
pub use palindrome_cyclic::{build_palindrome, PalindromeCyclic};
pub use periodic::{Periodic, PeriodicConfig};
pub use slope::{Slope, SlopeConfig};
pub use waveform_matching::{WaveformMatching, WaveformMatchingConfig};

/// Represents detected extrema in a signal.
#[derive(Debug, Clone)]
pub struct Extrema {
    /// Indices of local maxima in the signal.
    pub maxima_indices: Vec<usize>,
    /// Indices of local minima in the signal.
    pub minima_indices: Vec<usize>,
}

/// Result of extending a signal with boundary conditions.
#[derive(Debug, Clone)]
pub struct ExtendedSignal {
    /// The extended signal values (includes original + boundary extensions)
    pub values: Vec<f64>,
    /// Index in `values` where the original signal starts
    pub original_start: usize,
    /// Index in `values` where the original signal ends (exclusive)
    pub original_end: usize,
}

/// Trait for boundary condition strategies used in EMD envelope computation.
///
/// Each implementation extends a signal beyond its boundaries to enable
/// accurate spline interpolation of extrema. The sifting engine depends on
/// this trait, not concrete implementations (SOLID-DIP).
pub trait BoundaryCondition: Send + Sync {
    /// Extend the signal beyond its boundaries.
    ///
    /// # Arguments
    /// * `signal` — The original signal values
    /// * `extrema` — Detected maxima and minima positions
    ///
    /// # Returns
    /// An `ExtendedSignal` containing the original signal plus boundary extensions.
    /// The original signal must be preserved intact within the extended signal.
    fn extend(&self, signal: &[f64], extrema: &Extrema) -> ExtendedSignal;

    /// Human-readable name of this boundary strategy.
    fn name(&self) -> &str;
}

/// Available boundary condition strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryConditionType {
    /// Characteristic wave extension using the waveform shape near each boundary.
    CharacteristicWave,
    /// Even mirror extension: reflects signal values symmetrically without sign change.
    MirrorEven,
    /// Odd mirror extension: reflects signal values symmetrically with sign inversion.
    MirrorOdd,
    /// Palindrome-cyclic extension: pre-extends the signal to a `2N-1` palindrome at the
    /// EMD level, then sifts with `SplineType::Periodic`. Provides more stable end-effect
    /// suppression than sample mirroring for non-periodic signals.
    PalindromeCyclic,
    /// Periodic extension: tiles the signal on both sides to enforce periodicity.
    Periodic,
    /// Slope extension: linearly extrapolates from the endpoint derivative.
    Slope,
    /// AR model extension: forecasts and backcasts using Yule-Walker AR coefficients.
    ARModel,
    /// Waveform matching extension: finds interior segments similar to each boundary region.
    WaveformMatching,
}

/// Factory function to get a concrete BoundaryCondition implementation.
pub fn get_strategy(bc_type: BoundaryConditionType) -> Box<dyn BoundaryCondition> {
    match bc_type {
        BoundaryConditionType::CharacteristicWave => Box::new(CharacteristicWave::default()),
        BoundaryConditionType::MirrorEven => Box::new(Mirror::even()),
        BoundaryConditionType::MirrorOdd => Box::new(Mirror::odd()),
        BoundaryConditionType::PalindromeCyclic => Box::new(PalindromeCyclic::new()),
        BoundaryConditionType::Periodic => Box::new(Periodic::default()),
        BoundaryConditionType::Slope => Box::new(Slope::default()),
        BoundaryConditionType::ARModel => Box::new(ARModel::default()),
        BoundaryConditionType::WaveformMatching => Box::new(WaveformMatching::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy implementation for testing the trait contract.
    struct IdentityBoundary;

    impl BoundaryCondition for IdentityBoundary {
        fn extend(&self, signal: &[f64], _extrema: &Extrema) -> ExtendedSignal {
            ExtendedSignal {
                values: signal.to_vec(),
                original_start: 0,
                original_end: signal.len(),
            }
        }

        fn name(&self) -> &str {
            "identity"
        }
    }

    #[test]
    fn test_boundary_condition_trait_can_be_implemented() {
        let boundary = IdentityBoundary;
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let extrema = Extrema { maxima_indices: vec![2], minima_indices: vec![0] };

        let result = boundary.extend(&signal, &extrema);

        assert_eq!(result.values, signal);
        assert_eq!(boundary.name(), "identity");
    }

    #[test]
    fn test_extended_signal_preserves_original() {
        let boundary = IdentityBoundary;
        let signal = vec![10.0, 20.0, 30.0];
        let extrema = Extrema { maxima_indices: vec![1], minima_indices: vec![0] };

        let result = boundary.extend(&signal, &extrema);

        let original_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(original_slice, &signal[..]);
        assert_eq!(result.original_start, 0);
        assert_eq!(result.original_end, 3);
    }

    #[test]
    fn test_boundary_condition_type_serializes() {
        let bc_type = BoundaryConditionType::MirrorEven;

        let serialized = serde_json::to_string(&bc_type).unwrap();
        let deserialized: BoundaryConditionType = serde_json::from_str(&serialized).unwrap();

        assert_eq!(bc_type, deserialized);
    }
}
