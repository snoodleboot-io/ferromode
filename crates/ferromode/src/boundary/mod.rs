use serde::{Deserialize, Serialize};

pub mod ar_model;
pub mod characteristic_wave;
pub mod extrema_mirror;
pub mod mirror;
pub mod periodic;
pub mod slope;
pub mod waveform_matching;

pub use ar_model::{ARModel, ARModelConfig};
pub use characteristic_wave::{CharacteristicWave, CharacteristicWaveConfig};
pub use extrema_mirror::{build_envelope_knots, ExtremasMirror, ExtremasMirrorConfig};
pub use mirror::{Mirror, MirrorConfig, MirrorVariant};
pub use periodic::{Periodic, PeriodicConfig};
pub use slope::{Slope, SlopeConfig};
pub use waveform_matching::{WaveformMatching, WaveformMatchingConfig};

/// Represents detected extrema in a signal.
#[derive(Debug, Clone)]
pub struct Extrema {
    pub maxima_indices: Vec<usize>,
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
    CharacteristicWave,
    /// Reflect the `nbsym` outermost extrema across each boundary (Huang 1998).
    /// Matches the default behaviour of PyEMD. Best general-purpose choice.
    ExtremasMirror,
    MirrorEven,
    MirrorOdd,
    Periodic,
    Slope,
    ARModel,
    WaveformMatching,
}

/// Factory function to get a concrete BoundaryCondition implementation.
pub fn get_strategy(bc_type: BoundaryConditionType) -> Box<dyn BoundaryCondition> {
    match bc_type {
        BoundaryConditionType::CharacteristicWave => Box::new(CharacteristicWave::default()),
        BoundaryConditionType::ExtremasMirror => Box::new(ExtremasMirror::default()),
        BoundaryConditionType::MirrorEven => Box::new(Mirror::even()),
        BoundaryConditionType::MirrorOdd => Box::new(Mirror::odd()),
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
