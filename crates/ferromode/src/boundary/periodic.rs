/// Periodic / Cyclic Extension (Zeng & He 2004).
///
/// Concatenates even-extended and odd-extended copies to build a periodic series.
/// Uses periodic cubic spline boundary conditions for envelope computation.
///
/// Reference: Zeng & He (2004), "A new method to eliminate end effects in EMD"
use crate::boundary::{BoundaryCondition, ExtendedSignal, Extrema};

/// Configuration for periodic extension.
#[derive(Debug, Clone)]
pub struct PeriodicConfig {
    /// Number of periods to extend on each side (default: 1)
    pub periods: usize,
}

impl Default for PeriodicConfig {
    fn default() -> Self {
        Self { periods: 1 }
    }
}

/// Periodic/cyclic boundary extension.
pub struct Periodic {
    config: PeriodicConfig,
}

impl Default for Periodic {
    fn default() -> Self {
        Self { config: PeriodicConfig::default() }
    }
}

impl Periodic {
    pub fn new(config: PeriodicConfig) -> Self {
        Self { config }
    }

    pub fn with_periods(periods: usize) -> Self {
        Self { config: PeriodicConfig { periods } }
    }

    fn build_periodic_extension(signal: &[f64], periods: usize) -> (Vec<f64>, Vec<f64>) {
        let mut left_ext = Vec::new();
        let mut right_ext = Vec::new();

        for _ in 0..periods {
            left_ext.extend_from_slice(signal);
            right_ext.extend_from_slice(signal);
        }

        (left_ext, right_ext)
    }
}

impl BoundaryCondition for Periodic {
    fn extend(&self, signal: &[f64], _extrema: &Extrema) -> ExtendedSignal {
        if signal.is_empty() {
            return ExtendedSignal { values: vec![], original_start: 0, original_end: 0 };
        }

        let (left_ext, right_ext) = Self::build_periodic_extension(signal, self.config.periods);

        let mut values = Vec::with_capacity(left_ext.len() + signal.len() + right_ext.len());
        values.extend_from_slice(&left_ext);
        values.extend_from_slice(signal);
        values.extend_from_slice(&right_ext);

        let original_start = left_ext.len();
        let original_end = original_start + signal.len();

        ExtendedSignal { values, original_start, original_end }
    }

    fn name(&self) -> &str {
        "periodic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodic_on_known_periodic_signal() {
        let n = 100;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin()).collect();

        let extrema = Extrema { maxima_indices: vec![25, 75], minima_indices: vec![0, 50] };

        let strategy = Periodic::with_periods(1);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_periodic_preserves_original() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let extrema = Extrema { maxima_indices: vec![4], minima_indices: vec![0] };

        let strategy = Periodic::default();
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
    }

    #[test]
    fn test_periodic_multiple_periods() {
        let signal = vec![1.0, 2.0, 3.0];
        let extrema = Extrema { maxima_indices: vec![2], minima_indices: vec![0] };

        let strategy = Periodic::with_periods(3);
        let result = strategy.extend(&signal, &extrema);

        let expected_left_len = 3 * 3;
        let expected_right_len = 3 * 3;
        assert_eq!(result.original_start, expected_left_len);
        assert_eq!(result.original_end, expected_left_len + signal.len());
        assert_eq!(result.values.len(), expected_left_len + signal.len() + expected_right_len);
    }

    #[test]
    fn test_periodic_on_composite_signal() {
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * std::f64::consts::PI * t * 3.0).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * t * 7.0).sin()
            })
            .collect();

        let extrema = Extrema {
            maxima_indices: vec![10, 40, 70, 100, 130, 160, 190],
            minima_indices: vec![25, 55, 85, 115, 145, 175],
        };

        let strategy = Periodic::with_periods(2);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_periodic_empty_signal() {
        let signal: Vec<f64> = vec![];
        let extrema = Extrema { maxima_indices: vec![], minima_indices: vec![] };

        let strategy = Periodic::default();
        let result = strategy.extend(&signal, &extrema);

        assert!(result.values.is_empty());
    }

    #[test]
    fn test_periodic_name() {
        let strategy = Periodic::default();
        assert_eq!(strategy.name(), "periodic");
    }
}
