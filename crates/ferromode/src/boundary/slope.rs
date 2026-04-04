/// Slope-Based Extension.
///
/// Computes the first derivative at endpoints using finite differences,
/// then extrapolates linearly beyond each boundary to generate artificial extrema.
use crate::boundary::{BoundaryCondition, ExtendedSignal, Extrema};

/// Configuration for slope-based extension.
#[derive(Debug, Clone)]
pub struct SlopeConfig {
    /// Number of samples to extrapolate on each side (default: 10)
    pub extrapolation_length: usize,
}

impl Default for SlopeConfig {
    fn default() -> Self {
        Self { extrapolation_length: 10 }
    }
}

/// Slope-based boundary extension.
pub struct Slope {
    config: SlopeConfig,
}

impl Slope {
    pub fn new(config: SlopeConfig) -> Self {
        Self { config }
    }

    pub fn with_extrapolation_length(length: usize) -> Self {
        Self { config: SlopeConfig { extrapolation_length: length } }
    }

    fn compute_slope_at_start(signal: &[f64]) -> f64 {
        if signal.len() < 2 {
            return 0.0;
        }
        signal[1] - signal[0]
    }

    fn compute_slope_at_end(signal: &[f64]) -> f64 {
        if signal.len() < 2 {
            return 0.0;
        }
        signal[signal.len() - 1] - signal[signal.len() - 2]
    }

    fn extrapolate_left(signal: &[f64], n: usize) -> Vec<f64> {
        if signal.is_empty() {
            return vec![];
        }
        let slope = Self::compute_slope_at_start(signal);
        let start = signal[0];

        (1..=n).map(|i| start - slope * i as f64).rev().collect()
    }

    fn extrapolate_right(signal: &[f64], n: usize) -> Vec<f64> {
        if signal.is_empty() {
            return vec![];
        }
        let slope = Self::compute_slope_at_end(signal);
        let end = signal[signal.len() - 1];

        (1..=n).map(|i| end + slope * i as f64).collect()
    }
}

impl BoundaryCondition for Slope {
    fn extend(&self, signal: &[f64], _extrema: &Extrema) -> ExtendedSignal {
        if signal.is_empty() {
            return ExtendedSignal { values: vec![], original_start: 0, original_end: 0 };
        }

        let left_ext = Self::extrapolate_left(signal, self.config.extrapolation_length);
        let right_ext = Self::extrapolate_right(signal, self.config.extrapolation_length);

        let mut values = Vec::with_capacity(left_ext.len() + signal.len() + right_ext.len());
        values.extend_from_slice(&left_ext);
        values.extend_from_slice(signal);
        values.extend_from_slice(&right_ext);

        let original_start = left_ext.len();
        let original_end = original_start + signal.len();

        ExtendedSignal { values, original_start, original_end }
    }

    fn name(&self) -> &str {
        "slope"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slope_on_monotone_increasing() {
        let signal: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let extrema = Extrema { maxima_indices: vec![19], minima_indices: vec![0] };

        let strategy = Slope::with_extrapolation_length(5);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_slope_on_monotone_decreasing() {
        let signal: Vec<f64> = (0..20).map(|i| (19 - i) as f64).collect();
        let extrema = Extrema { maxima_indices: vec![0], minima_indices: vec![19] };

        let strategy = Slope::with_extrapolation_length(5);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
    }

    #[test]
    fn test_slope_preserves_original() {
        let signal = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let extrema = Extrema { maxima_indices: vec![1, 3], minima_indices: vec![0, 2, 4] };

        let strategy = Slope::default();
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
    }

    #[test]
    fn test_slope_linear_extrapolation() {
        let signal = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let extrema = Extrema { maxima_indices: vec![4], minima_indices: vec![0] };

        let strategy = Slope::with_extrapolation_length(3);
        let result = strategy.extend(&signal, &extrema);

        let left_ext = &result.values[..result.original_start];
        assert_eq!(left_ext.len(), 3);

        let right_ext = &result.values[result.original_end..];
        assert_eq!(right_ext.len(), 3);

        assert!(right_ext[0] > 4.0);
    }

    #[test]
    fn test_slope_empty_signal() {
        let signal: Vec<f64> = vec![];
        let extrema = Extrema { maxima_indices: vec![], minima_indices: vec![] };

        let strategy = Slope::default();
        let result = strategy.extend(&signal, &extrema);

        assert!(result.values.is_empty());
    }

    #[test]
    fn test_slope_single_element() {
        let signal = vec![5.0];
        let extrema = Extrema { maxima_indices: vec![0], minima_indices: vec![] };

        let strategy = Slope::default();
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() >= signal.len());
    }

    #[test]
    fn test_slope_name() {
        let strategy = Slope::default();
        assert_eq!(strategy.name(), "slope");
    }
}
