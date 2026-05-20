/// Mirror Extending (Zhao & Huang 2001).
///
/// Extends the signal by appending a time-reversed copy on each side:
///   [ signal_reversed | signal | signal_reversed ]
///
/// The reversal ensures the values match at both seams (signal[0] touches
/// reversed[last] = signal[0], signal[N-1] touches reversed[0] = signal[N-1]),
/// making the combined sequence naturally smooth and quasi-periodic.
///
/// The sifting engine detects extrema in the full extended signal and then fits
/// a **periodic cubic spline** (not not-a-knot) through those extrema.  The
/// periodic end-condition is correct here because the palindrome structure
/// means the extended extrema sequence is genuinely closed — no free endpoint
/// derivative assumptions are needed.
///
/// Reference: Zhao & Huang (2001), "Mirror extending and circular spline
/// function for empirical mode decomposition method", Journal of Zhejiang
/// University-SCIENCE, 2(3):247-252.
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

    /// Build the Zhao-Huang closed periodic sequence.
    ///
    /// Concatenates: `signal | signal_reversed` (total length 2N).
    ///
    /// Both seams are smooth:
    ///   - Middle seam: `signal[N-1]` meets `signal[N-1]` (identical value)
    ///   - Wrap-around: `signal_reversed[N-1] = signal[0]` meets `signal[0]`
    ///     (identical value, satisfying the periodic spline's y[0]==y[-1] requirement)
    ///
    /// The `periods` parameter is unused for the 2N construction but kept for
    /// API compatibility.
    fn build_closed_sequence(signal: &[f64]) -> Vec<f64> {
        let reversed: Vec<f64> = signal.iter().rev().copied().collect();
        let mut seq = Vec::with_capacity(signal.len() * 2);
        seq.extend_from_slice(signal);
        seq.extend_from_slice(&reversed);
        seq
    }
}

impl BoundaryCondition for Periodic {
    fn extend(&self, signal: &[f64], _extrema: &Extrema) -> ExtendedSignal {
        if signal.is_empty() {
            return ExtendedSignal { values: vec![], original_start: 0, original_end: 0 };
        }

        // Build [ signal | signal_reversed ] — a 2N closed periodic sequence.
        // The original signal occupies [0, N): original_start=0, original_end=N.
        // The mirrored copy in [N, 2N) provides boundary knots near both edges:
        //   - The left edge (position 0) sees mirrored knots wrapping from the
        //     end of the 2N period.
        //   - The right edge (position N-1) sees mirrored knots just past it in [N, 2N).
        let values = Self::build_closed_sequence(signal);
        let original_end = signal.len();

        ExtendedSignal { values, original_start: 0, original_end }
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
