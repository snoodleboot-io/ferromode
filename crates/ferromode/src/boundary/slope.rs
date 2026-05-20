/// Slope-Based Boundary Extension.
///
/// Extends the signal using a detrend-mirror-retrend strategy:
///
/// 1. Estimate the local linear trend (slope) at each endpoint via linear
///    regression over a short window.
/// 2. Subtract the trend to isolate the oscillatory component.
/// 3. Mirror the oscillation about the boundary (like ExtremasMirror), then
///    re-add the extrapolated trend.
///
/// This ensures the extension has extrema at positions that correspond to the
/// mirrored interior extrema — giving the envelope spline meaningful boundary
/// knots — while still respecting any linear drift in the signal.
///
/// The extension length is chosen adaptively: long enough to cover at least
/// the first two interior extrema on each side, so the spline always has ≥ 2
/// knots in the extension region.
use crate::boundary::{BoundaryCondition, ExtendedSignal, Extrema};

/// Configuration for slope-based boundary extension.
#[derive(Debug, Clone)]
pub struct SlopeConfig {
    /// Number of samples to extrapolate on each side.
    /// `0` (default) = adaptive: auto-sized to the distance of the second
    /// interior extremum on each side.
    pub extrapolation_length: usize,
    /// Window length (samples) for the linear-regression slope estimate.
    /// `0` (default) = `max(2, extrapolation_length / 2)`.
    pub slope_window: usize,
}

impl Default for SlopeConfig {
    fn default() -> Self {
        Self { extrapolation_length: 0, slope_window: 0 }
    }
}

/// Slope-based boundary extension.
pub struct Slope {
    config: SlopeConfig,
}

impl Default for Slope {
    fn default() -> Self {
        Self { config: SlopeConfig::default() }
    }
}

impl Slope {
    pub fn new(config: SlopeConfig) -> Self {
        Self { config }
    }

    pub fn with_extrapolation_length(length: usize) -> Self {
        Self { config: SlopeConfig { extrapolation_length: length, slope_window: 0 } }
    }

    /// Linear-regression slope over `signal[0..window]`.
    fn slope_at_start(signal: &[f64], window: usize) -> f64 {
        let n = window.min(signal.len());
        if n < 2 {
            return 0.0;
        }
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean: f64 = signal[..n].iter().sum::<f64>() / n as f64;
        let num: f64 =
            (0..n).map(|i| (i as f64 - x_mean) * (signal[i] - y_mean)).sum();
        let den: f64 = (0..n).map(|i| (i as f64 - x_mean).powi(2)).sum();
        if den.abs() < 1e-12 { 0.0 } else { num / den }
    }

    /// Linear-regression slope over `signal[n-window..n]`.
    fn slope_at_end(signal: &[f64], window: usize) -> f64 {
        let n = window.min(signal.len());
        let start = signal.len() - n;
        Self::slope_at_start(&signal[start..], n)
    }

    /// Build the left extension of length `d` using detrend-mirror-retrend.
    ///
    /// Value at mirrored position `-i`:  `signal[i] - 2 · slope · i`
    ///
    /// This is equivalent to reflecting the signal about the tangent line at
    /// the left boundary, producing extrema at the mirrored interior positions.
    fn build_left(signal: &[f64], d: usize, slope: f64) -> Vec<f64> {
        let max_d = d.min(signal.len().saturating_sub(1)).max(1);
        // Innermost point last → collect in reverse so index 0 is furthest left
        (1..=max_d)
            .map(|i| signal[i] - 2.0 * slope * i as f64)
            .rev()
            .collect()
    }

    /// Build the right extension of length `d` using detrend-mirror-retrend.
    ///
    /// Value at position `N-1+i`:  `signal[N-1-i] + 2 · slope · i`
    fn build_right(signal: &[f64], d: usize, slope: f64) -> Vec<f64> {
        let n = signal.len();
        let max_d = d.min(n.saturating_sub(1)).max(1);
        (1..=max_d)
            .map(|i| signal[n - 1 - i] + 2.0 * slope * i as f64)
            .collect()
    }
}

impl BoundaryCondition for Slope {
    fn extend(&self, signal: &[f64], extrema: &Extrema) -> ExtendedSignal {
        if signal.len() < 3 {
            return ExtendedSignal {
                values: signal.to_vec(),
                original_start: 0,
                original_end: signal.len(),
            };
        }

        let n = signal.len();

        // ── Extension length ─────────────────────────────────────────────────
        let d = if self.config.extrapolation_length > 0 {
            self.config.extrapolation_length
        } else {
            // Adaptive: extend to just past the second nearest interior extremum
            // (so we always have ≥ 2 extrema in the extension for a stable spline).
            let mut all_sorted: Vec<usize> = extrema
                .maxima_indices
                .iter()
                .chain(extrema.minima_indices.iter())
                .copied()
                .collect();
            all_sorted.sort_unstable();

            let d_left = all_sorted
                .get(1)
                .or(all_sorted.first())
                .copied()
                .unwrap_or(n / 4)
                + 1;
            let d_right = all_sorted
                .iter()
                .rev()
                .nth(1)
                .or(all_sorted.last())
                .map(|&p| n - 1 - p)
                .unwrap_or(n / 4)
                + 1;

            d_left.max(d_right).min(n / 2).max(10)
        };

        // ── Slope window ─────────────────────────────────────────────────────
        let window = if self.config.slope_window > 0 {
            self.config.slope_window
        } else {
            (d / 2).max(2).min(n / 4).max(2)
        };

        let slope_l = Self::slope_at_start(signal, window);
        let slope_r = Self::slope_at_end(signal, window);

        let left_ext = Self::build_left(signal, d, slope_l);
        let right_ext = Self::build_right(signal, d, slope_r);

        let orig_start = left_ext.len();
        let mut values = Vec::with_capacity(left_ext.len() + n + right_ext.len());
        values.extend_from_slice(&left_ext);
        values.extend_from_slice(signal);
        values.extend_from_slice(&right_ext);

        ExtendedSignal { values, original_start: orig_start, original_end: orig_start + n }
    }

    fn name(&self) -> &str {
        "slope"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extrema::detect_extrema;

    fn make_extrema(signal: &[f64]) -> Extrema {
        let e = detect_extrema(signal);
        Extrema { maxima_indices: e.maxima, minima_indices: e.minima }
    }

    #[test]
    fn test_slope_preserves_original() {
        let signal = vec![1.0, 3.0, 2.0, 5.0, 4.0, 1.0, 3.0, 2.0, 5.0, 4.0];
        let extrema = make_extrema(&signal);
        let strategy = Slope::default();
        let result = strategy.extend(&signal, &extrema);
        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
    }

    #[test]
    fn test_slope_extension_has_extrema() {
        // Pure sine: extension should create extrema in the padding region
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 8.0).sin())
            .collect();
        let extrema = make_extrema(&signal);
        let strategy = Slope::default();
        let result = strategy.extend(&signal, &extrema);

        let ext_extrema = detect_extrema(&result.values);
        let n_ext_left = ext_extrema
            .maxima
            .iter()
            .chain(ext_extrema.minima.iter())
            .filter(|&&i| i < result.original_start)
            .count();
        let n_ext_right = ext_extrema
            .maxima
            .iter()
            .chain(ext_extrema.minima.iter())
            .filter(|&&i| i >= result.original_end)
            .count();

        assert!(
            n_ext_left >= 1,
            "Expected ≥1 extremum in left extension, got {n_ext_left}"
        );
        assert!(
            n_ext_right >= 1,
            "Expected ≥1 extremum in right extension, got {n_ext_right}"
        );
    }

    #[test]
    fn test_slope_all_finite() {
        let signal: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let extrema = make_extrema(&signal);
        let strategy = Slope::default();
        let result = strategy.extend(&signal, &extrema);
        assert!(
            result.values.iter().all(|v| v.is_finite()),
            "Extension contains non-finite values"
        );
    }

    #[test]
    fn test_slope_short_signal() {
        let signal = vec![1.0, 2.0];
        let extrema = Extrema { maxima_indices: vec![], minima_indices: vec![] };
        let strategy = Slope::default();
        let result = strategy.extend(&signal, &extrema);
        // Should not panic; original preserved
        let orig = &result.values[result.original_start..result.original_end];
        assert_eq!(orig, &signal[..]);
    }

    #[test]
    fn test_slope_name() {
        assert_eq!(Slope::default().name(), "slope");
    }
}
