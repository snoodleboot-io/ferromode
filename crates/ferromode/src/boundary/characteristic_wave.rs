/// Characteristic Wave Extension (Huang et al. 1998).
///
/// Identifies the two nearest extrema at each boundary, constructs an implicit
/// wave from their spacing, and appends copies per end.
///
/// Reference: Huang et al. (1998), "A new view of nonlinear water waves: the Hilbert spectrum"
use crate::boundary::{BoundaryCondition, ExtendedSignal, Extrema};

/// Configuration for characteristic wave extension.
#[derive(Debug, Clone)]
pub struct CharacteristicWaveConfig {
    /// Number of copies to append per end (default: 4)
    pub copies: usize,
}

impl Default for CharacteristicWaveConfig {
    fn default() -> Self {
        Self { copies: 4 }
    }
}

/// Characteristic wave boundary extension.
pub struct CharacteristicWave {
    config: CharacteristicWaveConfig,
}

impl Default for CharacteristicWave {
    fn default() -> Self {
        Self { config: CharacteristicWaveConfig::default() }
    }
}

impl CharacteristicWave {
    /// Construct a `CharacteristicWave` boundary strategy from the given configuration.
    pub fn new(config: CharacteristicWaveConfig) -> Self {
        Self { config }
    }

    /// Construct a `CharacteristicWave` boundary strategy with the specified number of wave copies.
    pub fn with_copies(copies: usize) -> Self {
        Self { config: CharacteristicWaveConfig { copies } }
    }

    fn build_extension(
        signal: &[f64],
        extrema: &Extrema,
        boundary: Boundary,
        copies: usize,
    ) -> Vec<f64> {
        let all_extrema = Self::sorted_extrema_indices(extrema);

        if all_extrema.len() < 2 {
            return Self::fallback_extend(signal, boundary, copies);
        }

        let (first_idx, second_idx) = match boundary {
            Boundary::Left => {
                let first = all_extrema[0];
                let second = all_extrema[1];
                (first, second)
            }
            Boundary::Right => {
                let last = *all_extrema.last().unwrap();
                let second_last = all_extrema[all_extrema.len() - 2];
                (second_last, last)
            }
        };

        let wave_length =
            if second_idx > first_idx { second_idx - first_idx } else { first_idx - second_idx };

        if wave_length == 0 || wave_length >= signal.len() {
            return Self::fallback_extend(signal, boundary, copies);
        }

        let mut extension = Vec::new();
        let start_idx = match boundary {
            Boundary::Left => 0,
            Boundary::Right => signal.len() - wave_length,
        };

        for _ in 0..copies {
            for i in 0..wave_length {
                let idx = start_idx + i;
                if idx < signal.len() {
                    extension.push(signal[idx]);
                }
            }
        }

        if extension.is_empty() {
            return Self::fallback_extend(signal, boundary, copies);
        }

        extension
    }

    fn fallback_extend(signal: &[f64], boundary: Boundary, copies: usize) -> Vec<f64> {
        let n = if signal.len() >= 4 { 4 } else { signal.len() };
        let mut ext = Vec::new();
        for _ in 0..copies {
            for i in 0..n {
                match boundary {
                    Boundary::Left => ext.push(signal[i]),
                    Boundary::Right => ext.push(signal[signal.len() - n + i]),
                }
            }
        }
        ext
    }

    fn sorted_extrema_indices(extrema: &Extrema) -> Vec<usize> {
        let mut all: Vec<usize> = extrema.maxima_indices.clone();
        all.extend(extrema.minima_indices.iter());
        all.sort();
        all.dedup();
        all
    }
}

enum Boundary {
    Left,
    Right,
}

impl BoundaryCondition for CharacteristicWave {
    fn extend(&self, signal: &[f64], extrema: &Extrema) -> ExtendedSignal {
        if signal.is_empty() {
            return ExtendedSignal { values: vec![], original_start: 0, original_end: 0 };
        }

        let left_ext = Self::build_extension(signal, extrema, Boundary::Left, self.config.copies);
        let right_ext = Self::build_extension(signal, extrema, Boundary::Right, self.config.copies);

        let mut values = Vec::with_capacity(left_ext.len() + signal.len() + right_ext.len());
        values.extend_from_slice(&left_ext);
        values.extend_from_slice(signal);
        values.extend_from_slice(&right_ext);

        let original_start = left_ext.len();
        let original_end = original_start + signal.len();

        ExtendedSignal { values, original_start, original_end }
    }

    fn name(&self) -> &str {
        "characteristic_wave"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_characteristic_wave_on_pure_sine() {
        let n = 100;
        let signal: Vec<f64> =
            (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();

        let maxima: Vec<usize> = (12..n).step_by(50).collect();
        let minima: Vec<usize> = (37..n).step_by(50).collect();
        let extrema = Extrema { maxima_indices: maxima, minima_indices: minima };

        let strategy = CharacteristicWave::with_copies(4);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_characteristic_wave_on_chirp() {
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * std::f64::consts::PI * (5.0 * t + 10.0 * t * t)).sin()
            })
            .collect();

        let extrema = Extrema {
            maxima_indices: vec![10, 30, 50, 70, 90, 110, 130, 150, 170, 190],
            minima_indices: vec![20, 40, 60, 80, 100, 120, 140, 160, 180],
        };

        let strategy = CharacteristicWave::with_copies(4);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_characteristic_wave_on_noisy_signal() {
        let n = 150;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * i as f64 / 30.0).sin() + 0.1 * (i as f64 * 0.7).sin()
            })
            .collect();

        let extrema = Extrema {
            maxima_indices: vec![5, 25, 55, 85, 115, 145],
            minima_indices: vec![15, 45, 75, 105, 135],
        };

        let strategy = CharacteristicWave::with_copies(4);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_characteristic_wave_few_extrema() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let extrema = Extrema { maxima_indices: vec![2], minima_indices: vec![0] };

        let strategy = CharacteristicWave::with_copies(2);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
    }

    #[test]
    fn test_characteristic_wave_empty_signal() {
        let signal: Vec<f64> = vec![];
        let extrema = Extrema { maxima_indices: vec![], minima_indices: vec![] };

        let strategy = CharacteristicWave::default();
        let result = strategy.extend(&signal, &extrema);

        assert!(result.values.is_empty());
        assert_eq!(result.original_start, 0);
        assert_eq!(result.original_end, 0);
    }

    #[test]
    fn test_characteristic_wave_no_extrema() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let extrema = Extrema { maxima_indices: vec![], minima_indices: vec![] };

        let strategy = CharacteristicWave::with_copies(2);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
    }

    #[test]
    fn test_name_returns_correct_value() {
        let strategy = CharacteristicWave::default();
        assert_eq!(strategy.name(), "characteristic_wave");
    }
}
