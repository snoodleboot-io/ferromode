/// Waveform Matching Extension.
///
/// Uses cross-correlation to find interior segments most similar to each
/// endpoint region, then appends the matched segment beyond the boundary.
use crate::boundary::{BoundaryCondition, ExtendedSignal, Extrema};

/// Configuration for waveform matching extension.
#[derive(Debug, Clone)]
pub struct WaveformMatchingConfig {
    /// Length of the endpoint region to match (in samples, default: 20)
    pub match_length: usize,
}

impl Default for WaveformMatchingConfig {
    fn default() -> Self {
        Self { match_length: 20 }
    }
}

/// Waveform matching boundary extension.
pub struct WaveformMatching {
    config: WaveformMatchingConfig,
}

impl WaveformMatching {
    pub fn new(config: WaveformMatchingConfig) -> Self {
        Self { config }
    }

    pub fn with_match_length(length: usize) -> Self {
        Self { config: WaveformMatchingConfig { match_length: length } }
    }

    fn cross_correlate(signal: &[f64], template: &[f64]) -> Vec<f64> {
        if template.is_empty() || signal.len() < template.len() {
            return vec![];
        }

        let n = signal.len() - template.len() + 1;
        let mut correlations = Vec::with_capacity(n);

        let template_mean: f64 = template.iter().sum::<f64>() / template.len() as f64;
        let template_centered: Vec<f64> = template.iter().map(|v| v - template_mean).collect();
        let template_norm: f64 = template_centered.iter().map(|v| v * v).sum::<f64>().sqrt();

        if template_norm < 1e-15 {
            return vec![0.0; n];
        }

        for i in 0..n {
            let window = &signal[i..i + template.len()];
            let window_mean: f64 = window.iter().sum::<f64>() / window.len() as f64;

            let mut dot = 0.0;
            let mut window_norm = 0.0;
            for j in 0..template.len() {
                let wc = window[j] - window_mean;
                dot += wc * template_centered[j];
                window_norm += wc * wc;
            }
            window_norm = window_norm.sqrt();

            if window_norm < 1e-15 {
                correlations.push(0.0);
            } else {
                correlations.push(dot / (template_norm * window_norm));
            }
        }

        correlations
    }

    fn find_best_match(signal: &[f64], template: &[f64]) -> Option<(usize, f64)> {
        let correlations = Self::cross_correlate(signal, template);
        if correlations.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_corr = correlations[0];

        for (i, &corr) in correlations.iter().enumerate() {
            if corr > best_corr {
                best_idx = i;
                best_corr = corr;
            }
        }

        Some((best_idx, best_corr))
    }

    fn match_and_extend(signal: &[f64], match_len: usize, from_left: bool) -> Vec<f64> {
        if signal.len() < match_len * 2 {
            let n = signal.len().min(match_len);
            if from_left {
                return signal[..n].to_vec();
            } else {
                return signal[signal.len() - n..].to_vec();
            }
        }

        let template = if from_left {
            signal[..match_len].to_vec()
        } else {
            signal[signal.len() - match_len..].to_vec()
        };

        let search_start = if from_left { match_len } else { 0 };
        let search_end = if from_left { signal.len() - match_len } else { signal.len() };

        if search_end <= search_start {
            return if from_left {
                signal[..match_len.min(signal.len())].to_vec()
            } else {
                signal[signal.len().saturating_sub(match_len)..].to_vec()
            };
        }

        let search_region = &signal[search_start..search_end];
        if let Some((best_idx, _corr)) = Self::find_best_match(search_region, &template) {
            let match_start = search_start + best_idx;
            let match_end = (match_start + match_len).min(signal.len());
            return signal[match_start..match_end].to_vec();
        }

        if from_left {
            signal[..match_len.min(signal.len())].to_vec()
        } else {
            signal[signal.len().saturating_sub(match_len)..].to_vec()
        }
    }
}

impl BoundaryCondition for WaveformMatching {
    fn extend(&self, signal: &[f64], _extrema: &Extrema) -> ExtendedSignal {
        if signal.is_empty() {
            return ExtendedSignal { values: vec![], original_start: 0, original_end: 0 };
        }

        let left_ext = Self::match_and_extend(signal, self.config.match_length, true);
        let right_ext = Self::match_and_extend(signal, self.config.match_length, false);

        let mut values = Vec::with_capacity(left_ext.len() + signal.len() + right_ext.len());
        values.extend_from_slice(&left_ext);
        values.extend_from_slice(signal);
        values.extend_from_slice(&right_ext);

        let original_start = left_ext.len();
        let original_end = original_start + signal.len();

        ExtendedSignal { values, original_start, original_end }
    }

    fn name(&self) -> &str {
        "waveform_matching"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waveform_matching_on_quasi_periodic() {
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * std::f64::consts::PI * t * 5.0).sin()
                    + 0.2 * (2.0 * std::f64::consts::PI * t * 5.3).sin()
            })
            .collect();

        let extrema = Extrema {
            maxima_indices: vec![10, 30, 50, 70, 90, 110, 130, 150, 170, 190],
            minima_indices: vec![20, 40, 60, 80, 100, 120, 140, 160, 180],
        };

        let strategy = WaveformMatching::with_match_length(20);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_waveform_matching_preserves_original() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let extrema = Extrema { maxima_indices: vec![4], minima_indices: vec![0, 8] };

        let strategy = WaveformMatching::default();
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
    }

    #[test]
    fn test_waveform_matching_configurable_match_length() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
        let extrema = Extrema {
            maxima_indices: vec![3, 15, 28, 40, 53, 65, 78, 90],
            minima_indices: vec![9, 22, 34, 47, 59, 72, 84, 97],
        };

        let strategy = WaveformMatching::with_match_length(10);
        let result = strategy.extend(&signal, &extrema);

        assert!(result.values.len() > signal.len());
    }

    #[test]
    fn test_waveform_matching_empty_signal() {
        let signal: Vec<f64> = vec![];
        let extrema = Extrema { maxima_indices: vec![], minima_indices: vec![] };

        let strategy = WaveformMatching::default();
        let result = strategy.extend(&signal, &extrema);

        assert!(result.values.is_empty());
    }

    #[test]
    fn test_waveform_matching_short_signal() {
        let signal = vec![1.0, 2.0, 3.0];
        let extrema = Extrema { maxima_indices: vec![2], minima_indices: vec![0] };

        let strategy = WaveformMatching::with_match_length(20);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() >= signal.len());
    }

    #[test]
    fn test_waveform_matching_name() {
        let strategy = WaveformMatching::default();
        assert_eq!(strategy.name(), "waveform_matching");
    }

    #[test]
    fn test_cross_correlate_perfect_match() {
        let signal = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0];
        let template = vec![0.0, 1.0, 2.0, 3.0];

        let correlations = WaveformMatching::cross_correlate(&signal, &template);
        assert!(!correlations.is_empty());

        let max_corr = correlations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_corr > 0.9);
    }
}
