/// AR Model Extension.
///
/// Implements Yule-Walker AR coefficient estimation for forecasting beyond
/// the right boundary and backcasting before the left boundary.
use crate::boundary::{BoundaryCondition, ExtendedSignal, Extrema};

/// Configuration for AR model extension.
#[derive(Debug, Clone)]
pub struct ARModelConfig {
    /// AR model order (default: 5)
    pub ar_order: usize,
    /// Number of samples to forecast/backcast (default: 10)
    pub prediction_length: usize,
}

impl Default for ARModelConfig {
    fn default() -> Self {
        Self { ar_order: 5, prediction_length: 10 }
    }
}

/// AR model boundary extension.
pub struct ARModel {
    config: ARModelConfig,
}

impl Default for ARModel {
    fn default() -> Self {
        Self { config: ARModelConfig::default() }
    }
}

impl ARModel {
    /// Construct an `ARModel` boundary strategy from the given configuration.
    pub fn new(config: ARModelConfig) -> Self {
        Self { config }
    }

    /// Construct an `ARModel` boundary strategy with a specific AR order.
    pub fn with_order(order: usize) -> Self {
        Self { config: ARModelConfig { ar_order: order, ..Default::default() } }
    }

    fn yule_walker(signal: &[f64], order: usize) -> Vec<f64> {
        let n = signal.len();
        if n <= order {
            return vec![0.0; order];
        }

        let mut r = vec![0.0; order + 1];
        for k in 0..=order {
            let mut sum = 0.0;
            for i in 0..(n - k) {
                sum += signal[i] * signal[i + k];
            }
            r[k] = sum / n as f64;
        }

        if r[0].abs() < 1e-15 {
            return vec![0.0; order];
        }

        let mut a = vec![0.0; order];
        let mut reflection = vec![0.0; order];

        reflection[0] = -r[1] / r[0];
        a[0] = reflection[0];

        let mut err = r[0] * (1.0 - reflection[0] * reflection[0]);

        for k in 1..order {
            let mut num = r[k + 1];
            for j in 0..k {
                num += a[j] * r[k - j];
            }

            if err.abs() < 1e-15 {
                break;
            }

            reflection[k] = -num / err;

            let mut a_new = a.clone();
            for j in 0..k {
                a_new[j] += reflection[k] * a[k - 1 - j];
            }
            a_new[k] = reflection[k];
            a = a_new;

            err *= 1.0 - reflection[k] * reflection[k];
        }

        a
    }

    fn forecast(signal: &[f64], coefficients: &[f64], n: usize) -> Vec<f64> {
        if signal.is_empty() || coefficients.is_empty() {
            return vec![0.0; n];
        }

        let order = coefficients.len();
        let mut history: Vec<f64> = signal.to_vec();
        let mut predictions = Vec::with_capacity(n);

        for _ in 0..n {
            let start = history.len().saturating_sub(order);
            let window = &history[start..];

            let mut pred = 0.0;
            for (i, &coef) in coefficients.iter().enumerate().rev() {
                let idx = window.len().saturating_sub(1 + i);
                if idx < window.len() {
                    pred += coef * window[idx];
                }
            }

            history.push(pred);
            predictions.push(pred);
        }

        predictions
    }

    fn backcast(signal: &[f64], coefficients: &[f64], n: usize) -> Vec<f64> {
        if signal.is_empty() || coefficients.is_empty() {
            return vec![0.0; n];
        }

        let mut reversed = signal.to_vec();
        reversed.reverse();

        let mut forward_pred = Self::forecast(&reversed, coefficients, n);
        forward_pred.reverse();
        forward_pred
    }
}

impl BoundaryCondition for ARModel {
    fn extend(&self, signal: &[f64], _extrema: &Extrema) -> ExtendedSignal {
        if signal.is_empty() {
            return ExtendedSignal { values: vec![], original_start: 0, original_end: 0 };
        }

        let effective_order = self.config.ar_order.min(signal.len().saturating_sub(1)).max(1);
        let coefficients = Self::yule_walker(signal, effective_order);

        let left_ext = Self::backcast(signal, &coefficients, self.config.prediction_length);
        let right_ext = Self::forecast(signal, &coefficients, self.config.prediction_length);

        let mut values = Vec::with_capacity(left_ext.len() + signal.len() + right_ext.len());
        values.extend_from_slice(&left_ext);
        values.extend_from_slice(signal);
        values.extend_from_slice(&right_ext);

        let original_start = left_ext.len();
        let original_end = original_start + signal.len();

        ExtendedSignal { values, original_start, original_end }
    }

    fn name(&self) -> &str {
        "ar_model"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ar_model_on_ar2_synthetic() {
        let n = 200;
        let mut signal = vec![0.0; n];
        signal[0] = 1.0;
        signal[1] = 0.5;

        let phi1 = 1.5;
        let phi2 = -0.7;

        for i in 2..n {
            signal[i] = phi1 * signal[i - 1] + phi2 * signal[i - 2];
        }

        let extrema =
            Extrema { maxima_indices: vec![2, 10, 20, 30], minima_indices: vec![5, 15, 25, 35] };

        let strategy = ARModel::with_order(5);
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ar_model_preserves_original() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0];
        let extrema = Extrema { maxima_indices: vec![2], minima_indices: vec![0, 6] };

        let strategy = ARModel::new(ARModelConfig { ar_order: 3, prediction_length: 5 });
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
    }

    #[test]
    fn test_ar_model_configurable_order() {
        let signal: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
        let extrema =
            Extrema { maxima_indices: vec![5, 15, 26, 36], minima_indices: vec![10, 21, 31, 42] };

        let strategy = ARModel::with_order(3);
        let result = strategy.extend(&signal, &extrema);

        assert_eq!(result.original_start, 10);
        assert!(result.values.len() > signal.len());
    }

    #[test]
    fn test_ar_model_empty_signal() {
        let signal: Vec<f64> = vec![];
        let extrema = Extrema { maxima_indices: vec![], minima_indices: vec![] };

        let strategy = ARModel::default();
        let result = strategy.extend(&signal, &extrema);

        assert!(result.values.is_empty());
    }

    #[test]
    fn test_ar_model_single_element() {
        let signal = vec![42.0];
        let extrema = Extrema { maxima_indices: vec![0], minima_indices: vec![] };

        let strategy = ARModel::default();
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
    }

    #[test]
    fn test_ar_model_name() {
        let strategy = ARModel::default();
        assert_eq!(strategy.name(), "ar_model");
    }

    #[test]
    fn test_yule_walker_constant_signal() {
        let signal = vec![5.0; 20];
        let coeffs = ARModel::yule_walker(&signal, 3);

        assert_eq!(coeffs.len(), 3);
        for &c in &coeffs {
            assert!(c.is_finite());
        }
    }
}
