//! Boundary prediction models for streaming decomposition.
//!
//! This module implements predictors that extend signals at chunk boundaries
//! to reduce end effects and maintain envelope continuity across chunks.

use crate::adapters::streaming::state::PredictorState;
use crate::error::EmdError;
use serde::{Deserialize, Serialize};

/// Trait for boundary prediction strategies.
///
/// Implementations predict extension samples to reduce end effects when
/// computing envelopes at chunk boundaries.
pub trait BoundaryPrediction: Send + Sync {
    /// Predict next n samples based on signal.
    ///
    /// # Arguments
    /// * `signal` — The signal chunk
    /// * `n_ahead` — Number of samples to predict
    ///
    /// # Returns
    /// Vector of predicted values (length = n_ahead)
    fn predict(&self, signal: &[f64], n_ahead: usize) -> Vec<f64>;

    /// Fit/update the model with new signal data.
    ///
    /// # Arguments
    /// * `signal` — Recent signal chunk to incorporate
    fn fit(&mut self, signal: &[f64]) -> Result<(), EmdError>;

    /// Clone for use in Arc
    fn clone_box(&self) -> Box<dyn BoundaryPrediction>;
}

/// AR (Auto-Regressive) model for boundary prediction.
///
/// Uses Yule-Walker equations to estimate AR coefficients and predict
/// next samples as a linear combination of past values.
///
/// Model: x[n] = a[0]*x[n-1] + a[1]*x[n-2] + ... + a[p-1]*x[n-p]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArModel {
    /// AR coefficients (order p)
    coefficients: Vec<f64>,
    /// AR order
    order: usize,
    /// Last p samples for prediction
    state: Vec<f64>,
}

impl ArModel {
    /// Create a new AR model with specified order.
    ///
    /// # Arguments
    /// * `order` — Number of AR coefficients (p). Common values: 3-10
    ///
    /// # Returns
    /// A new AR model with zero coefficients (unfit)
    pub fn new(order: usize) -> Result<Self, EmdError> {
        if order == 0 {
            return Err(EmdError::InvalidConfig("AR order must be > 0".to_string()));
        }
        Ok(Self { coefficients: vec![0.0; order], order, state: Vec::with_capacity(order) })
    }

    /// Fit AR model using Yule-Walker equations.
    ///
    /// Computes autocorrelation and solves the normal equations
    /// to estimate AR coefficients.
    ///
    /// # Arguments
    /// * `signal` — Signal to fit (length > order + 1 recommended)
    pub fn fit_yule_walker(&mut self, signal: &[f64]) -> Result<(), EmdError> {
        if signal.len() < self.order + 2 {
            return Err(EmdError::InsufficientData);
        }

        // Compute autocorrelation
        let acf = self.compute_acf(signal);

        // Solve Yule-Walker normal equations using Levinson recursion
        self.solve_levinson(&acf)?;

        // Store recent samples as state for prediction
        self.state.clear();
        let start = signal.len().saturating_sub(self.order);
        self.state.extend_from_slice(&signal[start..]);

        Ok(())
    }

    /// Compute autocorrelation coefficients.
    fn compute_acf(&self, signal: &[f64]) -> Vec<f64> {
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        let centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

        let c0 = centered.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
        let mut acf = vec![c0];

        for lag in 1..=self.order {
            let c = centered[..signal.len() - lag]
                .iter()
                .zip(&centered[lag..])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / signal.len() as f64;
            acf.push(c / c0);
        }

        acf
    }

    /// Solve Yule-Walker equations using Levinson recursion.
    ///
    /// # Arguments
    /// * `acf` — Autocorrelation coefficients (length = order + 1)
    fn solve_levinson(&mut self, acf: &[f64]) -> Result<(), EmdError> {
        if acf.is_empty() {
            return Err(EmdError::InvalidValue);
        }

        let mut phi = vec![0.0; self.order * self.order];
        let mut a = vec![0.0; self.order];

        // Initialize
        a[0] = -acf[1] / acf[0];
        let mut e = acf[0] * (1.0 - a[0] * a[0]);

        if self.order == 1 {
            self.coefficients[0] = -a[0];
            return Ok(());
        }

        phi[1 * self.order + 0] = 1.0;
        phi[0 * self.order + 1] = a[0];

        // Levinson recursion
        for m in 1..self.order {
            let mut mu = acf[m + 1];
            for k in 0..m {
                mu += a[k] * acf[m - k];
            }

            let lambda = -mu / e;

            for k in 0..m {
                a[k] += lambda * phi[k * self.order + (m - k - 1)];
            }
            a[m] = lambda;

            e = e * (1.0 - lambda * lambda);

            if e <= 1e-10 {
                break;
            }

            for i in 0..=m {
                for j in 0..=m {
                    phi[i * self.order + j] += lambda * phi[(m - j) * self.order + (m - i)];
                }
            }
        }

        // Store coefficients
        for i in 0..self.order {
            self.coefficients[i] = a[i];
        }

        Ok(())
    }

    /// Get AR coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Get AR order.
    pub fn order(&self) -> usize {
        self.order
    }
}

impl BoundaryPrediction for ArModel {
    fn predict(&self, signal: &[f64], n_ahead: usize) -> Vec<f64> {
        if signal.is_empty() || self.coefficients.iter().all(|&x| x == 0.0) {
            return vec![0.0; n_ahead];
        }

        let mut predictions = Vec::with_capacity(n_ahead);
        let mut history = signal[signal.len().saturating_sub(self.order)..].to_vec();

        // Pad history if too short
        while history.len() < self.order {
            history.insert(0, signal[0]);
        }

        for _ in 0..n_ahead {
            let mut pred = 0.0;
            for (i, &coef) in self.coefficients.iter().enumerate() {
                if i < history.len() {
                    let idx = history.len() - 1 - i;
                    pred += coef * history[idx];
                }
            }
            predictions.push(pred);
            history.remove(0);
            history.push(pred);
        }

        predictions
    }

    fn fit(&mut self, signal: &[f64]) -> Result<(), EmdError> {
        self.fit_yule_walker(signal)
    }

    fn clone_box(&self) -> Box<dyn BoundaryPrediction> {
        Box::new(self.clone())
    }
}

impl PredictorState for ArModel {
    fn predict_next(&self, signal: &[f64], n_ahead: usize) -> Vec<f64> {
        <Self as BoundaryPrediction>::predict(self, signal, n_ahead)
    }

    fn update(&mut self, signal: &[f64]) -> Result<(), EmdError> {
        <Self as BoundaryPrediction>::fit(self, signal)
    }

    fn clone_box(&self) -> Box<dyn PredictorState> {
        Box::new(self.clone())
    }
}

/// LSTM-based boundary predictor (stub for v2.2+).
///
/// Currently a placeholder with zero weights. Will be populated in future
/// releases with trained weights for improved boundary prediction on
/// non-stationary signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmModel {
    /// Pre-trained weights (quantized, packed)
    _weights: Vec<f32>,
}

impl LstmModel {
    /// Create a new LSTM model (stub).
    pub fn new() -> Self {
        Self { _weights: Vec::new() }
    }
}

impl Default for LstmModel {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundaryPrediction for LstmModel {
    fn predict(&self, signal: &[f64], n_ahead: usize) -> Vec<f64> {
        // Stub: return last value repeated
        if signal.is_empty() {
            vec![0.0; n_ahead]
        } else {
            vec![signal[signal.len() - 1]; n_ahead]
        }
    }

    fn fit(&mut self, _signal: &[f64]) -> Result<(), EmdError> {
        // Stub: no-op
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn BoundaryPrediction> {
        Box::new(self.clone())
    }
}

impl PredictorState for LstmModel {
    fn predict_next(&self, signal: &[f64], n_ahead: usize) -> Vec<f64> {
        <Self as BoundaryPrediction>::predict(self, signal, n_ahead)
    }

    fn update(&mut self, signal: &[f64]) -> Result<(), EmdError> {
        <Self as BoundaryPrediction>::fit(self, signal)
    }

    fn clone_box(&self) -> Box<dyn PredictorState> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ar_model_new() {
        let ar = ArModel::new(3).unwrap();
        assert_eq!(ar.order(), 3);
        assert_eq!(ar.coefficients().len(), 3);
    }

    #[test]
    fn test_ar_model_zero_order() {
        let result = ArModel::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ar_model_fit_constant_signal() {
        let mut ar = ArModel::new(2).unwrap();
        let signal = vec![5.0; 100];
        ar.fit(&signal).unwrap();

        // Constant signal should predict itself
        let pred = ar.predict(&signal[50..], 5);
        for &p in &pred {
            assert!((p - 5.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_ar_model_insufficient_data() {
        let mut ar = ArModel::new(3).unwrap();
        let signal = vec![1.0, 2.0, 3.0];
        let result = ar.fit(&signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_ar_model_predict_empty_signal() {
        let ar = ArModel::new(2).unwrap();
        let pred = ar.predict(&[], 3);
        assert_eq!(pred.len(), 3);
        assert!(pred.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ar_model_boundary_prediction_trait() {
        let mut ar = ArModel::new(2).unwrap();
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        ar.fit(&signal).unwrap();

        let pred = <ArModel as BoundaryPrediction>::predict(&ar, &signal, 3);
        assert_eq!(pred.len(), 3);
    }

    #[test]
    fn test_lstm_model_new() {
        let lstm = LstmModel::new();
        assert!(lstm._weights.is_empty());
    }

    #[test]
    fn test_lstm_model_predict_stub() {
        let lstm = LstmModel::new();
        let signal = vec![1.0, 2.0, 3.0];
        let pred = lstm.predict(&signal, 2);

        assert_eq!(pred.len(), 2);
        assert_eq!(pred[0], 3.0);
        assert_eq!(pred[1], 3.0);
    }

    #[test]
    fn test_lstm_model_fit_stub() {
        let mut lstm = LstmModel::new();
        let result = lstm.fit(&[1.0, 2.0, 3.0]);
        assert!(result.is_ok());
    }
}
