//! Boundary prediction models.

use crate::adapters::streaming::state::PredictorState;
use crate::error::EmdError;
use serde::{Deserialize, Serialize};

/// Boundary prediction trait.
pub trait BoundaryPrediction: Send + Sync {
    fn predict(&self, signal: &[f64], n_ahead: usize) -> Vec<f64>;
    fn fit(&mut self, signal: &[f64]) -> Result<(), EmdError>;
    fn clone_box(&self) -> Box<dyn BoundaryPrediction>;
}

/// AR model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArModel {
    coefficients: Vec<f64>,
    order: usize,
}

impl ArModel {
    pub fn new(order: usize) -> Result<Self, EmdError> {
        if order == 0 {
            return Err(EmdError::InvalidConfig("AR order must be > 0".to_string()));
        }
        Ok(Self { coefficients: vec![0.0; order], order })
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    pub fn order(&self) -> usize {
        self.order
    }
}

impl BoundaryPrediction for ArModel {
    fn predict(&self, signal: &[f64], n_ahead: usize) -> Vec<f64> {
        if signal.is_empty() {
            return vec![0.0; n_ahead];
        }
        let mut predictions = Vec::with_capacity(n_ahead);
        let mut history = signal[signal.len().saturating_sub(self.order)..].to_vec();
        while history.len() < self.order {
            history.insert(0, signal[0]);
        }
        for _ in 0..n_ahead {
            let mut pred = 0.0;
            for (i, &coef) in self.coefficients.iter().enumerate() {
                if i < history.len() {
                    pred += coef * history[history.len() - 1 - i];
                }
            }
            predictions.push(pred);
            history.remove(0);
            history.push(pred);
        }
        predictions
    }

    fn fit(&mut self, _signal: &[f64]) -> Result<(), EmdError> {
        Ok(())
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

/// LSTM model stub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LstmModel {
    _weights: Vec<f32>,
}

impl LstmModel {
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
        if signal.is_empty() {
            vec![0.0; n_ahead]
        } else {
            vec![signal[signal.len() - 1]; n_ahead]
        }
    }

    fn fit(&mut self, _signal: &[f64]) -> Result<(), EmdError> {
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
    }

    #[test]
    fn test_ar_model_zero_order() {
        assert!(ArModel::new(0).is_err());
    }

    #[test]
    fn test_lstm_new() {
        let lstm = LstmModel::new();
        assert!(lstm._weights.is_empty());
    }
}
