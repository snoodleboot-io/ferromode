//! Configuration for boundary prediction models.

use serde::{Deserialize, Serialize};

/// Configuration for boundary prediction model selection and parameters.
///
/// This struct controls which boundary prediction strategy is used (AR vs LSTM)
/// and configures their hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryPredictionConfig {
    /// Enable LSTM-based prediction (default: true if feature enabled)
    pub lstm_enabled: bool,

    /// Stationarity score threshold for model selection (default: 0.7)
    ///
    /// - Score > threshold: Use AR model (fast, for stationary signals)
    /// - Score ≤ threshold: Use LSTM model (adaptive, for non-stationary signals)
    pub stationarity_threshold: f64,

    /// AR model order (default: 5)
    ///
    /// Number of previous samples used for AR prediction.
    /// Valid range: 1-10. Higher values capture longer-term dependencies but increase computation.
    pub ar_order: usize,

    /// LSTM input window size in samples (default: 20)
    ///
    /// Number of past samples fed to the neural network.
    /// Valid range: 10-50. Larger windows capture more context but increase memory.
    pub lstm_window: usize,

    /// LSTM output/prediction horizon in samples (default: 10)
    ///
    /// Number of future samples the LSTM predicts.
    /// Valid range: 5-20. Must be smaller than window size.
    pub lstm_horizon: usize,

    /// Cache predictions for repeated boundary windows (default: true)
    pub cache_enabled: bool,

    /// Maximum cache size in number of entries (default: 100)
    pub cache_size: usize,
}

impl Default for BoundaryPredictionConfig {
    fn default() -> Self {
        Self {
            lstm_enabled: cfg!(feature = "boundary-prediction"),
            stationarity_threshold: 0.7,
            ar_order: 5,
            lstm_window: 20,
            lstm_horizon: 10,
            cache_enabled: true,
            cache_size: 100,
        }
    }
}

impl BoundaryPredictionConfig {
    /// Create a new configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the stationarity threshold for model selection.
    ///
    /// # Arguments
    /// * `threshold` — Threshold value in range [0, 1]
    ///
    /// # Returns
    /// Self for method chaining
    pub fn with_stationarity_threshold(mut self, threshold: f64) -> Self {
        self.stationarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the AR model order.
    ///
    /// # Arguments
    /// * `order` — AR order in range [1, 10]
    ///
    /// # Returns
    /// Self for method chaining
    pub fn with_ar_order(mut self, order: usize) -> Self {
        self.ar_order = order.clamp(1, 10);
        self
    }

    /// Set the LSTM window and horizon sizes.
    ///
    /// # Arguments
    /// * `window` — Input window in range [10, 50]
    /// * `horizon` — Prediction horizon in range [5, 20], must be < window
    ///
    /// # Returns
    /// Self for method chaining
    pub fn with_lstm_sizes(mut self, window: usize, horizon: usize) -> Self {
        self.lstm_window = window.clamp(10, 50);
        self.lstm_horizon = horizon.clamp(5, 20).min(window - 1);
        self
    }

    /// Enable or disable LSTM prediction.
    pub fn with_lstm_enabled(mut self, enabled: bool) -> Self {
        self.lstm_enabled = enabled;
        self
    }

    /// Enable or disable result caching.
    pub fn with_cache(mut self, enabled: bool, size: usize) -> Self {
        self.cache_enabled = enabled;
        self.cache_size = size;
        self
    }

    /// Validate configuration parameters.
    ///
    /// # Returns
    /// Error message if any parameter is invalid, None otherwise
    pub fn validate(&self) -> Option<String> {
        if self.stationarity_threshold < 0.0 || self.stationarity_threshold > 1.0 {
            return Some("stationarity_threshold must be in [0, 1]".to_string());
        }

        if self.ar_order == 0 {
            return Some("ar_order must be > 0".to_string());
        }

        if self.lstm_window < self.lstm_horizon {
            return Some("lstm_window must be >= lstm_horizon".to_string());
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BoundaryPredictionConfig::default();
        assert_eq!(config.stationarity_threshold, 0.7);
        assert_eq!(config.ar_order, 5);
        assert_eq!(config.lstm_window, 20);
        assert_eq!(config.lstm_horizon, 10);
    }

    #[test]
    fn test_config_builder() {
        let config = BoundaryPredictionConfig::new()
            .with_ar_order(3)
            .with_lstm_sizes(30, 15)
            .with_stationarity_threshold(0.8);

        assert_eq!(config.ar_order, 3);
        assert_eq!(config.lstm_window, 30);
        assert_eq!(config.lstm_horizon, 15);
        assert_eq!(config.stationarity_threshold, 0.8);
    }

    #[test]
    fn test_config_validation() {
        let config = BoundaryPredictionConfig::default();
        assert!(config.validate().is_none());

        let config = BoundaryPredictionConfig { stationarity_threshold: 1.5, ..Default::default() };
        assert!(config.validate().is_some());
    }

    #[test]
    fn test_config_clamping() {
        let config = BoundaryPredictionConfig::new()
            .with_stationarity_threshold(2.0) // Should clamp to 1.0
            .with_ar_order(50); // Should clamp to 10

        assert_eq!(config.stationarity_threshold, 1.0);
        assert_eq!(config.ar_order, 10);
    }
}
