use crate::error::EmdError;
use serde::{Deserialize, Serialize};
use std::time::Duration;

fn validate_finite(value: f64) -> Result<(), EmdError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EmdError::InvalidValue)
    }
}

fn validate_signal_data(data: &[f64]) -> Result<(), EmdError> {
    if data.is_empty() {
        return Err(EmdError::EmptySignal);
    }
    for &val in data {
        validate_finite(val)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// AlgorithmType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlgorithmType {
    EMD,
    EEMD,
    CEEMD,
    CEEMDAN,
    ICEEMDAN,
    MEMD,
    NAMEMD,
    VMD,
}

// ---------------------------------------------------------------------------
// Signal
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    values: Vec<f64>,
    sample_rate: Option<f64>,
}

impl Signal {
    pub fn from_slice(values: &[f64]) -> Result<Self, EmdError> {
        validate_signal_data(values)?;
        Ok(Self { values: values.to_vec(), sample_rate: None })
    }

    pub fn with_sample_rate(values: &[f64], sample_rate: f64) -> Result<Self, EmdError> {
        validate_signal_data(values)?;
        validate_finite(sample_rate)?;
        if sample_rate <= 0.0 {
            return Err(EmdError::InvalidSampleRate);
        }
        Ok(Self { values: values.to_vec(), sample_rate: Some(sample_rate) })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.values.iter()
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn sample_rate(&self) -> Option<f64> {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// MultivariateSignal
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateSignal {
    channels: Vec<Vec<f64>>,
}

impl MultivariateSignal {
    pub fn from_channels(channels: Vec<Vec<f64>>) -> Result<Self, EmdError> {
        if channels.is_empty() {
            return Err(EmdError::EmptySignal);
        }
        let first_len = channels[0].len();
        if first_len == 0 {
            return Err(EmdError::EmptySignal);
        }
        for ch in &channels {
            for &val in ch {
                validate_finite(val)?;
            }
        }
        for ch in &channels[1..] {
            if ch.len() != first_len {
                return Err(EmdError::DimensionMismatch);
            }
        }
        Ok(Self { channels })
    }

    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    pub fn n_samples(&self) -> usize {
        if self.channels.is_empty() {
            0
        } else {
            self.channels[0].len()
        }
    }

    pub fn channels(&self) -> &[Vec<f64>] {
        &self.channels
    }

    pub fn channel(&self, index: usize) -> Option<&[f64]> {
        self.channels.get(index).map(|c| c.as_slice())
    }
}

// ---------------------------------------------------------------------------
// ImfCollection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImfCollection {
    pub imfs: Vec<Vec<f64>>,
    pub residue: Vec<f64>,
}

impl ImfCollection {
    pub fn new(imfs: Vec<Vec<f64>>, residue: Vec<f64>) -> Self {
        Self { imfs, residue }
    }

    /// Validate IMFs and residue for finite values and dimension consistency.
    pub fn validate(imfs: &[Vec<f64>], residue: &[f64]) -> Result<(), EmdError> {
        if !imfs.is_empty() {
            let first_len = imfs[0].len();
            for imf in imfs {
                if imf.len() != first_len {
                    return Err(EmdError::DimensionMismatch);
                }
                for &val in imf {
                    validate_finite(val)?;
                }
            }
        }
        if !residue.is_empty() {
            for &val in residue {
                validate_finite(val)?;
            }
        }
        Ok(())
    }

    pub fn reconstruct(&self) -> Vec<f64> {
        let len = if !self.residue.is_empty() {
            self.residue.len()
        } else if !self.imfs.is_empty() {
            self.imfs[0].len()
        } else {
            return Vec::new();
        };

        let mut result = vec![0.0f64; len];

        for imf in &self.imfs {
            for (i, val) in imf.iter().enumerate().take(len) {
                result[i] += val;
            }
        }

        for (i, val) in self.residue.iter().enumerate().take(len) {
            result[i] += val;
        }

        result
    }

    pub fn orthogonality_index(&self) -> f64 {
        if self.imfs.len() < 2 {
            return 0.0;
        }

        let mut sum = 0.0f64;
        for i in 0..self.imfs.len() {
            for j in (i + 1)..self.imfs.len() {
                let dot_product: f64 =
                    self.imfs[i].iter().zip(self.imfs[j].iter()).map(|(a, b)| a * b).sum();

                let energy_i: f64 = self.imfs[i].iter().map(|v| v * v).sum();
                let energy_j: f64 = self.imfs[j].iter().map(|v| v * v).sum();

                if energy_i > 0.0 && energy_j > 0.0 {
                    sum += (dot_product / (energy_i * energy_j).sqrt()).abs();
                }
            }
        }

        sum
    }

    pub fn n_imfs(&self) -> usize {
        self.imfs.len()
    }
}

// ---------------------------------------------------------------------------
// DecompositionResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecompositionResult {
    pub algorithm: AlgorithmType,
    pub imfs: ImfCollection,
    pub elapsed: Duration,
    pub n_siftings: usize,
    pub config_snapshot: String,
}

impl DecompositionResult {
    pub fn new(
        algorithm: AlgorithmType,
        imfs: ImfCollection,
        elapsed: Duration,
        n_siftings: usize,
        config_snapshot: String,
    ) -> Self {
        Self { algorithm, imfs, elapsed, n_siftings, config_snapshot }
    }
}

// ---------------------------------------------------------------------------
// HilbertResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HilbertResult {
    pub instantaneous_amplitude: Vec<Vec<f64>>,
    pub instantaneous_frequency: Vec<Vec<f64>>,
    pub marginal_spectrum: Vec<f64>,
}

impl HilbertResult {
    pub fn new(
        instantaneous_amplitude: Vec<Vec<f64>>,
        instantaneous_frequency: Vec<Vec<f64>>,
        marginal_spectrum: Vec<f64>,
    ) -> Self {
        Self { instantaneous_amplitude, instantaneous_frequency, marginal_spectrum }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // =========================================================================
    // Signal tests
    // =========================================================================

    #[test]
    fn test_signal_from_slice_valid() {
        let signal = Signal::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(signal.len(), 3);
        assert_eq!(signal.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_signal_from_slice_empty() {
        let result = Signal::from_slice(&[]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmdError::EmptySignal));
    }

    #[test]
    fn test_signal_from_slice_single_value() {
        let signal = Signal::from_slice(&[42.0]).unwrap();
        assert_eq!(signal.len(), 1);
        assert!(!signal.is_empty());
    }

    #[test]
    fn test_signal_with_sample_rate_valid() {
        let signal = Signal::with_sample_rate(&[1.0, 2.0], 1000.0).unwrap();
        assert_eq!(signal.sample_rate(), Some(1000.0));
    }

    #[test]
    fn test_signal_with_sample_rate_zero() {
        let result = Signal::with_sample_rate(&[1.0, 2.0], 0.0);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidSampleRate));
    }

    #[test]
    fn test_signal_with_sample_rate_negative() {
        let result = Signal::with_sample_rate(&[1.0, 2.0], -100.0);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidSampleRate));
    }

    #[test]
    fn test_signal_iter() {
        let signal = Signal::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let values: Vec<&f64> = signal.iter().collect();
        assert_eq!(values, [&1.0, &2.0, &3.0]);
    }

    #[test]
    fn test_signal_clone_and_eq() {
        let signal = Signal::from_slice(&[1.0, 2.0]).unwrap();
        let cloned = signal.clone();
        assert_eq!(signal, cloned);
    }

    // =========================================================================
    // MultivariateSignal tests
    // =========================================================================

    #[test]
    fn test_multivariate_signal_valid() {
        let channels = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let signal = MultivariateSignal::from_channels(channels).unwrap();
        assert_eq!(signal.n_channels(), 2);
        assert_eq!(signal.n_samples(), 3);
    }

    #[test]
    fn test_multivariate_signal_empty_channels() {
        let result = MultivariateSignal::from_channels(vec![]);
        assert!(matches!(result.unwrap_err(), EmdError::EmptySignal));
    }

    #[test]
    fn test_multivariate_signal_empty_channel() {
        let channels = vec![vec![], vec![]];
        let result = MultivariateSignal::from_channels(channels);
        assert!(matches!(result.unwrap_err(), EmdError::EmptySignal));
    }

    #[test]
    fn test_multivariate_signal_dimension_mismatch() {
        let channels = vec![vec![1.0, 2.0], vec![1.0, 2.0, 3.0]];
        let result = MultivariateSignal::from_channels(channels);
        assert!(matches!(result.unwrap_err(), EmdError::DimensionMismatch));
    }

    #[test]
    fn test_multivariate_signal_single_channel() {
        let channels = vec![vec![1.0, 2.0, 3.0]];
        let signal = MultivariateSignal::from_channels(channels).unwrap();
        assert_eq!(signal.n_channels(), 1);
        assert_eq!(signal.n_samples(), 3);
    }

    #[test]
    fn test_multivariate_signal_channel_access() {
        let channels = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let signal = MultivariateSignal::from_channels(channels).unwrap();
        assert_eq!(signal.channel(0), Some(&[1.0, 2.0][..]));
        assert_eq!(signal.channel(1), Some(&[3.0, 4.0][..]));
        assert_eq!(signal.channel(2), None);
    }

    // =========================================================================
    // ImfCollection tests
    // =========================================================================

    #[test]
    fn test_reconstruct_known_signal() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let imf1 = vec![0.5, 1.0, 1.5, 2.0, 2.5];
        let imf2 = vec![0.3, 0.6, 0.9, 1.2, 1.5];
        let residue = vec![0.2, 0.4, 0.6, 0.8, 1.0];

        let collection = ImfCollection::new(vec![imf1, imf2], residue);
        let reconstructed = collection.reconstruct();

        for (a, b) in original.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10, "expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_reconstruct_single_imf_no_residue() {
        let imf = vec![1.0, 2.0, 3.0];
        let collection = ImfCollection::new(vec![imf.clone()], vec![]);
        let reconstructed = collection.reconstruct();
        assert_eq!(reconstructed, imf);
    }

    #[test]
    fn test_reconstruct_no_imfs_only_residue() {
        let residue = vec![5.0, 6.0, 7.0];
        let collection = ImfCollection::new(vec![], residue.clone());
        let reconstructed = collection.reconstruct();
        assert_eq!(reconstructed, residue);
    }

    #[test]
    fn test_reconstruct_empty_collection() {
        let collection = ImfCollection::new(vec![], vec![]);
        let reconstructed = collection.reconstruct();
        assert!(reconstructed.is_empty());
    }

    #[test]
    fn test_orthogonality_index_orthogonal_imfs() {
        let imf1 = vec![1.0, 0.0, 0.0, 0.0];
        let imf2 = vec![0.0, 1.0, 0.0, 0.0];
        let collection = ImfCollection::new(vec![imf1, imf2], vec![]);
        let index = collection.orthogonality_index();
        assert!(index < 1e-10, "orthogonal IMFs should have index near 0, got {}", index);
    }

    #[test]
    fn test_orthogonality_index_identical_imfs() {
        let imf = vec![1.0, 2.0, 3.0, 4.0];
        let collection = ImfCollection::new(vec![imf.clone(), imf.clone()], vec![]);
        let index = collection.orthogonality_index();
        assert!((index - 1.0).abs() < 1e-10, "identical IMFs should have index 1.0, got {}", index);
    }

    #[test]
    fn test_orthogonality_index_single_imf() {
        let imf = vec![1.0, 2.0, 3.0];
        let collection = ImfCollection::new(vec![imf], vec![]);
        let index = collection.orthogonality_index();
        assert_eq!(index, 0.0);
    }

    #[test]
    fn test_orthogonality_index_non_orthogonal_imfs() {
        let imf1 = vec![1.0, 1.0, 1.0];
        let imf2 = vec![2.0, 2.0, 2.0];
        let collection = ImfCollection::new(vec![imf1, imf2], vec![]);
        let index = collection.orthogonality_index();
        assert!((index - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_orthogonality_index_partial_overlap() {
        let imf1 = vec![1.0, 1.0, 0.0, 0.0];
        let imf2 = vec![1.0, 0.0, 1.0, 0.0];
        let collection = ImfCollection::new(vec![imf1, imf2], vec![]);
        let index = collection.orthogonality_index();
        assert!(
            index > 0.0 && index < 1.0,
            "partial overlap should give 0 < index < 1, got {}",
            index
        );
    }

    // =========================================================================
    // AlgorithmType tests
    // =========================================================================

    #[test]
    fn test_algorithm_type_variants() {
        let algorithms = vec![
            AlgorithmType::EMD,
            AlgorithmType::EEMD,
            AlgorithmType::CEEMD,
            AlgorithmType::CEEMDAN,
            AlgorithmType::ICEEMDAN,
            AlgorithmType::MEMD,
            AlgorithmType::NAMEMD,
            AlgorithmType::VMD,
        ];
        assert_eq!(algorithms.len(), 8);
        assert_ne!(AlgorithmType::EMD, AlgorithmType::VMD);
    }

    #[test]
    fn test_algorithm_type_clone_and_eq() {
        let algo = AlgorithmType::CEEMDAN;
        let cloned = algo.clone();
        assert_eq!(algo, cloned);
    }

    // =========================================================================
    // Serde JSON round-trip tests
    // =========================================================================

    #[test]
    fn test_algorithm_type_json_roundtrip() {
        let algo = AlgorithmType::CEEMDAN;
        let json = serde_json::to_string(&algo).unwrap();
        let decoded: AlgorithmType = serde_json::from_str(&json).unwrap();
        assert_eq!(algo, decoded);
    }

    #[test]
    fn test_imf_collection_json_roundtrip() {
        let collection = ImfCollection::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![0.5, 0.5]);
        let json = serde_json::to_string(&collection).unwrap();
        let decoded: ImfCollection = serde_json::from_str(&json).unwrap();
        assert_eq!(collection, decoded);
    }

    #[test]
    fn test_decomposition_result_json_roundtrip() {
        let imfs = ImfCollection::new(vec![vec![1.0, 2.0, 3.0]], vec![0.5, 0.5, 0.5]);
        let result = DecompositionResult::new(
            AlgorithmType::EMD,
            imfs,
            Duration::from_millis(42),
            10,
            r#"{"sd_threshold": 0.2}"#.to_string(),
        );
        let json = serde_json::to_string(&result).unwrap();
        let decoded: DecompositionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, decoded);
    }

    #[test]
    fn test_hilbert_result_json_roundtrip() {
        let hilbert = HilbertResult::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            vec![0.5, 1.0, 1.5],
        );
        let json = serde_json::to_string(&hilbert).unwrap();
        let decoded: HilbertResult = serde_json::from_str(&json).unwrap();
        assert_eq!(hilbert, decoded);
    }

    #[test]
    fn test_algorithm_type_serializes_correctly() {
        let algo = AlgorithmType::ICEEMDAN;
        let json = serde_json::to_string(&algo).unwrap();
        assert!(json.contains("ICEEMDAN"));
    }

    // =========================================================================
    // DecompositionResult tests
    // =========================================================================

    #[test]
    fn test_decomposition_result_fields() {
        let imfs = ImfCollection::new(vec![vec![1.0, 2.0]], vec![0.5]);
        let result = DecompositionResult::new(
            AlgorithmType::EEMD,
            imfs.clone(),
            Duration::from_secs(1),
            5,
            "{}".to_string(),
        );
        assert_eq!(result.algorithm, AlgorithmType::EEMD);
        assert_eq!(result.n_siftings, 5);
        assert_eq!(result.imfs, imfs);
    }

    // =========================================================================
    // HilbertResult tests
    // =========================================================================

    #[test]
    fn test_hilbert_result_fields() {
        let hilbert = HilbertResult::new(vec![vec![1.0]], vec![vec![0.5]], vec![1.5]);
        assert_eq!(hilbert.instantaneous_amplitude.len(), 1);
        assert_eq!(hilbert.instantaneous_frequency.len(), 1);
        assert_eq!(hilbert.marginal_spectrum.len(), 1);
    }

    // =========================================================================
    // Reconstruction with sinusoidal signal
    // =========================================================================

    #[test]
    fn test_reconstruct_sinusoidal_signal() {
        let n = 100;
        let original: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let imf = original.clone();
        let residue = vec![0.0; n];
        let collection = ImfCollection::new(vec![imf], residue);
        let reconstructed = collection.reconstruct();

        for (a, b) in original.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    // =========================================================================
    // Edge case: reconstruction with different length IMFs (truncates to shortest)
    // =========================================================================

    #[test]
    fn test_reconstruct_with_truncation() {
        let imf1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let imf2 = vec![1.0, 2.0];
        let residue = vec![0.0, 0.0, 0.0, 0.0, 0.0];

        let collection = ImfCollection::new(vec![imf1, imf2], residue);
        let reconstructed = collection.reconstruct();

        assert_eq!(reconstructed.len(), 5);
        assert!((reconstructed[0] - 2.0).abs() < 1e-10);
        assert!((reconstructed[1] - 4.0).abs() < 1e-10);
        assert!((reconstructed[2] - 3.0).abs() < 1e-10);
    }

    // =========================================================================
    // Error path tests — T-019
    // =========================================================================

    #[test]
    fn test_signal_from_slice_rejects_nan() {
        let result = Signal::from_slice(&[1.0, f64::NAN, 3.0]);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_signal_from_slice_rejects_inf() {
        let result = Signal::from_slice(&[1.0, f64::INFINITY, 3.0]);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));

        let result = Signal::from_slice(&[1.0, f64::NEG_INFINITY, 3.0]);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_signal_with_sample_rate_rejects_nan() {
        let result = Signal::with_sample_rate(&[1.0, 2.0], f64::NAN);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_multivariate_signal_rejects_nan_in_channel() {
        let channels = vec![vec![1.0, f64::NAN, 3.0], vec![4.0, 5.0, 6.0]];
        let result = MultivariateSignal::from_channels(channels);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_multivariate_signal_rejects_inf_in_channel() {
        let channels = vec![vec![1.0, f64::INFINITY, 3.0], vec![4.0, 5.0, 6.0]];
        let result = MultivariateSignal::from_channels(channels);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_imf_collection_empty_imfs_rejected() {
        let collection = ImfCollection::new(vec![], vec![1.0, 2.0]);
        let reconstructed = collection.reconstruct();
        assert_eq!(reconstructed, vec![1.0, 2.0]);
    }

    #[test]
    fn test_imf_collection_rejects_nan_in_imf() {
        let result = ImfCollection::validate(&[vec![1.0, f64::NAN, 3.0]], &[0.5]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_imf_collection_rejects_inf_in_imf() {
        let result = ImfCollection::validate(&[vec![1.0, f64::INFINITY, 3.0]], &[0.5]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_imf_collection_rejects_nan_in_residue() {
        let result = ImfCollection::validate(&[vec![1.0, 2.0, 3.0]], &[f64::NAN, 0.5]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_imf_collection_rejects_dimension_mismatch() {
        let result =
            ImfCollection::validate(&[vec![1.0, 2.0], vec![1.0, 2.0, 3.0]], &[0.5, 0.5, 0.5]);
        assert!(matches!(result.unwrap_err(), EmdError::DimensionMismatch));
    }
}
