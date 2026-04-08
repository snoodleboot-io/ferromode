//! Streaming decomposition orchestrator.

use crate::adapters::streaming::state::{IntermittencyMetrics, PredictorState, StreamingState};
use crate::algorithms::emd::{self, EmdConfig};
use crate::error::EmdError;
use crate::types::Signal;

/// Result of decomposing a single chunk.
#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub imfs: Vec<Vec<f64>>,
    pub residue: Vec<f64>,
    pub metrics: IntermittencyMetrics,
}

/// Streaming EMD decomposer with state management.
pub struct StreamingDecomposer {
    config: EmdConfig,
    state: StreamingState,
}

impl StreamingDecomposer {
    pub fn new(
        config: EmdConfig,
        predictor: Box<dyn PredictorState>,
        buffer_size: usize,
    ) -> Result<Self, EmdError> {
        let state = StreamingState::new(predictor, buffer_size, 100);
        Ok(Self { config, state })
    }

    pub fn decompose_chunk(&mut self, chunk: &Signal) -> Result<ChunkResult, EmdError> {
        if chunk.len() < 10 {
            return Err(EmdError::InsufficientData);
        }

        let chunk_data = chunk.values();
        let metrics = self.compute_intermittency_metrics(chunk_data);
        self.state.update_metrics(metrics);

        let n_extend = (chunk_data.len() / 4).min(64).max(10);
        let pred_start = self.state.predictor_state.predict_next(chunk_data, n_extend);
        let pred_end = self.state.predictor_state.predict_next(chunk_data, n_extend);

        let mut extended = Vec::with_capacity(chunk_data.len() + 2 * n_extend);
        extended.extend(pred_start.iter().rev());
        extended.extend_from_slice(chunk_data);
        extended.extend(&pred_end);

        let extended_signal = Signal::from_slice(&extended)?;
        let result = emd::emd(extended_signal.values(), &self.config)?;

        let imfs = result
            .imfs
            .imfs
            .iter()
            .map(|imf| {
                let start = n_extend;
                let end = (start + chunk_data.len()).min(imf.len());
                imf[start..end].to_vec()
            })
            .collect();

        let residue = {
            let start = n_extend;
            let end = (start + chunk_data.len()).min(result.imfs.residue.len());
            result.imfs.residue[start..end].to_vec()
        };

        self.state.push_samples(chunk_data);
        self.state.next_chunk();

        Ok(ChunkResult { imfs, residue, metrics })
    }

    fn compute_intermittency_metrics(&self, _signal: &[f64]) -> IntermittencyMetrics {
        IntermittencyMetrics::new(0.3, 0.2, 0.8)
    }

    pub fn chunk_id(&self) -> u64 {
        self.state.chunk_id
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::streaming::state::PredictorState;

    struct MockPredictor;

    impl PredictorState for MockPredictor {
        fn predict_next(&self, signal: &[f64], n_ahead: usize) -> Vec<f64> {
            if signal.is_empty() {
                vec![0.0; n_ahead]
            } else {
                vec![signal[signal.len() - 1]; n_ahead]
            }
        }

        fn update(&mut self, _signal: &[f64]) -> Result<(), EmdError> {
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn PredictorState> {
            Box::new(MockPredictor)
        }
    }

    #[test]
    fn test_streaming_decomposer_new() {
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let decomposer = StreamingDecomposer::new(config, predictor, 1024).unwrap();
        assert_eq!(decomposer.chunk_id(), 0);
    }

    #[test]
    fn test_chunk_result_creation() {
        let imfs = vec![vec![1.0, 2.0]];
        let residue = vec![3.0, 4.0];
        let metrics = IntermittencyMetrics::new(0.3, 0.2, 0.8);

        let result = ChunkResult { imfs, residue, metrics };
        assert_eq!(result.imfs.len(), 1);
    }
}
