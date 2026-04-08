//! Streaming state management.

use crate::error::EmdError;
use serde::{Deserialize, Serialize};

/// Sifting iteration tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiftingIteration {
    pub iteration: usize,
    pub standard_deviation: f64,
    pub extrema_count: usize,
    pub converged: bool,
}

impl SiftingIteration {
    pub fn new(iteration: usize, sd: f64, extrema_count: usize, converged: bool) -> Self {
        Self { iteration, standard_deviation: sd, extrema_count, converged }
    }
}

/// Fixed-size ring buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingBuffer<T: Clone> {
    data: Vec<T>,
    write_pos: usize,
    count: usize,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self { data: Vec::with_capacity(capacity), write_pos: 0, count: 0, capacity }
    }

    pub fn push(&mut self, item: T) {
        if self.data.len() < self.capacity {
            self.data.push(item);
            self.count = self.data.len();
        } else {
            self.data[self.write_pos] = item;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.count {
            return None;
        }
        let actual_index = if self.count < self.capacity {
            index
        } else {
            (self.write_pos + index) % self.capacity
        };
        self.data.get(actual_index)
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
        self.write_pos = 0;
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn to_vec(&self) -> Vec<T> {
        (0..self.count)
            .map(|i| {
                let actual_index = if self.count < self.capacity {
                    i
                } else {
                    (self.write_pos + i) % self.capacity
                };
                self.data[actual_index].clone()
            })
            .collect()
    }
}

/// Predictor state trait.
pub trait PredictorState: Send + Sync {
    fn predict_next(&self, signal: &[f64], n_ahead: usize) -> Vec<f64>;
    fn update(&mut self, signal: &[f64]) -> Result<(), EmdError>;
    fn clone_box(&self) -> Box<dyn PredictorState>;
}

/// Intermittency metrics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IntermittencyMetrics {
    pub spectral_entropy: f64,
    pub extrema_spacing_cv: f64,
    pub stationarity_score: f64,
}

impl IntermittencyMetrics {
    pub fn new(spectral_entropy: f64, extrema_spacing_cv: f64, stationarity_score: f64) -> Self {
        Self { spectral_entropy, extrema_spacing_cv, stationarity_score }
    }

    pub fn recommended_algorithm(&self) -> AdaptiveAlgorithm {
        match self.stationarity_score {
            s if s > 0.8 => AdaptiveAlgorithm::EMD,
            s if s >= 0.5 => AdaptiveAlgorithm::EEMD,
            _ => AdaptiveAlgorithm::CEEMDAN,
        }
    }
}

/// Adaptive algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveAlgorithm {
    EMD,
    EEMD,
    CEEMDAN,
}

/// Streaming state.
// Debug impl below
pub struct StreamingState {
    pub chunk_id: u64,
    pub sifting_history: RingBuffer<SiftingIteration>,
    pub last_envelope: (Vec<f64>, Vec<f64>),
    pub predictor_state: Box<dyn PredictorState>,
    pub buffer: RingBuffer<f64>,
    pub last_metrics: Option<IntermittencyMetrics>,
    pub current_algorithm: AdaptiveAlgorithm,
}

impl StreamingState {
    pub fn new(
        predictor: Box<dyn PredictorState>,
        buffer_size: usize,
        sifting_size: usize,
    ) -> Self {
        Self {
            chunk_id: 0,
            sifting_history: RingBuffer::new(sifting_size),
            last_envelope: (Vec::new(), Vec::new()),
            predictor_state: predictor,
            buffer: RingBuffer::new(buffer_size),
            last_metrics: None,
            current_algorithm: AdaptiveAlgorithm::EMD,
        }
    }

    pub fn next_chunk(&mut self) {
        self.chunk_id += 1;
    }

    pub fn record_sifting_iteration(&mut self, iter: SiftingIteration) {
        self.sifting_history.push(iter);
    }

    pub fn update_envelope(&mut self, upper: Vec<f64>, lower: Vec<f64>) {
        self.last_envelope = (upper, lower);
    }

    pub fn push_sample(&mut self, v: f64) {
        self.buffer.push(v);
    }

    pub fn push_samples(&mut self, samples: &[f64]) {
        for &s in samples {
            self.buffer.push(s);
        }
    }

    pub fn update_metrics(&mut self, metrics: IntermittencyMetrics) {
        self.current_algorithm = metrics.recommended_algorithm();
        self.last_metrics = Some(metrics);
    }

    pub fn algorithm(&self) -> AdaptiveAlgorithm {
        self.current_algorithm
    }

    pub fn reset(&mut self) {
        self.chunk_id = 0;
        self.sifting_history.clear();
        self.last_envelope = (Vec::new(), Vec::new());
        self.buffer.clear();
        self.last_metrics = None;
        self.current_algorithm = AdaptiveAlgorithm::EMD;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_ring_buffer_new() {
        let rb: RingBuffer<i32> = RingBuffer::new(5);
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 5);
    }

    #[test]
    fn test_ring_buffer_push() {
        let mut rb = RingBuffer::new(3);
        rb.push(10);
        rb.push(20);
        assert_eq!(rb.len(), 2);
    }

    #[test]
    fn test_streaming_state_new() {
        let pred = Box::new(MockPredictor);
        let state = StreamingState::new(pred, 100, 50);
        assert_eq!(state.chunk_id, 0);
    }

    #[test]
    fn test_intermittency_metrics() {
        let m = IntermittencyMetrics::new(0.1, 0.05, 0.9);
        assert_eq!(m.recommended_algorithm(), AdaptiveAlgorithm::EMD);
    }
}
