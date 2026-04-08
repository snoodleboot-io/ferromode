//! Streaming state management for chunk-based EMD decomposition.
//!
//! This module provides the `StreamingState` struct and supporting types
//! that maintain continuity across chunks in streaming decomposition.

use crate::error::EmdError;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Represents a single sifting iteration with tracking information.
///
/// Used to track the sifting history across multiple chunks
/// to ensure envelope continuity and algorithm stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiftingIteration {
    /// Iteration index within the current sifting loop
    pub iteration: usize,
    /// Standard deviation computed in this iteration (sifting criterion)
    pub standard_deviation: f64,
    /// Number of extrema (maxima + minima) found in signal
    pub extrema_count: usize,
    /// Whether stopping criterion was met
    pub converged: bool,
}

impl SiftingIteration {
    /// Create a new sifting iteration record.
    pub fn new(
        iteration: usize,
        standard_deviation: f64,
        extrema_count: usize,
        converged: bool,
    ) -> Self {
        Self { iteration, standard_deviation, extrema_count, converged }
    }
}

/// Ring buffer implementation for fixed-size, memory-efficient chunk storage.
///
/// The ring buffer never grows beyond `capacity`. When full, new entries
/// overwrite the oldest. This is used to maintain a rolling window of
/// signal history for boundary prediction and continuity checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingBuffer<T: Clone> {
    /// Underlying storage
    data: Vec<T>,
    /// Current write position (wraps at capacity)
    write_pos: usize,
    /// Number of elements currently stored (0..=capacity)
    count: usize,
    /// Fixed capacity of the buffer (never grows)
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with fixed capacity.
    ///
    /// # Arguments
    /// * `capacity` — Maximum size (must be > 0)
    ///
    /// # Returns
    /// A new empty ring buffer
    ///
    /// # Panics
    /// Panics if capacity is 0
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        Self { data: Vec::with_capacity(capacity), write_pos: 0, count: 0, capacity }
    }

    /// Add an element to the ring buffer.
    ///
    /// If the buffer is full, overwrites the oldest element.
    /// Never allocates beyond initial capacity.
    pub fn push(&mut self, item: T) {
        if self.data.len() < self.capacity {
            self.data.push(item);
            self.count = self.data.len();
        } else {
            self.data[self.write_pos] = item;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }

    /// Get element at index in insertion order (0 = oldest).
    ///
    /// Returns None if index >= count
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

    /// Get mutable element at index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.count {
            return None;
        }
        let actual_index = if self.count < self.capacity {
            index
        } else {
            (self.write_pos + index) % self.capacity
        };
        self.data.get_mut(actual_index)
    }

    /// Return the number of elements stored (0..=capacity)
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if buffer is at capacity
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    /// Clear all elements (reset to empty state)
    pub fn clear(&mut self) {
        self.data.clear();
        self.write_pos = 0;
        self.count = 0;
    }

    /// Get capacity limit
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Iterator over elements in insertion order
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let capacity = self.capacity;
        let count = self.count;
        let write_pos = self.write_pos;
        let data = &self.data;

        (0..count).map(move |i| {
            let actual_index = if count < capacity { i } else { (write_pos + i) % capacity };
            &data[actual_index]
        })
    }

    /// Get all elements as a vector (in insertion order)
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }
}

/// Trait for managing predictor state across chunks.
///
/// The predictor state maintains model parameters (AR coefficients, LSTM weights, etc.)
/// and can generate predictions for boundary extension.
pub trait PredictorState: Send + Sync {
    /// Predict the next n samples based on current model state.
    ///
    /// # Arguments
    /// * `signal` — The current chunk signal
    /// * `n_ahead` — Number of samples to predict
    ///
    /// # Returns
    /// Vector of predicted values (length = n_ahead)
    fn predict_next(&self, signal: &[f64], n_ahead: usize) -> Vec<f64>;

    /// Update the predictor state based on a new signal chunk.
    ///
    /// This allows the model to adapt as new data arrives.
    ///
    /// # Arguments
    /// * `signal` — The new chunk data
    fn update(&mut self, signal: &[f64]) -> Result<(), EmdError>;

    /// Clone the state for use in Arc<dyn PredictorState>
    fn clone_box(&self) -> Box<dyn PredictorState>;
}

/// Metrics computed for intermittency and stationarity detection.
///
/// These metrics help determine which algorithm variant (EMD/EEMD/CEEMDAN)
/// is most suitable for the current signal characteristics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IntermittencyMetrics {
    /// Spectral entropy (0..1): lower = more stationary, higher = more complex
    pub spectral_entropy: f64,

    /// Coefficient of variation of extrema spacing: measures regularity of oscillations
    pub extrema_spacing_cv: f64,

    /// Stationarity score (0..1): composite measure of signal stationarity
    /// > 0.8 = use EMD, 0.5-0.8 = use EEMD, < 0.5 = use CEEMDAN
    pub stationarity_score: f64,
}

impl IntermittencyMetrics {
    /// Create metrics from components.
    pub fn new(spectral_entropy: f64, extrema_spacing_cv: f64, stationarity_score: f64) -> Self {
        Self { spectral_entropy, extrema_spacing_cv, stationarity_score }
    }

    /// Recommended algorithm based on stationarity score.
    pub fn recommended_algorithm(&self) -> AdaptiveAlgorithm {
        match self.stationarity_score {
            s if s > 0.8 => AdaptiveAlgorithm::EMD,
            s if s >= 0.5 => AdaptiveAlgorithm::EEMD,
            _ => AdaptiveAlgorithm::CEEMDAN,
        }
    }
}

/// Adaptive algorithm selection based on signal characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptiveAlgorithm {
    /// Classic EMD: good for highly stationary signals (stationarity > 0.8)
    EMD,
    /// EEMD with noise: moderate non-stationarity (0.5-0.8)
    EEMD,
    /// CEEMDAN: highly non-stationary signals (< 0.5)
    CEEMDAN,
}

/// Manages persistent state across streamed chunks.
///
/// The streaming adapter maintains this state to ensure:
/// - Envelope continuity at chunk boundaries
/// - Sifting history tracking
/// - Predictor parameter updates
/// - Memory-bounded operation via ring buffers
///
/// This is the core data structure for streaming decomposition.
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// Incremental chunk identifier (0, 1, 2, ...)
    pub chunk_id: u64,

    /// History of sifting iterations (limited to recent chunks via ring buffer)
    pub sifting_history: RingBuffer<SiftingIteration>,

    /// Last computed envelope (upper, lower) for continuity tracking
    pub last_envelope: (Vec<f64>, Vec<f64>),

    /// Predictor state for boundary prediction
    pub predictor_state: Box<dyn PredictorState>,

    /// Ring buffer of recent signal chunks (fixed memory)
    pub buffer: RingBuffer<f64>,

    /// Metrics from last chunk (for algorithm selection)
    pub last_metrics: Option<IntermittencyMetrics>,

    /// Currently selected algorithm (may adapt per chunk)
    pub current_algorithm: AdaptiveAlgorithm,
}

impl StreamingState {
    /// Create a new streaming state with given configuration.
    ///
    /// # Arguments
    /// * `predictor` — Predictor implementation for boundary extension
    /// * `buffer_size` — Ring buffer capacity for signal history (e.g., 1024)
    /// * `sifting_history_size` — Capacity for sifting iteration history (e.g., 100)
    ///
    /// # Returns
    /// A new streaming state, ready for chunk processing
    pub fn new(
        predictor: Box<dyn PredictorState>,
        buffer_size: usize,
        sifting_history_size: usize,
    ) -> Self {
        Self {
            chunk_id: 0,
            sifting_history: RingBuffer::new(sifting_history_size),
            last_envelope: (Vec::new(), Vec::new()),
            predictor_state: predictor,
            buffer: RingBuffer::new(buffer_size),
            last_metrics: None,
            current_algorithm: AdaptiveAlgorithm::EMD,
        }
    }

    /// Advance to next chunk and increment chunk_id.
    pub fn next_chunk(&mut self) {
        self.chunk_id += 1;
    }

    /// Record a sifting iteration in history.
    pub fn record_sifting_iteration(&mut self, iteration: SiftingIteration) {
        self.sifting_history.push(iteration);
    }

    /// Update the last envelope with new upper and lower curves.
    pub fn update_envelope(&mut self, upper: Vec<f64>, lower: Vec<f64>) {
        self.last_envelope = (upper, lower);
    }

    /// Add a sample to the ring buffer.
    pub fn push_sample(&mut self, value: f64) {
        self.buffer.push(value);
    }

    /// Add multiple samples to the ring buffer.
    pub fn push_samples(&mut self, samples: &[f64]) {
        for &sample in samples {
            self.buffer.push(sample);
        }
    }

    /// Update metrics and adapt algorithm if needed.
    pub fn update_metrics(&mut self, metrics: IntermittencyMetrics) {
        self.current_algorithm = metrics.recommended_algorithm();
        self.last_metrics = Some(metrics);
    }

    /// Get current algorithm.
    pub fn algorithm(&self) -> AdaptiveAlgorithm {
        self.current_algorithm
    }

    /// Get last metrics (if any).
    pub fn metrics(&self) -> Option<IntermittencyMetrics> {
        self.last_metrics
    }

    /// Get reference to the ring buffer.
    pub fn buffer_ref(&self) -> &RingBuffer<f64> {
        &self.buffer
    }

    /// Get mutable reference to the ring buffer.
    pub fn buffer_mut(&mut self) -> &mut RingBuffer<f64> {
        &mut self.buffer
    }

    /// Get reference to the predictor state.
    pub fn predictor_ref(&self) -> &dyn PredictorState {
        &*self.predictor_state
    }

    /// Get mutable reference to the predictor state.
    pub fn predictor_mut(&mut self) -> &mut Box<dyn PredictorState> {
        &mut self.predictor_state
    }

    /// Get sifting history as a vector.
    pub fn sifting_history_vec(&self) -> Vec<SiftingIteration> {
        self.sifting_history.to_vec()
    }

    /// Reset state to initial condition (for restart/cleanup).
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

    // Mock predictor for testing
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

    // =====================================================================
    // RingBuffer Tests
    // =====================================================================

    #[test]
    fn test_ring_buffer_new() {
        let rb: RingBuffer<i32> = RingBuffer::new(5);
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 5);
        assert!(rb.is_empty());
        assert!(!rb.is_full());
    }

    #[test]
    #[should_panic]
    fn test_ring_buffer_zero_capacity() {
        let _rb: RingBuffer<i32> = RingBuffer::new(0);
    }

    #[test]
    fn test_ring_buffer_push_and_get() {
        let mut rb = RingBuffer::new(3);
        rb.push(10);
        rb.push(20);
        rb.push(30);

        assert_eq!(rb.len(), 3);
        assert!(rb.is_full());
        assert_eq!(rb.get(0), Some(&10));
        assert_eq!(rb.get(1), Some(&20));
        assert_eq!(rb.get(2), Some(&30));
        assert_eq!(rb.get(3), None);
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let mut rb = RingBuffer::new(3);
        rb.push(10);
        rb.push(20);
        rb.push(30);
        // Now at capacity, next push should overwrite 10
        rb.push(40);

        assert_eq!(rb.len(), 3);
        assert_eq!(rb.get(0), Some(&20));
        assert_eq!(rb.get(1), Some(&30));
        assert_eq!(rb.get(2), Some(&40));
    }

    #[test]
    fn test_ring_buffer_iterator() {
        let mut rb = RingBuffer::new(4);
        rb.push(10);
        rb.push(20);
        rb.push(30);

        let collected: Vec<_> = rb.iter().copied().collect();
        assert_eq!(collected, vec![10, 20, 30]);
    }

    #[test]
    fn test_ring_buffer_to_vec() {
        let mut rb = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        rb.push(4); // overwrites 1

        let v = rb.to_vec();
        assert_eq!(v, vec![2, 3, 4]);
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut rb = RingBuffer::new(5);
        rb.push(10);
        rb.push(20);
        rb.clear();

        assert_eq!(rb.len(), 0);
        assert!(rb.is_empty());
        assert_eq!(rb.get(0), None);
    }

    #[test]
    fn test_ring_buffer_get_mut() {
        let mut rb = RingBuffer::new(3);
        rb.push(10);
        rb.push(20);
        rb.push(30);

        if let Some(val) = rb.get_mut(1) {
            *val = 99;
        }

        assert_eq!(rb.get(1), Some(&99));
    }

    // =====================================================================
    // SiftingIteration Tests
    // =====================================================================

    #[test]
    fn test_sifting_iteration_new() {
        let iter = SiftingIteration::new(5, 0.001, 42, true);
        assert_eq!(iter.iteration, 5);
        assert_eq!(iter.standard_deviation, 0.001);
        assert_eq!(iter.extrema_count, 42);
        assert!(iter.converged);
    }

    // =====================================================================
    // StreamingState Tests
    // =====================================================================

    #[test]
    fn test_streaming_state_new() {
        let predictor = Box::new(MockPredictor);
        let state = StreamingState::new(predictor, 100, 50);

        assert_eq!(state.chunk_id, 0);
        assert!(state.sifting_history.is_empty());
        assert!(state.last_envelope.0.is_empty());
        assert!(state.buffer.is_empty());
        assert_eq!(state.current_algorithm, AdaptiveAlgorithm::EMD);
    }

    #[test]
    fn test_streaming_state_next_chunk() {
        let predictor = Box::new(MockPredictor);
        let mut state = StreamingState::new(predictor, 100, 50);

        assert_eq!(state.chunk_id, 0);
        state.next_chunk();
        assert_eq!(state.chunk_id, 1);
        state.next_chunk();
        assert_eq!(state.chunk_id, 2);
    }

    #[test]
    fn test_streaming_state_push_samples() {
        let predictor = Box::new(MockPredictor);
        let mut state = StreamingState::new(predictor, 100, 50);

        state.push_samples(&[1.0, 2.0, 3.0]);
        assert_eq!(state.buffer.len(), 3);
    }

    #[test]
    fn test_streaming_state_update_envelope() {
        let predictor = Box::new(MockPredictor);
        let mut state = StreamingState::new(predictor, 100, 50);

        let upper = vec![1.0, 2.0, 3.0];
        let lower = vec![-1.0, -2.0, -3.0];
        state.update_envelope(upper.clone(), lower.clone());

        assert_eq!(state.last_envelope.0, upper);
        assert_eq!(state.last_envelope.1, lower);
    }

    #[test]
    fn test_streaming_state_metrics_and_algorithm() {
        let predictor = Box::new(MockPredictor);
        let mut state = StreamingState::new(predictor, 100, 50);

        // High stationarity -> EMD
        let metrics_high = IntermittencyMetrics::new(0.1, 0.05, 0.85);
        state.update_metrics(metrics_high);
        assert_eq!(state.algorithm(), AdaptiveAlgorithm::EMD);

        // Moderate stationarity -> EEMD
        let metrics_mid = IntermittencyMetrics::new(0.3, 0.2, 0.65);
        state.update_metrics(metrics_mid);
        assert_eq!(state.algorithm(), AdaptiveAlgorithm::EEMD);

        // Low stationarity -> CEEMDAN
        let metrics_low = IntermittencyMetrics::new(0.6, 0.5, 0.3);
        state.update_metrics(metrics_low);
        assert_eq!(state.algorithm(), AdaptiveAlgorithm::CEEMDAN);
    }

    #[test]
    fn test_streaming_state_sifting_history() {
        let predictor = Box::new(MockPredictor);
        let mut state = StreamingState::new(predictor, 100, 50);

        let iter1 = SiftingIteration::new(0, 0.1, 10, false);
        let iter2 = SiftingIteration::new(1, 0.05, 12, true);

        state.record_sifting_iteration(iter1.clone());
        state.record_sifting_iteration(iter2.clone());

        let history = state.sifting_history_vec();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_streaming_state_reset() {
        let predictor = Box::new(MockPredictor);
        let mut state = StreamingState::new(predictor, 100, 50);

        state.push_samples(&[1.0, 2.0, 3.0]);
        state.next_chunk();
        state.next_chunk();
        state.update_envelope(vec![1.0], vec![-1.0]);

        assert_eq!(state.chunk_id, 2);
        assert_eq!(state.buffer.len(), 3);
        assert!(!state.last_envelope.0.is_empty());

        state.reset();

        assert_eq!(state.chunk_id, 0);
        assert_eq!(state.buffer.len(), 0);
        assert!(state.last_envelope.0.is_empty());
    }

    #[test]
    fn test_intermittency_metrics_recommended_algorithm() {
        let m_emd = IntermittencyMetrics::new(0.1, 0.05, 0.9);
        assert_eq!(m_emd.recommended_algorithm(), AdaptiveAlgorithm::EMD);

        let m_eemd_high = IntermittencyMetrics::new(0.3, 0.2, 0.8);
        assert_eq!(m_eemd_high.recommended_algorithm(), AdaptiveAlgorithm::EEMD);

        let m_eemd_low = IntermittencyMetrics::new(0.3, 0.2, 0.5);
        assert_eq!(m_eemd_low.recommended_algorithm(), AdaptiveAlgorithm::EEMD);

        let m_ceemdan = IntermittencyMetrics::new(0.6, 0.5, 0.3);
        assert_eq!(m_ceemdan.recommended_algorithm(), AdaptiveAlgorithm::CEEMDAN);
    }
}
