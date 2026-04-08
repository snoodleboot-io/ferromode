# V2.2 Neural Boundary Prediction - Architecture Design

**Document:** V2.2 Neural Boundary Prediction Models  
**Version:** 1.0  
**Date:** 2026-04-08  
**Author:** Ferromode AI Team  
**Status:** Design Phase  

## 1. Overview

V2.2 introduces learnable LSTM-based boundary prediction models to improve end-effect handling in streaming and regular EMD. The system intelligently selects between LSTM (for non-stationary signals) and AR (for stationary signals) based on signal characteristics.

## 2. Model Architecture

### 2.1 LSTM Predictor (Neural Model)

**Purpose:** Capture complex patterns in non-stationary, high-intermittency signals

**Architecture:**
```
Input Layer:     N=20 samples
                 └─→ Dense(128) + Tanh
LSTM Layer 1:    128 hidden units
                 └─→ Dropout(0.2)
LSTM Layer 2:    128 hidden units
                 └─→ Dropout(0.2)
Output Layer:    K=10 predicted samples
                 └─→ Dense(10) + Tanh (bounded to [-1, 1])
```

**Hyperparameters:**
- **Input window:** N = 20 samples (default, configurable)
- **Output horizon:** K = 10 predicted samples (default, configurable)
- **Hidden units:** 2 × 128 LSTM cells
- **Activation:** Tanh (bounded predictions, better for oscillatory signals)
- **Dropout:** 0.2 (regularization to prevent overfitting)
- **Loss function:** MSE (Mean Squared Error) + L2 regularization
- **Optimizer:** Adam (lr=0.001, weight decay=1e-5)
- **Batch size:** 32
- **Epochs:** 50-100 with early stopping (patience=5)
- **Normalization:** Input signals normalized to [-1, 1] range

**Design Rationale:**
- Tanh activation provides bounded outputs, reducing extreme extrapolation errors
- Two LSTM layers capture temporal dependencies with sufficient depth
- Dropout regularization prevents overfitting on finite training data
- L2 regularization encourages smaller weights for smoother predictions
- Early stopping prevents memorization

### 2.2 AR Model Predictor (Baseline)

**Purpose:** Fast, lightweight baseline for stationary signals

**Architecture:**
```
AR(5) Model:  x[n] = a₀·x[n-1] + a₁·x[n-2] + a₂·x[n-3] + a₃·x[n-4] + a₄·x[n-5] + ε[n]
```

**Hyperparameters:**
- **Order:** 5 (default, typically sufficient for boundary prediction)
- **Estimation:** Yule-Walker equations (O(n) complexity)
- **Prediction:** Recursive substitution for K steps ahead

**Advantages:**
- No neural network overhead (fast inference: < 0.1ms)
- Transparent, interpretable coefficients
- Optimal for stationary signals
- No training data required (adaptive fitting)

### 2.3 Model Selection Strategy

**Decision Logic:**
```
signal_intermittency = IntermittencyMetrics::compute(signal)
stationarity_score = signal_intermittency.stationarity_score  // Range: [0, 1]

if stationarity_score > THRESHOLD (default: 0.7):
    use AR model (fast, lightweight)
else:
    use LSTM model (adaptive, better for non-stationary)
```

**Rationale:**
- High stationarity (score > 0.7) → signal is quasi-periodic, AR is optimal
- Low stationarity (score ≤ 0.7) → signal has time-varying characteristics, LSTM is better

**Configurable Threshold:**
```rust
pub struct BoundaryPredictionConfig {
    pub lstm_enabled: bool,
    pub stationarity_threshold: f64,  // Default: 0.7
    pub ar_order: usize,              // Default: 5
    pub lstm_window: usize,           // Default: 20
    pub lstm_horizon: usize,          // Default: 10
}
```

## 3. Model Serialization & Deployment

### 3.1 ONNX Format

**Choice:** ONNX (Open Neural Network Exchange)

**Advantages:**
- Industry-standard, language-agnostic format
- Excellent tooling for model optimization and quantization
- Rust support via `ort` crate (pure Rust, no Python dependency at runtime)
- Fast inference with optimizations
- Supports dynamic batching

### 3.2 Model Quantization

**Strategy:** Post-training quantization

**Formats:**
1. **FP32** (original): ~4 MB (for reference/benchmarking)
2. **FP16** (half precision): ~2 MB (recommended for deployment)
3. **INT8** (8-bit integer): ~1 MB (fastest, requires calibration)

**Process:**
```
PyTorch/TensorFlow → ONNX (FP32) 
                   → Quantize to FP16/INT8
                   → Validate accuracy drop < 2%
                   → Serialize to models/lstm_predictor.onnx
```

**Performance/Size Trade-off:**
- FP32: Max accuracy, 4 MB
- FP16: 99%+ accuracy, 2 MB ✓ **RECOMMENDED**
- INT8: 95%+ accuracy, 1 MB

### 3.3 Directory Structure

```
ferromode/
├── models/
│   ├── lstm_predictor.onnx     (2 MB, FP16 quantized)
│   ├── lstm_predictor_fp32.onnx (backup, 4 MB)
│   └── README.md               (model specs, sources, retraining guide)
├── src/
│   └── adapters/
│       ├── boundary_prediction/
│       │   ├── mod.rs          (public API, model selection logic)
│       │   ├── lstm.rs         (LSTM wrapper, ONNX inference)
│       │   ├── ar.rs           (AR model reuse from v2.0)
│       │   ├── models.rs       (serialization, loading)
│       │   └── config.rs       (configuration structs)
│       └── streaming/
│           └── predictor.rs    (existing BoundaryPrediction trait)
└── Cargo.toml                  (add ort, ndarray dependencies)
```

## 4. Inference Engine

### 4.1 ONNX Runtime Integration

**Crate:** `ort` (ONNX Runtime Rust bindings)

**Architecture:**
```rust
pub struct OnnxInferenceEngine {
    session: ort::Session,     // Cached ONNX session
    input_shape: Vec<usize>,   // [batch_size=1, seq_len=20, features=1]
    output_shape: Vec<usize>,  // [batch_size=1, horizon=10]
}

impl OnnxInferenceEngine {
    pub fn infer(&self, input: &[f64]) -> Result<Vec<f64>, EmdError>;
    pub fn batch_infer(&self, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, EmdError>;
}
```

### 4.2 LstmModel Implementation

**Public API:**
```rust
pub struct LstmModel {
    engine: OnnxInferenceEngine,
    normalization: SignalNormalizer,  // Maintains running min/max
}

impl LstmModel {
    pub fn load(model_path: &str) -> Result<Self, EmdError>;
    pub fn load_default() -> Result<Self, EmdError>;  // Load from embedded models/
    pub fn predict(&self, signal: &[f64], n_ahead: usize) -> Vec<f64>;
}

impl BoundaryPrediction for LstmModel {
    fn predict(&self, signal: &[f64], n_ahead: usize) -> Vec<f64>;
    fn fit(&mut self, signal: &[f64]) -> Result<(), EmdError>;
    fn clone_box(&self) -> Box<dyn BoundaryPrediction>;
}
```

### 4.3 Performance Requirements

**Inference Speed:**
- Target: < 1ms per prediction on CPU
- Baseline: AR model < 0.1ms
- Expected LSTM: 0.5-0.8ms (with FP16 quantization)

**Memory:**
- Model size: < 2 MB (FP16)
- Working memory: < 50 MB per thread
- Cache: Optional result caching for repeated boundaries

**Latency Budget:**
```
Signal chunk (256 samples) @ 1 kHz = 256 ms real time
Prediction overhead (10 samples @ < 1ms) ≈ negligible relative to EMD compute
```

## 5. Training Data Requirements

### 5.1 Synthetic Signal Types (1000+ training signals)

1. **Pure Tones** (10% of dataset)
   - Frequency: 0.05 to 0.5 (normalized)
   - Amplitude: 1.0
   - Duration: 500 samples

2. **Chirps** (15% of dataset)
   - Start freq: 0.05, End freq: 0.5
   - Linear frequency sweep
   - Duration: 500 samples

3. **AM/FM Modulated** (15% of dataset)
   - Carrier: 0.2, Modulation: 0.05-0.15
   - Depth: 50-90%
   - Duration: 500 samples

4. **Non-stationary (Frequency Sweeps)** (20% of dataset)
   - Multiple frequency components
   - Time-varying amplitudes
   - Duration: 500 samples

5. **Intermittent/Burst Signals** (20% of dataset)
   - Bursts of different frequencies
   - Quiet periods (near zero)
   - Duration: 500-1000 samples

6. **Real-world Signals** (20% of dataset)
   - EEG (publicly available datasets)
   - Seismic (IRIS seismic data)
   - Vibration (bearing degradation datasets)
   - Duration: 1000+ samples

### 5.2 Training Split

- **Training:** 70% (700 signals)
- **Validation:** 15% (150 signals, for early stopping)
- **Test:** 15% (150 signals, for final evaluation)

### 5.3 Data Augmentation

1. **Noise Injection:** Add Gaussian noise (SNR: 20-40 dB)
2. **Time Scaling:** Stretch/compress signals (0.9-1.1x)
3. **Amplitude Scaling:** Normalize to different energy levels
4. **Phase Shift:** Circular shift of training windows

## 6. Training Pipeline

### 6.1 Framework Choice

**Recommendation:** PyTorch

**Rationale:**
- Mature, well-supported ONNX export
- Rich ecosystem for signal processing
- Easy to implement custom LSTM layers
- Performance comparable to TensorFlow

### 6.2 Training Loop

```python
# Pseudocode
for epoch in range(50):
    for batch in training_dataloader:
        signals, targets = batch  # (batch_size, seq_len) → (batch_size, horizon)
        
        # Forward pass
        predictions = model(signals)
        loss = mse_loss(predictions, targets) + l2_regularization
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 5 == 0:
            val_loss = evaluate(model, val_dataloader)
            if val_loss > best_val_loss + patience_threshold:
                if patience_counter > 5:
                    break  # Early stopping
```

### 6.3 Hyperparameter Tuning

**Fixed Hyperparameters:**
- Learning rate: 0.001
- Batch size: 32
- Weight decay (L2): 1e-5
- Dropout: 0.2
- Early stopping patience: 5 epochs

**Tunable (if needed):**
- Hidden units: [64, 128, 256]
- Number of LSTM layers: [1, 2, 3]
- Input window size: [10, 20, 30]
- Output horizon: [5, 10, 15]

## 7. Model Validation

### 7.1 Accuracy Metrics

**Primary Metric: MSE (Mean Squared Error)**
```
MSE = (1/N) * Σ(predicted - actual)²
Target: MSE < 0.05 on test set
```

**Secondary Metrics:**
- **MAE (Mean Absolute Error):** Average absolute deviation
- **RMSE (Root Mean Squared Error):** Interpretable in original signal units
- **MAPE (Mean Absolute Percentage Error):** Relative error

### 7.2 End-Effect Validation

**Test Procedure:**
1. Decompose signal with boundary extension (AR vs LSTM)
2. Compute IMF orthogonality index (should be > 0.9)
3. Measure IMF reconstruction error (should be < 0.01)
4. Compare envelope quality at boundaries (visual inspection + metrics)

**Target: LSTM achieves 30%+ reduction in boundary artifacts vs AR**

### 7.3 Model Stability Tests

- **Constant signal:** Prediction should be constant
- **Periodic signal:** Prediction should extend correctly
- **Step function:** Handle discontinuities gracefully
- **Noisy signal:** Robust to input noise

## 8. Integration Points

### 8.1 Streaming EMD

**Update:** `StreamingEmd::new_with_predictor(config)`
```rust
let config = BoundaryPredictionConfig::default();
let streaming = StreamingEmd::new_with_predictor(signal, config)?;
// Automatically selects AR or LSTM based on signal characteristics
```

### 8.2 Batch EMD

**Update:** `emd()` function with optional model selection
```rust
pub fn emd_with_boundary_prediction(
    signal: &Signal,
    config: &EmdConfig,
    boundary_config: &BoundaryPredictionConfig,
) -> Result<ImfCollection, EmdError>;
```

### 8.3 Model Selection Logic

**Integration:** Compute intermittency metrics, select model, apply boundary extension

```rust
fn select_boundary_predictor(signal: &Signal, config: &BoundaryPredictionConfig) 
    -> Result<Box<dyn BoundaryPrediction>, EmdError> 
{
    let metrics = IntermittencyMetrics::compute(signal);
    
    if metrics.stationarity_score > config.stationarity_threshold {
        Ok(Box::new(ArModel::new(config.ar_order)?))
    } else {
        Ok(Box::new(LstmModel::load_default()?))
    }
}
```

## 9. Performance Targets

### 9.1 Runtime Performance

- **LSTM inference:** < 1 ms per prediction (10 samples)
- **AR inference:** < 0.1 ms (baseline comparison)
- **Total boundary prediction:** < 5% of EMD compute time

### 9.2 Model Size

- **Quantized LSTM:** < 2 MB (FP16 recommended)
- **Unquantized (reference):** < 4 MB (FP32)

### 9.3 Accuracy

- **Test MSE:** < 0.05
- **End-effect reduction:** > 30% vs AR baseline
- **IMF orthogonality:** > 0.9 on synthetic signals
- **Reconstruction error:** < 0.01

## 10. Backward Compatibility

- **Existing code:** No breaking changes
- **BoundaryPrediction trait:** Unchanged (both AR and LSTM implement it)
- **Default behavior:** AR model (no ONNX dependency required)
- **Optional feature flag:** `boundary-prediction` for LSTM support

## 11. Dependencies

### 11.1 Training (Python, not in final binary)

```
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.23.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

### 11.2 Runtime (Rust)

```
ort = { version = "2.0", features = ["load-dynamic"] }
ndarray = { version = "0.15", features = ["serde"] }
```

### 11.3 Optional Features

```toml
[features]
boundary-prediction = ["ort"]  # Enables LSTM predictor
```

## 12. Success Criteria

- [ ] LSTM model architecture finalized and documented
- [ ] AR(5) baseline integrated and validated
- [ ] Model selection logic implemented
- [ ] Training pipeline validated on synthetic data
- [ ] ONNX quantization pipeline tested
- [ ] < 1ms inference latency demonstrated
- [ ] 30%+ end-effect reduction measured
- [ ] Integration with streaming/EMD complete
- [ ] All tests passing (unit + integration + benchmark)
- [ ] Documentation complete with examples

## 13. Timeline

- **Week 1:** PHASE 1 (Architecture Design)
- **Week 2:** PHASE 2 (Training Data & Pipeline)
- **Week 3-4:** PHASE 3-4 (Integration & Training Infrastructure)
- **Week 4:** PHASE 5 (Validation & Optimization)

## 14. References

- Wu, Z., & Huang, N. E. (2009). Ensemble empirical mode decomposition: a noise-assisted data analysis method. Advances in adaptive data analysis, 1(01), 1-41.
- Huang, N. E., et al. (2009). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. Proceedings of the Royal Society, A454, 903-995.
- ONNX Runtime: https://onnxruntime.ai/
- PyTorch LSTM Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

---

**Document Status:** APPROVED for implementation  
**Next Step:** PHASE 1 - Create boundary_prediction/ module structure
