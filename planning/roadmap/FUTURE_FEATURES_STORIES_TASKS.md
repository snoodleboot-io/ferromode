# Features, Stories & Tasks — Future Releases (v2.x)
## Ferromode v2.0–v2.7: Detailed Feature Decomposition

**Created:** 2026-04-04  
**Status:** Active Planning Document  
**Linked Docs:** FUTURE_PRD.md · FUTURE_IMPLEMENTATION_MAP.md  

---

## 1. Release v2.0: Streaming & Real-Time Processing (Q3 2027)

### Feature: F-2.0.1 — Online EMD with Chunk-Based Processing

**Epic:** Enable processing of signals arriving in fixed-size chunks with continuous state maintenance.

#### User Story: US-2.0.1.1 — Stream EMD from IoT Sensor

```gherkin
Feature: Real-time EMD on streaming sensor data
Scenario: IoT sensor continuously publishes signal chunks

Given a time-series sensor emitting 1000-sample/second chunks
When each chunk arrives within 50ms
Then the decomposition must complete within 10ms
And the envelope must be smooth across chunk boundaries
And memory must not grow unbounded (ring buffer reuse)
And I can receive IMF updates via WebSocket in real-time
```

**Acceptance Criteria:**
- [ ] `StreamingDecomposer::new()` accepts config + boundary predictor
- [ ] `decompose_chunk()` returns IMFs + envelope for each chunk (< 10ms latency)
- [ ] State persists between chunks (stored in `StreamingState`)
- [ ] Envelope continuity metric > 0.95 at boundaries
- [ ] Memory profiling: peak RSS < 100 MB for 1-minute rolling window
- [ ] Works with both deterministic and probabilistic boundary predictors

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.0.1.1.1**: Design `StreamingState` struct | 2h | Arch | Todo |
| **T-2.0.1.1.2**: Implement `StreamingDecomposer::new()` | 3h | Dev | Todo |
| **T-2.0.1.1.3**: Implement `decompose_chunk()` method | 5h | Dev | Todo |
| **T-2.0.1.1.4**: Add stateful envelope tracking | 3h | Dev | Todo |
| **T-2.0.1.1.5**: Write acceptance tests (< 10ms latency) | 4h | Test | Todo |
| **T-2.0.1.1.6**: Benchmark vs batch mode on historical data | 3h | Perf | Todo |
| **T-2.0.1.1.7**: Document API + example (IoT use case) | 3h | Doc | Todo |

**Definition of Done:**
- All acceptance tests pass
- Performance benchmarks within targets
- Code review approved
- Documentation published
- Cross-language bindings updated (Python asyncio, JavaScript)

---

#### User Story: US-2.0.1.2 — Streaming EMD in Financial Data Pipeline

```gherkin
Feature: Real-time EMD for high-frequency trading signals
Scenario: Decompose tick-level price data with sub-millisecond latency

Given tick data (microsecond resolution) in 100-tick chunks
When each chunk arrives
Then EMD must complete in < 100µs
And decomposition must be numerically equivalent to batch-mode on historical data
And the system can handle 10,000 ticks/second
```

**Acceptance Criteria:**
- [ ] Latency < 100µs per 100-tick chunk (measured with nanosecond precision)
- [ ] Numerical equivalence to batch mode: error < 1e-10
- [ ] Throughput: ≥ 10,000 ticks/second on single CPU core
- [ ] No allocations in hot loop (pre-allocated buffers)
- [ ] Gradual buffer growth for longer time horizons (adaptive buffering)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.0.1.2.1**: Profile current `decompose_chunk()` latency | 2h | Perf | Todo |
| **T-2.0.1.2.2**: Implement zero-copy buffer ring | 4h | Dev | Todo |
| **T-2.0.1.2.3**: Optimize extrema detection for small chunks | 3h | Dev | Todo |
| **T-2.0.1.2.4**: Benchmark on real tick data | 2h | Perf | Todo |
| **T-2.0.1.2.5**: Write stress test (10k ticks/sec) | 3h | Test | Todo |

---

### Feature: F-2.0.2 — Boundary Prediction Models

**Epic:** Use machine learning to predict signal continuation at chunk boundaries, reducing end effects.

#### User Story: US-2.0.2.1 — Train & Deploy AR Model Boundary Predictor

```gherkin
Feature: AR(p) boundary prediction for streaming EMD
Scenario: Automatically predict boundary extension using autoregressive model

Given a signal chunk with last N points
When the model predicts the next N points
Then the prediction error should be < 10% of signal variance
And the predicted boundary should not introduce artifacts
And the model should be fast (< 1ms prediction)
```

**Acceptance Criteria:**
- [ ] AR model fits signal history via Yule-Walker or Burg's method
- [ ] Prediction error < 10% of signal variance
- [ ] Inference latency < 1ms per chunk
- [ ] Model stable for all boundary strategies
- [ ] Automated model selection (p chosen via AIC/BIC)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.0.2.1.1**: Implement AR model fit (Yule-Walker) | 3h | Dev | Todo |
| **T-2.0.2.1.2**: Implement boundary prediction from AR coefficients | 2h | Dev | Todo |
| **T-2.0.2.1.3**: Add model order selection (AIC/BIC) | 2h | Dev | Todo |
| **T-2.0.2.1.4**: Write tests: model stability on synthetic signals | 3h | Test | Todo |
| **T-2.0.2.1.5**: Benchmark: inference latency | 2h | Perf | Todo |
| **T-2.0.2.1.6**: Document + example (financial, geophysics) | 2h | Doc | Todo |

---

#### User Story: US-2.0.2.2 — LSTM Boundary Predictor (Optional)

```gherkin
Feature: Neural network boundary prediction
Scenario: Train LSTM on synthetic signals to predict boundaries

Given 10,000 synthetic signals with known properties
When LSTM trained on (last N points → next N points)
Then prediction error should beat AR model by > 20%
And inference latency should remain < 5ms
And model should transfer across signal types
```

**Acceptance Criteria:**
- [ ] LSTM beats AR baseline by > 20% on test set
- [ ] Inference < 5ms per chunk (GPU-compatible)
- [ ] Model size < 5 MB (for embedding in binaries)
- [ ] Transfer learning works: train on synthetic, test on real
- [ ] Gradual fallback to symmetric extension if LSTM fails

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.0.2.2.1**: Generate synthetic signal dataset (10k samples) | 3h | Dev | Todo |
| **T-2.0.2.2.2**: Define LSTM architecture + training pipeline | 4h | ML | Todo |
| **T-2.0.2.2.3**: Train on synthetic data | 8h | ML | Todo |
| **T-2.0.2.2.4**: Implement inference layer in Rust | 4h | Dev | Todo |
| **T-2.0.2.2.5**: Cross-validation: synthetic vs. real signals | 4h | Test | Todo |
| **T-2.0.2.2.6**: Model quantization (FP32 → FP16) | 3h | Dev | Todo |

---

### Feature: F-2.0.3 — Sliding Window EMD

**Epic:** Process signals with overlapping windows to detect time-varying modes.

#### User Story: US-2.0.3.1 — Time-Frequency Tracking via Sliding Windows

```gherkin
Feature: Sliding window decomposition with overlap
Scenario: Track how IMFs change over time

Given a 10-minute signal
When decomposed in 1-minute windows with 50% overlap
Then each window's IMFs should be smooth in time
And I can plot a time-frequency spectrogram showing IMF evolution
```

**Acceptance Criteria:**
- [ ] `SlidingWindowDecomposer` accepts window size, overlap, stride
- [ ] IMF stability metric > 0.9 between adjacent windows
- [ ] Output format compatible with matplotlib/plotly
- [ ] Latency scales linearly with window count

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.0.3.1.1**: Design `SlidingWindowDecomposer` struct | 2h | Arch | Todo |
| **T-2.0.3.1.2**: Implement sliding window logic | 3h | Dev | Todo |
| **T-2.0.3.1.3**: Add IMF stability tracking across windows | 3h | Dev | Todo |
| **T-2.0.3.1.4**: Write acceptance tests | 3h | Test | Todo |
| **T-2.0.3.1.5**: Benchmark: latency vs window overlap | 2h | Perf | Todo |
| **T-2.0.3.1.6**: Example: geophysical signal time-frequency tracking | 2h | Doc | Todo |

---

### Feature: F-2.0.4 — Adaptive Buffering

**Epic:** Automatically adjust buffer size based on signal stationarity to balance latency vs. quality.

#### User Story: US-2.0.4.1 — Auto-Buffering for Mixed Stationarity

```gherkin
Feature: Adaptive buffer sizing
Scenario: Stationary segments → larger buffer (better decomposition), 
          Transient segments → smaller buffer (lower latency)

Given a signal with varying nonstationarity
When the system measures local entropy
Then buffer size should adapt (small for spikes, large for trends)
And latency should remain predictable (bounded)
```

**Acceptance Criteria:**
- [ ] Stationarity metric computed via spectral entropy or time-frequency concentration
- [ ] Buffer size ranges from `min_buffer` to `max_buffer`
- [ ] Latency bounded: `buffer_size * sample_rate = max_latency`
- [ ] No mode mismatch: larger buffers still detect all IMFs

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.0.4.1.1**: Implement stationarity metric (spectral entropy) | 3h | Dev | Todo |
| **T-2.0.4.1.2**: Design buffer sizing policy (linear, exponential) | 2h | Arch | Todo |
| **T-2.0.4.1.3**: Implement adaptive buffering logic | 3h | Dev | Todo |
| **T-2.0.4.1.4**: Write tests: verify no mode mismatch | 3h | Test | Todo |
| **T-2.0.4.1.5**: Benchmark on geophysical/biomedical datasets | 2h | Perf | Todo |

---

### Feature: F-2.0.5 — WebSocket API for Real-Time IMF Streaming

**Epic:** Expose streaming EMD over WebSocket protocol for web-based visualization.

#### User Story: US-2.0.5.1 — Browser Visualization of Live EMD

```gherkin
Feature: WebSocket streaming to browser dashboard
Scenario: Live sensor data → server-side streaming EMD → browser display

Given a server streaming EMD of live data
When client connects via WebSocket
Then client receives IMF frames every 100ms
And can plot/animate decomposition in real-time
And bandwidth usage < 1 Mbps
```

**Acceptance Criteria:**
- [ ] Bidirectional WebSocket protocol (tungstenite or tokio-tungstenite)
- [ ] Server sends IMF frames as JSON or binary MessagePack
- [ ] Frame rate: ≥ 10 FPS (100ms latency)
- [ ] Bandwidth efficiency: < 1 Mbps for 1000 Hz signal, 8 IMFs
- [ ] Browser client example (HTML5 + Chart.js or Plotly)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.0.5.1.1**: Design WebSocket message format (schema) | 2h | Arch | Todo |
| **T-2.0.5.1.2**: Implement WebSocket server (tokio + tungstenite) | 4h | Dev | Todo |
| **T-2.0.5.1.3**: Implement binary serialization (MessagePack) | 2h | Dev | Todo |
| **T-2.0.5.1.4**: Write browser client (HTML5 + Plotly) | 4h | Frontend | Todo |
| **T-2.0.5.1.5**: Benchmark bandwidth usage | 2h | Perf | Todo |
| **T-2.0.5.1.6**: Security: rate limiting, auth token validation | 3h | Sec | Todo |

---

### Feature: F-2.0.6 — Python asyncio Integration

**Epic:** Native async support for Python streaming decomposition.

#### User Story: US-2.0.6.1 — Async Decomposition in Python

```gherkin
Feature: Python asyncio support for streaming EMD
Scenario: Decompose data while other async tasks run

Given an async function receiving signals
When EMD decomposes each chunk asynchronously
Then other tasks are not blocked
And I can use asyncio.gather() to decompose multiple signals in parallel
```

**Acceptance Criteria:**
- [ ] `ferromode_py.StreamingDecomposer` is async-compatible
- [ ] `await decomposer.decompose_chunk()` returns coroutine
- [ ] Works with `asyncio.gather()` for parallel processing
- [ ] No GIL contention (background thread or true async)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.0.6.1.1**: Wrap streaming API in PyO3 async wrapper | 3h | Dev | Todo |
| **T-2.0.6.1.2**: Test with asyncio.gather() | 2h | Test | Todo |
| **T-2.0.6.1.3**: Example: multi-signal async decomposition | 2h | Doc | Todo |

---

## 2. Release v2.1: GPU Acceleration (Q1 2028)

### Feature: F-2.1.1 — CUDA Kernel Implementation

**Epic:** Implement GPU-accelerated extrema detection and envelope interpolation.

#### User Story: US-2.1.1.1 — Parallel Extrema Detection

```gherkin
Feature: GPU-accelerated peak/valley detection
Scenario: Find extrema on 1M-sample signal in < 100ms

Given a signal on GPU memory
When extrema detection runs on GPU
Then speedup vs CPU is ≥ 10x
And results are bit-identical to CPU reference
```

**Acceptance Criteria:**
- [ ] CUDA kernel `find_extrema_gpu()` detects peaks & valleys
- [ ] Speedup: ≥ 10x for 1M samples
- [ ] Bit-identical results to CPU version
- [ ] Works with multiple GPUs

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.1.1.1.1**: Write CUDA kernel (extrema detection) | 6h | GPU Dev | Todo |
| **T-2.1.1.1.2**: Implement Rust wrapper + GPU memory management | 4h | Dev | Todo |
| **T-2.1.1.1.3**: Write parity tests (CPU vs GPU) | 4h | Test | Todo |
| **T-2.1.1.1.4**: Benchmark: latency, memory, speedup | 3h | Perf | Todo |

---

#### User Story: US-2.1.1.2 — GPU Cubic Spline Interpolation

```gherkin
Feature: GPU-accelerated envelope interpolation
Scenario: Interpolate upper/lower envelopes for 100k-sample signal

Given extrema points on GPU
When cubic spline runs on GPU
Then interpolation completes in < 50ms
And envelope quality is ≥ 99.9% match to CPU
```

**Acceptance Criteria:**
- [ ] GPU cubic spline implementation
- [ ] Latency: < 50ms for 100k samples
- [ ] Numerical equivalence to CPU (error < 1e-9)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.1.1.2.1**: Write CUDA kernel (cubic spline) | 8h | GPU Dev | Todo |
| **T-2.1.1.2.2**: Rust wrapper + memory management | 4h | Dev | Todo |
| **T-2.1.1.2.3**: Parity tests (CPU vs GPU) | 4h | Test | Todo |
| **T-2.1.1.2.4**: Benchmark | 2h | Perf | Todo |

---

### Feature: F-2.1.2 — CuPy Integration for Python

**Epic:** Expose GPU arrays directly in Python via CuPy.

#### User Story: US-2.1.2.1 — Python GPU Decomposition

```gherkin
Feature: CuPy-compatible Python API
Scenario: User passes GPU array, receives GPU result

Given a CuPy array on GPU
When ferromode-py decomposes it
Then result is also on GPU (no host transfer)
And user can further process on GPU
```

**Acceptance Criteria:**
- [ ] `ferromode_py.decompose_gpu(cupy_array)` accepts CuPy input
- [ ] Returns result as CuPy array
- [ ] Zero host-device copying (zero-copy binding)
- [ ] Performance: no overhead from Python->Rust->CUDA

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.1.2.1.1**: Design CuPy pointer passing protocol | 2h | Arch | Todo |
| **T-2.1.2.1.2**: Implement PyO3 wrapper for CuPy arrays | 4h | Dev | Todo |
| **T-2.1.2.1.3**: Test: zero-copy verification | 2h | Test | Todo |
| **T-2.1.2.1.4**: Example: GPU-based EEMD in Python | 3h | Doc | Todo |

---

### Feature: F-2.1.3 — PyTorch Layer

**Epic:** EMD as a differentiable PyTorch module.

#### User Story: US-2.1.3.1 — EMDLayer for End-to-End Learning

```gherkin
Feature: PyTorch EMD layer with gradient support
Scenario: Use EMD as preprocessing in neural network

Given a PyTorch model with EMD layer
When trained with backpropagation
Then gradients flow through EMD layer
And the model learns optimal preprocessing
```

**Acceptance Criteria:**
- [ ] `torch.nn.Module` subclass `EMDLayer`
- [ ] Forward pass: signal → IMFs
- [ ] Backward pass: gradients computed via implicit differentiation
- [ ] Works with mixed precision training
- [ ] Numerical gradient check passes (error < 1e-4)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.1.3.1.1**: Design differentiable EMD algorithm | 4h | Arch | Todo |
| **T-2.1.3.1.2**: Implement forward pass | 4h | Dev | Todo |
| **T-2.1.3.1.3**: Implement backward pass (implicit differentiation) | 6h | Dev | Todo |
| **T-2.1.3.1.4**: Write numerical gradient tests | 3h | Test | Todo |
| **T-2.1.3.1.5**: Example: classify signals via EMD→CNN | 4h | Doc | Todo |

---

## 3. Release v2.2: Advanced Boundary Conditions (Q3 2028)

### Feature: F-2.2.1 — Neural Boundary Prediction Models

**Epic:** Train and deploy neural networks for boundary effect mitigation.

#### User Story: US-2.2.1.1 — Pre-trained Boundary Predictor

```gherkin
Feature: Neural boundary prediction model
Scenario: User decomposes signal; model predicts optimal boundary extension

Given a signal with poor boundaries
When neural predictor estimates extension
Then decomposition quality improves by > 20%
And model is pre-trained (no user training required)
```

**Acceptance Criteria:**
- [ ] Pre-trained LSTM model bundled with library
- [ ] Quality improvement: > 20% on test signals (measured by HHT energy concentration)
- [ ] Model < 5 MB (for embedding)
- [ ] Inference latency < 50ms per signal

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.2.1.1.1**: Generate training dataset (10k synthetic signals) | 4h | ML | Todo |
| **T-2.2.1.1.2**: Define + train LSTM model | 8h | ML | Todo |
| **T-2.2.1.1.3**: Quantize model (FP32 → INT8) | 3h | ML | Todo |
| **T-2.2.1.1.4**: Implement inference in Rust | 4h | Dev | Todo |
| **T-2.2.1.1.5**: Cross-validation on real signals | 4h | Test | Todo |

---

### Feature: F-2.2.2 — Entropy-Based Mode Degradation Detection

**Epic:** Automatically flag IMFs contaminated by boundary effects.

#### User Story: US-2.2.2.1 — IMF Quality Flagging

```gherkin
Feature: Detect boundary-contaminated IMFs
Scenario: User receives warning if mode is unreliable

Given decomposition results
When entropy metric computed per IMF
Then degraded modes flagged with confidence score
And user can filter/exclude unreliable modes
```

**Acceptance Criteria:**
- [ ] Spectral entropy computed for each IMF
- [ ] Boundary contamination score (0–1) per IMF
- [ ] User can threshold: keep only IMFs with score > threshold
- [ ] Validated on synthetic signals with known contamination

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.2.2.1.1**: Implement spectral entropy computation | 2h | Dev | Todo |
| **T-2.2.2.1.2**: Compute boundary contamination metric | 3h | Dev | Todo |
| **T-2.2.2.1.3**: Write tests on synthetic data | 2h | Test | Todo |
| **T-2.2.2.1.4**: Example + visualization | 2h | Doc | Todo |

---

## 4. Release v2.3: Multidimensional EMD (Q1 2029)

### Feature: F-2.3.1 — 2D Image Decomposition

**Epic:** Extend EMD to grayscale and color images.

#### User Story: US-2.3.1.1 — Medical Image Analysis

```gherkin
Feature: 2D EMD for medical imaging
Scenario: Decompose CT scan into spatial IMFs

Given a 512×512 medical image
When 2D EMD applied
Then decomposition completes in < 5 seconds
And each IMF represents a spatial scale
```

**Acceptance Criteria:**
- [ ] `emd_2d()` function accepts image matrix
- [ ] Latency: < 5s for 512×512
- [ ] Output: vector of 2D IMFs (matrices)
- [ ] Separable and non-separable variants supported

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.3.1.1.1**: Design 2D EMD algorithm (separable vs. non-sep) | 4h | Arch | Todo |
| **T-2.3.1.1.2**: Implement separable 2D EMD | 6h | Dev | Todo |
| **T-2.3.1.1.3**: Optimize boundary handling for images (periodic/symmetric) | 4h | Dev | Todo |
| **T-2.3.1.1.4**: Write tests on synthetic + real medical images | 4h | Test | Todo |
| **T-2.3.1.1.5**: Benchmark: latency, memory | 2h | Perf | Todo |
| **T-2.3.1.1.6**: Example: CT scan feature extraction | 3h | Doc | Todo |

---

### Feature: F-2.3.2 — 3D Volumetric Decomposition

**Epic:** Support 3D data (fMRI, CT volumes, seismic cubes).

#### User Story: US-2.3.2.1 — fMRI Brain Analysis

```gherkin
Feature: 3D EMD for volumetric brain data
Scenario: Decompose fMRI volume into functional components

Given a 64×64×30 fMRI volume
When 3D EMD applied (GPU-accelerated)
Then decomposition completes in < 30 seconds
And voxel-wise IMFs reveal functional organization
```

**Acceptance Criteria:**
- [ ] `emd_3d()` on GPU or CPU
- [ ] Latency: < 30s for 256×256×256 on GPU
- [ ] Memory-efficient (streaming friendly)
- [ ] Output: 3D array of IMFs

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.3.2.1.1**: Design 3D EMD algorithm | 4h | Arch | Todo |
| **T-2.3.2.1.2**: Implement 3D extrema detection | 6h | Dev | Todo |
| **T-2.3.2.1.3**: GPU implementation (CUDA kernel) | 8h | GPU Dev | Todo |
| **T-2.3.2.1.4**: Tests on synthetic fMRI data | 4h | Test | Todo |
| **T-2.3.2.1.5**: Memory profiling | 2h | Perf | Todo |

---

## 5. Release v2.4: Machine Learning Integration (Q3 2029)

### Feature: F-2.4.1 — Differentiable EMD

**Epic:** Make EMD backpropagable for neural network training.

#### User Story: US-2.4.1.1 — End-to-End Training with EMD Preprocessing

```gherkin
Feature: Differentiable EMD layer
Scenario: Train signal classifier with learned EMD preprocessing

Given raw signals + labels
When model with EMDLayer trained via SGD
Then EMD parameters optimize for classification task
And accuracy improves vs fixed EMD
```

**Acceptance Criteria:**
- [ ] Gradients computed correctly (finite difference check < 1e-4 error)
- [ ] Works in PyTorch + TensorFlow
- [ ] Training converges on synthetic + real data
- [ ] Comparable to other preprocessing methods

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.4.1.1.1**: Design implicit differentiation strategy | 4h | Arch | Todo |
| **T-2.4.1.1.2**: Implement forward + backward (PyTorch) | 8h | Dev | Todo |
| **T-2.4.1.1.3**: Implement for TensorFlow/Keras | 8h | Dev | Todo |
| **T-2.4.1.1.4**: Numerical gradient verification tests | 4h | Test | Todo |
| **T-2.4.1.1.5**: Example: signal classification network | 4h | Doc | Todo |

---

### Feature: F-2.4.2 — Learnable Boundary Conditions

**Epic:** Neural network learns optimal boundary extension for task.

#### User Story: US-2.4.2.1 — Task-Specific Boundary Learning

```gherkin
Feature: Learnable boundary predictor
Scenario: Network learns to predict boundaries optimized for classification

Given classification dataset with signals
When boundary predictor trained end-to-end
Then boundaries improve classification accuracy
And convergence is stable (no gradient explosion)
```

**Acceptance Criteria:**
- [ ] Neural boundary predictor trainable via backprop
- [ ] Improves downstream task accuracy by > 5%
- [ ] Gradient stability verified (< 1e6 magnitude)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.4.2.1.1**: Design learnable boundary module | 3h | Arch | Todo |
| **T-2.4.2.1.2**: Implement in PyTorch | 5h | Dev | Todo |
| **T-2.4.2.1.3**: End-to-end training tests | 3h | Test | Todo |
| **T-2.4.2.1.4**: Benchmark: accuracy improvement | 3h | Perf | Todo |

---

## 6. Release v2.5: Advanced Post-Processing (Q1 2030)

### Feature: F-2.5.1 — Time-Frequency Entropy Metrics

**Epic:** Quantify signal complexity and decomposition quality.

#### User Story: US-2.5.1.1 — Spectral Entropy Analysis

```gherkin
Feature: Time-frequency entropy computation
Scenario: Quantify mode mixing and signal nonlinearity

Given decomposition result
When entropy metrics computed
Then user can assess decomposition quality
And identify problematic modes
```

**Acceptance Criteria:**
- [ ] Spectral entropy, permutation entropy, sample entropy implemented
- [ ] Results comparable to reference implementations
- [ ] Visualization-friendly output format

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.5.1.1.1**: Implement spectral entropy | 2h | Dev | Todo |
| **T-2.5.1.1.2**: Implement permutation entropy | 2h | Dev | Todo |
| **T-2.5.1.1.3**: Implement sample entropy | 2h | Dev | Todo |
| **T-2.5.1.1.4**: Tests + benchmarks | 2h | Test | Todo |
| **T-2.5.1.1.5**: Example visualization | 2h | Doc | Todo |

---

### Feature: F-2.5.2 — Mode Mixing Detection

**Epic:** Quantify and visualize frequency overlap between IMFs.

#### User Story: US-2.5.2.1 — Mode Mixing Visualization

```gherkin
Feature: Mode mixing metrics and heatmap
Scenario: Visualize which IMFs have overlapping frequencies

Given decomposition
When mode mixing computed
Then user sees heatmap of IMF overlap
And can assess decomposition reliability
```

**Acceptance Criteria:**
- [ ] Mode mixing metric (0–1) for each IMF pair
- [ ] Heatmap visualization
- [ ] Interpretation guidance (warn if > 0.3 overlap)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.5.2.1.1**: Implement mode mixing metric | 3h | Dev | Todo |
| **T-2.5.2.1.2**: Generate heatmap data | 2h | Dev | Todo |
| **T-2.5.2.1.3**: Example visualization script | 2h | Doc | Todo |

---

## 7. Release v2.6: Distributed Systems (Q3 2030)

### Feature: F-2.6.1 — Apache Arrow Integration

**Epic:** Enable zero-copy data exchange with Python/R data ecosystems.

#### User Story: US-2.6.1.1 — Arrow-Backed Signal Loading

```gherkin
Feature: Apache Arrow signal I/O
Scenario: Load signals from Arrow format without copying

Given signals in Arrow/Parquet format
When loaded and decomposed
Then zero-copy data passing
And interop with pandas/polars/data.table seamless
```

**Acceptance Criteria:**
- [ ] `Signal::from_arrow_array()` zero-copy binding
- [ ] `decompose_arrow()` returns Arrow output
- [ ] Works with Parquet files

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.6.1.1.1**: Implement Arrow bindings | 4h | Dev | Todo |
| **T-2.6.1.1.2**: Parquet I/O | 3h | Dev | Todo |
| **T-2.6.1.1.3**: Performance test (copy-free validation) | 2h | Perf | Todo |

---

### Feature: F-2.6.2 — Kubernetes Operator

**Epic:** Deploy EMD ensemble processing on Kubernetes.

#### User Story: US-2.6.2.1 — K8s Autoscaling EMD Jobs

```gherkin
Feature: Kubernetes operator for EMD ensemble
Scenario: Submit CEEMDAN job; auto-scale based on trial count

Given Kubernetes cluster
When EMDJob CRD submitted with trial count
Then pods auto-scale
And results aggregated
And cleanup automatic
```

**Acceptance Criteria:**
- [ ] Custom Resource Definition (CRD) for EMDJob
- [ ] Auto-scaling based on trial count
- [ ] Results aggregation + persistence
- [ ] Pod ready time < 5 seconds

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.6.2.1.1**: Define EMDJob CRD schema | 2h | Arch | Todo |
| **T-2.6.2.1.2**: Implement operator (kopf or operator-rs) | 8h | Dev | Todo |
| **T-2.6.2.1.3**: Deployment manifests (Helm chart) | 4h | DevOps | Todo |
| **T-2.6.2.1.4**: E2E test on minikube | 4h | Test | Todo |

---

### Feature: F-2.6.3 — gRPC Microservice

**Epic:** Language-agnostic RPC interface for EMD.

#### User Story: US-2.6.3.1 — gRPC EMD Service

```gherkin
Feature: gRPC-based EMD microservice
Scenario: Call EMD from any language via gRPC

Given gRPC server running
When client sends protobuf request
Then gets decomposition result back
And response time < 100ms (local) / < 500ms (network)
```

**Acceptance Criteria:**
- [ ] Protobuf schema for signal + config + result
- [ ] Server streaming for large decompositions
- [ ] Client authentication (mTLS)
- [ ] Example clients (Python, JavaScript, Go)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.6.3.1.1**: Define Protobuf schema | 2h | Arch | Todo |
| **T-2.6.3.1.2**: Implement gRPC server (tonic) | 6h | Dev | Todo |
| **T-2.6.3.1.3**: Client examples (Python, JS) | 4h | Dev | Todo |
| **T-2.6.3.1.4**: Load testing | 3h | Perf | Todo |

---

## 8. Release v2.7: Validation & Benchmarking (Q1 2031)

### Feature: F-2.7.1 — Reference Signal Library

**Epic:** Comprehensive synthetic + real signal dataset with ground truth.

#### User Story: US-2.7.1.1 — Signal Benchmark Suite

```gherkin
Feature: Standardized signal library
Scenario: Validate against reference signals

Given library of 1000+ signals (synthetic + real)
When decomposed with Ferromode
Then results can be compared against ground truth
And cross-implementation validation enabled
```

**Acceptance Criteria:**
- [ ] 500+ synthetic AM/FM signals with known IMFs
- [ ] 500+ real signals (geophysics, biomedicine, finance)
- [ ] Metadata: signal properties, expected mode count, noise level
- [ ] Distributed as crate/package

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.7.1.1.1**: Generate synthetic signal set | 8h | ML | Todo |
| **T-2.7.1.1.2**: Collect + curate real signal datasets | 12h | Data | Todo |
| **T-2.7.1.1.3**: Create metadata (JSON schema) | 3h | Dev | Todo |
| **T-2.7.1.1.4**: Package as crate/PyPI | 4h | DevOps | Todo |

---

### Feature: F-2.7.2 — Cross-Implementation Validation

**Epic:** Compare Ferromode against R emd, PyEMD, MATLAB toolbox.

#### User Story: US-2.7.2.1 — Reference Implementation Comparison

```gherkin
Feature: Automated cross-implementation testing
Scenario: Verify Ferromode against multiple reference implementations

Given same signal + config
When decomposed with Ferromode, R emd, PyEMD, MATLAB
Then results agree within 1e-10 relative error
And differences documented (algorithm variant vs numerical)
```

**Acceptance Criteria:**
- [ ] ≥ 100 test signals run against all implementations
- [ ] Results agree within tolerance
- [ ] Discrepancies documented (intentional algorithm differences)

**Implementation Tasks:**

| Task | Effort | Owner | Status |
|------|--------|-------|--------|
| **T-2.7.2.1.1**: Write R emd wrapper + comparison script | 4h | Dev | Todo |
| **T-2.7.2.1.2**: PyEMD wrapper | 4h | Dev | Todo |
| **T-2.7.2.1.3**: MATLAB wrapper (via MEX if available) | 4h | Dev | Todo |
| **T-2.7.2.1.4**: CI integration (run all comparisons) | 4h | DevOps | Todo |

---

## 9. Dependency Map

```
v2.0 (Streaming)
    └── Boundary Prediction (v2.2)
    
v2.1 (GPU)
    └── Differentiable (v2.4)
    └── 2D/3D EMD (v2.3)
    
v2.2 (Boundary)
    └── Streaming (v2.0)
    
v2.3 (2D/3D)
    └── GPU (v2.1, optional)
    
v2.4 (ML)
    └── GPU (v2.1)
    └── Boundary Prediction (v2.2)
    
v2.5 (Analysis)
    └── (No dependencies; standalone)
    
v2.6 (Distributed)
    └── Streaming (v2.0, optional)
    
v2.7 (Validation)
    └── (All previous versions)
```

---

## 10. Effort Estimation Summary

| Release | Effort (Person-Months) | Stories | Tasks |
|---------|------------------------|---------|-------|
| v2.0 | 3–4 | 6 | ~40 |
| v2.1 | 2–3 | 3 | ~25 |
| v2.2 | 2 | 2 | ~15 |
| v2.3 | 2–3 | 2 | ~20 |
| v2.4 | 3–4 | 2 | ~25 |
| v2.5 | 2 | 2 | ~15 |
| v2.6 | 2–3 | 3 | ~20 |
| v2.7 | 2–3 | 2 | ~20 |
| **Total** | **18–23** | **22** | **~180** |

---

## 11. Success Criteria Checklist

- [ ] All user stories have acceptance tests
- [ ] All acceptance tests pass
- [ ] Performance targets met
- [ ] Cross-implementation validation passed
- [ ] Documentation complete + examples working
- [ ] Language bindings updated
- [ ] Release notes published

---

*Last updated: April 4, 2026*  
*Maintained by: Core Engineering Team*
