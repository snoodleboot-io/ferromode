# V2.0 Streaming Requirements Document

**Release:** v2.0 (Streaming Adapter)  
**Status:** Ready for Implementation  
**Updated:** April 8, 2026  
**Effort Estimate:** XL (4-6 weeks)

---

## 1. Executive Summary

V2.0 introduces **chunk-based, streaming decomposition** through the `StreamingAdapter`. This enables real-time EMD processing on continuous data streams with fixed-memory consumption and latency guarantees.

**Key Capabilities:**
- Stateful chunk-by-chunk processing for time-series streams
- Boundary prediction to minimize stream endpoints effects
- Streaming ↔ batch equivalence validation
- < 10ms latency per chunk (1000 samples)
- Ring-buffer memory model (fixed allocation)

**Non-Goal:** GPU acceleration (v2.1), differentiable computation (v2.4), multi-dimensional signals (v2.3).

---

## 2. Epic Breakdown

### Epic 1: Streaming State Management
**Features:**
- StreamingState struct with chunk tracking
- Ring buffer for fixed-memory streaming
- Sifting history and envelope persistence
- Predictor state integration

### Epic 2: Boundary Prediction Framework
**Features:**
- BoundaryPrediction trait (pluggable models)
- Fallback AR model for boundary extension
- Predictor state management
- Prediction caching

### Epic 3: Chunk Decomposition Engine
**Features:**
- StreamingAdapter struct with stateful decomposition
- decompose_chunk() method (primary API)
- Chunk alignment and trimming
- Stream state synchronization

### Epic 4: Intermittency Detection
**Features:**
- IntermittencyMetrics struct (spectral entropy, extrema spacing CV)
- Stationarity scoring
- Adaptive algorithm selection
- Metrics per-chunk evaluation

### Epic 5: Streaming vs Batch Validation
**Features:**
- Cross-validation test suite
- Equivalence criteria (IMF tolerance < 1e-6)
- Latency benchmarks (< 10ms target)
- Memory profiling (< 100 MB peak RSS)

---

## 3. User Stories

### Story 1: Streaming Decomposition of Real-Time Data
**As a** real-time signal processor  
**I want to** decompose continuous data streams chunk-by-chunk  
**So that** I can process live data without buffering the entire signal

**Acceptance Criteria:**
- [ ] `StreamingAdapter::new()` accepts EmdConfig and BoundaryPrediction trait
- [ ] `decompose_chunk(&mut self, chunk: &Signal) -> ChunkResult` processes without blocking
- [ ] Chunks can be 100–10k samples; latency scales linearly
- [ ] Memory footprint is deterministic (ring buffer, no growth)
- [ ] State persists across chunks automatically

**Definition of Done:**
- Core implementation complete
- All acceptance criteria verified
- Latency benchmark passes (< 10ms for 1000-sample chunks)
- Memory profiling shows < 100 MB RSS

---

### Story 2: Minimal Boundary Artifacts in Streaming
**As a** streaming decomposition user  
**I want to** use boundary prediction to extend chunks  
**So that** streaming results match batch decomposition (within tolerance)

**Acceptance Criteria:**
- [ ] BoundaryPrediction trait defines predict() interface
- [ ] AR model fallback extends chunks before sifting
- [ ] Predictor state updates after each chunk
- [ ] Streaming vs batch equivalence tests pass (tolerance: 1e-6)
- [ ] Boundary effects quantified and logged

**Definition of Done:**
- BoundaryPrediction trait implemented
- AR model integration complete
- Integration tests demonstrate streaming ≈ batch
- Documentation includes boundary effect comparison

---

### Story 3: Intermittency Awareness in Streaming
**As a** signal analyst  
**I want to** detect intermittency in real-time  
**So that** I can flag or adapt processing to non-stationary regions

**Acceptance Criteria:**
- [ ] IntermittencyMetrics computed per chunk
- [ ] Stationarity score in [0, 1]
- [ ] Adaptive algorithm selection (EMD/EEMD/CEEMDAN based on score)
- [ ] Metrics available in ChunkResult
- [ ] Threshold for "high intermittency" configurable

**Definition of Done:**
- IntermittencyMetrics struct implemented
- Detection algorithm integrated into StreamingAdapter
- Tests verify correct thresholds
- User documentation with examples

---

### Story 4: Streaming Latency SLA Compliance
**As a** real-time system architect  
**I want to** guarantee < 10ms latency per chunk  
**So that** streaming decomposition fits into real-time pipelines

**Acceptance Criteria:**
- [ ] Latency benchmarks for chunk sizes: 100, 500, 1000, 5000, 10000 samples
- [ ] p99 latency < 10ms for 1000-sample chunks
- [ ] Memory allocation in hot path is pre-allocated or borrowed
- [ ] No dynamic allocation in decompose_chunk() inner loop
- [ ] Latency tests run in CI on every commit

**Definition of Done:**
- Benchmark harness created (criterion.rs)
- All chunk sizes meet SLA
- Latency regression tests in CI
- Performance profile documented

---

### Story 5: Streaming and Batch Equivalence Validation
**As a** quality engineer  
**I want to** verify streaming mode produces equivalent results to batch  
**So that** I can trust streaming in production with same guarantees as batch

**Acceptance Criteria:**
- [ ] Equivalence tests compare streaming → batch on same signal
- [ ] IMF tolerance: < 1e-6 (element-wise L∞)
- [ ] Residue tolerance: < 1e-6
- [ ] Tests on diverse signals: stationary, non-stationary, intermittent
- [ ] Cross-platform validation (Linux, macOS, Windows)

**Definition of Done:**
- Test suite with 50+ signals (various characteristics)
- All tests passing on CI/CD
- Tolerance metrics documented
- Failure mode analysis documented

---

## 4. Acceptance Criteria

### Functional Requirements

1. **StreamingState Management**
   - [ ] Struct contains: chunk_id, sifting_history, last_envelope, predictor_state, buffer
   - [ ] State updates automatically after each chunk decomposition
   - [ ] History is bounded (configurable, default 100 iterations)
   - [ ] Serialization/deserialization works (for checkpointing)

2. **Boundary Prediction Interface**
   - [ ] BoundaryPrediction trait with predict(signal) → Vec<f64>
   - [ ] AR model implementation with configurable order
   - [ ] Fallback strategy if model fails
   - [ ] Predictor state persists and updates automatically

3. **Chunk Decomposition Engine**
   - [ ] StreamingAdapter holds config + state + predictor
   - [ ] decompose_chunk() accepts &Signal, returns ChunkResult
   - [ ] Chunks trimmed to original length (boundary extension is internal)
   - [ ] Result includes: IMFs, residue, intermittency metrics, state_valid flag

4. **Intermittency Metrics**
   - [ ] Spectral entropy computed per chunk (using FFT)
   - [ ] Extrema spacing coefficient of variation
   - [ ] Stationarity score ∈ [0, 1]
   - [ ] Adaptive algorithm selection based on stationarity

5. **Streaming vs Batch Equivalence**
   - [ ] Streaming decomposition of all chunks + concatenation = batch decomposition
   - [ ] Tolerance: < 1e-6 element-wise L∞ norm
   - [ ] Equivalence verified on 50+ test signals
   - [ ] Edge cases covered: short chunks, first/last chunks, discontinuities

### Performance Requirements

1. **Latency**
   - [ ] p50 latency < 5ms for 1000-sample chunks
   - [ ] p99 latency < 10ms for 1000-sample chunks
   - [ ] p999 latency < 20ms for 1000-sample chunks
   - [ ] Latency scales sub-linearly with chunk size

2. **Memory**
   - [ ] Peak RSS < 100 MB for 1-minute rolling window
   - [ ] Ring buffer capacity configurable, default 10k samples
   - [ ] No unbounded allocations in hot path
   - [ ] Memory profiling in CI

3. **Correctness**
   - [ ] Streaming ↔ batch IMF tolerance < 1e-6
   - [ ] Residue tolerance < 1e-6
   - [ ] Cross-platform consistency (variance < 1e-7)
   - [ ] Numerical stability over 10k+ chunks

### Quality Requirements

1. **Testing**
   - [ ] 100% code coverage (excluding fuzz targets)
   - [ ] Unit tests for StreamingState, BoundaryPrediction, ChunkResult
   - [ ] Integration tests for full decomposition
   - [ ] Property-based tests (proptest) for invariants
   - [ ] Stress tests: 10k+ chunks without memory leaks

2. **Documentation**
   - [ ] Inline code comments explaining non-obvious logic
   - [ ] Rustdoc for all public APIs
   - [ ] Architecture design document
   - [ ] User guide with examples
   - [ ] Boundary effect comparison (streaming vs batch)

3. **Regression**
   - [ ] Latency regression tests in CI
   - [ ] Memory regression profiling
   - [ ] Cross-platform regression (Linux, macOS, Windows)

---

## 5. Technical Design

### Module Structure

```rust
// New modules in crates/ferromode/src/adapters/

adapters/
├── mod.rs                    # Adapter layer facade
└── streaming/
    ├── mod.rs                # Public API
    ├── state.rs              # StreamingState struct
    ├── decomposer.rs         # StreamingAdapter struct
    ├── buffer.rs             # RingBuffer implementation
    ├── metrics.rs            # IntermittencyMetrics
    └── prediction.rs         # BoundaryPrediction trait + AR model

// Extend existing modules
types.rs                      # Add StreamingState, ChunkResult, etc.
analysis/                     # New module for entropy/stationarity
└── intermittency.rs          # Intermittency detection functions
```

### Key Types

```rust
// StreamingState — manages decomposition state across chunks
pub struct StreamingState {
    chunk_id: u64,
    sifting_history: Vec<SiftingIteration>,
    last_envelope: (Vec<f64>, Vec<f64>),
    predictor_state: Arc<dyn PredictorState>,
    buffer: RingBuffer<f64>,
}

pub trait PredictorState: Send + Sync {
    fn predict_next(&self, signal: &Signal) -> Vec<f64>;
    fn update(&mut self, signal: &Signal);
}

// IntermittencyMetrics — quantify signal non-stationarity
pub struct IntermittencyMetrics {
    pub spectral_entropy: f64,
    pub extrema_spacing_cv: f64,
    pub stationarity_score: f64,
}

pub enum AdaptiveAlgorithm {
    EMD,      // stationary: score > 0.8
    EEMD,     // mildly intermittent: 0.5 < score < 0.8
    CEEMDAN,  // highly intermittent: score < 0.5
}

// ChunkResult — output of chunk decomposition
pub struct ChunkResult {
    pub imfs: Vec<Signal>,
    pub residue: Signal,
    pub intermittency: IntermittencyMetrics,
    pub boundary_effect: BoundaryEffect,
    pub state_valid: bool,
}

// BoundaryPrediction — pluggable extension models
pub trait BoundaryPrediction: Send + Sync {
    fn predict(&self, signal: &Signal) -> Vec<f64>;
    fn update_state(&mut self, signal: &Signal);
}

pub struct ArModel {
    pub coefficients: Vec<f64>,
    pub order: usize,
}

// StreamingAdapter — main interface
pub struct StreamingAdapter {
    config: EmdConfig,
    state: StreamingState,
    predictor: Arc<dyn BoundaryPrediction>,
}

impl StreamingAdapter {
    pub fn new(
        config: EmdConfig,
        predictor: Arc<dyn BoundaryPrediction>,
    ) -> Result<Self> { ... }

    pub fn decompose_chunk(&mut self, chunk: &Signal) -> Result<ChunkResult> {
        // 1. Predict boundary extension
        // 2. Delegate to domain::decompose()
        // 3. Trim to original length
        // 4. Update state
        // 5. Return ChunkResult
    }

    pub fn state(&self) -> &StreamingState { ... }
    pub fn reset_state(&mut self) { ... }
}

pub struct RingBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    head: usize,
}
```

### Algorithm: decompose_chunk()

```
Input: chunk (Signal of length N)
Config: EmdConfig, BoundaryPrediction, RingBuffer

1. Check chunk validity (len > 0, no NaN/Inf)
2. Compute IntermittencyMetrics for chunk
3. Predict boundary extension using predictor
4. Extended = [chunk | predicted_boundary]
5. Decompose using domain::decompose(extended, config)
6. Trim IMFs back to length N
7. Update StreamingState:
   - Increment chunk_id
   - Append to sifting_history
   - Update last_envelope
   - Predictor::update_state()
   - RingBuffer::push()
8. Return ChunkResult {
     imfs: trimmed_imfs,
     residue: trimmed_residue,
     intermittency: metrics,
     boundary_effect: {...},
     state_valid: true
   }

Output: ChunkResult
```

### Intermittency Detection Algorithm

```
Input: signal (Signal of length N)

1. Compute spectral entropy using FFT
   entropy = -sum(P_k * log(P_k))  where P_k = |FFT_k|^2 / sum(|FFT|^2)
   max_entropy = log(N/2)
   normalized_entropy = entropy / max_entropy

2. Find extrema (local min/max)
   extrema_count = count of local min/max in signal

3. Compute spacing between extrema
   spacings = [extrema[i+1] - extrema[i] for i in range(len-1)]
   mean_spacing = mean(spacings)
   std_spacing = std(spacings)
   cv_spacing = std_spacing / mean_spacing  (coefficient of variation)

4. Compute stationarity score
   spectral_entropy_contribution = 1 - normalized_entropy  (high entropy → intermittent)
   spacing_regularity = 1 / (1 + cv_spacing)  (uniform spacing → stationary)
   stationarity_score = 0.6 * spacing_regularity + 0.4 * spectral_entropy_contribution
   clamp to [0, 1]

5. Determine adaptive algorithm
   if stationarity_score > 0.8: AdaptiveAlgorithm::EMD
   elif stationarity_score > 0.5: AdaptiveAlgorithm::EEMD
   else: AdaptiveAlgorithm::CEEMDAN

Output: IntermittencyMetrics {
  spectral_entropy: normalized_entropy,
  extrema_spacing_cv: cv_spacing,
  stationarity_score: score
}
```

### AR Model Implementation

```rust
impl BoundaryPrediction for ArModel {
    fn predict(&self, signal: &Signal) -> Vec<f64> {
        // Auto-regressive model: x[n] = sum(a[i] * x[n-i]) + noise
        let n = signal.len();
        let p = self.order;
        let mut extended = signal.to_vec();
        
        // Extend by 10% of signal length (configurable)
        let extend_len = (n as f64 * 0.1).max(10.0) as usize;
        
        for _ in 0..extend_len {
            let mut pred = 0.0;
            for i in 0..p {
                if n > i {
                    pred += self.coefficients[i] * extended[n - 1 - i];
                }
            }
            extended.push(pred);
        }
        
        extended[n..].to_vec()  // Return only the extension
    }

    fn update_state(&mut self, signal: &Signal) {
        // Re-fit AR model on new chunk (using Yule-Walker equations)
        // This updates self.coefficients
    }
}
```

---

## 6. File Structure Changes

### New Files

```
crates/ferromode/src/
├── adapters/
│   ├── mod.rs (new)
│   └── streaming/
│       ├── mod.rs (new)
│       ├── state.rs (new)
│       ├── decomposer.rs (new)
│       ├── buffer.rs (new)
│       ├── metrics.rs (new)
│       └── prediction.rs (new)
├── analysis/
│   ├── mod.rs (existing, extend)
│   └── intermittency.rs (new)
└── types.rs (extend with StreamingState, ChunkResult)

tests/
├── streaming_vs_batch_equivalence.rs (new)
├── streaming_latency_bench.rs (new)
├── streaming_memory_profile.rs (new)
└── streaming_intermittency.rs (new)

crates/ferromode-py/
├── python/ferromode_py/
│   └── streaming.pyi (new)
└── tests/
    └── test_streaming_async.py (new)

docs/
└── streaming_design.md (new)
```

### Modified Files

```
crates/ferromode/src/
├── lib.rs (add streaming adapter exports)
├── api.rs (add streaming APIs if public)
└── types.rs (extend with StreamingState, etc.)

crates/ferromode-py/src/
└── lib.rs (add PyO3 bindings for StreamingAdapter)

Cargo.toml (no new dependencies required)
```

---

## 7. Test Strategy

### Unit Tests (25 tests)

1. **StreamingState** (4 tests)
   - [ ] new() creates valid state
   - [ ] update() increments chunk_id
   - [ ] buffer operations (push, peek, clear)
   - [ ] serialization round-trip

2. **RingBuffer** (5 tests)
   - [ ] new() with capacity
   - [ ] push() overwrites on full
   - [ ] peek()/get() semantics
   - [ ] clear() resets state
   - [ ] capacity enforcement

3. **ArModel** (4 tests)
   - [ ] new() with coefficients
   - [ ] predict() extends signal
   - [ ] update_state() re-fits coefficients
   - [ ] handles edge cases (short signal, order > len)

4. **IntermittencyMetrics** (6 tests)
   - [ ] compute_spectral_entropy() gives [0, 1]
   - [ ] extrema_spacing_cv() handles few extrema
   - [ ] stationarity_score() in [0, 1]
   - [ ] adaptive_algorithm_selection() correct thresholds
   - [ ] metrics on known signals (stationary, intermittent)

5. **ChunkResult** (2 tests)
   - [ ] new() creates valid result
   - [ ] boundary_effect properly set

### Integration Tests (20 tests)

1. **Chunk Decomposition** (8 tests)
   - [ ] decompose_chunk() processes valid signal
   - [ ] state updates correctly after chunk
   - [ ] multiple chunks without reset
   - [ ] reset_state() clears history
   - [ ] boundary extension is hidden from result
   - [ ] latency < 10ms for 1000-sample chunk
   - [ ] memory doesn't grow with chunks
   - [ ] handles edge cases (very short chunks)

2. **Streaming vs Batch** (8 tests)
   - [ ] single chunk = batch decomposition (exactly)
   - [ ] 2 chunks concatenated ≈ batch (tolerance 1e-6)
   - [ ] 10 chunks concatenated ≈ batch (tolerance 1e-6)
   - [ ] different chunk sizes (100, 500, 1000, 5000)
   - [ ] diverse signals (sine, chirp, non-stationary, intermittent)
   - [ ] cross-platform consistency (Linux, macOS)
   - [ ] reproducibility with seeded RNG
   - [ ] boundary effects quantified

3. **Predictor Integration** (4 tests)
   - [ ] AR predictor extends signal
   - [ ] predictor_state updates per chunk
   - [ ] fallback if prediction fails
   - [ ] custom predictor trait impl

### Property-Based Tests (10 tests, proptest)

1. **Invariants**
   - [ ] Streaming state chunk_id always increases
   - [ ] IMF count ≤ max_imfs (from config)
   - [ ] All IMFs alternating extrema (HHT definition)
   - [ ] Residue is monotonic or has ≤ 2 extrema
   - [ ] Intermittency score in [0, 1] always
   - [ ] Memory footprint monotonic (not decreasing)
   - [ ] Latency per chunk grows slowly with chunk size
   - [ ] Boundary extension < 20% of chunk length
   - [ ] Sifting history length bounded
   - [ ] stationarity_score invertible with entropy

### Stress Tests (5 tests)

1. **Long Streams**
   - [ ] 10,000 chunks without memory leak
   - [ ] 1,000,000 sample streaming decomposition
   - [ ] Memory profiling over 1 hour

2. **Edge Cases**
   - [ ] Very short chunks (5 samples)
   - [ ] Very long chunks (100k samples)
   - [ ] Constant signal (zero variance)
   - [ ] Pathological signals (NaN recovery)

### Benchmark Tests (criterion.rs)

```rust
// Chunk sizes: 100, 500, 1000, 5000, 10000 samples
// Signals: sine, chirp, synthetic non-stationary
// Metrics: latency (p50, p99, p999), memory, throughput
//
// SLA: p99 latency < 10ms for 1000-sample chunks

#[bench]
fn bench_decompose_chunk_1000(b: &mut Bencher) {
    // Setup
    // Measure p99 latency
    // Assert < 10ms
}
```

---

## 8. Intermittency Integration

### How V2.0 Supports Intermittency Analysis

1. **Real-Time Detection**
   - IntermittencyMetrics computed per chunk
   - Stationarity score available immediately
   - User can flag/alert on high intermittency

2. **Adaptive Algorithm Selection**
   - Stationarity score determines EMD variant
   - Allows runtime switching (future in v2.2+)

3. **Boundary Effects**
   - AR model learns intermittency patterns
   - Predictor updates based on chunk characteristics
   - Helps mitigate end effects in non-stationary regions

4. **Foundation for V2.2–V2.7**
   - v2.2: Neural boundary prediction trained on intermittent signals
   - v2.4: Learnable boundaries optimized for intermittency
   - v2.5: Entropy metrics quantifying intermittency impact
   - v2.7: Intermittency-aware validation suite

---

## 9. Implementation Tasks (Ordered)

### Phase 1: Foundation (Week 1)

- [ ] **T-238:** Create `adapters/` module structure and public API
- [ ] **T-239:** Implement RingBuffer<T> with tests
- [ ] **T-240:** Design StreamingState struct and PredictorState trait
- [ ] **T-241:** Implement IntermittencyMetrics and stationarity scoring
- [ ] **T-242:** Create spectral entropy and extrema spacing functions
- [ ] **T-243:** Write unit tests for IntermittencyMetrics (6 tests)

### Phase 2: Core Streaming (Week 2)

- [ ] **T-244:** Implement StreamingState struct with serialization
- [ ] **T-245:** Implement ArModel for boundary prediction
- [ ] **T-246:** Create BoundaryPrediction trait and default impl
- [ ] **T-247:** Implement StreamingAdapter struct
- [ ] **T-248:** Write unit tests for StreamingState (4 tests)
- [ ] **T-249:** Write unit tests for ArModel (4 tests)
- [ ] **T-250:** Write integration tests for chunk decomposition (8 tests)

### Phase 3: Validation & Equivalence (Week 2.5)

- [ ] **T-251:** Create streaming_vs_batch_equivalence test suite
- [ ] **T-252:** Generate reference signals (50+ diverse signal types)
- [ ] **T-253:** Implement tolerance checking (1e-6 L∞ norm)
- [ ] **T-254:** Cross-platform validation (Linux, macOS, Windows)
- [ ] **T-255:** Document boundary effects vs batch decomposition

### Phase 4: Performance & Benchmarking (Week 3)

- [ ] **T-256:** Create criterion.rs benchmark harness
- [ ] **T-257:** Implement latency benchmarks (p50, p99, p999)
- [ ] **T-258:** Profile memory usage (peak RSS, allocations)
- [ ] **T-259:** Implement latency SLA tests (< 10ms for 1000 samples)
- [ ] **T-260:** Create regression test suite in CI
- [ ] **T-261:** Document performance characteristics

### Phase 5: Stress Testing & Edge Cases (Week 3.5)

- [ ] **T-262:** Implement long-stream tests (10k+ chunks)
- [ ] **T-263:** Test edge cases (short chunks, discontinuities)
- [ ] **T-264:** Implement property-based tests (proptest, 10 invariants)
- [ ] **T-265:** Memory leak detection tests
- [ ] **T-266:** Cross-platform stress testing

### Phase 6: Python Bindings & Integration (Week 4)

- [ ] **T-267:** Add PyO3 wrappers for StreamingAdapter
- [ ] **T-268:** Create Python type stubs (streaming.pyi)
- [ ] **T-269:** Implement async streaming in Python (if needed)
- [ ] **T-270:** Write Python integration tests
- [ ] **T-271:** Create Python examples and documentation

### Phase 7: Documentation & Code Review (Week 4.5)

- [ ] **T-272:** Write Rustdoc for all public APIs
- [ ] **T-273:** Create architecture design document (streaming-design.md)
- [ ] **T-274:** Write user guide with examples
- [ ] **T-275:** Create boundary effect comparison document
- [ ] **T-276:** Code review and cleanup
- [ ] **T-277:** Final validation and sign-off

---

## 10. Exit Criteria

**All of the following must be true for v2.0 to ship:**

1. ✓ StreamingState, StreamingAdapter, BoundaryPrediction fully implemented
2. ✓ All 60 tests passing (25 unit + 20 integration + 10 property + 5 stress)
3. ✓ Streaming vs batch equivalence verified (tolerance 1e-6) on 50+ signals
4. ✓ Latency SLA met (p99 < 10ms for 1000-sample chunks)
5. ✓ Memory SLA met (peak RSS < 100 MB)
6. ✓ Cross-platform validation passing (Linux, macOS, Windows)
7. ✓ IntermittencyMetrics working end-to-end
8. ✓ Python bindings complete and tested
9. ✓ All public APIs documented (Rustdoc + user guide)
10. ✓ No performance regressions vs v1.x
11. ✓ Latency regression tests added to CI
12. ✓ Code review approved by 2+ maintainers
13. ✓ Security audit passed (no unsafe code, bounds checking)

---

## 11. Dependencies

### Rust Crates
- **rayon** (existing) — for parallelization if needed
- **ndarray** (existing) — for array operations
- **serde** (existing) — for state serialization
- **rustfft** (existing) — for spectral entropy
- **proptest** (dev only) — for property-based tests
- **criterion** (dev only) — for benchmarking

### No new external dependencies required.

---

## 12. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Streaming diverges from batch | Medium | High | Continuous validation tests; every CI run |
| Latency exceeds 10ms SLA | Medium | High | Early benchmarking; optimize hot path |
| Boundary effects inadequate | Medium | Medium | Cross-validation with batch; research papers |
| Memory grows unbounded | Low | High | Ring buffer enforces cap; profiling in CI |
| Predictor state becomes stale | Low | Medium | State updates after each chunk; tests verify |
| Intermittency metrics unreliable | Low | Medium | Validation on reference signals; threshold tuning |

---

## 13. Rollout Plan

### Phase 1: Internal Testing (Week 5)
- Run full test suite multiple times
- Cross-platform validation
- Load testing with real-world data

### Phase 2: Beta Release (Week 5.5)
- Tag v2.0-beta1
- Publish to crates.io with `features = ["streaming"]`
- Gather feedback from early adopters

### Phase 3: Production Release (Week 6)
- Address beta feedback
- Final performance tuning
- Tag v2.0 stable
- Publish to all package registries (PyPI, CRAN, etc.)

---

## 14. Success Metrics

After v2.0 ships, measure:

1. **Adoption:** # of users streaming decomposition (from telemetry)
2. **Reliability:** Uptime/errors in streaming mode (target: 99.99%)
3. **Performance:** Actual latency in production (p99 < 10ms)
4. **Correctness:** Cross-validation with existing EMD libraries
5. **Maintainability:** Time to onboard new streaming features

---

## 15. References

- [ARD_UPDATED_v2x.md](../next_work/ARD_UPDATED_v2x.md) — System architecture
- [HHT Definition](https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform) — IMF alternating extrema
- [AR Models](https://en.wikipedia.org/wiki/Autoregressive_model) — Boundary extension
- Existing v1.x test suite (for reference signals)
- Criterion.rs benchmarking guide

---

*Document Version: 1.0*  
*Last Updated: April 8, 2026*  
*Status: Ready for Implementation*  
