# Execution Checklist — Future Releases (v2.x)
## Ferromode Project — Task-Level Tracking for v2.0–v2.7

**Created:** 2026-04-04  
**Total Tasks:** ~180 | **Status:** Planning Phase  
**Linked Docs:** FUTURE_PRD.md · FUTURE_IMPLEMENTATION_MAP.md · FUTURE_FEATURES_STORIES_TASKS.md  

---

## Legend

| Symbol | Meaning |
|--------|---------|
| 🔲 | TODO — Not started |
| 🔄 | IN_PROGRESS — Currently being worked on |
| 🔍 | REVIEW — Implementation complete, awaiting review |
| ✅ | DONE — Completed and approved |
| 🚫 | BLOCKED — Waiting on dependency or decision |

### Mode Tracking
- `architect` — Design, task breakdown, technical decisions
- `code` — Implementation
- `test` — Test writing and coverage
- `review` — Code review and approval
- `perf` — Performance optimization & benchmarking
- `docs` — Documentation
- `devops` — Deployment, CI/CD, infrastructure

### Engineering Practice Compliance
Each task must satisfy before marked DONE:
- **TDD**: Test written BEFORE implementation
- **ATDD**: Acceptance criteria automated as tests
- **DDD**: Domain-driven types and bounded contexts
- **Clean Code**: Clarity, single responsibility, DRY
- **Clean Architecture**: Inward-pointing dependencies
- **Numerical Validation**: Cross-language/cross-implementation testing

---

## Release v2.0: Streaming & Real-Time Processing (Q3 2027)

**Target:** Q3 2027  
**Exit Criteria:** Streaming EMD < 10ms latency; envelope continuity > 0.95; memory bounded  
**Total Tasks:** ~40  

### Feature F-2.0.1: Online EMD with Chunk-Based Processing

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-238 | Design `StreamingState` struct with fields for sifting history, envelope tracking, chunk counter | 🔲 | architect | — | 2h | Dependency: None |
| T-239 | Define acceptance tests (< 10ms latency, continuity > 0.95) | 🔲 | test | — | 2h | RED phase: failing tests first |
| T-240 | Implement `StreamingDecomposer::new()` and validation | 🔲 | code | — | 3h | GREEN phase: minimal code |
| T-241 | Implement `decompose_chunk()` core algorithm | 🔲 | code | — | 5h | Handle state persistence |
| T-242 | Add envelope continuity tracking across chunks | 🔲 | code | — | 3h | Use boundary predictor |
| T-243 | Implement ring buffer for memory-efficient chunk storage | 🔲 | code | — | 3h | Zero allocations in hot loop |
| T-244 | Write unit tests: state accumulation, envelope smoothness | 🔲 | test | — | 4h | ATDD: acceptance tests pass |
| T-245 | Benchmark: latency vs chunk size on synthetic signals | 🔲 | perf | — | 2h | Regression tests for future PRs |
| T-246 | Document streaming API + IoT example | 🔲 | docs | — | 3h | Code examples work end-to-end |
| T-247 | Add Python asyncio binding via PyO3 | 🔲 | code | — | 3h | async-compatible wrapper |
| T-248 | Add JavaScript Promise binding (WASM) | 🔲 | code | — | 3h | Async/await in browser |
| T-249 | Cross-language validation: streaming vs batch on historical data | 🔲 | test | — | 3h | Error < 1e-10 relative |

**Feature Status:** 🔲 Not Started | **Completion:** 0/12

---

### Feature F-2.0.2: Boundary Prediction Models

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-250 | Design AR(p) model interface and trait `BoundaryPrediction` | 🔲 | architect | — | 2h | Generic over model type |
| T-251 | Implement AR model fitting (Yule-Walker equation) | 🔲 | code | — | 3h | Numerical stability critical |
| T-252 | Implement AR boundary prediction (`next_n_samples()`) | 🔲 | code | — | 2h | Integrate with streaming |
| T-253 | Add automated order selection (AIC/BIC) | 🔲 | code | — | 2h | User-friendly defaults |
| T-254 | Write tests: model stability on synthetic signals | 🔲 | test | — | 3h | Edge cases: short signals, white noise |
| T-255 | Benchmark: AR prediction latency + accuracy | 🔲 | perf | — | 2h | < 1ms inference target |
| T-256 | Document AR boundary method with examples | 🔲 | docs | — | 2h | Compare vs symmetric/periodic |
| T-257 | (Optional) Design LSTM boundary predictor architecture | 🔲 | architect | — | 3h | May defer to v2.1 if time-constrained |
| T-258 | (Optional) Generate synthetic training data (10k signals) | 🔲 | ml | — | 4h | Diverse signal types |
| T-259 | (Optional) Train LSTM on GPU | 🔲 | ml | — | 8h | Transfer learning validation |
| T-260 | (Optional) Integrate LSTM inference in Rust | 🔲 | code | — | 4h | Model quantization |

**Feature Status:** 🔲 Not Started | **Completion:** 0/11 (core 6, optional 5)

---

### Feature F-2.0.3: Sliding Window EMD

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-261 | Design `SlidingWindowDecomposer` with configurable window/overlap/stride | 🔲 | architect | — | 2h | Reuse streaming infrastructure |
| T-262 | Implement sliding window iteration logic | 🔲 | code | — | 3h | Handle boundary padding |
| T-263 | Add IMF stability tracking (Pearson correlation between windows) | 🔲 | code | — | 3h | Metric: > 0.9 for good stability |
| T-264 | Write acceptance tests: stability metric validation | 🔲 | test | — | 3h | Test on multi-component signals |
| T-265 | Document + example: geophysical time-frequency tracking | 🔲 | docs | — | 2h | Visualization guidance |

**Feature Status:** 🔲 Not Started | **Completion:** 0/5

---

### Feature F-2.0.4: Adaptive Buffering

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-266 | Implement stationarity metric (spectral entropy) | 🔲 | code | — | 3h | Efficient computation |
| T-267 | Design buffer sizing policy (adaptive curve: min/max bounds) | 🔲 | architect | — | 2h | Bounded latency guarantee |
| T-268 | Implement adaptive buffering logic | 🔲 | code | — | 3h | Switch buffers based on entropy |
| T-269 | Write tests: verify no mode mismatch with varying buffer sizes | 🔲 | test | — | 3h | Critical for reliability |
| T-270 | Benchmark on real geophysical + biomedical datasets | 🔲 | perf | — | 2h | Latency variance measurement |

**Feature Status:** 🔲 Not Started | **Completion:** 0/5

---

### Feature F-2.0.5: WebSocket API for Real-Time Streaming

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-271 | Design WebSocket message protocol (schema for IMF frames) | 🔲 | architect | — | 2h | Binary vs JSON trade-off |
| T-272 | Implement WebSocket server (tokio + tungstenite) | 🔲 | code | — | 4h | Bidirectional streaming |
| T-273 | Implement efficient serialization (MessagePack or Arrow) | 🔲 | code | — | 2h | < 1 Mbps bandwidth target |
| T-274 | Write browser client (HTML5 + Plotly.js) | 🔲 | frontend | — | 4h | Real-time animation |
| T-275 | Benchmark: frame rate, bandwidth, latency | 🔲 | perf | — | 2h | 10 FPS minimum (100ms latency) |
| T-276 | Add security layer: rate limiting + token auth | 🔲 | code | — | 3h | Production-ready |
| T-277 | Document + deploy example server | 🔲 | devops | — | 3h | Docker container |

**Feature Status:** 🔲 Not Started | **Completion:** 0/7

---

## Release v2.1: GPU Acceleration (Q1 2028)

**Target:** Q1 2028  
**Exit Criteria:** 50x speedup for EEMD 200 trials on GPU; numerical equivalence to CPU  
**Total Tasks:** ~25  

### Feature F-2.1.1: CUDA Kernel Implementation

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-278 | Design CUDA kernel architecture for extrema detection | 🔲 | architect | — | 2h | Warp-level reduction patterns |
| T-279 | Write CUDA kernel code (find_extrema_gpu) | 🔲 | code | — | 6h | Data-parallel extrema detection |
| T-280 | Implement Rust wrapper + GPU memory management | 🔲 | code | — | 4h | RAII guards for deallocation |
| T-281 | Write CPU reference implementation for validation | 🔲 | code | — | 2h | Required for parity testing |
| T-282 | Write parity tests (CPU vs GPU on identical signals) | 🔲 | test | — | 4h | Bit-for-bit comparison |
| T-283 | Benchmark: speedup, memory, latency for 1M samples | 🔲 | perf | — | 3h | Target: ≥ 10x speedup |
| T-284 | Document CUDA kernel + build instructions | 🔲 | docs | — | 2h | CI build matrix (CUDA versions) |
| T-285 | Implement cubic spline CUDA kernel | 🔲 | code | — | 8h | More complex parallelization |
| T-286 | Write spline parity tests | 🔲 | test | — | 3h | Error < 1e-9 |
| T-287 | Benchmark spline: latency, accuracy | 🔲 | perf | — | 2h | Target: < 50ms for 100k samples |

**Feature Status:** 🔲 Not Started | **Completion:** 0/10

---

### Feature F-2.1.2: CuPy Integration

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-288 | Design CuPy pointer passing protocol (zero-copy) | 🔲 | architect | — | 2h | DLpack or similar |
| T-289 | Implement PyO3 wrapper for CuPy arrays | 🔲 | code | — | 4h | `decompose_gpu(cupy_array)` |
| T-290 | Write zero-copy verification test | 🔲 | test | — | 2h | Validate no host transfer |
| T-291 | Example: GPU-based EEMD in Python | 🔲 | docs | — | 3h | End-to-end workflow |

**Feature Status:** 🔲 Not Started | **Completion:** 0/4

---

### Feature F-2.1.3: PyTorch Layer

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-292 | Design differentiable EMD algorithm (implicit or unrolled) | 🔲 | architect | — | 4h | Critical design decision |
| T-293 | Implement forward pass in PyTorch | 🔲 | code | — | 4h | Signal → IMFs |
| T-294 | Implement backward pass (custom autograd.Function) | 🔲 | code | — | 6h | Gradients computed correctly |
| T-295 | Write numerical gradient tests (finite difference check) | 🔲 | test | — | 3h | Error < 1e-4 acceptance |
| T-296 | Test with mixed precision training | 🔲 | test | — | 2h | FP16 + FP32 stability |
| T-297 | Example: signal classification network with EMD layer | 🔲 | docs | — | 4h | Train end-to-end |

**Feature Status:** 🔲 Not Started | **Completion:** 0/6

---

## Release v2.2: Advanced Boundary Conditions (Q3 2028)

**Target:** Q3 2028  
**Exit Criteria:** Neural predictor improves quality by > 20%; entropy-based flagging reliable  
**Total Tasks:** ~15  

### Feature F-2.2.1: Neural Boundary Prediction

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-298 | Generate synthetic signal training dataset (10k samples) | 🔲 | ml | — | 4h | Diverse signal types |
| T-299 | Define + train LSTM model | 🔲 | ml | — | 8h | Cross-validation on holdout set |
| T-300 | Quantize model (FP32 → INT8) | 🔲 | ml | — | 3h | < 5 MB target |
| T-301 | Implement inference in Rust (ort or similar) | 🔲 | code | — | 4h | Model bundled with library |
| T-302 | Write cross-validation tests (synthetic vs real) | 🔲 | test | — | 4h | Transfer learning validation |

**Feature Status:** 🔲 Not Started | **Completion:** 0/5

---

### Feature F-2.2.2: Entropy-Based Mode Degradation Detection

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-303 | Implement spectral entropy computation per IMF | 🔲 | code | — | 2h | Efficient via FFT |
| T-304 | Design boundary contamination metric (0–1 score) | 🔲 | architect | — | 2h | Statistical basis |
| T-305 | Implement degradation flagging in DecompositionResult | 🔲 | code | — | 3h | User can filter IMFs |
| T-306 | Write tests on synthetic signals with known contamination | 🔲 | test | — | 2h | Validation signals |
| T-307 | Document + visualization example | 🔲 | docs | — | 2h | Heatmap of mode quality |

**Feature Status:** 🔲 Not Started | **Completion:** 0/5

---

## Release v2.3: Multidimensional EMD (Q1 2029)

**Target:** Q1 2029  
**Exit Criteria:** 2D EMD on 512×512 image in < 5s; 3D on 128³ in < 30s  
**Total Tasks:** ~20  

### Feature F-2.3.1: 2D Image Decomposition

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-308 | Design 2D EMD algorithm (separable vs non-separable) | 🔲 | architect | — | 4h | Benchmark both approaches |
| T-309 | Implement separable 2D EMD | 🔲 | code | — | 6h | Row-wise then column-wise |
| T-310 | Optimize boundary handling for images (periodic/symmetric) | 🔲 | code | — | 4h | Edge artifacts critical |
| T-311 | Write tests on synthetic + real medical images | 🔲 | test | — | 4h | CT, MRI validation |
| T-312 | Benchmark: latency, memory on various resolutions | 🔲 | perf | — | 2h | Regression testing |
| T-313 | Document + example: CT scan feature extraction | 🔲 | docs | — | 3h | Medical imaging use case |

**Feature Status:** 🔲 Not Started | **Completion:** 0/6

---

### Feature F-2.3.2: 3D Volumetric Decomposition

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-314 | Design 3D EMD algorithm | 🔲 | architect | — | 4h | Streaming-friendly design |
| T-315 | Implement 3D extrema detection | 🔲 | code | — | 6h | Voxel-wise extrema |
| T-316 | Implement GPU CUDA kernel for 3D (optional, v2.1 dependency) | 🔲 | code | — | 8h | Significant speedup |
| T-317 | Write tests on synthetic fMRI volumes | 🔲 | test | — | 4h | Voxel-wise validation |
| T-318 | Memory profiling: peak RSS on large volumes | 🔲 | perf | — | 2h | Identify bottlenecks |

**Feature Status:** 🔲 Not Started | **Completion:** 0/5

---

## Release v2.4: Machine Learning Integration (Q3 2029)

**Target:** Q3 2029  
**Exit Criteria:** Differentiable EMD gradient-correct; learnable boundaries improve accuracy  
**Total Tasks:** ~25  

### Feature F-2.4.1: Differentiable EMD

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-319 | Design implicit differentiation strategy (or unrolled loop) | 🔲 | architect | — | 4h | Numerical stability critical |
| T-320 | Implement forward pass in PyTorch | 🔲 | code | — | 4h | Signal → IMFs |
| T-321 | Implement backward pass (custom autograd.Function) | 🔲 | code | — | 6h | Sifting loop gradient |
| T-322 | Implement TensorFlow/Keras layer | 🔲 | code | — | 6h | tf.keras.layers.Layer |
| T-323 | Write numerical gradient tests (finite difference) | 🔲 | test | — | 4h | Error < 1e-4 |
| T-324 | Test gradient stability: NaN/Inf detection | 🔲 | test | — | 2h | Gradient clipping strategy |
| T-325 | Example: signal classification network | 🔲 | docs | — | 4h | End-to-end training |

**Feature Status:** 🔲 Not Started | **Completion:** 0/7

---

### Feature F-2.4.2: Learnable Boundary Conditions

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-326 | Design learnable boundary module (neural network) | 🔲 | architect | — | 3h | Task-specific optimization |
| T-327 | Implement in PyTorch | 🔲 | code | — | 5h | Backpropagable |
| T-328 | End-to-end training tests | 🔲 | test | — | 3h | Convergence validation |
| T-329 | Measure accuracy improvement on classification task | 🔲 | perf | — | 3h | Benchmark vs fixed boundaries |

**Feature Status:** 🔲 Not Started | **Completion:** 0/4

---

## Release v2.5: Advanced Post-Processing (Q1 2030)

**Target:** Q1 2030  
**Exit Criteria:** Entropy metrics implemented; mode mixing visualization working  
**Total Tasks:** ~15  

### Feature F-2.5.1: Time-Frequency Entropy Metrics

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-330 | Implement spectral entropy | 🔲 | code | — | 2h | Efficient computation |
| T-331 | Implement permutation entropy | 🔲 | code | — | 2h | Ordinal patterns |
| T-332 | Implement sample entropy | 🔲 | code | — | 2h | Self-similarity measure |
| T-333 | Write tests comparing against reference implementations | 🔲 | test | — | 2h | Validation |
| T-334 | Benchmark: entropy computation latency | 🔲 | perf | — | 2h | < 100ms for 100k samples |
| T-335 | Document + example visualization | 🔲 | docs | — | 2h | Matplotlib/Plotly templates |

**Feature Status:** 🔲 Not Started | **Completion:** 0/6

---

### Feature F-2.5.2: Mode Mixing Detection & Visualization

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-336 | Implement mode mixing metric (IMF pair overlap) | 🔲 | code | — | 3h | Frequency domain analysis |
| T-337 | Generate heatmap data structure | 🔲 | code | — | 2h | Matrix of overlap scores |
| T-338 | Write interpretation guidance (warn if overlap > 0.3) | 🔲 | docs | — | 2h | Reliability assessment |
| T-339 | Example visualization script | 🔲 | docs | — | 2h | Matplotlib/Seaborn |

**Feature Status:** 🔲 Not Started | **Completion:** 0/4

---

## Release v2.6: Distributed Systems (Q3 2030)

**Target:** Q3 2030  
**Exit Criteria:** Kubernetes operator working; gRPC service < 100ms latency; Arrow I/O zero-copy  
**Total Tasks:** ~20  

### Feature F-2.6.1: Apache Arrow Integration

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-340 | Implement Arrow bindings (`Signal::from_arrow_array`) | 🔲 | code | — | 4h | Zero-copy binding |
| T-341 | Implement Parquet I/O | 🔲 | code | — | 3h | Read/write signals |
| T-342 | Performance test: copy-free validation | 🔲 | perf | — | 2h | Memory address verification |
| T-343 | Example: pandas ↔ Arrow ↔ Ferromode pipeline | 🔲 | docs | — | 2h | Seamless interop |

**Feature Status:** 🔲 Not Started | **Completion:** 0/4

---

### Feature F-2.6.2: Kubernetes Operator

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-344 | Define EMDJob CRD schema | 🔲 | architect | — | 2h | Trial count, timeout, etc. |
| T-345 | Implement operator (kopf or operator-rs) | 🔲 | code | — | 8h | Auto-scaling logic |
| T-346 | Create Helm chart + deployment manifests | 🔲 | devops | — | 4h | Production-ready |
| T-347 | E2E test on minikube cluster | 🔲 | test | — | 4h | Full workflow validation |

**Feature Status:** 🔲 Not Started | **Completion:** 0/4

---

### Feature F-2.6.3: gRPC Microservice

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-348 | Define Protobuf schema for signal/config/result | 🔲 | architect | — | 2h | Language-agnostic |
| T-349 | Implement gRPC server (tonic crate) | 🔲 | code | — | 6h | Server streaming support |
| T-350 | Write client examples (Python, JavaScript, Go) | 🔲 | code | — | 4h | Demonstrate interop |
| T-351 | Load testing: response time, throughput | 🔲 | perf | — | 3h | Target: < 500ms network |

**Feature Status:** 🔲 Not Started | **Completion:** 0/4

---

## Release v2.7: Validation & Benchmarking (Q1 2031)

**Target:** Q1 2031  
**Exit Criteria:** 1000+ signal library; cross-impl validation automated; benchmark suite CI-integrated  
**Total Tasks:** ~20  

### Feature F-2.7.1: Reference Signal Library

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-352 | Generate 500+ synthetic AM/FM signals with ground truth | 🔲 | ml | — | 8h | Known IMF properties |
| T-353 | Collect + curate 500+ real signals (geophysics, biomedicine, finance) | 🔲 | data | — | 12h | Licensed/public domain |
| T-354 | Create JSON metadata schema | 🔲 | dev | — | 3h | Signal properties, noise level |
| T-355 | Package as Rust crate + PyPI + CRAN | 🔲 | devops | — | 4h | Distribution |

**Feature Status:** 🔲 Not Started | **Completion:** 0/4

---

### Feature F-2.7.2: Cross-Implementation Validation

| Task ID | Description | Status | Mode | Owner | Effort | Notes |
|---------|-------------|--------|------|-------|--------|-------|
| T-356 | Write R emd wrapper + comparison script | 🔲 | code | — | 4h | Reference implementation |
| T-357 | Write PyEMD wrapper | 🔲 | code | — | 4h | Python reference |
| T-358 | Write MATLAB wrapper (MEX-compatible) | 🔲 | code | — | 4h | If available |
| T-359 | Integrate all comparisons into CI | 🔲 | devops | — | 4h | Auto-run on every PR |
| T-360 | Document discrepancies (intentional algorithm differences) | 🔲 | docs | — | 2h | Transparency |

**Feature Status:** 🔲 Not Started | **Completion:** 0/5

---

## Intermittency Integration Across v2.x

Intermittency handling is **woven throughout** the v2.x roadmap. Here's how:

### Direct Intermittency Features

| Feature | Release | Task IDs | Description |
|---------|---------|----------|-------------|
| **Online Intermittency Detection** | v2.0 | (Adaptive Buffering) | Real-time stationarity metric via spectral entropy; adaptive buffer sizing |
| **Adaptive Algorithm Selection** | v2.0 | (Streaming) | Switch between EMD/EEMD based on detected intermittency |
| **Advanced Noise-Assisted Methods** | v2.2–v2.4 | (Future integration) | LSTM boundary prediction; learnable boundary conditions for intermittent signals |
| **Intermittency-Specific Test Suite** | v2.7 | T-352–T-360 | Generate signals with known intermittency; validate all algorithms |

### Indirect Intermittency Benefits

| Feature | Release | Intermittency Impact |
|---------|---------|---------------------|
| **v2.0 Streaming** | Q3 2027 | Continuous intermittency monitoring; real-time adaptation |
| **v2.1 GPU** | Q1 2028 | Ensemble methods (EEMD/CEEMDAN) run faster; more trials possible |
| **v2.2 Boundary Prediction** | Q3 2028 | Neural models reduce end effects that masquerade as intermittency |
| **v2.3 2D/3D EMD** | Q1 2029 | Spatial decomposition reveals intermittent events in images/volumes |
| **v2.4 ML Integration** | Q3 2029 | Learn signal-adaptive stopping criteria for intermittent signals |
| **v2.5 Advanced Metrics** | Q1 2030 | Entropy/mode-mixing metrics quantify intermittency-related degradation |
| **v2.6 Distributed** | Q3 2030 | Scale ensemble trials; robust handling of heterogeneous signals |
| **v2.7 Validation** | Q1 2031 | Cross-algorithm benchmarking on intermittent signals |

---

## Design Changes Required for v2.x

### 1. **Core Architecture Changes**

#### A. Adapter Layer for Streaming/GPU/ML
**Current (v1.x):**
```
Bindings → Domain (EMD, EEMD, etc.) → Foundation (Signal, Sifting)
```

**New (v2.x):**
```
Bindings → Adapter Layer (Streaming, GPU, ML, Distributed) 
          → Domain (EMD, EEMD, MEMD, etc.)
          → Foundation (Signal, Sifting, Boundary, Hilbert)
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-001:** Define adapter trait hierarchy | 2h | 🔲 |
| **DES-002:** Refactor domain types for adapter compatibility | 4h | 🔲 |
| **DES-003:** Document adapter contract (pure delegation) | 2h | 🔲 |

#### B. Streaming State Management
**New Type:**
```rust
pub struct StreamingState {
    chunk_id: u64,
    sifting_history: Vec<SiftingIteration>,
    last_envelope: (Vec<f64>, Vec<f64>),  // upper, lower
    predictor_state: Box<dyn PredictorState>,
    buffer: RingBuffer<f64>,
}
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-004:** Design RingBuffer for fixed memory footprint | 3h | 🔲 |
| **DES-005:** Design PredictorState trait (generic over model type) | 2h | 🔲 |

#### C. GPU Memory Management
**New Abstraction:**
```rust
pub trait GpuExecutor: Send + Sync {
    fn decompose(&self, signal: &[f64], config: &EmdConfig) -> Result<Vec<Vec<f64>>>;
    fn device_name(&self) -> String;
}

pub struct CudaExecutor { /* CUDA context + device */ }
pub struct CpuExecutor { /* Fallback */ }
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-006:** Design GpuExecutor trait + CPU fallback | 2h | 🔲 |
| **DES-007:** Implement device selection (CUDA, ROCm fallback) | 3h | 🔲 |

#### D. Differentiable EMD Layer
**New Module:**
```rust
pub mod ml {
    pub struct DifferentiableEmd {
        config: EmdConfig,
        implicit_fn: Box<dyn Fn(&Signal) -> Result<ImfCollection>>,
    }
    
    impl DifferentiableEmd {
        pub fn forward(&self, signal: &Signal) -> Result<ImfCollection> { ... }
        pub fn vjp(&self, signal: &Signal, grad_output: &[Vec<f64>]) -> Result<Vec<f64>> { ... }
    }
}
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-008:** Design implicit differentiation strategy | 4h | 🔲 |
| **DES-009:** Document gradient flow through sifting | 2h | 🔲 |

### 2. **Configuration & API Changes**

#### A. Hierarchical Config
**v1.x (flat):**
```rust
pub struct EmdConfig {
    max_imfs: usize,
    boundary: BoundaryStrategy,
    stopping: StoppingCriterion,
}
```

**v2.x (hierarchical):**
```rust
pub struct EmdConfig { /* base */ }

pub struct StreamingConfig {
    base: EmdConfig,
    chunk_size: usize,
    predictor: Arc<dyn BoundaryPrediction>,
}

pub struct GpuConfig {
    base: EmdConfig,
    device_id: u32,
    mixed_precision: bool,
}
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-010:** Design config inheritance pattern | 2h | 🔲 |
| **DES-011:** Implement validation for each config subtype | 3h | 🔲 |

### 3. **Intermittency-Specific Design**

#### A. Intermittency Detection & Adaptation
**New Type:**
```rust
pub struct IntermittencyMetrics {
    spectral_entropy: f64,
    extrema_spacing_cv: f64,  // coefficient of variation
    stationarity_score: f64,  // 0–1, higher = more stationary
}

pub enum AdaptiveAlgorithm {
    EMD,         // stationary
    EEMD,        // mildly intermittent
    CEEMDAN,     // highly intermittent
}

impl AdaptiveAlgorithm {
    pub fn select_from_metrics(metrics: &IntermittencyMetrics) -> Self { ... }
}
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-012:** Design IntermittencyMetrics struct | 2h | 🔲 |
| **DES-013:** Implement adaptive algorithm selection policy | 3h | 🔲 |
| **DES-014:** Integrate with streaming mode | 2h | 🔲 |

#### B. Intermittency-Aware Boundary Prediction
**Integration with v2.2:**
- Neural boundary predictor trained on intermittent signals
- Boundary extension accounts for signal non-stationarity
- Task T-298: Training dataset includes high-intermittency signals

### 4. **Binding & FFI Changes**

#### A. Async Support for Python/JavaScript
**v2.x Python:**
```python
async def decompose_chunk_async(signal, config):
    return await streaming_decomposer.decompose_chunk(signal)
```

**v2.x JavaScript:**
```javascript
const imfs = await emddComposer.decomposeChunkAsync(chunk);
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-015:** Design async FFI protocol | 2h | 🔲 |
| **DES-016:** Implement PyO3 async wrappers | 3h | 🔲 |
| **DES-017:** Implement WASM Promise bindings | 3h | 🔲 |

#### B. GPU Array Bindings
**v2.x Python (CuPy):**
```python
gpu_result = ferromode_py.decompose_gpu(cupy_array)  # GPU → GPU
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-018:** Design zero-copy pointer protocol | 2h | 🔲 |
| **DES-019:** Implement CuPy PyO3 bindings | 4h | 🔲 |

### 5. **Testing & Validation Architecture**

#### A. Streaming vs Batch Equivalence Testing
**New Test Framework:**
```rust
#[test]
fn test_streaming_batch_equivalence() {
    // Run same signal through batch and streaming
    // Compare IMFs chunk-by-chunk
    // Assert error < 1e-10
}
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-020:** Design streaming/batch comparison harness | 2h | 🔲 |
| **DES-021:** Implement automated error reporting | 2h | 🔲 |

#### B. Cross-Implementation Validation Infrastructure
**New CI Job:**
```yaml
test-cross-impl:
  - Run Ferromode on reference signal library
  - Compare against R emd, PyEMD, MATLAB
  - Report concordance per algorithm
```

**Design Tasks:**
| Task | Effort | Status |
|------|--------|--------|
| **DES-022:** Design test harness for external implementations | 3h | 🔲 |
| **DES-023:** Integrate into CI/CD pipeline | 3h | 🔲 |

---

## Design Change Summary

| Design Change | Effort | Priority | v2.0 | v2.1 | v2.2 | v2.3 | v2.4 | v2.5 | v2.6 | v2.7 |
|--------------|--------|----------|------|------|------|------|------|------|------|------|
| **Adapter Layer** | 8h | High | ✓ | — | — | — | — | — | — | — |
| **Streaming State** | 5h | High | ✓ | — | — | — | — | — | — | — |
| **GPU Abstraction** | 5h | High | — | ✓ | — | — | — | — | — | — |
| **Differentiable Layer** | 6h | High | — | — | — | — | ✓ | — | — | — |
| **Config Hierarchy** | 5h | Medium | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Intermittency Adaptation** | 7h | Medium | ✓ | — | ✓ | — | ✓ | ✓ | — | ✓ |
| **Async FFI** | 8h | Medium | ✓ | — | — | — | — | — | — | — |
| **Zero-Copy Arrays** | 6h | Medium | — | — | — | — | — | — | — | ✓ |

**Total Design Effort:** ~50 hours across all v2.x releases.

---

## Execution Readiness Checklist

Before starting v2.0 work:

- [ ] All design documents reviewed (FUTURE_PRD.md, FUTURE_IMPLEMENTATION_MAP.md)
- [ ] Design changes (DES-001 through DES-023) approved by team
- [ ] Adapter layer architecture agreed upon
- [ ] Intermittency integration strategy signed off
- [ ] Streaming state management finalized
- [ ] CI/CD pipeline extended for v2.x tests (streaming, GPU, distributed)
- [ ] Resource allocation confirmed (team size, GPU hardware, etc.)
- [ ] Risk mitigation plans in place (GPU instability, streaming edge cases, etc.)

---

*Last updated: April 4, 2026*  
*Maintained by: Core Engineering Team*
