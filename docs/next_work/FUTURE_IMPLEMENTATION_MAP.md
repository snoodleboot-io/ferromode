# Implementation Map — Future Releases (v2.x)
## Ferromode v2.0–v2.7: Engineering Strategy & Execution Framework

**Created:** 2026-04-04  
**Status:** Active Planning Document  
**Linked Docs:** FUTURE_PRD.md · FUTURE_FEATURES_STORIES_TASKS.md · EXECUTION_CHECKLIST.md  

---

## 1. Engineering Practices — Extending v1.x Standards

All future work inherits the six non-negotiable practices from v1.x and extends them with release-specific considerations.

### 1.1 TDD (Test-Driven Development)

**Rule:** Tests written BEFORE implementation. Every algorithm and feature has a failing test first.

#### 1.1.1 Streaming Mode (v2.0) TDD Specifics

```rust
// STEP 1: RED — Write failing test
#[test]
fn test_online_emd_maintains_envelope_continuity() {
    // Given: signal with known properties
    // When: decomposed in two chunks with boundary prediction
    // Then: envelope at chunk boundary is smooth (curvature continuous)
    
    let signal = create_test_signal(1000);
    let (chunk1, chunk2) = signal.split_at(500);
    
    // Decompose separately → get envelopes at boundary
    let result1 = online_emd::decompose_chunk(chunk1, boundary_predictor);
    let result2 = online_emd::decompose_chunk(chunk2, boundary_predictor);
    
    // Validate: envelope smoothness at junction
    let junction_smoothness = compute_curvature_continuity(
        result1.upper_envelope.last_point(),
        result2.upper_envelope.first_point(),
    );
    assert!(junction_smoothness > 0.95); // 95% continuity
}

// STEP 2: GREEN — Minimum implementation
// In streaming.rs
pub fn decompose_chunk(
    chunk: &Signal,
    predictor: &BoundaryPredictor,
) -> ChunkDecompositionResult {
    // Predict signal continuation
    let extension = predictor.predict(chunk);
    
    // Run standard EMD on extended signal
    let extended = chunk.concatenate(extension);
    let result = emd::decompose(&extended, config);
    
    // Trim IMFs to original chunk size
    ChunkDecompositionResult {
        imfs: trim_to_original_length(result.imfs, chunk.len()),
        state: StreamingState { last_extension: extension },
    }
}

// STEP 3: REFACTOR — Clean up, optimize, handle edge cases
// Improved with:
// - Stateful sifting (carry over extrema positions)
// - Incremental IMF extraction
// - Memory pooling for buffers
```

#### 1.1.2 Coverage Targets for v2.x Features

| Feature | Line Coverage | Branch Coverage | Function Coverage |
|---------|--------------|-----------------|------------------|
| Streaming EMD | ≥ 90% | ≥ 80% | 100% |
| GPU kernels (Rust wrapper) | ≥ 85% | ≥ 75% | 100% |
| Neural boundary prediction | ≥ 95% | ≥ 85% | 100% |
| Differentiable EMD | ≥ 90% | ≥ 80% | 100% |
| 2D/3D EMD | ≥ 90% | ≥ 80% | 100% |
| Distributed coordination | ≥ 95% | ≥ 85% | 100% |

### 1.2 ATDD (Acceptance Test-Driven Development)

**Rule:** Acceptance criteria written as executable tests BEFORE task starts.

#### 1.2.1 Story Template for v2.x Features

```gherkin
# Feature: Streaming EMD for real-time signals
# Scenario: Online decomposition with continuous envelope

Given a 10-second signal arriving in 1-second chunks
When each chunk is decomposed with online EMD
And boundary prediction is applied
Then the reconstructed signal equals the batch-mode result
And the envelope at chunk boundaries is smooth
And latency per chunk is less than 10 ms
And memory does not accumulate (ring buffer reuse)
```

#### 1.2.2 Acceptance Test Checklist (per story)

Before implementation starts, these tests must be written:

- [ ] **Functional acceptance test** — Happy path behavior
- [ ] **Edge case tests** — Boundary conditions, empty input, extreme values
- [ ] **Performance acceptance test** — Meets latency/memory targets
- [ ] **Numerical validation test** — Matches batch mode or reference implementation
- [ ] **Integration test** — Works with language bindings and tooling

### 1.3 DDD (Domain-Driven Design)

**Rule:** Each major feature is a distinct bounded context with explicit types and invariants.

#### 1.3.1 v2.x Bounded Contexts

| Context | Module | Responsibility |
|---------|--------|-----------------|
| **Streaming Domain** | `streaming/` | `ChunkDecomposer`, `StreamingState`, `BoundaryPredictor` — chunk-based processing |
| **GPU Domain** | `gpu/` | CUDA kernels, GPU memory management, device-agnostic trait `GpuExecutor` |
| **Boundary Prediction Domain** | `boundary_prediction/` | Neural models, AR models, Wavelet-based extension — predicting signal continuation |
| **2D/3D Domain** | `multidimensional/` | `Image2D`, `Volume3D`, `SpatialExtremum` — spatial decomposition |
| **ML Integration Domain** | `ml/` | `DifferentiableDecomposition`, `GradientFlow`, `ImpicitDifferentiable` — learnable components |
| **Distributed Domain** | `distributed/` | `DistributedEnsemble`, `TaskCoordinator`, `ResultAggregator` — cloud/HPC scaling |
| **Analysis Domain** | `analysis/` | Time-frequency entropy, mode mixing metrics, change detection — post-processing |

#### 1.3.2 Domain Type Example: Streaming

```rust
/// Streaming EMD decomposer — maintains state across chunks
pub struct StreamingDecomposer {
    config: EmdConfig,
    state: StreamingState,
    boundary_predictor: Arc<dyn BoundaryPrediction>,
}

/// Invariants:
/// - config.max_imfs > 0
/// - state.buffer_size >= config.expected_imf_count
/// - boundary_predictor is deterministic (same input → same output)
impl StreamingDecomposer {
    pub fn new(config: EmdConfig, predictor: Arc<dyn BoundaryPrediction>) -> Result<Self> {
        config.validate()?; // Ensures invariants
        Ok(Self { config, state: StreamingState::default(), boundary_predictor: predictor })
    }
    
    /// Decompose a single chunk; may return partial IMFs
    /// Postcondition: state.last_chunk_id incremented
    pub fn decompose_chunk(&mut self, chunk: Signal) -> Result<ChunkResult> {
        chunk.validate()?; // Ensures input invariants
        
        // Core algorithm
        let extended = self.boundary_predictor.predict(&chunk);
        let result = emd::decompose(&extended, &self.config)?;
        
        // Update state
        self.state.last_chunk_id += 1;
        self.state.last_envelope = result.envelope.clone();
        
        Ok(ChunkResult { imfs: trim(result.imfs, chunk.len()), is_final: false })
    }
    
    /// Finalize decomposition (may apply final sifting iterations)
    pub fn finalize(self) -> Result<FinalResult> {
        // Aggregate partial IMFs; apply final refinement
        Ok(FinalResult { imfs: self.state.aggregated_imfs })
    }
}
```

### 1.4 Clean Code Standards for v2.x

All the rules from v1.x apply, plus release-specific guidelines:

#### 1.4.1 Streaming Code
- **Naming:** `decompose_chunk()` not `proc()` or `onl()` — clarity over brevity
- **No blocking calls:** Streaming code must not call blocking I/O or allocate in hot loops
- **Explicit state management:** StreamingState explicitly tracks what persists between chunks

#### 1.4.2 GPU Code
- **CUDA wrapper safety:** Every CUDA call wrapped in `Result<T, GpuError>`; no panics
- **Memory safety:** All GPU allocations explicitly deallocated; RAII guards preferred
- **Device agnostic:** Trait `GpuExecutor` abstracts CUDA/ROCm/WebGPU; no hardcoded device code

#### 1.4.3 Differentiable Code
- **Gradient computation:** Explicitly documented (forward, adjoint, or implicit differentiation)
- **Numerical stability:** Document gradient clipping strategy; test for NaN/Inf propagation
- **Composable:** Differentiable layers can be nested without breaking autodiff

### 1.5 Clean Architecture for v2.x

**Rule:** Dependencies point inward; domain knows nothing about streaming, GPU, or ML frameworks.

#### 1.5.1 Dependency Hierarchy

```
┌─────────────────────────────────────┐
│    Binding Layer                    │
│  (Python, R, JavaScript, gRPC)      │
├─────────────────────────────────────┤
│    Adapter Layer                    │
│  (Streaming, GPU, Distributed)      │
├─────────────────────────────────────┤
│    Domain Layer                     │
│  (EMD, EEMD, MEMD algorithms)      │
├─────────────────────────────────────┤
│    Foundation Layer                 │
│  (Types, Signal, Extrema, Sifting)  │
└─────────────────────────────────────┘
```

- **Foundation** is pure and testable in isolation
- **Domain** has no dependencies on Adapter or Binding layers
- **Adapter** depends on Domain; does translation/orchestration
- **Binding** is thin marshalling only

#### 1.5.2 Example: Streaming Adapter

```rust
// In adapter layer: streaming.rs
// This layer adapts the core domain to streaming context
// Domain stays unchanged

pub struct StreamingAdapter {
    core_decomposer: Arc<EmdDecomposer>,  // Domain type
    chunk_buffer: VecDeque<Signal>,        // Adapter concern
    predictor: Box<dyn BoundaryPrediction>,  // Adapter concern
}

impl StreamingAdapter {
    pub fn process_chunk(&mut self, chunk: Signal) -> Result<PartialResult> {
        // Adapter logic: predict boundaries, manage buffers
        let extended = self.predictor.predict(&chunk);
        
        // Delegate to domain
        let domain_result = self.core_decomposer.decompose(&extended)?;
        
        // Adapt result: trim to original, convert format
        self.convert_to_streaming_result(domain_result)
    }
}

// In domain layer: decomposition.rs (UNCHANGED from v1.x)
pub fn decompose(signal: &Signal, config: &EmdConfig) -> Result<DecompositionResult> {
    // Pure algorithm; no knowledge of streaming
    // ...
}
```

### 1.6 Numerical Validation & Stability for v2.x

#### 1.6.1 Cross-Mode Validation

Every new feature must be validated against existing modes:

| Feature | Validation Strategy | Acceptance Criteria |
|---------|-------------------|-------------------|
| **Streaming (v2.0)** | Run streaming on historical batch data; compare chunk-by-chunk IMFs | Relative error < 1e-10 |
| **GPU (v2.1)** | CPU vs GPU comparison on same signal/config | Relative error < 1e-9 (FP32) |
| **Boundary Prediction (v2.2)** | Compare neural vs. non-neural boundary strategies | HHT quality score within 5% |
| **2D EMD (v2.3)** | Compare separable vs. non-separable on synthetic images | Signal reconstruction error < 1e-11 |
| **Differentiable (v2.4)** | Numerical gradient check: |∂f/∂x - ε-grad| < 1e-4 | All tests pass |
| **Distributed (v2.6)** | Single-machine vs. distributed on same data | Identical results (exact floating-point) |

#### 1.6.2 Stability Testing Patterns

```rust
#[test]
fn test_streaming_vs_batch_numerical_equivalence() {
    let signal = Signal::synthetic_am_fm(10000);
    
    // Batch mode
    let batch_result = emd::decompose(&signal, &config)?;
    
    // Streaming mode: process in 500-sample chunks
    let mut streaming_result = Vec::new();
    for chunk in signal.chunks(500) {
        let chunk_result = streaming_decomposer.decompose_chunk(chunk)?;
        streaming_result.extend(chunk_result.imfs);
    }
    
    // Validate: max relative error across all IMFs
    for (batch_imf, stream_imf) in batch_result.imfs.iter().zip(streaming_result.iter()) {
        let rel_error = (batch_imf - stream_imf).norm() / batch_imf.norm();
        assert!(rel_error < 1e-10, "Streaming mode diverged: error = {}", rel_error);
    }
}
```

---

## 2. Release-Specific Engineering Considerations

### 2.1 v2.0 (Streaming) Specific Practices

#### 2.1.1 Asynchronous Testing

Streaming features must be tested with async-compatible test frameworks:

```rust
#[tokio::test]
async fn test_streaming_with_async_input() {
    let signal_stream = create_async_signal_stream(); // Simulated sensor data
    let mut decomposer = StreamingDecomposer::new(config, predictor)?;
    
    let mut imfs = Vec::new();
    while let Some(chunk) = signal_stream.next().await {
        let result = decomposer.decompose_chunk(chunk)?;
        imfs.push(result);
    }
    
    assert_eq!(imfs.len(), expected_chunk_count);
}
```

#### 2.1.2 Stateful Testing

Streaming state is mutable; tests must verify state transitions:

```rust
#[test]
fn test_streaming_state_accumulation() {
    let mut decomposer = StreamingDecomposer::new(...)?;
    
    // Chunk 1: boundary prediction trained on first chunk
    decomposer.decompose_chunk(chunk1)?;
    assert_eq!(decomposer.state.last_chunk_id, 1);
    
    // Chunk 2: uses predictor trained on chunk 1
    decomposer.decompose_chunk(chunk2)?;
    assert_eq!(decomposer.state.last_chunk_id, 2);
    
    // Validate envelope continuity between chunks
    let continuity = measure_envelope_smoothness(&decomposer.state);
    assert!(continuity > 0.95);
}
```

#### 2.1.3 Memory Leak Testing

Streaming code runs continuously; memory profiling is mandatory:

```bash
# In CI: Profile memory usage over 10k chunks
valgrind --leak-check=full --show-leak-kinds=all cargo test streaming_memory_leak

# In benchmarks: Track peak RSS and sustained RSS
criterion::black_box(
    (0..10_000).for_each(|_| {
        decomposer.decompose_chunk(&chunk)?;
    })
)?;
// Assert: peak_rss < 100 MB, sustained_rss < 50 MB
```

### 2.2 v2.1 (GPU) Specific Practices

#### 2.2.1 GPU-Host Parity Testing

Every GPU kernel must have a reference CPU implementation:

```rust
#[test]
fn test_gpu_extrema_detection_matches_cpu() {
    let signal = Signal::synthetic(...);
    
    // CPU reference
    let cpu_extrema = cpu::find_extrema(&signal);
    
    // GPU implementation
    let gpu_extrema = gpu::find_extrema(&signal)?;
    
    // Compare
    assert_eq!(cpu_extrema.len(), gpu_extrema.len());
    for (cpu_ext, gpu_ext) in cpu_extrema.iter().zip(gpu_extrema.iter()) {
        assert!((cpu_ext - gpu_ext).abs() < 1e-9);
    }
}
```

#### 2.2.2 Device Selection Testing

```rust
#[test]
fn test_gpu_device_fallback() {
    // Try GPU; fall back to CPU if unavailable
    let executor = GpuExecutor::new_with_fallback()?;
    
    // Should work on any hardware
    let result = executor.decompose(&signal, &config)?;
    assert!(!result.imfs.is_empty());
}
```

#### 2.2.3 Mixed Precision Stability

```rust
#[test]
fn test_mixed_precision_fp16_fp32_equivalence() {
    let signal = Signal::synthetic(...);
    
    // FP32 baseline
    let result_fp32 = gpu::decompose_fp32(&signal)?;
    
    // FP16 compute, FP32 accumulation
    let result_fp16 = gpu::decompose_mixed_fp16(&signal)?;
    
    // Validate: FP16 is within expected numerical error
    let rel_error = (result_fp32.imf[0] - result_fp16.imf[0]).norm() / result_fp32.imf[0].norm();
    assert!(rel_error < 1e-5); // Looser bound for FP16
}
```

### 2.3 v2.4 (Differentiable) Specific Practices

#### 2.3.1 Gradient Correctness Testing

```rust
#[test]
fn test_emd_gradient_correctness_finite_difference() {
    let signal = Signal::synthetic(...);
    let emd_layer = DifferentiableEmd::new(config)?;
    
    // Numerical gradient (ε = 1e-4)
    let eps = 1e-4;
    let mut numerical_grad = vec![0.0; signal.len()];
    for i in 0..signal.len() {
        let mut signal_plus = signal.clone();
        signal_plus[i] += eps;
        let f_plus = emd_layer.forward(&signal_plus)?;
        
        let mut signal_minus = signal.clone();
        signal_minus[i] -= eps;
        let f_minus = emd_layer.forward(&signal_minus)?;
        
        numerical_grad[i] = (f_plus - f_minus) / (2.0 * eps);
    }
    
    // Autodiff gradient
    let autodiff_grad = emd_layer.backward(&signal)?;
    
    // Compare
    let max_error = numerical_grad.iter().zip(autodiff_grad.iter())
        .map(|(n, a)| (n - a).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    
    assert!(max_error < 1e-4, "Gradient mismatch: {}", max_error);
}
```

#### 2.3.2 Integration with DL Frameworks

```rust
// PyTorch binding test
#[test]
fn test_pytorch_emd_layer_integration() {
    // Create EMD layer in PyTorch
    let module = "
        import torch
        from ferromode_torch import EMDLayer
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emd = EMDLayer(max_imfs=8)
                self.fc = torch.nn.Linear(8, 1)
            
            def forward(self, x):
                imfs = self.emd(x)
                out = self.fc(imfs)
                return out
        
        model = MyModel()
        x = torch.randn(32, 1000)  # Batch of signals
        loss = model(x).mean()
        loss.backward()  # Gradient flow through EMD
    ";
    
    // Verify: gradients computed successfully
    assert!(backward_successful);
}
```

### 2.4 v2.6 (Distributed) Specific Practices

#### 2.4.1 Deterministic Testing

Distributed code is non-deterministic by nature; tests must be idempotent:

```rust
#[test]
fn test_distributed_ensemble_idempotency() {
    let signal = Signal::synthetic(...);
    
    // Run 1
    let result1 = distributed::ceemdan(&signal, num_trials=100, num_workers=10)?;
    
    // Run 2 (same config)
    let result2 = distributed::ceemdan(&signal, num_trials=100, num_workers=10)?;
    
    // Same random seed → same results
    // (Or verify statistical equivalence: same mean/variance of ensemble)
    assert_eq!(result1.ensemble_mean, result2.ensemble_mean);
}
```

#### 2.4.2 Scalability Testing

```bash
# Benchmark: measure efficiency as we scale workers
for WORKERS in 2 4 8 16 32 64; do
    cargo bench distributed --bench-group ensemble -- --num-workers $WORKERS
done

# Report: speedup should be >= 0.8 * num_workers (80% efficiency)
```

---

## 3. Cross-Release Consistency

### 3.1 API Stability Contract

Each v2.x release maintains backward compatibility with v1.x:

```rust
// v1.x API (stable forever)
pub fn decompose(signal: &Signal, config: &EmdConfig) -> Result<DecompositionResult> { ... }

// v2.0 additions (new API, no breaking changes)
pub fn decompose_streaming(
    chunk: &Signal,
    state: &mut StreamingState,
    predictor: &dyn BoundaryPrediction,
) -> Result<ChunkResult> { ... }

// v2.1 additions (new API)
pub fn decompose_gpu(signal: &Signal, config: &EmdConfig, executor: &dyn GpuExecutor) 
    -> Result<DecompositionResult> { ... }

// All old code continues to compile and work
```

### 3.2 Configuration Inheritance

New config types inherit from base:

```rust
// v1.x
#[derive(Clone, Debug)]
pub struct EmdConfig {
    pub max_imfs: usize,
    pub boundary_strategy: BoundaryStrategy,
    pub stopping_criterion: StoppingCriterion,
}

// v2.0 extends (no breaking changes to EmdConfig)
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    pub base: EmdConfig,  // Inherit all v1.x settings
    pub chunk_size: usize,  // New v2.0 setting
    pub predictor_model: String,  // New v2.0 setting
}

// v2.1 extends
#[derive(Clone, Debug)]
pub struct GpuConfig {
    pub base: EmdConfig,  // Inherit all v1.x settings
    pub device_id: u32,  // New v2.1 setting
    pub mixed_precision: bool,  // New v2.1 setting
}
```

### 3.3 Documentation Synchronization

Every v2.x feature must have parallel documentation:

| Artifact | v1.x | v2.0 | v2.1 | v2.2 | v2.3 | v2.4 | v2.5 | v2.6 |
|----------|------|------|------|------|------|------|------|------|
| README.md | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| API docs | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Examples | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Tutorials | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Benchmark suite | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 4. Tooling & Infrastructure for v2.x

### 4.1 Expanded CI/CD Pipeline

```yaml
# .github/workflows/v2x-testing.yml

name: v2.x Extended Testing

on: [push, pull_request]

jobs:
  streaming-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Test streaming mode
        run: cargo test --features streaming
      
      - name: Memory profiling
        run: valgrind --leak-check=full cargo test streaming

  gpu-tests:
    runs-on: [ubuntu-latest, self-hosted-gpu]
    steps:
      - name: Test CUDA kernels
        run: cargo test --features gpu-cuda
      - name: GPU device fallback
        run: cargo test gpu_device_selection

  distributed-tests:
    runs-on: ubuntu-latest
    services:
      - redis
      - grpc-service
    steps:
      - name: Test distributed ensemble
        run: cargo test --features distributed

  cross-validation:
    runs-on: ubuntu-latest
    steps:
      - name: Compare against reference implementations
        run: |
          cargo test cross_impl_validation
          # Runs Ferromode vs R emd vs PyEMD vs MATLAB
```

### 4.2 Release Checklist Template

Every v2.x release must verify:

```markdown
# Release v2.X Checklist

## Code Quality
- [ ] All tests passing (streaming, GPU, distributed, etc.)
- [ ] Coverage >= 90% (core), >= 85% (adapters)
- [ ] Clippy clean; rustfmt applied
- [ ] No security advisories

## Numerical Validation
- [ ] Cross-implementation tests passing (vs reference)
- [ ] Floating-point equivalence tests within tolerance
- [ ] Gradient tests for differentiable features
- [ ] Memory profiling: peak RSS, sustained RSS within targets

## Documentation
- [ ] CHANGELOG updated
- [ ] API docs generated and reviewed
- [ ] Examples work (tested)
- [ ] Tutorials updated

## Performance
- [ ] Benchmarks passing
- [ ] No regressions vs previous release
- [ ] Streaming latency < target
- [ ] GPU speedup >= target

## Release
- [ ] Version bumped (SemVer)
- [ ] Git tag created
- [ ] crates.io published
- [ ] PyPI, NPM, etc. updated
- [ ] GitHub release notes published
```

---

## 5. Phased Rollout Strategy

Each v2.x release follows a 3-phase rollout:

### Phase 1: Alpha (4 weeks)
- **Audience:** Internal team + volunteer beta testers
- **Testing:** Intensive cross-validation against reference implementations
- **Feedback:** Focus on API design; gather user feedback before stabilization
- **Artifact:** `--pre` version on crates.io (v2.0.0-alpha.1)

### Phase 2: Beta (4 weeks)
- **Audience:** Expanded community; production-like testing
- **Testing:** Real-world scenarios; edge cases discovered in alpha
- **Feedback:** Performance optimization; API refinement
- **Artifact:** `--pre` version on crates.io (v2.0.0-beta.1)

### Phase 3: Stable Release (ongoing)
- **Audience:** General availability
- **Testing:** Regression testing; monitoring in production
- **Feedback:** Bug reports; feature requests for next version
- **Artifact:** Stable version (v2.0.0)

---

## 6. Failure Modes & Recovery

### 6.1 Known Risks & Mitigation

| Risk | Detection | Recovery |
|------|-----------|----------|
| Streaming mode diverges from batch | Continuous validation tests fail | Revert predictor model; fall back to symmetric extension |
| GPU memory exhaustion | `cuda_malloc()` returns OOM | Implement spilling to host memory; emit warning |
| Differentiable gradient explosion | NaN detected in backprop | Clip gradients; use mixed precision; reduce max_imfs |
| Distributed ensemble inconsistency | Cross-worker result variance excessive | Increase trial count; validate RNG seeding |

---

## 7. Success Criteria for v2.x Engineering

By the end of v2.x (v2.7 stable, Q1 2031):

- [ ] All six engineering practices enforced in CI
- [ ] Zero production numerical bugs reported
- [ ] 90%+ test coverage across all features
- [ ] Backward compatibility with v1.x: 100%
- [ ] Cross-language binding parity: 100%
- [ ] Performance targets met: 100%
- [ ] Documentation completeness: 100%

---

## 8. Glossary for v2.x

| Term | Definition |
|------|-----------|
| **Streaming State** | Mutable context maintained between chunk decompositions; includes sifting history, envelope estimates, and learned boundaries |
| **Boundary Predictor** | Function/model that extends signal beyond endpoints; used by streaming and boundary-effect-mitigation features |
| **GPU Executor** | Trait abstracting GPU device; enables device-agnostic code (CUDA, ROCm, WebGPU) |
| **Differentiable** | Supports automatic differentiation; gradients can flow through component |
| **Ensemble Aggregation** | Combining multiple decomposition results (e.g., mean of EEMD modes) |
| **Numerical Equivalence** | Results are identical within floating-point rounding error (1e-10 relative) |
| **Cross-Implementation Validation** | Comparing Ferromode against R emd, PyEMD, MATLAB reference to verify correctness |

---

## References

- **Streaming**: Rilling & Flandrin (2008). "One or two frequencies? The empirical mode decomposition answers." *IEEE Transactions on Signal Processing*, 56(1), 85–95.
- **GPU**: NVIDIA CUDA Toolkit Documentation (12.0+)
- **Differentiability**: Maclaurin, D., et al. (2015). "Autograd: Automatic differentiation of NumPy programs."
- **Distributed**: Zaharia, M., et al. "Apache Spark: a unified engine for big data processing."

---

*Last updated: April 4, 2026*  
*Maintained by: Core Engineering Team*
