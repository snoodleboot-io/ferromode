# Future v2.x Planning: Summary & Answers to Key Questions

**Created:** 2026-04-04  
**Status:** Planning Document  

---

## Question 1: Did We Include Intermittency?

### ✅ YES — Intermittency is Integrated Throughout v2.x

**The short answer:** Intermittency handling is **not a separate v2.x feature** but rather **woven into every major release** because it's fundamental to signal processing. Here's how:

#### What Was Already Done in v1.x

Your INTERMITTENCY_ROADMAP.md clearly states that v1.x **already handles intermittency** through:

1. **Ensemble Methods (M3)** — EEMD, CEEMD, CEEMDAN, ICEEMDAN
   - These algorithms are specifically designed to combat mode mixing caused by intermittency
   - Noise-assisted trials improve decomposition of non-stationary signals

2. **Boundary Condition Strategies (M1)** — 8 boundary methods
   - Proper boundaries reduce end effects that can masquerade as intermittency
   - AR model extension is adaptive to signal properties

3. **Intermittency Detection (T-064)** — Pre-check based on extrema spacing variability
   - Measures signal non-stationarity via coefficient of variation

### What v2.x Adds for Intermittency

#### **v2.0: Streaming (Q3 2027)**
- **Online Intermittency Detection** (Feature F-2.0.4: Adaptive Buffering)
  - Task T-266: Spectral entropy metric for stationarity detection
  - Real-time buffer sizing: large for stationary segments, small for transients
  - Enables continuous monitoring as data arrives

- **Adaptive Algorithm Selection** (v2.0 Streaming Core)
  - Automatically switch between EMD/EEMD/CEEMDAN based on detected stationarity
  - Tasks: Intermittency metrics inform model choice

#### **v2.1: GPU (Q1 2028)**
- **Scaling Ensemble Methods** for Intermittent Signals
  - Ensemble methods (EEMD/CEEMDAN) run faster on GPU
  - More trials possible → better noise averaging → better handling of intermittency
  - Task T-279–T-287: GPU kernels for parallel trials

#### **v2.2: Boundary Prediction (Q3 2028)**
- **Neural Boundary Prediction for Intermittent Signals**
  - Task T-298: Training dataset explicitly includes high-intermittency synthetic signals
  - LSTM learns to predict boundaries that minimize end effects
  - Reduces artifacts that confound intermittency analysis

#### **v2.3: 2D/3D EMD (Q1 2029)**
- **Spatial Intermittency Detection**
  - Decompose images/volumes to reveal intermittent spatial events
  - Useful for: medical imaging (transient lesions), geophysics (localized anomalies)
  - Tasks T-308–T-318: 2D/3D implementations

#### **v2.4: ML Integration (Q3 2029)**
- **Learnable Signal-Adaptive Stopping Criteria**
  - Neural networks learn optimal sifting stopping rules per signal type
  - Task T-326–T-329: Learnable boundaries optimized for intermittent signal classification
  - Differentiable EMD enables end-to-end training

#### **v2.5: Advanced Post-Processing (Q1 2030)**
- **Intermittency-Related Metrics**
  - Task T-330–T-339: Entropy measures, mode mixing detection
  - Spectral entropy flagged high for intermittent signals
  - Mode mixing heatmap reveals where intermittency causes decomposition issues

#### **v2.6: Distributed (Q3 2030)**
- **Scaling Robustness Across Heterogeneous Signals**
  - Dask/Ray parallelism on cloud clusters
  - Handle mixed-stationarity ensembles efficiently
  - Fault tolerance for long-running ensemble trials on intermittent signals

#### **v2.7: Validation (Q1 2031)**
- **Intermittency-Specific Test Suite**
  - Task T-352–T-360: Reference signal library includes signals with **known intermittency levels**
  - Cross-implementation validation on intermittent signals
  - Benchmark EEMD/CEEMDAN/ICEEMDAN/AN-EEMD on intermittent signals

### How Intermittency Flows Through v2.x

```
v2.0 (Streaming): Real-time stationarity monitoring via entropy
    ↓
v2.2 (Boundary): Neural models learn intermittency-aware extensions
    ↓
v2.4 (ML): Learnable stopping rules adapted to intermittency
    ↓
v2.5 (Metrics): Quantify intermittency-related decomposition quality
    ↓
v2.6 (Distributed): Scale robust ensemble methods
    ↓
v2.7 (Validation): Benchmark all algorithms on intermittent signals
```

### Key Intermittency Integration Points in Checklist

| Feature | Release | Intermittency Task | Impact |
|---------|---------|-------------------|--------|
| **Adaptive Buffering** | v2.0 | T-266–T-270 | Real-time intermittency detection |
| **Boundary Prediction** | v2.2 | T-298 (training data includes intermittent signals) | Neural model learns intermittency patterns |
| **ML Integration** | v2.4 | T-326–T-329 | Learnable boundaries for intermittent tasks |
| **Entropy Metrics** | v2.5 | T-330–T-335 | Quantify intermittency impact |
| **Test Suite** | v2.7 | T-352 (synthetic signals with known intermittency) | Validation on intermittent signals |

---

## Question 2: Do We Have a Checklist?

### ✅ YES — Comprehensive Execution Checklist Created

We've created **FUTURE_EXECUTION_CHECKLIST.md** with complete task breakdowns for all v2.0–v2.7 releases.

#### Structure of the Checklist

**Format:** Task-level tracking (like your v1.x EXECUTION_CHECKLIST.md)

Each task includes:
- **Task ID** (T-238 through T-360)
- **Description** — What needs to be done
- **Status** — 🔲 TODO | 🔄 IN_PROGRESS | 🔍 REVIEW | ✅ DONE
- **Mode** — architect | code | test | review | perf | docs | devops
- **Owner** — Team member assigned
- **Effort** — Hours estimated
- **Notes** — Dependencies, special considerations

#### Checklist by Release

| Release | Total Tasks | Status | Key Features |
|---------|------------|--------|--------------|
| **v2.0 (Streaming)** | ~40 | 🔲 | T-238–T-277: Online EMD, boundary prediction, sliding windows, WebSocket API |
| **v2.1 (GPU)** | ~25 | 🔲 | T-278–T-291: CUDA kernels, CuPy, PyTorch |
| **v2.2 (Boundary)** | ~15 | 🔲 | T-292–T-307: Neural prediction, entropy-based flagging |
| **v2.3 (2D/3D)** | ~20 | 🔲 | T-308–T-318: Image & volumetric decomposition |
| **v2.4 (ML)** | ~25 | 🔲 | T-319–T-329: Differentiable EMD, learnable boundaries |
| **v2.5 (Analysis)** | ~15 | 🔲 | T-330–T-339: Entropy metrics, mode mixing |
| **v2.6 (Distributed)** | ~20 | 🔲 | T-340–T-351: Arrow, Kubernetes, gRPC |
| **v2.7 (Validation)** | ~20 | 🔲 | T-352–T-360: Reference library, cross-impl validation |
| **TOTAL** | **~180** | 🔲 | All v2.0–v2.7 features |

#### Example: Feature F-2.0.1 (Online EMD)

```markdown
| Task ID | Description | Status | Mode | Owner | Effort |
|---------|-------------|--------|------|-------|--------|
| T-238   | Design StreamingState struct | 🔲 | architect | — | 2h |
| T-239   | Define acceptance tests (< 10ms) | 🔲 | test | — | 2h |
| T-240   | Implement StreamingDecomposer::new() | 🔲 | code | — | 3h |
| ...     | ... | ... | ... | ... | ... |
| T-249   | Cross-language validation | 🔲 | test | — | 3h |
```

#### Key Features of the Checklist

1. **ATDD Format** — Acceptance criteria defined as tests before implementation
2. **Engineering Practice Compliance** — Every task must satisfy:
   - TDD (test-first)
   - ATDD (acceptance test-driven)
   - DDD (domain-driven design)
   - Clean Code / Clean Architecture
   - Numerical Validation

3. **Dependency Tracking** — Tasks reference v1.x dependencies (e.g., streaming depends on boundary prediction)

4. **Effort Estimation** — Total v2.x effort: ~18–23 person-months

5. **Intermittency Integration Points** — Clearly marked where intermittency is addressed

---

## Question 3: Are There New Design Changes Needed?

### ✅ YES — 5 Major Design Changes Required for v2.x

We've documented **23 design tasks (DES-001 through DES-023)** totaling ~50 hours of architectural work.

### 1. **Adapter Layer for Streaming/GPU/ML** (v2.0 critical)

**Current Architecture (v1.x):**
```
Bindings → Domain (EMD, EEMD) → Foundation (Signal, Sifting)
```

**New Architecture (v2.x):**
```
Bindings 
    ↓
Adapter Layer ← (NEW) handles Streaming, GPU, ML, Distributed
    ↓
Domain (EMD, EEMD, MEMD)
    ↓
Foundation (Signal, Sifting, Boundary, Hilbert)
```

**Why Needed:**
- Streaming, GPU, and ML features are **orthogonal concerns** that shouldn't leak into the core domain
- Pure-wrap contract (established in v1.x) must extend to v2.x
- Domain types (Signal, ImfCollection) need stable adapter boundaries

**Design Tasks:**
| Task ID | Description | Effort | Status |
|---------|-------------|--------|--------|
| DES-001 | Define adapter trait hierarchy | 2h | 🔲 |
| DES-002 | Refactor domain types for adapter compatibility | 4h | 🔲 |
| DES-003 | Document adapter contract | 2h | 🔲 |

**Code Example:**
```rust
// ADAPTER LAYER (new)
pub trait DecompositionAdapter {
    fn decompose(&self, signal: &Signal) -> Result<ImfCollection>;
}

pub struct StreamingAdapter {
    core: Arc<EmdDecomposer>,
    predictor: Box<dyn BoundaryPrediction>,
}

pub struct GpuAdapter {
    executor: Box<dyn GpuExecutor>,
}

// DOMAIN LAYER (unchanged from v1.x)
pub fn decompose(signal: &Signal, config: &EmdConfig) -> Result<DecompositionResult> {
    // Pure algorithm; no knowledge of streaming/GPU
}
```

---

### 2. **Streaming State Management** (v2.0 critical)

**New Type Needed:**
```rust
pub struct StreamingState {
    chunk_id: u64,
    sifting_history: Vec<SiftingIteration>,
    last_envelope: (Vec<f64>, Vec<f64>),  // upper, lower
    predictor_state: Box<dyn PredictorState>,
    buffer: RingBuffer<f64>,  // Fixed-size ring buffer
}
```

**Why Needed:**
- v1.x assumes stateless, single-signal processing
- Streaming requires maintaining state between chunks
- Ring buffer prevents memory leaks in long-running processes

**Design Tasks:**
| Task ID | Description | Effort | Status |
|---------|-------------|--------|--------|
| DES-004 | Design RingBuffer for fixed memory footprint | 3h | 🔲 |
| DES-005 | Design PredictorState trait | 2h | 🔲 |

---

### 3. **GPU Memory Abstraction** (v2.1 critical)

**New Trait Needed:**
```rust
pub trait GpuExecutor: Send + Sync {
    fn decompose(&self, signal: &[f64], config: &EmdConfig) 
        -> Result<Vec<Vec<f64>>>;
    fn device_name(&self) -> String;
}

pub struct CudaExecutor { /* ... */ }
pub struct RocmExecutor { /* ... */ }
pub struct CpuExecutor { /* fallback */ }
```

**Why Needed:**
- Isolate CUDA/ROCm specifics behind trait boundary
- Support device fallback (if GPU unavailable, use CPU)
- Enable future GPU backends (Intel SYCL, WebGPU)

**Design Tasks:**
| Task ID | Description | Effort | Status |
|---------|-------------|--------|--------|
| DES-006 | Design GpuExecutor trait + CPU fallback | 2h | 🔲 |
| DES-007 | Implement device selection logic | 3h | 🔲 |

---

### 4. **Differentiable EMD Module** (v2.4 critical)

**New Module Needed:**
```rust
pub mod ml {
    pub struct DifferentiableEmd {
        config: EmdConfig,
        implicit_fn: Box<dyn Fn(&Signal) -> Result<ImfCollection>>,
    }
    
    impl DifferentiableEmd {
        pub fn forward(&self, signal: &Signal) -> Result<ImfCollection>;
        pub fn vjp(&self, signal: &Signal, grad: &[Vec<f64>]) 
            -> Result<Vec<f64>>;  // Vector-Jacobian product
    }
}
```

**Why Needed:**
- Sifting loop is iterative; implicit differentiation required
- Must support backpropagation through EMD for neural networks
- Gradient stability is non-trivial (may explode without care)

**Design Tasks:**
| Task ID | Description | Effort | Status |
|---------|-------------|--------|--------|
| DES-008 | Design implicit differentiation strategy | 4h | 🔲 |
| DES-009 | Document gradient flow through sifting | 2h | 🔲 |

---

### 5. **Hierarchical Configuration System** (v2.0+, all releases)

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
pub struct EmdConfig { /* base settings */ }

pub struct StreamingConfig {
    base: EmdConfig,          // inherit all v1.x settings
    chunk_size: usize,        // new for v2.0
    predictor: Arc<dyn ...>,  // new for v2.0
}

pub struct GpuConfig {
    base: EmdConfig,
    device_id: u32,           // new for v2.1
    mixed_precision: bool,    // new for v2.1
}

pub struct MlConfig {
    base: EmdConfig,
    max_gradient_norm: f64,   // new for v2.4
    learning_rate: f64,       // new for v2.4
}
```

**Why Needed:**
- v1.x must remain stable (backward-compatible)
- v2.x features require new config parameters
- Composition pattern avoids config explosion

**Design Tasks:**
| Task ID | Description | Effort | Status |
|---------|-------------|--------|--------|
| DES-010 | Design config inheritance pattern | 2h | 🔲 |
| DES-011 | Implement validation for each subtype | 3h | 🔲 |

---

### 6. **Intermittency Detection & Adaptation** (v2.0, v2.4, v2.5)

**New Types Needed:**
```rust
pub struct IntermittencyMetrics {
    spectral_entropy: f64,        // 0–1, lower = more ordered
    extrema_spacing_cv: f64,      // coefficient of variation
    stationarity_score: f64,      // 0–1, higher = more stationary
}

pub enum AdaptiveAlgorithm {
    EMD,      // stationarity_score > 0.8
    EEMD,     // 0.5 < stationarity_score < 0.8
    CEEMDAN,  // stationarity_score < 0.5
}

impl AdaptiveAlgorithm {
    pub fn select_from_metrics(metrics: &IntermittencyMetrics) -> Self;
}
```

**Why Needed:**
- Real-time signal monitoring (v2.0 streaming) requires intermittency assessment
- Adaptive algorithm selection improves decomposition quality
- Learnable boundaries (v2.4) need intermittency context

**Design Tasks:**
| Task ID | Description | Effort | Status |
|---------|-------------|--------|--------|
| DES-012 | Design IntermittencyMetrics struct | 2h | 🔲 |
| DES-013 | Implement adaptive algorithm selection policy | 3h | 🔲 |
| DES-014 | Integrate with streaming mode | 2h | 🔲 |

---

### 7. **Async FFI for Python/JavaScript** (v2.0)

**Why Needed:**
- v1.x is synchronous (blocking calls)
- Streaming EMD should not block event loops
- Python asyncio and JavaScript Promises need async bindings

**Design Tasks:**
| Task ID | Description | Effort | Status |
|---------|-------------|--------|--------|
| DES-015 | Design async FFI protocol | 2h | 🔲 |
| DES-016 | Implement PyO3 async wrappers | 3h | 🔲 |
| DES-017 | Implement WASM Promise bindings | 3h | 🔲 |

**Code Example (Python):**
```python
# v1.x (synchronous)
imfs = ferromode_py.decompose(signal, config)

# v2.0 (asynchronous)
imfs = await ferromode_py.decompose_async(signal, config)
imfs = await streaming_decomposer.decompose_chunk_async(chunk)
```

---

### 8. **Zero-Copy Array Bindings for GPU/Arrow** (v2.1, v2.6)

**Why Needed:**
- CuPy arrays live on GPU; copying to host is expensive
- Arrow columnar format enables zero-copy data sharing
- FFI must preserve pointer identity without copying

**Design Tasks:**
| Task ID | Description | Effort | Status |
|---------|-------------|--------|--------|
| DES-018 | Design zero-copy pointer protocol (DLpack) | 2h | 🔲 |
| DES-019 | Implement CuPy PyO3 bindings | 4h | 🔲 |

**Code Example (Python):**
```python
# CuPy GPU array (lives on GPU)
signal_gpu = cupy.array([1, 2, 3, ...])

# Decompose WITHOUT copying to host
result_gpu = ferromode_py.decompose_gpu(signal_gpu)
# Result is also on GPU; user can further process on GPU
```

---

### 9. **Testing & Validation Architecture** (all releases)

**Why Needed:**
- Streaming/batch equivalence testing requires specialized harness
- Cross-implementation validation (Ferromode vs R emd vs PyEMD) needs CI integration
- Intermittency-specific test suite requires signal library with known properties

**Design Tasks:**
| Task ID | Description | Effort | Status |
|---------|-------------|--------|--------|
| DES-020 | Design streaming/batch comparison harness | 2h | 🔲 |
| DES-021 | Implement automated error reporting | 2h | 🔲 |
| DES-022 | Design cross-implementation test harness | 3h | 🔲 |
| DES-023 | Integrate into CI/CD pipeline | 3h | 🔲 |

---

## Design Changes Summary Table

| Change | Effort | Priority | Needed By | Releases | Status |
|--------|--------|----------|-----------|----------|--------|
| **1. Adapter Layer** | 8h | 🔴 High | v2.0 | v2.0 | 🔲 |
| **2. Streaming State** | 5h | 🔴 High | v2.0 | v2.0 | 🔲 |
| **3. GPU Abstraction** | 5h | 🔴 High | v2.1 | v2.1 | 🔲 |
| **4. Differentiable EMD** | 6h | 🔴 High | v2.4 | v2.4 | 🔲 |
| **5. Config Hierarchy** | 5h | 🟡 Medium | v2.0 | v2.0+ | 🔲 |
| **6. Intermittency Adapt** | 7h | 🟡 Medium | v2.0 | v2.0, v2.4, v2.5 | 🔲 |
| **7. Async FFI** | 8h | 🟡 Medium | v2.0 | v2.0 | 🔲 |
| **8. Zero-Copy Arrays** | 6h | 🟡 Medium | v2.1, v2.6 | v2.1, v2.6 | 🔲 |
| **9. Testing Architecture** | 10h | 🟡 Medium | v2.0 | v2.0+ | 🔲 |
| **TOTAL** | **~60h** | — | — | — | 🔲 |

---

## Execution Readiness Checklist

Before starting v2.0 work, complete these prerequisites:

- [ ] **Design Review:** All 9 design changes reviewed and approved by team
- [ ] **Architecture Sign-Off:** Adapter layer pattern agreed; adapter contract documented
- [ ] **Intermittency Integration:** Confirm intermittency flows through all 8 releases
- [ ] **Config Inheritance:** Finalize EmdConfig → StreamingConfig → (v2.1+)Config pattern
- [ ] **Testing Strategy:** Cross-validation framework (streaming vs batch, cross-impl) planned
- [ ] **Async Support:** FFI design for Python asyncio and JavaScript Promises finalized
- [ ] **GPU Strategy:** CUDA-first or multi-backend (CUDA, ROCm) from start?
- [ ] **Resource Allocation:** Team assigned to design work (DES-001 through DES-023)
- [ ] **CI/CD Extended:** GitHub Actions updated to test streaming, GPU, distributed features
- [ ] **Risk Mitigation:** Known risks (streaming edge cases, GPU memory, gradient stability) have mitigation plans

---

## Summary

### 1. Intermittency ✅
- **Already in v1.x:** Ensemble methods + 8 boundary strategies
- **Added in v2.x:** Real-time detection, adaptive algorithm selection, neural boundaries, entropy metrics
- **Integrated throughout:** v2.0, v2.2, v2.4, v2.5, v2.6, v2.7

### 2. Checklist ✅
- **FUTURE_EXECUTION_CHECKLIST.md:** 180 tasks across v2.0–v2.7
- **Task-level tracking:** Status, mode (code/test/docs), effort, dependencies
- **Intermittency tasks clearly marked:** T-266–T-270, T-298, T-326–T-329, T-330–T-335, T-352

### 3. Design Changes ✅
- **9 major architectural changes needed**
- **~60 hours of design work** (DES-001 through DES-023)
- **High-priority:** Adapter layer, streaming state, GPU abstraction (v2.0–v2.1)
- **Medium-priority:** Config hierarchy, intermittency adaptation, async FFI

---

*Next Steps:*
1. **Review** design changes (DES-001–DES-023) with team
2. **Prioritize** design work before v2.0 implementation starts
3. **Track** progress in FUTURE_EXECUTION_CHECKLIST.md
4. **Monitor** intermittency integration across all releases
5. **Plan** resource allocation for ~18–23 person-months of v2.x work

---

*Last updated: April 4, 2026*
