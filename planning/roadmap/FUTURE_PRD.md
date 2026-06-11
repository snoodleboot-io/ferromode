# Product Requirements Document (PRD) — Future Releases
## Ferromode v2.x Roadmap

**Version:** 1.0  
**Date:** April 2026  
**Status:** Living Document  
**Owner:** Core Engineering  

---

## 1. Executive Summary

This document defines the product vision for Ferromode v2.x through v2.7, extending the v1.x foundation (core EMD algorithms with multi-language bindings) into new domains: real-time processing, GPU acceleration, machine learning integration, distributed computing, and advanced signal analysis.

Each major version (v2.0 through v2.7) targets a specific capability maturity level, enabling users to deploy Ferromode in increasingly complex environments: IoT/edge devices (streaming), high-performance computing (GPU), research pipelines (ML differentiation), and cloud infrastructure (distributed systems).

---

## 2. Vision & Strategic Goals

### 2.1 Vision Statement

**"Make Ferromode the default choice for EMD across real-time, GPU-accelerated, and machine learning workloads — extending beyond research into production systems."**

### 2.2 Strategic Goals

| Goal | Rationale | Success Metric |
|------|-----------|-----------------|
| **Real-time EMD** | Enable IoT sensors, live financial data, and streaming signals to leverage EMD without batch processing delays | Streaming latency < 10ms for 1000-sample windows |
| **GPU Acceleration** | Support massive-scale ensemble methods and multi-modal data processing (images, video, 3D) | 50–100x speedup for EEMD/CEEMDAN on GPU vs. CPU |
| **ML Integration** | Make EMD a first-class component in PyTorch/TensorFlow pipelines, differentiable end-to-end | Train models with EMD as a learnable preprocessing layer |
| **Distributed Processing** | Scale to cloud and HPC clusters for embarrassingly parallel ensemble trials | Process 1000+ EEMD trials across 100 GPU nodes |
| **Spatial EMD** | Decompose 2D/3D data (images, CT scans, video) natively, not just 1D signals | Decompose 1024×1024 medical image in < 5s |
| **Advanced Analysis** | Go beyond basic IMF extraction: time-frequency entropy, mode mixing metrics, change detection | Detect abrupt shifts in nonstationary signals; quantify decomposition quality |

---

## 3. Release Roadmap

### 3.0: v2.0 — Streaming & Real-Time Processing (Q3 2027)

**Motivation:** Enable EMD on live sensor data, IoT devices, and real-time signal processing pipelines.

#### 3.0.1 Functional Requirements

| Requirement | Description |
|------------|-------------|
| **FR-2.0.1: Online EMD** | Process data in fixed-size chunks as it arrives; maintain sifting state between chunks; guarantee envelope continuity across boundaries |
| **FR-2.0.2: Sliding Window EMD** | Fixed-size window with configurable overlap; emit IMFs for stable segments; discard unstable transients |
| **FR-2.0.3: Adaptive Buffering** | Dynamically adjust buffer size based on signal stationarity; short buffers for high-nonstationarity, larger for quasi-stationary segments |
| **FR-2.0.4: Boundary Prediction** | Use AR(p) or LSTM models to predict signal continuation at chunk boundaries, improving envelope estimates |
| **FR-2.0.5: Incremental IMF Extraction** | Extract and emit IMFs as they stabilize rather than waiting for full convergence; reduce latency |
| **FR-2.0.6: WebSocket API** | Bidirectional streaming protocol; client sends signal chunks, server streams back IMF frames |
| **FR-2.0.7: Zero-Copy Streaming** | Memory-efficient ring buffers; no allocations between chunks; embed-device friendly (< 10 MB footprint) |

#### 3.0.2 Technical Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Maintaining envelope continuity across chunk boundaries | Use boundary prediction model trained on stationary-signal statistics; validate envelope smoothness at boundaries |
| Adapting stopping criteria for partial signals | Deploy time-aware stopping: max_iterations adjusted based on window size and historical convergence rates |
| Managing memory footprint in long-running processes | Ring buffer for input; single-buffer reuse for intermediate computations; explicit drop of old chunk data |
| Ensuring numerical equivalence with batch processing | Offline validation: run streaming on historical data, compare chunk-by-chunk IMFs with batch results; target 1e-10 error |
| Handling transient signals with poor endpoints | Predictive models for boundary; fallback to symmetric extension if prediction fails |

#### 3.0.3 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.0.1 | Streaming latency (EMD, 1000 samples) | < 10ms (CPU) |
| NFR-2.0.2 | Memory: peak RSS for 1-minute rolling window | < 100 MB |
| NFR-2.0.3 | API stability | Backward-compatible with v1.x config types |
| NFR-2.0.4 | Cross-language bindings | Python (asyncio), JavaScript (Node.js), Rust (tokio) |
| NFR-2.0.5 | Numerical equivalence to batch | Within 1e-10 relative error for identical signals |

#### 3.0.4 Out of Scope (Deferred to v2.1+)
- GPU acceleration for streaming
- Complex signal (IQ/analytic signal) streaming
- Multivariate streaming (MEMD in streaming mode)

---

### 3.1: v2.1 — GPU Acceleration (Q1 2028)

**Motivation:** Leverage GPU parallelism for large-scale EMD processing on massive datasets.

#### 3.1.1 Functional Requirements

| Requirement | Description |
|------------|-------------|
| **FR-2.1.1: CUDA Kernels** | Custom kernels for extrema detection, cubic spline interpolation, sifting loop — data-parallel operations |
| **FR-2.1.2: CuPy Integration** | Python binding exposing GPU-accelerated NumPy-like API; arrays live on GPU; zero host-device copying |
| **FR-2.1.3: PyTorch Layers** | `torch.nn.Module` subclass for EMD; gradient-compatible for backprop through ensemble aggregation |
| **FR-2.1.4: WebGPU WASM** | Browser-based GPU acceleration; fallback to CPU if unavailable; Chrome/Firefox/Safari support |
| **FR-2.1.5: Multi-GPU Support** | Distribute EEMD/CEEMDAN trials across 2+ GPUs; MPI or gRPC coordination |
| **FR-2.1.6: Mixed Precision** | FP16 compute, FP32 accumulation; memory savings for large-scale ensemble trials |

#### 3.1.2 Target Algorithms

| Algorithm | Parallelization Strategy |
|-----------|-------------------------|
| **EEMD** | Embarrassingly parallel: each trial on a separate GPU stream |
| **CEEMDAN** | Trials parallelized; within-trial sifting sequential |
| **VMD** | Frequency-domain FFT on GPU; iterative refinement on GPU |
| **MEMD** | Direction-wise parallelism: decompose each direction on separate GPU |

#### 3.1.3 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.1.1 | Speedup: EEMD 200 trials, 100k samples, GPU vs. 8-core CPU | ≥ 50x |
| NFR-2.1.2 | GPU memory: peak usage for EEMD 200 trials, 100k samples | < 8 GB (RTX 3080 capable) |
| NFR-2.1.3 | Numerical equivalence to CPU | Within 1e-9 relative error (FP32 rounding) |
| NFR-2.1.4 | Hardware support | NVIDIA CUDA 12.0+; AMD ROCm 5.0+; Intel SYCL roadmap (v2.2) |
| NFR-2.1.5 | Build artifact size | CUDA .so < 500 MB (packaged separately from CPU binary) |

---

### 3.2: v2.2 — Advanced Boundary Conditions (Q3 2028)

**Motivation:** Improve end-effect handling for non-stationary and transient signals.

#### 3.2.1 Functional Requirements

| Requirement | Description |
|------------|-------------|
| **FR-2.2.1: Neural Boundary Prediction** | LSTM/Transformer model trained on synthetic and real signals; predict signal extension for given endpoint; reduce end effects |
| **FR-2.2.2: Adaptive Boundary Selection** | Automatically choose optimal boundary condition per signal segment (peak density, trend, frequency content); evaluate via envelope quality metrics |
| **FR-2.2.3: Wavelet-Based Extension** | Use wavelet packet decomposition at boundaries; reconstruct likely signal continuation |
| **FR-2.2.4: Mirror-Synthesis** | Generate synthetic boundary data via autoencoder or VAE; statistically consistent with signal interior |
| **FR-2.2.5: Entropy-Based Thresholding** | Detect when boundary effects dominate IMF quality; flag IMFs with high boundary contamination; suggest alternative boundary strategy |
| **FR-2.2.6: Boundary-Aware IMF Selection** | Weight IMFs by estimated boundary contamination; allow user to filter out degraded modes |

#### 3.2.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.2.1 | Model training time | < 1 hour on 10k synthetic signals (GPU) |
| NFR-2.2.2 | Inference latency per signal | < 50 ms (boundary prediction) |
| NFR-2.2.3 | Numerical stability | No NaNs or Inf; graceful fallback to symmetric extension on model failure |
| NFR-2.2.4 | Model size | < 5 MB (quantized) |
| NFR-2.2.5 | Cross-domain transfer | Pre-trained model works on geophysics, biomedicine, finance with < 10% accuracy loss |

---

### 3.3: v2.3 — Multidimensional & Spatiotemporal EMD (Q1 2029)

**Motivation:** Extend beyond 1D signals to images, video, and volumetric data.

#### 3.3.1 Functional Requirements

| Requirement | Description |
|------------|-------------|
| **FR-2.3.1: 2D EMD** | Grayscale and color image decomposition via separable or non-separable extensions; IMF2D as 2D arrays |
| **FR-2.3.2: 3D EMD** | Volumetric decomposition (fMRI, CT, seismic); IMF3D as 3D arrays; optional voxel-wise or spatial smoothing |
| **FR-2.3.3: Complex Signal EMD** | Handle analytic signals and I/Q pairs natively; preserve complex envelope structure |
| **FR-2.3.4: Multichannel Image EMD** | Hyperspectral/multispectral images; decompose across channels simultaneously; leverage channel correlation |
| **FR-2.3.5: TensorEMD** | Decompose higher-order tensors (e.g., diffusion MRI tensors, video as time×height×width); Tucker or CP decomposition integration |
| **FR-2.3.6: Steerable Filters** | Oriented filter banks for directional feature extraction; apply EMD to oriented subbands |

#### 3.3.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.3.1 | 2D EMD: 512×512 grayscale image | < 5 seconds (CPU) |
| NFR-2.3.2 | 3D EMD: 128×128×128 volume | < 30 seconds (CPU) or < 3 seconds (GPU) |
| NFR-2.3.3 | Memory: 3D EMD peak usage | < 4 GB for 256×256×256 |
| NFR-2.3.4 | Numerical stability | No boundary artifacts at image edges; periodic or symmetric extension |
| NFR-2.3.5 | GPU acceleration | Available for 2D/3D EMD on CUDA; WASM fallback for browser |

---

### 3.4: v2.4 — Machine Learning Integration (Q3 2029)

**Motivation:** Make EMD a differentiable, learnable component in modern ML pipelines.

#### 3.4.1 Functional Requirements

| Requirement | Description |
|------------|-------------|
| **FR-2.4.1: Differentiable EMD** | Autograd-compatible implementation; gradients flow through sifting loop via implicit differentiation or unrolled loop |
| **FR-2.4.2: Learnable Boundary Conditions** | Neural network predicts optimal boundary extension; trained end-to-end with downstream task |
| **FR-2.4.3: IMF Selection Networks** | Learn which IMFs to retain; gating network predicts per-IMF importance; pruning via straight-through estimator |
| **FR-2.4.4: Keras/TensorFlow Layers** | `tf.keras.layers.Layer` subclass; integrates with Functional/Sequential/Model APIs |
| **FR-2.4.5: PyTorch Layers** | `torch.nn.Module` with backward() implementation; supports autograd, mixed precision, distributed data parallel |
| **FR-2.4.6: AutoEncoder-EMD** | Learn optimal number of IMFs via reconstruction loss; variational EMD with learned stopping criterion |
| **FR-2.4.7: Attention over IMFs** | Transformer-based attention weights IMFs by relevance to downstream prediction task |
| **FR-2.4.8: Physics-Informed EMD** | Incorporate domain constraints (e.g., energy conservation, causality); constrained optimization or Lagrange multiplier learning |

#### 3.4.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.4.1 | Gradient computation overhead | < 3x forward pass time |
| NFR-2.4.2 | Memory for gradient storage | < 2x peak forward memory |
| NFR-2.4.3 | Numerical stability | Gradients < 1e6 magnitude; no gradient explosion in deep networks |
| NFR-2.4.4 | Framework support | PyTorch 2.0+, TensorFlow 2.10+ |
| NFR-2.4.5 | Ease of integration | User can drop `EMDLayer()` into existing model; no boilerplate |

---

### 3.5: v2.5 — Advanced Post-Processing & Metrics (Q1 2030)

**Motivation:** Provide richer analysis tools beyond basic IMF extraction.

#### 3.5.1 Functional Requirements

| Requirement | Description |
|------------|-------------|
| **FR-2.5.1: Multivariate Hilbert Spectral Analysis** | Compute instantaneous frequency correlations across channels; joint time-frequency representations |
| **FR-2.5.2: Time-Frequency Entropy Measures** | Spectral entropy, permutation entropy, Lempel-Ziv complexity, sample entropy on Hilbert spectrum |
| **FR-2.5.3: Hilbert-Huang Transform Significance Testing** | Bootstrap and surrogate data methods; confidence intervals on marginal spectrum peaks |
| **FR-2.5.4: Mode Mixing Metrics** | Quantify frequency overlap between IMFs; visualize mode mixing map; suggest refinement strategies |
| **FR-2.5.5: Energy-Time-Frequency Distributions** | Joint distributions of energy vs. (time, frequency) beyond marginal spectrum; heatmaps |
| **FR-2.5.6: Nonlinearity Measures** | Higher-order spectra, bispectral analysis, third-order cumulant spectra; quantify signal nonlinearity |
| **FR-2.5.7: Change Point Detection** | Detect abrupt shifts in IMF characteristics (frequency, amplitude, energy); time-series segmentation |
| **FR-2.5.8: IMF Quality Metrics** | Compute orthogonality index, separation index, energy ratio between IMFs |

#### 3.5.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.5.1 | Entropy computation (100k sample signal) | < 100 ms |
| NFR-2.5.2 | Significance test (1000 surrogates) | < 10 seconds |
| NFR-2.5.3 | Memory efficiency | Use NumPy/GPU for large matrices; no intermediate allocations |
| NFR-2.5.4 | Visualization support | Return data structures compatible with matplotlib, plotly |

---

### 3.6: v2.6 — Streaming & Distributed Systems (Q3 2030)

**Motivation:** Enable EMD in cloud, edge, and HPC environments.

#### 3.6.1 Functional Requirements

| Requirement | Description |
|------------|-------------|
| **FR-2.6.1: Apache Arrow Integration** | Zero-copy columnar data exchange; Parquet I/O; Arrow Streaming IPC protocol |
| **FR-2.6.2: gRPC/Protobuf Interface** | Language-agnostic microservice boundary; EMD service accessible from any language; server streaming for real-time IMFs |
| **FR-2.6.3: Kubernetes Operator** | Auto-scaling EMD processing pods; CRD for EMD job specifications; integration with Knative for serverless |
| **FR-2.6.4: Dask Integration** | Parallel ensemble methods on distributed clusters; IMF aggregation across workers |
| **FR-2.6.5: Ray Integration** | Task-parallel EEMD/CEEMDAN on heterogeneous clusters; fault-tolerant distributed trials |
| **FR-2.6.6: AWS Lambda Layer** | Serverless EMD processing for event-driven architectures; S3 input/output; CloudWatch monitoring |
| **FR-2.6.7: Edge TPU Support** | Quantized boundary prediction models for on-device inference; sub-millisecond inference |

#### 3.6.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.6.1 | gRPC latency (single EMD call) | < 50 ms (local) |
| NFR-2.6.2 | Dask scaling efficiency | ≥ 80% (200 trials across 10 workers) |
| NFR-2.6.3 | Kubernetes deployment time | Pod ready < 5 seconds |
| NFR-2.6.4 | Lambda cold start | < 1 second |
| NFR-2.6.5 | Arrow serialization overhead | < 5% of total job time |

---

### 3.7: v2.7 — Validation & Benchmarking Suite (Q1 2031)

**Motivation:** Provide standardized testing and comparison against reference implementations.

#### 3.7.1 Functional Requirements

| Requirement | Description |
|------------|-------------|
| **FR-2.7.1: Reference Signal Library** | Expanded synthetic signals with ground-truth IMFs: AM/FM chirps, multicomponent signals, real geophysical/biomedical datasets |
| **FR-2.7.2: Cross-Implementation Validation** | Automated tests comparing Ferromode against R emd, Python PyEMD, MATLAB EMD toolbox, C reference (Rilling & Flandrin) |
| **FR-2.7.3: Benchmark Suite** | Standardized tests for speed, memory, accuracy; regression testing across versions; CI/CD enforcement |
| **FR-2.7.4: Regression Testing** | Historical version comparison; detect unintended performance degradations; alert on breaking changes |
| **FR-2.7.5: Property-Based Testing** | Generate signals with known IMF properties; verify invariants hold across implementations |
| **FR-2.7.6: Fuzzing Suite** | Malformed input testing: NaN, Inf, extreme values, empty arrays; security and robustness validation |
| **FR-2.7.7: Reproducibility Toolkit** | Export/import decomposition state; audit trail for parameter changes; publish results to registry |

#### 3.7.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.7.1 | Benchmark suite runtime | < 10 minutes (CI/CD) |
| NFR-2.7.2 | Cross-implementation test count | ≥ 100 (per algorithm family) |
| NFR-2.7.3 | Fuzz test cases | ≥ 10,000 (automated generation) |
| NFR-2.7.4 | Reproducibility data size | < 1 MB per decomposition (serialized) |

---

### 3.8: v2.x Extended Algorithm Family (Ongoing)

**Motivation:** Include related decomposition methods for comparison and hybridization.

#### Planned Additions

| Algorithm | Target Release | Status |
|-----------|-----------------|--------|
| **Synchrosqueezing Transform** | v2.4 | Roadmap |
| **Empirical Wavelet Transform** | v2.4 | Roadmap |
| **Complete Ensemble EMD with Adaptive Noise (CEEMDAN)** | v1.x | ✓ Done |
| **Adaptive Noise-Assisted MEMD (AN-AMEMD)** | v2.3 | Roadmap |
| **Proxy-Based EEMD** | v2.1 | Roadmap |
| **Improved Complete Ensemble EMD with Trial Optimization (ICEEMDTO)** | v2.5 | Roadmap |
| **Fractional Delay-Based EMD (FD-EMD)** | v2.2 | Roadmap |
| **EEMD with Adaptive Cluster Analysis (EEMD-ACA)** | v2.5 | Roadmap |

---

## 4. Engineering Practices for Future Work

All future work will follow the same standards established in v1.x:

### 4.1 TDD (Test-Driven Development)
- Tests written BEFORE implementation code
- Red-green-refactor cycle mandatory
- Minimum 90% line coverage, 80% branch coverage

### 4.2 ATDD (Acceptance Test-Driven Development)
- Acceptance criteria defined as executable tests before work begins
- Story not done until acceptance tests pass

### 4.3 DDD (Domain-Driven Design)
- Domain types encapsulate behavior; no anemic data structures
- Bounded contexts for each major concern (signal, boundary, algorithm, etc.)
- Anti-corruption layer at FFI/binding boundaries

### 4.4 Clean Code
- Max 20-line functions; single responsibility
- All public APIs documented with examples
- Clippy and rustfmt enforcement
- Boy Scout Rule: leave code cleaner than found

### 4.5 Clean Architecture
- Dependencies point inward; domain independent of frameworks
- Testable in isolation; no hidden coupling

### 4.6 Numerical Validation
- Cross-language/cross-implementation bit-equivalence testing
- Surrogate data and bootstrap validation for statistical claims
- Regression testing for numerical stability

---

## 5. Success Metrics for v2.x

| Metric | v2.0 | v2.1 | v2.2 | v2.3 | v2.4 | v2.5 | v2.6 |
|--------|------|------|------|------|------|------|------|
| GitHub stars | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
| PyPI monthly downloads | 15k | 30k | 50k | 75k | 100k | 125k | 150k |
| Academic citations | 20 | 40 | 60 | 80 | 100 | 120 | 150 |
| Production deployments | 5 | 20 | 50 | 100 | 200 | 300 | 500+ |
| Issue resolution (< 14 days) | 85% | 85% | 85% | 85% | 85% | 85% | 90% |

---

## 6. Constraints & Assumptions

### 6.1 Technical Constraints
- Backward compatibility with v1.x APIs maintained throughout v2.x
- MSRV (Rust): 1.75+ for all v2.x releases
- No proprietary dependencies
- All code under MIT or Apache-2.0

### 6.2 Resource Constraints
- Core team: 2–3 full-time engineers + community contributors
- Budget: Open source (volunteer-driven); potential sponsorship for GPU/infrastructure costs

### 6.3 Assumptions
- GPU hardware (NVIDIA CUDA) widely available and economically viable through cloud providers
- ML frameworks (PyTorch, TensorFlow) continue to evolve and maintain compatibility
- Kubernetes ecosystem continues to dominate container orchestration

---

## 7. Dependencies & Risks

### 7.1 Major Dependencies
- CUDA Toolkit 12.0+ for GPU features (external)
- PyTorch 2.0+ for differentiable layers (external)
- Apache Arrow for columnar data (external)
- gRPC/Protobuf for microservices (external)

### 7.2 Key Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| GPU code becomes stale if CUDA API changes | Medium | High | Maintain CI testing on multiple CUDA versions; use wrapper layer |
| Streaming mode introduces subtle bugs in edge cases | High | Medium | Extensive fuzzing; cross-validation with batch mode; field testing |
| ML layer gradient computation unstable for deep models | Medium | High | Research implicit differentiation; limit sifting iterations for stable gradients |
| Distributed scaling hits communication bottleneck | Low | Medium | Profile early; use batching and compression; consider async patterns |

---

## 8. Open Questions for v2.x

1. **Streaming:** Should we support arbitrary-length signals with online prediction, or require knowledge of expected signal properties upfront?
2. **GPU:** CUDA-only initially, or invest in AMD ROCm/Intel SYCL from the start?
3. **ML:** Should differentiable EMD support all boundary strategies, or just a subset (e.g., symmetric, AR model)?
4. **Multidimensional:** Should 2D/3D EMD decompose separably (per-axis) or non-separably (true 2D/3D extrema)?
5. **Distributed:** Is Kubernetes the primary deployment target, or should we prioritize HPC (Slurm) integration?
6. **Benchmarking:** Which external implementations (R, Python, MATLAB) should be the reference for cross-validation?

---

## 9. Glossary & Terminology

| Term | Definition |
|------|-----------|
| **Online EMD** | Processing fixed-size chunks as data arrives; state maintained between chunks |
| **Streaming** | Continuous, real-time data processing; latency-sensitive |
| **Boundary Prediction** | Using ML to forecast signal continuation beyond observed endpoints |
| **Differentiable** | Supporting automatic differentiation (gradients); compatible with backpropagation |
| **Ensemble** | Multiple decomposition runs (EEMD, CEEMDAN) aggregated for robustness |
| **IMF** | Intrinsic Mode Function; a decomposed component |
| **Hilbert Transform** | Transform that produces the analytic signal; basis for instantaneous frequency |
| **Mode Mixing** | Aliasing where one IMF contains multiple frequency components; undesirable in EMD |
| **Zero-Copy** | Data sharing without allocation/deallocation; memory-efficient |

---

## 10. References

- **Rilling, G., Flandrin, P., Goncalves, P.** (2003). "On Empirical Mode Decomposition and its applications." *IEEE Signal Processing Letters*, 10(8), 241–244.
- **Huang, N. E., et al.** (1998). "The empirical mode decomposition and the Hilbert spectrum for nonlinear and nonstationary time series analysis." *Proc. Royal Soc. London*, 454(1971), 903–995.
- **Wu, Z., Huang, N. E.** (2009). "Ensemble Empirical Mode Decomposition: A Noise-Assisted Data Analysis Method." *Advances in Adaptive Data Analysis*, 1(01), 1–41.
- **Torres, M. E., et al.** (2011). "A complete ensemble empirical mode decomposition with adaptive noise." *ICASSP 2011*, 4144–4147.

---

*Last updated: April 4, 2026*  
*Maintained by: Core Engineering Team*  
*Contributions welcome: See CONTRIBUTING.md*
