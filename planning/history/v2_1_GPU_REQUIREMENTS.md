# V2.1 GPU Requirements Document

**Release:** v2.1 (GPU Adapter)  
**Status:** Ready for Implementation  
**Updated:** April 8, 2026  
**Effort Estimate:** XL (5-7 weeks)

---

## 1. Executive Summary

V2.1 introduces **GPU acceleration** for EMD ensemble methods through the `GpuAdapter`. This enables **50x+ speedup** on NVIDIA CUDA and AMD ROCm devices while maintaining bit-identical parity with CPU decomposition.

**Key Capabilities:**
- Multi-GPU device detection and selection
- GPU memory pool management (VRAM allocation/deallocation)
- Parallel ensemble trial execution (EEMD, CEEMDAN)
- Speedup target: > 50x for EEMD 200 trials on 10k samples
- Fallback to CPU if GPU unavailable
- Mixed-precision support (FP32/FP16)

**Platform Support:**
- NVIDIA CUDA 11.8+ (Tesla, RTX, A100)
- AMD ROCm 5.0+ (MI100, MI200)
- Apple Metal (v2.8+, out of scope for v2.1)
- WebGPU in browsers (v2.9+, out of scope for v2.1)

**Non-Goal:** Distributed training (v2.6), machine learning optimization (v2.4), sparse decomposition.

---

## 2. Epic Breakdown

### Epic 1: GPU Device Management
**Features:**
- Device detection (CUDA/ROCm/Metal)
- Device selection and validation
- Multi-GPU support
- Device memory queries

### Epic 2: GPU Memory Management
**Features:**
- GpuMemoryPool for pre-allocation
- Memory fragmentation mitigation
- Automatic garbage collection
- OOM recovery strategies

### Epic 3: Parallel Kernel Execution
**Features:**
- GPU kernels for extrema finding
- GPU kernels for spline interpolation
- GPU kernels for FFT (via cufft/rocfft)
- Batch kernel execution

### Epic 4: GPU Adapter Layer
**Features:**
- GpuAdapter struct with device management
- decompose_gpu() method signature
- CPU ↔ GPU transfer orchestration
- Mixed-precision support

### Epic 5: Ensemble Parallelization
**Features:**
- Parallel EEMD trials on GPU
- Parallel CEEMDAN trials on GPU
- Result aggregation from GPU
- Trial batching strategy

### Epic 6: GPU vs CPU Parity Validation
**Features:**
- Bit-identity tests (within FP32 tolerance)
- Performance benchmarks (50x target)
- Cross-device validation (CUDA, ROCm)
- Numerical stability analysis

---

## 3. User Stories

### Story 1: GPU Acceleration for Ensemble Methods
**As a** EMD researcher analyzing large datasets  
**I want to** accelerate EEMD and CEEMDAN on GPU  
**So that** I can process 200+ trials in seconds instead of minutes

**Acceptance Criteria:**
- [ ] `decompose_gpu()` accepts GPU-resident arrays or auto-transfers from CPU
- [ ] EEMD 200 trials on 10k samples completes in < 20 seconds (> 50x speedup)
- [ ] CEEMDAN 200 trials on 10k samples completes in < 20 seconds
- [ ] GPU memory usage capped at configured max (e.g., 8 GB)
- [ ] Falls back to CPU if GPU OOM or unavailable
- [ ] Results bit-identical to CPU within FP32 tolerance

**Definition of Done:**
- GpuAdapter fully implemented
- Ensemble parallelization working
- Speedup benchmarks passing
- Fallback tested and working
- Documentation with performance expectations

---

### Story 2: Multi-GPU Support and Device Selection
**As a** machine with multiple GPUs  
**I want to** distribute ensemble trials across available GPUs  
**So that** I can maximize hardware utilization

**Acceptance Criteria:**
- [ ] Device auto-detection works on CUDA and ROCm
- [ ] Device selection API: `select_device(device_id)` or `auto_select()`
- [ ] Multi-GPU decomposition: trials split across devices
- [ ] Memory queries per device
- [ ] Device failover if one GPU fails
- [ ] Documentation of multi-GPU strategy

**Definition of Done:**
- Device management layer complete
- Multi-GPU tests passing
- Auto-selection logic verified
- Failover tested

---

### Story 3: Safe Memory Management on GPU
**As a** long-running GPU decomposition service  
**I want to** ensure predictable memory usage without leaks  
**So that** the service remains stable over days/weeks of operation

**Acceptance Criteria:**
- [ ] GpuMemoryPool with pre-allocation strategy
- [ ] Memory pool size configurable
- [ ] No implicit GPU allocations in hot path
- [ ] Memory leak detection tests
- [ ] OOM graceful recovery (fall back to CPU or error cleanly)
- [ ] Memory usage profiling in CI

**Definition of Done:**
- GpuMemoryPool fully implemented
- Memory leak tests passing
- Profiling data collected
- Documentation of memory model

---

### Story 4: Mixed-Precision GPU Computation
**As a** researcher optimizing for speed over precision  
**I want to** use FP16 computation on capable GPUs  
**So that** I can achieve higher throughput for exploratory analysis

**Acceptance Criteria:**
- [ ] GpuConfig supports `mixed_precision: bool` flag
- [ ] FP16 kernels use tensor cores (if available)
- [ ] Accumulation in FP32 for numerical stability
- [ ] Accuracy loss quantified (vs FP32)
- [ ] Fallback to FP32 if device doesn't support FP16
- [ ] Speedup and accuracy trade-off documented

**Definition of Done:**
- Mixed-precision kernels implemented
- Accuracy benchmarks showing trade-offs
- Documentation with usage examples

---

### Story 5: CPU-GPU Parity Validation
**As a** quality engineer  
**I want to** verify GPU results are numerically equivalent to CPU  
**So that** I can trust GPU acceleration with same confidence as CPU

**Acceptance Criteria:**
- [ ] Parity tests on 50+ diverse signals
- [ ] Tolerance: element-wise L∞ < 1e-5 (FP32 machine epsilon)
- [ ] Cross-device validation (CUDA, ROCm separately)
- [ ] Edge cases: very small/large values, nearly-zero IMFs
- [ ] Numerical stability analysis in report
- [ ] Bit-level testing for deterministic operations

**Definition of Done:**
- Parity test suite passing
- Tolerance analysis documented
- Cross-platform validation complete
- Numerical stability report

---

## 4. Acceptance Criteria

### Functional Requirements

1. **GPU Device Management**
   - [ ] `cuda_available()` → bool
   - [ ] `rocm_available()` → bool
   - [ ] `device_count()` → usize
   - [ ] `device_name(id)` → String
   - [ ] `device_memory(id)` → (total, free, allocated)
   - [ ] `select_device(id)` → Result<()>
   - [ ] Error handling for invalid device IDs

2. **GPU Memory Pool**
   - [ ] GpuMemoryPool struct with pre-allocated buffers
   - [ ] `allocate(size)` → *mut f64 from pool
   - [ ] `deallocate(ptr)` frees back to pool
   - [ ] `reset()` clears all allocations
   - [ ] Fragmentation metrics tracked
   - [ ] OOM detection and reporting
   - [ ] Thread-safe access (Arc<Mutex<>>)

3. **GPU Kernels**
   - [ ] Extrema finding kernel (local min/max)
   - [ ] Cubic spline interpolation kernel
   - [ ] FFT kernel (via cufft/rocfft)
   - [ ] Ensemble iteration kernel (batch sifting)
   - [ ] Result reduction/aggregation kernels
   - [ ] Kernel launch validation (grid/block sizes)

4. **GpuAdapter Interface**
   - [ ] `new(config: EmdConfig, device: u32) → Result<Self>`
   - [ ] `decompose_gpu(signal: &Signal) → Result<DecompositionResult>`
   - [ ] `decompose_gpu_ensemble(signal, config, num_trials) → Result<>`
   - [ ] `device_info() → DeviceInfo`
   - [ ] `reset_device() → Result<()>`
   - [ ] Automatic CPU fallback on GPU error
   - [ ] Config validation (device memory vs buffer size)

5. **Ensemble Parallelization**
   - [ ] Batch trial execution on GPU
   - [ ] Trial distribution strategy (per-GPU queue)
   - [ ] Result aggregation (averaging, sorting)
   - [ ] Progress tracking for long-running trials
   - [ ] Cancellation support (cancel all trials)
   - [ ] Trial-level error handling

6. **Mixed Precision**
   - [ ] `GpuConfig::mixed_precision: bool`
   - [ ] FP16 tensor core usage detection
   - [ ] Fallback to FP32 if unsupported
   - [ ] Accumulation in FP32 (for stability)
   - [ ] Configuration of precision per operation

### Performance Requirements

1. **Speedup**
   - [ ] EEMD (200 trials, 10k samples): **> 50x** speedup
   - [ ] CEEMDAN (200 trials, 10k samples): **> 50x** speedup
   - [ ] Single-trial EMD: **> 10x** speedup
   - [ ] Speedup scales linearly with trial count (up to GPU limit)

2. **Latency**
   - [ ] Kernel launch overhead < 1ms
   - [ ] Data transfer (to GPU) < 5ms for 10k samples
   - [ ] Data transfer (from GPU) < 5ms for result
   - [ ] Trial batching overhead < 2% of total

3. **Memory**
   - [ ] Peak GPU memory < configured max (default 8 GB)
   - [ ] Memory fragmentation < 10% waste
   - [ ] No memory leaks over 1000+ decompositions
   - [ ] CPU↔GPU transfer buffer overhead < 5%

### Quality Requirements

1. **Testing**
   - [ ] 100% code coverage (excluding unsafe FFI)
   - [ ] Unit tests for device detection
   - [ ] Unit tests for memory pool
   - [ ] Integration tests for GPU decomposition
   - [ ] Cross-device tests (CUDA and ROCm)
   - [ ] Stress tests: 1000+ decompositions without leak
   - [ ] Parity tests: 50+ signals, tolerance 1e-5

2. **Documentation**
   - [ ] Inline comments for unsafe code
   - [ ] Rustdoc for all public APIs
   - [ ] Architecture design document (gpu-design.md)
   - [ ] Performance tuning guide
   - [ ] Multi-GPU setup guide
   - [ ] Troubleshooting guide (OOM, device errors)

3. **Reliability**
   - [ ] Graceful fallback to CPU on GPU error
   - [ ] Device reset on unrecoverable error
   - [ ] Memory cleanup in all code paths (RAII)
   - [ ] Error messages actionable (specific, recoverable)

---

## 5. Technical Design

### Module Structure

```rust
// New modules in crates/ferromode/src/adapters/

adapters/
├── mod.rs                      # Adapter layer facade
└── gpu/
    ├── mod.rs                  # Public GPU API
    ├── executor.rs             # GpuAdapter, device management
    ├── memory.rs               # GpuMemoryPool
    ├── kernels/                # GPU kernel wrappers
    │   ├── mod.rs
    │   ├── extrema.rs          # Extrema finding kernel
    │   ├── spline.rs           # Spline interpolation kernel
    │   ├── fft.rs              # FFT kernel (cufft/rocfft)
    │   └── ensemble.rs         # Ensemble iteration kernel
    ├── cuda/                   # CUDA-specific (feature-gated)
    │   ├── mod.rs
    │   ├── device.rs           # CUDA device mgmt
    │   ├── kernels.cu          # CUDA kernel code
    │   └── ffi.rs              # CUDA FFI bindings
    ├── rocm/                   # ROCm-specific (feature-gated)
    │   ├── mod.rs
    │   ├── device.rs           # ROCm device mgmt
    │   ├── kernels.cpp         # HIP kernel code
    │   └── ffi.rs              # ROCm FFI bindings
    └── cpu_fallback.rs         # CPU execution fallback

types.rs                        # Extend with GpuConfig, DeviceInfo
```

### Key Types

```rust
// GpuConfig — GPU-specific decomposition config
pub struct GpuConfig {
    pub base: EmdConfig,
    pub device_id: u32,
    pub max_memory_bytes: u64,    // e.g., 8 GB
    pub mixed_precision: bool,     // FP16 if available
    pub batch_size: usize,         // Trials per batch
}

// DeviceInfo — GPU device information
pub struct DeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub compute_capability: (u32, u32),  // CUDA: (major, minor)
    pub total_memory: u64,
    pub free_memory: u64,
    pub device_type: DeviceType,
}

pub enum DeviceType {
    CUDA,
    ROCm,
    Metal,
    WebGPU,
}

// GpuMemoryPool — pre-allocated GPU memory
pub struct GpuMemoryPool {
    device: u32,
    buffers: HashMap<String, *mut f64>,
    total_capacity: u64,
    allocated: u64,
    free_list: Vec<(u64, usize)>,  // (addr, size)
}

impl GpuMemoryPool {
    pub fn new(device: u32, capacity: u64) -> Result<Self> { ... }
    pub fn allocate(&mut self, size: usize) -> Result<*mut f64> { ... }
    pub fn deallocate(&mut self, ptr: *mut f64) -> Result<()> { ... }
    pub fn reset(&mut self) -> Result<()> { ... }
    pub fn stats(&self) -> MemoryStats { ... }
}

pub struct MemoryStats {
    pub allocated: u64,
    pub free: u64,
    pub fragmentation_pct: f64,
}

// GpuAdapter — main GPU interface
pub struct GpuAdapter {
    config: GpuConfig,
    device: u32,
    memory_pool: Arc<Mutex<GpuMemoryPool>>,
    stream: CudaStream,  // or HIP stream
}

impl GpuAdapter {
    pub fn new(config: GpuConfig) -> Result<Self> { ... }

    pub fn decompose_gpu(&self, signal: &Signal) -> Result<DecompositionResult> {
        // 1. Validate input on CPU
        // 2. Transfer signal to GPU
        // 3. Decompose using GPU kernels
        // 4. Transfer result back to CPU
        // 5. Return DecompositionResult
    }

    pub fn decompose_gpu_ensemble(
        &self,
        signal: &Signal,
        config: &EnsembleConfig,
        num_trials: usize,
    ) -> Result<DecompositionResult> {
        // 1. Batch trials into GPU batches
        // 2. Parallel execution across batches
        // 3. Aggregate results
        // 4. Return combined DecompositionResult
    }

    pub fn device_info(&self) -> Result<DeviceInfo> { ... }
    pub fn reset(&self) -> Result<()> { ... }
}

pub struct GpuKernelConfig {
    pub block_size: u32,
    pub grid_size: u32,
    pub shared_memory: u32,
    pub stream: CudaStream,
}
```

### Algorithm: decompose_gpu()

```
Input: signal (Signal of length N) on CPU
Config: GpuConfig, EmdConfig

1. Validate signal on CPU (no NaN/Inf)
2. Select device using device_id
3. Allocate GPU memory for signal, envelopes, IMFs
4. Transfer signal to GPU (H2D copy, async)

5. GPU Decomposition Loop:
   while not_converged:
     a) Find extrema using GPU kernel (local min/max)
     b) Interpolate upper/lower envelopes (cubic spline kernel)
     c) Compute mean envelope on GPU
     d) Compute residual = signal - mean
     e) Transfer residual back to CPU for HHT checks
     f) Apply CPU-side stopping criterion
     g) If stops sifting, transfer residual to GPU for next IMF

6. Extract results from GPU IMFs
7. Transfer residue to CPU

Output: DecompositionResult {
  imfs: Vec<Signal>,
  residue: Signal,
  num_imfs: usize
}
```

### Algorithm: decompose_gpu_ensemble()

```
Input: signal (length N), num_trials (e.g., 200)
Config: GpuConfig, EnsembleConfig

1. Determine batch size B (e.g., 10 trials per batch)
2. Compute total batches: ceil(num_trials / B)

3. Allocate GPU memory for B trials worth of noise-added signals

4. For each batch:
   a) Generate B noise arrays on GPU
   b) Add noise to signal in parallel (batch kernel)
   c) Decompose each of B signals in parallel
   d) Store results in pre-allocated GPU memory
   e) Asynchronously transfer batch results to CPU

5. Aggregate results on CPU:
   - Average all IMFs across trials
   - Compute residue as mean of trial residues

Output: DecompositionResult {
  imfs: Vec<Signal> (averaged),
  residue: Signal (averaged),
  num_imfs: usize
}
```

### GPU Kernel Signatures (CUDA)

```cuda
// Extrema finding kernel
__global__ void find_extrema_kernel(
    const float* signal,
    int length,
    uint8_t* extrema_type,  // 0=none, 1=min, 2=max
    int* extrema_indices,
    int* num_extrema
);

// Cubic spline interpolation kernel
__global__ void cubic_spline_kernel(
    const float* x_data,     // extrema x-coordinates
    const float* y_data,     // extrema y-values
    int num_extrema,
    const float* x_interp,   // interpolation points
    float* y_interp,         // output interpolated values
    int num_interp
);

// Mean envelope computation kernel
__global__ void compute_mean_envelope_kernel(
    const float* upper_envelope,
    const float* lower_envelope,
    int length,
    float* mean_envelope
);

// Ensemble noise addition kernel
__global__ void add_noise_kernel(
    const float* signal,
    float* noisy_signals,   // B × length
    const float* noise_arrays,  // B × length
    float std_noise,
    int length,
    int batch_size
);
```

### Memory Layout

```
GPU Memory Pool (total: 8 GB)
├── Signal buffer (1 GB, reused)
├── Upper envelope (1 GB, reused)
├── Lower envelope (1 GB, reused)
├── Mean envelope (1 GB, reused)
├── Extrema markers (100 MB, reused)
├── Ensemble noise buffers (2 GB, batch-sized)
└── Free pool (remaining)
```

### CUDA Kernel Optimization

```cuda
// Example: extrema finding (optimized with shared memory)
__global__ void find_extrema_kernel(
    const float* signal,
    int length,
    uint8_t* extrema_type
) {
    // Each thread block processes a chunk
    // Uses shared memory reduction for efficiency
    // Coalescent memory access patterns
    // Minimizes global memory traffic
}
```

---

## 6. File Structure Changes

### New Files

```
crates/ferromode/src/
├── adapters/
│   └── gpu/
│       ├── mod.rs (new)
│       ├── executor.rs (new)
│       ├── memory.rs (new)
│       ├── kernels/
│       │   ├── mod.rs (new)
│       │   ├── extrema.rs (new)
│       │   ├── spline.rs (new)
│       │   ├── fft.rs (new)
│       │   └── ensemble.rs (new)
│       ├── cuda/ (new, feature-gated)
│       │   ├── mod.rs (new)
│       │   ├── device.rs (new)
│       │   ├── kernels.cu (new, CUDA kernel code)
│       │   ├── kernels.cuh (new, CUDA header)
│       │   └── ffi.rs (new, CUDA FFI)
│       └── cpu_fallback.rs (new)

crates/ferromode-cuda/            # NEW: Optional CUDA crate
├── Cargo.toml
├── src/
│   └── lib.rs
├── kernels/
│   ├── mod.rs
│   ├── extrema.cu
│   └── ensemble.cu
└── build.rs                       # CUDA compilation

tests/
├── gpu_vs_cpu_parity.rs (new)
├── gpu_speedup_bench.rs (new)
├── gpu_memory_profile.rs (new)
├── gpu_multi_device.rs (new)
└── gpu_mixed_precision.rs (new)

crates/ferromode-py/
├── python/ferromode_py/
│   └── gpu.pyi (new)
└── tests/
    └── test_gpu_cupy.py (new)

docs/
└── gpu_design.md (new)
```

### Modified Files

```
crates/ferromode/src/
├── lib.rs (add GPU adapter exports, feature-gated)
├── types.rs (extend with GpuConfig, DeviceInfo)
└── api.rs (add GPU decomposition APIs)

Cargo.toml (add features: ["gpu", "gpu-cuda", "gpu-rocm"])
build.rs (GPU kernel compilation detection)

crates/ferromode-py/src/
└── lib.rs (add PyO3 bindings for GpuAdapter, optional)
```

---

## 7. Test Strategy

### Unit Tests (30 tests)

1. **Device Detection** (6 tests)
   - [ ] cuda_available() works
   - [ ] rocm_available() works
   - [ ] device_count() returns valid count
   - [ ] device_name() returns string
   - [ ] device_memory() queries valid sizes
   - [ ] select_device() validates device_id

2. **GPU Memory Pool** (10 tests)
   - [ ] new() creates valid pool
   - [ ] allocate() returns valid pointers
   - [ ] allocate() grows free list correctly
   - [ ] deallocate() returns memory to pool
   - [ ] deallocate() merges adjacent free blocks
   - [ ] reset() clears all allocations
   - [ ] stats() reports fragmentation
   - [ ] OOM detection on full pool
   - [ ] thread-safety (concurrent allocate/deallocate)
   - [ ] edge case: allocate then deallocate same size

3. **GPU Kernels** (8 tests)
   - [ ] extrema finding kernel correctness
   - [ ] spline interpolation kernel correctness
   - [ ] FFT kernel matches CPU FFT
   - [ ] ensemble noise kernel adds noise correctly
   - [ ] mean envelope kernel computation
   - [ ] kernel error handling (invalid launch)
   - [ ] asynchronous kernel launch
   - [ ] memory coalescing patterns

4. **GpuAdapter** (6 tests)
   - [ ] new() initializes device
   - [ ] device_info() returns correct info
   - [ ] decompose_gpu() on single signal
   - [ ] mixed_precision flag works
   - [ ] CPU fallback on GPU error
   - [ ] reset() clears state

### Integration Tests (25 tests)

1. **GPU Decomposition** (10 tests)
   - [ ] decompose_gpu() basic EMD on GPU
   - [ ] Single chunk equals CPU result (within tolerance)
   - [ ] Mixed precision vs FP32 accuracy
   - [ ] Large signal (100k samples)
   - [ ] Edge cases: very short (10 samples), very long (1M samples)
   - [ ] Batch vs individual decomposition
   - [ ] Device switching between trials
   - [ ] Memory reuse across decompositions
   - [ ] Determinism (same seed = same result)
   - [ ] Error recovery (invalid input, device error)

2. **Ensemble Parallelization** (8 tests)
   - [ ] decompose_gpu_ensemble() 10 trials
   - [ ] decompose_gpu_ensemble() 200 trials (EEMD)
   - [ ] CEEMDAN on GPU vs CPU (same results)
   - [ ] Batch size variation (1, 5, 10, 50)
   - [ ] Trial-level error handling
   - [ ] Progress tracking over long runs
   - [ ] Cancellation support
   - [ ] Memory efficiency for large trial counts

3. **CPU-GPU Parity** (7 tests)
   - [ ] Parity on 50+ reference signals
   - [ ] Tolerance: L∞ < 1e-5
   - [ ] Edge cases: nearly-zero IMFs
   - [ ] Very small values (denormals)
   - [ ] Very large values (near max FP32)
   - [ ] Reproducibility across runs
   - [ ] Cross-device consistency (CUDA vs ROCm)

### Performance Tests (criterion.rs)

```rust
// Benchmarks
#[bench]
fn bench_decompose_gpu_single(b: &mut Bencher) {
    // Measure latency for single EMD on GPU
    // Assert > 10x speedup vs CPU
}

#[bench]
fn bench_decompose_gpu_eemd_200(b: &mut Bencher) {
    // Measure EEMD 200 trials on 10k samples
    // Assert > 50x speedup vs CPU
}

#[bench]
fn bench_decompose_gpu_ceemdan_200(b: &mut Bencher) {
    // Measure CEEMDAN 200 trials
    // Assert > 50x speedup vs CPU
}

#[bench]
fn bench_memory_transfer_overhead(b: &mut Bencher) {
    // Measure H2D + D2H transfer time
    // Assert < 10ms for 10k samples
}

#[bench]
fn bench_kernel_launch_overhead(b: &mut Bencher) {
    // Measure kernel launch + sync time
    // Assert < 1ms per kernel
}
```

### Stress Tests (5 tests)

1. **Long-Running**
   - [ ] 1000+ decompositions without memory leak
   - [ ] 1-hour continuous GPU execution
   - [ ] Device memory stable (no growth)

2. **Edge Cases**
   - [ ] Device reset under load
   - [ ] OOM fallback to CPU
   - [ ] Multi-device load balancing

### Cross-Device Tests (10 tests, if available)

1. **CUDA Tests** (5 tests, if CUDA available)
   - [ ] Device detection on CUDA
   - [ ] Decomposition on CUDA device
   - [ ] Memory management on CUDA
   - [ ] Multi-GPU on CUDA
   - [ ] Mixed precision on CUDA

2. **ROCm Tests** (5 tests, if ROCm available)
   - [ ] Device detection on ROCm
   - [ ] Decomposition on ROCm device
   - [ ] Memory management on ROCm
   - [ ] Multi-GPU on ROCm
   - [ ] Mixed precision on ROCm

---

## 8. Hardware Requirements

### NVIDIA CUDA (Primary)

- **Driver:** >= 480.06 (for CUDA 11.0 support)
- **Compute Capability:** >= 3.0 (Kepler, Maxwell, Pascal, Turing, Ampere, Hopper)
- **Memory:** >= 4 GB for testing, 8+ GB recommended for production
- **Examples:** RTX 3090, RTX 4090, A100, H100, Tesla K80, V100

### AMD ROCm (Secondary)

- **Driver:** rocm-driver >= 50000 (ROCm 5.0)
- **GPU:** GCN 2nd gen or newer (MI100, MI200, Radeon VII)
- **Memory:** >= 4 GB for testing, 16+ GB for production
- **Examples:** MI100, MI200, Radeon VII, RX 6800

### Validation Hardware

For CI/CD, maintain testing on:
- 1× NVIDIA CUDA device (e.g., RTX 3080 Ti or Tesla T4)
- 1× AMD ROCm device (if available, otherwise skip ROCm tests)
- CPU fallback always available

---

## 9. Implementation Tasks (Ordered)

### Phase 1: Foundation (Week 1)

- [ ] **T-278:** Create `adapters/gpu/` module structure
- [ ] **T-279:** Implement DeviceInfo struct and device queries
- [ ] **T-280:** Design GpuMemoryPool and memory allocation strategy
- [ ] **T-281:** Implement GpuMemoryPool with unit tests
- [ ] **T-282:** Create CUDA/ROCm feature flags and build.rs
- [ ] **T-283:** Write device detection tests (6 tests)

### Phase 2: GPU Kernels (Week 2)

- [ ] **T-284:** Write extrema finding kernel (CUDA)
- [ ] **T-285:** Write cubic spline interpolation kernel (CUDA)
- [ ] **T-286:** Write FFT wrapper kernel (using cufft)
- [ ] **T-287:** Write ensemble noise addition kernel (CUDA)
- [ ] **T-288:** Write mean envelope computation kernel (CUDA)
- [ ] **T-289:** Optimize kernels for coalesced memory access
- [ ] **T-290:** Write kernel unit tests (8 tests)
- [ ] **T-291:** Implement ROCm versions (HIP) if resources available

### Phase 3: GPU Adapter (Week 2.5)

- [ ] **T-292:** Implement GpuAdapter struct
- [ ] **T-293:** Implement decompose_gpu() method
- [ ] **T-294:** Implement decompose_gpu_ensemble() method
- [ ] **T-295:** Add CPU fallback on GPU error
- [ ] **T-296:** Mixed precision support (FP16 if available)
- [ ] **T-297:** Write GpuAdapter unit tests (6 tests)

### Phase 4: GPU Decomposition Integration (Week 3)

- [ ] **T-298:** Integrate kernels into decompose_gpu()
- [ ] **T-299:** Implement H2D and D2H transfers
- [ ] **T-300:** Test single decomposition on GPU
- [ ] **T-301:** Integration tests for GPU decomposition (10 tests)
- [ ] **T-302:** Verify correctness vs CPU baseline

### Phase 5: Ensemble Parallelization (Week 3.5)

- [ ] **T-303:** Implement batch trial execution strategy
- [ ] **T-304:** Implement trial distribution across GPUs
- [ ] **T-305:** Implement result aggregation
- [ ] **T-306:** Add progress tracking for ensembles
- [ ] **T-307:** Add cancellation support
- [ ] **T-308:** Integration tests for ensembles (8 tests)

### Phase 6: Parity & Performance (Week 4)

- [ ] **T-309:** Create GPU vs CPU parity test suite
- [ ] **T-310:** Run on 50+ reference signals
- [ ] **T-311:** Benchmark speedup targets (50x for EEMD)
- [ ] **T-312:** Profile memory usage (peak, fragmentation)
- [ ] **T-313:** Cross-device testing (CUDA + ROCm if available)
- [ ] **T-314:** Create performance comparison report

### Phase 7: Multi-GPU & Mixed Precision (Week 4.5)

- [ ] **T-315:** Implement multi-GPU device selection
- [ ] **T-316:** Test trial distribution across GPUs
- [ ] **T-317:** Implement mixed precision support
- [ ] **T-318:** Benchmark mixed precision vs FP32
- [ ] **T-319:** Document accuracy trade-offs

### Phase 8: Stress Testing & Robustness (Week 5)

- [ ] **T-320:** Implement long-running stress tests (1000+ decompositions)
- [ ] **T-321:** Memory leak detection tests
- [ ] **T-322:** Device reset and error recovery tests
- [ ] **T-323:** OOM fallback testing
- [ ] **T-324:** Edge case testing (short/long signals, NaN/Inf)

### Phase 9: Python Bindings & Documentation (Week 5.5)

- [ ] **T-325:** Add PyO3 wrappers for GpuAdapter (optional)
- [ ] **T-326:** Create Python type stubs (gpu.pyi)
- [ ] **T-327:** Write GPU user guide
- [ ] **T-328:** Create performance tuning guide
- [ ] **T-329:** Write troubleshooting guide (OOM, device errors)

### Phase 10: Code Review & Final Validation (Week 6)

- [ ] **T-330:** Code review of GPU kernels and memory management
- [ ] **T-331:** Performance regression testing
- [ ] **T-332:** Cross-platform validation (CUDA + ROCm)
- [ ] **T-333:** Documentation review
- [ ] **T-334:** Final benchmarking and sign-off

---

## 10. Exit Criteria

**All of the following must be true for v2.1 to ship:**

1. ✓ GpuAdapter, device detection, memory pool fully implemented
2. ✓ All GPU kernels (extrema, spline, FFT, ensemble) implemented
3. ✓ All 65 tests passing (30 unit + 25 integration + 10 stress)
4. ✓ GPU vs CPU parity verified (tolerance 1e-5) on 50+ signals
5. ✓ Speedup SLA met (> 50x for EEMD 200 trials on 10k samples)
6. ✓ Memory SLA met (peak < 8 GB, no leaks over 1000+ decompositions)
7. ✓ Cross-device validation passing (CUDA and ROCm if available)
8. ✓ Mixed precision support working (FP16 if available)
9. ✓ Multi-GPU support implemented and tested
10. ✓ CPU fallback working for all error cases
11. ✓ Python bindings complete and tested (optional)
12. ✓ All public APIs documented (Rustdoc + user guide)
13. ✓ Performance benchmarks documented
14. ✓ No performance regressions vs v1.x CPU
15. ✓ Memory leak tests passing
16. ✓ Code review approved by 2+ maintainers
17. ✓ Security audit passed (unsafe code audit, bounds checking)

---

## 11. Dependencies

### Rust Crates
- **cuda** or **hip** (feature-gated) — GPU kernel compilation
- **rustacuda** (optional) — CUDA FFI wrapper
- **criterion** (dev) — benchmarking
- **proptest** (dev) — property-based tests

### External
- **CUDA Toolkit** (11.8+) — if compiling with `gpu-cuda` feature
- **ROCm** (5.0+) — if compiling with `gpu-rocm` feature
- **cuFFT** — FFT kernel (included in CUDA Toolkit)
- **rocFFT** — FFT kernel (included in ROCm)

### Build Tools
- **NVCC** (NVIDIA) — CUDA C compiler
- **hipcc** (AMD) — HIP compiler
- **CMake** (optional) — for kernel compilation

---

## 12. Feature Flags

```toml
[features]
default = []
gpu = ["gpu-cuda"]  # Enable GPU support (CUDA by default)
gpu-cuda = []       # CUDA kernel compilation
gpu-rocm = []       # ROCm kernel compilation (mutually exclusive with CUDA)
gpu-mixed-precision = []  # FP16 support (requires Tensor Core or CDNA)
```

### Conditional Compilation

```rust
#[cfg(feature = "gpu-cuda")]
mod cuda { ... }

#[cfg(feature = "gpu-rocm")]
mod rocm { ... }

#[cfg(all(feature = "gpu", feature = "gpu-mixed-precision"))]
mod mixed_precision { ... }
```

---

## 13. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| GPU kernel bugs cause divergence | Medium | High | Extensive parity tests; bit-level checking |
| CUDA/ROCm API breaking changes | Medium | Medium | Wrapper traits + vendor version pinning in CI |
| Speedup < 50x target | Low | Medium | Early benchmarking; kernel optimization |
| Memory leaks in GPU code | Low | High | Valgrind-style tools; RAII + tests |
| Device compatibility issues | Medium | Medium | Support multiple architectures; fallback to CPU |
| Multi-GPU synchronization bugs | Low | Medium | Careful thread/stream synchronization; tests |
| Mixed precision accuracy loss | Low | Low | Quantify and document accuracy trade-offs |

---

## 14. Performance Targets & Trade-Offs

### Speedup by Operation

| Operation | Target | Achieved | Notes |
|-----------|--------|----------|-------|
| **Extrema Finding** | 50x | TBD | Highly parallelizable |
| **Spline Interpolation** | 20x | TBD | Memory-bound |
| **FFT** | 30x | TBD | Uses vendor library (cufft) |
| **Ensemble Iteration** | 100x | TBD | Embarrassingly parallel |
| **EEMD (200 trials)** | 50x | TBD | Overall speedup |
| **CEEMDAN (200 trials)** | 50x | TBD | Overall speedup |

### Memory Trade-Offs

| Feature | Memory Cost | Trade-Off |
|---------|-------------|-----------|
| Pre-allocated pool | 8 GB | Predictable latency |
| Batch trials (size 10) | +100 MB | 10x throughput gain |
| Mixed precision (FP16) | -50% | ±0.1% accuracy loss |
| Streaming (GPU) | +200 MB | Continuous processing |

---

## 15. Rollout Plan

### Phase 1: Limited Beta (Week 7)
- Release v2.1-beta1 with `gpu` feature flag
- Limited to CUDA, single-GPU
- Internal testing only

### Phase 2: Extended Beta (Week 7.5)
- v2.1-beta2 with ROCm support (if resources allow)
- Multi-GPU support
- Gather feedback from community GPU users

### Phase 3: Production Release (Week 8)
- Address beta feedback
- Final performance tuning
- Tag v2.1 stable
- Document hardware requirements

---

## 16. Success Metrics

After v2.1 ships, measure:

1. **Adoption:** # of users leveraging GPU decomposition
2. **Performance:** Actual speedup in production (target: > 50x)
3. **Reliability:** Uptime in GPU mode (target: 99.99%)
4. **Correctness:** Cross-validation with CPU
5. **Maintainability:** Time to onboard GPU features

---

## 17. References

- [ARD_UPDATED_v2x.md](../next_work/ARD_UPDATED_v2x.md) — System architecture
- [CUDA Toolkit Docs](https://docs.nvidia.com/cuda/) — NVIDIA GPU programming
- [ROCm Docs](https://rocmdocs.amd.com/) — AMD GPU programming
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Kernel optimization
- Existing v1.x CPU implementation (for algorithm reference)
- Research: GPU-accelerated signal processing papers

---

*Document Version: 1.0*  
*Last Updated: April 8, 2026*  
*Status: Ready for Implementation*  
