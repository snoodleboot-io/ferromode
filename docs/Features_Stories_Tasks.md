# Features, Stories & Tasks
## Ferromode Project

---

## EPIC 1: Rust Core Foundation

### Feature 1.1: Core Infrastructure & Types

**Description:** Establish the foundational Rust workspace, type system, error handling, and spline engine that all algorithms depend on.

---

#### Story 1.1.1: Workspace & Project Scaffolding
*As a contributor, I need a correctly configured Rust workspace so I can build and test all crates together.*

**Tasks:**
- [ ] **T-001** Initialize Cargo workspace with `ferromode`, `ferromode-py`, `ferromode-r`, `ferromode-wasm` crates
- [ ] **T-002** Configure `Cargo.toml` with shared dependency versions (ndarray 0.15, rayon 1.8, rustfft 6, rand 0.8, serde 1, thiserror 1)
- [ ] **T-003** Set up `.github/workflows/ci.yml` with matrix build (Linux, macOS, Windows × stable, beta)
- [ ] **T-004** Configure `rustfmt.toml` and `clippy.toml` with project-wide lint rules
- [ ] **T-005** Create `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE` (Apache-2.0)
- [ ] **T-006** Set up `criterion` benchmarking harness in `ferromode/benches/`
- [ ] **T-007** Create initial `README.md` with project overview and quick-start

**Acceptance Criteria:** `cargo build --workspace` succeeds on all three platforms; CI is green.

---

#### Story 1.1.2: Core Type System
*As a developer, I need well-defined types for signals, IMF collections, and results so the API is type-safe and ergonomic.*

**Tasks:**
- [ ] **T-008** Define `Signal` struct: `Vec<f64>` values + optional sample rate, with `from_slice`, `len`, `iter` methods
- [ ] **T-009** Define `MultivariateSignal` struct: `Vec<Vec<f64>>` channels with dimension validation
- [ ] **T-010** Define `ImfCollection` struct with `imfs: Vec<Vec<f64>>`, `residue: Vec<f64>`, `reconstruct()`, `orthogonality_index()` methods
- [ ] **T-011** Define `DecompositionResult` with algorithm metadata, elapsed time, config snapshot
- [ ] **T-012** Define `HilbertResult` struct with instantaneous amplitude, frequency, marginal spectrum fields
- [ ] **T-013** Define `AlgorithmType` enum (EMD, EEMD, CEEMD, CEEMDAN, ICEEMDAN, MEMD, NAMEMD, VMD)
- [ ] **T-014** Derive `serde::Serialize/Deserialize` on all result types; test JSON round-trip
- [ ] **T-015** Write unit tests for reconstruction error, orthogonality index

**Acceptance Criteria:** All types compile; `ImfCollection::reconstruct()` returns signal within 1e-12 of original for lossless algorithms.

---

#### Story 1.1.3: Error Handling
*As a user, I need clear, actionable error messages when inputs are invalid.*

**Tasks:**
- [ ] **T-016** Define `EmdError` enum with thiserror: `EmptySignal`, `InsufficientExtrema`, `InvalidConfig`, `NumericalFailure`, `DimensionMismatch`, `InvalidBoundary`
- [ ] **T-017** Implement `From<EmdError>` conversions for each binding's native error type
- [ ] **T-018** Add input validation at all public API entry points (check NaN, Inf, zero length, dimension mismatches)
- [ ] **T-019** Write tests for all error paths

---

#### Story 1.1.4: Cubic Spline Engine
*As a developer, I need a robust cubic spline interpolation that handles edge cases for envelope computation.*

**Tasks:**
- [ ] **T-020** Implement natural cubic spline from extrema knots: tridiagonal system solver (Thomas algorithm)
- [ ] **T-021** Implement periodic cubic spline (required for Zeng-He boundary condition)
- [ ] **T-022** Implement not-a-knot cubic spline variant
- [ ] **T-023** Validate against reference (scipy's CubicSpline) on test cases: uniform knots, non-uniform knots, single/double extrema edge cases
- [ ] **T-024** Benchmark spline vs. naive O(n²) approach; ensure O(n) Thomas algorithm is used
- [ ] **T-025** Handle degenerate cases: fewer than 2 knots, duplicate knot locations

**Acceptance Criteria:** Spline values match scipy within 1e-10 on all reference test signals.

---

### Feature 1.2: Boundary Condition Strategies

**Description:** Implement the full suite of boundary condition strategies as a trait-based plugin system.

---

#### Story 1.2.1: Boundary Trait & Registry
*As a developer, I need a trait abstraction for boundary conditions so strategies are interchangeable.*

**Tasks:**
- [ ] **T-026** Define `BoundaryCondition` trait with `extend(signal: &[f64], extrema: &Extrema) -> ExtendedSignal`
- [ ] **T-027** Define `BoundaryCondition` enum and `get_strategy()` factory function
- [ ] **T-028** Write test harness that applies each strategy and verifies: (1) extended signal contains original, (2) spline fit is smooth, (3) no NaN/Inf produced

---

#### Story 1.2.2: Characteristic Wave Extension
*Reference: Huang et al. 1998*

**Tasks:**
- [ ] **T-029** Implement `CharacteristicWave` strategy: identify two nearest extrema at each boundary, construct implicit wave, append 4 copies per end
- [ ] **T-030** Handle edge case: signal has fewer than 4 extrema total
- [ ] **T-031** Test on: pure sine, chirp, noisy signal with envelope modulation

---

#### Story 1.2.3: Mirror / Symmetric Extension

**Tasks:**
- [ ] **T-032** Implement even-extension mirror strategy
- [ ] **T-033** Implement odd-extension mirror strategy
- [ ] **T-034** Test both variants; verify that endpoint is a local extremum in extended signal

---

#### Story 1.2.4: Periodic / Cyclic Extension (Zeng & He 2004)
*Reference: K. Zeng & M.-X. He, IEEE IGARSS 2004*

**Tasks:**
- [ ] **T-035** Implement periodic extension: concatenate even-extended and odd-extended copies to build periodic series
- [ ] **T-036** Use periodic cubic spline BC for envelope computation on the periodic extended signal
- [ ] **T-037** Test on signals with known periodic structure; verify no endpoint artifacts
- [ ] **T-038** Add citation comment in source code pointing to Zeng & He 2004

---

#### Story 1.2.5: Slope-Based Extension

**Tasks:**
- [ ] **T-039** Compute first-derivative at each endpoint using finite differences
- [ ] **T-040** Extrapolate linearly beyond each boundary to generate artificial extrema
- [ ] **T-041** Test on signals with monotone ends

---

#### Story 1.2.6: AR Model Extension

**Tasks:**
- [ ] **T-042** Implement Yule-Walker AR coefficient estimation (order p, default p=5)
- [ ] **T-043** Forecast N samples beyond right boundary; backcast N samples before left boundary
- [ ] **T-044** Expose `ar_order` as a configuration parameter
- [ ] **T-045** Test on AR(2) synthetic signals; verify near-perfect extension

---

#### Story 1.2.7: Waveform Matching Extension

**Tasks:**
- [ ] **T-046** Implement cross-correlation search: find interior segment most similar to each endpoint region
- [ ] **T-047** Append matched segment beyond boundary
- [ ] **T-048** Expose `match_length` (in samples) as configuration parameter
- [ ] **T-049** Test on quasi-periodic signals; compare against mirror extension

---

### Feature 1.3: Core EMD Algorithm

**Description:** Implement the foundational EMD sifting process and basic EMD decomposition.

---

#### Story 1.3.1: Extrema Detection
*As a developer, I need reliable local extrema detection that handles plateaus and noisy signals.*

**Tasks:**
- [ ] **T-050** Implement local maxima detection: find indices where `x[i] > x[i-1] && x[i] > x[i+1]`
- [ ] **T-051** Implement local minima detection
- [ ] **T-052** Handle plateau extrema: detect flat tops/bottoms, use midpoint index
- [ ] **T-053** Handle boundary as potential extremum (needed for some boundary strategies)
- [ ] **T-054** Test on: sine wave, sawtooth, step function, constant signal

---

#### Story 1.3.2: Sifting Engine
*As a developer, I need the core sifting loop that extracts one IMF from a signal.*

**Tasks:**
- [ ] **T-055** Implement `sift_one(signal, config) -> (imf, residue)` function
- [ ] **T-056** Implement SD threshold stopping criterion: `SD = Σ|h_{k-1} - h_k|² / Σ|h_{k-1}|² < threshold`
- [ ] **T-057** Implement S-number criterion: count consecutive siftings where #extrema and #zero-crossings are equal or differ by 1
- [ ] **T-058** Implement fixed-iteration stopping
- [ ] **T-059** Implement energy-difference stopping
- [ ] **T-060** Add sifting iteration counter and max_sifting_iterations guard to prevent infinite loops

---

#### Story 1.3.3: Basic EMD
*Reference: Huang et al. 1998*

**Tasks:**
- [ ] **T-061** Implement `emd(signal, config) -> DecompositionResult` top-level function
- [ ] **T-062** Implement outer loop: repeatedly call `sift_one` on residue until residue has < 2 extrema or max_imfs reached
- [ ] **T-063** Implement `max_imfs` limit
- [ ] **T-064** Implement intermittency test option (frequency-range filter on extrema spacing)
- [ ] **T-065** Validate reconstruction: `Σ IMFs + residue = original signal` within 1e-12
- [ ] **T-066** Write reference tests using Huang's original test signals (sunspot data, EEG)

---

### Feature 1.4: Ensemble EMD Family

---

#### Story 1.4.1: EEMD
*Reference: Wu & Huang 2009*

**Tasks:**
- [ ] **T-067** Implement `eemd(signal, ensemble_config)`: add Gaussian white noise (std = noise_std × signal_std), run EMD per trial
- [ ] **T-068** Implement parallel trial execution via `rayon::par_iter`
- [ ] **T-069** Implement seeded RNG for reproducibility (`rand::SeedableRng`)
- [ ] **T-070** Average IMFs across trials (handle unequal IMF counts by zero-padding)
- [ ] **T-071** Validate: noise cancels in average; test on mode-mixed synthetic signal

---

#### Story 1.4.2: CEEMD
*Reference: Yeh, Shieh & Huang 2010*

**Tasks:**
- [ ] **T-072** Implement paired noise trials: for each trial i, compute `emd(signal + ε·noise_i)` and `emd(signal - ε·noise_i)`
- [ ] **T-073** Average complementary pairs; verify exact reconstruction
- [ ] **T-074** Test that RMS noise level in IMFs matches EEMD

---

#### Story 1.4.3: CEEMDAN
*Reference: Torres et al. 2011*

**Tasks:**
- [ ] **T-075** Implement stage-wise CEEMDAN: at stage k, add adaptive noise to current residue
- [ ] **T-076** Compute adaptive noise as: `ε_k = ε · std(residue_k) / std(noise)`
- [ ] **T-077** Extract first IMF of `emd(residue_k + ε_k·noise)` for each trial; average to get mean_IMF_k
- [ ] **T-078** Compute next residue: `r_{k+1} = r_k - mean_IMF_k`
- [ ] **T-079** Parallelize trial computation at each stage
- [ ] **T-080** Validate reconstruction error < 1e-10

---

#### Story 1.4.4: ICEEMDAN
*Reference: Colominas, Schlotthauer & Torres 2014*

**Tasks:**
- [ ] **T-081** Implement improved noise model: use EMD of noise-only signal at each stage
- [ ] **T-082** Compute `mean_IMF_k = E[first_IMF(r_k + ε_k · IMF_k_of_noise)]`
- [ ] **T-083** Test that residual noise in IMFs is further reduced vs. CEEMDAN

---

### Feature 1.5: Multivariate EMD

---

#### Story 1.5.1: Hypersphere Direction Sampling
*Reference: Rehman & Mandic 2010*

**Tasks:**
- [ ] **T-084** Implement uniform angular sampling on n-sphere for n=2,3,4,6,8 channels
- [ ] **T-085** Implement Hammersley low-discrepancy sequence sampling
- [ ] **T-086** Implement Halton sequence sampling
- [ ] **T-087** Expose `DirectionSampling` enum: `Uniform`, `Hammersley`, `Halton`
- [ ] **T-088** Validate distribution uniformity with statistical test (Kolmogorov-Smirnov on projection angles)

---

#### Story 1.5.2: MEMD
*Reference: Rehman & Mandic 2010*

**Tasks:**
- [ ] **T-089** Implement projection of n-variate signal onto each direction vector
- [ ] **T-090** Find extrema of each projection; interpolate component-wise to form n-D envelopes
- [ ] **T-091** Average envelopes over all directions to compute local mean
- [ ] **T-092** Implement multivariate sifting loop using multivariate stopping criterion
- [ ] **T-093** Test mode-alignment property: decompose synthetic hexavariate signal with known scales; verify IMFs align across channels

---

#### Story 1.5.3: NA-MEMD
*Reference: Rehman & Mandic 2011*

**Tasks:**
- [ ] **T-094** Implement NA-MEMD: add `n_noise_channels` of white noise as extra channels
- [ ] **T-095** Run MEMD on augmented signal; discard noise channel IMFs post-decomposition
- [ ] **T-096** Test noise suppression vs. plain MEMD on bivariate test signal

---

### Feature 1.6: Hilbert Transform & Post-Processing

---

#### Story 1.6.1: Hilbert Transform
**Tasks:**
- [ ] **T-097** Implement FFT-based Hilbert transform: FFT → zero negative frequencies → IFFT
- [ ] **T-098** Compute analytic signal: `z(t) = x(t) + i·H{x}(t)`
- [ ] **T-099** Compute instantaneous amplitude: `A(t) = |z(t)|`
- [ ] **T-100** Compute instantaneous phase: `φ(t) = arctan(H{x}/x)`
- [ ] **T-101** Compute instantaneous frequency: `f(t) = (1/2π) · dφ/dt`
- [ ] **T-102** Validate against known analytic signals (pure tone, AM signal)

---

#### Story 1.6.2: Marginal Spectrum & Metrics
**Tasks:**
- [ ] **T-103** Implement Hilbert marginal spectrum (time-integrated instantaneous energy at each frequency)
- [ ] **T-104** Implement IMF orthogonality index
- [ ] **T-105** Implement degree of stationarity metric
- [ ] **T-106** Implement IMF energy ratio per component

---

### Feature 1.7: VMD Implementation

**Description:** Implement Variational Mode Decomposition as a reference/comparison method.
*Reference: Dragomiretskiy & Zosso 2014*

---

#### Story 1.7.1: VMD Core
**Tasks:**
- [ ] **T-107** Implement VMD: alternating direction method of multipliers (ADMM) in frequency domain
- [ ] **T-108** Implement center frequency update step
- [ ] **T-109** Implement mode update step (Wiener filter)
- [ ] **T-110** Implement dual variable (Lagrange multiplier) update
- [ ] **T-111** Expose `n_modes` (K), `alpha` (bandwidth penalty), `tau` (noise tolerance), `tol` (convergence) parameters
- [ ] **T-112** Validate against MATLAB reference output from Dragomiretskiy & Zosso

---

## EPIC 2: Python Binding (ferromode-py)

**Binding contract:** This crate contains zero algorithmic logic. Every function body is: (1) convert inputs from NumPy/Python types to Rust slices, (2) call the corresponding `ferromode::api` function, (3) convert the result back to Python types.

### Feature 2.1: PyO3 Marshalling Layer

**Description:** Build a PyO3 extension module that exposes the Rust core to Python via type conversion only. The Python package `ferromode-py` is the compiled `.so`/`.pyd` extension plus a minimal `__init__.py` that re-exports and adds type stubs.

---

#### Story 2.1.1: PyO3 Extension Module Setup
*As a Python user, I can `pip install ferromode-py` and call `ferromode_py.emd(array, ...)` with the same semantics as the Rust API.*

**Tasks:**
- [ ] **T-113** Configure PyO3 dependency and `maturin` build in `ferromode-py/Cargo.toml`; create `#[pymodule] fn ferromode_py(m: &PyModule)` entry point
- [ ] **T-114** Expose `EmdConfig` as a `#[pyclass]` with `#[new]` accepting keyword arguments; fields mirror Rust struct exactly — no defaults computed here, defaults live in Rust
- [ ] **T-115** Expose `EnsembleConfig`, `MemdConfig` as `#[pyclass]` wrappers — same principle
- [ ] **T-116** Expose `BoundaryCondition` and `StoppingCriterion` as `#[pyclass]` enums
- [ ] **T-117** Expose `AlgorithmType` as a `#[pyclass]` enum for result inspection

#### Story 2.1.2: Array Marshalling — Input
*As a developer, inputs from NumPy must reach Rust with zero copy where the memory layout allows it.*

**Tasks:**
- [ ] **T-118** Implement `numpy_to_slice(arr: PyReadonlyArray1<f64>) -> &[f64]` — zero-copy borrow if C-contiguous; copy + warn if not
- [ ] **T-119** Implement `numpy2d_to_vecs(arr: PyReadonlyArray2<f64>) -> Vec<Vec<f64>>` for multivariate input (MEMD/NA-MEMD)
- [ ] **T-120** Validate input at marshalling boundary: reject non-finite values, zero-length arrays, wrong dtype — raise `ValueError` with message forwarded from `EmdError`

#### Story 2.1.3: Result Marshalling — Output
*As a Python user, results are returned as objects whose `.imfs` and `.residue` are `np.ndarray` without unnecessary copies.*

**Tasks:**
- [ ] **T-121** Implement `ImfCollectionPy` `#[pyclass]`: wraps `ImfCollection`; `.imfs` property returns `PyArray2<f64>` (n_imfs × n_samples); `.residue` returns `PyArray1<f64>`; `.reconstruct()` calls `ImfCollection::reconstruct()` in Rust and returns `PyArray1<f64>`
- [ ] **T-122** Implement `HilbertResultPy` `#[pyclass]`: wraps `HilbertResult`; `.instantaneous_amplitude`, `.instantaneous_frequency`, `.marginal_spectrum` as numpy arrays
- [ ] **T-123** Implement `DecompositionResultPy` `#[pyclass]`: exposes `.algorithm`, `.elapsed_ms`, `.imfs` (→ `ImfCollectionPy`), `.hilbert()` (→ `HilbertResultPy`)

#### Story 2.1.4: Function Wrappers
*As a Python user, I call `ferromode_py.emd(signal, config)` and receive a `DecompositionResult`.*

**Tasks:**
- [ ] **T-124** `#[pyfunction] fn emd(signal: PyReadonlyArray1<f64>, config: &EmdConfigPy) -> PyResult<DecompositionResultPy>` — marshal in, call `ferromode::api::emd()`, marshal out
- [ ] **T-125** Same for `eemd`, `ceemd`, `ceemdan`, `iceemdan` (all accepting `EnsembleConfigPy`)
- [ ] **T-126** Same for `memd`, `namemd` (accepting `MemdConfigPy` + `PyReadonlyArray2<f64>`)
- [ ] **T-127** Same for `vmd` (accepting `VmdConfigPy`)
- [ ] **T-128** Wrap all Rust `Result::Err` variants into typed Python exceptions: `EmdError` → `ferromode_py.EmdError(ValueError)` with the original message; no information lost
- [ ] **T-129** Release GIL inside all ensemble method wrappers: `py.allow_threads(|| ferromode::api::ceemdan(...))` so Python threads are not blocked during parallel Rust computation

#### Story 2.1.5: Packaging & Distribution
**Tasks:**
- [ ] **T-130** Write `pyproject.toml` with maturin build backend, classifier metadata, Python ≥3.10 constraint
- [ ] **T-131** Add `py.typed` marker (PEP 561) and hand-written `*.pyi` stub file for IDE autocompletion — stubs are the only Python-authored file with substance
- [ ] **T-132** Configure GitHub Actions release workflow: `maturin publish` on tag push; build wheels for Linux (manylinux), macOS (universal2), Windows
- [ ] **T-133** Write smoke-test suite (`tests/test_binding.py`): for each exposed function, assert output shape, dtype, and that `.reconstruct()` returns array of correct length — no numerical correctness tests here (those live in Rust)

---

## EPIC 3: R Binding (ferromode-r)

**Binding contract:** This crate contains zero algorithmic logic. The R package is the compiled shared library plus auto-generated extendr wrappers and a minimal `DESCRIPTION`. No R-level computation of IMFs, envelopes, or frequencies.

### Feature 3.1: extendr Marshalling Layer

---

#### Story 3.1.1: extendr Extension Module Setup
*As an R user, I can `install.packages("ferromode-r")` and call `emd(signal, config)` receiving an S3 object.*

**Tasks:**
- [ ] **T-134** Configure `extendr-api` in `ferromode-r/Cargo.toml`; create `#[extendr]` module entry; run `rextendr::document()` to generate `R/ferromode-r-extendr-wrappers.R` — this file is auto-generated and never manually edited
- [ ] **T-135** Expose `EmdConfig` as an `#[extendr]` struct with constructor accepting named R arguments; all defaults come from Rust `impl Default`
- [ ] **T-136** Expose `EnsembleConfig`, `MemdConfig`, `VmdConfig` as `#[extendr]` structs — same principle

#### Story 3.1.2: Array Marshalling
*As a developer, R numeric vectors must reach Rust with zero copy.*

**Tasks:**
- [ ] **T-137** Use `Robj::as_real_slice() -> Option<&[f64]>` for zero-copy input borrow; return `Err` if object is not a real numeric vector
- [ ] **T-138** For multivariate input (MEMD): accept R `matrix` object, use `as_real_vector()` + dimensions attribute to reconstruct channel layout — no R arithmetic
- [ ] **T-139** Validate inputs at marshalling boundary: reject NA, NaN, Inf, zero-length; raise R `stop()` with `EmdError` message

#### Story 3.1.3: Result Marshalling
*As an R user, results are returned as a named list with class `emd_result` containing numeric vectors.*

**Tasks:**
- [ ] **T-140** `ImfCollection` → R named list: `$imfs` (matrix n_imfs × n_samples), `$residue` (numeric vector), `$n_imfs` (integer); assign S3 class `"emd_result"`
- [ ] **T-141** `HilbertResult` → R named list: `$instantaneous_amplitude`, `$instantaneous_frequency` (matrices), `$marginal_spectrum` (numeric vector); class `"hilbert_result"`
- [ ] **T-142** `.reconstruct()` method on `emd_result` calls Rust `ImfCollection::reconstruct()` via extendr — no R summation

#### Story 3.1.4: Function Wrappers
**Tasks:**
- [ ] **T-143** `emd(signal, config)` → marshals `numeric` vector, calls `ferromode::api::emd()`, marshals result
- [ ] **T-144** Same for `eemd`, `ceemd`, `ceemdan`, `iceemdan`, `memd`, `namemd`, `vmd`
- [ ] **T-145** Wrap `EmdError` as R `simpleError` with class `c("emd_error", "error")`; message forwarded verbatim from Rust

#### Story 3.1.5: Packaging & Distribution
**Tasks:**
- [ ] **T-146** Write `DESCRIPTION` with correct `SystemRequirements: Cargo (Rust)`, `LinkingTo: extendr`; pass `R CMD check --as-cran`
- [ ] **T-147** Write minimal `tests/testthat/test-binding.R`: call each function, assert result is a list with correct field names and numeric vector types — no numerical assertions (those are Rust tests)
- [ ] **T-148** Configure GitHub Actions CRAN check on Linux + macOS + Windows; submit to CRAN

---

## EPIC 4: Julia Binding (Ferromode.jl)

**Binding contract:** `Ferromode.jl` is a thin `ccall` wrapper around a C-ABI shared library compiled from `ferromode`. The Julia package contains zero algorithmic logic — only `ccall` declarations, Julia struct definitions that mirror C structs, and finalizer registration for memory management.

### Feature 4.1: C ABI + Julia ccall Layer

---

#### Story 4.1.1: C-ABI Shared Library
*The Rust core must expose a stable C ABI so Julia can call it via `ccall`.*

**Tasks:**
- [ ] **T-149** Add `crate-type = ["cdylib"]` to `ferromode-julia/Cargo.toml`; this crate is a thin re-export of `ferromode::ffi` — no new logic
- [ ] **T-150** In `ferromode/src/ffi.rs`: define C-compatible structs (`#[repr(C)]`) mirroring all config and result types; define `extern "C"` functions for every public API: `ferromode_emd()`, `ferromode_eemd()`, `ferromode_ceemdan()`, `ferromode_iceemdan()`, `ferromode_memd()`, `ferromode_namemd()`, `ferromode_vmd()`
- [ ] **T-151** All FFI functions accept raw pointers (`*const f64`, `*mut f64`) and lengths (`usize`); all results returned as heap-allocated `*mut CImfCollection` pointer; caller must call `ferromode_free_result()` — no exceptions cross the FFI boundary, errors returned as null pointer + error code written to out-param
- [ ] **T-152** Run `cbindgen` in CI to auto-generate `ferromode.h` from `ffi.rs`; commit generated header; binding code must not duplicate struct definitions

#### Story 4.1.2: Julia Package Scaffolding
**Tasks:**
- [ ] **T-153** Create `Ferromode.jl` with standard `Project.toml`; `deps` contains only `Libdl` (stdlib) — no algorithm dependencies
- [ ] **T-154** On `__init__`: use `Libdl.find_library(["libferromode"])` or bundle platform-specific artifact via `JLLWrappers`; store library handle
- [ ] **T-155** Define `EmdConfig`, `EnsembleConfig`, `MemdConfig`, `VmdConfig` as Julia `struct` types with fields that exactly mirror the C structs from `ferromode.h` — no new fields, no defaults computed in Julia (defaults come from Rust `impl Default` exposed as `ferromode_default_emd_config()` FFI call)

#### Story 4.1.3: ccall Wrappers
*As a Julia user, I call `Ferromode.emd(signal, config)` and receive a Julia struct — no awareness of C pointers needed.*

**Tasks:**
- [ ] **T-156** Implement `emd(signal::Vector{Float64}, config::EmdConfig)::ImfCollection`: pin `signal` with `GC.@preserve`, call `ccall((:ferromode_emd, libferromode), Ptr{CImfCollection}, ...)`, wrap result in `ImfCollection` Julia struct, register `finalizer` that calls `ferromode_free_result()`
- [ ] **T-157** Same pattern for `eemd`, `ceemd`, `ceemdan`, `iceemdan` (accepting `EnsembleConfig`)
- [ ] **T-158** Same for `memd`, `namemd` (accepting `Matrix{Float64}` + `MemdConfig`)
- [ ] **T-159** Same for `vmd` (accepting `VmdConfig`)
- [ ] **T-160** Error handling: if FFI returns null, read error code out-param, throw `EmdError(message)` — no error logic, just translation
- [ ] **T-161** Implement `reconstruct(result::ImfCollection)::Vector{Float64}` via `ccall((:ferromode_reconstruct, libferromode), ...)` — calls Rust, no Julia summation

#### Story 4.1.4: Packaging & Distribution
**Tasks:**
- [ ] **T-162** Write `test/runtests.jl` using `@testset`: call each function, assert output types and array sizes — no numerical correctness (those are Rust tests)
- [ ] **T-163** Register in Julia General Registry; configure GitHub Actions to run `Pkg.test()` on Julia 1.9+, Linux + macOS + Windows
- [ ] **T-164** Document the `JLLWrappers` / artifact bundle approach for shipping the compiled library alongside the Julia package

---

## EPIC 5: JavaScript / TypeScript Binding (ferromode-js)

**Binding contract:** The WASM binding contains zero algorithmic logic. The Rust source in `ferromode-wasm/src/lib.rs` does only: copy `Float64Array` data from JS into WASM linear memory, call `ferromode::api`, expose result accessor methods. No mathematical operations in TypeScript.

### Feature 5.1: wasm-bindgen Marshalling Layer

---

#### Story 5.1.1: WASM Module Setup
*As a JS developer, I can `import init, { emd } from 'ferromode-js'` and call EMD on a `Float64Array`.*

**Tasks:**
- [ ] **T-165** Configure `wasm-bindgen` and `wasm-pack` in `emd-wasm/Cargo.toml`; entry point `lib.rs` is `#[wasm_bindgen]` only — no logic
- [ ] **T-166** Expose `WasmEmdConfig` as a `#[wasm_bindgen]` struct with `#[wasm_bindgen(constructor)]` accepting JS object; fields map 1:1 to `EmdConfig` — no defaults computed in JS, defaults from `EmdConfig::default()` in Rust
- [ ] **T-167** Same for `WasmEnsembleConfig`, `WasmMemdConfig`, `WasmVmdConfig`
- [ ] **T-168** Expose `WasmBoundaryCondition`, `WasmStoppingCriterion` as `#[wasm_bindgen]` enums

#### Story 5.1.2: Array Marshalling
*As a developer, `Float64Array` data must enter WASM memory efficiently.*

**Tasks:**
- [ ] **T-169** In each function wrapper: receive `Float64Array`, use `unsafe { std::slice::from_raw_parts(ptr, len) }` inside WASM linear memory to get `&[f64]` — no copy if possible; WASM memory model allows this safely
- [ ] **T-170** For multivariate input (MEMD): accept flat `Float64Array` + `n_channels: usize`; reconstruct `Vec<Vec<f64>>` by striding — this striding is marshalling, not algorithm logic
- [ ] **T-171** Validate: reject non-finite values at marshalling boundary; throw `EmdError` as JS `Error` with Rust message

#### Story 5.1.3: Result Accessors
*As a JS user, I access IMFs via `.getImf(n)` returning a `Float64Array` view.*

**Tasks:**
- [ ] **T-172** `WasmImfCollection` `#[wasm_bindgen]` struct: wraps `ImfCollection`; `.getImf(n: usize) -> Float64Array` returns a view into the underlying buffer; `.getResidue() -> Float64Array`; `.nImfs() -> usize`; `.reconstruct() -> Float64Array` calls Rust
- [ ] **T-173** `WasmHilbertResult`: `.getInstantaneousAmplitude(imf_idx)`, `.getInstantaneousFrequency(imf_idx)`, `.getMarginalSpectrum()` — all return `Float64Array` views
- [ ] **T-174** `WasmDecompositionResult`: `.imfs() -> WasmImfCollection`, `.hilbert() -> WasmHilbertResult`, `.algorithm() -> string`, `.elapsedMs() -> f64`
- [ ] **T-175** Implement `free()` on all result structs; call underlying Rust `drop` — required to avoid WASM memory leaks; document this clearly

#### Story 5.1.4: Function Wrappers
**Tasks:**
- [ ] **T-176** `#[wasm_bindgen] pub fn emd(signal: &[f64], config: &WasmEmdConfig) -> Result<WasmDecompositionResult, JsValue>` — marshal, call Rust, marshal out
- [ ] **T-177** Same for `eemd`, `ceemd`, `ceemdan`, `iceemdan`, `memd`, `namemd`, `vmd`
- [ ] **T-178** Map `EmdError` → `JsValue::from(js_sys::Error::new(&msg))`; no information lost

#### Story 5.1.5: Build, Types & Distribution
**Tasks:**
- [ ] **T-179** `wasm-pack build --target web` and `--target nodejs`; output to `pkg/`
- [ ] **T-180** `wasm-bindgen` auto-generates `.d.ts` for all `#[wasm_bindgen]` exports — review and supplement with hand-written `.d.ts` for the async `init()` pattern only
- [ ] **T-181** Write `package.json` with dual ESM/CJS exports; publish to npm as `ferromode-js`
- [ ] **T-182** Write Vitest test suite: import WASM, call each function, assert output `instanceof Float64Array` and correct `length` — no numerical assertions (those are Rust tests)
- [ ] **T-183** Configure GitHub Actions release: `wasm-pack publish` on tag push

---

## EPIC 6: Cross-Language Validation & Documentation

### Feature 6.1: Validation Suite

#### Story 6.1.1: Reference Signal Library
**Tasks:**
- [ ] **T-184** Create set of reference test signals with known analytical properties: pure tones, AM, FM, chirp, sunspot data, synthetic multivariate
- [ ] **T-185** Pre-compute expected IMF outputs using Rilling & Flandrin's C reference implementation
- [ ] **T-186** Store as JSON in `validation/reference/` directory

#### Story 6.1.2: Cross-Language Test Runner
**Tasks:**
- [ ] **T-187** Write script that runs same decomposition in Rust, Python, R, Julia, JS, MATLAB, and C++ and compares outputs
- [ ] **T-188** Assert all pairwise differences < 1e-10
- [ ] **T-189** Run Rust/Python/R/Julia/JS/C++ legs in CI on every PR; MATLAB leg on nightly (requires licence)

### Feature 6.2: Documentation

#### Story 6.2.1: Algorithm Documentation
**Tasks:**
- [ ] **T-190** Write mathematical description of each algorithm with LaTeX equations in `docs/algorithms/`
- [ ] **T-191** Include full bibliographic citations for each algorithm and boundary method
- [ ] **T-192** Create decision tree diagram: "Which algorithm should I use?"
- [ ] **T-193** Create comparison table: algorithm properties, use cases, computational cost

#### Story 6.2.2: API Documentation
**Tasks:**
- [ ] **T-194** Ensure every public Rust function has rustdoc with example
- [ ] **T-195** Build and publish to docs.rs
- [ ] **T-196** Write mkdocs site with `binding-guide.md` explaining the pure-wrap contract; narrative guide per binding
- [ ] **T-197** Create getting-started tutorial for each of the 6 language bindings (Python, R, Julia, JS, MATLAB, C++)

---

## EPIC 7: MATLAB / Octave Binding (ferromode-mex)

**Rationale:** MATLAB is the incumbent language for EMD research. The canonical Rilling & Flandrin reference implementation ships as MATLAB/C code, and virtually every major EMD paper ships MATLAB as the reference. A MEX binding lets researchers call Ferromode directly from their existing MATLAB/Octave workflows with no code changes beyond the function call — and they get a 10–100× speed improvement over `.m` implementations for free.

**Binding contract:** Zero algorithmic logic. The MEX entry point receives MATLAB arrays, marshals them to Rust slices, calls `ferromode::api`, and returns `mxArray` results. Identical pure-wrap contract as all other bindings.

**Technical approach:** MEX functions are compiled shared libraries (`.mexa64` / `.mexmaci64` / `.mexw64`) with a C entry point `mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])`. Ferromode reuses the C-ABI `cdylib` already compiled for the Julia binding — the MEX crate is a thin additional entry point over the same `ferromode::ffi` module.

### Feature 7.1: MEX Marshalling Layer

---

#### Story 7.1.1: MEX Entry Point & Build System
*As a MATLAB/Octave user, I can call `ferromode_emd(signal, ...)` as a native MEX function.*

**Tasks:**
- [ ] **T-198** Create `ferromode-mex` crate: `crate-type = ["cdylib"]`; implement `mexFunction` as `extern "C"` entry point linking against `ferromode::ffi` — no new logic
- [ ] **T-199** Configure build system: `cc` crate links against `libmex` and `libmx` from MATLAB SDK (path configurable via `MATLAB_ROOT` env var); also support Octave's `liboctave` for open-source compatibility
- [ ] **T-200** Write `build.rs` that detects MATLAB vs Octave installation and sets correct link flags and output extension (`.mexa64` / `.mexmaci64` / `.mexw64` for MATLAB; `.mex` for Octave)
- [ ] **T-201** Implement `mxGetPr()` / `mxGetM()` / `mxGetN()` based input extraction: `*const f64` + dimensions → `&[f64]` slice — marshalling only, no computation

#### Story 7.1.2: Input Marshalling
*As a developer, MATLAB `double` arrays must reach Rust with zero copy via `mxGetPr()`.*

**Tasks:**
- [ ] **T-202** Extract signal from `prhs[0]`: validate `mxIsDouble()`, `!mxIsComplex()`, column or row vector; get pointer via `mxGetPr()` and length via `mxGetNumberOfElements()` → `&[f64]`
- [ ] **T-203** Parse config struct from `prhs[1]` (optional MATLAB struct): use `mxGetField()` to extract named fields mapping to `FerromodeConfig` fields; all defaults from Rust `impl Default`
- [ ] **T-204** For MEMD/NA-MEMD: accept MATLAB matrix `prhs[0]`; extract via `mxGetPr()` + `mxGetM()` + `mxGetN()` to reconstruct channel layout — marshalling only
- [ ] **T-205** Validate at boundary: non-double, complex, empty, or non-finite inputs → `mexErrMsgIdAndTxt("Ferromode:invalidInput", msg)` with Rust error message verbatim

#### Story 7.1.3: Result Marshalling
*As a MATLAB user, results are returned as a struct with fields `imfs`, `residue`, `n_imfs`.*

**Tasks:**
- [ ] **T-206** Allocate output `mxArray` struct via `mxCreateStructMatrix(1,1,nfields,fieldnames)` with fields `imfs` (matrix), `residue` (vector), `n_imfs` (scalar), `algorithm` (string), `elapsed_ms` (scalar)
- [ ] **T-207** Copy IMF data from Rust `ImfCollection` into `mxCreateDoubleMatrix` allocations via `memcpy` — MATLAB owns output memory
- [ ] **T-208** Implement `ferromode_reconstruct(result_struct)` MEX: accepts the output struct, extracts `imfs` + `residue`, calls `ferromode::api::reconstruct()` in Rust, returns `mxArray` double vector
- [ ] **T-209** Implement `ferromode_hilbert(result_struct)` MEX: accepts output struct, calls `ferromode::api::hilbert()`, returns struct with `instantaneous_amplitude`, `instantaneous_frequency`, `marginal_spectrum` fields

#### Story 7.1.4: Function Wrappers (all algorithms)
**Tasks:**
- [ ] **T-210** `ferromode_emd.mexa64`: MEX entry point → marshal → call `ferromode_emd()` FFI → marshal result
- [ ] **T-211** Same for `ferromode_eemd`, `ferromode_ceemd`, `ferromode_ceemdan`, `ferromode_iceemdan`
- [ ] **T-212** Same for `ferromode_memd`, `ferromode_namemd` (matrix input)
- [ ] **T-213** Same for `ferromode_vmd`
- [ ] **T-214** Wrapper `.m` files for each function providing MATLAB-style `help` documentation and argument name sugar: `ferromode_emd(signal, 'BoundaryCondition', 'periodic', 'MaxIMFs', 8)` — these `.m` files call the MEX binary; they contain no computation

#### Story 7.1.5: Octave Compatibility & Distribution
**Tasks:**
- [ ] **T-215** Test all MEX functions under GNU Octave 8+; fix any `liboctave` API differences (Octave's MEX layer is largely compatible but has minor divergences in struct creation)
- [ ] **T-216** Write MATLAB test script `tests/test_ferromode.m`: call each function, assert output is struct with correct field names and sizes — `assert(size(result.imfs, 1) >= 1)` style; no numerical assertions
- [ ] **T-217** Package as MATLAB toolbox (`.mltbx`) for MATLAB Add-On Explorer submission; package as Octave package (`.tar.gz`) for Octave Forge submission
- [ ] **T-218** Document `MATLAB_ROOT` build configuration in `binding-guide.md`; add CI job that builds MEX on GitHub Actions with MATLAB licence (or Octave as free alternative for open CI)

---

## EPIC 8: C++ Binding (ferromode-cxx)

**Rationale:** A meaningful EMD user base works in C++ for embedded systems, real-time DSP, robotics, and HPC pipelines where MATLAB and Python are too heavy. C++ is also the natural integration point for engineers embedding Ferromode into production signal processing systems. The Rust Foundation's active C++/Rust interoperability initiative makes this increasingly well-supported tooling.

**Binding contract:** Zero algorithmic logic. The C++ binding is a header + compiled library. Headers expose idiomatic C++ types (`std::vector<std::vector<double>>`, `std::span<const double>`) and delegate entirely to the `ferromode::ffi` C-ABI layer. No C++ signal processing code.

**Technical approach:** Two complementary layers — a raw C header (already generated by `cbindgen` for Julia) for maximum compatibility, and a modern C++17 header-only wrapper using `cxx` or manual `extern "C"` that provides RAII, `std::span`, and `std::vector` ergonomics. The `cxx` crate is preferred for its safety properties but the manual approach is provided as a fallback for codebases not using a Cargo-based build.

### Feature 8.1: C++ Marshalling Layer

---

#### Story 8.1.1: C Header & Build Artefacts
*As a C++ developer, I can link against `libferromode` and include `ferromode.h` to call all algorithms.*

**Tasks:**
- [ ] **T-219** Confirm `ferromode.h` generated by `cbindgen` (already produced for Julia binding) is sufficient for C++ consumption — no additions needed; it is the C-ABI contract
- [ ] **T-220** Provide `CMakeLists.txt` in `ferromode-cxx/` that fetches the pre-built `libferromode` for the target platform (via `FetchContent` from GitHub Releases) and exposes `ferromode::ferromode` CMake target
- [ ] **T-221** Provide `ferromode.pc` pkg-config file for non-CMake build systems
- [ ] **T-222** Ship pre-built binaries for Linux (x86_64, aarch64), macOS (universal2), Windows (x64) via GitHub Releases as part of the standard release workflow

#### Story 8.1.2: C++17 Header-Only Wrapper
*As a C++ developer, I use idiomatic C++ types — `std::span`, `std::vector`, RAII — without touching raw pointers.*

**Tasks:**
- [ ] **T-223** Write `ferromode.hpp`: `namespace ferromode { ... }` with `EmdConfig`, `EnsembleConfig`, `MemdConfig`, `VmdConfig` C++ structs — plain aggregates mirroring the C structs; constructors delegate to `ferromode_default_*_config()` FFI for defaults
- [ ] **T-224** Implement `ImfCollection` RAII wrapper: holds `CImfCollection*`; destructor calls `ferromode_free_result()`; `.imfs() -> std::vector<std::span<const double>>`; `.residue() -> std::span<const double>`; `.reconstruct() -> std::vector<double>` calls Rust FFI — no C++ summation
- [ ] **T-225** Implement free functions: `ferromode::emd(std::span<const double> signal, const EmdConfig& config) -> ImfCollection` — extracts `.data()` + `.size()`, calls `ferromode_emd()` FFI, wraps result in `ImfCollection`; validation delegated to Rust
- [ ] **T-226** Same for `eemd`, `ceemd`, `ceemdan`, `iceemdan` (accepting `EnsembleConfig`)
- [ ] **T-227** Same for `memd`, `namemd` (accepting `std::span<const double*>` channels + `MemdConfig`)
- [ ] **T-228** Same for `vmd` (accepting `VmdConfig`)
- [ ] **T-229** Implement `ferromode::hilbert(const ImfCollection&) -> HilbertResult` RAII wrapper; calls Rust FFI; no C++ DSP
- [ ] **T-230** Error handling: FFI null returns → throw `ferromode::FerromodeError(std::string message)` derived from `std::runtime_error` — no error logic, just translation

#### Story 8.1.3: `cxx` Bridge (Optional Modern Path)
*As a Rust-integrated C++ developer using Cargo + cmake-rs, I get type-safe bridging without manual `extern "C"`.*

**Tasks:**
- [ ] **T-231** Create `ferromode-cxx` Rust crate using the `cxx` crate: define bridge with `#[cxx::bridge]` exposing all public API functions with C++ idiomatic signatures
- [ ] **T-232** Expose `rust::Vec<f64>` ↔ `std::vector<double>` conversions via `cxx` generated glue — no manual pointer arithmetic
- [ ] **T-233** Document both paths clearly in `binding-guide.md`: header-only (no Cargo required) vs `cxx` bridge (Cargo-integrated projects)

#### Story 8.1.4: Testing & Distribution
**Tasks:**
- [ ] **T-234** Write C++ test suite using Catch2: for each function, assert output vector sizes and that `reconstruct()` returns a vector of correct length — no numerical assertions
- [ ] **T-235** CI: build and test with GCC 12+, Clang 15+, MSVC 2022 on Linux/macOS/Windows; run under AddressSanitizer and UndefinedBehaviorSanitizer
- [ ] **T-236** Publish to `vcpkg` registry and `Conan Center Index` for easy integration into existing C++ projects
- [ ] **T-237** Add C++ getting-started tutorial to docs site covering both CMake + `FetchContent` path and `cxx` bridge path
