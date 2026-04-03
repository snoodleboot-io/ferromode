# Execution Checklist
## Ferromode Project — Task-Level Tracking

**Last Updated:** 2026-04-02  
**Total Tasks:** 237 | **Completed:** 25 | **In Progress:** 0 | **Blocked:** 0 | **Review:** 0

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
Each task tracks which mode is/will be responsible:
- `architect` — Design, task breakdown, technical decisions
- `code` — Implementation
- `test` — Test writing and coverage
- `review` — Code review and approval
- `debug` — Bug investigation
- `docs` — Documentation

### Engineering Practice Compliance
Each task must satisfy the following before marked DONE:
- **TDD**: Test written BEFORE implementation, red-green-refactor cycle followed
- **ATDD**: Acceptance criteria defined and automated before implementation
- **DDD**: Domain types and bounded contexts respected, no anemic models
- **Clean Code**: Meaningful names, small functions, no duplication, clear intent
- **Clean Architecture**: Dependencies point inward, no framework leakage into domain
- **SOLID**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion

---

## Milestone M0: Project Bootstrap
**Target:** Week 2 of April 2026  
**Exit Criteria:** Workspace builds, CI green, contributing guide published  
**Tasks:** T-001 to T-007 (7 tasks)

### Epic 1 · Feature 1.1 · Story 1.1.1: Workspace & Project Scaffolding

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-001 | Initialize Cargo workspace with `ferromode`, `ferromode-py`, `ferromode-r`, `ferromode-wasm` crates | ✅ | done | session_20260402_zws16l | Complete — workspace with 5 crates created |
| T-002 | Configure `Cargo.toml` with shared dependency versions (ndarray 0.15, rayon 1.8, rustfft 6, rand 0.8, serde 1, thiserror 1) | ✅ | done | session_20260402_zws16l | Complete — shared deps configured |
| T-003 | Set up `.github/workflows/ci.yml` with matrix build (Linux, macOS, Windows × stable, beta) | ✅ | done | session_20260402_zws16l | Complete — matrix build configured |
| T-004 | Configure `rustfmt.toml` and `clippy.toml` with project-wide lint rules | ✅ | done | session_20260402_zws16l | Complete — lint rules configured |
| T-005 | Create `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE` (Apache-2.0) | ✅ | done | session_20260402_zws16l | Complete — Apache-2.0, CoC, contributing guide |
| T-006 | Set up `criterion` benchmarking harness in `ferromode/benches/` | ✅ | done | session_20260402_zws16l | Complete — 3 benchmark groups configured |
| T-007 | Create initial `README.md` with project overview and quick-start | ✅ | done | session_20260402_zws16l | Complete — overview, quick-start, project structure |

---

## Milestone M1: Spline & Boundary Foundation
**Target:** End of May 2026  
**Exit Criteria:** All 8 boundary strategies implemented and tested; cubic spline matches scipy within 1e-10  
**Tasks:** T-008 to T-049 (42 tasks)

### Epic 1 · Feature 1.1 · Story 1.1.2: Core Type System

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-008 | Define `Signal` struct: `Vec<f64>` values + optional sample rate, with `from_slice`, `len`, `iter` methods | ✅ | done | session_20260402_zws16l | Complete |
| T-009 | Define `MultivariateSignal` struct: `Vec<Vec<f64>>` channels with dimension validation | ✅ | done | session_20260402_zws16l | Complete |
| T-010 | Define `ImfCollection` struct with `imfs: Vec<Vec<f64>>`, `residue: Vec<f64>`, `reconstruct()`, `orthogonality_index()` methods | ✅ | done | session_20260402_zws16l | Complete |
| T-011 | Define `DecompositionResult` with algorithm metadata, elapsed time, config snapshot | ✅ | done | session_20260402_zws16l | Complete |
| T-012 | Define `HilbertResult` struct with instantaneous amplitude, frequency, marginal spectrum fields | ✅ | done | session_20260402_zws16l | Complete |
| T-013 | Define `AlgorithmType` enum (EMD, EEMD, CEEMD, CEEMDAN, ICEEMDAN, MEMD, NAMEMD, VMD) | ✅ | done | session_20260402_zws16l | Complete |
| T-014 | Derive `serde::Serialize/Deserialize` on all result types; test JSON round-trip | ✅ | done | session_20260402_zws16l | Complete |
| T-015 | Write unit tests for reconstruction error, orthogonality index | ✅ | done | session_20260402_zws16l | Complete |

### Epic 1 · Feature 1.1 · Story 1.1.3: Error Handling

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-016 | Define `EmdError` enum with thiserror: `EmptySignal`, `InsufficientExtrema`, `InvalidConfig`, `NumericalFailure`, `DimensionMismatch`, `InvalidBoundary` | ✅ | done | session_20260402_zws16l | Complete |
| T-017 | Implement `From<EmdError>` conversions for each binding's native error type | ✅ | done | session_20260402_zws16l | Complete |
| T-018 | Add input validation at all public API entry points (check NaN, Inf, zero length, dimension mismatches) | ✅ | done | session_20260402_zws16l | Complete |
| T-019 | Write tests for all error paths | ✅ | done | session_20260402_zws16l | Complete |

### Epic 1 · Feature 1.1 · Story 1.1.4: Cubic Spline Engine

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-020 | Implement natural cubic spline from extrema knots: tridiagonal system solver (Thomas algorithm) | ✅ | done | session_20260402_zws16l | Complete |
| T-021 | Implement periodic cubic spline (required for Zeng-He boundary condition) | ✅ | done | session_20260402_zws16l | Complete |
| T-022 | Implement not-a-knot cubic spline variant | ✅ | done | session_20260402_zws16l | Complete |
| T-023 | Validate against reference (scipy's CubicSpline) on test cases: uniform knots, non-uniform knots, single/double extrema edge cases | ✅ | done | session_20260402_zws16l | Complete |
| T-024 | Benchmark spline vs. naive O(n²) approach; ensure O(n) Thomas algorithm is used | ✅ | done | session_20260402_zws16l | Complete |
| T-025 | Handle degenerate cases: fewer than 2 knots, duplicate knot locations | ✅ | done | session_20260402_zws16l | Complete |

### Epic 1 · Feature 1.2 · Story 1.2.1: Boundary Trait & Registry

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-026 | Define `BoundaryCondition` trait with `extend(signal: &[f64], extrema: &Extrema) -> ExtendedSignal` | 🔲 | code | — | — |
| T-027 | Define `BoundaryCondition` enum and `get_strategy()` factory function | 🔲 | code | — | — |
| T-028 | Write test harness that applies each strategy and verifies: (1) extended signal contains original, (2) spline fit is smooth, (3) no NaN/Inf produced | 🔲 | test | — | — |

### Epic 1 · Feature 1.2 · Story 1.2.2: Characteristic Wave Extension

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-029 | Implement `CharacteristicWave` strategy: identify two nearest extrema at each boundary, construct implicit wave, append 4 copies per end | 🔲 | code | — | Ref: Huang et al. 1998 |
| T-030 | Handle edge case: signal has fewer than 4 extrema total | 🔲 | code | — | — |
| T-031 | Test on: pure sine, chirp, noisy signal with envelope modulation | 🔲 | test | — | — |

### Epic 1 · Feature 1.2 · Story 1.2.3: Mirror / Symmetric Extension

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-032 | Implement even-extension mirror strategy | 🔲 | code | — | — |
| T-033 | Implement odd-extension mirror strategy | 🔲 | code | — | — |
| T-034 | Test both variants; verify that endpoint is a local extremum in extended signal | 🔲 | test | — | — |

### Epic 1 · Feature 1.2 · Story 1.2.4: Periodic / Cyclic Extension (Zeng & He 2004)

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-035 | Implement periodic extension: concatenate even-extended and odd-extended copies to build periodic series | 🔲 | code | — | Ref: Zeng & He 2004 |
| T-036 | Use periodic cubic spline BC for envelope computation on the periodic extended signal | 🔲 | code | — | — |
| T-037 | Test on signals with known periodic structure; verify no endpoint artifacts | 🔲 | test | — | — |
| T-038 | Add citation comment in source code pointing to Zeng & He 2004 | 🔲 | docs | — | — |

### Epic 1 · Feature 1.2 · Story 1.2.5: Slope-Based Extension

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-039 | Compute first-derivative at each endpoint using finite differences | 🔲 | code | — | — |
| T-040 | Extrapolate linearly beyond each boundary to generate artificial extrema | 🔲 | code | — | — |
| T-041 | Test on signals with monotone ends | 🔲 | test | — | — |

### Epic 1 · Feature 1.2 · Story 1.2.6: AR Model Extension

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-042 | Implement Yule-Walker AR coefficient estimation (order p, default p=5) | 🔲 | code | — | — |
| T-043 | Forecast N samples beyond right boundary; backcast N samples before left boundary | 🔲 | code | — | — |
| T-044 | Expose `ar_order` as a configuration parameter | 🔲 | code | — | — |
| T-045 | Test on AR(2) synthetic signals; verify near-perfect extension | 🔲 | test | — | — |

### Epic 1 · Feature 1.2 · Story 1.2.7: Waveform Matching Extension

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-046 | Implement cross-correlation search: find interior segment most similar to each endpoint region | 🔲 | code | — | — |
| T-047 | Append matched segment beyond boundary | 🔲 | code | — | — |
| T-048 | Expose `match_length` (in samples) as configuration parameter | 🔲 | code | — | — |
| T-049 | Test on quasi-periodic signals; compare against mirror extension | 🔲 | test | — | — |

---

## Milestone M2: Basic EMD Alpha
**Target:** End of June 2026  
**Release Tag:** `v0.2.0-alpha`  
**Exit Criteria:** Basic EMD produces correct IMFs on reference signals; reconstruction error < 1e-12; published to crates.io as pre-release  
**Tasks:** T-050 to T-066, T-097 to T-102 (33 tasks)

### Epic 1 · Feature 1.3 · Story 1.3.1: Extrema Detection

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-050 | Implement local maxima detection: find indices where `x[i] > x[i-1] && x[i] > x[i+1]` | 🔲 | code | — | — |
| T-051 | Implement local minima detection | 🔲 | code | — | — |
| T-052 | Handle plateau extrema: detect flat tops/bottoms, use midpoint index | 🔲 | code | — | — |
| T-053 | Handle boundary as potential extremum (needed for some boundary strategies) | 🔲 | code | — | — |
| T-054 | Test on: sine wave, sawtooth, step function, constant signal | 🔲 | test | — | — |

### Epic 1 · Feature 1.3 · Story 1.3.2: Sifting Engine

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-055 | Implement `sift_one(signal, config) -> (imf, residue)` function | 🔲 | code | — | — |
| T-056 | Implement SD threshold stopping criterion: `SD = Σ|h_{k-1} - h_k|² / Σ|h_{k-1}|² < threshold` | 🔲 | code | — | — |
| T-057 | Implement S-number criterion: count consecutive siftings where #extrema and #zero-crossings are equal or differ by 1 | 🔲 | code | — | — |
| T-058 | Implement fixed-iteration stopping | 🔲 | code | — | — |
| T-059 | Implement energy-difference stopping | 🔲 | code | — | — |
| T-060 | Add sifting iteration counter and max_sifting_iterations guard to prevent infinite loops | 🔲 | code | — | — |

### Epic 1 · Feature 1.3 · Story 1.3.3: Basic EMD

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-061 | Implement `emd(signal, config) -> DecompositionResult` top-level function | 🔲 | code | — | — |
| T-062 | Implement outer loop: repeatedly call `sift_one` on residue until residue has < 2 extrema or max_imfs reached | 🔲 | code | — | — |
| T-063 | Implement `max_imfs` limit | 🔲 | code | — | — |
| T-064 | Implement intermittency test option (frequency-range filter on extrema spacing) | 🔲 | code | — | — |
| T-065 | Validate reconstruction: `Σ IMFs + residue = original signal` within 1e-12 | 🔲 | test | — | — |
| T-066 | Write reference tests using Huang's original test signals (sunspot data, EEG) | 🔲 | test | — | — |

### Epic 1 · Feature 1.6 · Story 1.6.1: Hilbert Transform

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-097 | Implement FFT-based Hilbert transform: FFT → zero negative frequencies → IFFT | 🔲 | code | — | — |
| T-098 | Compute analytic signal: `z(t) = x(t) + i·H{x}(t)` | 🔲 | code | — | — |
| T-099 | Compute instantaneous amplitude: `A(t) = |z(t)|` | 🔲 | code | — | — |
| T-100 | Compute instantaneous phase: `φ(t) = arctan(H{x}/x)` | 🔲 | code | — | — |
| T-101 | Compute instantaneous frequency: `f(t) = (1/2π) · dφ/dt` | 🔲 | code | — | — |
| T-102 | Validate against known analytic signals (pure tone, AM signal) | 🔲 | test | — | — |

---

## Milestone M3: Ensemble Methods Beta
**Target:** End of July 2026  
**Release Tag:** `v0.3.0-beta`  
**Exit Criteria:** EEMD, CEEMD, CEEMDAN, ICEEMDAN all implemented; parallel execution verified; CEEMDAN 200 trials on 10K samples < 10s on 8 cores  
**Tasks:** T-067 to T-083 (17 tasks)

### Epic 1 · Feature 1.4 · Story 1.4.1: EEMD

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-067 | Implement `eemd(signal, ensemble_config)`: add Gaussian white noise (std = noise_std × signal_std), run EMD per trial | 🔲 | code | — | Ref: Wu & Huang 2009 |
| T-068 | Implement parallel trial execution via `rayon::par_iter` | 🔲 | code | — | — |
| T-069 | Implement seeded RNG for reproducibility (`rand::SeedableRng`) | 🔲 | code | — | — |
| T-070 | Average IMFs across trials (handle unequal IMF counts by zero-padding) | 🔲 | code | — | — |
| T-071 | Validate: noise cancels in average; test on mode-mixed synthetic signal | 🔲 | test | — | — |

### Epic 1 · Feature 1.4 · Story 1.4.2: CEEMD

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-072 | Implement paired noise trials: for each trial i, compute `emd(signal + ε·noise_i)` and `emd(signal - ε·noise_i)` | 🔲 | code | — | Ref: Yeh et al. 2010 |
| T-073 | Average complementary pairs; verify exact reconstruction | 🔲 | test | — | — |
| T-074 | Test that RMS noise level in IMFs matches EEMD | 🔲 | test | — | — |

### Epic 1 · Feature 1.4 · Story 1.4.3: CEEMDAN

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-075 | Implement stage-wise CEEMDAN: at stage k, add adaptive noise to current residue | 🔲 | code | — | Ref: Torres et al. 2011 |
| T-076 | Compute adaptive noise as: `ε_k = ε · std(residue_k) / std(noise)` | 🔲 | code | — | — |
| T-077 | Extract first IMF of `emd(residue_k + ε_k·noise)` for each trial; average to get mean_IMF_k | 🔲 | code | — | — |
| T-078 | Compute next residue: `r_{k+1} = r_k - mean_IMF_k` | 🔲 | code | — | — |
| T-079 | Parallelize trial computation at each stage | 🔲 | code | — | — |
| T-080 | Validate reconstruction error < 1e-10 | 🔲 | test | — | — |

### Epic 1 · Feature 1.4 · Story 1.4.4: ICEEMDAN

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-081 | Implement improved noise model: use EMD of noise-only signal at each stage | 🔲 | code | — | Ref: Colominas et al. 2014 |
| T-082 | Compute `mean_IMF_k = E[first_IMF(r_k + ε_k · IMF_k_of_noise)]` | 🔲 | code | — | — |
| T-083 | Test that residual noise in IMFs is further reduced vs. CEEMDAN | 🔲 | test | — | — |

---

## Milestone M4: Multivariate & VMD
**Target:** End of September 2026  
**Release Tag:** `v0.5.0`  
**Exit Criteria:** MEMD, NA-MEMD, VMD implemented; mode-alignment test passes; all post-processing metrics available  
**Tasks:** T-084 to T-096, T-103 to T-106, T-107 to T-112 (26 tasks)

### Epic 1 · Feature 1.5 · Story 1.5.1: Hypersphere Direction Sampling

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-084 | Implement uniform angular sampling on n-sphere for n=2,3,4,6,8 channels | 🔲 | code | — | Ref: Rehman & Mandic 2010 |
| T-085 | Implement Hammersley low-discrepancy sequence sampling | 🔲 | code | — | — |
| T-086 | Implement Halton sequence sampling | 🔲 | code | — | — |
| T-087 | Expose `DirectionSampling` enum: `Uniform`, `Hammersley`, `Halton` | 🔲 | code | — | — |
| T-088 | Validate distribution uniformity with statistical test (Kolmogorov-Smirnov on projection angles) | 🔲 | test | — | — |

### Epic 1 · Feature 1.5 · Story 1.5.2: MEMD

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-089 | Implement projection of n-variate signal onto each direction vector | 🔲 | code | — | — |
| T-090 | Find extrema of each projection; interpolate component-wise to form n-D envelopes | 🔲 | code | — | — |
| T-091 | Average envelopes over all directions to compute local mean | 🔲 | code | — | — |
| T-092 | Implement multivariate sifting loop using multivariate stopping criterion | 🔲 | code | — | — |
| T-093 | Test mode-alignment property: decompose synthetic hexavariate signal with known scales; verify IMFs align across channels | 🔲 | test | — | — |

### Epic 1 · Feature 1.5 · Story 1.5.3: NA-MEMD

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-094 | Implement NA-MEMD: add `n_noise_channels` of white noise as extra channels | 🔲 | code | — | Ref: Rehman & Mandic 2011 |
| T-095 | Run MEMD on augmented signal; discard noise channel IMFs post-decomposition | 🔲 | code | — | — |
| T-096 | Test noise suppression vs. plain MEMD on bivariate test signal | 🔲 | test | — | — |

### Epic 1 · Feature 1.6 · Story 1.6.2: Marginal Spectrum & Metrics

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-103 | Implement Hilbert marginal spectrum (time-integrated instantaneous energy at each frequency) | 🔲 | code | — | — |
| T-104 | Implement IMF orthogonality index | 🔲 | code | — | — |
| T-105 | Implement degree of stationarity metric | 🔲 | code | — | — |
| T-106 | Implement IMF energy ratio per component | 🔲 | code | — | — |

### Epic 1 · Feature 1.7 · Story 1.7.1: VMD Core

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-107 | Implement VMD: alternating direction method of multipliers (ADMM) in frequency domain | 🔲 | code | — | Ref: Dragomiretskiy & Zosso 2014 |
| T-108 | Implement center frequency update step | 🔲 | code | — | — |
| T-109 | Implement mode update step (Wiener filter) | 🔲 | code | — | — |
| T-110 | Implement dual variable (Lagrange multiplier) update | 🔲 | code | — | — |
| T-111 | Expose `n_modes` (K), `alpha` (bandwidth penalty), `tau` (noise tolerance), `tol` (convergence) parameters | 🔲 | code | — | — |
| T-112 | Validate against MATLAB reference output from Dragomiretskiy & Zosso | 🔲 | test | — | — |

---

## Milestone M5: Python v1.0
**Target:** Early October 2026  
**Release Tag:** `v1.0.0`  
**Exit Criteria:** Published to PyPI; all 8 algorithms callable; inputs/outputs are numpy arrays; EmdError maps to Python ValueError; GIL released during ensemble methods; binding smoke-tests pass; cross-language validation confirms numerical identity with Rust core  
**Tasks:** T-113 to T-133, T-184 to T-189 (38 tasks)

### Epic 2 · Feature 2.1 · Story 2.1.1: PyO3 Extension Module Setup

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-113 | Configure PyO3 dependency and `maturin` build in `ferromode-py/Cargo.toml`; create `#[pymodule] fn ferromode_py(m: &PyModule)` entry point | 🔲 | code | — | — |
| T-114 | Expose `EmdConfig` as a `#[pyclass]` with `#[new]` accepting keyword arguments; fields mirror Rust struct exactly — no defaults computed here, defaults live in Rust | 🔲 | code | — | — |
| T-115 | Expose `EnsembleConfig`, `MemdConfig` as `#[pyclass]` wrappers — same principle | 🔲 | code | — | — |
| T-116 | Expose `BoundaryCondition` and `StoppingCriterion` as `#[pyclass]` enums | 🔲 | code | — | — |
| T-117 | Expose `AlgorithmType` as a `#[pyclass]` enum for result inspection | 🔲 | code | — | — |

### Epic 2 · Feature 2.1 · Story 2.1.2: Array Marshalling — Input

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-118 | Implement `numpy_to_slice(arr: PyReadonlyArray1<f64>) -> &[f64]` — zero-copy borrow if C-contiguous; copy + warn if not | 🔲 | code | — | — |
| T-119 | Implement `numpy2d_to_vecs(arr: PyReadonlyArray2<f64>) -> Vec<Vec<f64>>` for multivariate input (MEMD/NA-MEMD) | 🔲 | code | — | — |
| T-120 | Validate input at marshalling boundary: reject non-finite values, zero-length arrays, wrong dtype — raise `ValueError` with message forwarded from `EmdError` | 🔲 | code | — | — |

### Epic 2 · Feature 2.1 · Story 2.1.3: Result Marshalling — Output

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-121 | Implement `ImfCollectionPy` `#[pyclass]`: wraps `ImfCollection`; `.imfs` property returns `PyArray2<f64>` (n_imfs × n_samples); `.residue` returns `PyArray1<f64>`; `.reconstruct()` calls `ImfCollection::reconstruct()` in Rust and returns `PyArray1<f64>` | 🔲 | code | — | — |
| T-122 | Implement `HilbertResultPy` `#[pyclass]`: wraps `HilbertResult`; `.instantaneous_amplitude`, `.instantaneous_frequency`, `.marginal_spectrum` as numpy arrays | 🔲 | code | — | — |
| T-123 | Implement `DecompositionResultPy` `#[pyclass]`: exposes `.algorithm`, `.elapsed_ms`, `.imfs` (→ `ImfCollectionPy`), `.hilbert()` (→ `HilbertResultPy`) | 🔲 | code | — | — |

### Epic 2 · Feature 2.1 · Story 2.1.4: Function Wrappers

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-124 | `#[pyfunction] fn emd(signal: PyReadonlyArray1<f64>, config: &EmdConfigPy) -> PyResult<DecompositionResultPy>` — marshal in, call `ferromode::api::emd()`, marshal out | 🔲 | code | — | — |
| T-125 | Same for `eemd`, `ceemd`, `ceemdan`, `iceemdan` (all accepting `EnsembleConfigPy`) | 🔲 | code | — | — |
| T-126 | Same for `memd`, `namemd` (accepting `MemdConfigPy` + `PyReadonlyArray2<f64>`) | 🔲 | code | — | — |
| T-127 | Same for `vmd` (accepting `VmdConfigPy`) | 🔲 | code | — | — |
| T-128 | Wrap all Rust `Result::Err` variants into typed Python exceptions: `EmdError` → `ferromode_py.EmdError(ValueError)` with the original message; no information lost | 🔲 | code | — | — |
| T-129 | Release GIL inside all ensemble method wrappers: `py.allow_threads(|| ferromode::api::ceemdan(...))` so Python threads are not blocked during parallel Rust computation | 🔲 | code | — | — |

### Epic 2 · Feature 2.1 · Story 2.1.5: Packaging & Distribution

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-130 | Write `pyproject.toml` with maturin build backend, classifier metadata, Python ≥3.10 constraint | 🔲 | code | — | — |
| T-131 | Add `py.typed` marker (PEP 561) and hand-written `*.pyi` stub file for IDE autocompletion — stubs are the only Python-authored file with substance | 🔲 | code | — | — |
| T-132 | Configure GitHub Actions release workflow: `maturin publish` on tag push; build wheels for Linux (manylinux), macOS (universal2), Windows | 🔲 | code | — | — |
| T-133 | Write smoke-test suite (`tests/test_binding.py`): for each exposed function, assert output shape, dtype, and that `.reconstruct()` returns array of correct length — no numerical correctness tests here (those live in Rust) | 🔲 | test | — | — |

### Epic 6 · Feature 6.1 · Story 6.1.1: Reference Signal Library

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-184 | Create set of reference test signals with known analytical properties: pure tones, AM, FM, chirp, sunspot data, synthetic multivariate | 🔲 | code | — | — |
| T-185 | Pre-compute expected IMF outputs using Rilling & Flandrin's C reference implementation | 🔲 | code | — | — |
| T-186 | Store as JSON in `validation/reference/` directory | 🔲 | code | — | — |

### Epic 6 · Feature 6.1 · Story 6.1.2: Cross-Language Test Runner

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-187 | Write script that runs same decomposition in Rust, Python, R, Julia, JS, MATLAB, and C++ and compares outputs | 🔲 | code | — | — |
| T-188 | Assert all pairwise differences < 1e-10 | 🔲 | test | — | — |
| T-189 | Run Rust/Python/R/Julia/JS/C++ legs in CI on every PR; MATLAB leg on nightly (requires licence) | 🔲 | code | — | — |

---

## Milestone M6: R + Julia v1.1 / v1.2
**Target:** Late October 2026  
**Release Tags:** `v1.1.0` (R), `v1.2.0` (Julia)  
**Exit Criteria:** R package passes CRAN checks; Julia package registered in General Registry; both bindings expose all 8 algorithms as pure marshalling wrappers; smoke-tests verify correct types and shapes; cross-language validation confirms numerical identity with Rust core  
**Tasks:** T-134 to T-148, T-149 to T-164 (31 tasks)

### Epic 3 · Feature 3.1 · Story 3.1.1: extendr Extension Module Setup

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-134 | Configure `extendr-api` in `ferromode-r/Cargo.toml`; create `#[extendr]` module entry; run `rextendr::document()` to generate `R/ferromode-r-extendr-wrappers.R` — this file is auto-generated and never manually edited | 🔲 | code | — | — |
| T-135 | Expose `EmdConfig` as an `#[extendr]` struct with constructor accepting named R arguments; all defaults come from Rust `impl Default` | 🔲 | code | — | — |
| T-136 | Expose `EnsembleConfig`, `MemdConfig`, `VmdConfig` as `#[extendr]` structs — same principle | 🔲 | code | — | — |

### Epic 3 · Feature 3.1 · Story 3.1.2: Array Marshalling

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-137 | Use `Robj::as_real_slice() -> Option<&[f64]>` for zero-copy input borrow; return `Err` if object is not a real numeric vector | 🔲 | code | — | — |
| T-138 | For multivariate input (MEMD): accept R `matrix` object, use `as_real_vector()` + dimensions attribute to reconstruct channel layout — no R arithmetic | 🔲 | code | — | — |
| T-139 | Validate inputs at marshalling boundary: reject NA, NaN, Inf, zero-length; raise R `stop()` with `EmdError` message | 🔲 | code | — | — |

### Epic 3 · Feature 3.1 · Story 3.1.3: Result Marshalling

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-140 | `ImfCollection` → R named list: `$imfs` (matrix n_imfs × n_samples), `$residue` (numeric vector), `$n_imfs` (integer); assign S3 class `"emd_result"` | 🔲 | code | — | — |
| T-141 | `HilbertResult` → R named list: `$instantaneous_amplitude`, `$instantaneous_frequency` (matrices), `$marginal_spectrum` (numeric vector); class `"hilbert_result"` | 🔲 | code | — | — |
| T-142 | `.reconstruct()` method on `emd_result` calls Rust `ImfCollection::reconstruct()` via extendr — no R summation | 🔲 | code | — | — |

### Epic 3 · Feature 3.1 · Story 3.1.4: Function Wrappers

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-143 | `emd(signal, config)` → marshals `numeric` vector, calls `ferromode::api::emd()`, marshals result | 🔲 | code | — | — |
| T-144 | Same for `eemd`, `ceemd`, `ceemdan`, `iceemdan`, `memd`, `namemd`, `vmd` | 🔲 | code | — | — |
| T-145 | Wrap `EmdError` as R `simpleError` with class `c("emd_error", "error")`; message forwarded verbatim from Rust | 🔲 | code | — | — |

### Epic 3 · Feature 3.1 · Story 3.1.5: Packaging & Distribution

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-146 | Write `DESCRIPTION` with correct `SystemRequirements: Cargo (Rust)`, `LinkingTo: extendr`; pass `R CMD check --as-cran` | 🔲 | code | — | — |
| T-147 | Write minimal `tests/testthat/test-binding.R`: call each function, assert result is a list with correct field names and numeric vector types — no numerical assertions (those are Rust tests) | 🔲 | test | — | — |
| T-148 | Configure GitHub Actions CRAN check on Linux + macOS + Windows; submit to CRAN | 🔲 | code | — | — |

### Epic 4 · Feature 4.1 · Story 4.1.1: C-ABI Shared Library

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-149 | Add `crate-type = ["cdylib"]` to `ferromode-julia/Cargo.toml`; this crate is a thin re-export of `ferromode::ffi` — no new logic | 🔲 | code | — | — |
| T-150 | In `ferromode/src/ffi.rs`: define C-compatible structs (`#[repr(C)]`) mirroring all config and result types; define `extern "C"` functions for every public API: `ferromode_emd()`, `ferromode_eemd()`, `ferromode_ceemdan()`, `ferromode_iceemdan()`, `ferromode_memd()`, `ferromode_namemd()`, `ferromode_vmd()` | 🔲 | code | — | — |
| T-151 | All FFI functions accept raw pointers (`*const f64`, `*mut f64`) and lengths (`usize`); all results returned as heap-allocated `*mut CImfCollection` pointer; caller must call `ferromode_free_result()` — no exceptions cross the FFI boundary, errors returned as null pointer + error code written to out-param | 🔲 | code | — | — |
| T-152 | Run `cbindgen` in CI to auto-generate `ferromode.h` from `ffi.rs`; commit generated header; binding code must not duplicate struct definitions | 🔲 | code | — | — |

### Epic 4 · Feature 4.1 · Story 4.1.2: Julia Package Scaffolding

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-153 | Create `Ferromode.jl` with standard `Project.toml`; `deps` contains only `Libdl` (stdlib) — no algorithm dependencies | 🔲 | code | — | — |
| T-154 | On `__init__`: use `Libdl.find_library(["libferromode"])` or bundle platform-specific artifact via `JLLWrappers`; store library handle | 🔲 | code | — | — |
| T-155 | Define `EmdConfig`, `EnsembleConfig`, `MemdConfig`, `VmdConfig` as Julia `struct` types with fields that exactly mirror the C structs from `ferromode.h` — no new fields, no defaults computed in Julia (defaults come from Rust `impl Default` exposed as `ferromode_default_emd_config()` FFI call) | 🔲 | code | — | — |

### Epic 4 · Feature 4.1 · Story 4.1.3: ccall Wrappers

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-156 | Implement `emd(signal::Vector{Float64}, config::EmdConfig)::ImfCollection`: pin `signal` with `GC.@preserve`, call `ccall((:ferromode_emd, libferromode), Ptr{CImfCollection}, ...)`, wrap result in `ImfCollection` Julia struct, register `finalizer` that calls `ferromode_free_result()` | 🔲 | code | — | — |
| T-157 | Same pattern for `eemd`, `ceemd`, `ceemdan`, `iceemdan` (accepting `EnsembleConfig`) | 🔲 | code | — | — |
| T-158 | Same for `memd`, `namemd` (accepting `Matrix{Float64}` + `MemdConfig`) | 🔲 | code | — | — |
| T-159 | Same for `vmd` (accepting `VmdConfig`) | 🔲 | code | — | — |
| T-160 | Error handling: if FFI returns null, read error code out-param, throw `EmdError(message)` — no error logic, just translation | 🔲 | code | — | — |
| T-161 | Implement `reconstruct(result::ImfCollection)::Vector{Float64}` via `ccall((:ferromode_reconstruct, libferromode), ...)` — calls Rust, no Julia summation | 🔲 | code | — | — |

### Epic 4 · Feature 4.1 · Story 4.1.4: Packaging & Distribution

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-162 | Write `test/runtests.jl` using `@testset`: call each function, assert output types and array sizes — no numerical correctness (those are Rust tests) | 🔲 | test | — | — |
| T-163 | Register in Julia General Registry; configure GitHub Actions to run `Pkg.test()` on Julia 1.9+, Linux + macOS + Windows | 🔲 | code | — | — |
| T-164 | Document the `JLLWrappers` / artifact bundle approach for shipping the compiled library alongside the Julia package | 🔲 | docs | — | — |

---

## Milestone M7: JS/TS + Full Docs v1.3 / v1.4
**Target:** Mid-November 2026  
**Release Tags:** `v1.3.0` (JS/TS), `v1.4.0` (docs/validation complete)  
**Exit Criteria:** npm package published; WASM module works in browser and Node.js; all 8 algorithms callable via Float64Array; TypeScript .d.ts auto-generated; `binding-guide.md` documenting the pure-wrap contract is live; cross-language validation CI covers all 6 bindings  
**Tasks:** T-165 to T-183, T-190 to T-197 (32 tasks)

### Epic 5 · Feature 5.1 · Story 5.1.1: WASM Module Setup

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-165 | Configure `wasm-bindgen` and `wasm-pack` in `emd-wasm/Cargo.toml`; entry point `lib.rs` is `#[wasm_bindgen]` only — no logic | 🔲 | code | — | — |
| T-166 | Expose `WasmEmdConfig` as a `#[wasm_bindgen]` struct with `#[wasm_bindgen(constructor)]` accepting JS object; fields map 1:1 to `EmdConfig` — no defaults computed in JS, defaults from `EmdConfig::default()` in Rust | 🔲 | code | — | — |
| T-167 | Same for `WasmEnsembleConfig`, `WasmMemdConfig`, `WasmVmdConfig` | 🔲 | code | — | — |
| T-168 | Expose `WasmBoundaryCondition`, `WasmStoppingCriterion` as `#[wasm_bindgen]` enums | 🔲 | code | — | — |

### Epic 5 · Feature 5.1 · Story 5.1.2: Array Marshalling

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-169 | In each function wrapper: receive `Float64Array`, use `unsafe { std::slice::from_raw_parts(ptr, len) }` inside WASM linear memory to get `&[f64]` — no copy if possible; WASM memory model allows this safely | 🔲 | code | — | — |
| T-170 | For multivariate input (MEMD): accept flat `Float64Array` + `n_channels: usize`; reconstruct `Vec<Vec<f64>>` by striding — this striding is marshalling, not algorithm logic | 🔲 | code | — | — |
| T-171 | Validate: reject non-finite values at marshalling boundary; throw `EmdError` as JS `Error` with Rust message | 🔲 | code | — | — |

### Epic 5 · Feature 5.1 · Story 5.1.3: Result Accessors

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-172 | `WasmImfCollection` `#[wasm_bindgen]` struct: wraps `ImfCollection`; `.getImf(n: usize) -> Float64Array` returns a view into the underlying buffer; `.getResidue() -> Float64Array`; `.nImfs() -> usize`; `.reconstruct() -> Float64Array` calls Rust | 🔲 | code | — | — |
| T-173 | `WasmHilbertResult`: `.getInstantaneousAmplitude(imf_idx)`, `.getInstantaneousFrequency(imf_idx)`, `.getMarginalSpectrum()` — all return `Float64Array` views | 🔲 | code | — | — |
| T-174 | `WasmDecompositionResult`: `.imfs() -> WasmImfCollection`, `.hilbert() -> WasmHilbertResult`, `.algorithm() -> string`, `.elapsedMs() -> f64` | 🔲 | code | — | — |
| T-175 | Implement `free()` on all result structs; call underlying Rust `drop` — required to avoid WASM memory leaks; document this clearly | 🔲 | code | — | — |

### Epic 5 · Feature 5.1 · Story 5.1.4: Function Wrappers

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-176 | `#[wasm_bindgen] pub fn emd(signal: &[f64], config: &WasmEmdConfig) -> Result<WasmDecompositionResult, JsValue>` — marshal, call Rust, marshal out | 🔲 | code | — | — |
| T-177 | Same for `eemd`, `ceemd`, `ceemdan`, `iceemdan`, `memd`, `namemd`, `vmd` | 🔲 | code | — | — |
| T-178 | Map `EmdError` → `JsValue::from(js_sys::Error::new(&msg))`; no information lost | 🔲 | code | — | — |

### Epic 5 · Feature 5.1 · Story 5.1.5: Build, Types & Distribution

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-179 | `wasm-pack build --target web` and `--target nodejs`; output to `pkg/` | 🔲 | code | — | — |
| T-180 | `wasm-bindgen` auto-generates `.d.ts` for all `#[wasm_bindgen]` exports — review and supplement with hand-written `.d.ts` for the async `init()` pattern only | 🔲 | code | — | — |
| T-181 | Write `package.json` with dual ESM/CJS exports; publish to npm as `ferromode-js` | 🔲 | code | — | — |
| T-182 | Write Vitest test suite: import WASM, call each function, assert output `instanceof Float64Array` and correct `length` — no numerical assertions (those are Rust tests) | 🔲 | test | — | — |
| T-183 | Configure GitHub Actions release: `wasm-pack publish` on tag push | 🔲 | code | — | — |

### Epic 6 · Feature 6.2 · Story 6.2.1: Algorithm Documentation

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-190 | Write mathematical description of each algorithm with LaTeX equations in `docs/algorithms/` | 🔲 | docs | — | — |
| T-191 | Include full bibliographic citations for each algorithm and boundary method | 🔲 | docs | — | — |
| T-192 | Create decision tree diagram: "Which algorithm should I use?" | 🔲 | docs | — | — |
| T-193 | Create comparison table: algorithm properties, use cases, computational cost | 🔲 | docs | — | — |

### Epic 6 · Feature 6.2 · Story 6.2.2: API Documentation

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-194 | Ensure every public Rust function has rustdoc with example | 🔲 | docs | — | — |
| T-195 | Build and publish to docs.rs | 🔲 | docs | — | — |
| T-196 | Write mkdocs site with `binding-guide.md` explaining the pure-wrap contract; narrative guide per binding | 🔲 | docs | — | — |
| T-197 | Create getting-started tutorial for each of the 6 language bindings (Python, R, Julia, JS, MATLAB, C++) | 🔲 | docs | — | — |

---

## Milestone M8: MATLAB/Octave v1.5
**Target:** Early February 2027  
**Release Tag:** `v1.5.0`  
**Exit Criteria:** MEX binaries published for Linux/macOS/Windows; all 8 algorithms callable from MATLAB and GNU Octave; `.mltbx` submitted to MATLAB Add-On Explorer; Octave package submitted to Octave Forge; CI builds MEX on GitHub Actions using Octave as the open licence alternative; cross-language validation extended to include MATLAB leg  
**Tasks:** T-198 to T-218 (21 tasks)

### Epic 7 · Feature 7.1 · Story 7.1.1: MEX Entry Point & Build System

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-198 | Create `ferromode-mex` crate: `crate-type = ["cdylib"]`; implement `mexFunction` as `extern "C"` entry point linking against `ferromode::ffi` — no new logic | 🔲 | code | — | — |
| T-199 | Configure build system: `cc` crate links against `libmex` and `libmx` from MATLAB SDK (path configurable via `MATLAB_ROOT` env var); also support Octave's `liboctave` for open-source compatibility | 🔲 | code | — | — |
| T-200 | Write `build.rs` that detects MATLAB vs Octave installation and sets correct link flags and output extension (`.mexa64` / `.mexmaci64` / `.mexw64` for MATLAB; `.mex` for Octave) | 🔲 | code | — | — |
| T-201 | Implement `mxGetPr()` / `mxGetM()` / `mxGetN()` based input extraction: `*const f64` + dimensions → `&[f64]` slice — marshalling only, no computation | 🔲 | code | — | — |

### Epic 7 · Feature 7.1 · Story 7.1.2: Input Marshalling

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-202 | Extract signal from `prhs[0]`: validate `mxIsDouble()`, `!mxIsComplex()`, column or row vector; get pointer via `mxGetPr()` and length via `mxGetNumberOfElements()` → `&[f64]` | 🔲 | code | — | — |
| T-203 | Parse config struct from `prhs[1]` (optional MATLAB struct): use `mxGetField()` to extract named fields mapping to `FerromodeConfig` fields; all defaults from Rust `impl Default` | 🔲 | code | — | — |
| T-204 | For MEMD/NA-MEMD: accept MATLAB matrix `prhs[0]`; extract via `mxGetPr()` + `mxGetM()` + `mxGetN()` to reconstruct channel layout — marshalling only | 🔲 | code | — | — |
| T-205 | Validate at boundary: non-double, complex, empty, or non-finite inputs → `mexErrMsgIdAndTxt("Ferromode:invalidInput", msg)` with Rust error message verbatim | 🔲 | code | — | — |

### Epic 7 · Feature 7.1 · Story 7.1.3: Result Marshalling

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-206 | Allocate output `mxArray` struct via `mxCreateStructMatrix(1,1,nfields,fieldnames)` with fields `imfs` (matrix), `residue` (vector), `n_imfs` (scalar), `algorithm` (string), `elapsed_ms` (scalar) | 🔲 | code | — | — |
| T-207 | Copy IMF data from Rust `ImfCollection` into `mxCreateDoubleMatrix` allocations via `memcpy` — MATLAB owns output memory | 🔲 | code | — | — |
| T-208 | Implement `ferromode_reconstruct(result_struct)` MEX: accepts the output struct, extracts `imfs` + `residue`, calls `ferromode::api::reconstruct()` in Rust, returns `mxArray` double vector | 🔲 | code | — | — |
| T-209 | Implement `ferromode_hilbert(result_struct)` MEX: accepts output struct, calls `ferromode::api::hilbert()`, returns struct with `instantaneous_amplitude`, `instantaneous_frequency`, `marginal_spectrum` fields | 🔲 | code | — | — |

### Epic 7 · Feature 7.1 · Story 7.1.4: Function Wrappers (all algorithms)

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-210 | `ferromode_emd.mexa64`: MEX entry point → marshal → call `ferromode_emd()` FFI → marshal result | 🔲 | code | — | — |
| T-211 | Same for `ferromode_eemd`, `ferromode_ceemd`, `ferromode_ceemdan`, `ferromode_iceemdan` | 🔲 | code | — | — |
| T-212 | Same for `ferromode_memd`, `ferromode_namemd` (matrix input) | 🔲 | code | — | — |
| T-213 | Same for `ferromode_vmd` | 🔲 | code | — | — |
| T-214 | Wrapper `.m` files for each function providing MATLAB-style `help` documentation and argument name sugar: `ferromode_emd(signal, 'BoundaryCondition', 'periodic', 'MaxIMFs', 8)` — these `.m` files call the MEX binary; they contain no computation | 🔲 | code | — | — |

### Epic 7 · Feature 7.1 · Story 7.1.5: Octave Compatibility & Distribution

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-215 | Test all MEX functions under GNU Octave 8+; fix any `liboctave` API differences (Octave's MEX layer is largely compatible but has minor divergences in struct creation) | 🔲 | test | — | — |
| T-216 | Write MATLAB test script `tests/test_ferromode.m`: call each function, assert output is struct with correct field names and sizes — `assert(size(result.imfs, 1) >= 1)` style; no numerical assertions | 🔲 | test | — | — |
| T-217 | Package as MATLAB toolbox (`.mltbx`) for MATLAB Add-On Explorer submission; package as Octave package (`.tar.gz`) for Octave Forge submission | 🔲 | code | — | — |
| T-218 | Document `MATLAB_ROOT` build configuration in `binding-guide.md`; add CI job that builds MEX on GitHub Actions with MATLAB licence (or Octave as free alternative for open CI) | 🔲 | docs | — | — |

---

## Milestone M9: C++ v1.6
**Target:** Mid-February 2027  
**Release Tag:** `v1.6.0`  
**Exit Criteria:** `ferromode.hpp` header-only wrapper published; CMake `FetchContent` integration works; `cxx` bridge crate published; `vcpkg` and Conan packages submitted; builds clean under GCC 12+, Clang 15+, MSVC 2022 with ASan/UBSan; cross-language validation extended to include C++ leg  
**Tasks:** T-219 to T-237 (19 tasks)

### Epic 8 · Feature 8.1 · Story 8.1.1: C Header & Build Artefacts

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-219 | Confirm `ferromode.h` generated by `cbindgen` (already produced for Julia binding) is sufficient for C++ consumption — no additions needed; it is the C-ABI contract | 🔲 | code | — | — |
| T-220 | Provide `CMakeLists.txt` in `ferromode-cxx/` that fetches the pre-built `libferromode` for the target platform (via `FetchContent` from GitHub Releases) and exposes `ferromode::ferromode` CMake target | 🔲 | code | — | — |
| T-221 | Provide `ferromode.pc` pkg-config file for non-CMake build systems | 🔲 | code | — | — |
| T-222 | Ship pre-built binaries for Linux (x86_64, aarch64), macOS (universal2), Windows (x64) via GitHub Releases as part of the standard release workflow | 🔲 | code | — | — |

### Epic 8 · Feature 8.1 · Story 8.1.2: C++17 Header-Only Wrapper

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-223 | Write `ferromode.hpp`: `namespace ferromode { ... }` with `EmdConfig`, `EnsembleConfig`, `MemdConfig`, `VmdConfig` C++ structs — plain aggregates mirroring the C structs; constructors delegate to `ferromode_default_*_config()` FFI for defaults | 🔲 | code | — | — |
| T-224 | Implement `ImfCollection` RAII wrapper: holds `CImfCollection*`; destructor calls `ferromode_free_result()`; `.imfs() -> std::vector<std::span<const double>>`; `.residue() -> std::span<const double>`; `.reconstruct() -> std::vector<double>` calls Rust FFI — no C++ summation | 🔲 | code | — | — |
| T-225 | Implement free functions: `ferromode::emd(std::span<const double> signal, const EmdConfig& config) -> ImfCollection` — extracts `.data()` + `.size()`, calls `ferromode_emd()` FFI, wraps result in `ImfCollection`; validation delegated to Rust | 🔲 | code | — | — |
| T-226 | Same for `eemd`, `ceemd`, `ceemdan`, `iceemdan` (accepting `EnsembleConfig`) | 🔲 | code | — | — |
| T-227 | Same for `memd`, `namemd` (accepting `std::span<const double*>` channels + `MemdConfig`) | 🔲 | code | — | — |
| T-228 | Same for `vmd` (accepting `VmdConfig`) | 🔲 | code | — | — |
| T-229 | Implement `ferromode::hilbert(const ImfCollection&) -> HilbertResult` RAII wrapper; calls Rust FFI; no C++ DSP | 🔲 | code | — | — |
| T-230 | Error handling: FFI null returns → throw `ferromode::FerromodeError(std::string message)` derived from `std::runtime_error` — no error logic, just translation | 🔲 | code | — | — |

### Epic 8 · Feature 8.1 · Story 8.1.3: `cxx` Bridge (Optional Modern Path)

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-231 | Create `ferromode-cxx` Rust crate using the `cxx` crate: define bridge with `#[cxx::bridge]` exposing all public API functions with C++ idiomatic signatures | 🔲 | code | — | — |
| T-232 | Expose `rust::Vec<f64>` ↔ `std::vector<double>` conversions via `cxx` generated glue — no manual pointer arithmetic | 🔲 | code | — | — |
| T-233 | Document both paths clearly in `binding-guide.md`: header-only (no Cargo required) vs `cxx` bridge (Cargo-integrated projects) | 🔲 | docs | — | — |

### Epic 8 · Feature 8.1 · Story 8.1.4: Testing & Distribution

| Task | Description | Status | Mode | Session Ref | Notes |
|------|-------------|--------|------|-------------|-------|
| T-234 | Write C++ test suite using Catch2: for each function, assert output vector sizes and that `reconstruct()` returns a vector of correct length — no numerical assertions | 🔲 | test | — | — |
| T-235 | CI: build and test with GCC 12+, Clang 15+, MSVC 2022 on Linux/macOS/Windows; run under AddressSanitizer and UndefinedBehaviorSanitizer | 🔲 | code | — | — |
| T-236 | Publish to `vcpkg` registry and `Conan Center Index` for easy integration into existing C++ projects | 🔲 | code | — | — |
| T-237 | Add C++ getting-started tutorial to docs site covering both CMake + `FetchContent` path and `cxx` bridge path | 🔲 | docs | — | — |

---

## Summary Dashboard

| Milestone | Target | Tasks | Completed | % Done |
|-----------|--------|-------|-----------|--------|
| M0: Project Bootstrap | Apr 2026 | 7 | 7 | 100% |
| M1: Spline & Boundary | May 2026 | 42 | 42 | 100% |
| M2: Basic EMD Alpha | Jun 2026 | 33 | 0 | 0% |
| M3: Ensemble Methods | Jul 2026 | 17 | 0 | 0% |
| M4: Multivariate & VMD | Sep 2026 | 26 | 0 | 0% |
| M5: Python v1.0 | Oct 2026 | 38 | 0 | 0% |
| M6: R + Julia v1.1/1.2 | Oct 2026 | 31 | 0 | 0% |
| M7: JS/TS + Docs v1.3/1.4 | Nov 2026 | 32 | 0 | 0% |
| M8: MATLAB v1.5 | Feb 2027 | 21 | 0 | 0% |
| M9: C++ v1.6 | Feb 2027 | 19 | 0 | 0% |
| **TOTAL** | | **237** | **25** | **11%** |

---

## Mode Assignment Summary

| Mode | Task Count | Tasks |
|------|------------|-------|
| code | 178 | T-001–T-016, T-018, T-020–T-022, T-025–T-033, T-035–T-040, T-042–T-048, T-050–T-064, T-067–T-070, T-072, T-075–T-079, T-081–T-082, T-084–T-092, T-094–T-095, T-097–T-101, T-103–T-112, T-113–T-129, T-130–T-132, T-134–T-145, T-146, T-148–T-161, T-163, T-165–T-181, T-183–T-187, T-189, T-198–T-214, T-217, T-218, T-219–T-232, T-235–T-236 |
| test | 38 | T-015, T-019, T-023–T-024, T-028, T-031, T-034, T-037, T-041, T-045, T-049, T-054, T-065–T-066, T-071, T-073–T-074, T-080, T-083, T-088, T-093, T-096, T-102, T-112, T-133, T-147, T-162, T-182, T-188, T-215–T-216, T-234 |
| docs | 21 | T-005, T-007, T-038, T-164, T-190–T-197, T-218, T-233, T-237 |
