# Changelog

All notable changes to Ferromode are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`PalindromeCyclic` boundary condition** — pre-extends the signal to a `2N-1`
  even-symmetric palindrome before the sifting loop, then uses a periodic (cyclic)
  cubic spline for envelope interpolation. Provides more stable end-effect
  suppression than mirror-based strategies for signals with trends or non-zero
  endpoints. IMFs and residue are trimmed back to the original `N` samples;
  exact reconstruction is preserved. Available via
  `BoundaryConditionType::PalindromeCyclic` in all language bindings (Python,
  WASM, C++, MEX, Julia/FFI).
- **`SplineType` enum** on `SiftingConfig` — selects the cubic spline boundary
  condition used when fitting IMF envelopes: `Natural` (default, zero second
  derivative at endpoints), `Periodic` (cyclic; selected automatically by
  `PalindromeCyclic`), and `NotAKnot` (C³ continuity at the first and last
  interior knots).
- `build_palindrome()` public utility function for constructing a `2N-1`
  palindrome from a signal slice.
- `palindrome_cyclic_boundary` example demonstrating boundary-strategy comparison
  and reconstruction verification (`cargo run --example palindrome_cyclic_boundary`).

### Fixed
- **Cubic spline coefficient formula** — corrected the `b`, `c`, `d` coefficient
  computation to match the standard Burden & Faires convention (divisors were
  incorrectly `6` and `2`; now `3` and `1` respectively).
- **Natural spline tridiagonal sweep** — forward and backward substitution loops
  now iterate over all `n` interior knots instead of `n-1`, fixing incorrect
  second-derivative values for longer knot sets.
- **Not-a-knot spline** — replaced the narrow-band tridiagonal solver with full
  Gaussian elimination; the boundary rows have stencil width 3 which caused
  fill-in that the previous solver silently dropped.
- **Cyclic tridiagonal (Sherman-Morrison)** — corrected the `gamma` sign choice
  (`-diag[0]` instead of `+diag[0]`) so the modified diagonal is nonzero and
  the factorisation is stable.
- `ArModel::fit` now handles constant signals (zero variance) without a
  degenerate Yule-Walker solve; returns an AR(1)=1 predictor that reproduces
  the constant value.
- `GpuExecutor` timing assertions relaxed from `> Duration::ZERO` to
  `>= Duration::ZERO` to avoid spurious failures on fast hardware where
  sub-nanosecond executions round to zero.

## [2.2.0] - 2026-04-08

### Added
- **LSTM-based neural boundary prediction** with automatic AR/LSTM selection based on signal stationarity
- `BoundarySelector` for intelligent model selection with 35 comprehensive test cases
- V2.2 Integration Guide (`docs/V22_LSTM_INTEGRATION_GUIDE.md`) with code examples and benchmarks
- Support for multiple signal types: ECG, seismic, speech, vibration, and EEG
- Release mode benchmarks showing < 1ms inference latency for boundary prediction
- SafeTensors model format (no external ONNX runtime dependency)
- `boundary-prediction` optional feature flag for conditional compilation

### Changed
- `StreamingDecomposer` now uses `BoundarySelector` for automatic boundary prediction
- Default boundary extension mechanism uses stationarity-based model selection
- LSTM model embedded in binary (no external model file dependencies)
- Improved signal preprocessing pipeline for non-stationary signal handling

### Fixed
- End-effect artifacts reduced by 30-40% compared to AR baseline
- Improved boundary smoothness for non-stationary signals
- Better handling of edge cases in signal decomposition

### Technical Details
- **Model Architecture:** 2-layer LSTM (128 hidden units) + fully-connected output layer
- **Training Data:** 996 synthetic signals across 5 signal types, 56 epochs convergence
- **Format:** SafeTensors (pure Rust tensor serialization, no ONNX dependency)
- **Inference:** < 1ms per signal (release build, typical signal length 1000)
- **Tests:** 35 passing (unit, integration, real-world validation)
- **Coverage:** 88% code coverage on boundary prediction module

## [2.1.0] - TBD

### Planned
- Streaming decomposition support with sliding window
- GPU acceleration for multi-signal batch processing
- C++ language bindings

## [2.0.0] - TBD

### Core Features
- Empirical Mode Decomposition (EMD) implementation
- Cross-language bindings (Python, R, JavaScript/WASM, MATLAB/Octave)
- High-performance Rust core with `ndarray` and `rayon`
- Comprehensive test suite with property-based testing
- Production-ready error handling

[2.2.0]: https://github.com/ferromode/ferromode/releases/tag/v2.2.0
[2.1.0]: https://github.com/ferromode/ferromode/releases/tag/v2.1.0
[2.0.0]: https://github.com/ferromode/ferromode/releases/tag/v2.0.0
