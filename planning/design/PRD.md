# Product Requirements Document (PRD)
## Ferromode: A High-Performance Empirical Mode Decomposition Ecosystem

**Version:** 1.0  
**Date:** April 2026  
**Status:** Draft  
**Owner:** Core Engineering  

---

## 1. Executive Summary

Ferromode is an open-source, multi-language empirical mode decomposition ecosystem built on a high-performance Rust core library, with idiomatic bindings for Python, R, Julia, and JavaScript/TypeScript. It implements the full family of EMD algorithms — from original EMD through CEEMDAN, MEMD, NA-MEMD — along with a comprehensive suite of boundary condition strategies, sifting stopping criteria, and auxiliary signal analysis tools (Hilbert transform, instantaneous frequency, marginal spectrum).

The project targets researchers, data scientists, and engineers working with nonlinear and nonstationary time series across domains: geophysics, biomedicine, structural health monitoring, finance, oceanography, and signal processing.

---

## 2. Problem Statement

Existing EMD implementations are fragmented, language-specific, and suffer from:

- **Performance bottlenecks:** Python/R implementations are slow for large datasets or ensemble methods requiring hundreds of trials.
- **Incomplete algorithm coverage:** Most packages implement only one or two variants (e.g., only EMD + EEMD), omitting CEEMDAN, ICEEMDAN, MEMD, NA-MEMD.
- **Poor boundary handling:** Most packages offer only one or two boundary strategies, often poorly documented.
- **No cross-language consistency:** Results differ between R, Python, and MATLAB reference implementations due to differing defaults, spline implementations, and stopping criteria.
- **Limited documentation:** Algorithm parameter choices are opaque; users cannot easily understand trade-offs.
- **No interoperability:** Research pipelines mixing R and Python must duplicate decompositions.

---

## 3. Goals & Non-Goals

### Goals
- Build a single, authoritative, high-performance Rust core (`ferromode`) implementing all major EMD variants
- Provide idiomatic, well-documented language bindings: Python (`ferromode-py`), R (`ferromode-r`), Julia (`Ferromode.jl`), JavaScript/TypeScript (`ferromode-js`), MATLAB/Octave (`ferromode-mex`), C++ (`ferromode-cxx`)
- Achieve numerical equivalence across all language bindings (same algorithm, same defaults → same results)
- Implement ≥8 boundary condition strategies with clear documentation of trade-offs
- Provide comprehensive test suite with reference signals and cross-language validation
- Enable parallel computation (ensemble methods) via Rayon in Rust
- Each language binding is a pure marshalling wrapper — zero algorithmic logic duplicated across languages
- Maintain full API documentation and algorithm-level mathematical documentation

### Non-Goals
- Real-time streaming EMD (deferred to a later release)
- GPU acceleration (out of scope for v1.x)
- A graphical UI application (out of scope; CLI tool only in v1)
- Proprietary/closed-source licensing
- Support for MATLAB directly (MATLAB bindings are out of scope)
- Visualization layers built into any language binding (plotting is the user's responsibility; bindings return arrays)
- High-level convenience APIs in bindings (e.g. `decompose()` helpers, pandas DataFrame output, tidy tibble formatters) — these are user-space concerns, not binding concerns

---

## 4. Target Users

| Persona | Profile | Key Needs |
|---------|---------|-----------|
| **Academic Researcher** | PhD student or faculty in geophysics, neuroscience, climate science | Full algorithm coverage, reproducibility, publication citations |
| **Data Scientist** | Python-first practitioner in industry | Fast Pythonic API, numpy integration, clear docs |
| **Statistician / R User** | Biostatistics, econometrics | R idiomatic API, CRAN publishable |
| **Scientific Programmer** | Julia user doing numerical methods | High performance, composable, Julia type system friendly |
| **MATLAB Engineer** | Research engineer in DSP, geophysics, biomedicine using MATLAB/Octave | Drop-in MEX replacement; same workflow, dramatically faster |
| **C++ Developer** | Embedded systems, real-time DSP, HPC, robotics | Header-only integration, no Python/JVM overhead, RAII safe |
| **Web Developer / SciComm** | JavaScript data viz, browser-based tools | WASM-compiled, lightweight, TypeScript-typed |
| **Open Source Maintainer** | Wants to build on top of Ferromode | Stable Rust API, documented extension points, pure-wrap contract |

---

## 5. Core Requirements

### 5.1 Functional Requirements

#### FR-001: Algorithm Coverage
The system shall implement:
- EMD (Huang et al. 1998)
- EEMD (Wu & Huang 2009)
- CEEMD (Yeh et al. 2010)
- CEEMDAN (Torres et al. 2011)
- ICEEMDAN (Colominas et al. 2014)
- MEMD (Rehman & Mandic 2010)
- NA-MEMD (Rehman & Mandic 2011)
- VMD (Dragomiretskiy & Zosso 2014) — as a comparable reference method

#### FR-002: Boundary Condition Strategies
The system shall implement ≥8 boundary strategies:
- Characteristic Wave (Huang et al. 1998)
- Mirror / Symmetric extension
- Periodic / Cyclic extension (Zeng & He 2004)
- Odd extension
- Slope-based extrapolation
- AR model extrapolation
- SVR-based extension
- Waveform matching / cross-correlation extension
- None (raw endpoints, for comparison)

#### FR-003: Sifting Stopping Criteria
- Standard deviation threshold (Cauchy criterion)
- S-number criterion
- Fixed iteration count
- Energy-difference threshold

#### FR-004: Post-Processing
- Hilbert transform (IMF → analytic signal)
- Instantaneous frequency calculation
- Instantaneous amplitude
- Hilbert marginal spectrum
- IMF orthogonality index
- Signal reconstruction and error metric

#### FR-005: Language Bindings
All six language bindings shall expose:
- The full algorithm set via type-marshalling wrappers (zero algorithmic logic in any binding)
- All config types, boundary strategies, and stopping criteria as host-language-idiomatic types
- Result types that provide access to IMFs and residue as native arrays
- Error forwarding from Rust `EmdError` to each host language's native exception/condition type
- Serialization/deserialization of results (delegated to Rust serde layer)

**Bindings:** Python (PyO3), R (extendr), Julia (ccall/cdylib), JavaScript/TypeScript (wasm-bindgen), MATLAB/Octave (MEX cdylib), C++ (header-only + optional cxx bridge)

**Binding contract enforced in CI:** Cross-language validation confirms each binding produces bit-for-bit identical results to the Rust core. Any divergence indicates logic has been incorrectly added to a binding rather than delegated to the core.

#### FR-006: Numerical Equivalence
Given the same inputs, parameters, and random seed, all language bindings shall produce numerically equivalent results (within floating-point rounding tolerance).

#### FR-007: Parallelism
Ensemble methods (EEMD, CEEMD, CEEMDAN, ICEEMDAN, NA-MEMD) shall support configurable parallelism via thread-pool (Rayon in Rust).

### 5.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-001 | Performance: EMD on 10,000-sample signal | < 50ms (single thread) |
| NFR-002 | Performance: CEEMDAN, 200 trials, 10,000 samples | < 10s (8 threads) |
| NFR-003 | Memory: peak RSS for CEEMDAN 200 trials | < 2GB |
| NFR-004 | Test coverage (Rust core) | ≥ 90% line coverage |
| NFR-005 | Cross-language numerical equivalence | Within 1e-10 relative error |
| NFR-006 | API stability | SemVer; no breaking changes within major version |
| NFR-007 | Documentation | Every public API has docs + example |

---

## 6. Success Metrics

| Metric | Target (12 months post-launch) |
|--------|-------------------------------|
| GitHub stars | ≥ 500 |
| PyPI monthly downloads | ≥ 5,000 |
| CRAN downloads | ≥ 1,000/month |
| Academic citations | ≥ 10 |
| Open issues resolved within 14 days | ≥ 80% |
| Cross-language test suite pass rate | 100% |

---

## 7. Constraints

- All code under MIT or Apache-2.0 license
- No proprietary dependencies
- Rust MSRV: 1.75+
- Python: 3.10+
- R: 4.2+
- Julia: 1.9+
- Node.js: 18+ / Browser (via WASM)
- CI/CD via GitHub Actions
- Documentation hosted on GitHub Pages + crates.io / PyPI / CRAN / Julia General registry

---

## 8. Open Questions

1. Should WASM build support SIMD intrinsics? (performance vs. compatibility trade-off)
2. Should we implement a streaming/online EMD variant in v2?
3. Should Hilbert transform use FFT-based or direct computation?
4. What is the reference implementation for cross-language validation? (Rilling & Flandrin's C code is the de facto standard)
5. How to handle NA-MEMD's choice of hypersphere sampling direction count — user-configurable or auto?
