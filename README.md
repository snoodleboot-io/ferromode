# Ferromode

> High-performance Empirical Mode Decomposition family algorithms in Rust, with bindings for Python, R, Julia, JavaScript/WASM, and MATLAB/Octave.

[![Crates.io](https://img.shields.io/crates/v/ferromode.svg)](https://crates.io/crates/ferromode)
[![PyPI](https://img.shields.io/pypi/v/ferromode.svg)](https://pypi.org/project/ferromode/)
[![CI](https://github.com/snoodleboot-io/ferromode/actions/workflows/ci.yml/badge.svg)](https://github.com/snoodleboot-io/ferromode/actions/workflows/ci.yml)
[![Docs](https://docs.rs/ferromode/badge.svg)](https://docs.rs/ferromode)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Overview

Ferromode is a high-performance, numerically stable implementation of the Empirical Mode Decomposition (EMD) family of algorithms, written in Rust. It provides fast, reliable signal decomposition for non-linear and non-stationary time series data.

The core Rust library (`ferromode`) is built on `ndarray` for efficient multi-dimensional array operations, `rayon` for parallel computation, and `rustfft` for FFT-accelerated interpolation. All bindings share the same battle-tested core, ensuring consistent results across every language.

### Why Ferromode?

- **Speed** — Rust's zero-cost abstractions and parallel execution via Rayon deliver orders-of-magnitude speedups over pure-Python or MATLAB implementations.
- **Numerical stability** — Careful handling of spline interpolation, envelope estimation, and stopping criteria to avoid the common numerical pitfalls in EMD algorithms. Multiple boundary condition strategies are available, including `PalindromeCyclic` — the most stable option for signals with trends or non-zero endpoints — which pre-extends the signal into a symmetric palindrome before sifting and guarantees exact reconstruction.
- **Cross-language** — One core, many languages. Python, R, Julia, JavaScript/WASM, MATLAB/Octave, and C++ all use the same underlying implementation.
- **Production-ready** — Comprehensive test coverage, property-based testing with proptest, and benchmark suites with Criterion.

## Algorithms Supported

| Algorithm | Description | Status |
|-----------|-------------|--------|
| **EMD** | Empirical Mode Decomposition (Huang et al., 1998) | ✅ Implemented |
| **EEMD** | Ensemble EMD (Wu & Huang, 2009) | ✅ Implemented |
| **CEEMD** | Complementary EEMD (Yeh et al., 2010) | ✅ Implemented |
| **CEEMDAN** | Complete EEMD with Adaptive Noise (Torres et al., 2011) | ✅ Implemented |
| **ICEEMDAN** | Improved CEEMDAN (Colominas et al., 2014) | ✅ Implemented |
| **MEMD** | Multivariate EMD (Rehman & Mandic, 2010) | ✅ Implemented |
| **NA-MEMD** | Noise-Assisted MEMD (Rehman & Mandic, 2011) | ✅ Implemented |
| **VMD** | Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014) | ✅ Implemented |

## Language Bindings

| Language | Crate | Package Manager | Status |
|----------|-------|-----------------|--------|
| **Rust** | `ferromode` | [crates.io](https://crates.io/crates/ferromode) | Active |
| **Python** | `ferromode-py` | [PyPI](https://pypi.org/project/ferromode/) | Active |
| **R** | `ferromode-r` | [CRAN](https://cran.r-project.org/) | Active |
| **JavaScript / WASM** | `ferromode-wasm` | [npm](https://www.npmjs.com/package/ferromode) | Active |
| **MATLAB / Octave** | `ferromode-mex` | — | Active |
| **C++** | `ferromode-cpp` | [Conan](https://conan.io/) / [vcpkg](https://vcpkg.io/) | Planned |

## Quick Start (Rust)

Add Ferromode to your `Cargo.toml`:

```toml
[dependencies]
ferromode = "0.1"
```

Basic EMD decomposition:

```rust
use ferromode::Emd;
use ndarray::Array1;

// Create a sample signal: sum of two sinusoids
let n = 1000;
let t: Array1<f64> = Array1::linspace(0.0, 1.0, n);
let signal = &t.mapv(|x| (2.0 * std::f64::consts::PI * 5.0 * x).sin())
    + &t.mapv(|x| 0.5 * (2.0 * std::f64::consts::PI * 20.0 * x).sin());

// Run EMD decomposition
let emd = Emd::default();
let imfs = emd.decompose(&signal)?;

println!("Extracted {} IMFs", imfs.len());
for (i, imf) in imfs.iter().enumerate() {
    println!("  IMF {}: {} samples", i + 1, imf.len());
}
```

## Quick Start (Python)

Install via pip:

```bash
pip install ferromode
```

Basic usage with NumPy:

```python
import numpy as np
from ferromode import EMD

# Create a sample signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

# Run EMD decomposition
emd = EMD()
imfs = emd.decompose(signal)

print(f"Extracted {len(imfs)} IMFs")
for i, imf in enumerate(imfs):
    print(f"  IMF {i+1}: {len(imf)} samples")
```

## Project Structure

Ferromode is organized as a Cargo workspace with five crates:

```
ferromode/
├── crates/
│   ├── ferromode/       # Core Rust library — algorithms, signal processing
│   ├── ferromode-py/    # Python bindings via PyO3
│   ├── ferromode-r/     # R bindings via extendr-api
│   ├── ferromode-wasm/  # WebAssembly bindings via wasm-bindgen
│   └── ferromode-mex/   # MATLAB/Octave MEX bindings
├── Cargo.toml           # Workspace root
├── docs/                # Documentation, design docs, ADRs
└── .github/             # CI workflows, issue templates
```

| Crate | Purpose | Key Dependencies |
|-------|---------|-----------------|
| `ferromode` | Core algorithms, signal processing, math primitives | `ndarray`, `rayon`, `rustfft`, `rand` |
| `ferromode-py` | Python extension module | `pyo3`, `ferromode` |
| `ferromode-r` | R shared library | `extendr-api`, `ferromode` |
| `ferromode-wasm` | WebAssembly module | `wasm-bindgen`, `ferromode` |
| `ferromode-mex` | MEX shared library | `mex`, `ferromode` |

## Development

### Prerequisites

- **Rust 1.75+** — Install via [rustup](https://rustup.rs/)
- **Python 3.9+** — For building and testing Python bindings
- **R 4.0+** — For building and testing R bindings (optional)
- **MATLAB or Octave** — For building and testing MEX bindings (optional)
- **wasm-pack** — For building WASM bindings (`cargo install wasm-pack`)

### Build

```bash
# Build all crates
cargo build --workspace

# Build in release mode
cargo build --workspace --release

# Build a specific crate
cargo build -p ferromode-py
```

### Test

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p ferromode

# Run with output visible
cargo test --workspace -- --nocapture

# Run property-based tests
cargo test -p ferromode -- --ignored
```

### Benchmark

```bash
# Run benchmarks (requires nightly for some features)
cargo bench -p ferromode
```

### Lint & Format

```bash
# Run Clippy linter
cargo clippy --workspace -- -D warnings

# Format all code
cargo fmt --workspace

# Check formatting without modifying
cargo fmt --workspace -- --check
```

### Build Bindings

```bash
# Python (requires maturin)
pip install maturin
cd crates/ferromode-py && maturin develop

# R (requires extendr)
cd crates/ferromode-r && R CMD INSTALL .

# WASM
cd crates/ferromode-wasm && wasm-pack build --target web

# MATLAB/Octave MEX
cd crates/ferromode-mex && cargo build --release
# Copy target/release/libferromode_mex.so (or .dll/.dylib) to MATLAB path
```

## Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct, development process, and how to submit pull requests.

### Key Guidelines

- **TDD is required** — All new code must include tests. We aim for 80%+ line coverage and 70%+ branch coverage.
- **Conventional Commits** — Use `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:` prefixes.
- **Clippy clean** — All code must pass `cargo clippy -- -D warnings`.
- **Formatted** — Run `cargo fmt` before committing.
- **Feature branches** — Use `feat/TICKET-description` or `bugfix/TICKET-description` naming.

### Reporting Issues

- [Bug reports](https://github.com/snoodleboot-io/ferromode/issues/new?template=bug_report.md)
- [Feature requests](https://github.com/snoodleboot-io/ferromode/issues/new?template=feature_request.md)

## License

Ferromode is licensed under the [Apache License, Version 2.0](LICENSE).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Ferromode by you shall be licensed as above, without any additional terms or conditions.

## Citation

If you use Ferromode in academic work, please cite:

```bibtex
@software{ferromode2026,
  author  = {Ferromode Contributors},
  title   = {Ferromode: High-Performance Empirical Mode Decomposition in Rust},
  year    = {2026},
  url     = {https://github.com/snoodleboot-io/ferromode},
  version = {0.1.0},
}
```

A peer-reviewed journal article is in preparation. Check back for the formal citation once published.
