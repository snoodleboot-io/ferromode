# Contributing to Ferromode

Welcome to Ferromode! We're building a high-performance statistical modeling library in Rust with bindings for Python, R, WebAssembly, and MEX (MATLAB). We appreciate your interest in contributing.

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Issues

- Search existing issues before creating a new one
- Use the issue template when available
- Include reproduction steps, environment details, and expected vs. actual behavior
- Tag issues appropriately (bug, enhancement, documentation, question)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (see naming conventions below)
3. Write tests first (TDD is required)
4. Implement the feature or fix
5. Ensure all tests pass and lints are clean
6. Submit a PR with a clear description of changes

### Code Review

- All PRs require at least one review before merging
- Address review feedback promptly
- Keep discussions respectful and constructive
- Squash commits before merging if requested

## Development Workflow

### Branch Naming

Use the following format for all branches:

- `feat/PROJ-NNN-short-description` — New features
- `bugfix/PROJ-NNN-fix-description` — Bug fixes

Examples:
- `feat/PROJ-001-add-linear-mixed-model`
- `bugfix/PROJ-042-fix-convergence-criterion`

### Commits

- One commit per logical task
- Use [Conventional Commits](https://www.conventionalcommits.org/) format:
  - `feat: add support for Bayesian estimation`
  - `fix: resolve NaN issue in variance calculation`
  - `test: add convergence edge case coverage`
  - `docs: update API reference for LMM`
  - `refactor: extract solver into separate module`

### Test-Driven Development (TDD)

TDD is **required** for all new code:

1. Write the test first, describing expected behavior
2. Run the test — it should fail
3. Implement the minimum code to make it pass
4. Refactor while keeping tests green

### Running Tests

```bash
# Run all tests across the workspace
cargo test --workspace

# Run tests for a specific crate
cargo test -p ferromode

# Run with output
cargo test --workspace -- --nocapture
```

### Running Lints

```bash
# Run Clippy with project-wide lint rules
cargo clippy --workspace -- -D warnings
```

### Formatting

```bash
# Format all code
cargo fmt --all

# Check formatting without modifying
cargo fmt --all -- --check
```

## Project Structure

Ferromode is a Cargo workspace with five crates:

| Crate | Purpose |
|-------|---------|
| `ferromode` | Core Rust library — statistical models, solvers, and algorithms |
| `ferromode-py` | Python bindings via PyO3 |
| `ferromode-r` | R bindings via extendr |
| `ferromode-wasm` | WebAssembly bindings for browser/Node.js usage |
| `ferromode-mex` | MEX bindings for MATLAB integration |

```
ferromode/
├── Cargo.toml              # Workspace manifest
├── ferromode/              # Core library
│   ├── Cargo.toml
│   └── src/
├── ferromode-py/           # Python bindings
│   ├── Cargo.toml
│   └── src/
├── ferromode-r/            # R bindings
│   ├── Cargo.toml
│   └── src/
├── ferromode-wasm/         # WebAssembly bindings
│   ├── Cargo.toml
│   └── src/
└── ferromode-mex/          # MEX bindings
    ├── Cargo.toml
    └── src/
```

## Engineering Practices

Ferromode follows established engineering practices:

- **TDD (Test-Driven Development)** — Tests before implementation
- **ATDD (Acceptance Test-Driven Development)** — Define acceptance criteria before writing code
- **DDD (Domain-Driven Design)** — Model the statistical domain explicitly; use ubiquitous language
- **Clean Code** — Readable, maintainable, self-documenting code
- **Clean Architecture** — Separate concerns; depend on abstractions; keep the core pure
- **SOLID Principles** — Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion

## Getting Started

1. Install Rust (1.75 or later): `rustup install stable`
2. Clone the repository
3. Run `cargo build --workspace` to verify the setup
4. Run `cargo test --workspace` to ensure all tests pass
5. Pick an issue labeled `good first issue` to get started

## Questions?

Open a discussion or tag a maintainer in an issue. We're happy to help you get started.
