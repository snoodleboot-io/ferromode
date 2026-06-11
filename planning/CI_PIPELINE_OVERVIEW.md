# CI Pipeline Overview

**Last Updated:** 2026-05-29

This document describes what each crate builds, what artifact it produces, where that artifact is
published, and how the CI pipeline gates each step.

---

## Crate Map: What Builds What

Ferromode ships ten crates. Each has a distinct interop target and output artifact.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ferromode workspace                                     │
│                                                                                 │
│  ┌──────────────┐   Core Rust library — pure EMD algorithms, no FFI             │
│  │  ferromode   │   Output: rlib / staticlib                                    │
│  └──────┬───────┘   Published: crates.io                                        │
│         │                                                                       │
│         │  (all interop crates depend on this)                                  │
│         │                                                                       │
│    ┌────┴────────────────────────────────────────────────────┐                  │
│    │                                                         │                  │
│  ┌─┴─────────────┐  ┌────────────────┐  ┌─────────────────┐ │                  │
│  │ ferromode-cxx │  │  ferromode-py  │  │ ferromode-wasm  │ │                  │
│  │ C++ via CXX   │  │ Python/PyO3    │  │ WASM/wasm-pack  │ │                  │
│  │ bridge        │  │ via maturin    │  │                 │ │                  │
│  └───────┬───────┘  └──────┬─────────┘  └──────┬──────────┘ │                  │
│          │                 │                   │            │                  │
│     cdylib/staticlib    .whl files          .wasm + JS     │                  │
│     (no registry)       → PyPI              → npm          │                  │
│                                                            │                  │
│  ┌─────────────────┐  ┌────────────────┐  ┌───────────────┴─┐                  │
│  │   ferromode-r   │  │ ferromode-julia│  │  ferromode-mex  │                  │
│  │ R via extendr   │  │ Julia C-ABI    │  │  MATLAB MEX     │                  │
│  │                 │  │ cdylib         │  │  cdylib         │                  │
│  └────────┬────────┘  └──────┬─────────┘  └──────┬──────────┘                  │
│           │                  │                   │                              │
│      .tar.gz src          .so/.dylib/         .mexa64/                         │
│       → CRAN               .dll               .mexmaci64/                      │
│                          → GH Release         .mexw64                          │
│                                             → GH Release                       │
│                                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐                │
│  │ ferromode_grpc  │  │ferromode_datasets│  │ferromode_validator│               │
│  │ tonic gRPC svc  │  │ test datasets    │  │ result validator │               │
│  └────────┬────────┘  └──────────────────┘  └─────────────────┘                │
│           │                 internal only         internal only                 │
│      Docker image /                                                             │
│      binary release                                                             │
│      (no workflow yet)                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Per-Crate Detail

### `ferromode` → crates.io

- **Crate type:** `lib` (rlib)
- **Published by:** `publish-crates.yml` (triggered by `release.yml`)
- **What it delivers:** The Rust API — the foundational dependency for all other interop crates.
  Users add it as `cargo add ferromode`.
- **CI coverage:** Full — `cargo test`, `cargo clippy`, `cargo fmt` across 3 OSes and 2 toolchains.

### `ferromode-cxx` → no external registry

- **Crate type:** `cdylib` + `staticlib` via CXX bridge
- **Published by:** Not published to a registry — consumed as a build dependency by C++ projects.
- **What it delivers:** A C++ header + shared/static library so C++ projects can call Ferromode
  algorithms without touching Rust tooling.
- **CI coverage:** Compiled and tested in the `rust` job alongside the core crate.

### `ferromode-py` → PyPI

- **Crate type:** `cdylib` compiled by maturin into a Python wheel (`.whl`)
- **Published by:** `publish-pypi.yml`
- **What it delivers:** A native Python extension importable as `import ferromode`. Users install
  with `pip install ferromode`. One wheel per Python version × OS × architecture combination.
- **Wheel matrix required:**
  ```
  OS:             linux-x86_64  linux-aarch64  macos-x86_64  macos-arm64  windows-x86_64
  Python:         3.10  3.11  3.12
  ```
- **CI coverage:** Tested on ubuntu-latest only, 3 Python versions. Missing: macOS, Windows, ARM64.

### `ferromode-wasm` → npm

- **Crate type:** `cdylib` compiled by wasm-pack targeting `wasm32-unknown-unknown`
- **Published by:** `publish-npm.yml` (triggered by `release.yml`)
- **What it delivers:** A `.wasm` binary + JavaScript glue module. Users install with
  `npm install @ferromode/ferromode`. Runs in browsers and Node.js.
- **Architecture:** See [binding-guide.md](../_old_design/binding-guide.md) — pure-wrap contract,
  no algorithm logic in the WASM layer.
- **CI coverage:** Built and tested on ubuntu-latest only.

### `ferromode-r` → CRAN

- **Crate type:** `cdylib` + R package source tarball (`.tar.gz`)
- **Published by:** `publish-cran.yml` — CRAN review takes 3–7 days; it is not instant.
- **What it delivers:** An R package installable via `install.packages("ferromode")`. The Rust
  cdylib is compiled on the user's machine during `R CMD INSTALL`.
- **CI coverage:** Tested via `devtools::check()` in a `rocker/tidyverse` container.

### `ferromode-julia` → GitHub Release assets

- **Crate type:** `cdylib` (produces `libferromode.so` / `.dylib` / `.dll`)
- **Published by:** ⚠️ **No publish workflow exists.** `release.yml` finalize step notes
  "Julia: distributed via GitHub Release assets only" but no job builds or uploads the `.so`.
- **What it delivers:** A shared library that Julia code calls via `ccall`. Users download the
  prebuilt `.so`/`.dylib`/`.dll` from the GitHub Release and configure Julia to find it.
- **CI coverage:** ⚠️ **None.** The crate is never compiled or tested in CI.

### `ferromode-mex` → GitHub Release assets

- **Crate type:** `cdylib` compiled into MATLAB MEX files (`.mexa64`, `.mexmaci64`, `.mexw64`)
- **Published by:** ⚠️ **No publish workflow exists.** Same situation as Julia.
- **What it delivers:** MEX files callable from MATLAB as native functions. Users download the
  `.mex*` file for their platform from the GitHub Release.
- **CI coverage:** ⚠️ **None.** The crate is never compiled or tested in CI.

### `ferromode_grpc` → Docker image / binary (TBD)

- **Crate type:** Binary — tonic gRPC server
- **Published by:** ⚠️ **No publish workflow exists.** Distribution strategy not defined.
- **What it delivers:** A standalone server process. Clients in any language can call Ferromode
  algorithms over gRPC without embedding Rust. Intended for distributed/cloud deployments.
- **CI coverage:** ⚠️ **None.** Never compiled in CI.

### `ferromode_datasets` — internal

- **Crate type:** lib (rlib)
- **Published by:** Not published.
- **What it delivers:** Reference signal datasets used by tests and benchmarks.
- **CI coverage:** ⚠️ Never compiled in CI (not included in `cargo check` command in `ci.yml`).

### `ferromode_validator` — internal

- **Crate type:** lib (rlib)
- **Published by:** Not published.
- **What it delivers:** Result validation utilities for comparing decomposition outputs.
- **CI coverage:** ⚠️ Never compiled in CI.

---

## CI Flow: Current State

```
ci.yml  (on push to main / feat/* / PRs)
─────────────────────────────────────────────────────────────────
  rust      ─────────────────────────────────────┐
  (3 OS × stable + beta)                         │  ALL START
                                                 │  IN PARALLEL
  coverage  ─────────────────────────────────────┤  (no ordering)
  (ubuntu, tarpaulin)                            │
                                                 │
  python    ─────────────────────────────────────┤
  (ubuntu, 3 Python versions)                    │
                                                 │
  wasm      ─────────────────────────────────────┤
  (ubuntu, --dev build)                          │
                                                 │
  r         ─────────────────────────────────────┘
  (ubuntu, rocker container)

⚠ Problems:
  - Coverage runs even when rust tests are failing
  - No gate between core and binding tests
  - ferromode_datasets, ferromode_grpc, ferromode-julia,
    ferromode-mex, ferromode_validator: never touched
```

---

## CI Flow: What It Should Look Like

```
release.yml  (on semver tag push)
─────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────┐
  │  Stage 1 — Parse & verify tag                       │
  │  parse_version                                      │
  └─────────────────────────┬───────────────────────────┘
                            │ (all Stage 2 jobs need this)
  ┌─────────────────────────▼───────────────────────────┐
  │  Stage 2 — Test gate (ALL must green before Stage 3)│
  │                                                     │
  │  test_rust    test_python    test_wasm    test_r    │
  │  (parallel — each needs parse_version)              │
  │                                                     │
  │  ← also needs: test_julia, test_mex, test_grpc →   │  ← MISSING
  └─────────────────────────┬───────────────────────────┘
                            │ (all must succeed)
  ┌─────────────────────────▼───────────────────────────┐
  │  Stage 3 — Build release artifacts                  │
  │  build_python_wheels   (matrix: 5 OS × 3 Python)   │
  │  build_wasm            (--release flag)             │
  │  build_julia_cdylib    (3 OS targets)               │  ← MISSING
  │  build_mex             (3 platform targets)         │  ← MISSING
  │  build_grpc_binary     (linux amd64/arm64)          │  ← MISSING
  └─────────────────────────┬───────────────────────────┘
                            │
  ┌─────────────────────────▼───────────────────────────┐
  │  Stage 4 — Create GitHub Release                    │
  │  create_release                                     │
  │  Uploads: Python wheels, WASM pkg, Julia .so/.dylib │
  │           MEX files, gRPC binary, R tarball         │
  └─────────────────────────┬───────────────────────────┘
                            │
  ┌─────────────────────────▼───────────────────────────┐
  │  Stage 5 — Publish (each requires manual approval)  │
  │                                                     │
  │  ┌──────────────────┐   environment: publish-crates │
  │  │ publish_rust     │   [Requires reviewer approval] │
  │  │ (crates.io)      │   → cargo publish              │
  │  └──────────────────┘                               │
  │                                                     │
  │  ┌──────────────────┐   environment: publish-pypi   │
  │  │ publish_python   │   [Requires reviewer approval] │
  │  │ (PyPI)           │   → twine upload (from Stage 3│
  │  └──────────────────┘      artifacts, not rebuild)  │
  │                                                     │
  │  ┌──────────────────┐   environment: publish-npm    │
  │  │ publish_npm      │   [Requires reviewer approval] │
  │  │ (npm)            │   → npm publish                │
  │  └──────────────────┘                               │
  │                                                     │
  │  ┌──────────────────┐   environment: publish-cran   │
  │  │ publish_r        │   [Requires reviewer approval] │
  │  │ (CRAN)           │   → R CMD check + submit       │
  │  └──────────────────┘                               │
  └─────────────────────────────────────────────────────┘

  "Requires reviewer approval" = GitHub Environment protection rule.
  Each publish_* job declares `environment: publish-<registry>`.
  That environment has Required Reviewers set in GitHub Settings.
  The job pauses and sends a notification; a human approves before it runs.
```

---

## Known Gaps (Work Required)

| Gap | Impact | Fix |
|-----|--------|-----|
| `ferromode-julia` not built in CI | Julia users get zero test coverage | Add `test_julia` job, build cdylib on ubuntu/macos/windows, upload to GH Release |
| `ferromode-mex` not built in CI | MATLAB users get zero test coverage | Add `test_mex` job, build `.mex*` files, upload to GH Release |
| `ferromode_grpc` not compiled in CI | gRPC server may be broken silently | Add to `cargo check` and `cargo test` in `ci.yml`; decide distribution strategy |
| `ferromode_datasets`, `ferromode_validator` not in CI | Workspace compile errors invisible | Add `-p ferromode_datasets -p ferromode_validator` to `cargo check`/`cargo test` |
| `publish-pypi.yml` publishes untested artifacts | Tested wheels discarded; fresh untested wheel published | Build once in Stage 3, upload artifacts, download in Stage 5 `publish` job |
| No ARM64 runners | ARM64 wheels missing; release notes incorrect | Add `ubuntu-22.04-arm` and `macos-latest` (Apple Silicon) to wheel matrix |
| WASM build mode mismatch | CI tests debug build; release ships release build | Add a `--release` CI build check to match release behavior |
| Manual approval missing | Any tag triggers full publish with no human gate | Add GitHub Environment protection rules with required reviewers |
| Python version mismatch | 3.12 untested in publish flow | Align python-version matrix to `["3.10", "3.11", "3.12"]` in both ci.yml and publish-pypi.yml |

---

## Artifact Handoff Pattern (correct approach for PyPI)

The current `publish-pypi.yml` bug: tested wheels are rebuilt from scratch in the publish job
and the tested artifacts are never published. The correct pattern is:

```
Stage 3 — build_python_wheels (matrix job)
│
│  maturin build --release --out ../../dist/
│  pip install dist/ferromode*.whl && pytest tests/   ← test the exact wheel
│  upload-artifact: dist/ferromode-*.whl              ← save it
│
Stage 5 — publish_python (single job, protected environment)
│
│  download-artifact: dist/ferromode-*.whl            ← same artifact tested above
│  twine upload dist/ferromode-*.whl                  ← publish exactly what was tested
```

This guarantees that what is tested is what is published, on every platform.
