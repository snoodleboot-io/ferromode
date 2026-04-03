import { useState, useEffect, useRef } from "react";

// ─── DATA ────────────────────────────────────────────────────────────────────

const releases = [
  { id: "v0.1", label: "v0.1", name: "Foundation", color: "#2dd4bf", date: "Apr–May 2026" },
  { id: "v0.2", label: "v0.2", name: "Core EMD Alpha", color: "#60a5fa", date: "Jun 2026" },
  { id: "v0.3", label: "v0.3", name: "Ensemble Beta", color: "#a78bfa", date: "Jul 2026" },
  { id: "v0.5", label: "v0.5", name: "Multivariate & VMD", color: "#f472b6", date: "Sep 2026" },
  { id: "v1.0", label: "v1.0", name: "Python Binding", color: "#fb923c", date: "Oct 2026" },
  { id: "v1.1", label: "v1.1", name: "R + Julia Bindings", color: "#facc15", date: "Nov 2026" },
  { id: "v1.3", label: "v1.3", name: "JS/TS + Docs", color: "#4ade80", date: "Dec 2026" },
  { id: "v1.5", label: "v1.5", name: "MATLAB/Octave Binding", color: "#e879f9", date: "Feb 2027" },
  { id: "v1.6", label: "v1.6", name: "C++ Binding", color: "#f43f5e", date: "Feb 2027" },
];

const milestones = [
  { id: "M0", label: "M0", name: "Project Bootstrap", release: "v0.1", epics: ["E1"], tasks: "T-001–T-007" },
  { id: "M1", label: "M1", name: "Spline & Boundary Foundation", release: "v0.1", epics: ["E1"], tasks: "T-008–T-049" },
  { id: "M2", label: "M2", name: "Basic EMD Alpha", release: "v0.2", epics: ["E1"], tasks: "T-050–T-112" },
  { id: "M3", label: "M3", name: "Ensemble Methods Beta", release: "v0.3", epics: ["E1"], tasks: "T-067–T-083" },
  { id: "M4", label: "M4", name: "Multivariate & VMD", release: "v0.5", epics: ["E1"], tasks: "T-084–T-112" },
  { id: "M5", label: "M5", name: "Python v1.0", release: "v1.0", epics: ["E2", "E6"], tasks: "T-113–T-189" },
  { id: "M6", label: "M6", name: "R + Julia v1.1/v1.2", release: "v1.1", epics: ["E3", "E4"], tasks: "T-134–T-164" },
  { id: "M7", label: "M7", name: "JS + Full Docs v1.3/v1.4", release: "v1.3", epics: ["E5", "E6"], tasks: "T-165–T-197" },
  { id: "M8", label: "M8", name: "MATLAB/Octave v1.5", release: "v1.5", epics: ["E7"], tasks: "T-198–T-218" },
  { id: "M9", label: "M9", name: "C++ v1.6", release: "v1.6", epics: ["E8"], tasks: "T-219–T-237" },
];

const epics = [
  { id: "E1", label: "Epic 1", name: "Rust Core Foundation", color: "#2dd4bf", features: ["F1.1","F1.2","F1.3","F1.4","F1.5","F1.6","F1.7"], taskCount: 112, weeks: 22,
    contract: "ALL algorithmic logic lives here. No algorithm code in any binding." },
  { id: "E2", label: "Epic 2", name: "Python Binding (PyO3)", color: "#60a5fa", features: ["F2.1"], taskCount: 21, weeks: 3,
    contract: "Pure marshalling: np.ndarray ↔ &[f64], call emd_core::api, return PyObject. Zero algorithm logic." },
  { id: "E3", label: "Epic 3", name: "R Binding (extendr)", color: "#a78bfa", features: ["F3.1"], taskCount: 15, weeks: 3,
    contract: "Pure marshalling: Robj ↔ &[f64], call emd_core::api, return named R list. Zero algorithm logic." },
  { id: "E4", label: "Epic 4", name: "Julia Binding (ccall)", color: "#f472b6", features: ["F4.1"], taskCount: 16, weeks: 3,
    contract: "Pure marshalling: ccall into C ABI cdylib, Array{Float64} ↔ *const f64. Zero algorithm logic." },
  { id: "E5", label: "Epic 5", name: "JS/TS Binding (WASM)", color: "#fb923c", features: ["F5.1"], taskCount: 19, weeks: 3,
    contract: "Pure marshalling: Float64Array ↔ WASM linear memory, call emd_core via wasm-bindgen. Zero algorithm logic." },
  { id: "E6", label: "Epic 6", name: "Docs & Validation", color: "#4ade80", features: ["F6.1","F6.2"], taskCount: 14, weeks: 6,
    contract: "Cross-language validation proves all bindings produce identical numerical output to the Rust core." },
  { id: "E7", label: "Epic 7", name: "MATLAB/Octave Binding (MEX)", color: "#e879f9", features: ["F7.1"], taskCount: 21, weeks: 4,
    contract: "Pure marshalling via MEX entry point: mxGetPr() → &[f64], call ferromode::ffi, return mxArray. Reuses Julia C-ABI cdylib. Zero algorithm logic." },
  { id: "E8", label: "Epic 8", name: "C++ Binding (ferromode-cxx)", color: "#f43f5e", features: ["F8.1"], taskCount: 19, weeks: 4,
    contract: "Header-only C++17 wrapper over the C-ABI layer (ferromode.h). RAII ImfCollection wrapper + cxx bridge option. Zero algorithm logic." },
];

const features = [
  { id: "F1.1", epic: "E1", name: "Core Infrastructure & Types", stories: ["S1.1.1","S1.1.2","S1.1.3","S1.1.4"], milestone: "M0" },
  { id: "F1.2", epic: "E1", name: "Boundary Condition Strategies", stories: ["S1.2.1","S1.2.2","S1.2.3","S1.2.4","S1.2.5","S1.2.6","S1.2.7"], milestone: "M1" },
  { id: "F1.3", epic: "E1", name: "Core EMD Algorithm", stories: ["S1.3.1","S1.3.2","S1.3.3"], milestone: "M2" },
  { id: "F1.4", epic: "E1", name: "Ensemble EMD Family", stories: ["S1.4.1","S1.4.2","S1.4.3","S1.4.4"], milestone: "M3" },
  { id: "F1.5", epic: "E1", name: "Multivariate EMD", stories: ["S1.5.1","S1.5.2","S1.5.3"], milestone: "M4" },
  { id: "F1.6", epic: "E1", name: "Hilbert Transform & Post-Processing", stories: ["S1.6.1","S1.6.2"], milestone: "M2" },
  { id: "F1.7", epic: "E1", name: "VMD Implementation", stories: ["S1.7.1"], milestone: "M4" },
  { id: "F2.1", epic: "E2", name: "PyO3 Marshalling Layer", stories: ["S2.1.1","S2.1.2","S2.1.3","S2.1.4","S2.1.5"], milestone: "M5" },
  { id: "F3.1", epic: "E3", name: "extendr Marshalling Layer", stories: ["S3.1.1","S3.1.2","S3.1.3","S3.1.4","S3.1.5"], milestone: "M6" },
  { id: "F4.1", epic: "E4", name: "C-ABI + Julia ccall Layer", stories: ["S4.1.1","S4.1.2","S4.1.3","S4.1.4"], milestone: "M6" },
  { id: "F5.1", epic: "E5", name: "wasm-bindgen Marshalling Layer", stories: ["S5.1.1","S5.1.2","S5.1.3","S5.1.4","S5.1.5"], milestone: "M7" },
  { id: "F6.1", epic: "E6", name: "Cross-Language Validation Suite", stories: ["S6.1.1","S6.1.2"], milestone: "M5" },
  { id: "F6.2", epic: "E6", name: "Documentation Site", stories: ["S6.2.1","S6.2.2"], milestone: "M7" },
  { id: "F7.1", epic: "E7", name: "MEX Marshalling Layer", stories: ["S7.1.1","S7.1.2","S7.1.3","S7.1.4","S7.1.5"], milestone: "M8" },
  { id: "F8.1", epic: "E8", name: "C++ Marshalling Layer", stories: ["S8.1.1","S8.1.2","S8.1.3","S8.1.4"], milestone: "M9" },
];

const stories = [
  { id: "S1.1.1", feature: "F1.1", name: "Workspace & Project Scaffolding", tasks: ["T-001","T-002","T-003","T-004","T-005","T-006","T-007"], ac: "cargo build --workspace succeeds on all platforms; CI green" },
  { id: "S1.1.2", feature: "F1.1", name: "Core Type System", tasks: ["T-008","T-009","T-010","T-011","T-012","T-013","T-014","T-015"], ac: "ImfCollection::reconstruct() returns signal within 1e-12" },
  { id: "S1.1.3", feature: "F1.1", name: "Error Handling", tasks: ["T-016","T-017","T-018","T-019"], ac: "All error paths tested; actionable error messages" },
  { id: "S1.1.4", feature: "F1.1", name: "Cubic Spline Engine", tasks: ["T-020","T-021","T-022","T-023","T-024","T-025"], ac: "Spline matches scipy within 1e-10 on all reference signals" },
  { id: "S1.2.1", feature: "F1.2", name: "Boundary Trait & Registry", tasks: ["T-026","T-027","T-028"], ac: "All strategies interchangeable via enum; no NaN/Inf produced" },
  { id: "S1.2.2", feature: "F1.2", name: "Characteristic Wave Extension", tasks: ["T-029","T-030","T-031"], ac: "Handles signals with <4 extrema; tested on sine + chirp" },
  { id: "S1.2.3", feature: "F1.2", name: "Mirror / Symmetric Extension", tasks: ["T-032","T-033","T-034"], ac: "Endpoint is local extremum in extended signal" },
  { id: "S1.2.4", feature: "F1.2", name: "Periodic / Cyclic Extension (Zeng & He 2004)", tasks: ["T-035","T-036","T-037","T-038"], ac: "No endpoint artifacts on periodic test signals; Zeng 2004 cited in source" },
  { id: "S1.2.5", feature: "F1.2", name: "Slope-Based Extension", tasks: ["T-039","T-040","T-041"], ac: "Tested on monotone-end signals" },
  { id: "S1.2.6", feature: "F1.2", name: "AR Model Extension", tasks: ["T-042","T-043","T-044","T-045"], ac: "Near-perfect extension on AR(2) synthetic signals" },
  { id: "S1.2.7", feature: "F1.2", name: "Waveform Matching Extension", tasks: ["T-046","T-047","T-048","T-049"], ac: "Cross-correlation match beats mirror on quasi-periodic signals" },
  { id: "S1.3.1", feature: "F1.3", name: "Extrema Detection", tasks: ["T-050","T-051","T-052","T-053","T-054"], ac: "Plateau extrema handled; tested on step/sawtooth/constant" },
  { id: "S1.3.2", feature: "F1.3", name: "Sifting Engine", tasks: ["T-055","T-056","T-057","T-058","T-059","T-060"], ac: "All 4 stopping criteria implemented; infinite loop guard in place" },
  { id: "S1.3.3", feature: "F1.3", name: "Basic EMD", tasks: ["T-061","T-062","T-063","T-064","T-065","T-066"], ac: "Reconstruction error < 1e-12; reference signal tests pass" },
  { id: "S1.4.1", feature: "F1.4", name: "EEMD", tasks: ["T-067","T-068","T-069","T-070","T-071"], ac: "Parallel trial execution; noise cancels in average" },
  { id: "S1.4.2", feature: "F1.4", name: "CEEMD", tasks: ["T-072","T-073","T-074"], ac: "Exact reconstruction; RMS noise matches EEMD" },
  { id: "S1.4.3", feature: "F1.4", name: "CEEMDAN", tasks: ["T-075","T-076","T-077","T-078","T-079","T-080"], ac: "Reconstruction error < 1e-10; 200 trials on 10K samples < 10s (8 threads)" },
  { id: "S1.4.4", feature: "F1.4", name: "ICEEMDAN", tasks: ["T-081","T-082","T-083"], ac: "Residual noise reduced vs. CEEMDAN" },
  { id: "S1.5.1", feature: "F1.5", name: "Hypersphere Direction Sampling", tasks: ["T-084","T-085","T-086","T-087","T-088"], ac: "KS test passes for uniform distribution of projections" },
  { id: "S1.5.2", feature: "F1.5", name: "MEMD", tasks: ["T-089","T-090","T-091","T-092","T-093"], ac: "Mode-alignment property verified on synthetic hexavariate signal" },
  { id: "S1.5.3", feature: "F1.5", name: "NA-MEMD", tasks: ["T-094","T-095","T-096"], ac: "Noise suppression better than plain MEMD" },
  { id: "S1.6.1", feature: "F1.6", name: "Hilbert Transform", tasks: ["T-097","T-098","T-099","T-100","T-101","T-102"], ac: "Validated on pure tone and AM signal" },
  { id: "S1.6.2", feature: "F1.6", name: "Marginal Spectrum & Metrics", tasks: ["T-103","T-104","T-105","T-106"], ac: "All metrics implemented and documented" },
  { id: "S1.7.1", feature: "F1.7", name: "VMD Core", tasks: ["T-107","T-108","T-109","T-110","T-111","T-112"], ac: "Matches MATLAB reference output from Dragomiretskiy & Zosso" },
  { id: "S2.1.1", feature: "F2.1", name: "PyO3 Module Setup & Config Types", tasks: ["T-113","T-114","T-115","T-116","T-117"], ac: "All config types constructible from Python kwargs; map 1:1 to Rust structs; no defaults computed in Python" },
  { id: "S2.1.2", feature: "F2.1", name: "Input Array Marshalling", tasks: ["T-118","T-119","T-120"], ac: "np.ndarray → &[f64] zero-copy when C-contiguous; non-finite inputs raise ValueError with Rust message" },
  { id: "S2.1.3", feature: "F2.1", name: "Result Object Marshalling", tasks: ["T-121","T-122","T-123"], ac: ".imfs returns np.ndarray; .reconstruct() delegates to Rust; no Python arithmetic" },
  { id: "S2.1.4", feature: "F2.1", name: "Function Wrappers (all algorithms)", tasks: ["T-124","T-125","T-126","T-127","T-128","T-129"], ac: "All 8 algorithms callable from Python; GIL released during ensemble methods; EmdError → Python ValueError" },
  { id: "S2.1.5", feature: "F2.1", name: "Packaging & Distribution", tasks: ["T-130","T-131","T-132","T-133"], ac: "pip install works; wheels for Linux/macOS/Windows; smoke tests pass; py.typed marker present" },

  { id: "S3.1.1", feature: "F3.1", name: "extendr Module Setup & Config Types", tasks: ["T-134","T-135","T-136"], ac: "All config types constructible from R; auto-generated wrappers not manually edited" },
  { id: "S3.1.2", feature: "F3.1", name: "Input Array Marshalling", tasks: ["T-137","T-138","T-139"], ac: "as_real_slice() zero-copy; NA/NaN/Inf inputs raise R stop() with Rust message" },
  { id: "S3.1.3", feature: "F3.1", name: "Result Object Marshalling", tasks: ["T-140","T-141","T-142"], ac: "Results are named R lists with S3 class; .reconstruct() calls Rust; no R summation" },
  { id: "S3.1.4", feature: "F3.1", name: "Function Wrappers (all algorithms)", tasks: ["T-143","T-144","T-145"], ac: "All 8 algorithms callable from R; EmdError → R simpleError with correct class" },
  { id: "S3.1.5", feature: "F3.1", name: "Packaging & CRAN Submission", tasks: ["T-146","T-147","T-148"], ac: "R CMD check --as-cran passes; smoke tests check structure, not numerics; submitted to CRAN" },

  { id: "S4.1.1", feature: "F4.1", name: "C-ABI Shared Library & Header", tasks: ["T-149","T-150","T-151","T-152"], ac: "cdylib compiled; cbindgen generates emd_core.h; all public API exported; errors via out-param not exceptions" },
  { id: "S4.1.2", feature: "F4.1", name: "Julia Package Scaffolding", tasks: ["T-153","T-154","T-155"], ac: "Ferromode.jl loads cdylib at init; config structs mirror C structs exactly; no Julia-side defaults" },
  { id: "S4.1.3", feature: "F4.1", name: "ccall Wrappers (all algorithms)", tasks: ["T-156","T-157","T-158","T-159","T-160","T-161"], ac: "All 8 algorithms callable; GC.@preserve used correctly; finalizers registered; reconstruct() calls Rust" },
  { id: "S4.1.4", feature: "F4.1", name: "Packaging & Registry", tasks: ["T-162","T-163","T-164"], ac: "Registered in Julia General; @testset checks types/sizes not numerics; JLL artifact ships the cdylib" },

  { id: "S5.1.1", feature: "F5.1", name: "WASM Module Setup & Config Types", tasks: ["T-165","T-166","T-167","T-168"], ac: "wasm-pack builds successfully; all config types constructible from JS; no JS defaults computation" },
  { id: "S5.1.2", feature: "F5.1", name: "Input Array Marshalling", tasks: ["T-169","T-170","T-171"], ac: "Float64Array → WASM memory efficiently; non-finite inputs throw JS Error with Rust message" },
  { id: "S5.1.3", feature: "F5.1", name: "Result Accessors", tasks: ["T-172","T-173","T-174","T-175"], ac: "getImf(n) returns Float64Array view; reconstruct() delegates to Rust; free() must be called; no JS math" },
  { id: "S5.1.4", feature: "F5.1", name: "Function Wrappers (all algorithms)", tasks: ["T-176","T-177","T-178"], ac: "All 8 algorithms callable; EmdError → JS Error; no TS arithmetic" },
  { id: "S5.1.5", feature: "F5.1", name: "Build, Types & Distribution", tasks: ["T-179","T-180","T-181","T-182","T-183"], ac: "npm published; dual ESM/CJS; auto-generated .d.ts; Vitest smoke tests check types/lengths only" },
  { id: "S6.1.1", feature: "F6.1", name: "Reference Signal Library", tasks: ["T-184","T-185","T-186"], ac: "JSON reference outputs from Rilling-Flandrin C implementation in validation/reference/" },
  { id: "S6.1.2", feature: "F6.1", name: "Cross-Language Test Runner", tasks: ["T-187","T-188","T-189"], ac: "All pairwise differences < 1e-10 across Rust, Python, R, Julia, JS, MATLAB, C++; runs in CI on every PR" },
  { id: "S6.2.1", feature: "F6.2", name: "Algorithm Documentation", tasks: ["T-190","T-191","T-192","T-193"], ac: "Every algorithm has mathematical docs + full citations + decision tree" },
  { id: "S6.2.2", feature: "F6.2", name: "API Documentation", tasks: ["T-194","T-195","T-196","T-197"], ac: "docs.rs published; binding-guide.md (pure-wrap contract); tutorials for all 6 language bindings" },

  { id: "S7.1.1", feature: "F7.1", name: "MEX Entry Point & Build System", tasks: ["T-198","T-199","T-200","T-201"], ac: "ferromode_emd.mexa64 builds on Linux/macOS/Windows; links libmex + libmx; Octave alternative works" },
  { id: "S7.1.2", feature: "F7.1", name: "Input Marshalling", tasks: ["T-202","T-203","T-204","T-205"], ac: "mxGetPr() → &[f64] zero-copy; non-double/complex/empty inputs → mexErrMsgIdAndTxt with Rust message" },
  { id: "S7.1.3", feature: "F7.1", name: "Result Marshalling", tasks: ["T-206","T-207","T-208","T-209"], ac: "Output is MATLAB struct with imfs matrix, residue vector, algorithm string; reconstruct() calls Rust" },
  { id: "S7.1.4", feature: "F7.1", name: "Function Wrappers (all algorithms)", tasks: ["T-210","T-211","T-212","T-213","T-214"], ac: "All 8 algorithms callable from MATLAB/Octave; .m helper files provide named argument sugar only" },
  { id: "S7.1.5", feature: "F7.1", name: "Octave Compatibility & Distribution", tasks: ["T-215","T-216","T-217","T-218"], ac: "Works under Octave 8+; .mltbx submitted; .m test script checks structure not numerics" },

  { id: "S8.1.1", feature: "F8.1", name: "C Header & Build Artefacts", tasks: ["T-219","T-220","T-221","T-222"], ac: "ferromode.h sufficient for C++ consumption; CMake FetchContent target works; pkg-config file present" },
  { id: "S8.1.2", feature: "F8.1", name: "C++17 Header-Only Wrapper", tasks: ["T-223","T-224","T-225","T-226","T-227","T-228","T-229","T-230"], ac: "ferromode.hpp: RAII ImfCollection; std::span inputs; all 8 algorithms; FerromodeError thrown on null FFI return" },
  { id: "S8.1.3", feature: "F8.1", name: "cxx Bridge (Optional Modern Path)", tasks: ["T-231","T-232","T-233"], ac: "ferromode-cxx crate builds; rust::Vec<f64> ↔ std::vector<double> works; both paths documented" },
  { id: "S8.1.4", feature: "F8.1", name: "Testing & Distribution", tasks: ["T-234","T-235","T-236","T-237"], ac: "Catch2 suite passes GCC/Clang/MSVC; ASan/UBSan clean; vcpkg + Conan packages submitted" },
];

const tasks = [
  { id: "T-001", story: "S1.1.1", name: "Initialize Cargo workspace", status: "todo" },
  { id: "T-002", story: "S1.1.1", name: "Configure shared Cargo.toml dependencies", status: "todo" },
  { id: "T-003", story: "S1.1.1", name: "Setup GitHub Actions CI matrix", status: "todo" },
  { id: "T-004", story: "S1.1.1", name: "Configure rustfmt + clippy rules", status: "todo" },
  { id: "T-005", story: "S1.1.1", name: "Create CONTRIBUTING, CODE_OF_CONDUCT, LICENSE", status: "todo" },
  { id: "T-006", story: "S1.1.1", name: "Setup criterion benchmarking harness", status: "todo" },
  { id: "T-007", story: "S1.1.1", name: "Create initial README with quickstart", status: "todo" },
  { id: "T-008", story: "S1.1.2", name: "Define Signal struct", status: "todo" },
  { id: "T-009", story: "S1.1.2", name: "Define MultivariateSignal struct", status: "todo" },
  { id: "T-010", story: "S1.1.2", name: "Define ImfCollection struct", status: "todo" },
  { id: "T-011", story: "S1.1.2", name: "Define DecompositionResult struct", status: "todo" },
  { id: "T-012", story: "S1.1.2", name: "Define HilbertResult struct", status: "todo" },
  { id: "T-013", story: "S1.1.2", name: "Define AlgorithmType enum", status: "todo" },
  { id: "T-014", story: "S1.1.2", name: "Derive serde on all result types", status: "todo" },
  { id: "T-015", story: "S1.1.2", name: "Unit tests: reconstruction + orthogonality", status: "todo" },
  { id: "T-016", story: "S1.1.3", name: "Define EmdError enum with thiserror", status: "todo" },
  { id: "T-017", story: "S1.1.3", name: "Implement From<EmdError> for binding error types", status: "todo" },
  { id: "T-018", story: "S1.1.3", name: "Input validation at all public API entry points", status: "todo" },
  { id: "T-019", story: "S1.1.3", name: "Tests for all error paths", status: "todo" },
  { id: "T-020", story: "S1.1.4", name: "Implement natural cubic spline (Thomas algorithm)", status: "todo" },
  { id: "T-021", story: "S1.1.4", name: "Implement periodic cubic spline", status: "todo" },
  { id: "T-022", story: "S1.1.4", name: "Implement not-a-knot cubic spline", status: "todo" },
  { id: "T-023", story: "S1.1.4", name: "Validate against scipy CubicSpline reference", status: "todo" },
  { id: "T-024", story: "S1.1.4", name: "Benchmark spline: verify O(n) Thomas algorithm", status: "todo" },
  { id: "T-025", story: "S1.1.4", name: "Handle degenerate cases (<2 knots, duplicates)", status: "todo" },
  { id: "T-026", story: "S1.2.1", name: "Define BoundaryCondition trait", status: "todo" },
  { id: "T-027", story: "S1.2.1", name: "Define BoundaryCondition enum + factory", status: "todo" },
  { id: "T-028", story: "S1.2.1", name: "Write boundary strategy test harness", status: "todo" },
  { id: "T-029", story: "S1.2.2", name: "Implement CharacteristicWave strategy", status: "todo" },
  { id: "T-030", story: "S1.2.2", name: "Handle edge case: <4 total extrema", status: "todo" },
  { id: "T-031", story: "S1.2.2", name: "Test on sine, chirp, noisy modulated signal", status: "todo" },
  { id: "T-032", story: "S1.2.3", name: "Implement even-extension mirror strategy", status: "todo" },
  { id: "T-033", story: "S1.2.3", name: "Implement odd-extension mirror strategy", status: "todo" },
  { id: "T-034", story: "S1.2.3", name: "Verify endpoint is local extremum in extended signal", status: "todo" },
  { id: "T-035", story: "S1.2.4", name: "Implement Zeng-He periodic extension", status: "todo" },
  { id: "T-036", story: "S1.2.4", name: "Use periodic cubic spline BC for envelope", status: "todo" },
  { id: "T-037", story: "S1.2.4", name: "Test on periodic signals; verify no endpoint artifacts", status: "todo" },
  { id: "T-038", story: "S1.2.4", name: "Add Zeng & He 2004 citation in source comment", status: "todo" },
  { id: "T-039", story: "S1.2.5", name: "Compute endpoint derivatives via finite differences", status: "todo" },
  { id: "T-040", story: "S1.2.5", name: "Linear extrapolation for artificial extrema", status: "todo" },
  { id: "T-041", story: "S1.2.5", name: "Test on monotone-end signals", status: "todo" },
  { id: "T-042", story: "S1.2.6", name: "Implement Yule-Walker AR estimation", status: "todo" },
  { id: "T-043", story: "S1.2.6", name: "Forecast/backcast N samples beyond boundaries", status: "todo" },
  { id: "T-044", story: "S1.2.6", name: "Expose ar_order configuration parameter", status: "todo" },
  { id: "T-045", story: "S1.2.6", name: "Test on AR(2) synthetic signals", status: "todo" },
  { id: "T-046", story: "S1.2.7", name: "Implement cross-correlation interior search", status: "todo" },
  { id: "T-047", story: "S1.2.7", name: "Append matched segment beyond boundary", status: "todo" },
  { id: "T-048", story: "S1.2.7", name: "Expose match_length config parameter", status: "todo" },
  { id: "T-049", story: "S1.2.7", name: "Compare vs mirror on quasi-periodic signals", status: "todo" },
  { id: "T-050", story: "S1.3.1", name: "Implement local maxima detection", status: "todo" },
  { id: "T-051", story: "S1.3.1", name: "Implement local minima detection", status: "todo" },
  { id: "T-052", story: "S1.3.1", name: "Handle plateau extrema (midpoint index)", status: "todo" },
  { id: "T-053", story: "S1.3.1", name: "Handle boundary as potential extremum", status: "todo" },
  { id: "T-054", story: "S1.3.1", name: "Test: sine, sawtooth, step, constant", status: "todo" },
  { id: "T-055", story: "S1.3.2", name: "Implement sift_one() function", status: "todo" },
  { id: "T-056", story: "S1.3.2", name: "Implement SD threshold stopping criterion", status: "todo" },
  { id: "T-057", story: "S1.3.2", name: "Implement S-number stopping criterion", status: "todo" },
  { id: "T-058", story: "S1.3.2", name: "Implement fixed-iteration stopping", status: "todo" },
  { id: "T-059", story: "S1.3.2", name: "Implement energy-difference stopping", status: "todo" },
  { id: "T-060", story: "S1.3.2", name: "Add max_sifting_iterations guard", status: "todo" },
  { id: "T-061", story: "S1.3.3", name: "Implement emd() top-level function", status: "todo" },
  { id: "T-062", story: "S1.3.3", name: "Outer sifting loop over residue", status: "todo" },
  { id: "T-063", story: "S1.3.3", name: "Implement max_imfs limit", status: "todo" },
  { id: "T-064", story: "S1.3.3", name: "Implement intermittency test option", status: "todo" },
  { id: "T-065", story: "S1.3.3", name: "Validate reconstruction within 1e-12", status: "todo" },
  { id: "T-066", story: "S1.3.3", name: "Reference tests: sunspot data, synthetic EEG", status: "todo" },
  { id: "T-067", story: "S1.4.1", name: "Implement EEMD noise injection", status: "todo" },
  { id: "T-068", story: "S1.4.1", name: "Parallel trial execution via rayon::par_iter", status: "todo" },
  { id: "T-069", story: "S1.4.1", name: "Seeded RNG for reproducibility", status: "todo" },
  { id: "T-070", story: "S1.4.1", name: "Average IMFs across trials", status: "todo" },
  { id: "T-071", story: "S1.4.1", name: "Test mode-mixing resolution on synthetic signal", status: "todo" },
  { id: "T-072", story: "S1.4.2", name: "Implement paired ±noise trials", status: "todo" },
  { id: "T-073", story: "S1.4.2", name: "Average complementary pairs; verify reconstruction", status: "todo" },
  { id: "T-074", story: "S1.4.2", name: "Test RMS noise level matches EEMD", status: "todo" },
  { id: "T-075", story: "S1.4.3", name: "Implement CEEMDAN stage-wise loop", status: "todo" },
  { id: "T-076", story: "S1.4.3", name: "Compute adaptive noise scaling per stage", status: "todo" },
  { id: "T-077", story: "S1.4.3", name: "Extract first IMF per trial; average to get mean_IMF_k", status: "todo" },
  { id: "T-078", story: "S1.4.3", name: "Compute next residue", status: "todo" },
  { id: "T-079", story: "S1.4.3", name: "Parallelize trial computation at each stage", status: "todo" },
  { id: "T-080", story: "S1.4.3", name: "Validate reconstruction error < 1e-10", status: "todo" },
  { id: "T-081", story: "S1.4.4", name: "Implement improved noise model (EMD of noise-only)", status: "todo" },
  { id: "T-082", story: "S1.4.4", name: "Compute mean_IMF_k = E[first_IMF(r_k + ε·IMF_k_noise)]", status: "todo" },
  { id: "T-083", story: "S1.4.4", name: "Test residual noise reduction vs CEEMDAN", status: "todo" },
  { id: "T-084", story: "S1.5.1", name: "Uniform angular sampling on n-spheres", status: "todo" },
  { id: "T-085", story: "S1.5.1", name: "Implement Hammersley low-discrepancy sequences", status: "todo" },
  { id: "T-086", story: "S1.5.1", name: "Implement Halton sequences", status: "todo" },
  { id: "T-087", story: "S1.5.1", name: "Expose DirectionSampling enum", status: "todo" },
  { id: "T-088", story: "S1.5.1", name: "KS test for uniform projection distribution", status: "todo" },
  { id: "T-089", story: "S1.5.2", name: "Projection of n-variate signal onto direction vectors", status: "todo" },
  { id: "T-090", story: "S1.5.2", name: "Component-wise interpolation for n-D envelopes", status: "todo" },
  { id: "T-091", story: "S1.5.2", name: "Average envelopes over all directions", status: "todo" },
  { id: "T-092", story: "S1.5.2", name: "Multivariate sifting loop", status: "todo" },
  { id: "T-093", story: "S1.5.2", name: "Test mode-alignment on synthetic hexavariate signal", status: "todo" },
  { id: "T-094", story: "S1.5.3", name: "Add noise channels to signal before MEMD", status: "todo" },
  { id: "T-095", story: "S1.5.3", name: "Discard noise channel IMFs post-decomposition", status: "todo" },
  { id: "T-096", story: "S1.5.3", name: "Test noise suppression vs plain MEMD", status: "todo" },
  { id: "T-097", story: "S1.6.1", name: "FFT-based Hilbert transform (zero neg freqs)", status: "todo" },
  { id: "T-098", story: "S1.6.1", name: "Compute analytic signal z(t)", status: "todo" },
  { id: "T-099", story: "S1.6.1", name: "Compute instantaneous amplitude A(t)", status: "todo" },
  { id: "T-100", story: "S1.6.1", name: "Compute instantaneous phase φ(t)", status: "todo" },
  { id: "T-101", story: "S1.6.1", name: "Compute instantaneous frequency f(t)", status: "todo" },
  { id: "T-102", story: "S1.6.1", name: "Validate on pure tone and AM signal", status: "todo" },
  { id: "T-103", story: "S1.6.2", name: "Implement Hilbert marginal spectrum", status: "todo" },
  { id: "T-104", story: "S1.6.2", name: "Implement IMF orthogonality index", status: "todo" },
  { id: "T-105", story: "S1.6.2", name: "Implement degree of stationarity metric", status: "todo" },
  { id: "T-106", story: "S1.6.2", name: "Implement IMF energy ratio per component", status: "todo" },
  { id: "T-107", story: "S1.7.1", name: "VMD: ADMM in frequency domain", status: "todo" },
  { id: "T-108", story: "S1.7.1", name: "Center frequency update step", status: "todo" },
  { id: "T-109", story: "S1.7.1", name: "Mode update step (Wiener filter)", status: "todo" },
  { id: "T-110", story: "S1.7.1", name: "Dual variable (Lagrange multiplier) update", status: "todo" },
  { id: "T-111", story: "S1.7.1", name: "Expose K, alpha, tau, tol parameters", status: "todo" },
  { id: "T-112", story: "S1.7.1", name: "Validate vs MATLAB reference output", status: "todo" },
  { id: "T-113", story: "S2.1.1", name: "Configure PyO3 + maturin; create #[pymodule] entry point", status: "todo" },
  { id: "T-114", story: "S2.1.1", name: "Expose EmdConfig as #[pyclass] with #[new]; all defaults from Rust impl Default", status: "todo" },
  { id: "T-115", story: "S2.1.1", name: "Expose EnsembleConfig, MemdConfig as #[pyclass] wrappers", status: "todo" },
  { id: "T-116", story: "S2.1.1", name: "Expose BoundaryCondition, StoppingCriterion as #[pyclass] enums", status: "todo" },
  { id: "T-117", story: "S2.1.1", name: "Expose AlgorithmType as #[pyclass] enum", status: "todo" },
  { id: "T-118", story: "S2.1.2", name: "numpy_to_slice(): PyReadonlyArray1<f64> → &[f64] zero-copy if C-contiguous", status: "todo" },
  { id: "T-119", story: "S2.1.2", name: "numpy2d_to_vecs(): PyReadonlyArray2<f64> → Vec<Vec<f64>> for MEMD input", status: "todo" },
  { id: "T-120", story: "S2.1.2", name: "Validate inputs at marshalling boundary; reject non-finite; forward EmdError as ValueError", status: "todo" },
  { id: "T-121", story: "S2.1.3", name: "ImfCollectionPy #[pyclass]: .imfs → PyArray2, .residue → PyArray1, .reconstruct() calls Rust", status: "todo" },
  { id: "T-122", story: "S2.1.3", name: "HilbertResultPy #[pyclass]: .instantaneous_amplitude/frequency/marginal_spectrum as numpy arrays", status: "todo" },
  { id: "T-123", story: "S2.1.3", name: "DecompositionResultPy #[pyclass]: .algorithm, .elapsed_ms, .imfs, .hilbert()", status: "todo" },
  { id: "T-124", story: "S2.1.4", name: "#[pyfunction] emd(): marshal in → call emd_core::api::emd() → marshal out", status: "todo" },
  { id: "T-125", story: "S2.1.4", name: "Same for eemd, ceemd, ceemdan, iceemdan (EnsembleConfigPy)", status: "todo" },
  { id: "T-126", story: "S2.1.4", name: "Same for memd, namemd (MemdConfigPy + 2D array)", status: "todo" },
  { id: "T-127", story: "S2.1.4", name: "Same for vmd (VmdConfigPy)", status: "todo" },
  { id: "T-128", story: "S2.1.4", name: "Map EmdError variants → typed Python exceptions; no information lost", status: "todo" },
  { id: "T-129", story: "S2.1.4", name: "py.allow_threads() in all ensemble wrappers to release GIL during Rayon parallel work", status: "todo" },
  { id: "T-130", story: "S2.1.5", name: "Write pyproject.toml with maturin build backend, Python ≥3.10 constraint, classifiers", status: "todo" },
  { id: "T-131", story: "S2.1.5", name: "Add py.typed marker; write *.pyi stubs for IDE autocompletion", status: "todo" },
  { id: "T-132", story: "S2.1.5", name: "CI release: maturin publish on tag; build wheels Linux/macOS/Windows", status: "todo" },
  { id: "T-133", story: "S2.1.5", name: "Smoke-test suite: assert output shape, dtype, reconstruct() length — no numerical assertions", status: "todo" },

  { id: "T-134", story: "S3.1.1", name: "Configure extendr-api; create #[extendr] module; run rextendr::document() to generate wrappers", status: "todo" },
  { id: "T-135", story: "S3.1.1", name: "Expose EmdConfig as #[extendr] struct; defaults from Rust impl Default", status: "todo" },
  { id: "T-136", story: "S3.1.1", name: "Expose EnsembleConfig, MemdConfig, VmdConfig as #[extendr] structs", status: "todo" },
  { id: "T-137", story: "S3.1.2", name: "Robj::as_real_slice() → &[f64] zero-copy for 1D; reject non-numeric with EmdError message", status: "todo" },
  { id: "T-138", story: "S3.1.2", name: "Matrix input for MEMD: read dimensions attribute + as_real_vector() to reconstruct channels", status: "todo" },
  { id: "T-139", story: "S3.1.2", name: "Validate at boundary: NA/NaN/Inf → R stop() with Rust message", status: "todo" },
  { id: "T-140", story: "S3.1.3", name: "ImfCollection → R named list: $imfs matrix, $residue vector, S3 class emd_result", status: "todo" },
  { id: "T-141", story: "S3.1.3", name: "HilbertResult → R named list: $instantaneous_amplitude, $frequency matrices, class hilbert_result", status: "todo" },
  { id: "T-142", story: "S3.1.3", name: ".reconstruct() on emd_result calls Rust ImfCollection::reconstruct() via extendr — no R summation", status: "todo" },
  { id: "T-143", story: "S3.1.4", name: "#[extendr] emd(): marshal → call emd_core::api::emd() → marshal result", status: "todo" },
  { id: "T-144", story: "S3.1.4", name: "Same for eemd, ceemd, ceemdan, iceemdan, memd, namemd, vmd", status: "todo" },
  { id: "T-145", story: "S3.1.4", name: "Map EmdError → R simpleError with class c('emd_error','error'); message verbatim from Rust", status: "todo" },
  { id: "T-146", story: "S3.1.5", name: "Write DESCRIPTION: SystemRequirements: Cargo; pass R CMD check --as-cran", status: "todo" },
  { id: "T-147", story: "S3.1.5", name: "Smoke tests: call each function; assert list structure + field names + vector types", status: "todo" },
  { id: "T-148", story: "S3.1.5", name: "CI CRAN check on Linux/macOS/Windows; submit to CRAN", status: "todo" },

  { id: "T-149", story: "S4.1.1", name: "emd-julia crate: crate-type=[cdylib]; re-exports emd_core::ffi — no new logic", status: "todo" },
  { id: "T-150", story: "S4.1.1", name: "emd_core/src/ffi.rs: #[repr(C)] structs + extern 'C' fn for every public API", status: "todo" },
  { id: "T-151", story: "S4.1.1", name: "FFI error protocol: null return + error code out-param; no Rust panic across FFI", status: "todo" },
  { id: "T-152", story: "S4.1.1", name: "cbindgen in CI: auto-generate emd_core.h; commit header; no duplicate struct definitions", status: "todo" },
  { id: "T-153", story: "S4.1.2", name: "Create Ferromode.jl Project.toml; deps: Libdl only; no algorithm dependencies", status: "todo" },
  { id: "T-154", story: "S4.1.2", name: "__init__: Libdl.find_library or JLLWrappers artifact; store library handle", status: "todo" },
  { id: "T-155", story: "S4.1.2", name: "Define EmdConfig, EnsembleConfig, MemdConfig, VmdConfig Julia structs mirroring C structs exactly", status: "todo" },
  { id: "T-156", story: "S4.1.3", name: "emd(signal, config): GC.@preserve, ccall emd_core_emd, wrap result, register finalizer → emd_core_free", status: "todo" },
  { id: "T-157", story: "S4.1.3", name: "Same for eemd, ceemd, ceemdan, iceemdan", status: "todo" },
  { id: "T-158", story: "S4.1.3", name: "Same for memd, namemd (Matrix{Float64} input)", status: "todo" },
  { id: "T-159", story: "S4.1.3", name: "Same for vmd", status: "todo" },
  { id: "T-160", story: "S4.1.3", name: "Error: null return → read error code → throw EmdError(message); no error logic", status: "todo" },
  { id: "T-161", story: "S4.1.3", name: "reconstruct(): ccall emd_core_reconstruct; no Julia summation", status: "todo" },
  { id: "T-162", story: "S4.1.4", name: "runtests.jl: @testset for each function; assert output types and sizes only", status: "todo" },
  { id: "T-163", story: "S4.1.4", name: "Register in Julia General Registry; CI on Julia 1.9+, 3 platforms", status: "todo" },
  { id: "T-164", story: "S4.1.4", name: "Document JLLWrappers artifact approach for shipping compiled cdylib", status: "todo" },

  { id: "T-165", story: "S5.1.1", name: "Configure wasm-bindgen + wasm-pack in emd-wasm/Cargo.toml; #[wasm_bindgen] entry", status: "todo" },
  { id: "T-166", story: "S5.1.1", name: "WasmEmdConfig #[wasm_bindgen]: constructor from JS object; all defaults from Rust", status: "todo" },
  { id: "T-167", story: "S5.1.1", name: "WasmEnsembleConfig, WasmMemdConfig, WasmVmdConfig as #[wasm_bindgen] structs", status: "todo" },
  { id: "T-168", story: "S5.1.1", name: "WasmBoundaryCondition, WasmStoppingCriterion as #[wasm_bindgen] enums", status: "todo" },
  { id: "T-169", story: "S5.1.2", name: "Float64Array → WASM linear memory slice via from_raw_parts; no copy in WASM model", status: "todo" },
  { id: "T-170", story: "S5.1.2", name: "MEMD flat Float64Array + n_channels: reconstruct Vec<Vec<f64>> by striding (marshalling only)", status: "todo" },
  { id: "T-171", story: "S5.1.2", name: "Validate non-finite inputs at boundary; throw JS Error with Rust message", status: "todo" },
  { id: "T-172", story: "S5.1.3", name: "WasmImfCollection: getImf(n) → Float64Array view; getResidue(); nImfs(); reconstruct() calls Rust", status: "todo" },
  { id: "T-173", story: "S5.1.3", name: "WasmHilbertResult: getInstantaneousAmplitude/Frequency(idx); getMarginalSpectrum()", status: "todo" },
  { id: "T-174", story: "S5.1.3", name: "WasmDecompositionResult: .imfs(), .hilbert(), .algorithm(), .elapsedMs()", status: "todo" },
  { id: "T-175", story: "S5.1.3", name: "free() on all result structs calls Rust drop; document memory ownership clearly", status: "todo" },
  { id: "T-176", story: "S5.1.4", name: "#[wasm_bindgen] emd(): marshal → call emd_core::api::emd() → marshal out", status: "todo" },
  { id: "T-177", story: "S5.1.4", name: "Same for eemd, ceemd, ceemdan, iceemdan, memd, namemd, vmd", status: "todo" },
  { id: "T-178", story: "S5.1.4", name: "Map EmdError → JsValue::from(js_sys::Error::new(&msg)); no information lost", status: "todo" },
  { id: "T-179", story: "S5.1.5", name: "wasm-pack build --target web and --target nodejs; output to pkg/", status: "todo" },
  { id: "T-180", story: "S5.1.5", name: "Review auto-generated .d.ts; supplement with hand-written types for async init() only", status: "todo" },
  { id: "T-181", story: "S5.1.5", name: "package.json with dual ESM/CJS exports; publish to npm as ferromode-js", status: "todo" },
  { id: "T-182", story: "S5.1.5", name: "Vitest smoke tests: output instanceof Float64Array, correct .length — no numerical assertions", status: "todo" },
  { id: "T-183", story: "S5.1.5", name: "CI release: wasm-pack publish on tag push", status: "todo" },
  { id: "T-184", story: "S6.1.1", name: "Create reference test signal library (JSON): tones, AM, FM, chirp, sunspot, multivariate", status: "todo" },
  { id: "T-185", story: "S6.1.1", name: "Pre-compute expected IMFs from Rilling-Flandrin C reference implementation", status: "todo" },
  { id: "T-186", story: "S6.1.1", name: "Store reference outputs in validation/reference/", status: "todo" },
  { id: "T-187", story: "S6.1.2", name: "Write cross-language comparison script: run identical decomposition in all 6 bindings (Rust, Python, R, Julia, JS, C++; MATLAB on nightly)", status: "todo" },
  { id: "T-188", story: "S6.1.2", name: "Assert all pairwise differences < 1e-10", status: "todo" },
  { id: "T-189", story: "S6.1.2", name: "Run cross-language validation in CI on every PR", status: "todo" },
  { id: "T-190", story: "S6.2.1", name: "Write per-algorithm mathematical docs with LaTeX equations", status: "todo" },
  { id: "T-191", story: "S6.2.1", name: "Full bibliographic citations for all algorithms and boundary methods", status: "todo" },
  { id: "T-192", story: "S6.2.1", name: "Create algorithm selection decision tree diagram", status: "todo" },
  { id: "T-193", story: "S6.2.1", name: "Create algorithm comparison table (properties, use cases, cost)", status: "todo" },
  { id: "T-194", story: "S6.2.2", name: "Ensure every public Rust fn has rustdoc + example", status: "todo" },
  { id: "T-195", story: "S6.2.2", name: "Build and publish to docs.rs", status: "todo" },
  { id: "T-196", story: "S6.2.2", name: "Write mkdocs site with binding-guide.md: the pure-wrap contract explained", status: "todo" },
  { id: "T-197", story: "S6.2.2", name: "Getting-started tutorial for each of the 6 language bindings (Python, R, Julia, JS, MATLAB, C++)", status: "todo" },

  { id: "T-198", story: "S7.1.1", name: "Create ferromode-mex crate (cdylib); implement mexFunction extern C entry point over ferromode::ffi", status: "todo" },
  { id: "T-199", story: "S7.1.1", name: "build.rs: link libmex + libmx from MATLAB_ROOT env var; support Octave liboctave as alternative", status: "todo" },
  { id: "T-200", story: "S7.1.1", name: "Auto-detect MATLAB vs Octave and set correct output extension (.mexa64 / .mexmaci64 / .mexw64 / .mex)", status: "todo" },
  { id: "T-201", story: "S7.1.1", name: "mxGetPr() / mxGetM() / mxGetN() input extraction helpers — marshalling only, no computation", status: "todo" },
  { id: "T-202", story: "S7.1.2", name: "Extract signal from prhs[0]: validate mxIsDouble, !mxIsComplex; get pointer via mxGetPr() → &[f64]", status: "todo" },
  { id: "T-203", story: "S7.1.2", name: "Parse config from prhs[1] MATLAB struct: mxGetField() → FerromodeConfig fields; defaults from Rust", status: "todo" },
  { id: "T-204", story: "S7.1.2", name: "MEMD matrix input: mxGetPr() + mxGetM() + mxGetN() to reconstruct channel layout — marshalling only", status: "todo" },
  { id: "T-205", story: "S7.1.2", name: "Validation: non-double, complex, empty, non-finite → mexErrMsgIdAndTxt with Rust error message verbatim", status: "todo" },
  { id: "T-206", story: "S7.1.3", name: "Output struct: mxCreateStructMatrix with imfs (matrix), residue (vector), n_imfs, algorithm, elapsed_ms", status: "todo" },
  { id: "T-207", story: "S7.1.3", name: "Copy IMF data from ImfCollection into mxCreateDoubleMatrix allocations via memcpy — MATLAB owns output", status: "todo" },
  { id: "T-208", story: "S7.1.3", name: "ferromode_reconstruct MEX: accepts output struct, calls ferromode::api::reconstruct() in Rust, returns mxArray", status: "todo" },
  { id: "T-209", story: "S7.1.3", name: "ferromode_hilbert MEX: calls ferromode::api::hilbert(), returns struct with amplitude/frequency/spectrum fields", status: "todo" },
  { id: "T-210", story: "S7.1.4", name: "ferromode_emd MEX: marshal → ferromode_emd() FFI → marshal result", status: "todo" },
  { id: "T-211", story: "S7.1.4", name: "Same for ferromode_eemd, ferromode_ceemd, ferromode_ceemdan, ferromode_iceemdan", status: "todo" },
  { id: "T-212", story: "S7.1.4", name: "Same for ferromode_memd, ferromode_namemd (matrix input)", status: "todo" },
  { id: "T-213", story: "S7.1.4", name: "Same for ferromode_vmd", status: "todo" },
  { id: "T-214", story: "S7.1.4", name: "Wrapper .m files per function: named argument sugar calling MEX binary; zero computation in .m files", status: "todo" },
  { id: "T-215", story: "S7.1.5", name: "Test all MEX functions under GNU Octave 8+; fix liboctave API differences in struct creation", status: "todo" },
  { id: "T-216", story: "S7.1.5", name: "MATLAB test script tests/test_ferromode.m: assert output is struct with correct field names and sizes only", status: "todo" },
  { id: "T-217", story: "S7.1.5", name: "Package as .mltbx for MATLAB Add-On Explorer; package as .tar.gz for Octave Forge submission", status: "todo" },
  { id: "T-218", story: "S7.1.5", name: "Document MATLAB_ROOT build config in binding-guide.md; CI builds MEX using Octave (no licence required)", status: "todo" },

  { id: "T-219", story: "S8.1.1", name: "Confirm ferromode.h (cbindgen output) is sufficient for C++ consumption — no additions needed", status: "todo" },
  { id: "T-220", story: "S8.1.1", name: "CMakeLists.txt: FetchContent pre-built libferromode; expose ferromode::ferromode CMake target", status: "todo" },
  { id: "T-221", story: "S8.1.1", name: "ferromode.pc pkg-config file for non-CMake build systems", status: "todo" },
  { id: "T-222", story: "S8.1.1", name: "Ship pre-built binaries (Linux x86_64/aarch64, macOS universal2, Windows x64) via GitHub Releases", status: "todo" },
  { id: "T-223", story: "S8.1.2", name: "ferromode.hpp: namespace ferromode; EmdConfig, EnsembleConfig, MemdConfig, VmdConfig as C++ aggregates mirroring C structs; constructors delegate to ferromode_default_*_config() FFI", status: "todo" },
  { id: "T-224", story: "S8.1.2", name: "ImfCollection RAII wrapper: destructor calls ferromode_free_result(); .imfs() → vector<span<const double>>; .reconstruct() calls Rust FFI", status: "todo" },
  { id: "T-225", story: "S8.1.2", name: "ferromode::emd(span<const double>, const EmdConfig&) → ImfCollection; extracts .data() + .size(), calls FFI, wraps result", status: "todo" },
  { id: "T-226", story: "S8.1.2", name: "Same for eemd, ceemd, ceemdan, iceemdan (EnsembleConfig)", status: "todo" },
  { id: "T-227", story: "S8.1.2", name: "Same for memd, namemd (span<const double*> channels + MemdConfig)", status: "todo" },
  { id: "T-228", story: "S8.1.2", name: "Same for vmd (VmdConfig)", status: "todo" },
  { id: "T-229", story: "S8.1.2", name: "ferromode::hilbert(const ImfCollection&) → HilbertResult RAII wrapper; calls Rust FFI; no C++ DSP", status: "todo" },
  { id: "T-230", story: "S8.1.2", name: "Error handling: FFI null return → throw ferromode::FerromodeError(message) : std::runtime_error", status: "todo" },
  { id: "T-231", story: "S8.1.3", name: "ferromode-cxx Rust crate: #[cxx::bridge] exposing all public API functions with C++ idiomatic signatures", status: "todo" },
  { id: "T-232", story: "S8.1.3", name: "rust::Vec<f64> ↔ std::vector<double> conversions via cxx generated glue — no manual pointer arithmetic", status: "todo" },
  { id: "T-233", story: "S8.1.3", name: "Document both paths in binding-guide.md: header-only (no Cargo) vs cxx bridge (Cargo-integrated)", status: "todo" },
  { id: "T-234", story: "S8.1.4", name: "Catch2 test suite: assert output vector sizes and reconstruct() length — no numerical assertions", status: "todo" },
  { id: "T-235", story: "S8.1.4", name: "CI: GCC 12+, Clang 15+, MSVC 2022 on Linux/macOS/Windows; AddressSanitizer + UBSan", status: "todo" },
  { id: "T-236", story: "S8.1.4", name: "Publish to vcpkg registry and Conan Center Index", status: "todo" },
  { id: "T-237", story: "S8.1.4", name: "C++ getting-started tutorial covering CMake+FetchContent path and cxx bridge path", status: "todo" },
];

// ─── HELPERS ─────────────────────────────────────────────────────────────────

const epicById = Object.fromEntries(epics.map(e => [e.id, e]));
const featureById = Object.fromEntries(features.map(f => [f.id, f]));
const storyById = Object.fromEntries(stories.map(s => [s.id, s]));
const releaseById = Object.fromEntries(releases.map(r => [r.id, r]));
const milestoneById = Object.fromEntries(milestones.map(m => [m.id, m]));

function getEpicColor(epicId) {
  return epicById[epicId]?.color || "#6b7280";
}
function getFeatureEpicColor(featureId) {
  const f = featureById[featureId];
  return f ? getEpicColor(f.epic) : "#6b7280";
}
function getMilestoneRelease(milestoneId) {
  const m = milestoneById[milestoneId];
  return m ? releaseById[m.release] : null;
}

// ─── COMPONENTS ──────────────────────────────────────────────────────────────

function Badge({ label, color, small }) {
  return (
    <span
      style={{
        background: color + "22",
        border: `1px solid ${color}55`,
        color: color,
        borderRadius: 4,
        padding: small ? "1px 6px" : "2px 8px",
        fontSize: small ? 10 : 11,
        fontFamily: "monospace",
        fontWeight: 700,
        letterSpacing: "0.05em",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}

function StatusDot({ status }) {
  const colors = { todo: "#6b7280", inprogress: "#f59e0b", done: "#22c55e" };
  return (
    <span
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: colors[status] || "#6b7280",
        flexShrink: 0,
        marginTop: 2,
      }}
    />
  );
}

function Collapsible({ title, badge, color, children, defaultOpen, depth = 0 }) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  return (
    <div style={{ marginBottom: depth === 0 ? 12 : 6 }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          width: "100%",
          background: open ? color + "11" : "transparent",
          border: `1px solid ${color}33`,
          borderRadius: 6,
          padding: depth === 0 ? "10px 14px" : "7px 12px",
          cursor: "pointer",
          textAlign: "left",
          transition: "all 0.15s",
        }}
      >
        <span style={{ color, fontSize: 12, width: 12, flexShrink: 0 }}>{open ? "▼" : "▶"}</span>
        <span style={{ flex: 1, color: "#e2e8f0", fontWeight: depth === 0 ? 700 : 600, fontSize: depth === 0 ? 14 : 13 }}>
          {title}
        </span>
        {badge}
      </button>
      {open && (
        <div style={{ marginLeft: depth === 0 ? 16 : 12, marginTop: 4, borderLeft: `2px solid ${color}33`, paddingLeft: 12 }}>
          {children}
        </div>
      )}
    </div>
  );
}

// ─── VIEWS ───────────────────────────────────────────────────────────────────

function PRDView() {
  return (
    <div style={{ maxWidth: 900 }}>
      <Section title="Executive Summary" icon="📋">
        <P>Ferromode is an open-source, multi-language empirical mode decomposition ecosystem built on a high-performance Rust core, with idiomatic bindings for Python, R, Julia, and JavaScript/TypeScript. It implements the full EMD algorithm family — EMD through CEEMDAN, MEMD, NA-MEMD — plus a comprehensive suite of boundary condition strategies and post-processing tools.</P>
      </Section>

      <Section title="Problem Statement" icon="⚠️">
        <Grid cols={2}>
          {[
            ["Performance Bottlenecks", "Python/R implementations are slow for large datasets or ensemble methods (100s of trials)"],
            ["Incomplete Coverage", "Most packages implement only EMD + EEMD, omitting CEEMDAN, ICEEMDAN, MEMD, NA-MEMD"],
            ["Poor Boundary Handling", "Most packages offer only 1–2 boundary strategies, often poorly documented"],
            ["No Cross-Language Consistency", "Results differ between R, Python, and MATLAB due to differing defaults"],
            ["Limited Documentation", "Algorithm parameter choices are opaque; trade-offs undocumented"],
            ["No Interoperability", "Research pipelines mixing R and Python must duplicate decompositions"],
          ].map(([t, d]) => (
            <ProblemCard key={t} title={t} desc={d} />
          ))}
        </Grid>
      </Section>

      <Section title="Goals" icon="🎯">
        <ul style={{ color: "#94a3b8", lineHeight: 1.8, paddingLeft: 20, fontSize: 13 }}>
          {[
            "Single authoritative Rust core (ferromode) implementing all major EMD variants",
            "Idiomatic bindings: Python (ferromode-py), R (ferromode-r), Julia (Ferromode.jl), JS/TS (ferromode-js)",
            "Numerical equivalence across all language bindings (same params → same results)",
            "≥8 boundary condition strategies with documented trade-offs",
            "Comprehensive test suite with cross-language validation",
            "Parallel computation for ensemble methods via Rayon",
            "Publication-quality visualizations in each binding",
          ].map(g => <li key={g}>{g}</li>)}
        </ul>
      </Section>

      <Section title="Target Users" icon="👥">
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {[
            ["🎓", "Academic Researcher", "PhD/faculty in geophysics, neuroscience, climate", "Full algorithm coverage, reproducibility, citations"],
            ["💻", "Data Scientist", "Python-first practitioner in industry", "Fast Pythonic API, pandas/numpy integration"],
            ["📊", "Statistician / R User", "Biostatistics, econometrics", "R idiomatic API, tidyverse, CRAN publishable"],
            ["⚡", "Scientific Programmer", "Julia numerical methods", "High performance, composable, Julia types"],
            ["🌐", "Web Developer", "JS data viz, browser tools", "WASM, lightweight, TypeScript typed"],
          ].map(([icon, role, profile, needs]) => (
            <div key={role} style={{ display: "flex", gap: 12, background: "#1e293b", borderRadius: 6, padding: "10px 14px", border: "1px solid #334155" }}>
              <span style={{ fontSize: 20 }}>{icon}</span>
              <div>
                <div style={{ color: "#e2e8f0", fontWeight: 700, fontSize: 13 }}>{role}</div>
                <div style={{ color: "#64748b", fontSize: 12 }}>{profile}</div>
                <div style={{ color: "#94a3b8", fontSize: 12, marginTop: 2 }}>→ {needs}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Success Metrics" icon="📈">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}>
          {[
            ["GitHub Stars", "≥ 500", "12 months"],
            ["PyPI Monthly Downloads", "≥ 5,000", "12 months"],
            ["CRAN Downloads", "≥ 1,000/mo", "12 months"],
            ["Academic Citations", "≥ 10", "12 months"],
            ["Issue Resolution", "≥ 80%", "within 14 days"],
            ["Cross-lang Tests", "100%", "pass rate"],
          ].map(([label, val, note]) => (
            <div key={label} style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, padding: "12px 14px", textAlign: "center" }}>
              <div style={{ color: "#2dd4bf", fontWeight: 800, fontSize: 18 }}>{val}</div>
              <div style={{ color: "#e2e8f0", fontSize: 11, fontWeight: 600, marginTop: 2 }}>{label}</div>
              <div style={{ color: "#64748b", fontSize: 10 }}>{note}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Non-Functional Requirements" icon="⚙️">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ background: "#1e293b" }}>
              {["ID","Requirement","Target"].map(h => <th key={h} style={{ padding: "8px 12px", color: "#94a3b8", textAlign: "left", border: "1px solid #334155" }}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {[
              ["NFR-001","EMD on 10K-sample signal","< 50ms (single thread)"],
              ["NFR-002","CEEMDAN 200 trials, 10K samples","< 10s (8 threads)"],
              ["NFR-003","Peak RAM for CEEMDAN 200 trials","< 2 GB"],
              ["NFR-004","Rust core test coverage","≥ 90% line coverage"],
              ["NFR-005","Cross-language numerical equivalence","Within 1e-10 relative error"],
              ["NFR-006","API stability","SemVer; no breaking changes within major"],
              ["NFR-007","Documentation","Every public API: docs + example"],
            ].map(([id,req,tgt]) => (
              <tr key={id} style={{ background: "#0f172a" }}>
                <td style={{ padding: "7px 12px", border: "1px solid #1e293b", fontFamily: "monospace", color: "#2dd4bf", fontSize: 11 }}>{id}</td>
                <td style={{ padding: "7px 12px", border: "1px solid #1e293b", color: "#e2e8f0" }}>{req}</td>
                <td style={{ padding: "7px 12px", border: "1px solid #1e293b", color: "#94a3b8" }}>{tgt}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
    </div>
  );
}

function ARDView() {
  return (
    <div style={{ maxWidth: 900 }}>
      <Section title="Architecture Overview" icon="🏗️">
        <P>Ferromode follows a layered monorepo architecture. All algorithmic logic lives in the Rust core. Language bindings are thin wrappers handling type marshalling, idiomatic API surface, and packaging.</P>
        <ArchDiagram />
      </Section>

      <Section title="Core Module Structure" icon="📦">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 8 }}>
          {[
            { name: "algorithms/", color: "#2dd4bf", items: ["emd.rs — Basic EMD","eemd.rs — Ensemble EMD","ceemdan.rs — CEEMDAN","iceemdan.rs — Improved CEEMDAN","memd.rs — Multivariate EMD","namemd.rs — NA-MEMD","vmd.rs — VMD"] },
            { name: "boundary/", color: "#60a5fa", items: ["characteristic_wave.rs","mirror.rs","periodic.rs (Zeng & He 2004)","slope.rs","ar_model.rs","svr.rs","waveform_match.rs"] },
            { name: "sifting/", color: "#a78bfa", items: ["engine.rs — Core sifting loop","criteria.rs — Stopping criteria (SD, S-num, fixed, energy)"] },
            { name: "hilbert/", color: "#f472b6", items: ["transform.rs — FFT-based Hilbert","instantaneous.rs — IF + IA"] },
            { name: "spline/", color: "#fb923c", items: ["cubic.rs — Natural + Periodic + Not-a-knot"] },
            { name: "parallel/", color: "#4ade80", items: ["pool.rs — Rayon thread pool config","Enables ensemble method parallelism"] },
          ].map(mod => (
            <div key={mod.name} style={{ background: "#1e293b", border: `1px solid ${mod.color}33`, borderRadius: 6, padding: "12px 14px" }}>
              <div style={{ color: mod.color, fontFamily: "monospace", fontWeight: 700, fontSize: 13, marginBottom: 8 }}>{mod.name}</div>
              {mod.items.map(i => <div key={i} style={{ color: "#94a3b8", fontSize: 12, marginBottom: 3, paddingLeft: 8, borderLeft: `2px solid ${mod.color}44` }}>{i}</div>)}
            </div>
          ))}
        </div>
      </Section>

      <Section title="Language Binding Architecture" icon="🔗">
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {[
            { lang: "Python", pkg: "ferromode-py", bridge: "PyO3", color: "#60a5fa", icon: "🐍", notes: "np.ndarray ↔ &[f64] zero-copy; #[pyfunction] wrappers call emd_core::api directly; GIL released during ensemble methods; EmdError → Python ValueError; maturin build + PyPI" },
            { lang: "R", pkg: "ferromode-r", bridge: "extendr", color: "#a78bfa", icon: "📊", notes: "Robj::as_real_slice() → &[f64] zero-copy; #[extendr] wrappers call emd_core::api; results → named R list with S3 class; EmdError → R simpleError; CRAN submission" },
            { lang: "Julia", pkg: "Ferromode.jl", bridge: "ccall into cdylib", color: "#f472b6", icon: "⚡", notes: "C-ABI cdylib compiled via cbindgen; ccall wrappers with GC.@preserve; finalizers call ferromode_free(); EmdError via out-param error code; JLLWrappers artifact bundle; Julia General Registry" },
            { lang: "JavaScript/TypeScript", pkg: "ferromode-js", bridge: "wasm-bindgen", color: "#fb923c", icon: "🌐", notes: "Float64Array ↔ WASM linear memory; #[wasm_bindgen] wrappers call emd_core::api; result accessors (.getImf, .getResidue, .free()); auto-generated .d.ts; wasm-pack build web+nodejs; npm" },
          ].map(b => (
            <div key={b.lang} style={{ display: "flex", gap: 14, background: "#1e293b", border: `1px solid ${b.color}33`, borderRadius: 6, padding: "12px 14px", alignItems: "flex-start" }}>
              <span style={{ fontSize: 24 }}>{b.icon}</span>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 4 }}>
                  <span style={{ color: b.color, fontWeight: 700, fontSize: 14 }}>{b.lang}</span>
                  <Badge label={b.pkg} color={b.color} small />
                  <Badge label={`via ${b.bridge}`} color="#64748b" small />
                </div>
                <div style={{ color: "#94a3b8", fontSize: 12 }}>{b.notes}</div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Performance Targets" icon="⚡">
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ background: "#1e293b" }}>
              {["Signal Length","Algorithm","Threads","Target"].map(h => <th key={h} style={{ padding: "8px 12px", color: "#94a3b8", textAlign: "left", border: "1px solid #334155" }}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {[
              ["1,000","EMD","1","< 5ms"],
              ["10,000","EMD","1","< 50ms"],
              ["100,000","EMD","1","< 500ms"],
              ["10,000","CEEMDAN (200 trials)","8","< 10s"],
              ["10,000","MEMD (4 channels)","4","< 2s"],
            ].map(row => (
              <tr key={row[0]+row[1]} style={{ background: "#0f172a" }}>
                {row.map((cell, i) => <td key={i} style={{ padding: "7px 12px", border: "1px solid #1e293b", color: i === 3 ? "#2dd4bf" : "#e2e8f0", fontFamily: i === 0 || i === 3 ? "monospace" : "inherit" }}>{cell}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="Key Architectural Decisions" icon="🧠">
        {[
          ["Rust Core with FFI Bindings", "All algorithmic logic in one place. Language bindings are thin wrappers. One bug fix propagates to all 6 language bindings simultaneously."],
          ["Trait-Based Boundary Conditions", "BoundaryCondition trait allows new strategies without modifying core. Users can implement custom strategies."],
          ["Rayon for Parallelism", "Ensemble trials are embarrassingly parallel. Rayon's work-stealing thread pool maximises CPU utilisation with minimal code complexity."],
          ["Seeded RNG for Reproducibility", "All ensemble methods accept an optional seed, making research results fully reproducible across languages."],
          ["Panic-Free Public API", "All errors returned as Result<T, EmdError>. Panics only in truly unexpected internal logic failures."],
        ].map(([title, desc]) => (
          <div key={title} style={{ background: "#1e293b", borderRadius: 6, padding: "10px 14px", marginBottom: 8, borderLeft: "3px solid #2dd4bf" }}>
            <div style={{ color: "#e2e8f0", fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{title}</div>
            <div style={{ color: "#94a3b8", fontSize: 12 }}>{desc}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

function FeaturesView() {
  const [selectedEpic, setSelectedEpic] = useState(null);
  const filtered = selectedEpic ? features.filter(f => f.epic === selectedEpic) : features;

  return (
    <div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
        <button onClick={() => setSelectedEpic(null)} style={filterBtn(!selectedEpic, "#64748b")}>All Epics</button>
        {epics.map(e => (
          <button key={e.id} onClick={() => setSelectedEpic(e.id)} style={filterBtn(selectedEpic === e.id, e.color)}>
            {e.label}: {e.name}
          </button>
        ))}
      </div>

      {filtered.map(f => {
        const epic = epicById[f.epic];
        const release = getMilestoneRelease(f.milestone);
        const featureStories = stories.filter(s => s.feature === f.id);
        return (
          <Collapsible
            key={f.id}
            title={`${f.id} — ${f.name}`}
            color={epic.color}
            badge={
              <div style={{ display: "flex", gap: 6 }}>
                <Badge label={epic.label} color={epic.color} small />
                {release && <Badge label={release.id} color={release.color} small />}
                <Badge label={`${featureStories.length} stories`} color="#64748b" small />
              </div>
            }
          >
            {featureStories.map(s => (
              <StorySummary key={s.id} story={s} epicColor={epic.color} />
            ))}
          </Collapsible>
        );
      })}
    </div>
  );
}

function StorySummary({ story, epicColor }) {
  const storyTasks = tasks.filter(t => t.story === story.id);
  return (
    <Collapsible
      key={story.id}
      title={`${story.id} — ${story.name}`}
      color={epicColor}
      depth={1}
      badge={<Badge label={`${storyTasks.length} tasks`} color="#64748b" small />}
    >
      <div style={{ marginBottom: 8, background: "#1e293b", borderRadius: 4, padding: "8px 12px" }}>
        <div style={{ color: "#64748b", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>Acceptance Criteria</div>
        <div style={{ color: "#94a3b8", fontSize: 12 }}>{story.ac}</div>
      </div>
      {storyTasks.map(t => (
        <div key={t.id} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "5px 8px", borderRadius: 4, marginBottom: 2, background: "#0f172a" }}>
          <StatusDot status={t.status} />
          <span style={{ fontFamily: "monospace", color: epicColor, fontSize: 11, flexShrink: 0, marginTop: 1 }}>{t.id}</span>
          <span style={{ color: "#94a3b8", fontSize: 12 }}>{t.name}</span>
        </div>
      ))}
    </Collapsible>
  );
}

function MatrixView() {
  return (
    <div style={{ maxWidth: 1100 }}>
      <Section title="Release × Epic Matrix" icon="🗺️">
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ background: "#1e293b" }}>
                <th style={{ padding: "10px 14px", color: "#64748b", textAlign: "left", border: "1px solid #334155", minWidth: 140 }}>Epic</th>
                {releases.map(r => (
                  <th key={r.id} style={{ padding: "10px 14px", border: "1px solid #334155", textAlign: "center", minWidth: 100 }}>
                    <div style={{ color: r.color, fontWeight: 700 }}>{r.id}</div>
                    <div style={{ color: "#64748b", fontSize: 10 }}>{r.name}</div>
                    <div style={{ color: "#475569", fontSize: 10 }}>{r.date}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {epics.map((epic, ei) => (
                <tr key={epic.id} style={{ background: ei % 2 === 0 ? "#0f172a" : "#0a1628" }}>
                  <td style={{ padding: "10px 14px", border: "1px solid #1e293b" }}>
                    <div style={{ color: epic.color, fontWeight: 700, fontSize: 12 }}>{epic.label}</div>
                    <div style={{ color: "#94a3b8", fontSize: 11 }}>{epic.name}</div>
                  </td>
                  {releases.map(r => {
                    const ms = milestones.find(m => m.release === r.id && m.epics.includes(epic.id));
                    return (
                      <td key={r.id} style={{ padding: "8px 10px", border: "1px solid #1e293b", textAlign: "center" }}>
                        {ms ? (
                          <div>
                            <div style={{ background: epic.color + "22", border: `1px solid ${epic.color}55`, borderRadius: 4, padding: "4px 8px" }}>
                              <div style={{ color: epic.color, fontWeight: 700, fontSize: 11 }}>{ms.id}</div>
                              <div style={{ color: "#94a3b8", fontSize: 10 }}>{ms.name.split(" ").slice(0,3).join(" ")}</div>
                            </div>
                          </div>
                        ) : (
                          <span style={{ color: "#1e293b" }}>—</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Milestone Details" icon="🏁">
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {milestones.map(m => {
            const release = releaseById[m.release];
            const mEpics = m.epics.map(eid => epicById[eid]);
            return (
              <div key={m.id} style={{ background: "#1e293b", border: `1px solid #334155`, borderRadius: 8, padding: "14px 16px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, flexWrap: "wrap" }}>
                  <span style={{ color: release?.color || "#94a3b8", fontWeight: 800, fontSize: 16, fontFamily: "monospace" }}>{m.id}</span>
                  <span style={{ color: "#e2e8f0", fontWeight: 700, fontSize: 14 }}>{m.name}</span>
                  {release && <Badge label={release.id} color={release.color} small />}
                  <Badge label={m.tasks} color="#64748b" small />
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {mEpics.map(e => e && <Badge key={e.id} label={e.label + ": " + e.name} color={e.color} small />)}
                </div>
              </div>
            );
          })}
        </div>
      </Section>

      <Section title="Timeline (Gantt)" icon="📅">
        <GanttChart />
      </Section>

      <Section title="Task Count by Epic" icon="📊">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}>
          {epics.map(e => (
            <div key={e.id} style={{ background: "#1e293b", border: `1px solid ${e.color}33`, borderRadius: 6, padding: "14px 16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                <div style={{ color: e.color, fontWeight: 700, fontSize: 13 }}>{e.label}</div>
                <div style={{ color: e.color, fontWeight: 800, fontSize: 22 }}>{e.taskCount}</div>
              </div>
              <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 8 }}>{e.name}</div>
              <div style={{ background: "#0f172a", borderRadius: 4, overflow: "hidden" }}>
                <div style={{ height: 6, background: e.color, width: `${(e.taskCount / 237) * 100}%`, transition: "width 0.5s" }} />
              </div>
              <div style={{ color: "#64748b", fontSize: 11, marginTop: 4 }}>~{e.weeks} weeks</div>
            </div>
          ))}
        </div>
        <div style={{ background: "#1e293b", borderRadius: 6, padding: "12px 16px", marginTop: 8, display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "#94a3b8", fontSize: 13 }}>Total Tasks</span>
          <span style={{ color: "#2dd4bf", fontWeight: 800, fontSize: 18 }}>237</span>
        </div>
      </Section>
    </div>
  );
}

function GanttChart() {
  const ganttData = [
    { label: "v0.1 Foundation", items: [
      { name: "Workspace scaffolding", start: 0, len: 2, epic: "E1" },
      { name: "Core type system", start: 2, len: 2, epic: "E1" },
      { name: "Cubic spline engine", start: 4, len: 3, epic: "E1" },
      { name: "Boundary conditions", start: 4, len: 3, epic: "E1" },
    ]},
    { label: "v0.2 Core EMD", items: [
      { name: "Extrema detection", start: 7, len: 1, epic: "E1" },
      { name: "Sifting engine", start: 8, len: 2, epic: "E1" },
      { name: "Basic EMD", start: 10, len: 2, epic: "E1" },
      { name: "Hilbert transform", start: 12, len: 2, epic: "E1" },
    ]},
    { label: "v0.3 Ensemble", items: [
      { name: "EEMD", start: 12, len: 2, epic: "E1" },
      { name: "CEEMD + CEEMDAN", start: 14, len: 3, epic: "E1" },
      { name: "ICEEMDAN", start: 17, len: 1, epic: "E1" },
    ]},
    { label: "v0.5 Multivariate", items: [
      { name: "Hypersphere sampling", start: 18, len: 2, epic: "E1" },
      { name: "MEMD", start: 20, len: 3, epic: "E1" },
      { name: "NA-MEMD + VMD", start: 23, len: 3, epic: "E1" },
    ]},
    { label: "v1.0 Python", items: [
      { name: "PyO3 bindings", start: 23, len: 3, epic: "E2" },
      { name: "Python API + Viz", start: 26, len: 3, epic: "E2" },
    ]},
    { label: "v1.1/1.2 R+Julia", items: [
      { name: "R binding", start: 29, len: 5, epic: "E3" },
      { name: "Julia binding", start: 29, len: 5, epic: "E4" },
    ]},
    { label: "v1.3/1.4 JS+Docs", items: [
      { name: "WASM/TS binding", start: 34, len: 5, epic: "E5" },
      { name: "Docs & validation", start: 32, len: 7, epic: "E6" },
    ]},
    { label: "v1.5 MATLAB/Octave", items: [
      { name: "MEX entry point + build", start: 39, len: 2, epic: "E7" },
      { name: "Marshalling + wrappers", start: 41, len: 2, epic: "E7" },
      { name: "Octave + packaging", start: 43, len: 1, epic: "E7" },
    ]},
    { label: "v1.6 C++ Binding", items: [
      { name: "C header + CMake", start: 39, len: 1, epic: "E8" },
      { name: "C++17 header wrapper", start: 40, len: 2, epic: "E8" },
      { name: "cxx bridge + testing", start: 42, len: 2, epic: "E8" },
    ]},
  ];
  const totalWeeks = 45;

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ minWidth: 700 }}>
        <div style={{ display: "flex", marginBottom: 8, paddingLeft: 120 }}>
          {Array.from({ length: 10 }, (_, i) => i * 4).map(w => (
            <div key={w} style={{ flex: 4, color: "#475569", fontSize: 10, textAlign: "left" }}>W{w+1}</div>
          ))}
        </div>
        {ganttData.map(group => (
          <div key={group.label} style={{ marginBottom: 12 }}>
            <div style={{ color: "#64748b", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>{group.label}</div>
            {group.items.map(item => {
              const epic = epicById[item.epic];
              return (
                <div key={item.name} style={{ display: "flex", alignItems: "center", marginBottom: 3 }}>
                  <div style={{ width: 120, color: "#94a3b8", fontSize: 11, flexShrink: 0, paddingRight: 8, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.name}</div>
                  <div style={{ flex: 1, position: "relative", height: 20, background: "#1e293b", borderRadius: 3 }}>
                    <div style={{
                      position: "absolute",
                      left: `${(item.start / totalWeeks) * 100}%`,
                      width: `${(item.len / totalWeeks) * 100}%`,
                      height: "100%",
                      background: epic.color,
                      borderRadius: 3,
                      opacity: 0.85,
                    }} />
                  </div>
                </div>
              );
            })}
          </div>
        ))}
        <div style={{ display: "flex", gap: 12, marginTop: 12, flexWrap: "wrap" }}>
          {epics.map(e => (
            <div key={e.id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 12, height: 12, background: e.color, borderRadius: 2 }} />
              <span style={{ color: "#64748b", fontSize: 11 }}>{e.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ArchDiagram() {
  const boxes = [
    { id: "py", label: "Python", sub: "ferromode-py / PyO3", color: "#60a5fa", x: 0 },
    { id: "r", label: "R", sub: "ferromode-r / extendr", color: "#a78bfa", x: 1 },
    { id: "jl", label: "Julia", sub: "Ferromode.jl / ccall", color: "#f472b6", x: 2 },
    { id: "js", label: "JS/TS", sub: "ferromode-js / WASM", color: "#fb923c", x: 3 },
  ];
  const coreModules = [
    { label: "Basic EMD", color: "#2dd4bf" },
    { label: "EEMD / CEEMD", color: "#2dd4bf" },
    { label: "CEEMDAN / ICEEMDAN", color: "#2dd4bf" },
    { label: "MEMD / NA-MEMD", color: "#2dd4bf" },
    { label: "VMD", color: "#2dd4bf" },
    { label: "Boundary Conditions", color: "#60a5fa" },
    { label: "Sifting Engine", color: "#60a5fa" },
    { label: "Cubic Spline", color: "#60a5fa" },
    { label: "Hilbert Transform", color: "#f472b6" },
    { label: "Rayon Parallelism", color: "#4ade80" },
  ];

  return (
    <div style={{ background: "#0a1628", borderRadius: 8, padding: 16, border: "1px solid #1e293b" }}>
      <div style={{ color: "#64748b", fontSize: 11, textAlign: "center", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.1em" }}>Language Bindings Layer</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 16 }}>
        {boxes.map(b => (
          <div key={b.id} style={{ background: b.color + "18", border: `1px solid ${b.color}55`, borderRadius: 6, padding: "8px 10px", textAlign: "center" }}>
            <div style={{ color: b.color, fontWeight: 700, fontSize: 13 }}>{b.label}</div>
            <div style={{ color: b.color + "aa", fontSize: 10 }}>{b.sub}</div>
          </div>
        ))}
      </div>
      <div style={{ textAlign: "center", color: "#334155", marginBottom: 8 }}>↓ ↓ ↓ ↓</div>
      <div style={{ background: "#1e293b", borderRadius: 8, padding: 12, border: "1px solid #334155" }}>
        <div style={{ color: "#2dd4bf", fontSize: 11, fontWeight: 700, textAlign: "center", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.1em" }}>ferromode (Rust)</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 6 }}>
          {coreModules.map(m => (
            <div key={m.label} style={{ background: "#0f172a", border: `1px solid ${m.color}33`, borderRadius: 4, padding: "5px 10px", display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: m.color, flexShrink: 0 }} />
              <span style={{ color: "#94a3b8", fontSize: 11 }}>{m.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── SHARED UI ───────────────────────────────────────────────────────────────

function Section({ title, icon, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 18 }}>{icon}</span>
        <h3 style={{ margin: 0, color: "#e2e8f0", fontSize: 16, fontWeight: 700 }}>{title}</h3>
      </div>
      {children}
    </div>
  );
}

function P({ children }) {
  return <p style={{ color: "#94a3b8", fontSize: 13, lineHeight: 1.7, margin: "0 0 12px" }}>{children}</p>;
}

function Grid({ cols, children }) {
  return <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 10 }}>{children}</div>;
}

function ProblemCard({ title, desc }) {
  return (
    <div style={{ background: "#1e293b", border: "1px solid #ef444433", borderRadius: 6, padding: "10px 14px" }}>
      <div style={{ color: "#ef4444", fontWeight: 700, fontSize: 12, marginBottom: 4 }}>⚠ {title}</div>
      <div style={{ color: "#94a3b8", fontSize: 12 }}>{desc}</div>
    </div>
  );
}

function filterBtn(active, color) {
  return {
    background: active ? color + "22" : "transparent",
    border: `1px solid ${active ? color : "#334155"}`,
    borderRadius: 4,
    padding: "5px 10px",
    color: active ? color : "#64748b",
    fontSize: 11,
    cursor: "pointer",
    fontWeight: active ? 700 : 400,
  };
}

// ─── SEARCH ──────────────────────────────────────────────────────────────────

function SearchView({ query }) {
  if (!query) return <div style={{ color: "#64748b", fontSize: 13, padding: 16 }}>Start typing to search tasks, stories, and features…</div>;
  const q = query.toLowerCase();
  const matchedTasks = tasks.filter(t => t.id.toLowerCase().includes(q) || t.name.toLowerCase().includes(q));
  const matchedStories = stories.filter(s => s.id.toLowerCase().includes(q) || s.name.toLowerCase().includes(q));
  const matchedFeatures = features.filter(f => f.id.toLowerCase().includes(q) || f.name.toLowerCase().includes(q));

  if (!matchedTasks.length && !matchedStories.length && !matchedFeatures.length) {
    return <div style={{ color: "#64748b", fontSize: 13, padding: 16 }}>No results for "{query}"</div>;
  }

  return (
    <div>
      {matchedFeatures.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ color: "#64748b", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>Features ({matchedFeatures.length})</div>
          {matchedFeatures.map(f => {
            const epic = epicById[f.epic];
            return (
              <div key={f.id} style={{ display: "flex", gap: 8, padding: "7px 12px", background: "#1e293b", borderRadius: 6, marginBottom: 4, alignItems: "center" }}>
                <Badge label={f.id} color={epic.color} small />
                <span style={{ color: "#e2e8f0", fontSize: 13 }}>{f.name}</span>
                <Badge label={epic.label} color={epic.color} small />
              </div>
            );
          })}
        </div>
      )}
      {matchedStories.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ color: "#64748b", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>Stories ({matchedStories.length})</div>
          {matchedStories.map(s => {
            const feature = featureById[s.feature];
            const color = getFeatureEpicColor(s.feature);
            return (
              <div key={s.id} style={{ padding: "7px 12px", background: "#1e293b", borderRadius: 6, marginBottom: 4 }}>
                <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
                  <Badge label={s.id} color={color} small />
                  <span style={{ color: "#e2e8f0", fontSize: 13 }}>{s.name}</span>
                </div>
                <div style={{ color: "#64748b", fontSize: 11 }}>→ {feature?.name}</div>
              </div>
            );
          })}
        </div>
      )}
      {matchedTasks.length > 0 && (
        <div>
          <div style={{ color: "#64748b", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>Tasks ({matchedTasks.length})</div>
          {matchedTasks.map(t => {
            const story = storyById[t.story];
            const color = getFeatureEpicColor(story?.feature || "");
            return (
              <div key={t.id} style={{ display: "flex", gap: 8, padding: "6px 12px", background: "#1e293b", borderRadius: 6, marginBottom: 3, alignItems: "flex-start" }}>
                <StatusDot status={t.status} />
                <Badge label={t.id} color={color} small />
                <span style={{ color: "#94a3b8", fontSize: 12, flex: 1 }}>{t.name}</span>
                <span style={{ color: "#475569", fontSize: 11 }}>{t.story}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── MAIN APP ────────────────────────────────────────────────────────────────

const TABS = [
  { id: "prd", label: "PRD", icon: "📋", desc: "Product Requirements" },
  { id: "ard", label: "ARD", icon: "🏗️", desc: "Architecture" },
  { id: "features", label: "Features & Stories", icon: "📖", desc: "Epics → Features → Stories → Tasks" },
  { id: "matrix", label: "Matrix", icon: "🗺️", desc: "Releases × Milestones × Epics" },
  { id: "search", label: "Search", icon: "🔍", desc: "Find any item" },
];

export default function App() {
  const [tab, setTab] = useState("prd");
  const [search, setSearch] = useState("");
  const [activeSearch, setActiveSearch] = useState("");

  useEffect(() => {
    if (tab === "search") {
      setActiveSearch(search);
    }
  }, [search, tab]);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a1628",
      fontFamily: "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif",
      color: "#e2e8f0",
    }}>
      {/* Header */}
      <div style={{ background: "#060e1a", borderBottom: "1px solid #1e293b", padding: "14px 24px", position: "sticky", top: 0, zIndex: 100, display: "flex", alignItems: "center", gap: 16 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 28, height: 28, background: "linear-gradient(135deg, #2dd4bf, #60a5fa)", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>〜</div>
            <span style={{ color: "#e2e8f0", fontWeight: 800, fontSize: 16, letterSpacing: "-0.01em" }}>Ferromode</span>
            <Badge label="v0.1 planning" color="#2dd4bf" small />
          </div>
          <div style={{ color: "#475569", fontSize: 11, marginTop: 2 }}>High-Performance EMD · Rust Core · 6 Language Bindings</div>
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ color: "#475569", fontSize: 11 }}>237 tasks</span>
          <span style={{ color: "#334155" }}>·</span>
          <span style={{ color: "#475569", fontSize: 11 }}>53 stories</span>
          <span style={{ color: "#334155" }}>·</span>
          <span style={{ color: "#475569", fontSize: 11 }}>8 epics</span>
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 57px)" }}>
        {/* Sidebar */}
        <div style={{ width: 220, background: "#060e1a", borderRight: "1px solid #1e293b", padding: "16px 12px", flexShrink: 0, overflowY: "auto" }}>
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                display: "flex", alignItems: "flex-start", gap: 10, width: "100%",
                background: tab === t.id ? "#1e293b" : "transparent",
                border: tab === t.id ? "1px solid #334155" : "1px solid transparent",
                borderRadius: 6, padding: "9px 12px", cursor: "pointer", textAlign: "left", marginBottom: 4,
              }}
            >
              <span style={{ fontSize: 16, marginTop: 1 }}>{t.icon}</span>
              <div>
                <div style={{ color: tab === t.id ? "#e2e8f0" : "#64748b", fontWeight: tab === t.id ? 700 : 400, fontSize: 13 }}>{t.label}</div>
                <div style={{ color: "#475569", fontSize: 10 }}>{t.desc}</div>
              </div>
            </button>
          ))}

          <div style={{ marginTop: 16, borderTop: "1px solid #1e293b", paddingTop: 16 }}>
            <div style={{ color: "#475569", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>Epics</div>
            {epics.map(e => (
              <div key={e.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 6px", marginBottom: 2 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: e.color, flexShrink: 0 }} />
                <span style={{ color: "#64748b", fontSize: 11 }}>{e.label}: {e.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Main content */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px" }}>
          {tab === "search" && (
            <div style={{ marginBottom: 16 }}>
              <input
                autoFocus
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search tasks, stories, features…"
                style={{
                  width: "100%", background: "#1e293b", border: "1px solid #334155", borderRadius: 6,
                  padding: "10px 14px", color: "#e2e8f0", fontSize: 14, outline: "none", boxSizing: "border-box"
                }}
              />
            </div>
          )}

          {tab === "prd" && <PRDView />}
          {tab === "ard" && <ARDView />}
          {tab === "features" && <FeaturesView />}
          {tab === "matrix" && <MatrixView />}
          {tab === "search" && <SearchView query={activeSearch} />}
        </div>
      </div>
    </div>
  );
}
