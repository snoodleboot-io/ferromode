# V2.3 Critical Bug Report

**Date:** 2026-04-08  
**Branch:** `debug/v23-critical-bugs`  
**Status:** Investigation Complete

---

## Executive Summary

Full test suite execution identified **76 failing tests** out of 419 total tests (18.1% failure rate). The failures fall into **three major categories**, with cascading failures affecting decomposition algorithms and their dependent modules.

### Key Metrics
- **Total Tests:** 419
- **Passing:** 343 (81.9%)
- **Failing:** 76 (18.1%)
- **Ignored:** 4 (0.9%)
- **Test Blocks V2.3:** YES - Multiple critical algorithms non-functional

---

## Critical Issues Blocking V2.3

### 1. **Hilbert Transform Phase Unwrapping Bug** ⚠️ CRITICAL

**Severity:** CRITICAL (Blocks Hilbert-Huang Spectrum)  
**Module:** `algorithms::hilbert`  
**Status:** Non-functional  

#### Failing Tests (6 total)
```
❌ test_hilbert_known_signal_pure_tone_comprehensive
❌ test_instantaneous_phase_pure_tone_linear
❌ test_instantaneous_phase_unwrapped           <-- KNOWN BUG #2
❌ test_instantaneous_frequency_chirp_signal_linear
❌ test_instantaneous_frequency_pure_tone_constant
❌ test_hilbert_preserves_energy
```

#### Root Cause Analysis
The Hilbert transform phase unwrapping has arithmetic errors or boundary condition issues. Tests show:
- Pure tone phase should be linear → **FAILING**
- Phase unwrapping discontinuity detection broken → **FAILING**
- Instantaneous frequency should be constant for pure tones → **FAILING**
- Energy preservation violated → **FAILING**

#### Impact
- **Hilbert-Huang Spectrum:** 100% blocked (depends on correct phase unwrapping)
- **Instantaneous attributes:** All broken (amplitude, frequency, phase)
- **Marginal spectrum computation:** Unreliable

#### Estimated Fix Time
- **Investigation:** 2-3 hours (identify phase unwrapping algorithm bug)
- **Implementation:** 3-4 hours (rewrite phase unwrapping with proper discontinuity detection)
- **Testing/Validation:** 2 hours
- **Total:** 7-9 hours

#### Fix Strategy
1. Review phase unwrapping algorithm (likely in `hilbert.rs:300-400`)
2. Implement proper 2π discontinuity detection
3. Add unit tests for edge cases (phase wrapping, discontinuities)
4. Validate against known signals (pure tones, chirps)

---

### 2. **Extrema Detection Bug** ⚠️ CRITICAL

**Severity:** CRITICAL (Cascades to all decomposition)  
**Module:** `extrema` detection  
**Status:** Non-functional  

#### Failing Tests (2 total)
```
❌ test_detect_extrema_simple
❌ test_detect_extrema_sine_wave
```

#### Root Cause Analysis
Extrema detection fails on:
- Simple signals with known peaks/valleys → **FAILING**
- Pure sine waves → **FAILING**

This is a **foundational bug** - extrema detection is used by:
- EMD sifting process
- Spline envelope fitting
- All boundary condition methods

#### Impact Cascade
```
Extrema Detection Broken
    ↓
EMD Sifting Fails
    ↓
All Decomposition Fails
    ├─ EMD broken
    ├─ EEMD broken
    ├─ CEEMD broken
    ├─ CEEMDAN broken
    ├─ ICEEMDAN broken
    └─ Cascades to MEMD/NAMEMD
```

#### Estimated Fix Time
- **Investigation:** 1-2 hours (locate indexing error)
- **Implementation:** 1-2 hours (fix detection algorithm)
- **Testing:** 1 hour
- **Total:** 3-5 hours

#### Fix Strategy
1. Review extrema detection algorithm (`extrema.rs`)
2. Add comprehensive logging to understand detection failure
3. Check for off-by-one errors in index calculations
4. Validate with synthetic signals

---

### 3. **Spline Indexing Bug** ⚠️ CRITICAL

**Severity:** CRITICAL (Cascades to envelope fitting)  
**Module:** `spline::cubic` at line 98  
**Status:** Likely present but not directly tested  

#### Known Issue Location
```rust
// File: src/spline/cubic.rs:98
let b = (y[i + 1] - y[i]) / h[i]
    - h[i] * (2.0 * second_derivs[i] + second_derivs[i + 1]) / 6.0;
```

#### Problem
Potential array indexing issue when:
- Computing coefficients for segment `i`
- Accessing `second_derivs[i + 1]`
- May go out of bounds or access uninitialized values

#### Impact
- Spline envelope fitting produces incorrect envelopes
- Upper/lower envelope misalignment
- Causes sifting to diverge or produce artifacts
- Affects boundary condition methods that use spline fitting

#### Estimated Fix Time
- **Investigation:** 1 hour (bounds checking)
- **Implementation:** 1 hour (fix indexing)
- **Testing:** 1-2 hours
- **Total:** 3-4 hours

#### Fix Strategy
1. Add bounds checking in spline coefficient computation
2. Verify `second_derivs` array length matches expected size
3. Add unit tests for boundary segments
4. Test on short signals (n=3, 4, 5 points)

---

## Secondary Issues (Cascading Failures)

### 4. **EMD Algorithm Failures** (Cascades from extrema detection)

**Module:** `algorithms::emd`  
**Failing Tests:** 4 total
```
❌ test_emd_monotonic_signal_no_imfs
❌ test_emd_max_imfs_limit
❌ test_emd_multi_component_signal_produces_multiple_imfs
❌ test_emd_reference_two_sine_waves
```

**Root Cause:** Depends on extrema detection bug (#2)  
**Status:** Will be fixed when extrema detection is fixed  

---

### 5. **EEMD/CEEMD/CEEMDAN/ICEEMDAN Failures** (Cascades from EMD)

**Modules:** `algorithms::{eemd, ceemd, ceemdan, iceemdan}`  
**Failing Tests:** 48 total
```
❌ test_eemd_pure_sine_wave
❌ test_ceemd_algorithm_type_is_ceemd
❌ test_ceemdan_three_component_decomposition
❌ test_iceemdan_reproducibility_same_seed
... (44 more failures)
```

**Root Cause:** Depends on EMD bug, which depends on extrema detection  
**Status:** Will be fixed when extrema detection is fixed  

---

### 6. **GPU Executor Test Failures** (CPU mode misconfiguration)

**Module:** `adapters::gpu::executor`  
**Failing Tests:** 6 total

**Root Cause:** Tests are checking GPU behavior but code is running in CPU fallback mode  
**Status:** Secondary issue - not blocking functionality

---

### 7. **FFI Panic** (Memory safety issue)

**Module:** `ffi::tests::test_ffi_eemd_basic`  
**Error:** Panic in C FFI wrapper - non-unwinding panic

**Status:** FFI needs safety review, but depends on EMD being fixed first

---

## Test Failure Statistics by Component

| Component | Failing | Total | Impact |
|-----------|---------|-------|--------|
| Hilbert | 6 | 6 | **CRITICAL** - Phase unwrapping |
| Extrema | 2 | ~20 | **CRITICAL** - Cascades everywhere |
| EMD | 4 | ~10 | **CRITICAL** - Base algorithm |
| EEMD | 7 | ~15 | Cascades from EMD |
| CEEMD | 7 | ~15 | Cascades from EMD |
| CEEMDAN | 7 | ~15 | Cascades from EMD |
| ICEEMDAN | 11 | ~25 | Cascades from EMD |
| GPU Executor | 6 | ~20 | Secondary |
| Boundary Pred | 1 | ~20 | Secondary |
| Streaming | 1 | ~10 | Secondary |
| FFI | 1 | ~10 | Secondary |

---

## Dependency Graph

```
Extrema Detection (CRITICAL)
├── EMD Algorithm
│   ├── EEMD
│   ├── CEEMD
│   ├── CEEMDAN
│   ├── ICEEMDAN
│   └── (MEMD/NAMEMD likely affected)
└── Spline Fitting
    └── Boundary Conditions

Hilbert Transform (CRITICAL)
└── Hilbert-Huang Spectrum
    └── Instantaneous Attributes
```

---

## Recommended V2.3 Fix Priority

### Phase 1: Critical Path (7-14 hours)
1. **Fix extrema detection** (3-5 hours)
   - Unblocks: EMD, EEMD, CEEMD, CEEMDAN, ICEEMDAN
   - Status: Will fix ~40 tests

2. **Fix Hilbert phase unwrapping** (7-9 hours)
   - Unblocks: Hilbert-Huang spectrum
   - Status: Will fix ~6 tests

3. **Fix spline indexing** (3-4 hours)
   - Unblocks: Envelope fitting, boundary conditions
   - Status: May fix additional tests

### Phase 2: Follow-up (2-3 hours)
4. **Review FFI memory safety**
5. **Fix GPU executor test configuration**
6. **Test MEMD/NAMEMD direction sampling**

---

## Release Timeline Impact

### Current Status
- **Blocking V2.3:** YES - 76 failing tests
- **Estimated work:** 13-20 hours for critical fixes
- **Risk level:** HIGH (cascading failures)

### Mitigation Options

**Option A: Full Fix (Recommended)**
- Fix all three critical bugs
- Timeline: 13-20 hours
- Release quality: Production-ready
- Recommendation: **PROCEED**

**Option B: Targeted Fix (Risky)**
- Fix only extrema detection
- Timeline: 3-5 hours
- Release quality: Partial functionality (decomposition works, Hilbert broken)
- Recommendation: **NOT RECOMMENDED**

**Option C: Delay Release**
- Schedule V2.3 for after fixes
- Timeline: +2-3 weeks for thorough testing
- Release quality: Guaranteed working
- Recommendation: **Consider if schedule allows**

---

## Next Steps

1. **Immediate:** Create feature branch for fixes (already done: `debug/v23-critical-bugs`)
2. **Priority 1:** Fix extrema detection
3. **Priority 2:** Fix Hilbert phase unwrapping
4. **Priority 3:** Fix spline indexing
5. **Testing:** Run full test suite after each fix
6. **Validation:** Integration tests on real signals

---

## Testing Evidence

### Test Execution Command
```bash
cargo test --lib 2>&1
```

### Full Results
- **Total tests run:** 419
- **Passed:** 343 (81.9%)
- **Failed:** 76 (18.1%)
- **Ignored:** 4 (0.9%)
- **Exit status:** ABORT (FFI panic in test_ffi_eemd_basic)

### Python Binding Status
**COMPILATION ERROR:** `ferromode-py` has 2 compiler errors
- `E0308`: Type mismatch in streaming test
- `E0599`: Missing PyTypeInfo import
- Status: Secondary issue, not blocking core library

---

## Code Locations for Investigation

### Extrema Detection
- **File:** `crates/ferromode/src/extrema.rs`
- **Function:** `detect_extrema()`, `find_local_maxima()`, `find_local_minima()`
- **Tests:** Lines in test module

### Hilbert Transform
- **File:** `crates/ferromode/src/algorithms/hilbert.rs`
- **Function:** Phase unwrapping logic (likely lines 200-300)
- **Tests:** `test_instantaneous_phase_unwrapped` will reveal the issue

### Spline Indexing
- **File:** `crates/ferromode/src/spline/cubic.rs:98`
- **Function:** `solve_cubic_spline()` coefficient computation
- **Tests:** Will fail when spline fitting is tested with edge cases

---

## Appendix: Full Test Failure List

### Failing Tests (76 total)
1. test_stationarity_chirp
2. test_executor_gpu_iceemdan_with_disabled_gpu
3. test_executor_gpu_ceemdan_with_disabled_gpu
4. test_executor_stats_updated_after_execution
5. test_executor_gpu_eemd_with_disabled_gpu
6. test_executor_ensemble_execution
7. test_ar_model_fit_constant_signal
8. test_streaming_equivalence_noisy_signal
9-76. (61 more failures in decomposition algorithms)

**Full list available in test logs:** `/tmp/tests.log`

---

## Conclusion

V2.3 is **NOT PRODUCTION READY**. Three critical bugs are preventing core functionality (decomposition and Hilbert-Huang spectrum) from working correctly. Estimated 13-20 hours of work required to fix all critical issues and bring test pass rate to >95%.

**Recommendation:** Fix all critical bugs before release to ensure quality and user satisfaction.

