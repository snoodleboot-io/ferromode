# Bug Fix Week Plan: V2.3 Critical Issues

**Start Date:** 2026-04-09  
**Target Duration:** 5 days (13-20 hours)  
**Branch:** `bugfix/v2x-critical-issues` (create from main)  
**Expected Outcome:** >95% test pass rate, V2.3 unblocked

---

## Overview

This document provides day-by-day breakdown for fixing the three critical bugs blocking V2.3 development. Each day includes:
- Root cause analysis
- Specific file locations
- Code examples of bugs and fixes
- Test commands to verify
- Rollback procedures
- Expected test pass rate improvement

**Total estimated effort:** 15-24 hours distributed across 1-2 weeks

---

## Day 1: Fix Extrema Detection (3-5 hours)

### Issue Summary

**Bug:** Extrema detection fails on simple signals  
**Tests Failing:** 2 direct + ~40 cascading  
**Impact:** Breaks all decomposition algorithms (EMD, EEMD, CEEMDAN, MEMD, NAMEMD)  
**Severity:** CRITICAL  
**Fix Priority:** 1st (highest ROI)

### Root Cause Analysis

**Location:** `crates/ferromode/src/extrema.rs`

**Symptoms:**
- `test_detect_extrema_simple` failing (basic functionality)
- `test_detect_extrema_sine_wave` failing (pure signal)
- Cascades to all EMD-based algorithms (40+ tests fail)

**Likely Issues:**
1. Off-by-one error in index calculations
2. Boundary condition miscalculation (first/last samples)
3. Peak/valley comparison logic error
4. Array access out of bounds

### Investigation Steps

**Step 1: Examine the failing tests**

```rust
// File: crates/ferromode/src/extrema.rs
// Find these tests:

#[test]
fn test_detect_extrema_simple() {
    // Input: Simple signal with known peaks/valleys
    // Expected: Correct peak and valley indices
    // Actual: FAILING - wrong indices or empty result
}

#[test]
fn test_detect_extrema_sine_wave() {
    // Input: Pure sine wave
    // Expected: Regular peaks at known positions
    // Actual: FAILING - peaks not detected
}
```

**Step 2: Review the extrema detection algorithm**

```rust
// Expected structure in extrema.rs:

pub fn detect_extrema(signal: &[f64]) -> Extrema {
    let maxima = find_local_maxima(signal);
    let minima = find_local_minima(signal);
    Extrema { maxima, minima }
}

fn find_local_maxima(signal: &[f64]) -> Vec<usize> {
    // ISSUE: Likely off-by-one in loop indices
    // Check: for i in 1..signal.len()-1 vs for i in 1..signal.len()
    // Check: Comparison logic for peaks (y[i] > y[i-1] && y[i] > y[i+1])
}

fn find_local_minima(signal: &[f64]) -> Vec<usize> {
    // ISSUE: Same as maxima
    // Check: Comparison logic for valleys
}
```

**Step 3: Run tests with debug output**

```bash
# Enable backtrace for debugging
RUST_BACKTRACE=1 cargo test test_detect_extrema_simple -- --nocapture

# Look for:
# - What indices are being returned
# - How many extrema found vs expected
# - Are boundary indices included when they shouldn't be
```

### Likely Fixes

#### Possibility A: Off-by-One in Loop

**Before (INCORRECT):**
```rust
fn find_local_maxima(signal: &[f64]) -> Vec<usize> {
    let mut maxima = Vec::new();
    
    // BUG: Loop goes to len() instead of len()-1
    for i in 1..signal.len() {
        if signal[i] > signal[i-1] && signal[i] > signal[i+1] {
            // ❌ CRASH: When i == len()-1, signal[i+1] is out of bounds
            maxima.push(i);
        }
    }
    maxima
}
```

**After (CORRECT):**
```rust
fn find_local_maxima(signal: &[f64]) -> Vec<usize> {
    let mut maxima = Vec::new();
    
    // FIX: Loop explicitly ends at len()-1
    for i in 1..signal.len()-1 {
        if signal[i] > signal[i-1] && signal[i] > signal[i+1] {
            // ✅ CORRECT: i+1 always in bounds
            maxima.push(i);
        }
    }
    maxima
}
```

#### Possibility B: Boundary Logic Error

**Before (INCORRECT):**
```rust
fn find_local_maxima(signal: &[f64]) -> Vec<usize> {
    let mut maxima = Vec::new();
    
    for i in 1..signal.len()-1 {
        // BUG: Wrong comparison logic
        if signal[i] >= signal[i-1] && signal[i] >= signal[i+1] {
            // ❌ This includes plateaus and ties, causing false positives
            maxima.push(i);
        }
    }
    maxima
}
```

**After (CORRECT):**
```rust
fn find_local_maxima(signal: &[f64]) -> Vec<usize> {
    let mut maxima = Vec::new();
    
    for i in 1..signal.len()-1 {
        // FIX: Strict inequality for true local maxima
        if signal[i] > signal[i-1] && signal[i] > signal[i+1] {
            // ✅ CORRECT: Only returns true local maxima
            maxima.push(i);
        }
    }
    maxima
}
```

#### Possibility C: Missing Edge Case Handling

**Before (INCOMPLETE):**
```rust
fn find_local_maxima(signal: &[f64]) -> Vec<usize> {
    if signal.is_empty() {
        return Vec::new();  // Missing: handle len == 1, len == 2
    }
    
    let mut maxima = Vec::new();
    // ... rest of implementation
    maxima
}
```

**After (COMPLETE):**
```rust
fn find_local_maxima(signal: &[f64]) -> Vec<usize> {
    // Handle edge cases explicitly
    if signal.len() < 3 {
        return Vec::new();  // Can't have local extrema with < 3 points
    }
    
    let mut maxima = Vec::new();
    for i in 1..signal.len()-1 {
        if signal[i] > signal[i-1] && signal[i] > signal[i+1] {
            maxima.push(i);
        }
    }
    maxima
}
```

### Implementation Strategy

**Step 1: Read the actual code**
```bash
# Examine the implementation
cat crates/ferromode/src/extrema.rs | head -100

# Look specifically for:
# - Loop bounds in find_local_maxima()
# - Loop bounds in find_local_minima()
# - Comparison operators (>, >=, <, <=)
# - Edge case handling
```

**Step 2: Add defensive logging**
```rust
// Add this before committing fix:
#[cfg(test)]
fn log_extrema_detection(signal: &[f64], maxima: &[usize], minima: &[usize]) {
    eprintln!("Signal length: {}", signal.len());
    eprintln!("Found {} maxima: {:?}", maxima.len(), maxima);
    eprintln!("Found {} minima: {:?}", minima.len(), minima);
    
    // Verify indices are valid
    for &idx in maxima.iter().chain(minima.iter()) {
        assert!(idx < signal.len(), "Index {} out of bounds for signal of length {}", idx, signal.len());
    }
}
```

**Step 3: Write comprehensive test to catch regression**
```rust
#[test]
fn test_extrema_bounds_checking() {
    // Test short signals
    assert_eq!(detect_extrema(&[1.0]).minima.len(), 0);  // Len 1
    assert_eq!(detect_extrema(&[1.0, 2.0]).maxima.len(), 0);  // Len 2
    
    // Test 3-point signal with clear extrema
    let signal = vec![1.0, 2.0, 1.0];  // Peak at index 1
    let extrema = detect_extrema(&signal);
    assert_eq!(extrema.maxima, vec![1], "Should detect peak at index 1");
    
    // Test 5-point signal
    let signal = vec![1.0, 2.0, 1.0, 2.0, 1.0];  // Peaks at 1, 3; valleys at 0, 2, 4
    let extrema = detect_extrema(&signal);
    assert_eq!(extrema.maxima, vec![1, 3], "Should detect peaks at indices 1 and 3");
    assert_eq!(extrema.minima, vec![2], "Should detect valley at index 2");
    
    // Test pure sine wave
    let signal: Vec<f64> = (0..101)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin())
        .collect();
    let extrema = detect_extrema(&signal);
    assert!(extrema.maxima.len() > 0, "Should detect maxima in sine wave");
    assert!(extrema.minima.len() > 0, "Should detect minima in sine wave");
}
```

### Verification

**Test Command to Verify Fix:**

```bash
# Build and run extrema-specific tests
cargo test test_detect_extrema_ -- --nocapture

# Expected output:
# test test_detect_extrema_simple ... ok
# test test_detect_extrema_sine_wave ... ok

# Run full test suite to check cascading fixes
cargo test --lib 2>&1 | grep "test result:"

# Expected improvement:
# Before: 343 passed, 76 failed
# After: ~385+ passed, ~50 failed (40 cascading extrema tests fixed)
```

**Validation Checklist:**
- [ ] `test_detect_extrema_simple` passes
- [ ] `test_detect_extrema_sine_wave` passes
- [ ] No crashes on edge cases (empty, 1-element, 2-element signals)
- [ ] EMD tests start passing (cascading fix)
- [ ] No new test failures introduced

### Rollback Procedure

If the fix introduces regressions:

```bash
# Option 1: Revert to last known good state
git reset --hard main

# Option 2: Cherry-pick specific commits
git log --oneline  # Find commit hash
git revert <commit-hash>

# Option 3: Use git bisect to find regression
git bisect start
git bisect bad HEAD
git bisect good main
# Test incrementally
```

### Time Tracking

| Activity | Estimated | Actual |
|----------|-----------|--------|
| Investigation & reading code | 1-1.5h | — |
| Implementing fix | 1-2h | — |
| Testing & validation | 0.5-1h | — |
| Debugging if needed | 0.5-1h | — |
| **Total** | **3-5h** | — |

---

## Day 2: Fix Hilbert Phase Unwrapping (7-9 hours)

### Issue Summary

**Bug:** Phase unwrapping has arithmetic/boundary errors  
**Tests Failing:** 6 direct (no cascading)  
**Impact:** Breaks Hilbert-Huang spectral analysis, instantaneous attributes  
**Severity:** CRITICAL  
**Fix Priority:** 2nd (after extrema)

### Root Cause Analysis

**Location:** `crates/ferromode/src/algorithms/hilbert.rs`

**Failing Tests:**
```
❌ test_hilbert_known_signal_pure_tone_comprehensive
❌ test_instantaneous_phase_pure_tone_linear
❌ test_instantaneous_phase_unwrapped
❌ test_instantaneous_frequency_chirp_signal_linear
❌ test_instantaneous_frequency_pure_tone_constant
❌ test_hilbert_preserves_energy
```

**Symptoms:**
- Pure tone phase should be linear but isn't
- Phase unwrapping has discontinuities at 2π boundaries
- Energy not preserved in transform
- Instantaneous frequency calculation is wrong

### Mathematical Background

**Hilbert Transform:**
```
H(x(t)) = (1/π) ∫ x(τ)/(t-τ) dτ
```

**Analytic Signal:**
```
z(t) = x(t) + i*H(x(t))
     = A(t) * e^(i*φ(t))
```

**Instantaneous Phase (WRAPPED):**
```
φ(t) = atan2(H(x(t)), x(t))  // Range: [-π, π]
```

**Instantaneous Phase (UNWRAPPED):**
```
φ_unwrapped(t) = continuous phase without 2π jumps
```

**Pure Tone Phase (Should Be Linear):**
```
x(t) = sin(ωt + φ₀)
φ(t) = ωt + φ₀  (linear relationship!)
```

### Investigation Steps

**Step 1: Review the phase unwrapping algorithm**

```rust
// File: crates/ferromode/src/algorithms/hilbert.rs
// Look for: Phase unwrapping function (likely ~lines 200-300)

// INCORRECT version might look like:
fn unwrap_phase(wrapped_phase: &[f64]) -> Vec<f64> {
    let mut unwrapped = vec![wrapped_phase[0]];
    
    for i in 1..wrapped_phase.len() {
        // BUG: Not handling 2π discontinuities correctly
        let diff = wrapped_phase[i] - wrapped_phase[i-1];
        // Missing: if diff > π, subtract 2π; if diff < -π, add 2π
        unwrapped.push(unwrapped[i-1] + diff);
    }
    
    unwrapped
}
```

**Step 2: Test with pure tone**

```bash
# Run a specific test with output
RUST_LOG=debug cargo test test_instantaneous_phase_pure_tone_linear -- --nocapture --test-threads=1

# Look for:
# - Phase values over time
# - Are they linear?
# - Are there sudden jumps?
```

**Step 3: Check phase computation**

```rust
// Expected Hilbert transform steps:

pub fn hilbert_transform(signal: &[f64]) -> Vec<Complex64> {
    let analytic = compute_analytic_signal(signal);
    analytic
}

fn compute_analytic_signal(signal: &[f64]) -> Vec<Complex64> {
    // Step 1: Compute FFT of signal
    // Step 2: Zero out negative frequencies
    // Step 3: Inverse FFT → analytic signal
    // Step 4: Extract phase from complex values
}

pub fn instantaneous_phase(analytic: &[Complex64]) -> Vec<f64> {
    analytic.iter()
        .map(|z| z.arg())  // This gives wrapped phase [-π, π]
        .collect()
}

pub fn unwrap_phase_discontinuities(wrapped: &[f64]) -> Vec<f64> {
    // BUG LOCATION: This function likely has the issue
    
    let mut unwrapped = Vec::with_capacity(wrapped.len());
    unwrapped.push(wrapped[0]);
    
    let mut cumulative_offset = 0.0;
    
    for i in 1..wrapped.len() {
        let diff = wrapped[i] - wrapped[i-1];
        
        // Detect and correct 2π jumps
        let corrected_diff = if diff > std::f64::consts::PI {
            diff - 2.0 * std::f64::consts::PI  // Descending jump
        } else if diff < -std::f64::consts::PI {
            diff + 2.0 * std::f64::consts::PI  // Ascending jump
        } else {
            diff  // Normal progression
        };
        
        cumulative_offset += corrected_diff;
        unwrapped.push(wrapped[0] + cumulative_offset);
    }
    
    unwrapped
}
```

### Likely Fixes

#### Fix 1: Implement proper phase unwrapping

**Before (BROKEN):**
```rust
fn unwrap_phase(wrapped: &[f64]) -> Vec<f64> {
    let mut unwrapped = vec![wrapped[0]];
    
    for i in 1..wrapped.len() {
        let diff = wrapped[i] - wrapped[i-1];
        unwrapped.push(unwrapped[i-1] + diff);  // ❌ No discontinuity handling
    }
    unwrapped
}
```

**After (CORRECT):**
```rust
fn unwrap_phase(wrapped: &[f64]) -> Vec<f64> {
    if wrapped.is_empty() {
        return Vec::new();
    }
    
    let mut unwrapped = Vec::with_capacity(wrapped.len());
    unwrapped.push(wrapped[0]);
    
    let two_pi = 2.0 * std::f64::consts::PI;
    let pi = std::f64::consts::PI;
    
    for i in 1..wrapped.len() {
        let diff = wrapped[i] - wrapped[i-1];
        
        // Correct for 2π discontinuities
        let corrected_diff = if diff > pi {
            diff - two_pi  // Wrapped backwards (ascending jump)
        } else if diff < -pi {
            diff + two_pi  // Wrapped forwards (descending jump)
        } else {
            diff  // No discontinuity
        };
        
        unwrapped.push(unwrapped[i-1] + corrected_diff);  // ✅ CORRECT
    }
    
    unwrapped
}
```

#### Fix 2: Ensure phase computation correctness

```rust
// Verify complex-to-phase conversion
pub fn instantaneous_phase(analytic: &[Complex64]) -> Vec<f64> {
    analytic.iter()
        .map(|z| {
            // atan2(imag, real) gives phase in [-π, π]
            z.im.atan2(z.re)  // ✅ CORRECT order
            // NOT: z.re.atan2(z.im)  // ❌ WRONG order
        })
        .collect()
}

pub fn instantaneous_amplitude(analytic: &[Complex64]) -> Vec<f64> {
    analytic.iter()
        .map(|z| z.norm())  // sqrt(re² + im²)
        .collect()
}
```

#### Fix 3: Implement energy preservation check

```rust
// Energy should be preserved: E(x) = E(H(x))
fn verify_hilbert_energy_preservation(signal: &[f64], analytic: &[Complex64]) -> bool {
    let signal_energy: f64 = signal.iter().map(|&x| x * x).sum();
    
    // For analytic signal, real part is original, imaginary is Hilbert transform
    let analytic_energy: f64 = analytic.iter()
        .map(|z| z.re * z.re + z.im * z.im)
        .sum();
    
    // Should be approximately equal
    let relative_error = (analytic_energy - signal_energy).abs() / signal_energy;
    
    // ✅ Allow small numerical error
    relative_error < 0.01  // 1% tolerance
}
```

### Implementation Strategy

**Step 1: Review current implementation**

```bash
# Find the exact implementation
grep -n "unwrap" crates/ferromode/src/algorithms/hilbert.rs

# Review phase computation
grep -n "phase\|arg" crates/ferromode/src/algorithms/hilbert.rs
```

**Step 2: Add comprehensive tests**

```rust
#[test]
fn test_phase_unwrapping_linear_for_pure_tone() {
    // Generate pure tone: x(t) = sin(2π*f*t)
    let freq = 5.0;  // 5 Hz
    let samples = 100;
    let signal: Vec<f64> = (0..samples)
        .map(|i| {
            let t = i as f64 / samples as f64;
            (2.0 * std::f64::consts::PI * freq * t).sin()
        })
        .collect();
    
    let analytic = compute_analytic_signal(&signal);
    let phase = instantaneous_phase(&analytic);
    let unwrapped = unwrap_phase(&phase);
    
    // Phase should increase linearly (with small numerical noise)
    let mut phase_differences = Vec::new();
    for i in 1..unwrapped.len() {
        phase_differences.push(unwrapped[i] - unwrapped[i-1]);
    }
    
    // Average phase increment
    let avg_diff: f64 = phase_differences.iter().sum::<f64>() / phase_differences.len() as f64;
    
    // Standard deviation
    let variance: f64 = phase_differences.iter()
        .map(|&d| (d - avg_diff).powi(2))
        .sum::<f64>() / phase_differences.len() as f64;
    let std_dev = variance.sqrt();
    
    // Phase should be very linear (low standard deviation)
    assert!(std_dev < 0.1, "Phase not linear: std_dev = {}", std_dev);
}

#[test]
fn test_phase_unwrapping_detects_discontinuities() {
    // Synthetic wrapped phase with 2π jumps
    let wrapped = vec![
        0.0, 0.1, 0.2, 0.3, 0.4,      // Increasing normally
        -3.0, -2.9, -2.8,              // Jump at 2π boundary (3.1 ≈ π)
        -2.7, -2.6, -2.5, -2.4, -2.3,  // Continue increasing
    ];
    
    let unwrapped = unwrap_phase(&wrapped);
    
    // Unwrapped should be monotonically increasing
    for i in 1..unwrapped.len() {
        assert!(unwrapped[i] > unwrapped[i-1], 
                "Unwrapped phase should be monotonic at index {}", i);
    }
}
```

**Step 3: Validate with reference values**

```rust
#[test]
fn test_instantaneous_frequency_for_chirp() {
    // Chirp: f(t) = f0 + (f1-f0)*t
    let f0 = 1.0;
    let f1 = 10.0;
    let duration = 1.0;
    let samples = 1000;
    
    let signal: Vec<f64> = (0..samples)
        .map(|i| {
            let t = i as f64 / samples as f64;
            let f = f0 + (f1 - f0) * t;
            (2.0 * std::f64::consts::PI * f * t).sin()
        })
        .collect();
    
    let analytic = compute_analytic_signal(&signal);
    let phase = instantaneous_phase(&analytic);
    let unwrapped = unwrap_phase(&phase);
    let inst_freq = compute_instantaneous_frequency(&unwrapped);
    
    // Frequency should increase linearly from f0 to f1
    // Check a few key points
    assert!(inst_freq[0] > f0 * 0.8, "Start frequency too low");
    assert!(inst_freq[inst_freq.len()-1] < f1 * 1.2, "End frequency too high");
}
```

### Verification

**Test Command:**

```bash
# Run Hilbert transform specific tests
cargo test hilbert -- --nocapture

# Expected output:
# test test_hilbert_known_signal_pure_tone_comprehensive ... ok
# test test_instantaneous_phase_pure_tone_linear ... ok
# test test_instantaneous_phase_unwrapped ... ok
# test test_instantaneous_frequency_chirp_signal_linear ... ok
# test test_instantaneous_frequency_pure_tone_constant ... ok
# test test_hilbert_preserves_energy ... ok

# Verify with full test run
cargo test --lib 2>&1 | tail -3
# Expected: test result: ok. (with all 6 Hilbert tests fixed)
```

**Validation Checklist:**
- [ ] Pure tone phase is linear (regression < 0.1)
- [ ] Phase unwrapping handles 2π discontinuities
- [ ] Instantaneous frequency constant for pure tones
- [ ] Instantaneous frequency increases for chirps
- [ ] Energy preservation within 1% tolerance
- [ ] All 6 Hilbert tests passing

### Time Tracking

| Activity | Estimated | Actual |
|----------|-----------|--------|
| Understanding phase unwrapping math | 1-1.5h | — |
| Reading existing implementation | 1h | — |
| Implementing fix & validation | 3-4h | — |
| Writing comprehensive tests | 1-2h | — |
| **Total** | **7-9h** | — |

---

## Day 3: Fix Spline Indexing (3-4 hours)

### Issue Summary

**Bug:** Array indexing issue in spline coefficient computation  
**Tests Failing:** 5-10 indirect (envelope quality)  
**Impact:** Spline envelope fitting produces incorrect results  
**Severity:** CRITICAL  
**Fix Priority:** 3rd (quality improvement)

### Root Cause Analysis

**Location:** `crates/ferromode/src/spline/cubic.rs:98`

**Known Issue:**
```rust
// File: src/spline/cubic.rs:98
let b = (y[i + 1] - y[i]) / h[i]
    - h[i] * (2.0 * second_derivs[i] + second_derivs[i + 1]) / 6.0;
    //                                  ↑ Potential out of bounds
```

**Symptoms:**
- Spline envelope fitting produces artifacts
- Upper/lower envelopes misaligned
- Sifting may diverge
- Boundary segments computed incorrectly

### Cubic Spline Mathematical Background

**Natural Cubic Spline:**
- Divides domain into segments
- Each segment is a cubic polynomial
- Continuous second derivative at knots
- Second derivatives solved via tridiagonal system

**Coefficient Computation:**
```
For segment i:
- h[i] = x[i+1] - x[i]  (distance between knots)
- a[i] = y[i]
- b[i], c[i], d[i] = computed from second derivatives
```

**Key Loop:**
```rust
for i in 0..n-1 {  // n = number of data points
    // Here, accessing second_derivs[i+1] is valid
    // because: i ranges from 0 to n-2
    // so i+1 ranges from 1 to n-1 ✓
    
    if i == n-1 {  // BUG if this check is missing
        // second_derivs[i+1] = second_derivs[n]
        // OUT OF BOUNDS! second_derivs has n elements (indices 0..n-1)
    }
}
```

### Investigation Steps

**Step 1: Review spline implementation**

```bash
# Find the spline code
cat crates/ferromode/src/spline/cubic.rs | head -150

# Look for:
# - Array size n
# - Loop bounds
# - second_derivs allocation
# - Coefficient computation loops
```

**Step 2: Check array allocation**

```rust
// Expected correct implementation:

pub fn solve_cubic_spline(x: &[f64], y: &[f64]) -> Vec<SplineCoefficients> {
    let n = x.len();
    
    if n < 2 {
        return Vec::new();
    }
    
    // second_derivs has n elements (indices 0..n-1)
    let mut second_derivs = vec![0.0; n];
    
    // ... tridiagonal system solving ...
    // second_derivs gets values filled in
    
    // Coefficient computation loop
    let mut coeffs = Vec::with_capacity(n - 1);
    
    // ✅ CORRECT: Loop goes from 0 to n-2
    // So i+1 goes from 1 to n-1 (all valid indices)
    for i in 0..n-1 {
        let h = x[i+1] - x[i];
        let a = y[i];
        
        let b = (y[i+1] - y[i]) / h 
            - h * (2.0 * second_derivs[i] + second_derivs[i+1]) / 6.0;
        // ✓ second_derivs[i] valid: i ranges 0..n-2, so second_derivs[0..n-2] ✓
        // ✓ second_derivs[i+1] valid: i+1 ranges 1..n-1, so second_derivs[1..n-1] ✓
        
        let c = second_derivs[i] / 2.0;
        let d = (second_derivs[i+1] - second_derivs[i]) / (6.0 * h);
        
        coeffs.push(SplineCoefficients { a, b, c, d });
    }
    
    coeffs
}
```

**Step 3: Test edge cases**

```bash
# Run spline-specific tests
cargo test spline -- --nocapture

# Check for panics on:
# - n = 2 (minimal valid)
# - n = 3 (first boundary segment)
# - Large n
```

### Likely Fixes

#### Fix 1: Add bounds checking

**Before (UNSAFE):**
```rust
pub fn solve_cubic_spline(x: &[f64], y: &[f64]) -> Vec<SplineCoefficients> {
    let n = x.len();
    // Missing: Check n >= 2
    
    let mut second_derivs = vec![0.0; n];
    
    // ... solving ...
    
    let mut coeffs = Vec::new();
    for i in 0..n {  // ❌ BUG: Goes to n instead of n-1
        // When i == n-1, accessing second_derivs[i+1] is out of bounds
        let b = ... - ... * second_derivs[i+1] ...;  // ❌ PANIC
        coeffs.push(...);
    }
    
    coeffs
}
```

**After (SAFE):**
```rust
pub fn solve_cubic_spline(x: &[f64], y: &[f64]) -> Vec<SplineCoefficients> {
    let n = x.len();
    
    // ✅ Add bounds check
    if n < 2 {
        return Vec::new();  // Can't create spline with < 2 points
    }
    
    if x.len() != y.len() {
        panic!("x and y must have same length");
    }
    
    let mut second_derivs = vec![0.0; n];
    
    // ... solving ...
    
    let mut coeffs = Vec::with_capacity(n - 1);
    
    // ✅ Loop explicitly ends at n-1
    for i in 0..n-1 {
        let h = x[i+1] - x[i];
        
        // ✅ Verify h is positive
        assert!(h > 0.0, "x values must be strictly increasing");
        
        let a = y[i];
        let b = (y[i+1] - y[i]) / h 
            - h * (2.0 * second_derivs[i] + second_derivs[i+1]) / 6.0;
        let c = second_derivs[i] / 2.0;
        let d = (second_derivs[i+1] - second_derivs[i]) / (6.0 * h);
        
        coeffs.push(SplineCoefficients { a, b, c, d });
    }
    
    coeffs
}
```

#### Fix 2: Add comprehensive validation

```rust
fn validate_spline_input(x: &[f64], y: &[f64]) -> Result<(), String> {
    // Check lengths match
    if x.len() != y.len() {
        return Err(format!("Length mismatch: x={}, y={}", x.len(), y.len()));
    }
    
    // Check minimum points
    if x.len() < 2 {
        return Err(format!("Need at least 2 points, got {}", x.len()));
    }
    
    // Check x values are strictly increasing
    for i in 1..x.len() {
        if x[i] <= x[i-1] {
            return Err(format!("x values must be strictly increasing at index {}", i));
        }
    }
    
    // Check for NaN or Inf
    for (i, val) in x.iter().chain(y.iter()).enumerate() {
        if !val.is_finite() {
            return Err(format!("Non-finite value at position {}", i));
        }
    }
    
    Ok(())
}
```

#### Fix 3: Add edge case tests

```rust
#[test]
fn test_spline_minimal_valid_input() {
    let x = vec![0.0, 1.0];
    let y = vec![0.0, 1.0];
    
    let coeffs = solve_cubic_spline(&x, &y);
    assert_eq!(coeffs.len(), 1, "2-point input should produce 1 segment");
}

#[test]
fn test_spline_boundary_segments() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 1.0, 0.5, 1.0];
    
    let coeffs = solve_cubic_spline(&x, &y);
    assert_eq!(coeffs.len(), 3, "4-point input should produce 3 segments");
    
    // Verify no panic and all segments are valid
    for (i, coeff) in coeffs.iter().enumerate() {
        assert!(coeff.a.is_finite(), "Segment {} coefficient a not finite", i);
        assert!(coeff.b.is_finite(), "Segment {} coefficient b not finite", i);
        assert!(coeff.c.is_finite(), "Segment {} coefficient c not finite", i);
        assert!(coeff.d.is_finite(), "Segment {} coefficient d not finite", i);
    }
}

#[test]
fn test_spline_evaluates_correctly() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 0.0];  // Quadratic-like curve
    
    let coeffs = solve_cubic_spline(&x, &y);
    
    // Evaluate at knot points
    assert!((evaluate_spline(&coeffs, &x, 0.0) - y[0]).abs() < 1e-10);
    assert!((evaluate_spline(&coeffs, &x, 1.0) - y[1]).abs() < 1e-10);
    assert!((evaluate_spline(&coeffs, &x, 2.0) - y[2]).abs() < 1e-10);
    
    // Evaluate between knots
    let mid_val = evaluate_spline(&coeffs, &x, 0.5);
    assert!(mid_val > 0.0 && mid_val < 1.0, "Spline between points should be interpolated");
}
```

### Verification

**Test Command:**

```bash
# Run spline-specific tests
cargo test test_spline -- --nocapture

# Expected: All spline tests passing without panics

# Run full test to check cascading improvements
cargo test --lib 2>&1 | grep "test result:"

# Expected improvement:
# Before: 343 passed, 76 failed
# After: ~390+ passed, ~29 failed (50+ tests fixed from all 3 bugs)
```

**Validation Checklist:**
- [ ] No panics on valid 2-point input
- [ ] No panics on valid 3+ point input
- [ ] Spline evaluates correctly at knot points
- [ ] Spline interpolates between knots
- [ ] No out-of-bounds array access
- [ ] Envelope fitting quality improves
- [ ] EMD sifting convergence improves

### Time Tracking

| Activity | Estimated | Actual |
|----------|-----------|--------|
| Review spline algorithm | 0.5h | — |
| Implement bounds checking | 0.5-1h | — |
| Add validation functions | 0.5h | — |
| Write edge case tests | 1h | — |
| Testing & validation | 0.5-1h | — |
| **Total** | **3-4h** | — |

---

## Day 4: Regression Testing (2-3 hours)

### Purpose

Verify that all fixes work together and don't introduce new regressions.

### Test Plan

**Full Test Suite Execution:**

```bash
# Run complete test suite
cargo test --lib 2>&1

# Capture output for analysis
cargo test --lib 2>&1 | tee bug_fix_test_results.log

# Expected result:
# test result: ok. XXX passed; YY failed
# With target: >95% pass rate (>400/419 tests)
```

### Expected Results

**Before bug fixes:**
```
test result: FAILED. 343 passed; 76 failed; 4 ignored
Pass rate: 81.9%
```

**After all 3 fixes:**
```
test result: ok. 410+ passed; <9 failed; 4 ignored
Pass rate: 97.9%+
```

### Detailed Test Categories

**Category 1: V2.0-V2.2 Features (Should remain unchanged)**
```bash
# Streaming tests (V2.0)
cargo test test_streaming -- --nocapture
# Expected: 40 passing

# GPU tests (V2.1)
cargo test test_gpu -- --nocapture
cargo test test_executor -- --nocapture
# Expected: 59 passing

# LSTM tests (V2.2)
cargo test test_lstm -- --nocapture
cargo test test_boundary -- --nocapture
# Expected: 75 passing
```

**Category 2: Core algorithm fixes (Should improve)**
```bash
# Extrema detection
cargo test test_extrema -- --nocapture
cargo test test_detect -- --nocapture
# Expected: Before 2 failing → After 0 failing
# And cascading: Before 40 failing → After 0 failing

# Hilbert transform
cargo test hilbert -- --nocapture
# Expected: Before 6 failing → After 0 failing

# Spline
cargo test spline -- --nocapture
# Expected: Before ~5 failing → After 0 failing
```

**Category 3: Cascading fixes**
```bash
# EMD (should improve after extrema fix)
cargo test emd -- --nocapture
# Expected: Before 4 failing → After 0 failing

# EEMD/CEEMDAN (should improve)
cargo test eemd -- --nocapture
cargo test ceemdan -- --nocapture
# Expected: 30+ tests fixed

# FFI (should unblock after EMD fix)
cargo test ffi -- --nocapture
# Expected: Before panic → After safe
```

### Analysis

**Step 1: Count test results**

```bash
# Extract summary
grep "test result:" bug_fix_test_results.log

# Count by status
grep "ok\|FAILED" bug_fix_test_results.log | wc -l

# List failures (if any)
grep "FAILED\|failed\|error" bug_fix_test_results.log
```

**Step 2: Verify no regressions**

```bash
# Compare with baseline
diff <(cargo test --lib 2>&1) <(git stash && cargo test --lib 2>&1)

# Key check: V2.0/V2.1/V2.2 tests should NOT increase
```

**Step 3: Generate report**

```markdown
# Test Results After Bug Fixes

## Summary
- Total tests: 419
- Passed: XXX (XX.X%)
- Failed: XXX (XX.X%)
- Ignored: 4 (0.9%)

## Fixed
- ✅ Extrema detection: 2 → 0 failing
- ✅ Hilbert transform: 6 → 0 failing
- ✅ Spline indexing: 5 → 0 failing
- ✅ EMD cascading: 4 → 0 failing
- ✅ EEMD cascading: 7 → 0 failing
- ✅ CEEMDAN cascading: 7 → 0 failing
- ✅ ICEEMDAN cascading: 11 → 0 failing

## Regressions
- ✅ V2.0 Streaming: 40 → 40 passing (no change)
- ✅ V2.1 GPU: 59 → 59 passing (no change)
- ✅ V2.2 LSTM: 75 → 75 passing (no change)
```

### Decision Gate

**Can proceed to Day 5 if:**
- ✅ Test pass rate > 95% (>400/419)
- ✅ V2.0, V2.1, V2.2 still passing (174/174)
- ✅ No new test failures introduced
- ✅ All 3 critical bugs appear fixed

**If gate fails:**
- Investigate failing tests
- Identify regression cause
- Fix regression (usually small change)
- Re-run tests
- Document what went wrong

---

## Day 5: V2.3 Readiness Verification (2-3 hours)

### Purpose

Verify that V2.3 can now proceed with implementation.

### Verification Checklist

**Technical Checks:**

- [ ] All 3 critical bugs fixed (extrema, Hilbert, spline)
- [ ] Test pass rate > 95% (>400/419)
- [ ] V2.0-V2.2 features unaffected (174/174 passing)
- [ ] MEMD/NAMEMD tests unblocked
- [ ] Cascading failures resolved
- [ ] No memory safety issues in FFI
- [ ] Code compiles without errors or warnings

**Code Review Checks:**

- [ ] All fixes follow coding conventions
- [ ] No hardcoded values (use named constants)
- [ ] Edge cases handled properly
- [ ] Tests comprehensive and meaningful
- [ ] Comments explain "why" not "what"
- [ ] Performance acceptable (no regressions)

**Documentation Checks:**

- [ ] Bug fix PRs have clear descriptions
- [ ] Root cause documented in commit messages
- [ ] Fix strategy explained in code comments
- [ ] Tests serve as documentation

### Pre-V2.3 Architecture Review

**Review V2.3 design against fixes:**

1. **MEMD depends on:** Extrema detection, spline fitting, EMD base algorithm
   - ✅ All fixed, MEMD can proceed

2. **NAMEMD depends on:** Extrema detection, EMD, noise addition
   - ✅ All fixed, NAMEMD can proceed

3. **GPU V2.1 Phase 2 (kernels) depends on:** GPU framework (already done)
   - ✅ Framework complete, kernels can proceed in parallel

4. **Boundary prediction (V2.2 integration) depends on:** None of the bugs
   - ✅ Already working, can be used in V2.3

### Generate V2.3 Readiness Report

```markdown
# V2.3 Readiness Report

**Date:** 2026-04-15 (After bug fixes)  
**Status:** READY TO PROCEED

## Critical Bugs Fixed
- ✅ Extrema detection (3-5h actual)
- ✅ Hilbert phase unwrapping (7-9h actual)
- ✅ Spline indexing bounds (3-4h actual)

## Test Results
- Pass rate: 410/419 (97.9%)
- Improvement: +67 tests fixed
- V2.0-V2.2: 174/174 (100%)

## Unblocked Features
- ✅ MEMD base algorithm
- ✅ NAMEMD variant
- ✅ 2D/3D extrema detection
- ✅ Multivariate streaming integration
- ✅ GPU kernel implementation (parallel track)

## V2.3 Implementation Can Begin
- Estimated effort: 4-5 weeks
- Tasks: 30-35 implementation tasks
- Team size: 1-2 engineers
- Parallel GPU kernel work: 1 engineer

## Risks Mitigated
- ❌ → ✅ Extrema detection broken (FIXED)
- ❌ → ✅ Hilbert-Huang analysis broken (FIXED)
- ❌ → ✅ Spline quality issues (FIXED)

## Dependencies Satisfied
- ✅ V2.0 streaming foundation
- ✅ V2.1 GPU framework
- ✅ V2.2 neural boundaries
- ✅ Core algorithm stability

## Recommendation
**PROCEED WITH V2.3 IMPLEMENTATION**

Next steps:
1. Create V2.3 feature branch
2. Decompose V2.3 task list
3. Begin MEMD architecture design
4. Start GPU kernel parallel work
5. Plan integration testing
```

### Create V2.3 Feature Branch

```bash
# Ensure main is up to date and clean
git checkout main
git pull origin main

# Create V2.3 feature branch
git checkout -b feat/v2-3-memd-namemd-multivariate

# Verify branch
git branch --show-current
# Output: feat/v2-3-memd-namemd-multivariate
```

### Time Tracking

| Activity | Estimated | Actual |
|----------|-----------|--------|
| Architecture review | 0.5h | — |
| V2.3 readiness checklist | 0.5h | — |
| Test validation | 0.5h | — |
| V2.3 branch setup | 0.5h | — |
| Report generation | 0.5h | — |
| **Total** | **2-3h** | — |

---

## Summary: All Days Combined

### Timeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPLETE BUG FIX WEEK TIMELINE                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Day 1 (Mon-Tue):  Extrema Detection Fix          3-5 hours      │
│  ├─ Investigation & root cause                    1-2h           │
│  ├─ Implementation                                1-2h           │
│  └─ Testing & validation                          0.5-1h        │
│                                                                   │
│  Day 2 (Wed-Thu):  Hilbert Phase Unwrapping       7-9 hours      │
│  ├─ Phase unwrapping math review                  1-2h           │
│  ├─ Algorithm analysis                            1h             │
│  ├─ Implementation                                3-4h           │
│  └─ Comprehensive testing                         1-2h           │
│                                                                   │
│  Day 3 (Fri-Sat):  Spline Indexing Fix            3-4 hours      │
│  ├─ Code review & understanding                   0.5-1h        │
│  ├─ Implementation                                1h             │
│  ├─ Validation & edge case tests                  1-2h           │
│  └─ Integration testing                           0.5h           │
│                                                                   │
│  Day 4 (Sun-Mon):  Regression Testing             2-3 hours      │
│  ├─ Full test suite run                           0.5h           │
│  ├─ Analysis & comparison                         1h             │
│  └─ Report generation                             0.5-1h        │
│                                                                   │
│  Day 5 (Tue):      V2.3 Readiness                 2-3 hours      │
│  ├─ Architecture review                           0.5h           │
│  ├─ Readiness checklist                           0.5h           │
│  ├─ V2.3 branch setup                             0.5h           │
│  └─ Documentation                                 0.5h           │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│  TOTAL:                                           15-24 hours     │
│  Typical pace (40h/week):                        2-3 days        │
│  Realistic pace (managing other work):           5-7 days        │
└──────────────────────────────────────────────────────────────────┘
```

### Success Metrics

**After all 5 days:**
```
✅ Test pass rate:          343/419 (81.9%) → 410+/419 (97.9%+)
✅ Critical bugs fixed:     3/3 bugs fixed
✅ V2.0-V2.2 unchanged:     174/174 tests (100%)
✅ V2.3 unblocked:          All dependencies resolved
✅ Code quality:            Reviewed, tested, documented
✅ Regression risk:         Minimal (<1%)
```

### Go/No-Go Decision Gate

**Can merge to main if:**
- ✅ All 3 bugs fixed (code review approved)
- ✅ >95% test pass rate (410+/419)
- ✅ No new regressions (V2.0-V2.2 unchanged)
- ✅ All changes documented
- ✅ Team approval

**Merge command (once approved):**
```bash
# Ensure clean working directory
git status  # Should show "nothing to commit"

# Create final summary commit
git log --oneline -5  # Show last 5 commits

# Merge to main
git checkout main
git pull origin main
git merge bugfix/v2x-critical-issues

# Verify merge
git log --oneline -5
# Should show all bug fix commits
```

---

## Appendix: Troubleshooting

### Issue: Test still failing after fix

**Diagnosis:**
```bash
# Run specific test with output
cargo test test_name -- --nocapture --test-threads=1

# Add debug logging
RUST_LOG=debug cargo test test_name -- --nocapture

# Check what values are actually being computed
```

**Common causes:**
1. Fix not complete (edge case missed)
2. Related bug elsewhere affecting same code
3. Test expectations wrong (validate with math)
4. Numerical precision issues (loosen tolerance slightly)

### Issue: New test failures after fix

**Diagnosis:**
```bash
# Find new failures
cargo test --lib 2>&1 | grep FAILED

# Bisect to find which fix caused it
git stash
cargo test --lib 2>&1 | grep "test result:" # Get baseline
git stash pop
# Apply fixes one by one
```

**Common causes:**
1. Typo in fix code
2. Changed function signature (cascading impact)
3. Off-by-one or array indexing still wrong
4. Numerical instability from change

### Issue: Can't compile after fix

**Diagnosis:**
```bash
# See full error
cargo build 2>&1 | head -30

# Check syntax
cargo check

# Look for:
# - Mismatched braces/parentheses
# - Type mismatches
# - Missing imports
```

**Resolution:**
1. Review exact error message
2. Find line in editor
3. Compare with working code nearby
4. Fix syntax issue

---

**Document Version:** 1.0  
**Last Updated:** 2026-04-08 16:45 UTC  
**Next Review:** After bug fix execution  
**Status:** Ready for use
