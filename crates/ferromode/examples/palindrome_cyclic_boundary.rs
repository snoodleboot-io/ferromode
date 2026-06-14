//! Palindrome-Cyclic Boundary Condition
//!
//! This example demonstrates the `PalindromeCyclic` boundary condition and
//! compares it against the default `MirrorEven` strategy to show why it
//! produces more stable end-effect suppression.
//!
//! ## Background
//!
//! EMD is sensitive to boundary conditions because cubic spline envelope
//! interpolation must extrapolate beyond the first and last extrema.
//! `MirrorEven` adds a short reflected window at each end. `PalindromeCyclic`
//! instead pre-extends the whole signal into a `2N-1` palindrome before
//! the sifting loop begins, making the signal naturally even-periodic,
//! then uses a cyclic (periodic) cubic spline throughout sifting.
//!
//! The palindrome approach is typically more stable when the signal has
//! non-trivial values near its endpoints or contains a trend.
//!
//! ## What this example shows
//!
//! 1. Reconstruct signal identically with both strategies
//! 2. Measure boundary residue in the first and last 10 samples
//! 3. Print a side-by-side comparison of end-effect magnitude
//!
//! Run with: `cargo run --example palindrome_cyclic_boundary --release`

use ferromode::algorithms::emd::{emd, EmdConfig};
use ferromode::boundary::BoundaryConditionType;
use ferromode::sifting::SiftingConfig;
use ferromode::spline::SplineType;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Signal generators
// ---------------------------------------------------------------------------

/// Composite signal with a nonzero slope at both endpoints.
///
/// The strong low-frequency trend means the first and last samples are far
/// from zero, which is where mirror-based strategies struggle most.
fn generate_boundary_stress_signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64; // t in [0, 1)
            let fast = (2.0 * PI * 8.0 * t).sin();
            let medium = 0.6 * (2.0 * PI * 3.0 * t).sin();
            let trend = 1.5 * t; // rising DC trend — hard for boundary strategies
            fast + medium + trend
        })
        .collect()
}

/// Pure sine wave — a best-case signal for boundary strategies.
fn generate_sine(n: usize) -> Vec<f64> {
    (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect()
}

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

/// Root-mean-square of the first `k` and last `k` samples of an IMF.
///
/// This measures how much "end energy" leaked into an IMF — a proxy for
/// end-effect contamination. Lower is better near boundaries.
fn boundary_rms(imf: &[f64], k: usize) -> (f64, f64) {
    let k = k.min(imf.len() / 4); // don't let k exceed a quarter of the signal
    let left: f64 = imf[..k].iter().map(|v| v * v).sum::<f64>() / k as f64;
    let right: f64 = imf[imf.len() - k..].iter().map(|v| v * v).sum::<f64>() / k as f64;
    (left.sqrt(), right.sqrt())
}

/// Maximum absolute reconstruction error.
fn max_reconstruction_error(signal: &[f64], result: &ferromode::types::DecompositionResult) -> f64 {
    let reconstructed = result.imfs.reconstruct();
    signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f64, f64::max)
}

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------

fn mirror_even_config() -> EmdConfig {
    EmdConfig::default() // MirrorEven + Natural spline is the default
}

fn palindrome_cyclic_config() -> EmdConfig {
    EmdConfig {
        sifting_config: SiftingConfig {
            boundary_condition: BoundaryConditionType::PalindromeCyclic,
            ..SiftingConfig::default()
        },
        validate_reconstruction: true,
        reconstruction_tolerance: 1e-10,
        ..EmdConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn print_header(title: &str) {
    let w = 72;
    println!("\n{}", "=".repeat(w));
    println!("{:^width$}", title, width = w);
    println!("{}", "=".repeat(w));
}

fn print_section(title: &str) {
    println!("\n{}", title);
    println!("{}", "-".repeat(60));
}

fn print_boundary_comparison(
    label: &str,
    mirror_result: &ferromode::types::DecompositionResult,
    palindrome_result: &ferromode::types::DecompositionResult,
    signal: &[f64],
    boundary_samples: usize,
) {
    println!("\n  {:46}  {:>10}  {:>10}", "", "MirrorEven", "Palindrome");
    println!("  {}", "-".repeat(68));

    let n_imfs = mirror_result.imfs.n_imfs().min(palindrome_result.imfs.n_imfs()).min(4);
    for i in 0..n_imfs {
        let (ml, mr) = boundary_rms(&mirror_result.imfs.imfs[i], boundary_samples);
        let (pl, pr) = boundary_rms(&palindrome_result.imfs.imfs[i], boundary_samples);

        let improvement_l = if ml > 0.0 { (ml - pl) / ml * 100.0 } else { 0.0 };
        let improvement_r = if mr > 0.0 { (mr - pr) / mr * 100.0 } else { 0.0 };

        println!(
            "  IMF {:2}  left  RMS:  MirrorEven={:.5}  Palindrome={:.5}  ({:+.1}%)",
            i, ml, pl, -improvement_l
        );
        println!(
            "  IMF {:2}  right RMS:  MirrorEven={:.5}  Palindrome={:.5}  ({:+.1}%)",
            i, mr, pr, -improvement_r
        );
    }

    println!(
        "\n  Reconstruction error   MirrorEven={:.2e}  Palindrome={:.2e}",
        max_reconstruction_error(signal, mirror_result),
        max_reconstruction_error(signal, palindrome_result),
    );
    let _ = label;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    print_header("PALINDROME-CYCLIC BOUNDARY CONDITION DEMO");

    // ------------------------------------------------------------------
    // Section 1: basic usage
    // ------------------------------------------------------------------
    print_section("1. BASIC USAGE");

    let n = 200;
    let signal = generate_sine(n);

    println!("  Signal: {} samples, pure sine", n);

    let result = emd(&signal, &palindrome_cyclic_config())?;

    println!("  IMFs extracted: {}", result.imfs.n_imfs());
    println!(
        "  Elapsed: {:.2} ms",
        result.elapsed.as_secs_f64() * 1000.0
    );

    // Verify reconstruction
    let max_err = max_reconstruction_error(&signal, &result);
    println!("  Reconstruction error: {:.2e} (should be < 1e-10)", max_err);
    assert!(max_err < 1e-10, "reconstruction failed: {:.2e}", max_err);
    println!("  Reconstruction: OK");

    // ------------------------------------------------------------------
    // Section 2: boundary stress test
    // ------------------------------------------------------------------
    print_section("2. BOUNDARY STRESS TEST (signal with trend + fast oscillation)");

    let n = 300;
    let stress = generate_boundary_stress_signal(n);

    println!("  Signal: {} samples", n);
    println!("  Components: 8 Hz sine + 3 Hz sine + rising trend");
    println!("  Both endpoints are far from zero (hard boundary case)");

    let mirror_result = emd(&stress, &mirror_even_config())?;
    let palindrome_result = emd(&stress, &palindrome_cyclic_config())?;

    println!(
        "\n  MirrorEven:    {} IMFs, {:.2} ms",
        mirror_result.imfs.n_imfs(),
        mirror_result.elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "  PalindromeCyclic: {} IMFs, {:.2} ms",
        palindrome_result.imfs.n_imfs(),
        palindrome_result.elapsed.as_secs_f64() * 1000.0
    );

    print_boundary_comparison("stress", &mirror_result, &palindrome_result, &stress, 10);

    // ------------------------------------------------------------------
    // Section 3: max_imfs limit — still returns N-length output
    // ------------------------------------------------------------------
    print_section("3. MAX IMFS LIMIT");

    let limited_config = EmdConfig {
        sifting_config: SiftingConfig {
            boundary_condition: BoundaryConditionType::PalindromeCyclic,
            ..SiftingConfig::default()
        },
        max_imfs: 2,
        validate_reconstruction: false, // residue not fully extracted, so don't validate
        ..EmdConfig::default()
    };
    let limited = emd(&stress, &limited_config)?;

    println!("  max_imfs=2 → {} IMFs extracted", limited.imfs.n_imfs());
    for (i, imf) in limited.imfs.imfs.iter().enumerate() {
        println!("  IMF {:2}: {} samples (should equal input length {})", i, imf.len(), n);
        assert_eq!(imf.len(), n, "IMF length mismatch");
    }
    assert_eq!(limited.imfs.residue.len(), n, "residue length mismatch");
    println!("  All output lengths correct: OK");

    // ------------------------------------------------------------------
    // Section 4: SplineType::Periodic used directly (advanced)
    // ------------------------------------------------------------------
    print_section("4. SPLINE TYPE CONTROL (advanced)");

    println!("  SplineType can also be set independently of boundary strategy.");
    println!("  PalindromeCyclic automatically selects Periodic spline,");
    println!("  but you can combine any strategy with any spline type:");

    let periodic_spline_config = EmdConfig {
        sifting_config: SiftingConfig {
            boundary_condition: BoundaryConditionType::Periodic, // tile the signal
            spline_type: SplineType::Periodic,                   // cyclic spline
            ..SiftingConfig::default()
        },
        validate_reconstruction: false,
        ..EmdConfig::default()
    };
    let periodic_result = emd(&signal, &periodic_spline_config)?;
    println!(
        "  Periodic tiling + Periodic spline: {} IMFs",
        periodic_result.imfs.n_imfs()
    );

    let not_a_knot_config = EmdConfig {
        sifting_config: SiftingConfig {
            boundary_condition: BoundaryConditionType::MirrorEven,
            spline_type: SplineType::NotAKnot,
            ..SiftingConfig::default()
        },
        validate_reconstruction: false,
        ..EmdConfig::default()
    };
    let nak_result = emd(&signal, &not_a_knot_config)?;
    println!(
        "  MirrorEven + NotAKnot spline:       {} IMFs",
        nak_result.imfs.n_imfs()
    );

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    print_header("WHEN TO USE PALINDROME CYCLIC");

    println!(
        "
  USE PalindromeCyclic when:
    - Signal has non-trivial values or slope at its endpoints
    - Signal contains a low-frequency trend
    - You observe spurious oscillations at the start/end of IMF 1
    - You need the most stable boundary behaviour available

  PalindromeCyclic vs. alternatives:
    - MirrorEven (default):  fast, good for periodic or zero-mean signals
    - Periodic:              only appropriate for truly periodic inputs
    - ARModel:               good for stochastic signals with known AR order
    - PalindromeCyclic:      best general-purpose stability; ~2x slower
                             (operates on 2N-1 samples internally)

  Note: reconstruction is always exact regardless of boundary strategy.
    The difference is visible only in the boundary region of each IMF.
"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Embedded tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palindrome_cyclic_reconstructs_exactly() {
        let n = 150;
        let signal = generate_boundary_stress_signal(n);
        let result = emd(&signal, &palindrome_cyclic_config()).unwrap();
        let err = max_reconstruction_error(&signal, &result);
        assert!(err < 1e-10, "reconstruction error {:.2e}", err);
    }

    #[test]
    fn test_palindrome_cyclic_imf_lengths_match_input() {
        let n = 100;
        let signal = generate_sine(n);
        let result = emd(&signal, &palindrome_cyclic_config()).unwrap();
        for imf in &result.imfs.imfs {
            assert_eq!(imf.len(), n);
        }
        assert_eq!(result.imfs.residue.len(), n);
    }

    #[test]
    fn test_spline_type_natural_backward_compat() {
        let n = 100;
        let signal = generate_sine(n);

        // Default config must behave identically to before the SplineType field was added
        let default_result = emd(&signal, &EmdConfig::default()).unwrap();
        assert!(default_result.imfs.n_imfs() >= 1);
        let err = max_reconstruction_error(&signal, &default_result);
        assert!(err < 1e-10, "default config reconstruction error {:.2e}", err);
    }
}
