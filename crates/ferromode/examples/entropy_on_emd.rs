//! Entropy Metrics Analysis on EMD Decomposition
//!
//! This example demonstrates how to:
//! 1. Perform EMD decomposition on a composite signal
//! 2. Analyze entropy of each IMF
//! 3. Interpret entropy results for quality assessment
//! 4. Detect potential mode mixing issues
//!
//! Run with: cargo run --example entropy_on_emd --release

use ferromode::algorithms::emd::{emd, EmdConfig};
use ferromode::analysis::EntropyAnalysis;
use std::f64::consts::PI;

/// Generate a composite test signal for decomposition
///
/// Composed of three sinusoidal modes plus trend:
/// - 3 Hz mode (5 cycles)
/// - 7 Hz mode (10 cycles)
/// - 15 Hz mode (20 cycles)
/// - Linear trend
/// - Low-level noise
fn generate_composite_signal(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| {
            let t = i as f64 / 100.0; // 100 Hz effective sampling

            // Mode 1: 3 Hz sinusoid
            let mode1 = (2.0 * PI * 3.0 * t).sin();

            // Mode 2: 7 Hz sinusoid
            let mode2 = 0.5 * (2.0 * PI * 7.0 * t).sin();

            // Mode 3: 15 Hz sinusoid
            let mode3 = 0.25 * (2.0 * PI * 15.0 * t).sin();

            // Linear trend
            let trend = 0.5 * t;

            // Low-level noise
            let noise = 0.05 * (rand::random::<f64>() - 0.5);

            mode1 + mode2 + mode3 + trend + noise
        })
        .collect()
}

/// Print header with formatting
fn print_header(title: &str, width: usize) {
    println!("\n{}", "=".repeat(width));
    println!("{:^width$}", title, width = width);
    println!("{}", "=".repeat(width));
}

/// Print section separator
fn print_separator(width: usize) {
    println!("{}", "-".repeat(width));
}

/// Print formatted entropy metrics table
fn print_entropy_table(analysis: &EntropyAnalysis) {
    let width = 70;

    println!("\n{:<6} {:>14} {:>14} {:>14}", "IMF", "Spectral", "Permutation", "Sample");
    print_separator(width);

    for i in 0..analysis.spectral_entropy.len() {
        println!(
            "{:<6} {:>14.4} {:>14.4} {:>14.4}",
            i,
            analysis.spectral_entropy[i],
            analysis.permutation_entropy[i],
            analysis.sample_entropy[i]
        );
    }

    print_separator(width);
}

/// Interpret entropy results and provide recommendations
fn interpret_entropy_analysis(analysis: &EntropyAnalysis, _signal_length: usize) {
    println!("\nQUALITY ASSESSMENT");
    println!("{}", "-".repeat(70));

    // Overall complexity score assessment
    println!("\nComplexity Score: {:.4}", analysis.complexity_score);

    if analysis.complexity_score < 0.4 {
        println!("✓ Excellent decomposition quality");
        println!("  - Clear mode separation");
        println!("  - Low mode mixing");
        println!("  - Reliable IMF components");
    } else if analysis.complexity_score < 0.6 {
        println!("✓ Good decomposition quality");
        println!("  - Acceptable mode separation");
        println!("  - Minor mode mixing acceptable");
        println!("  - Components generally reliable");
    } else if analysis.complexity_score < 0.75 {
        println!("⚠ Fair decomposition quality");
        println!("  - Some mode mixing detected");
        println!("  - Consider validation with other methods");
        println!("  - May need parameter adjustment");
    } else {
        println!("✗ Poor decomposition quality");
        println!("  - Significant mode mixing indicated");
        println!("  - Recommend increasing max_imf");
        println!("  - Consider alternative stopping criteria");
    }

    // Entropy metric interpretation
    println!("\nSPECTRAL ENTROPY (Frequency disorder)");
    println!("Mean: {:.4}", analysis.mean_spectral_entropy);
    if analysis.mean_spectral_entropy < 0.4 {
        println!("✓ Signal has narrow-band characteristics");
    } else if analysis.mean_spectral_entropy > 0.7 {
        println!("! Signal has broad-band/noisy characteristics");
    }

    println!("\nPERMUTATION ENTROPY (Temporal complexity)");
    println!("Mean: {:.4}", analysis.mean_permutation_entropy);
    if analysis.mean_permutation_entropy < 0.3 {
        println!("✓ Signal has regular/periodic patterns");
    } else if analysis.mean_permutation_entropy > 0.7 {
        println!("! Signal has chaotic/random patterns");
    }

    println!("\nSAMPLE ENTROPY (Self-similarity)");
    println!("Mean: {:.4}", analysis.mean_sample_entropy);
    if analysis.mean_sample_entropy < 0.5 {
        println!("✓ Signal is highly regular/predictable");
    } else if analysis.mean_sample_entropy > 1.5 {
        println!("! Signal is highly complex/random");
    }

    // Mode mixing detection
    if !analysis.high_complexity_imfs.is_empty() {
        println!("\n⚠️ WARNING: HIGH-COMPLEXITY IMFs DETECTED");
        println!("{}", "-".repeat(70));
        println!("Affected IMF indices: {:?}", analysis.high_complexity_imfs);
        println!("\nPossible causes:");
        println!("  1. Mode mixing: Multiple modes in same IMF");
        println!("  2. Noise contamination: High-frequency noise not separated");
        println!("  3. Signal non-stationarity: Decomposition boundary effects");
        println!("\nRecommendations:");
        println!("  - Review decomposition parameters");
        println!("  - Increase max_imf for finer decomposition");
        println!("  - Check signal preprocessing");
        println!("  - Consider alternative stopping criteria");
    } else {
        println!("\n✓ No high-complexity IMFs detected");
        println!("  Mode separation appears successful");
    }

    // Entropy trend analysis
    println!("\nENTROPY TRENDS");
    println!("{}", "-".repeat(70));

    if analysis.spectral_entropy.len() > 1 {
        let first_spectral = analysis.spectral_entropy[0];
        let last_spectral = analysis.spectral_entropy[analysis.spectral_entropy.len() - 1];

        if last_spectral < first_spectral {
            println!("✓ Spectral entropy decreases across IMFs");
            println!("  (Expected: higher frequencies → lower frequencies)");
        }
    }
}

/// Analyze signal preprocessing impact
#[allow(dead_code)]
fn analyze_preprocessing_impact(original: &[f64], processed: &[f64]) {
    use ferromode::analysis::spectral_entropy_normalized;

    match (spectral_entropy_normalized(original), spectral_entropy_normalized(processed)) {
        (Ok(entropy_orig), Ok(entropy_proc)) => {
            let change = (entropy_proc - entropy_orig).abs();

            println!("\nPREPROCESSING VALIDATION");
            println!("{}", "-".repeat(70));
            println!("Entropy before: {:.4}", entropy_orig);
            println!("Entropy after:  {:.4}", entropy_proc);
            println!("Absolute change: {:.4}", change);

            if change > 0.3 {
                println!("⚠️ Preprocessing significantly altered signal characteristics");
            } else {
                println!("✓ Preprocessing had minimal impact on frequency content");
            }
        }
        _ => println!("⚠️ Could not compute preprocessing analysis"),
    }
}

/// Main analysis workflow
fn main() -> Result<(), Box<dyn std::error::Error>> {
    print_header("ENTROPY METRICS ANALYSIS ON EMD DECOMPOSITION", 70);

    // Generate test signal
    println!("\n1. SIGNAL GENERATION");
    println!("{}", "-".repeat(70));

    let signal = generate_composite_signal(1000);
    println!("Generated composite signal:");
    println!("  Length: {} samples", signal.len());
    println!("  Composition:");
    println!("    - 3 Hz sinusoid (dominant)");
    println!("    - 7 Hz sinusoid (medium)");
    println!("    - 15 Hz sinusoid (fine)");
    println!("    - Linear trend");
    println!("    - 5% white noise");

    // Perform decomposition
    println!("\n2. EMD DECOMPOSITION");
    println!("{}", "-".repeat(70));

    let config = EmdConfig::default();
    println!("Configuration:");
    println!("  Max IMF: {}", config.max_imfs);
    println!("  Boundary condition: {:?}", config.boundary_condition);

    println!("\nDecomposing signal...");
    let decomposition = emd(&signal, &config)?;

    println!("✓ Decomposition complete");
    println!("  Number of IMFs: {}", decomposition.imfs.imfs.len());
    println!("  Siftings: {}", decomposition.n_siftings);
    println!("  Decomposition time: {:.2}ms", decomposition.elapsed.as_secs_f64() * 1000.0);

    // Analyze entropy
    println!("\n3. ENTROPY ANALYSIS");
    println!("{}", "-".repeat(70));

    println!("Computing entropy metrics for all IMFs...");
    let analysis = EntropyAnalysis::from_decomposition(&decomposition)?;
    println!("✓ Entropy analysis complete\n");

    // Print detailed entropy table
    print_entropy_table(&analysis);

    // Print summary statistics
    println!("\nSUMMARY STATISTICS");
    println!("{}", "-".repeat(70));
    println!("Mean Spectral Entropy:    {:.4}", analysis.mean_spectral_entropy);
    println!("Mean Permutation Entropy: {:.4}", analysis.mean_permutation_entropy);
    println!("Mean Sample Entropy:      {:.4}", analysis.mean_sample_entropy);
    println!("\nOverall Complexity Score: {:.4}", analysis.complexity_score);

    // Interpret results
    interpret_entropy_analysis(&analysis, signal.len());

    // Per-IMF insights
    println!("\nPER-IMF INSIGHTS");
    println!("{}", "-".repeat(70));

    for i in 0..analysis.spectral_entropy.len() {
        let spec_ent = analysis.spectral_entropy[i];
        let perm_ent = analysis.permutation_entropy[i];
        let samp_ent = analysis.sample_entropy[i];

        print!("IMF {}: ", i);

        // Characterize based on entropy
        let mut characteristics = Vec::new();

        if spec_ent < 0.3 {
            characteristics.push("narrowband");
        } else if spec_ent > 0.7 {
            characteristics.push("broadband");
        }

        if perm_ent < 0.3 {
            characteristics.push("regular");
        } else if perm_ent > 0.7 {
            characteristics.push("chaotic");
        }

        if samp_ent < 0.5 {
            characteristics.push("predictable");
        } else if samp_ent > 1.5 {
            characteristics.push("complex");
        }

        if characteristics.is_empty() {
            println!("balanced characteristics");
        } else {
            println!("{}", characteristics.join(", "));
        }
    }

    // Recommendations
    println!("\n4. RECOMMENDATIONS");
    println!("{}", "-".repeat(70));

    if analysis.complexity_score < 0.5 {
        println!("✓ Decomposition is reliable for further analysis");
        println!("  - Proceed with confidence in IMF interpretation");
        println!("  - Use IMFs for feature extraction or signal processing");
    } else if analysis.complexity_score < 0.7 {
        println!("✓ Decomposition acceptable, but validate with caution");
        println!("  - Cross-check results with domain knowledge");
        println!("  - Consider sensitivity analysis");
        println!("  - May benefit from parameter tuning");
    } else {
        println!("⚠️ Re-evaluate decomposition approach");
        println!("  - Try different stopping criteria");
        println!("  - Preprocess signal (denoise, detrend)");
        println!("  - Consider alternative decomposition methods");
        println!("  - Consult domain expert for signal interpretation");
    }

    print_header("ANALYSIS COMPLETE", 70);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_signal_generation() {
        let signal = generate_composite_signal(100);
        assert_eq!(signal.len(), 100);

        // Check signal is not all zeros or NaN
        assert!(signal.iter().all(|&x| x.is_finite()));
        assert!(!signal.iter().all(|&x| (x - 0.0).abs() < 1e-10));
    }

    #[test]
    fn test_entropy_analysis_structure() {
        // Verify EntropyAnalysis can be created and used
        let signal = generate_composite_signal(500);

        match emd(&signal, &EmdConfig::default()) {
            Ok(decomposition) => {
                match EntropyAnalysis::from_decomposition(&decomposition) {
                    Ok(analysis) => {
                        // Verify structure
                        assert!(!analysis.spectral_entropy.is_empty());
                        assert_eq!(
                            analysis.spectral_entropy.len(),
                            analysis.permutation_entropy.len()
                        );
                        assert_eq!(analysis.spectral_entropy.len(), analysis.sample_entropy.len());

                        // Verify values are in reasonable ranges
                        assert!(analysis.mean_spectral_entropy >= 0.0);
                        assert!(analysis.mean_spectral_entropy <= 1.0);
                        assert!(analysis.complexity_score >= 0.0);
                        assert!(analysis.complexity_score <= 1.0);
                    }
                    Err(e) => panic!("Failed to create entropy analysis: {}", e),
                }
            }
            Err(e) => panic!("Failed to decompose signal: {}", e),
        }
    }
}
