/// Mode Mixing Analysis Example
///
/// Demonstrates mode mixing detection and analysis using Ferromode
use ferromode::algorithms::emd::emd;
use ferromode::mode_mixing::{analyze_mode_mixing, compute_mode_mixing_overlap, ModeMixingHeatmap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a multi-component test signal: 1 Hz + 5 Hz + 15 Hz
    let n_samples = 1000;
    let sample_rate = 100.0; // Hz
    let mut signal = vec![0.0; n_samples];

    for i in 0..n_samples {
        let t = i as f64 / sample_rate;
        signal[i] = 1.0 * (2.0 * std::f64::consts::PI * 1.0 * t).sin()
            + 0.6 * (2.0 * std::f64::consts::PI * 5.0 * t).sin()
            + 0.4 * (2.0 * std::f64::consts::PI * 15.0 * t).sin();
    }

    // Decompose signal
    let config = Default::default();
    let decomposition = emd(&signal, &config)?;

    println!("Decomposition: {} IMFs computed", decomposition.imfs.imfs.len());

    // Analyze mode mixing
    let analysis = compute_mode_mixing_overlap(&decomposition.imfs.imfs)?;
    let heatmap = ModeMixingHeatmap::new(analysis);
    let interpretation = analyze_mode_mixing(&decomposition.imfs.imfs)?;

    // Print results
    println!("\n{}", heatmap.generate_severity_report());
    println!("\nInterpretation: {}", interpretation.interpretation);

    if !interpretation.remediation.is_empty() {
        println!("\nRecommended Actions:");
        for (i, remedy) in interpretation.remediation.iter().enumerate() {
            println!("  {}. {}", i + 1, remedy);
        }
    }

    Ok(())
}
