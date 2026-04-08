//! Adaptive algorithm selection based on signal metrics.
//!
//! Demonstrates how streaming decomposer automatically adapts to signal
//! characteristics through intermittency metrics.

use ferromode::adapters::streaming::{ArModel, StreamingDecomposer};
use ferromode::algorithms::emd::EmdConfig;
use ferromode::types::Signal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming Decomposition: Adaptive Algorithm Selection ===\n");

    // Create decomposer
    let config = EmdConfig::default();
    let predictor = Box::new(ArModel::new(3)?);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 4096)?;

    println!("Testing adaptive algorithm selection based on stationarity:\n");

    // Test 1: Stationary signal (pure sine)
    println!("--- Test 1: Stationary Signal (Pure Sine) ---");
    test_signal(&mut decomposer, "pure_sine", |i| (i as f64 * 0.02).sin())?;

    // Test 2: Mildly non-stationary (sine + noise)
    println!("\n--- Test 2: Mildly Non-Stationary (Sine + Light Noise) ---");
    use rand::Rng;
    test_signal(&mut decomposer, "sine_noise_light", {
        let mut rng = rand::thread_rng();
        move |i: usize| (i as f64 * 0.02).sin() + rng.gen_range(-0.1..0.1)
    })?;

    // Test 3: Highly non-stationary (random noise)
    println!("\n--- Test 3: Highly Non-Stationary (Random Noise) ---");
    test_signal(&mut decomposer, "noise_heavy", {
        let mut rng = rand::thread_rng();
        move |_i: usize| rng.gen_range(-1.0..1.0)
    })?;

    // Test 4: Composite signal (multiple frequencies)
    println!("\n--- Test 4: Composite Signal (Multiple Frequencies) ---");
    test_signal(&mut decomposer, "composite", |i| {
        (0.5 * (i as f64 * 0.02).sin())
            + (0.3 * (i as f64 * 0.05).cos())
            + (0.2 * (i as f64 * 0.1).sin())
    })?;

    println!("\n✓ Adaptive algorithm selection demonstration complete!");
    println!("\nAlgorithm Selection Rules:");
    println!("  Stationarity > 0.8  → EMD (minimal oscillations)");
    println!("  Stationarity 0.5-0.8 → EEMD (moderate noise)");
    println!("  Stationarity < 0.5  → CEEMDAN (high noise/complexity)");

    Ok(())
}

fn test_signal<F>(
    decomposer: &mut StreamingDecomposer,
    name: &str,
    mut signal_fn: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(usize) -> f64,
{
    decomposer.reset();

    let chunk_size = 512;
    let num_chunks = 10;
    let mut chunk = vec![0.0; chunk_size];

    let mut entropies = Vec::new();
    let mut stationarity_scores = Vec::new();

    for chunk_idx in 0..num_chunks {
        // Generate signal chunk
        for i in 0..chunk_size {
            let sample_idx = chunk_idx * chunk_size + i;
            chunk[i] = signal_fn(sample_idx);
        }

        // Convert to Signal and decompose
        let signal = Signal::from_slice(&chunk)?;
        let result = decomposer.decompose_chunk(&signal)?;

        entropies.push(result.metrics.spectral_entropy);
        stationarity_scores.push(result.metrics.stationarity_score);

        println!(
            "  Chunk {}: entropy={:.3}, stationarity={:.3}, imfs={}",
            chunk_idx,
            result.metrics.spectral_entropy,
            result.metrics.stationarity_score,
            result.imfs.len()
        );
    }

    // Compute statistics
    let mean_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let mean_stationarity =
        stationarity_scores.iter().sum::<f64>() / stationarity_scores.len() as f64;

    println!("\nStatistics for {}:", name);
    println!("  Mean entropy: {:.3}", mean_entropy);
    println!("  Mean stationarity: {:.3}", mean_stationarity);

    let recommended_algo = if mean_stationarity > 0.8 {
        "EMD"
    } else if mean_stationarity > 0.5 {
        "EEMD"
    } else {
        "CEEMDAN"
    };

    println!("  Recommended algorithm: {}", recommended_algo);

    Ok(())
}
