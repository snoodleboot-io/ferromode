//! Real-time streaming example with state persistence.
//!
//! Simulates continuous signal input and shows how state is maintained
//! across chunks for improved boundary handling.

use ferromode::adapters::streaming::{ArModel, StreamingDecomposer};
use ferromode::algorithms::emd::EmdConfig;
use ferromode::types::Signal;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming Decomposition: Real-Time Example ===\n");

    // Create decomposer configured for real-time operation
    let config = EmdConfig::default();
    let predictor = Box::new(ArModel::new(3)?);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 4096)?;

    println!("Processing continuous signal stream...");
    println!("Chunk size: 512 samples, Processing 20 chunks\n");

    let chunk_size = 512;
    let num_chunks = 20;
    let mut chunk = vec![0.0; chunk_size];

    // Simulate continuous signal (frequency increases over time)
    let start_time = Instant::now();
    let mut total_samples_processed = 0;

    for chunk_idx in 0..num_chunks {
        // Generate signal chunk with time-varying frequency
        for i in 0..chunk_size {
            let sample_idx = chunk_idx * chunk_size + i;
            let t = sample_idx as f64 * 0.001;

            // Frequency increases linearly (chirp signal)
            let freq = 0.01 + (sample_idx as f64) * 0.00001;
            chunk[i] = (2.0 * std::f64::consts::PI * freq * t).sin();
        }

        // Convert to Signal
        let signal = Signal::from_slice(&chunk)?;

        // Decompose with timing
        let decompose_start = Instant::now();
        let result = decomposer.decompose_chunk(&signal)?;
        let decompose_time = decompose_start.elapsed();

        total_samples_processed += chunk_size;

        // Display results
        println!(
            "[{:3}] {:.3}ms | {} IMFs | Entropy: {:.3} | Stationarity: {:.3}",
            chunk_idx,
            decompose_time.as_secs_f64() * 1000.0,
            result.imfs.len(),
            result.metrics.spectral_entropy,
            result.metrics.stationarity_score
        );

        // Show residue stats
        let residue_mean = result.remainder.iter().sum::<f64>() / result.remainder.len() as f64;
        let residue_std =
            (result.remainder.iter().map(|x| (x - residue_mean).powi(2)).sum::<f64>()
                / result.remainder.len() as f64)
                .sqrt();

        println!("      Residue: mean={:.4}, std={:.4}", residue_mean, residue_std);
    }

    let total_time = start_time.elapsed();
    println!(
        "\n✓ Processed {} samples in {:.2}s",
        total_samples_processed,
        total_time.as_secs_f64()
    );
    println!(
        "  Throughput: {:.0} samples/sec",
        total_samples_processed as f64 / total_time.as_secs_f64()
    );
    println!(
        "  Latency: {:.2}ms/chunk average",
        (total_time.as_secs_f64() / num_chunks as f64) * 1000.0
    );

    Ok(())
}
