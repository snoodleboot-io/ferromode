//! Basic streaming decomposition example.
//!
//! Demonstrates simple chunk-by-chunk decomposition of a synthetic sine signal.

use ferromode::adapters::streaming::{ArModel, StreamingDecomposer};
use ferromode::algorithms::emd::EmdConfig;
use ferromode::types::Signal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming Decomposition: Basic Example ===\n");

    // Create decomposer with default configuration
    let config = EmdConfig::default();
    let predictor = Box::new(ArModel::new(3)?);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 4096)?;

    // Process 5 chunks of a 1280-sample signal
    let chunk_size = 256;
    let num_chunks = 5;
    let total_samples = chunk_size * num_chunks;

    println!("Configuration:");
    println!("  Chunk size: {} samples", chunk_size);
    println!("  Number of chunks: {}", num_chunks);
    println!("  Total samples: {}\n", total_samples);

    let mut chunk = vec![0.0; chunk_size];

    for chunk_idx in 0..num_chunks {
        // Generate sine signal for this chunk
        for i in 0..chunk_size {
            let sample_idx = chunk_idx * chunk_size + i;
            let t = sample_idx as f64 * 0.01;
            chunk[i] = t.sin();
        }

        // Convert to Signal
        let signal = Signal::from_slice(&chunk)?;

        // Decompose chunk
        let result = decomposer.decompose_chunk(&signal)?;

        // Display results
        println!(
            "Chunk {}: {} IMFs extracted, metrics: entropy={:.3}, stationarity={:.3}",
            chunk_idx,
            result.imfs.len(),
            result.metrics.spectral_entropy,
            result.metrics.stationarity_score
        );

        // Show shape of residue
        println!(
            "  Residue shape: {} samples, mean value: {:.4}\n",
            result.remainder.len(),
            result.remainder.iter().sum::<f64>() / result.remainder.len() as f64
        );
    }

    println!("✓ Decomposition complete!");
    Ok(())
}
