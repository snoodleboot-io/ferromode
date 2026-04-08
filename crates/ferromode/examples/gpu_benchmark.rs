// Example: GPU vs CPU Benchmarking
//
// This example demonstrates how to benchmark GPU execution against CPU
// and measure performance improvements.
//
// Run with:
//   cargo run --example gpu_benchmark --release

use ferromode::adapters::gpu::{EnsembleExecutor, ExecutorConfig};
use ferromode::algorithms::eemd::EnsembleConfig;
use ferromode::algorithms::emd::EmdConfig;
use ferromode::types::Signal;
use std::time::Instant;

/// Generate a synthetic test signal.
fn generate_signal(samples: usize) -> Vec<f64> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / samples as f64 * 4.0 * std::f64::consts::PI;
            let low_freq = (t).sin();
            let high_freq = 0.3 * (5.0 * t).sin();
            let noise = ((i * 7919) % 256) as f64 / 128.0 - 1.0;
            low_freq + high_freq + 0.05 * noise
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sep = "=".repeat(70);
    println!("{}", sep);
    println!("GPU vs CPU EEMD Benchmarking Example");
    println!("{}", sep);
    println!();

    // Configuration
    let signal_size = 10_240;
    let num_trials = 100;
    let num_runs = 3;

    println!("Configuration:");
    println!("  Signal size: {} samples", signal_size);
    println!("  Ensemble trials: {}", num_trials);
    println!("  Number of runs: {}", num_runs);
    println!();

    // Generate signal
    let signal_data = generate_signal(signal_size);
    let signal = Signal::with_sample_rate(&signal_data, 1.0)?;

    // Setup configurations
    let ensemble_config =
        EnsembleConfig { num_ensembles: num_trials, noise_std: 0.2, seed: Some(42) };
    let emd_config = EmdConfig::default();

    // GPU execution (with fallback to CPU if GPU unavailable)
    println!("Running GPU EEMD {} times...", num_runs);
    let mut gpu_times = Vec::new();

    for run in 1..=num_runs {
        let executor_config = ExecutorConfig {
            max_gpu_memory: 8 * 1024 * 1024 * 1024,
            batch_size: 16,
            profiling_enabled: true,
        };
        let mut executor = EnsembleExecutor::new(executor_config)?;

        let start = Instant::now();
        let gpu_result = executor.execute_gpu_eemd(&signal, &ensemble_config)?;
        let elapsed = start.elapsed();
        gpu_times.push(elapsed.as_secs_f64());

        let stats = executor.stats();
        println!(
            "  Run {}: {:.3}s ({} IMFs, {} samples residue)",
            run,
            elapsed.as_secs_f64(),
            gpu_result.imfs.len(),
            gpu_result.residue.len()
        );
        println!(
            "    GPU time: {:.3}s | CPU time: {:.3}s | Available memory: {} MB",
            stats.gpu_time.as_secs_f64(),
            stats.cpu_time.as_secs_f64(),
            executor.available_memory() / (1024 * 1024)
        );
    }

    println!();

    // CPU execution (for comparison)
    println!("Running CPU EEMD {} times...", num_runs);
    let mut cpu_times = Vec::new();

    for run in 1..=num_runs {
        let start = Instant::now();
        let cpu_result =
            ferromode::algorithms::eemd::eemd(&signal_data, &ensemble_config, &emd_config)?;
        let elapsed = start.elapsed();
        cpu_times.push(elapsed.as_secs_f64());

        println!(
            "  Run {}: {:.3}s ({} IMFs, {} samples residue)",
            run,
            elapsed.as_secs_f64(),
            cpu_result.imfs.imfs.len(),
            cpu_result.imfs.residue.len()
        );
    }

    println!();
    let sep = "=".repeat(70);
    println!("{}", sep);
    println!("Summary");
    println!("{}", sep);

    let avg_gpu = gpu_times.iter().sum::<f64>() / gpu_times.len() as f64;
    let avg_cpu = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;
    let speedup = avg_cpu / avg_gpu;

    println!("Average GPU time: {:.3}s", avg_gpu);
    println!("Average CPU time: {:.3}s", avg_cpu);
    println!("Speedup: {:.1}x", speedup);
    println!();

    // Analysis
    if speedup > 1.0 {
        println!("✓ GPU execution is {:.1}x faster than CPU", speedup);
    } else {
        println!("✗ GPU execution is {:.2}x slower than CPU", 1.0 / speedup);
        println!("  (This is expected if no GPU is available - using CPU fallback)");
    }

    println!();

    // Try to get device info
    if let Ok(executor) = EnsembleExecutor::new(ExecutorConfig::default()) {
        println!("GPU executor device: {}", executor.current_device());
    }

    println!();

    Ok(())
}
