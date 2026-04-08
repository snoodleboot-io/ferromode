//! Performance benchmarks for streaming decomposer.
//!
//! Measures:
//! - Latency per 1000-sample chunk (mean, p50, p99)
//! - Memory usage (peak RSS during continuous operation)
//! - Chunk sizes: 256, 512, 1024, 2048 samples
//! - Duration: 1 minute continuous processing

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferromode::adapters::streaming::{ArModel, StreamingDecomposer};
use ferromode::algorithms::emd::EmdConfig;
use ferromode::types::Signal;
use std::time::Instant;

/// Benchmark latency for a single chunk decomposition.
fn bench_chunk_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_latency");

    for chunk_size in [256, 512, 1024, 2048].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}samples", chunk_size)),
            chunk_size,
            |b, &chunk_size| {
                b.iter_custom(|iters| {
                    let config = EmdConfig::default();
                    let predictor = Box::new(ArModel::new(3).unwrap());
                    let mut decomposer =
                        StreamingDecomposer::new(config, predictor, chunk_size * 4).unwrap();

                    // Generate test signal (sine)
                    let mut chunk = vec![0.0; chunk_size];
                    for i in 0..chunk_size {
                        chunk[i] = ((i as f64 * 0.02).sin());
                    }
                    let signal = Signal::from_slice(&chunk).unwrap();

                    // Time the decomposition
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _ = decomposer.decompose_chunk(black_box(&signal));
                    }
                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark latency statistics over many chunks.
///
/// Measures mean, p50, p99 latency for 100+ chunks.
fn bench_latency_percentiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_percentiles");
    group.sample_size(50); // Smaller sample size for longer-running benchmark

    group.bench_function("1000samples_100chunks_p99", |b| {
        b.iter_custom(|_iters| {
            let config = EmdConfig::default();
            let predictor = Box::new(ArModel::new(3).unwrap());
            let mut decomposer = StreamingDecomposer::new(config, predictor, 4096).unwrap();

            // Generate signal (sine with varying frequency)
            let mut chunk = vec![0.0; 1000];
            let mut durations = Vec::new();

            for chunk_idx in 0..100 {
                for i in 0..1000 {
                    let freq = 0.02 + (chunk_idx as f64 * 0.001);
                    chunk[i] = ((i as f64 * freq).sin());
                }

                let signal = Signal::from_slice(&chunk).unwrap();
                let start = Instant::now();
                let _ = decomposer.decompose_chunk(black_box(&signal));
                let elapsed = start.elapsed();
                durations.push(elapsed);
            }

            // Calculate p99
            durations.sort();
            let p99_idx = (durations.len() as f64 * 0.99) as usize;
            durations[p99_idx]
        });
    });

    group.finish();
}

/// Benchmark memory usage during continuous operation.
///
/// Simulates 1-minute continuous processing with rolling window.
fn bench_memory_continuous(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_memory");
    group.measurement_time(std::time::Duration::from_secs(30));
    group.sample_size(10);

    for chunk_size in [256, 512, 1024, 2048].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}samples_1min", chunk_size)),
            chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    let config = EmdConfig::default();
                    let predictor = Box::new(ArModel::new(3).unwrap());
                    let mut decomposer =
                        StreamingDecomposer::new(config, predictor, chunk_size * 4).unwrap();

                    // Generate signal chunks
                    let mut chunk = vec![0.0; chunk_size];
                    for i in 0..chunk_size {
                        chunk[i] = ((i as f64 * 0.02).sin());
                    }
                    let signal = Signal::from_slice(&chunk).unwrap();

                    // Process 3600 chunks (1 second each @ 1000 Hz = 60 chunks/sec, 60 sec)
                    for _chunk_idx in 0..3600 {
                        let _ = decomposer.decompose_chunk(black_box(&signal));
                    }

                    black_box(decomposer)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark with chirp signal (frequency sweep).
///
/// More realistic signal: frequency increases linearly over chunk.
fn bench_chirp_signal(c: &mut Criterion) {
    c.bench_function("streaming_chirp_1024samples", |b| {
        b.iter_custom(|iters| {
            let config = EmdConfig::default();
            let predictor = Box::new(ArModel::new(3).unwrap());
            let mut decomposer = StreamingDecomposer::new(config, predictor, 4096).unwrap();

            // Generate chirp signal
            let mut chunk = vec![0.0; 1024];
            let start_freq = 0.01;
            let end_freq = 0.1;
            for i in 0..1024 {
                let t = i as f64;
                let freq = start_freq + (end_freq - start_freq) * (t / 1024.0);
                chunk[i] = (2.0 * std::f64::consts::PI * freq * t).sin();
            }
            let signal = Signal::from_slice(&chunk).unwrap();

            // Time decomposition
            let start = Instant::now();
            for _ in 0..iters {
                let _ = decomposer.decompose_chunk(black_box(&signal));
            }
            start.elapsed()
        });
    });
}

/// Benchmark with composite signal (multiple frequencies).
fn bench_composite_signal(c: &mut Criterion) {
    c.bench_function("streaming_composite_1024samples", |b| {
        b.iter_custom(|iters| {
            let config = EmdConfig::default();
            let predictor = Box::new(ArModel::new(3).unwrap());
            let mut decomposer = StreamingDecomposer::new(config, predictor, 4096).unwrap();

            // Generate composite signal (sum of 3 sines)
            let mut chunk = vec![0.0; 1024];
            for i in 0..1024 {
                let t = i as f64 / 1024.0;
                chunk[i] = (0.5 * (2.0 * std::f64::consts::PI * 0.02 * i as f64).sin())
                    + (0.3 * (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin())
                    + (0.2 * (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin());
            }
            let signal = Signal::from_slice(&chunk).unwrap();

            // Time decomposition
            let start = Instant::now();
            for _ in 0..iters {
                let _ = decomposer.decompose_chunk(black_box(&signal));
            }
            start.elapsed()
        });
    });
}

criterion_group!(
    benches,
    bench_chunk_latency,
    bench_latency_percentiles,
    bench_memory_continuous,
    bench_chirp_signal,
    bench_composite_signal
);
criterion_main!(benches);
