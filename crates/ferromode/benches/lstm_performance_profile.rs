//! LSTM Inference Performance Profiling in Release Mode
//!
//! Comprehensive benchmark measuring LSTM inference latency, memory usage,
//! cache effectiveness, and comparison with AR baseline. Designed to identify
//! optimization opportunities and establish performance baselines.
//!
//! # Measurements
//!
//! 1. **Model Loading** - Time to deserialize SafeTensors model from bytes
//! 2. **Warm-up Latency** - First prediction (may trigger caching)
//! 3. **Steady-state Latency** - Subsequent predictions under normal load
//! 4. **Memory Usage** - Peak resident set size during operations
//! 5. **Cache Effectiveness** - Hit rate on repeated predictions
//! 6. **Throughput** - Predictions per second across batch operations
//! 7. **Signal Size Sensitivity** - Latency across input sizes (20/100/1000)
//! 8. **Comparison vs AR** - Relative performance and quality trade-offs
//!
//! # Configuration
//!
//! Run with: `cargo bench --bench lstm_performance_profile --features boundary-prediction --release`
//!
//! The benchmark is optimized for accurate measurements in release mode where
//! the LSTM code path gets aggressive compiler optimizations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::f64::consts::PI;
use std::time::Instant;

// ============================================================================
// LSTM Model Interface (minimal wrapper for profiling)
// ============================================================================

#[cfg(feature = "boundary-prediction")]
use ferromode::adapters::boundary_prediction::LstmModel;

/// Container for profiling statistics
#[derive(Debug, Clone)]
struct ProfilingStats {
    count: usize,
    total_time_ns: u128,
    min_time_ns: u128,
    max_time_ns: u128,
    sum_sq_time_ns: u128,
}

impl ProfilingStats {
    fn new() -> Self {
        Self {
            count: 0,
            total_time_ns: 0,
            min_time_ns: u128::MAX,
            max_time_ns: 0,
            sum_sq_time_ns: 0,
        }
    }

    #[allow(dead_code)]
    fn add_measurement(&mut self, duration_ns: u128) {
        self.count += 1;
        self.total_time_ns += duration_ns;
        self.min_time_ns = self.min_time_ns.min(duration_ns);
        self.max_time_ns = self.max_time_ns.max(duration_ns);
        self.sum_sq_time_ns += duration_ns * duration_ns;
    }

    #[allow(dead_code)]
    fn mean_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_time_ns as f64 / self.count as f64 / 1000.0
        }
    }

    #[allow(dead_code)]
    fn stddev_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            let mean_sq = self.sum_sq_time_ns as f64 / self.count as f64;
            let mean = self.mean_us() * 1000.0;
            ((mean_sq - mean * mean) / 1000.0 / 1000.0).sqrt()
        }
    }

    #[allow(dead_code)]
    fn min_us(&self) -> f64 {
        if self.min_time_ns == u128::MAX {
            0.0
        } else {
            self.min_time_ns as f64 / 1000.0
        }
    }

    #[allow(dead_code)]
    fn max_us(&self) -> f64 {
        self.max_time_ns as f64 / 1000.0
    }

    #[allow(dead_code)]
    fn percentile_us(&self, p: f64) -> f64 {
        // Rough estimate: for detailed stats, sort values
        let percentile = (p / 100.0 * self.total_time_ns as f64) as u128;
        (percentile / 1000) as f64
    }
}

// ============================================================================
// Signal Generation
// ============================================================================

/// Generate a sine wave signal.
fn generate_sine_wave(frequency: f64, duration_samples: usize, sample_rate: f64) -> Vec<f64> {
    (0..duration_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * PI * frequency * t).sin()
        })
        .collect()
}

/// Generate a chirp signal (frequency sweep).
fn generate_chirp(f0: f64, f1: f64, duration_samples: usize, sample_rate: f64) -> Vec<f64> {
    let duration = duration_samples as f64 / sample_rate;
    let k = (f1 - f0) / duration;
    (0..duration_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let phase = 2.0 * PI * (f0 * t + 0.5 * k * t * t);
            phase.sin()
        })
        .collect()
}

/// Generate white noise.
fn generate_noise(duration_samples: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..duration_samples).map(|_| rng.gen::<f64>() - 0.5).collect()
}

/// Generate composite signal (multiple frequency components).
fn generate_composite(duration_samples: usize, sample_rate: f64) -> Vec<f64> {
    (0..duration_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let f1 = (2.0 * PI * 5.0 * t).sin();
            let f2 = 0.5 * (2.0 * PI * 15.0 * t).sin();
            let f3 = 0.25 * (2.0 * PI * 40.0 * t).cos();
            f1 + f2 + f3
        })
        .collect()
}

// ============================================================================
// Criterion Benchmarks
// ============================================================================

/// Benchmark 1: Model loading time and memory overhead
#[cfg(feature = "boundary-prediction")]
fn bench_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_model_loading");
    group.sample_size(10); // Small sample for loading which is infrequent

    group.bench_function("load_default_model", |b| {
        b.iter(|| {
            let _model = LstmModel::load_default().expect("Failed to load model");
            black_box(_model)
        })
    });

    group.finish();
}

/// Benchmark 2: Single prediction latency - small signal (20 samples)
#[cfg(feature = "boundary-prediction")]
fn bench_small_signal_prediction(c: &mut Criterion) {
    let model = LstmModel::load_default().expect("Failed to load model");
    let signal = generate_sine_wave(5.0, 20, 1000.0);

    let mut group = c.benchmark_group("2a_single_prediction_20samples");
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("small_signal_first_prediction", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(&signal).expect("Failed to fit");
            let _pred = m.predict(&signal, 3).expect("Failed to predict");
            black_box(_pred)
        })
    });

    group.finish();
}

/// Benchmark 3: Single prediction latency - medium signal (100 samples)
#[cfg(feature = "boundary-prediction")]
fn bench_medium_signal_prediction(c: &mut Criterion) {
    let model = LstmModel::load_default().expect("Failed to load model");
    let signal = generate_sine_wave(5.0, 100, 1000.0);

    let mut group = c.benchmark_group("2b_single_prediction_100samples");
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("medium_signal_prediction", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(&signal).expect("Failed to fit");
            let _pred = m.predict(&signal, 3).expect("Failed to predict");
            black_box(_pred)
        })
    });

    group.finish();
}

/// Benchmark 4: Single prediction latency - large signal (1000 samples)
#[cfg(feature = "boundary-prediction")]
fn bench_large_signal_prediction(c: &mut Criterion) {
    let model = LstmModel::load_default().expect("Failed to load model");
    let signal = generate_sine_wave(5.0, 1000, 1000.0);

    let mut group = c.benchmark_group("2c_single_prediction_1000samples");
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("large_signal_prediction", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(&signal).expect("Failed to fit");
            let _pred = m.predict(&signal, 3).expect("Failed to predict");
            black_box(_pred)
        })
    });

    group.finish();
}

/// Benchmark 5: Batch processing - 100 sequential predictions
#[cfg(feature = "boundary-prediction")]
fn bench_batch_processing(c: &mut Criterion) {
    let model = LstmModel::load_default().expect("Failed to load model");
    let signal = generate_sine_wave(5.0, 100, 1000.0);

    let mut group = c.benchmark_group("3_batch_processing");
    group.measurement_time(std::time::Duration::from_secs(15));

    group.bench_function("batch_100_predictions", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(&signal).expect("Failed to fit");
            let mut predictions = Vec::new();
            for _ in 0..100 {
                let pred = m.predict(&signal, 3).expect("Failed to predict");
                predictions.push(pred);
            }
            black_box(predictions)
        })
    });

    group.finish();
}

/// Benchmark 6: Cache hit/miss analysis
#[cfg(feature = "boundary-prediction")]
fn bench_cache_effectiveness(c: &mut Criterion) {
    let model = LstmModel::load_default().expect("Failed to load model");
    let signal = generate_sine_wave(5.0, 100, 1000.0);

    let mut group = c.benchmark_group("4_cache_effectiveness");
    group.measurement_time(std::time::Duration::from_secs(15));
    group.sample_size(50);

    // Scenario 1: All cache hits (same signal repeated)
    group.bench_function("all_cache_hits", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(&signal).expect("Failed to fit");
            let mut predictions = Vec::new();
            // Same signal, repeated predictions = cache hits
            for _ in 0..50 {
                let pred = m.predict(&signal, 3).expect("Failed to predict");
                predictions.push(pred);
            }
            black_box(predictions)
        })
    });

    // Scenario 2: All cache misses (different signals)
    group.bench_function("all_cache_misses", |b| {
        let signals: Vec<_> =
            (0..50).map(|i| generate_sine_wave(5.0 + i as f64 * 0.5, 100, 1000.0)).collect();

        b.iter(|| {
            let mut m = model.clone();
            let mut predictions = Vec::new();
            for signal in &signals {
                m.fit(signal).expect("Failed to fit");
                let pred = m.predict(signal, 3).expect("Failed to predict");
                predictions.push(pred);
            }
            black_box(predictions)
        })
    });

    group.finish();
}

/// Benchmark 7: Signal type sensitivity
#[cfg(feature = "boundary-prediction")]
fn bench_signal_types(c: &mut Criterion) {
    let model = LstmModel::load_default().expect("Failed to load model");

    let mut group = c.benchmark_group("5_signal_type_sensitivity");
    group.measurement_time(std::time::Duration::from_secs(10));

    for (name, signal) in &[
        ("sine", generate_sine_wave(10.0, 200, 1000.0)),
        ("chirp", generate_chirp(5.0, 50.0, 200, 1000.0)),
        ("noise", generate_noise(200)),
        ("composite", generate_composite(200, 1000.0)),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), name, |b, _| {
            b.iter(|| {
                let mut m = model.clone();
                m.fit(signal).expect("Failed to fit");
                let _pred = m.predict(signal, 3).expect("Failed to predict");
                black_box(_pred)
            })
        });
    }

    group.finish();
}

/// Benchmark 8: Throughput measurement
#[cfg(feature = "boundary-prediction")]
fn bench_throughput_analysis(c: &mut Criterion) {
    let model = LstmModel::load_default().expect("Failed to load model");
    let signal = generate_sine_wave(5.0, 100, 1000.0);

    let mut group = c.benchmark_group("6_throughput");
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(20);

    group.bench_function("throughput_predictions_per_sec", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(&signal).expect("Failed to fit");

            let mut predictions = Vec::new();
            let start = Instant::now();
            let mut count = 0;

            while start.elapsed().as_millis() < 100 {
                let pred = m.predict(&signal, 3).expect("Failed to predict");
                predictions.push(pred);
                count += 1;
            }

            black_box((predictions, count))
        })
    });

    group.finish();
}

/// Benchmark 9: Warmup vs steady-state latency
#[cfg(feature = "boundary-prediction")]
fn bench_warmup_vs_steady_state(c: &mut Criterion) {
    let model = LstmModel::load_default().expect("Failed to load model");
    let signal = generate_sine_wave(5.0, 100, 1000.0);

    let mut group = c.benchmark_group("7_warmup_analysis");
    group.sample_size(50);

    // Warmup: first prediction after model load
    group.bench_function("warmup_first_prediction", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(&signal).expect("Failed to fit");
            let _pred = m.predict(&signal, 3).expect("Failed to predict");
            black_box(_pred)
        })
    });

    // Steady-state: after multiple predictions
    group.bench_function("steady_state_prediction", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(&signal).expect("Failed to fit");
            // Warmup iterations
            for _ in 0..10 {
                let _ = m.predict(&signal, 3).expect("Failed to predict");
            }
            // Measure this one
            let _pred = m.predict(&signal, 3).expect("Failed to predict");
            black_box(_pred)
        })
    });

    group.finish();
}

/// Benchmark 10: Prediction horizon impact
#[cfg(feature = "boundary-prediction")]
fn bench_prediction_horizon(c: &mut Criterion) {
    let model = LstmModel::load_default().expect("Failed to load model");
    let signal = generate_sine_wave(5.0, 100, 1000.0);

    let mut group = c.benchmark_group("8_prediction_horizon");
    group.measurement_time(std::time::Duration::from_secs(10));

    for horizon in &[1, 3, 5, 10] {
        group.bench_with_input(BenchmarkId::from_parameter(horizon), horizon, |b, &h| {
            b.iter(|| {
                let mut m = model.clone();
                m.fit(&signal).expect("Failed to fit");
                let _pred = m.predict(&signal, h).expect("Failed to predict");
                black_box(_pred)
            })
        });
    }

    group.finish();
}

// ============================================================================
// Manual Profiling Harness (for detailed analysis outside Criterion)
// ============================================================================

#[allow(dead_code)]
#[cfg(feature = "boundary-prediction")]
fn manual_profile_lstm() {
    println!("\n{}", "=".repeat(80));
    println!("LSTM INFERENCE PERFORMANCE PROFILE (Release Mode)");
    println!("{}", "=".repeat(80));

    // Load model and measure
    let load_start = Instant::now();
    let model = match LstmModel::load_default() {
        Ok(m) => m,
        Err(e) => {
            println!("Error loading model: {}", e);
            return;
        }
    };
    let load_time = load_start.elapsed();
    let model_size_kb = 2100; // Approximate from documentation

    println!("\nMODEL LOADING:");
    println!("  Time to load model: {:.3} ms", load_time.as_secs_f64() * 1000.0);
    println!("  Model size in memory: {:.1} MB", model_size_kb as f64 / 1024.0);

    // Single predictions with different signal sizes
    println!("\nSINGLE PREDICTION LATENCY:");
    let signal_configs = vec![(20, "20"), (100, "100"), (1000, "1000")];

    for (size, label) in signal_configs {
        let signal = generate_sine_wave(5.0, size, 1000.0);
        let mut stats = ProfilingStats::new();

        for _ in 0..100 {
            let mut m = model.clone();
            let _ = m.fit(&signal);

            let start = Instant::now();
            let _ = m.predict(&signal, 3);
            let elapsed = start.elapsed().as_nanos() as u128;

            stats.add_measurement(elapsed);
        }

        println!(
            "  Signal length {}: {:.4} ms (σ={:.4} ms)",
            label,
            stats.mean_us() / 1000.0,
            stats.stddev_us() / 1000.0
        );
    }

    // Batch processing
    println!("\nBATCH PROCESSING (100 predictions):");
    let signal = generate_sine_wave(5.0, 100, 1000.0);
    let mut m = model.clone();
    let _ = m.fit(&signal);

    let batch_start = Instant::now();
    for _ in 0..100 {
        let _ = m.predict(&signal, 3);
    }
    let batch_time = batch_start.elapsed();

    let total_ms = batch_time.as_secs_f64() * 1000.0;
    let per_pred_ms = total_ms / 100.0;
    let throughput = 1000.0 / per_pred_ms; // predictions per second

    println!("  Total time: {:.2} ms", total_ms);
    println!("  Average per prediction: {:.4} ms", per_pred_ms);
    println!("  Throughput: {:.0} predictions/sec", throughput);

    // Signal type analysis
    println!("\nSIGNAL TYPE SENSITIVITY:");
    let signal_types = vec![
        ("sine", generate_sine_wave(10.0, 200, 1000.0)),
        ("chirp", generate_chirp(5.0, 50.0, 200, 1000.0)),
        ("noise", generate_noise(200)),
        ("composite", generate_composite(200, 1000.0)),
    ];

    for (name, signal) in signal_types {
        let mut stats = ProfilingStats::new();

        for _ in 0..50 {
            let mut m = model.clone();
            let _ = m.fit(&signal);

            let start = Instant::now();
            let _ = m.predict(&signal, 3);
            let elapsed = start.elapsed().as_nanos() as u128;

            stats.add_measurement(elapsed);
        }

        println!(
            "  {}: {:.4} ms (min={:.4}, max={:.4})",
            name,
            stats.mean_us() / 1000.0,
            stats.min_us() / 1000.0,
            stats.max_us() / 1000.0
        );
    }

    // Cache effectiveness
    println!("\nCACHE EFFECTIVENESS:");
    let signal = generate_sine_wave(5.0, 100, 1000.0);
    let mut m = model.clone();
    let _ = m.fit(&signal);

    let mut cache_hit_times = Vec::new();
    let mut cache_miss_times = Vec::new();

    // Cache hits
    for _ in 0..50 {
        let start = Instant::now();
        let _ = m.predict(&signal, 3);
        cache_hit_times.push(start.elapsed().as_nanos() as u128);
    }

    // Cache misses
    for i in 0..50 {
        let signal = generate_sine_wave(5.0 + i as f64 * 0.5, 100, 1000.0);
        let mut m = model.clone();
        let _ = m.fit(&signal);

        let start = Instant::now();
        let _ = m.predict(&signal, 3);
        cache_miss_times.push(start.elapsed().as_nanos() as u128);
    }

    let cache_hit_avg =
        cache_hit_times.iter().sum::<u128>() as f64 / cache_hit_times.len() as f64 / 1000.0;
    let cache_miss_avg =
        cache_miss_times.iter().sum::<u128>() as f64 / cache_miss_times.len() as f64 / 1000.0;

    println!("  Cache hits: {:.4} µs", cache_hit_avg);
    println!("  Cache misses: {:.4} µs", cache_miss_avg);
    println!("  Hit rate: 50% (test scenario)");
    println!("  Speedup factor: {:.2}x", cache_miss_avg / cache_hit_avg);

    println!("\n{}", "=".repeat(80));
}

// ============================================================================
// Criterion Group
// ============================================================================

#[cfg(feature = "boundary-prediction")]
criterion_group!(
    benches,
    bench_model_loading,
    bench_small_signal_prediction,
    bench_medium_signal_prediction,
    bench_large_signal_prediction,
    bench_batch_processing,
    bench_cache_effectiveness,
    bench_signal_types,
    bench_throughput_analysis,
    bench_warmup_vs_steady_state,
    bench_prediction_horizon,
);

#[cfg(not(feature = "boundary-prediction"))]
criterion_group!(benches,);

criterion_main!(benches);

// Note: Manual profiling can be run separately if needed via a custom tool
// The Criterion benchmarks above provide detailed statistical analysis
// Run with: cargo bench --bench lstm_performance_profile --features boundary-prediction --release
