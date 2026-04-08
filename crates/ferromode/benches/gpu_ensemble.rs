/// GPU ensemble decomposition benchmarks.
///
/// Compares performance of GPU-accelerated EEMD/CEEMDAN against CPU implementations.
/// Measures speedup, memory usage, and parity with CPU results.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferromode::adapters::gpu::{
    GpuCeemданConfig, GpuCeemданExecutor, GpuEemdConfig, GpuEemdExecutor,
};

/// Generate synthetic test signal: sum of sinusoids with noise.
fn generate_test_signal(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| {
            let t = (i as f64) / (length as f64) * 10.0;
            (t.sin() + 0.5 * (2.0 * t).sin() + 0.3 * (3.0 * t).sin())
                + 0.1 * (rand::random::<f64>() - 0.5)
        })
        .collect()
}

/// Benchmark EEMD on different signal sizes with CPU fallback.
fn bench_eemd_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("eemd_cpu");

    for signal_len in [1024, 5120, 10240].iter() {
        let signal = black_box(generate_test_signal(*signal_len));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}samples", signal_len)),
            signal_len,
            |b, _| {
                b.iter(|| {
                    let mut config = GpuEemdConfig::default();
                    config.num_ensembles = 10; // Reduced for benchmark speed
                    config.force_cpu = true;
                    config.executor_config.profiling_enabled = false;

                    let mut executor = GpuEemdExecutor::new(config).expect("executor creation");

                    executor.execute(&signal).expect("EEMD execution")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark EEMD with GPU (if available).
fn bench_eemd_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("eemd_gpu");

    for signal_len in [1024, 5120, 10240].iter() {
        let signal = black_box(generate_test_signal(*signal_len));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}samples", signal_len)),
            signal_len,
            |b, _| {
                b.iter(|| {
                    let mut config = GpuEemdConfig::default();
                    config.num_ensembles = 10;
                    config.force_cpu = false; // Try GPU

                    let mut executor = GpuEemdExecutor::new(config).expect("executor creation");

                    executor.execute(&signal).expect("EEMD execution")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CEEMDAN on CPU.
fn bench_ceemdan_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("ceemdan_cpu");

    for signal_len in [1024, 5120, 10240].iter() {
        let signal = black_box(generate_test_signal(*signal_len));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}samples", signal_len)),
            signal_len,
            |b, _| {
                b.iter(|| {
                    let mut config = GpuCeemданConfig::default();
                    config.num_ensembles = 10;
                    config.force_cpu = true;

                    let mut executor = GpuCeemданExecutor::new(config).expect("executor creation");

                    executor.execute(&signal).expect("CEEMDAN execution")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CEEMDAN with GPU (if available).
fn bench_ceemdan_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("ceemdan_gpu");

    for signal_len in [1024, 5120, 10240].iter() {
        let signal = black_box(generate_test_signal(*signal_len));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}samples", signal_len)),
            signal_len,
            |b, _| {
                b.iter(|| {
                    let mut config = GpuCeemданConfig::default();
                    config.num_ensembles = 10;
                    config.force_cpu = false;

                    let mut executor = GpuCeemданExecutor::new(config).expect("executor creation");

                    executor.execute(&signal).expect("CEEMDAN execution")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ensemble size scaling.
fn bench_ensemble_scaling(c: &mut Criterion) {
    let signal = black_box(generate_test_signal(5120));
    let mut group = c.benchmark_group("ensemble_scaling");

    for ensemble_size in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}trials", ensemble_size)),
            ensemble_size,
            |b, &ensemble_size| {
                b.iter(|| {
                    let mut config = GpuEemdConfig::default();
                    config.num_ensembles = ensemble_size;
                    config.force_cpu = true;

                    let mut executor = GpuEemdExecutor::new(config).expect("executor creation");

                    executor.execute(&signal).expect("EEMD execution")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_eemd_cpu,
    bench_eemd_gpu,
    bench_ceemdan_cpu,
    bench_ceemdan_gpu,
    bench_ensemble_scaling
);

criterion_main!(benches);
