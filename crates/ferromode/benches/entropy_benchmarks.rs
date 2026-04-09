// Entropy computation latency benchmarks
// T-334: Benchmark entropy computation and validate <100ms target for full analysis

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferromode::analysis::{
    permutation_entropy_normalized, sample_entropy, spectral_entropy_normalized, EntropyAnalysis,
};
use ferromode::types::{AlgorithmType, DecompositionResult, ImfCollection};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Signal Generation Helpers
// ---------------------------------------------------------------------------

/// Generate a synthetic sinusoidal signal with noise
fn generate_sinusoid(n_samples: usize, freq: f64, noise_level: f64) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..n_samples)
        .map(|i| {
            let t = i as f64 / 1000.0;
            let signal = (2.0 * PI * freq * t).sin();
            let noise = (2.0 * (i as f64 % 1000.0) - 1000.0) / 1000.0; // Simple pseudo-noise
            signal + noise_level * noise
        })
        .collect()
}

/// Generate white noise signal
fn generate_noise(n_samples: usize) -> Vec<f64> {
    (0..n_samples)
        .map(|i| {
            // Simple pseudo-random using modulo operations
            let x = ((i as f64 * 7919.0) % 1000.0 - 500.0) / 500.0;
            x
        })
        .collect()
}

/// Generate a complex signal with multiple frequency components
fn generate_multifrequency(n_samples: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..n_samples)
        .map(|i| {
            let t = i as f64 / 1000.0;
            let f1 = (2.0 * PI * 5.0 * t).sin();
            let f2 = 0.5 * (2.0 * PI * 15.0 * t).sin();
            let f3 = 0.25 * (2.0 * PI * 25.0 * t).sin();
            let noise = ((i as f64 * 1234.0) % 1000.0 - 500.0) / 5000.0;
            f1 + f2 + f3 + noise
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark: Spectral Entropy (T-330)
// ---------------------------------------------------------------------------

/// Benchmark spectral entropy on 10k sample signal
/// Target: <1 ms
/// Expected: 0.3-0.8 ms (FFT is O(N log N))
fn bench_spectral_entropy_10k(c: &mut Criterion) {
    let signal = black_box(generate_sinusoid(10000, 5.0, 0.1));

    c.bench_function("spectral_entropy_10k", |b| b.iter(|| spectral_entropy_normalized(&signal)));
}

/// Benchmark spectral entropy on different signal sizes
fn bench_spectral_entropy_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_entropy_scaling");

    for size in [1000, 5000, 10000, 50000].iter() {
        let signal = black_box(generate_sinusoid(*size, 5.0, 0.1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| spectral_entropy_normalized(&signal))
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Permutation Entropy (T-331)
// ---------------------------------------------------------------------------

/// Benchmark permutation entropy on 10k sample signal
/// Target: <10 ms
/// Expected: 2-5 ms (O(N) with pattern matching)
fn bench_permutation_entropy_10k(c: &mut Criterion) {
    let signal = black_box(generate_multifrequency(10000));

    c.bench_function("permutation_entropy_10k", |b| {
        b.iter(|| permutation_entropy_normalized(&signal, 3))
    });
}

/// Benchmark permutation entropy with different embedding dimensions
fn bench_permutation_entropy_embedding_dims(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutation_entropy_dims");
    let signal = black_box(generate_multifrequency(10000));

    for dim in [2, 3, 4, 5].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("dim_{}", dim)),
            dim,
            |b, &dim| b.iter(|| permutation_entropy_normalized(&signal, dim)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Sample Entropy (T-332)
// ---------------------------------------------------------------------------

/// Benchmark sample entropy on 10k sample signal
/// Target: <50 ms
/// Expected: 30-40 ms (O(N²) comparison-based)
fn bench_sample_entropy_10k(c: &mut Criterion) {
    let signal = black_box(generate_noise(10000));

    c.bench_function("sample_entropy_10k", |b| b.iter(|| sample_entropy(&signal, 2, None)));
}

/// Benchmark sample entropy with different embedding dimensions
fn bench_sample_entropy_embedding_dims(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_entropy_dims");
    group.sample_size(10); // Reduce sample size for slower benchmark

    let signal = black_box(generate_noise(10000));

    for dim in [1, 2, 3].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("dim_{}", dim)),
            dim,
            |b, &dim| b.iter(|| sample_entropy(&signal, dim, None)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Full Entropy Analysis
// ---------------------------------------------------------------------------

/// Create a synthetic decomposition result with N IMFs of given size each
fn create_synthetic_decomposition(n_imfs: usize, imf_size: usize) -> DecompositionResult {
    let imfs: Vec<Vec<f64>> = (0..n_imfs)
        .map(|idx| {
            let freq = 3.0 + (idx as f64 * 2.0); // Increasing frequency per IMF
            generate_sinusoid(imf_size, freq, 0.05)
        })
        .collect();

    let residue = vec![0.0; imf_size];

    DecompositionResult::new(
        AlgorithmType::EMD,
        ImfCollection::new(imfs, residue),
        Duration::from_millis(0), // Dummy duration
        0,                        // Dummy sifting count
        "benchmark".to_string(),  // Dummy config
    )
}

/// Benchmark full entropy analysis on synthetic 10-IMF decomposition
/// Total: 10 × 10k = 100k samples
/// Target: <100 ms for full analysis
/// Expected: 60-80 ms (sum of all 3 entropy metrics)
fn bench_entropy_analysis_full(c: &mut Criterion) {
    let decomposition = black_box(create_synthetic_decomposition(10, 10000));

    c.bench_function("entropy_analysis_full_10imfs", |b| {
        b.iter(|| EntropyAnalysis::from_decomposition(&decomposition))
    });
}

/// Benchmark full entropy analysis with different IMF counts
fn bench_entropy_analysis_scaling_imfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_analysis_imf_scaling");
    group.sample_size(10); // Reduce sample size for intensive benchmark

    for n_imfs in [5, 10, 15, 20].iter() {
        let decomposition = black_box(create_synthetic_decomposition(*n_imfs, 10000));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_imfs", n_imfs)),
            n_imfs,
            |b, _| b.iter(|| EntropyAnalysis::from_decomposition(&decomposition)),
        );
    }

    group.finish();
}

/// Benchmark full entropy analysis with different signal sizes per IMF
fn bench_entropy_analysis_scaling_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_analysis_size_scaling");
    group.sample_size(10);

    for size in [1000, 5000, 10000, 50000].iter() {
        let decomposition = black_box(create_synthetic_decomposition(10, *size));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}samples", size)),
            size,
            |b, _| b.iter(|| EntropyAnalysis::from_decomposition(&decomposition)),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Real-world Scenario: Analysis on Real EMD Results
// ---------------------------------------------------------------------------

/// Simulate realistic decomposition with decreasing IMF energy
fn create_realistic_decomposition(n_imfs: usize, signal_size: usize) -> DecompositionResult {
    let imfs: Vec<Vec<f64>> = (0..n_imfs)
        .map(|idx| {
            // IMF amplitude decreases with index (typical EMD behavior)
            let amplitude_decay = 0.85_f64.powi(idx as i32);
            let freq = 5.0 + (idx as f64 * 3.0); // Increasing frequency
            let noise = 0.02 * amplitude_decay; // Noise decreases with amplitude

            generate_sinusoid(signal_size, freq, noise)
                .iter()
                .map(|&x| x * amplitude_decay)
                .collect()
        })
        .collect();

    let residue = vec![0.0; signal_size];

    DecompositionResult::new(
        AlgorithmType::EMD,
        ImfCollection::new(imfs, residue),
        Duration::from_millis(0),
        0,
        "benchmark".to_string(),
    )
}

/// Benchmark entropy analysis on realistic decomposition (1000 samples)
/// Expected: <10 ms
fn bench_entropy_analysis_realistic(c: &mut Criterion) {
    let decomposition = black_box(create_realistic_decomposition(8, 1000));

    c.bench_function("entropy_analysis_realistic_1k", |b| {
        b.iter(|| EntropyAnalysis::from_decomposition(&decomposition))
    });
}

// ---------------------------------------------------------------------------
// Criterion Configuration
// ---------------------------------------------------------------------------

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(10));
    targets =
        bench_spectral_entropy_10k,
        bench_spectral_entropy_scaling,
        bench_permutation_entropy_10k,
        bench_permutation_entropy_embedding_dims,
        bench_sample_entropy_10k,
        bench_sample_entropy_embedding_dims,
        bench_entropy_analysis_full,
        bench_entropy_analysis_scaling_imfs,
        bench_entropy_analysis_scaling_size,
        bench_entropy_analysis_realistic
);

criterion_main!(benches);
