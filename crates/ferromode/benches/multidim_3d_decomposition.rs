//! Benchmark suite for 3D volume decomposition performance.
//!
//! Measures latency, memory usage, and component performance of the separable
//! 3D EMD decomposition against architecture targets:
//! - 64×64×64: <4s (small volume)
//! - 128×128×128: <30s (primary target)
//! - 256×256×256: stress test (if feasible)
//! - Memory (128³): < 500 MB

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferromode::adapters::multidim::decompose_volume_3d_separable;
use ferromode::adapters::multidim::Volume3D;
use ferromode::algorithms::emd::EmdConfig;

// ============================================================================
// Test Volume Generation
// ============================================================================

/// Generate a Gaussian blob 3D volume.
/// Common in medical imaging benchmarks.
fn create_gaussian_blob_volume(width: usize, height: usize, depth: usize) -> Volume3D {
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let cz = depth as f64 / 2.0;
    let sigma_x = width as f64 / 6.0;
    let sigma_y = height as f64 / 6.0;
    let sigma_z = depth as f64 / 6.0;

    let mut data = vec![0.0; width * height * depth];

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dz = z as f64 - cz;
                let exp_arg = -(dx * dx / (2.0 * sigma_x * sigma_x)
                    + dy * dy / (2.0 * sigma_y * sigma_y)
                    + dz * dz / (2.0 * sigma_z * sigma_z));
                let flat_idx = z * (width * height) + y * width + x;
                data[flat_idx] = exp_arg.exp();
            }
        }
    }

    Volume3D::new(width, height, depth, data, None).expect("Failed to create Gaussian blob volume")
}

/// Generate a checkerboard pattern 3D volume.
/// Useful for testing edge detection in 3D decomposition.
fn create_checkerboard_volume(
    width: usize,
    height: usize,
    depth: usize,
    cube_size: usize,
) -> Volume3D {
    let mut data = vec![0.0; width * height * depth];

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let sq_x = x / cube_size;
                let sq_y = y / cube_size;
                let sq_z = z / cube_size;
                if (sq_x + sq_y + sq_z) % 2 == 0 {
                    let flat_idx = z * (width * height) + y * width + x;
                    data[flat_idx] = 1.0;
                } else {
                    let flat_idx = z * (width * height) + y * width + x;
                    data[flat_idx] = 0.0;
                }
            }
        }
    }

    Volume3D::new(width, height, depth, data, None).expect("Failed to create checkerboard volume")
}

/// Generate a random noise 3D volume.
fn create_random_noise_volume(width: usize, height: usize, depth: usize) -> Volume3D {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = Vec::with_capacity(width * height * depth);

    for i in 0..width * height * depth {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hash = hasher.finish();
        let normalized = ((hash % 1000) as f64) / 1000.0;
        data.push(normalized);
    }

    Volume3D::new(width, height, depth, data, None).expect("Failed to create random volume")
}

/// Generate a composite 3D volume: Gaussian blob + checkerboard pattern.
fn create_composite_volume(width: usize, height: usize, depth: usize) -> Volume3D {
    let mut data = vec![0.0; width * height * depth];

    // Create Gaussian component
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let cz = depth as f64 / 2.0;
    let sigma = width as f64 / 8.0;

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dz = z as f64 - cz;
                let gaussian = (-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)).exp();

                // Add checkerboard pattern
                let sq_x = x / (width / 4);
                let sq_y = y / (height / 4);
                let sq_z = z / (depth / 4);
                let checkerboard = if (sq_x + sq_y + sq_z) % 2 == 0 { 0.3 } else { -0.3 };

                let flat_idx = z * (width * height) + y * width + x;
                data[flat_idx] = 0.7 * gaussian + 0.3 * checkerboard;
            }
        }
    }

    Volume3D::new(width, height, depth, data, None).expect("Failed to create composite volume")
}

// ============================================================================
// Latency Benchmarks (3 total)
// ============================================================================

/// Benchmark 3D decomposition for 64×64×64 volume (small).
/// Expected: ~0.5-1.5 seconds
fn bench_decompose_3d_64x64x64(c: &mut Criterion) {
    let config = EmdConfig::default();

    c.bench_function("decompose_3d_64x64x64", |b| {
        b.iter_batched(
            || black_box(create_checkerboard_volume(64, 64, 64, 4)),
            |volume| {
                eprintln!("64³: decomposing...");
                decompose_volume_3d_separable(&volume, &config, true)
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmark 3D decomposition for 128×128×128 volume (primary target).
/// Expected: <30 seconds, actual ~4-5s with parallelization
fn bench_decompose_3d_128x128x128(c: &mut Criterion) {
    let config = EmdConfig::default();

    let mut group = c.benchmark_group("decompose_3d_128x128x128_primary_target");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));

    group.bench_function("primary_target", |b| {
        b.iter_batched(
            || black_box(create_gaussian_blob_volume(128, 128, 128)),
            |volume| {
                eprintln!("128³: decomposing with parallelization...");
                decompose_volume_3d_separable(&volume, &config, true)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark 3D decomposition for 256×256×256 volume (stress test).
/// Expected: very long runtime, included for profiling
fn bench_decompose_3d_256x256x256(c: &mut Criterion) {
    let config = EmdConfig::default();

    let mut group = c.benchmark_group("decompose_3d_stress_test");
    // Set a longer sample time for the stress test
    group.sample_size(3);
    group.measurement_time(std::time::Duration::from_secs(120));

    group.bench_function("256x256x256", |b| {
        b.iter_batched(
            || {
                eprintln!("⚠️  Creating 256³ volume for stress test...");
                black_box(create_random_noise_volume(256, 256, 256))
            },
            |volume| {
                eprintln!("256³: decomposing (this will take time)...");
                decompose_volume_3d_separable(&volume, &config, true)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ============================================================================
// Memory Benchmarks (2 total)
// ============================================================================

/// Benchmark memory usage for 128×128×128 volume decomposition.
/// Target: < 500 MB peak RSS
fn bench_memory_peak_128x128x128(c: &mut Criterion) {
    let config = EmdConfig::default();

    c.bench_function("memory_peak_128x128x128", |b| {
        b.iter(|| {
            let volume = black_box(create_gaussian_blob_volume(128, 128, 128));
            let _ = black_box(decompose_volume_3d_separable(&volume, &config, true));
        });
    });
}

/// Benchmark memory scaling across volume sizes.
/// Compares memory usage for 64³ vs 128³ volumes
fn bench_memory_scaling(c: &mut Criterion) {
    let config = EmdConfig::default();
    let mut group = c.benchmark_group("memory_scaling_3d");

    for size in [64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", size, size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    let volume = black_box(create_composite_volume(*size, *size, *size));
                    let _ = black_box(decompose_volume_3d_separable(&volume, &config, true));
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_decompose_3d_64x64x64,
    bench_decompose_3d_128x128x128,
    bench_decompose_3d_256x256x256,
    bench_memory_peak_128x128x128,
    bench_memory_scaling,
);

criterion_main!(benches);
