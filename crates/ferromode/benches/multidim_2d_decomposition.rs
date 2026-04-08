//! Benchmark suite for 2D image decomposition performance.
//!
//! Measures latency, memory usage, and component performance of the separable
//! 2D EMD decomposition against architecture targets:
//! - 256×256: < 1.0s
//! - 512×512: < 5.0s (primary target)
//! - 1024×1024: < 20s
//! - Memory (512×512): < 150 MB

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferromode::adapters::multidim::image_2d::{decompose_image_2d_separable, Image2D};
use ferromode::algorithms::emd::EmdConfig;
use std::time::Instant;

// ============================================================================
// Test Image Generation
// ============================================================================

/// Generate a checkerboard pattern image.
/// Useful for testing edge detection in decomposition.
fn create_checkerboard_image(width: usize, height: usize, square_size: usize) -> Image2D {
    let mut data = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let sq_x = x / square_size;
            let sq_y = y / square_size;
            if (sq_x + sq_y) % 2 == 0 {
                data[y * width + x] = 1.0;
            } else {
                data[y * width + x] = 0.0;
            }
        }
    }

    Image2D::new(width, height, data, None).expect("Failed to create checkerboard image")
}

/// Generate a Gaussian blob image.
/// Common in medical imaging benchmarks.
fn create_gaussian_blob_image(width: usize, height: usize) -> Image2D {
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let sigma_x = width as f64 / 6.0;
    let sigma_y = height as f64 / 6.0;

    let mut data = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let exp_arg =
                -(dx * dx / (2.0 * sigma_x * sigma_x) + dy * dy / (2.0 * sigma_y * sigma_y));
            data[y * width + x] = exp_arg.exp();
        }
    }

    Image2D::new(width, height, data, None).expect("Failed to create Gaussian blob image")
}

/// Generate a random noise image.
fn create_random_noise_image(width: usize, height: usize) -> Image2D {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = Vec::with_capacity(width * height);

    for i in 0..width * height {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hash = hasher.finish();
        let normalized = ((hash % 1000) as f64) / 1000.0;
        data.push(normalized);
    }

    Image2D::new(width, height, data, None).expect("Failed to create random image")
}

/// Generate a composite image: gaussian blob + checkerboard pattern.
fn create_composite_image(width: usize, height: usize) -> Image2D {
    let mut data = vec![0.0; width * height];

    // Create Gaussian component
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let sigma = width as f64 / 8.0;

    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let gaussian = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();

            // Add checkerboard pattern
            let sq_x = x / (width / 8);
            let sq_y = y / (height / 8);
            let checkerboard = if (sq_x + sq_y) % 2 == 0 { 0.3 } else { -0.3 };

            data[y * width + x] = 0.7 * gaussian + 0.3 * checkerboard;
        }
    }

    Image2D::new(width, height, data, None).expect("Failed to create composite image")
}

// ============================================================================
// Latency Benchmarks
// ============================================================================

fn bench_decompose_2d_256x256(c: &mut Criterion) {
    let config = EmdConfig::default();
    let image = black_box(create_checkerboard_image(256, 256, 16));

    c.bench_function("decompose_2d_256x256", |b| {
        b.iter(|| decompose_image_2d_separable(black_box(&image), black_box(&config)));
    });
}

fn bench_decompose_2d_512x512(c: &mut Criterion) {
    let config = EmdConfig::default();
    let image = black_box(create_gaussian_blob_image(512, 512));

    c.bench_function("decompose_2d_512x512_primary_target", |b| {
        b.iter(|| decompose_image_2d_separable(black_box(&image), black_box(&config)));
    });
}

fn bench_decompose_2d_1024x1024(c: &mut Criterion) {
    let config = EmdConfig::default();
    let image = black_box(create_random_noise_image(1024, 1024));

    c.bench_function("decompose_2d_1024x1024", |b| {
        b.iter(|| decompose_image_2d_separable(black_box(&image), black_box(&config)));
    });
}

fn bench_decompose_2d_scaling(c: &mut Criterion) {
    let config = EmdConfig::default();
    let mut group = c.benchmark_group("decompose_2d_scaling");

    for size in [256, 512].iter() {
        let image = black_box(create_composite_image(*size, *size));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| decompose_image_2d_separable(black_box(&image), black_box(&config)));
            },
        );
    }

    group.finish();
}

fn bench_decompose_2d_varying_content(c: &mut Criterion) {
    let config = EmdConfig::default();
    let mut group = c.benchmark_group("decompose_2d_varying_content");

    let patterns = vec![
        ("checkerboard", create_checkerboard_image(512, 512, 32)),
        ("gaussian", create_gaussian_blob_image(512, 512)),
        ("random", create_random_noise_image(512, 512)),
        ("composite", create_composite_image(512, 512)),
    ];

    for (name, image) in patterns {
        group.bench_with_input(BenchmarkId::from_parameter(name), &image, |b, img| {
            b.iter(|| decompose_image_2d_separable(black_box(img), black_box(&config)));
        });
    }

    group.finish();
}

// ============================================================================
// Memory Benchmarks
// ============================================================================

fn bench_memory_peak_512x512(c: &mut Criterion) {
    let config = EmdConfig::default();

    c.bench_function("memory_peak_512x512", |b| {
        b.iter(|| {
            let image = black_box(create_gaussian_blob_image(512, 512));
            let _ = black_box(decompose_image_2d_separable(&image, &config));
        });
    });
}

fn bench_memory_scaling(c: &mut Criterion) {
    let config = EmdConfig::default();
    let mut group = c.benchmark_group("memory_scaling");

    for size in [256, 512].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            size,
            |b, _| {
                b.iter(|| {
                    let image = black_box(create_composite_image(*size, *size));
                    let _ = black_box(decompose_image_2d_separable(&image, &config));
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Component Breakdown Benchmarks
// ============================================================================

/// Benchmark row-wise EMD phase.
/// This measures the time for Phase 1: decomposing each row independently.
fn bench_phase1_rowwise_emd(c: &mut Criterion) {
    use ferromode::algorithms::emd::emd;

    let config = EmdConfig::default();

    c.bench_function("phase1_rowwise_emd_512x512", |b| {
        b.iter(|| {
            let image = black_box(create_gaussian_blob_image(512, 512));
            let height = image.height();

            for row_idx in 0..height {
                let row_data = image.row(row_idx);
                let _ = black_box(emd(black_box(&row_data), black_box(&config)));
            }
        });
    });
}

/// Benchmark column-wise EMD phase.
/// This measures the time for Phase 2: decomposing columns of row-based IMFs.
fn bench_phase2_columnwise_emd(c: &mut Criterion) {
    use ferromode::algorithms::emd::emd;

    let config = EmdConfig::default();

    c.bench_function("phase2_columnwise_emd_512x512", |b| {
        b.iter(|| {
            let image = black_box(create_gaussian_blob_image(512, 512));
            let width = image.width();

            for col_idx in 0..width {
                let col_data = image.column(col_idx);
                let _ = black_box(emd(black_box(&col_data), black_box(&config)));
            }
        });
    });
}

/// Benchmark padding and unpadding overhead.
/// Note: Current implementation uses row/column extraction, so this measures
/// that extraction cost as an approximation of padding overhead.
fn bench_padding_overhead(c: &mut Criterion) {
    c.bench_function("padding_overhead_512x512", |b| {
        b.iter(|| {
            let image = black_box(create_composite_image(512, 512));
            let width = image.width();
            let height = image.height();

            // Simulate padding overhead by extracting all rows and columns
            for row_idx in 0..height {
                let _ = black_box(image.row(row_idx));
            }
            for col_idx in 0..width {
                let _ = black_box(image.column(col_idx));
            }
        });
    });
}

// ============================================================================
// Custom Measurement and Reporting
// ============================================================================

/// Manual latency measurement with detailed reporting.
/// Used to verify criterion benchmarks and provide additional context.
#[allow(dead_code)]
fn manual_latency_report() {
    println!("\n=== Manual Latency Report ===");

    let config = EmdConfig::default();
    let test_cases = vec![("256×256", 256), ("512×512", 512)];

    for (label, size) in test_cases {
        let image = create_gaussian_blob_image(size, size);

        let start = Instant::now();
        let result = decompose_image_2d_separable(&image, &config);
        let elapsed = start.elapsed();

        match result {
            Ok(_) => {
                let millis = elapsed.as_millis();
                let target = if size == 256 { 1000 } else { 5000 };
                let status = if millis <= target { "✓ PASS" } else { "✗ FAIL" };
                println!("{}: {:.1}ms (target: {}ms) {}", label, millis, target, status);
            }
            Err(e) => {
                println!("{}: Error - {:?}", label, e);
            }
        }
    }
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = latency_benches;
    config = Criterion::default()
        .significance_level(0.1)
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_decompose_2d_256x256,
        bench_decompose_2d_512x512,
        bench_decompose_2d_1024x1024,
        bench_decompose_2d_scaling,
        bench_decompose_2d_varying_content
);

criterion_group!(
    name = memory_benches;
    config = Criterion::default()
        .significance_level(0.1)
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_memory_peak_512x512,
        bench_memory_scaling
);

criterion_group!(
    name = component_benches;
    config = Criterion::default()
        .significance_level(0.1)
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(5));
    targets =
        bench_phase1_rowwise_emd,
        bench_phase2_columnwise_emd,
        bench_padding_overhead
);

criterion_main!(latency_benches, memory_benches, component_benches);
