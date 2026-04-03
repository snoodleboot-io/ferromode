use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// ---------------------------------------------------------------------------
// Benchmark group: signal_creation
// ---------------------------------------------------------------------------

fn bench_signal_creation(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![100, 1_000, 10_000];
    let mut group = c.benchmark_group("signal_creation");

    for &size in &sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.01).sin()).collect();
                black_box(data)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group: extrema_detection
// ---------------------------------------------------------------------------

fn bench_extrema_detection(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![100, 1_000, 10_000];
    let mut group = c.benchmark_group("extrema_detection");

    for &size in &sizes {
        // Pre-generate signal data so we benchmark only the detection logic
        let signal: Vec<f64> = (0..size).map(|i| (i as f64 * 0.01).sin()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &signal, |b, signal| {
            b.iter(|| find_local_extrema(black_box(signal)));
        });
    }

    group.finish();
}

/// Find local maxima and minima indices in a signal slice.
/// Returns a tuple of (maxima_indices, minima_indices).
fn find_local_extrema(signal: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut maxima = Vec::new();
    let mut minima = Vec::new();

    for i in 1..signal.len().saturating_sub(1) {
        let prev = signal[i - 1];
        let curr = signal[i];
        let next = signal[i + 1];

        if curr > prev && curr > next {
            maxima.push(i);
        } else if curr < prev && curr < next {
            minima.push(i);
        }
    }

    (maxima, minima)
}

// ---------------------------------------------------------------------------
// Benchmark group: spline_interpolation
// ---------------------------------------------------------------------------

fn bench_spline_interpolation(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![10, 50, 100];
    let mut group = c.benchmark_group("spline_interpolation");

    for &size in &sizes {
        // Pre-generate knot points
        let knots: Vec<f64> = (0..size).map(|i| (i as f64 * 0.1).sin()).collect();
        let x_points: Vec<f64> = (0..size).map(|i| i as f64).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(x_points, knots),
            |b, (xs, ys)| {
                b.iter(|| {
                    // Placeholder: linear interpolation until cubic spline is implemented
                    linear_interpolate(black_box(xs), black_box(ys))
                });
            },
        );
    }

    group.finish();
}

/// Simple linear interpolation placeholder.
/// Returns interpolated values at query points between the knots.
fn linear_interpolate(x_points: &[f64], y_points: &[f64]) -> Vec<f64> {
    if x_points.len() < 2 {
        return y_points.to_vec();
    }

    let n_queries = x_points.len() * 2 - 1;
    let mut result = Vec::with_capacity(n_queries);

    for i in 0..x_points.len().saturating_sub(1) {
        let x0 = x_points[i];
        let x1 = x_points[i + 1];
        let y0 = y_points[i];
        let y1 = y_points[i + 1];

        result.push(y0);

        // Midpoint interpolation
        let xm = (x0 + x1) / 2.0;
        let t = (xm - x0) / (x1 - x0);
        let ym = y0 * (1.0 - t) + y1 * t;
        result.push(ym);
    }

    // Last point
    if let Some(&last) = y_points.last() {
        result.push(last);
    }

    result
}

// ---------------------------------------------------------------------------
// Criterion macros
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_signal_creation,
    bench_extrema_detection,
    bench_spline_interpolation,
);
criterion_main!(benches);
