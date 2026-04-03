use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferromode::spline::cubic::naive_gaussian_solve;
use ferromode::spline::{CubicSpline, Spline};

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
// Benchmark group: spline_solve_comparison — Thomas O(n) vs naive O(n³)
// ---------------------------------------------------------------------------

fn bench_spline_solve_comparison(c: &mut Criterion) {
    let sizes: Vec<usize> = vec![10, 100, 1000];
    let mut group = c.benchmark_group("spline_solve_comparison");

    for &size in &sizes {
        let x: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

        // Build tridiagonal system for natural spline
        let n = size - 1;
        let h: Vec<f64> = (0..n).map(|i| x[i + 1] - x[i]).collect();

        // Thomas algorithm input (tridiagonal form)
        let thomas_lower: Vec<f64> = (1..n - 1).map(|i| h[i - 1]).collect();
        let thomas_diag: Vec<f64> = (1..n - 1).map(|i| 2.0 * (h[i - 1] + h[i])).collect();
        let thomas_upper: Vec<f64> = (1..n - 1).map(|i| h[i]).collect();
        let thomas_rhs: Vec<f64> = (1..n - 1)
            .map(|i| 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]))
            .collect();

        // Naive Gaussian elimination input (full matrix)
        let sys_size = n - 1;
        let naive_mat: Vec<Vec<f64>> = (0..sys_size)
            .map(|i| {
                let mut row = vec![0.0; sys_size];
                if i > 0 {
                    row[i - 1] = h[i - 1];
                }
                row[i] = 2.0 * (h[i] + h.get(i + 1).unwrap_or(&h[i]));
                if i < sys_size - 1 {
                    row[i + 1] = h[i];
                }
                row
            })
            .collect();
        let naive_rhs: Vec<f64> = (1..n - 1)
            .map(|i| 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]))
            .collect();

        // Benchmark Thomas algorithm — O(n)
        group.bench_with_input(
            BenchmarkId::new("thomas_O_n", size),
            &(&thomas_lower, &thomas_diag, &thomas_upper, &thomas_rhs),
            |b, (lower, diag, upper, rhs)| {
                b.iter(|| {
                    solve_thomas(
                        black_box(lower),
                        black_box(diag),
                        black_box(upper),
                        black_box(rhs),
                    )
                });
            },
        );

        // Benchmark naive Gaussian elimination — O(n³)
        // Only for small sizes to avoid excessive runtime
        if size <= 100 {
            group.bench_with_input(
                BenchmarkId::new("naive_gaussian_O_n3", size),
                &(naive_mat, naive_rhs),
                |b, (mat, rhs)| {
                    b.iter(|| naive_gaussian_solve(black_box(mat), black_box(rhs)));
                },
            );
        }
    }

    group.finish();
}

/// Thomas algorithm implementation for benchmarking.
fn solve_thomas(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - lower[i - 1] * c_prime[i - 1];
        if i < n - 1 {
            c_prime[i] = upper[i] / denom;
        }
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    x
}

// ---------------------------------------------------------------------------
// Criterion macros
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_signal_creation,
    bench_extrema_detection,
    bench_spline_interpolation,
    bench_spline_solve_comparison,
);
criterion_main!(benches);
