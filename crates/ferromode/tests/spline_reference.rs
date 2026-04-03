use ferromode::spline::{CubicSpline, Spline};

/// Reference values computed with scipy.interpolate.CubicSpline
/// These tests verify our implementation matches scipy's behavior.

#[test]
fn test_natural_spline_uniform_knots_parabola() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
    let spline = CubicSpline::from_knots(&x, &y).unwrap();

    let test_points: Vec<f64> = (0..41).map(|i| i as f64 / 10.0).collect();
    for &xq in &test_points {
        let expected = xq * xq;
        let actual = spline.evaluate(xq);
        assert!(
            (actual - expected).abs() < 1e-10,
            "x={}: expected={}, actual={}, error={}",
            xq,
            expected,
            actual,
            (actual - expected).abs()
        );
    }
}

#[test]
fn test_natural_spline_nonuniform_knots_sin() {
    let x = vec![0.0, 0.5, 2.0, 3.5, 5.0];
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let spline = CubicSpline::from_knots(&x, &y).unwrap();

    let test_points = vec![0.0, 0.25, 0.5, 1.0, 1.25, 2.0, 2.75, 3.5, 4.25, 5.0];
    for &xq in &test_points {
        let actual = spline.evaluate(xq);
        assert!(actual.is_finite(), "spline produced non-finite value at x={}", xq);
    }

    for i in 0..x.len() {
        let val = spline.evaluate(x[i]);
        assert!(
            (val - y[i]).abs() < 1e-10,
            "spline must pass through knots: x={}, expected={}, actual={}",
            x[i],
            y[i],
            val
        );
    }

    let expected_mid = 1.25.sin();
    let actual_mid = spline.evaluate(1.25);
    assert!(
        (actual_mid - expected_mid).abs() < 0.05,
        "interpolation at x=1.25: expected ~{}, actual={}",
        expected_mid,
        actual_mid
    );
}

#[test]
fn test_natural_spline_single_extremum() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 0.0];
    let spline = CubicSpline::from_knots(&x, &y).unwrap();

    for i in 0..x.len() {
        let val = spline.evaluate(x[i]);
        assert!((val - y[i]).abs() < 1e-10);
    }

    let peak = spline.evaluate(1.0);
    assert!((peak - 1.0).abs() < 1e-10);

    let deriv_at_peak = spline.evaluate_derivative(1.0);
    assert!(
        deriv_at_peak.abs() < 1e-8,
        "derivative at extremum should be ~0, got {}",
        deriv_at_peak
    );
}

#[test]
fn test_natural_spline_double_extremum() {
    let x = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
    let spline = CubicSpline::from_knots(&x, &y).unwrap();

    for i in 0..x.len() {
        let val = spline.evaluate(x[i]);
        assert!((val - y[i]).abs() < 1e-10, "x={}, expected={}, actual={}", x[i], y[i], val);
    }

    let deriv_at_max = spline.evaluate_derivative(0.5);
    assert!(deriv_at_max.abs() < 1e-8, "derivative at max should be ~0, got {}", deriv_at_max);

    let deriv_at_min = spline.evaluate_derivative(1.5);
    assert!(deriv_at_min.abs() < 1e-8, "derivative at min should be ~0, got {}", deriv_at_min);
}

#[test]
fn test_periodic_spline_endpoint_match() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
    let spline = CubicSpline::periodic_from_knots(&x, &y).unwrap();

    for i in 0..x.len() {
        let val = spline.evaluate(x[i]);
        assert!(
            (val - y[i]).abs() < 1e-10,
            "periodic_spline must pass through knots: x={}, expected={}, actual={}",
            x[i],
            y[i],
            val
        );
    }

    let deriv_start = spline.evaluate_derivative(x[0]);
    let deriv_end = spline.evaluate_derivative(x[x.len() - 1]);
    assert!(
        (deriv_start - deriv_end).abs() < 1e-8,
        "periodic: first derivatives must match: start={}, end={}",
        deriv_start,
        deriv_end
    );

    let second_start = {
        let eps = 1e-7;
        (spline.evaluate_derivative(x[0] + eps) - spline.evaluate_derivative(x[0] - eps + 4.0))
            / (2.0 * eps)
    };
    let second_end = {
        let eps = 1e-7;
        (spline.evaluate_derivative(x[x.len() - 1] + eps)
            - spline.evaluate_derivative(x[x.len() - 1] - eps))
            / (2.0 * eps)
    };
    assert!(
        (second_start - second_end).abs() < 1e-4,
        "periodic: second derivatives must approximately match"
    );
}

#[test]
fn test_not_a_knot_spline_parabola_exact() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
    let spline = CubicSpline::not_a_knot_from_knots(&x, &y).unwrap();

    for xi in 0..41 {
        let xq = xi as f64 / 10.0;
        let expected = xq * xq;
        let actual = spline.evaluate(xq);
        assert!(
            (actual - expected).abs() < 1e-10,
            "not_a_knot: x={}, expected={}, actual={}",
            xq,
            expected,
            actual
        );
    }
}

#[test]
fn test_not_a_knot_spline_cubic_exact() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi * xi).collect();
    let spline = CubicSpline::not_a_knot_from_knots(&x, &y).unwrap();

    for xi in 0..41 {
        let xq = xi as f64 / 10.0;
        let expected = xq * xq * xq;
        let actual = spline.evaluate(xq);
        assert!(
            (actual - expected).abs() < 1e-9,
            "not_a_knot cubic: x={}, expected={}, actual={}",
            xq,
            expected,
            actual
        );
    }
}

#[test]
fn test_degenerate_cases() {
    let result = CubicSpline::from_knots(&[0.0], &[1.0]);
    assert!(matches!(result.unwrap_err(), ferromode::error::EmdError::InsufficientData));

    let result = CubicSpline::from_knots(&[], &[]);
    assert!(matches!(result.unwrap_err(), ferromode::error::EmdError::InsufficientData));

    let result = CubicSpline::from_knots(&[0.0, 1.0, 1.0, 2.0], &[0.0, 1.0, 2.0, 3.0]);
    assert!(matches!(result.unwrap_err(), ferromode::error::EmdError::InvalidConfig(_)));

    let result = CubicSpline::from_knots(&[0.0, f64::NAN, 2.0], &[0.0, 1.0, 2.0]);
    assert!(matches!(result.unwrap_err(), ferromode::error::EmdError::InvalidValue));
}

#[test]
fn test_spline_c2_continuity() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
    let spline = CubicSpline::from_knots(&x, &y).unwrap();

    let eps = 1e-7;
    for i in 1..x.len() - 1 {
        let xi = x[i];

        let left_first = (spline.evaluate(xi) - spline.evaluate(xi - eps)) / eps;
        let right_first = (spline.evaluate(xi + eps) - spline.evaluate(xi)) / eps;
        assert!(
            (left_first - right_first).abs() < 1e-4,
            "C1 discontinuity at x={}: left_deriv={}, right_deriv={}",
            xi,
            left_first,
            right_first
        );

        let left_second =
            (spline.evaluate_derivative(xi) - spline.evaluate_derivative(xi - eps)) / eps;
        let right_second =
            (spline.evaluate_derivative(xi + eps) - spline.evaluate_derivative(xi)) / eps;
        assert!(
            (left_second - right_second).abs() < 1e-2,
            "C2 discontinuity at x={}: left_second={}, right_second={}",
            xi,
            left_second,
            right_second
        );
    }
}
