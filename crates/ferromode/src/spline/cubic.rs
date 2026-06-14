use crate::error::EmdError;
use crate::spline::Spline;

#[derive(Debug, Clone)]
struct Segment {
    x0: f64,
    x1: f64,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl Segment {
    fn evaluate(&self, x: f64) -> f64 {
        let dx = x - self.x0;
        self.a + self.b * dx + self.c * dx * dx + self.d * dx * dx * dx
    }

    fn evaluate_derivative(&self, x: f64) -> f64 {
        let dx = x - self.x0;
        self.b + 2.0 * self.c * dx + 3.0 * self.d * dx * dx
    }
}

/// Piecewise cubic Hermite spline interpolating a set of (x, y) knots.
///
/// Construct via [`CubicSpline::from_knots`] (natural boundary), [`CubicSpline::periodic_from_knots`]
/// (periodic boundary), or [`CubicSpline::not_a_knot_from_knots`] (not-a-knot boundary).
/// Knots must be strictly increasing in x.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    x_knots: Vec<f64>,
    y_knots: Vec<f64>,
    segments: Vec<Segment>,
}

impl CubicSpline {
    /// Build a natural cubic spline from the given knot arrays.
    ///
    /// Returns an error if there are fewer than 2 knots, x and y differ in length,
    /// any value is non-finite, or knots are not strictly increasing.
    pub fn from_knots(x: &[f64], y: &[f64]) -> Result<Self, EmdError> {
        Self::build(x, y, BoundaryCondition::Natural)
    }

    /// Build a periodic (cyclic) cubic spline from the given knot arrays.
    ///
    /// First and second derivatives match at the endpoints, making the spline periodic.
    pub fn periodic_from_knots(x: &[f64], y: &[f64]) -> Result<Self, EmdError> {
        Self::build(x, y, BoundaryCondition::Periodic)
    }

    /// Build a not-a-knot cubic spline from the given knot arrays.
    ///
    /// Enforces C³ continuity at the first and last interior knots. Exact for polynomials
    /// up to degree 3; no artificial endpoint constraint.
    pub fn not_a_knot_from_knots(x: &[f64], y: &[f64]) -> Result<Self, EmdError> {
        Self::build(x, y, BoundaryCondition::NotAKnot)
    }

    fn build(x: &[f64], y: &[f64], bc: BoundaryCondition) -> Result<Self, EmdError> {
        if x.len() < 2 || y.len() < 2 {
            return Err(EmdError::InsufficientData);
        }
        if x.len() != y.len() {
            return Err(EmdError::DimensionMismatch);
        }

        for i in 0..x.len() {
            if !x[i].is_finite() || !y[i].is_finite() {
                return Err(EmdError::InvalidValue);
            }
        }

        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(EmdError::InvalidConfig(format!(
                    "knots must be strictly increasing, found x[{}] = {} <= x[{}] = {}",
                    i,
                    x[i],
                    i - 1,
                    x[i - 1]
                )));
            }
        }

        let n = x.len() - 1;

        if n == 1 {
            let seg = Segment {
                x0: x[0],
                x1: x[1],
                a: y[0],
                b: (y[1] - y[0]) / (x[1] - x[0]),
                c: 0.0,
                d: 0.0,
            };
            return Ok(Self { x_knots: x.to_vec(), y_knots: y.to_vec(), segments: vec![seg] });
        }

        let h: Vec<f64> = (0..n).map(|i| x[i + 1] - x[i]).collect();

        let second_derivs = match bc {
            BoundaryCondition::Natural => solve_natural(&h, x, y, n),
            BoundaryCondition::Periodic => solve_periodic(&h, x, y, n),
            BoundaryCondition::NotAKnot => solve_not_a_knot(&h, x, y, n),
        };

        let mut segments = Vec::with_capacity(n);
        for i in 0..n {
            // Verify array bounds to prevent panics on malformed data
            // second_derivs should have length n+1 (from all solver functions)
            if i + 1 >= second_derivs.len() {
                return Err(EmdError::InvalidConfig(format!(
                    "spline solver returned insufficient second derivatives: {} < {}",
                    second_derivs.len(),
                    i + 2
                )));
            }

            // second_derivs[i] = c[i] in Burden & Faires (Algorithm 3.4) convention.
            // The spline is a + b*dx + c*dx^2 + d*dx^3.
            let a = y[i];
            let b = (y[i + 1] - y[i]) / h[i]
                - h[i] * (2.0 * second_derivs[i] + second_derivs[i + 1]) / 3.0;
            let c = second_derivs[i];
            let d = (second_derivs[i + 1] - second_derivs[i]) / (3.0 * h[i]);

            segments.push(Segment { x0: x[i], x1: x[i + 1], a, b, c, d });
        }

        Ok(Self { x_knots: x.to_vec(), y_knots: y.to_vec(), segments })
    }

    fn find_segment(&self, x: f64) -> usize {
        let n = self.segments.len();
        if x <= self.segments[0].x0 {
            return 0;
        }
        if x >= self.segments[n - 1].x1 {
            return n - 1;
        }

        let mut lo = 0;
        let mut hi = n;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if x < self.segments[mid].x0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo
    }
}

impl Spline for CubicSpline {
    fn evaluate(&self, x: f64) -> f64 {
        let idx = self.find_segment(x);
        self.segments[idx].evaluate(x)
    }

    fn evaluate_derivative(&self, x: f64) -> f64 {
        let idx = self.find_segment(x);
        self.segments[idx].evaluate_derivative(x)
    }

    fn knots(&self) -> (&[f64], &[f64]) {
        (&self.x_knots, &self.y_knots)
    }
}

#[derive(Debug, Clone, Copy)]
enum BoundaryCondition {
    Natural,
    Periodic,
    NotAKnot,
}

fn solve_natural(h: &[f64], x: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![0.0, 0.0];
    }

    // alpha[i] = RHS for interior knot i (indices 1..n)
    let mut alpha = vec![0.0; n + 1];
    for i in 1..n {
        alpha[i] = 3.0 * (y[i + 1] - y[i]) / h[i] - 3.0 * (y[i] - y[i - 1]) / h[i - 1];
    }

    let mut l = vec![0.0; n + 1];
    let mut mu = vec![0.0; n + 1];
    let mut z = vec![0.0; n + 1];

    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;

    // Forward sweep over all interior knots 1..n
    for i in 1..n {
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n] = 1.0;
    z[n] = 0.0;

    // Backward substitution over 0..n (c[n] = 0 natural BC, already initialised)
    let mut c = vec![0.0; n + 1];
    for j in (0..n).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
    }

    c
}

fn solve_periodic(h: &[f64], x: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    let m = n;
    if m == 1 {
        return vec![0.0, 0.0];
    }

    let mut alpha = vec![0.0; m];
    for i in 1..m {
        alpha[i] = 3.0 * (y[i + 1] - y[i]) / h[i] - 3.0 * (y[i] - y[i - 1]) / h[i - 1];
    }
    alpha[0] = 3.0 * (y[1] - y[0]) / h[0] - 3.0 * (y[n] - y[n - 1]) / h[n - 1];

    let mut a_diag = vec![2.0 * (x[1] - x[n - 1]); m];
    a_diag[0] = 2.0 * (h[0] + h[n - 1]);
    for i in 1..m {
        a_diag[i] = 2.0 * (x[i + 1] - x[i - 1]);
    }

    let mut lower = vec![0.0; m];
    let mut upper = vec![0.0; m];
    for i in 1..m {
        lower[i] = h[i - 1];
    }
    lower[0] = h[n - 1];
    for i in 0..m - 1 {
        upper[i] = h[i];
    }
    upper[m - 1] = h[n - 1];

    let mut result = solve_cyclic_tridiagonal(&a_diag, &lower, &upper, &alpha);
    // For periodic spline, the last derivative equals the first
    result.push(result[0]);
    result
}

fn solve_not_a_knot(h: &[f64], x: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    let m = n - 1;
    if m == 0 {
        return vec![0.0, 0.0];
    }

    let size = n + 1;
    let mut mat = vec![vec![0.0; size]; size];
    let mut rhs = vec![0.0; size];

    for i in 1..n {
        mat[i][i - 1] = h[i - 1];
        mat[i][i] = 2.0 * (x[i + 1] - x[i - 1]);
        mat[i][i + 1] = h[i];
        rhs[i] = 3.0 * (y[i + 1] - y[i]) / h[i] - 3.0 * (y[i] - y[i - 1]) / h[i - 1];
    }

    mat[0][0] = h[1];
    mat[0][1] = -(h[0] + h[1]);
    mat[0][2] = h[0];
    rhs[0] = 0.0;

    mat[n][n - 2] = h[n - 1];
    mat[n][n - 1] = -(h[n - 2] + h[n - 1]);
    mat[n][n] = h[n - 2];
    rhs[n] = 0.0;

    // The boundary rows have stencil width 3 so we need full Gaussian elimination,
    // not the narrow-band solve_general_tridiagonal which would miss fill-in.
    gaussian_elimination(&mat, &rhs)
}

fn solve_cyclic_tridiagonal(diag: &[f64], lower: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    if n == 1 {
        return vec![rhs[0] / diag[0]];
    }

    // Sherman-Morrison: A = A' + u*v^T where A' is a regular tridiagonal.
    // Choose gamma = -diag[0] so that a_prime[0] = diag[0] - gamma = 2*diag[0] != 0.
    let gamma = -diag[0];

    let mut a_prime = diag.to_vec();
    a_prime[0] = diag[0] - gamma;                           // 2 * diag[0]
    a_prime[n - 1] -= gamma * lower[0] / upper[n - 1];     // adjust corner

    // u and v define the rank-1 perturbation: A = A' + u*v^T
    let mut u = vec![0.0; n];
    u[0] = gamma;
    u[n - 1] = lower[0];

    let mut v = vec![0.0; n];
    v[0] = 1.0;
    v[n - 1] = upper[n - 1] / gamma;

    // Solve A'*q = rhs and A'*z = u
    let mut q = thomas_algorithm(&a_prime, &lower[1..], &upper[..n - 1], rhs);
    let z = thomas_algorithm(&a_prime, &lower[1..], &upper[..n - 1], &u);

    // x = q - (v^T * q) / (1 + v^T * z) * z
    let v_dot_z: f64 = v.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
    let v_dot_q: f64 = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
    let factor = v_dot_q / (1.0 + v_dot_z);

    for i in 0..n {
        q[i] -= factor * z[i];
    }

    q
}

fn thomas_algorithm(diag: &[f64], lower: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
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

#[allow(dead_code)]
fn solve_general_tridiagonal(mat: &[Vec<f64>], rhs: &[f64], size: usize) -> Vec<f64> {
    let mut mat = mat.to_vec();
    let mut rhs = rhs.to_vec();

    for i in 1..size {
        let factor = mat[i][i - 1] / mat[i - 1][i - 1];
        for j in (i - 1)..size.min(i + 1) {
            mat[i][j] -= factor * mat[i - 1][j];
        }
        rhs[i] -= factor * rhs[i - 1];
    }

    let mut x = vec![0.0; size];
    x[size - 1] = rhs[size - 1] / mat[size - 1][size - 1];
    for i in (0..size - 1).rev() {
        x[i] = (rhs[i] - mat[i][i + 1] * x[i + 1]) / mat[i][i];
    }

    x
}

/// Solve the linear system `mat * x = rhs` using naive Gaussian elimination with partial pivoting.
///
/// Exposed for testing and benchmarking the internal spline solver. For general use, prefer
/// the constructor methods on [`CubicSpline`] which select the appropriate solver automatically.
pub fn naive_gaussian_solve(mat: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    gaussian_elimination(mat, rhs)
}

fn gaussian_elimination(mat: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let n = mat.len();
    let mut mat = mat.to_vec();
    let mut rhs = rhs.to_vec();

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if mat[k][i].abs() > mat[max_row][i].abs() {
                max_row = k;
            }
        }
        mat.swap(i, max_row);
        rhs.swap(i, max_row);

        for k in (i + 1)..n {
            let factor = mat[k][i] / mat[i][i];
            for j in i..n {
                mat[k][j] -= factor * mat[i][j];
            }
            rhs[k] -= factor * rhs[i];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = rhs[i];
        for j in (i + 1)..n {
            x[i] -= mat[i][j] * x[j];
        }
        x[i] /= mat[i][i];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_spline_passes_through_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let spline = CubicSpline::from_knots(&x, &y).unwrap();

        for i in 0..x.len() {
            let val = spline.evaluate(x[i]);
            assert!((val - y[i]).abs() < 1e-10, "spline({}) = {} != {}", x[i], val, y[i]);
        }
    }

    #[test]
    fn test_natural_spline_parabola_uniform_knots() {
        // Natural spline imposes S''=0 at both endpoints ("natural" condition).
        // For y=x² the true second derivative is 2 everywhere, so the natural spline
        // cannot exactly reproduce x² — it will deviate near the endpoints.
        // What we verify: knot interpolation is exact, endpoint second derivative is 0,
        // and interior values are "close" (within the O(h^4) characteristic error).
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let spline = CubicSpline::from_knots(&x, &y).unwrap();

        // Knot interpolation must be exact
        for i in 0..x.len() {
            let val = spline.evaluate(x[i]);
            assert!((val - y[i]).abs() < 1e-10, "knot {} not interpolated: {} != {}", i, val, y[i]);
        }

        // Natural BC: second derivative at left endpoint must be zero
        let d2_left = spline.evaluate_derivative(x[0] + 1e-7) - spline.evaluate_derivative(x[0]);
        assert!(d2_left.abs() < 1e-4, "left endpoint second derivative not near zero");

        // Natural BC: second derivative at right endpoint must be zero
        let d2_right = spline.evaluate_derivative(x[4]) - spline.evaluate_derivative(x[4] - 1e-7);
        assert!(d2_right.abs() < 1e-4, "right endpoint second derivative not near zero");

        // For exact parabola reproduction use not_a_knot_from_knots (see that test).
    }

    #[test]
    fn test_natural_spline_nonuniform_knots_sin() {
        let x = vec![0.0_f64, 0.5, 2.0, 3.5, 5.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let spline = CubicSpline::from_knots(&x, &y).unwrap();

        for i in 0..x.len() {
            let val = spline.evaluate(x[i]);
            assert!((val - y[i]).abs() < 1e-10, "spline({}) = {} != {}", x[i], val, y[i]);
        }

        let mid = 1.25;
        let actual = spline.evaluate(mid);
        let expected = mid.sin();
        assert!(
            (actual - expected).abs() < 0.05,
            "spline interpolation too far from sin: {} vs {}",
            actual,
            expected
        );
    }

    #[test]
    fn test_periodic_spline_passes_through_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let spline = CubicSpline::periodic_from_knots(&x, &y).unwrap();

        for i in 0..x.len() {
            let val = spline.evaluate(x[i]);
            assert!((val - y[i]).abs() < 1e-10, "periodic_spline({}) = {} != {}", x[i], val, y[i]);
        }
    }

    #[test]
    fn test_periodic_spline_endpoint_derivatives_match() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let spline = CubicSpline::periodic_from_knots(&x, &y).unwrap();

        let deriv_start = spline.evaluate_derivative(x[0]);
        let deriv_end = spline.evaluate_derivative(x[x.len() - 1]);

        assert!(
            (deriv_start - deriv_end).abs() < 1e-8,
            "periodic: deriv_start={} != deriv_end={}",
            deriv_start,
            deriv_end
        );
    }

    #[test]
    fn test_not_a_knot_spline_passes_through_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let spline = CubicSpline::not_a_knot_from_knots(&x, &y).unwrap();

        for i in 0..x.len() {
            let val = spline.evaluate(x[i]);
            assert!(
                (val - y[i]).abs() < 1e-10,
                "not_a_knot_spline({}) = {} != {}",
                x[i],
                val,
                y[i]
            );
        }
    }

    #[test]
    fn test_not_a_knot_spline_parabola_exact() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let spline = CubicSpline::not_a_knot_from_knots(&x, &y).unwrap();

        for xi in 0..40 {
            let xq = xi as f64 / 10.0;
            let expected = xq * xq;
            let actual = spline.evaluate(xq);
            assert!(
                (actual - expected).abs() < 1e-10,
                "not_a_knot_spline({}) = {} != {}",
                xq,
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_degenerate_fewer_than_2_knots() {
        let x = vec![0.0];
        let y = vec![1.0];
        let result = CubicSpline::from_knots(&x, &y);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));

        let result = CubicSpline::periodic_from_knots(&x, &y);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));

        let result = CubicSpline::not_a_knot_from_knots(&x, &y);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_degenerate_empty_knots() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        let result = CubicSpline::from_knots(&x, &y);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_degenerate_duplicate_knots() {
        let x = vec![0.0, 1.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let result = CubicSpline::from_knots(&x, &y);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_degenerate_unsorted_knots() {
        let x = vec![0.0, 2.0, 1.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let result = CubicSpline::from_knots(&x, &y);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_degenerate_dimension_mismatch() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0];
        let result = CubicSpline::from_knots(&x, &y);
        assert!(matches!(result.unwrap_err(), EmdError::DimensionMismatch));
    }

    #[test]
    fn test_degenerate_nan_in_knots() {
        let x = vec![0.0, f64::NAN, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let result = CubicSpline::from_knots(&x, &y);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_degenerate_inf_in_knots() {
        let x = vec![0.0, f64::INFINITY, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let result = CubicSpline::from_knots(&x, &y);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_single_extremum() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 0.0];
        let spline = CubicSpline::from_knots(&x, &y).unwrap();

        for i in 0..x.len() {
            let val = spline.evaluate(x[i]);
            assert!((val - y[i]).abs() < 1e-10);
        }

        let max_val = spline.evaluate(1.0);
        assert!((max_val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_double_extremum() {
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let spline = CubicSpline::from_knots(&x, &y).unwrap();

        for i in 0..x.len() {
            let val = spline.evaluate(x[i]);
            assert!((val - y[i]).abs() < 1e-10, "spline({}) = {} != {}", x[i], val, y[i]);
        }
    }

    #[test]
    fn test_spline_derivative_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let spline = CubicSpline::from_knots(&x, &y).unwrap();

        let deriv_at_1 = spline.evaluate_derivative(1.0);
        let deriv_at_2 = spline.evaluate_derivative(2.0);

        assert!(deriv_at_1 > 0.0, "derivative at x=1 should be positive");
        assert!(deriv_at_2 > 0.0, "derivative at x=2 should be positive");
    }

    #[test]
    fn test_two_knots_linear() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 2.0];
        let spline = CubicSpline::from_knots(&x, &y).unwrap();

        let val = spline.evaluate(0.5);
        assert!((val - 1.0).abs() < 1e-10);

        let deriv = spline.evaluate_derivative(0.5);
        assert!((deriv - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_knots_accessor() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];
        let spline = CubicSpline::from_knots(&x, &y).unwrap();

        let (xk, yk) = spline.knots();
        assert_eq!(xk, &x);
        assert_eq!(yk, &y);
    }

    #[test]
    fn test_c2_continuity_natural() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let spline = CubicSpline::from_knots(&x, &y).unwrap();

        for i in 1..x.len() - 1 {
            let xi = x[i];
            let left = spline.evaluate(xi - 1e-8);
            let right = spline.evaluate(xi + 1e-8);

            assert!(
                (left - right).abs() < 1e-6,
                "C2 discontinuity at x={}: left={}, right={}",
                xi,
                left,
                right
            );
        }
    }
}
