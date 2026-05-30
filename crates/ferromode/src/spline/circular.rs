/// Circular (Zhao-Huang) spline for periodic signals.
///
/// Wraps a periodic-end-condition cubic spline so that any evaluation point
/// is first mapped into `[x_first_knot, x_last_knot)` via modular arithmetic
/// before being handed to the underlying segments.  This is the correct
/// interpolant for the Zhao-Huang `[signal | signal_reversed]` palindrome
/// boundary: the cyclic tridiagonal system enforces S''(first) = S''(last) and
/// S'(first) = S'(last), so the knot range is one closed period — evaluation
/// outside that range must wrap, not extrapolate.
///
/// Reference: Zhao & Huang (2001), "Mirror extending and circular spline
/// function for empirical mode decomposition method", Journal of Zhejiang
/// University-SCIENCE, 2(3):247-252.
use crate::error::EmdError;
use crate::spline::{cubic::CubicSpline, Spline};

/// A periodic cubic spline that wraps evaluation positions modulo its period.
#[derive(Debug, Clone)]
pub struct CircularSpline {
    inner: CubicSpline,
    x_first: f64,
    period: f64,
}

impl CircularSpline {
    /// Construct from knot arrays using a periodic (cyclic tridiagonal) spline.
    ///
    /// `x` must be strictly increasing.  The period is taken as
    /// `x[last] - x[first]`, which matches the palindrome construction where
    /// `extended[0..N]` is one full period.
    pub fn from_knots(x: &[f64], y: &[f64]) -> Result<Self, EmdError> {
        if x.len() < 2 {
            return Err(EmdError::InsufficientData);
        }
        let x_first = x[0];
        let period = x[x.len() - 1] - x_first;
        if period <= 0.0 {
            return Err(EmdError::InvalidConfig(
                "CircularSpline requires strictly increasing knots with positive period".into(),
            ));
        }
        let inner = CubicSpline::periodic_from_knots(x, y)?;
        Ok(Self { inner, x_first, period })
    }

    /// Map an arbitrary position into `[x_first, x_first + period)`.
    #[inline]
    fn wrap(&self, x: f64) -> f64 {
        self.x_first + (x - self.x_first).rem_euclid(self.period)
    }
}

impl Spline for CircularSpline {
    fn evaluate(&self, x: f64) -> f64 {
        self.inner.evaluate(self.wrap(x))
    }

    fn evaluate_derivative(&self, x: f64) -> f64 {
        self.inner.evaluate_derivative(self.wrap(x))
    }

    fn knots(&self) -> (&[f64], &[f64]) {
        self.inner.knots()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_before_first_knot() {
        // sin knots at period 2π — value at 0 should equal value at 2π
        let n = 9;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * std::f64::consts::TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let spl = CircularSpline::from_knots(&x, &y).unwrap();

        // Evaluate just before the first knot — should wrap to near the end
        let v_before = spl.evaluate(-0.5);
        let v_wrapped = spl.evaluate(std::f64::consts::TAU - 0.5);
        assert!((v_before - v_wrapped).abs() < 1e-10,
            "wrap failed: before={v_before}, wrapped={v_wrapped}");
    }

    #[test]
    fn wrap_after_last_knot() {
        let n = 9;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * std::f64::consts::TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let spl = CircularSpline::from_knots(&x, &y).unwrap();

        let period = std::f64::consts::TAU;
        let v_after = spl.evaluate(period + 0.5);
        let v_wrapped = spl.evaluate(0.5);
        assert!((v_after - v_wrapped).abs() < 1e-10,
            "wrap failed: after={v_after}, wrapped={v_wrapped}");
    }

    #[test]
    fn interior_unchanged() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let spl = CircularSpline::from_knots(&x, &y).unwrap();
        // Interior points should evaluate normally (no wrapping needed)
        let v = spl.evaluate(2.0);
        assert!(v.is_finite(), "interior eval NaN");
    }
}
