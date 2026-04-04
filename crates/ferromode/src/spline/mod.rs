/// Spline trait — defines the interface for all spline interpolation variants.
pub trait Spline {
    /// Evaluate the spline at a given point `x`.
    fn evaluate(&self, x: f64) -> f64;

    /// Evaluate the first derivative of the spline at a given point `x`.
    fn evaluate_derivative(&self, x: f64) -> f64;

    /// Returns references to the knot arrays (x_knots, y_knots).
    fn knots(&self) -> (&[f64], &[f64]);
}

pub mod cubic;

pub use cubic::CubicSpline;
