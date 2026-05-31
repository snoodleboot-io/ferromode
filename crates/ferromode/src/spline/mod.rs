//! Spline interpolation for EMD envelope computation.
//!
//! [`CubicSpline`] is the concrete implementation. The boundary condition
//! used when solving the spline system is controlled by [`SplineType`],
//! which is a field of [`crate::sifting::SiftingConfig`].
//!
//! [`SplineType::Natural`] (the default) is appropriate for most signals.
//! [`SplineType::Periodic`] is selected automatically when
//! [`crate::boundary::BoundaryConditionType::PalindromeCyclic`] is used.

use serde::{Deserialize, Serialize};

/// Cubic spline boundary condition used when fitting IMF envelopes.
///
/// Controls how second derivatives at the endpoints of the knot set are
/// constrained when solving the spline system.
///
/// # Choosing a type
///
/// - [`SplineType::Natural`] — default; zero second derivative at both ends.
///   Works well for most signals.
/// - [`SplineType::Periodic`] — first and second derivatives match at the
///   endpoints of the knot set. Selected automatically by
///   [`crate::boundary::BoundaryConditionType::PalindromeCyclic`].
/// - [`SplineType::NotAKnot`] — C³ continuity at the first and last interior
///   knots; no artificial boundary constraint. Accurate for arbitrary data but
///   slightly more expensive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SplineType {
    /// Natural boundary: second derivative = 0 at both endpoints. Default.
    #[default]
    Natural,
    /// Periodic (cyclic) boundary: first and second derivatives match at endpoints.
    /// Appropriate when the knot set is known to be periodic.
    Periodic,
    /// Not-a-knot: C³ continuity at the first and last interior knots.
    /// No artificial boundary constraint; accurate for arbitrary data.
    NotAKnot,
}

/// Spline trait — defines the interface for all spline interpolation variants.
pub trait Spline {
    /// Evaluate the spline at a given point `x`.
    fn evaluate(&self, x: f64) -> f64;

    /// Evaluate the first derivative of the spline at a given point `x`.
    fn evaluate_derivative(&self, x: f64) -> f64;

    /// Returns references to the knot arrays (x_knots, y_knots).
    fn knots(&self) -> (&[f64], &[f64]);
}

/// Cubic spline implementation and solver routines.
pub mod cubic;

pub use cubic::CubicSpline;
