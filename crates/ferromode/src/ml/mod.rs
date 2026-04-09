#![warn(missing_docs)]

//! Machine Learning and differentiation infrastructure for ferromode.
//!
//! This module provides the foundation for differentiable signal processing,
//! enabling gradient computation through EMD and related algorithms.
//!
//! The module is organized as follows:
//!
//! - `differentiable` — Differentiable EMD with forward/backward pass infrastructure
//! - `linear_algebra` — Matrix operations, LU decomposition, linear system solving
//! - `implicit_diff` — Implicit differentiation via Jacobian computation (T-321+)

pub mod differentiable;
pub mod linear_algebra;
// pub mod implicit_diff;  // TODO: Complete in T-321 (Jacobian computation)

pub use differentiable::{DifferentiableEmd, ImplicitEmdContext};
pub use linear_algebra::{condition_number, matrix_inverse, solve_linear_system, Matrix};
