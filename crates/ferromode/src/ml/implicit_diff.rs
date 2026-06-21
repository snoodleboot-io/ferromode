#![warn(missing_docs)]

//! Differentiable EMD backward pass via exact linearization.
//!
//! # Why this is exact
//!
//! At its realized extrema configuration, the EMD forward pass is *exactly* a
//! product of linear operators. Each sifting iteration applies
//! ```text
//! h <- h - mean_env(h) = (I - P) h
//! ```
//! where `mean_env` is the average of the upper and lower cubic-spline envelopes.
//! Cubic-spline interpolation through a fixed set of knots is linear in the knot
//! *values*, so once the extrema indices are fixed (which they are, given the
//! converged decomposition), `P` is a fixed linear operator determined solely by
//! those indices. Empirically the forward map's Jacobian is constant to machine
//! precision in a neighborhood of the input (see `examples/diff_groundtruth.rs`).
//!
//! Therefore the whole decomposition `x -> (c_1, ..., c_K, r_K)` is, locally, an
//! exact linear map `M`, and the vector–Jacobian product is `grad_x = Mᵀ g`. We
//! never form `M`: we replay the recorded per-iteration operators in transpose.
//!
//! # Sequential structure
//!
//! With `r_0 = x` (the working signal) and, for each IMF `k`,
//! ```text
//! c_k = A_k r_{k-1},   r_k = (I - A_k) r_{k-1},   A_k = (I - P_{k,T}) ... (I - P_{k,1})
//! ```
//! the reverse pass (with `g_k = ∂L/∂c_k`, no gradient on the final residue) is
//! ```text
//! bar_r_K = 0
//! bar_r_{k-1} = A_kᵀ (g_k - bar_r_k) + bar_r_k       for k = K..1
//! grad_x = bar_r_0
//! ```
//! `A_kᵀ` applies the step transposes in reverse order. For the PalindromeCyclic
//! boundary the signal is first mirrored to length `2N-1`; that mirror is also
//! linear and is folded back in transpose at the end.

use super::differentiable::ImplicitEmdContext;
use crate::algorithms::emd::EmdBackwardTrace;
use crate::error::EmdError;
use crate::sifting::SiftStep;
use crate::spline::cubic::CubicSpline;
use crate::spline::{Spline, SplineType};

/// Build a cubic spline using the same constructor the sifting forward uses.
fn build_spline(xs: &[f64], ys: &[f64], spline_type: SplineType) -> Result<CubicSpline, EmdError> {
    match spline_type {
        SplineType::Natural => CubicSpline::from_knots(xs, ys),
        SplineType::Periodic => CubicSpline::periodic_from_knots(xs, ys),
        SplineType::NotAKnot => CubicSpline::not_a_knot_from_knots(xs, ys),
    }
}

/// Accumulate the transpose of a single envelope operator into `out`.
///
/// The envelope is `E u = spline(knots_x = idx, knots_y = u[idx])` evaluated at
/// `0..n`, i.e. `E = B · Sel` where `Sel` gathers `u` at the knot indices and
/// `B[i, m]` is the value at sample `i` of the cardinal spline basis for knot
/// `m`. Its transpose scatters `Bᵀ u` back to the knot indices. We obtain column
/// `m` of `B` exactly by interpolating the unit knot vector `e_m` (the spline is
/// linear in the knot values, so this is exact, not finite-difference).
///
/// `weight` is `0.5` because the mean envelope averages the two envelopes; this
/// matches the forward's `mean_env = (upper + lower) / 2`.
fn add_envelope_transpose(
    knot_idx: &[usize],
    n: usize,
    spline_type: SplineType,
    u: &[f64],
    weight: f64,
    out: &mut [f64],
) {
    // Mirrors `build_env` in sifting: fewer than 2 knots => zero envelope.
    if knot_idx.len() < 2 {
        return;
    }
    let xs: Vec<f64> = knot_idx.iter().map(|&i| i as f64).collect();
    let mut ys = vec![0.0; knot_idx.len()];
    for m in 0..knot_idx.len() {
        ys[m] = 1.0;
        // The forward falls back to a zero envelope if the spline build fails;
        // since failure depends only on the knot positions (not values), every
        // column would fail identically, so skipping the column matches.
        if let Ok(spline) = build_spline(&xs, &ys, spline_type) {
            let mut dot = 0.0;
            for (i, &ui) in u.iter().enumerate().take(n) {
                dot += spline.evaluate(i as f64) * ui;
            }
            out[knot_idx[m]] += weight * dot;
        }
        ys[m] = 0.0;
    }
}

/// Apply `Pᵀ u` for one sifting step (`P` = mean-envelope operator).
fn p_transpose_apply(step: &SiftStep, n: usize, spline_type: SplineType, u: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n];
    add_envelope_transpose(&step.maxima, n, spline_type, u, 0.5, &mut out);
    add_envelope_transpose(&step.minima, n, spline_type, u, 0.5, &mut out);
    out
}

/// Apply `A_kᵀ v` — the transpose of one IMF's full sifting chain.
///
/// `A_k = (I - P_T) ... (I - P_1)`, so `A_kᵀ = (I - P_1)ᵀ ... (I - P_T)ᵀ`; we
/// apply `(I - P_j)ᵀ u = u - P_jᵀ u` for the steps in reverse order.
fn chain_transpose(steps: &[SiftStep], n: usize, spline_type: SplineType, v: &[f64]) -> Vec<f64> {
    let mut u = v.to_vec();
    for step in steps.iter().rev() {
        let pt = p_transpose_apply(step, n, spline_type, &u);
        for (ui, pti) in u.iter_mut().zip(pt) {
            *ui -= pti;
        }
    }
    u
}

/// Compute `∂L/∂signal` given `∂L/∂IMF_k` for every emitted IMF.
///
/// `grad_imfs[k]` is the upstream gradient w.r.t. IMF `k` (length `n_out`); the
/// final residue is treated as having zero upstream gradient (loss-on-IMFs
/// convention, matching the FFI `ferromode_diff_backward` signature). Returns
/// the gradient w.r.t. the input signal (length `n_out`).
pub fn emd_signal_gradient(
    trace: &EmdBackwardTrace,
    grad_imfs: &[Vec<f64>],
) -> Result<Vec<f64>, EmdError> {
    let k = trace.imf_steps.len();
    if grad_imfs.len() != k {
        return Err(EmdError::DimensionMismatch);
    }
    let n_work = trace.n_work;
    let n_out = trace.n_out;

    // Reverse sweep over IMFs: bar_r is the adjoint of the running residual.
    let mut bar_r = vec![0.0; n_work];
    for idx in (0..k).rev() {
        let g = &grad_imfs[idx];
        if g.len() != n_out {
            return Err(EmdError::DimensionMismatch);
        }
        // Lift g into working space (transpose of truncation: zero-pad). For the
        // non-palindrome case n_work == n_out, so this is just a copy.
        let mut gk = vec![0.0; n_work];
        gk[..n_out].copy_from_slice(&g[..n_out]);

        // diff = g_k - bar_r_k
        let diff: Vec<f64> = gk.iter().zip(&bar_r).map(|(&a, &b)| a - b).collect();
        let t = chain_transpose(&trace.imf_steps[idx], n_work, trace.spline_type, &diff);
        // bar_r_{k-1} = A_kᵀ(g_k - bar_r_k) + bar_r_k
        for (br, ti) in bar_r.iter_mut().zip(t) {
            *br += ti;
        }
    }

    if trace.is_palindrome {
        // Transpose of build_palindrome: w = [x, reverse(x[..N-1])] (length 2N-1).
        let n = n_out;
        let mut grad = vec![0.0; n];
        grad[..n].copy_from_slice(&bar_r[..n]);
        for (k_idx, gk) in grad.iter_mut().enumerate().take(n.saturating_sub(1)) {
            *gk += bar_r[2 * n - 2 - k_idx];
        }
        Ok(grad)
    } else {
        // n_work == n_out
        bar_r.truncate(n_out);
        Ok(bar_r)
    }
}

impl ImplicitEmdContext {
    /// Backward pass: gradient of a loss w.r.t. the input signal.
    ///
    /// `grad_imfs[k]` is `∂L/∂IMF_k`. Requires a context produced by
    /// [`DifferentiableEmd::forward`](super::differentiable::DifferentiableEmd::forward)
    /// (which records the linearization trace).
    ///
    /// # Errors
    /// Returns [`EmdError::InvalidConfig`] if the context has no trace, or
    /// [`EmdError::DimensionMismatch`] if `grad_imfs` has the wrong shape.
    pub fn backward(&self, grad_imfs: &[Vec<f64>]) -> Result<Vec<f64>, EmdError> {
        let trace = self.trace.as_ref().ok_or_else(|| {
            EmdError::InvalidConfig("context has no backward trace (was it built via forward?)".into())
        })?;
        emd_signal_gradient(trace, grad_imfs)
    }
}

// ---------------------------------------------------------------------------
// Tests — validated against the finite-difference oracle.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::emd::{emd, EmdConfig};
    use crate::ml::differentiable::DifferentiableEmd;
    use std::f64::consts::PI;

    fn signal_two_tone(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 12.0 * t).sin()
            })
            .collect()
    }

    /// Finite-difference d(IMF_imf)/d(signal) column-by-column on the real emd.
    fn fd_jacobian(signal: &[f64], cfg: &EmdConfig, imf: usize, eps: f64) -> Vec<Vec<f64>> {
        let n = signal.len();
        let base = emd(signal, cfg).unwrap();
        let n_out = base.imfs.imfs[imf].len();
        let mut j = vec![vec![0.0; n]; n_out];
        for col in 0..n {
            let mut sp = signal.to_vec();
            sp[col] += eps;
            let cp = emd(&sp, cfg).unwrap().imfs.imfs[imf].clone();
            let mut sm = signal.to_vec();
            sm[col] -= eps;
            let cm = emd(&sm, cfg).unwrap().imfs.imfs[imf].clone();
            for row in 0..n_out {
                j[row][col] = (cp[row] - cm[row]) / (2.0 * eps);
            }
        }
        j
    }

    /// The analytic VJP must equal the FD Jacobian: for a one-hot upstream
    /// gradient on IMF `imf` at row `r`, backward returns row `r` of dc_imf/dx.
    fn check_against_fd(signal: &[f64], cfg: &EmdConfig, tol: f64) {
        let decomposer = DifferentiableEmd::new(cfg.clone());
        let ctx = decomposer.forward(signal).unwrap();
        let n = signal.len();
        let k = ctx.imfs.len();
        assert!(k >= 1, "need at least one IMF");

        for imf in 0..k {
            let jac = fd_jacobian(signal, cfg, imf, 1e-5);
            let n_out = ctx.imfs[imf].len();
            // Probe a handful of rows to keep the test quick but representative.
            for &row in &[0usize, n_out / 3, n_out / 2, n_out - 1] {
                let mut grads = vec![vec![0.0; n_out]; k];
                grads[imf][row] = 1.0;
                let g = ctx.backward(&grads).unwrap();
                let mut max_err = 0.0f64;
                for col in 0..n {
                    max_err = max_err.max((g[col] - jac[row][col]).abs());
                }
                assert!(
                    max_err < tol,
                    "imf {imf} row {row}: max grad error {max_err:.3e} exceeds {tol:.0e}"
                );
            }
        }
    }

    #[test]
    fn backward_matches_finite_difference_default() {
        let signal = signal_two_tone(64);
        let cfg = EmdConfig { max_imfs: 3, ..Default::default() };
        check_against_fd(&signal, &cfg, 1e-6);
    }

    #[test]
    fn backward_matches_finite_difference_single_imf() {
        let signal = signal_two_tone(80);
        let cfg = EmdConfig { max_imfs: 1, ..Default::default() };
        check_against_fd(&signal, &cfg, 1e-6);
    }

    #[test]
    fn backward_matches_finite_difference_palindrome() {
        use crate::boundary::BoundaryConditionType;
        use crate::sifting::SiftingConfig;
        let signal = signal_two_tone(64);
        let cfg = EmdConfig {
            max_imfs: 2,
            sifting_config: SiftingConfig {
                boundary_condition: BoundaryConditionType::PalindromeCyclic,
                ..Default::default()
            },
            ..Default::default()
        };
        check_against_fd(&signal, &cfg, 1e-6);
    }

    #[test]
    fn backward_matches_finite_difference_notaknot_spline() {
        use crate::sifting::SiftingConfig;
        use crate::spline::SplineType;
        let signal = signal_two_tone(72);
        let cfg = EmdConfig {
            max_imfs: 2,
            sifting_config: SiftingConfig {
                spline_type: SplineType::NotAKnot,
                ..Default::default()
            },
            ..Default::default()
        };
        check_against_fd(&signal, &cfg, 1e-6);
    }

    #[test]
    fn backward_linearity_sum_of_grads() {
        // grad for (g_a + g_b) == grad(g_a) + grad(g_b): the map is linear.
        let signal = signal_two_tone(64);
        let cfg = EmdConfig { max_imfs: 2, ..Default::default() };
        let ctx = DifferentiableEmd::new(cfg).forward(&signal).unwrap();
        let k = ctx.imfs.len();
        let n = ctx.imfs[0].len();

        let mut ga = vec![vec![0.0; n]; k];
        let mut gb = vec![vec![0.0; n]; k];
        for i in 0..n {
            ga[0][i] = ((i * 3 % 7) as f64) - 3.0;
            gb[k - 1][i] = ((i * 5 % 11) as f64) - 5.0;
        }
        let sum_grad = ctx.backward(&{
            let mut s = vec![vec![0.0; n]; k];
            for kk in 0..k {
                for i in 0..n {
                    s[kk][i] = ga[kk][i] + gb[kk][i];
                }
            }
            s
        }).unwrap();
        let grad_a = ctx.backward(&ga).unwrap();
        let grad_b = ctx.backward(&gb).unwrap();
        for i in 0..n {
            assert!((sum_grad[i] - (grad_a[i] + grad_b[i])).abs() < 1e-9);
        }
    }

    #[test]
    fn backward_requires_trace() {
        let ctx = ImplicitEmdContext::new(
            vec![1.0, 2.0, 3.0],
            vec![vec![0.0, 0.0, 0.0]],
            vec![1.0, 2.0, 3.0],
            EmdConfig::default(),
        );
        assert!(ctx.backward(&[vec![1.0, 1.0, 1.0]]).is_err());
    }
}
