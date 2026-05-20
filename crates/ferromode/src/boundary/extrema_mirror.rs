/// Extrema Mirror Extension — full PyEMD-compatible implementation.
///
/// Ports the exact boundary mirroring logic from PyEMD (Laszuk et al.),
/// which itself implements the strategy described in:
///
///   Huang et al. (1998) "The empirical mode decomposition and the Hilbert
///   spectrum for nonlinear and non-stationary time series analysis."
///   Proc. R. Soc. Lond. A 454, 903–995.
///
/// Four conditional branches per boundary side (left / right) depending on:
///   1. Whether the nearest interior extremum is a maximum or minimum
///   2. Whether the signal endpoint value is on the same side as that extremum
///   3. Whether the gap to the first extremum exceeds the gap to the boundary
///
/// The reflected positions and values are assembled into extended knot lists
/// for upper (maxima) and lower (minima) envelope splines.
use crate::boundary::{BoundaryCondition, ExtendedSignal, Extrema};

/// Configuration for extrema mirror extension.
#[derive(Debug, Clone)]
pub struct ExtremasMirrorConfig {
    /// Number of extrema to mirror beyond each boundary (default: 2, matches PyEMD).
    pub nbsym: usize,
}

impl Default for ExtremasMirrorConfig {
    fn default() -> Self {
        Self { nbsym: 2 }
    }
}

/// Extrema mirror boundary extension (PyEMD-compatible).
pub struct ExtremasMirror {
    config: ExtremasMirrorConfig,
}

impl Default for ExtremasMirror {
    fn default() -> Self {
        Self { config: ExtremasMirrorConfig::default() }
    }
}

impl ExtremasMirror {
    pub fn new(config: ExtremasMirrorConfig) -> Self {
        Self { config }
    }

    pub fn with_nbsym(nbsym: usize) -> Self {
        Self { config: ExtremasMirrorConfig { nbsym } }
    }
}

impl BoundaryCondition for ExtremasMirror {
    /// Extend signal by padding with even-mirrored samples.
    ///
    /// The actual extrema reflection is done separately via `build_envelope_knots`
    /// (called directly by the sifting engine). This `extend` method just ensures
    /// the padded signal is wide enough for the spline evaluations.
    fn extend(&self, signal: &[f64], extrema: &Extrema) -> ExtendedSignal {
        if signal.is_empty() {
            return ExtendedSignal { values: vec![], original_start: 0, original_end: 0 };
        }
        let n = signal.len();

        // Pad enough to cover reflected extrema indices (which can be negative
        // in signal-space). Use the outermost extremum index as the pad width.
        let left_pad = extrema.maxima_indices.first()
            .copied()
            .unwrap_or(0)
            .max(extrema.minima_indices.first().copied().unwrap_or(0))
            .min(n)
            .max(4);

        let right_ext = n.saturating_sub(
            extrema.maxima_indices.last().copied().unwrap_or(n - 1)
                .min(extrema.minima_indices.last().copied().unwrap_or(n - 1))
        );
        let right_pad = right_ext.min(n).max(4);

        let mut values = Vec::with_capacity(left_pad + n + right_pad);
        for i in (0..left_pad).rev() {
            values.push(signal[i.min(n - 1)]);
        }
        values.extend_from_slice(signal);
        for i in (0..right_pad).rev() {
            values.push(signal[n - 1 - i.min(n - 1)]);
        }

        ExtendedSignal {
            values,
            original_start: left_pad,
            original_end: left_pad + n,
        }
    }

    fn name(&self) -> &str {
        "extrema_mirror"
    }
}

// ── Public knot builder — called directly by the sifting engine ──────────────

/// Build extended spline knots for upper (maxima) and lower (minima) envelopes
/// using the full PyEMD-compatible conditional mirroring strategy.
///
/// Returns `(max_idx, max_val, min_idx, min_val)` — all in signal-index space
/// (may contain values outside `[0, n-1]`). Feed directly to the spline builder.
pub fn build_envelope_knots(
    signal: &[f64],
    max_pos: &[usize],
    min_pos: &[usize],
    nbsym: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = signal.len();
    if max_pos.is_empty() || min_pos.is_empty() || n < 2 {
        let mx: Vec<f64> = max_pos.iter().map(|&i| i as f64).collect();
        let mv: Vec<f64> = max_pos.iter().map(|&i| signal[i]).collect();
        let nx: Vec<f64> = min_pos.iter().map(|&i| i as f64).collect();
        let nv: Vec<f64> = min_pos.iter().map(|&i| signal[i]).collect();
        return (mx, mv, nx, nv);
    }

    let max_val: Vec<f64> = max_pos.iter().map(|&i| signal[i]).collect();
    let min_val: Vec<f64> = min_pos.iter().map(|&i| signal[i]).collect();
    let max_pos_f: Vec<f64> = max_pos.iter().map(|&i| i as f64).collect();
    let min_pos_f: Vec<f64> = min_pos.iter().map(|&i| i as f64).collect();
    let t0 = 0.0_f64;
    let t_end = (n - 1) as f64;
    let s0 = signal[0];
    let s_end = signal[n - 1];

    // ── Left boundary ─────────────────────────────────────────────────────────
    let d_pos_left = max_pos_f[0] - min_pos_f[0];
    let left_ext_max = d_pos_left < 0.0; // true → nearest left extremum is a maximum

    let (mut lmax_pos, mut lmax_val, mut lmin_pos, mut lmin_val) = if left_ext_max {
        // Nearest left extremum is a maximum
        let dist_to_max = max_pos_f[0] - t0;
        if s0 > min_val[0] && d_pos_left.abs() > dist_to_max {
            // Mirror to first (maximum) extremum
            let lmax_p: Vec<f64> = (1..=nbsym)
                .take(max_pos.len() - 1)
                .map(|k| 2.0 * max_pos_f[0] - max_pos_f[k])
                .collect();
            let lmax_v: Vec<f64> = (1..=nbsym)
                .take(max_val.len() - 1)
                .map(|k| max_val[k])
                .collect();
            let lmin_p: Vec<f64> = (0..nbsym.min(min_pos.len()))
                .map(|k| 2.0 * max_pos_f[0] - min_pos_f[k])
                .collect();
            let lmin_v: Vec<f64> = (0..nbsym.min(min_val.len()))
                .map(|k| min_val[k])
                .collect();
            (lmax_p, lmax_v, lmin_p, lmin_v)
        } else {
            // Mirror to signal beginning
            let lmax_p: Vec<f64> = (0..nbsym.min(max_pos.len()))
                .map(|k| 2.0 * t0 - max_pos_f[k])
                .collect();
            let lmax_v: Vec<f64> = (0..nbsym.min(max_val.len()))
                .map(|k| max_val[k])
                .collect();
            // prepend s0 to min positions
            let mut lmin_p: Vec<f64> = vec![2.0 * t0 - t0]; // = 0 reflected = 0... actually t0 itself
            lmin_p.extend(
                (0..nbsym.saturating_sub(1).min(min_pos.len()))
                    .map(|k| 2.0 * t0 - min_pos_f[k])
            );
            let mut lmin_v: Vec<f64> = vec![s0];
            lmin_v.extend((0..nbsym.saturating_sub(1).min(min_val.len())).map(|k| min_val[k]));
            (lmax_p, lmax_v, lmin_p, lmin_v)
        }
    } else {
        // Nearest left extremum is a minimum
        let dist_to_min = min_pos_f[0] - t0;
        if s0 < max_val[0] && d_pos_left.abs() > dist_to_min {
            // Mirror to first (minimum) extremum
            let lmax_p: Vec<f64> = (0..nbsym.min(max_pos.len()))
                .map(|k| 2.0 * min_pos_f[0] - max_pos_f[k])
                .collect();
            let lmax_v: Vec<f64> = (0..nbsym.min(max_val.len()))
                .map(|k| max_val[k])
                .collect();
            let lmin_p: Vec<f64> = (1..=nbsym)
                .take(min_pos.len() - 1)
                .map(|k| 2.0 * min_pos_f[0] - min_pos_f[k])
                .collect();
            let lmin_v: Vec<f64> = (1..=nbsym)
                .take(min_val.len() - 1)
                .map(|k| min_val[k])
                .collect();
            (lmax_p, lmax_v, lmin_p, lmin_v)
        } else {
            // Mirror to signal beginning
            let mut lmax_p: Vec<f64> = vec![2.0 * t0 - t0];
            lmax_p.extend(
                (0..nbsym.saturating_sub(1).min(max_pos.len()))
                    .map(|k| 2.0 * t0 - max_pos_f[k])
            );
            let mut lmax_v: Vec<f64> = vec![s0];
            lmax_v.extend((0..nbsym.saturating_sub(1).min(max_val.len())).map(|k| max_val[k]));
            let lmin_p: Vec<f64> = (0..nbsym.min(min_pos.len()))
                .map(|k| 2.0 * t0 - min_pos_f[k])
                .collect();
            let lmin_v: Vec<f64> = (0..nbsym.min(min_val.len()))
                .map(|k| min_val[k])
                .collect();
            (lmax_p, lmax_v, lmin_p, lmin_v)
        }
    };

    // Sort and deduplicate left extensions (descending → ascending after reverse)
    sort_and_dedup(&mut lmax_pos, &mut lmax_val);
    sort_and_dedup(&mut lmin_pos, &mut lmin_val);

    // ── Right boundary ────────────────────────────────────────────────────────
    let end_max = max_pos.len();
    let end_min = min_pos.len();
    let d_pos_right = max_pos_f[end_max - 1] - min_pos_f[end_min - 1];
    let right_ext_max = d_pos_right > 0.0; // true → nearest right extremum is a maximum

    let (mut rmax_pos, mut rmax_val, mut rmin_pos, mut rmin_val) = if !right_ext_max {
        // Nearest right extremum is a minimum
        let dist_to_min = t_end - min_pos_f[end_min - 1];
        if s_end < max_val[end_max - 1] && d_pos_right.abs() > dist_to_min {
            // Mirror to last (minimum) extremum
            let idx_max = end_max.saturating_sub(nbsym);
            let idx_min = end_min.saturating_sub(nbsym + 1);
            let rmax_p: Vec<f64> = (idx_max..end_max)
                .map(|k| 2.0 * min_pos_f[end_min - 1] - max_pos_f[k])
                .collect();
            let rmax_v: Vec<f64> = (idx_max..end_max).map(|k| max_val[k]).collect();
            let rmin_p: Vec<f64> = (idx_min..end_min - 1)
                .map(|k| 2.0 * min_pos_f[end_min - 1] - min_pos_f[k])
                .collect();
            let rmin_v: Vec<f64> = (idx_min..end_min - 1).map(|k| min_val[k]).collect();
            (rmax_p, rmax_v, rmin_p, rmin_v)
        } else {
            // Mirror to signal end
            let idx_max = end_max.saturating_sub(nbsym - 1);
            let idx_min = end_min.saturating_sub(nbsym);
            let mut rmax_p: Vec<f64> = (idx_max..end_max)
                .map(|k| 2.0 * t_end - max_pos_f[k])
                .collect();
            rmax_p.push(2.0 * t_end - t_end); // = t_end
            let mut rmax_v: Vec<f64> = (idx_max..end_max).map(|k| max_val[k]).collect();
            rmax_v.push(s_end);
            let rmin_p: Vec<f64> = (idx_min..end_min)
                .map(|k| 2.0 * t_end - min_pos_f[k])
                .collect();
            let rmin_v: Vec<f64> = (idx_min..end_min).map(|k| min_val[k]).collect();
            (rmax_p, rmax_v, rmin_p, rmin_v)
        }
    } else {
        // Nearest right extremum is a maximum
        let dist_to_max = t_end - max_pos_f[end_max - 1];
        if s_end > min_val[end_min - 1] && end_max > 1 && d_pos_right.abs() > dist_to_max {
            // Mirror to last (maximum) extremum
            let idx_max = end_max.saturating_sub(nbsym + 1);
            let idx_min = end_min.saturating_sub(nbsym);
            let rmax_p: Vec<f64> = (idx_max..end_max - 1)
                .map(|k| 2.0 * max_pos_f[end_max - 1] - max_pos_f[k])
                .collect();
            let rmax_v: Vec<f64> = (idx_max..end_max - 1).map(|k| max_val[k]).collect();
            let rmin_p: Vec<f64> = (idx_min..end_min)
                .map(|k| 2.0 * max_pos_f[end_max - 1] - min_pos_f[k])
                .collect();
            let rmin_v: Vec<f64> = (idx_min..end_min).map(|k| min_val[k]).collect();
            (rmax_p, rmax_v, rmin_p, rmin_v)
        } else {
            // Mirror to signal end
            let idx_max = end_max.saturating_sub(nbsym);
            let idx_min = end_min.saturating_sub(nbsym - 1);
            let rmax_p: Vec<f64> = (idx_max..end_max)
                .map(|k| 2.0 * t_end - max_pos_f[k])
                .collect();
            let rmax_v: Vec<f64> = (idx_max..end_max).map(|k| max_val[k]).collect();
            let mut rmin_p: Vec<f64> = (idx_min..end_min)
                .map(|k| 2.0 * t_end - min_pos_f[k])
                .collect();
            rmin_p.push(2.0 * t_end - t_end);
            let mut rmin_v: Vec<f64> = (idx_min..end_min).map(|k| min_val[k]).collect();
            rmin_v.push(s_end);
            (rmax_p, rmax_v, rmin_p, rmin_v)
        }
    };

    sort_and_dedup(&mut rmax_pos, &mut rmax_val);
    sort_and_dedup(&mut rmin_pos, &mut rmin_val);

    // ── Assemble final knot lists ──────────────────────────────────────────────
    let (full_max_pos, full_max_val) =
        concat_knots(lmax_pos, lmax_val, &max_pos_f, &max_val, rmax_pos, rmax_val);
    let (full_min_pos, full_min_val) =
        concat_knots(lmin_pos, lmin_val, &min_pos_f, &min_val, rmin_pos, rmin_val);

    (full_max_pos, full_max_val, full_min_pos, full_min_val)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sort_and_dedup(pos: &mut Vec<f64>, val: &mut Vec<f64>) {
    let mut pairs: Vec<(f64, f64)> = pos.iter().copied().zip(val.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
    *pos = pairs.iter().map(|p| p.0).collect();
    *val = pairs.iter().map(|p| p.1).collect();
}

fn concat_knots(
    left_pos: Vec<f64>, left_val: Vec<f64>,
    interior_pos: &[f64], interior_val: &[f64],
    right_pos: Vec<f64>, right_val: Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut pos: Vec<f64> = left_pos;
    pos.extend_from_slice(interior_pos);
    pos.extend(right_pos);
    let mut val: Vec<f64> = left_val;
    val.extend_from_slice(interior_val);
    val.extend(right_val);

    // Final dedup (left reflection may coincide with interior knot 0)
    let mut pairs: Vec<(f64, f64)> = pos.into_iter().zip(val).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);

    (pairs.iter().map(|p| p.0).collect(), pairs.iter().map(|p| p.1).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_envelope_knots_basic() {
        let signal: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin()).collect();
        let max_pos: Vec<usize> = (0..20)
            .filter(|&i| i > 0 && i < 19 && signal[i] > signal[i-1] && signal[i] > signal[i+1])
            .collect();
        let min_pos: Vec<usize> = (0..20)
            .filter(|&i| i > 0 && i < 19 && signal[i] < signal[i-1] && signal[i] < signal[i+1])
            .collect();

        if max_pos.len() >= 2 && min_pos.len() >= 2 {
            let (mx, _, mn, _) = build_envelope_knots(&signal, &max_pos, &min_pos, 2);
            // Should have extended beyond boundaries
            assert!(mx.first().unwrap() < &0.0 || mx.last().unwrap() > &19.0
                    || mn.first().unwrap() < &0.0 || mn.last().unwrap() > &19.0,
                    "knots should extend beyond signal range");
            // Strictly increasing
            for w in mx.windows(2) { assert!(w[1] > w[0], "max knots not strictly increasing"); }
            for w in mn.windows(2) { assert!(w[1] > w[0], "min knots not strictly increasing"); }
        }
    }

    #[test]
    fn test_extend_preserves_original() {
        let signal: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin()).collect();
        let extrema = Extrema {
            maxima_indices: vec![5, 15, 25],
            minima_indices: vec![2, 10, 20],
        };
        let em = ExtremasMirror::default();
        let ext = em.extend(&signal, &extrema);
        assert_eq!(&ext.values[ext.original_start..ext.original_end], signal.as_slice());
    }
}
