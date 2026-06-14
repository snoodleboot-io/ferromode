/// Palindrome-Cyclic boundary extension.
///
/// Pre-extends a signal into a `2N-1` palindrome by appending the reversed signal
/// (excluding the final sample to avoid duplication):
///
/// ```text
/// [s_0, s_1, ..., s_{N-1}, s_{N-2}, ..., s_1, s_0]
/// ```
///
/// The resulting signal is even-symmetric around its midpoint `s_{N-1}`, which makes
/// the signal naturally periodic: the start and end values both equal `s_0`, and the
/// signal is smooth at the join point (derivative is zero by symmetry).
///
/// This pre-processing is applied once at the EMD level before the sifting loop begins.
/// During each sifting iteration on the `2N-1` signal, the per-iteration extension
/// delegates to `MirrorEven` as a small stability margin, and the sifting engine uses
/// the periodic cubic spline (`SplineType::Periodic`).
///
/// After decomposition, IMFs and residue are trimmed back to the original `N` samples.
/// Reconstruction is preserved exactly: `Σ imf[:N] + residue[:N] == original_signal`.
use crate::boundary::{BoundaryCondition, ExtendedSignal, Extrema};
use crate::boundary::mirror::{Mirror, MirrorConfig, MirrorVariant};

/// Build a `2N-1` palindrome from `signal`.
///
/// Returns `[s_0, ..., s_{N-1}, s_{N-2}, ..., s_0]`.
/// If `signal.len() < 2`, returns a copy of the signal unchanged.
pub fn build_palindrome(signal: &[f64]) -> Vec<f64> {
    if signal.len() < 2 {
        return signal.to_vec();
    }
    let mut palindrome = signal.to_vec();
    // Append reversed signal excluding the last element (which is already the midpoint)
    palindrome.extend(signal[..signal.len() - 1].iter().rev().copied());
    palindrome
}

/// Palindrome-cyclic boundary extension.
///
/// The `extend()` method on this struct is used as the per-sifting-iteration boundary
/// within the palindrome EMD. It delegates to `MirrorEven` to provide a small stability
/// extension. The palindrome pre-processing itself is handled by `build_palindrome()` at
/// the `emd()` level.
pub struct PalindromeCyclic {
    inner: Mirror,
}

impl Default for PalindromeCyclic {
    fn default() -> Self {
        Self {
            inner: Mirror::new(MirrorConfig { variant: MirrorVariant::Even, mirror_length: None }),
        }
    }
}

impl PalindromeCyclic {
    /// Construct a `PalindromeCyclic` boundary strategy with default settings.
    pub fn new() -> Self {
        Self::default()
    }
}

impl BoundaryCondition for PalindromeCyclic {
    fn extend(&self, signal: &[f64], extrema: &Extrema) -> ExtendedSignal {
        self.inner.extend(signal, extrema)
    }

    fn name(&self) -> &str {
        "palindrome_cyclic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_palindrome_basic() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = build_palindrome(&signal);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(result.len(), 2 * signal.len() - 1);
    }

    #[test]
    fn test_build_palindrome_preserves_first_half() {
        let signal = vec![0.5, 1.0, 1.5, 2.0];
        let result = build_palindrome(&signal);
        assert_eq!(&result[..signal.len()], &signal[..]);
    }

    #[test]
    fn test_build_palindrome_endpoints_match() {
        let signal = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let result = build_palindrome(&signal);
        // Both endpoints should be s_0
        assert_eq!(result[0], result[result.len() - 1]);
        assert_eq!(result[0], signal[0]);
    }

    #[test]
    fn test_build_palindrome_symmetric() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = build_palindrome(&signal);
        let n = result.len();
        for i in 0..n {
            assert!(
                (result[i] - result[n - 1 - i]).abs() < 1e-15,
                "palindrome not symmetric at index {}: {} != {}",
                i, result[i], result[n - 1 - i]
            );
        }
    }

    #[test]
    fn test_build_palindrome_two_samples() {
        let signal = vec![1.0, 2.0];
        let result = build_palindrome(&signal);
        assert_eq!(result, vec![1.0, 2.0, 1.0]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_build_palindrome_single_sample() {
        let signal = vec![42.0];
        let result = build_palindrome(&signal);
        assert_eq!(result, vec![42.0]);
    }

    #[test]
    fn test_build_palindrome_empty() {
        let signal: Vec<f64> = vec![];
        let result = build_palindrome(&signal);
        assert!(result.is_empty());
    }

    #[test]
    fn test_palindrome_cyclic_extend_delegates_to_mirror() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let extrema = Extrema { maxima_indices: vec![2], minima_indices: vec![0, 4] };

        let strategy = PalindromeCyclic::new();
        let result = strategy.extend(&signal, &extrema);

        // Should behave like MirrorEven — original is preserved
        let orig = &result.values[result.original_start..result.original_end];
        assert_eq!(orig, &signal[..]);
        assert!(result.values.len() > signal.len());
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_palindrome_cyclic_name() {
        assert_eq!(PalindromeCyclic::new().name(), "palindrome_cyclic");
    }
}
