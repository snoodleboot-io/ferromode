#![warn(missing_docs)]

//! Padding utilities for 1D signal extension.
//!
//! This module provides symmetric and other padding strategies used to reduce
//! edge artifacts when decomposing 2D/3D data via separable 1D EMD.

/// Pad a signal symmetrically (mirror boundary).
///
/// Extends the signal by reflecting it at the boundaries, which reduces
/// edge artifacts compared to zero-padding or periodic extension.
///
/// For a signal [1, 2, 3, 4] with pad_size=2, produces:
/// [3, 2, 1, 2, 3, 4, 3, 2]
/// (reflects signal[1] and signal[0] on left, signal[3] and signal[2] on right)
///
/// # Arguments
///
/// * `signal` - The original signal slice
/// * `pad_size` - Number of points to add at each boundary
///
/// # Returns
///
/// A new vector with padded data
///
/// # Panics
///
/// Panics if `pad_size > signal.len()` (not enough data to reflect)
pub fn pad_symmetric_1d(signal: &[f64], pad_size: usize) -> Vec<f64> {
    assert!(!signal.is_empty(), "Cannot pad empty signal");
    assert!(
        pad_size <= signal.len(),
        "pad_size {} exceeds signal length {}",
        pad_size,
        signal.len()
    );

    let mut result = Vec::with_capacity(signal.len() + 2 * pad_size);

    // Left padding: reflect backwards from index pad_size
    // For signal [1, 2, 3, 4] with pad_size=2:
    // i=0: signal[2-0-1] = signal[1] = 2
    // i=1: signal[2-1-1] = signal[0] = 1
    // Result: [2, 1, ...]
    for i in 0..pad_size {
        let idx = pad_size - i - 1;
        if idx < signal.len() {
            result.push(signal[idx]);
        }
    }

    // Original signal
    result.extend_from_slice(signal);

    // Right padding: reflect backwards from the last valid indices (excluding boundary)
    // For signal [1, 2, 3, 4] with pad_size=2:
    // n=4, we want indices 2, 1 (i.e., n-2, n-3)
    // i=1: signal[4-1-1] = signal[2] = 3
    // i=2: signal[4-2-1] = signal[1] = 2
    // Result: [3, 2]
    let n = signal.len();
    for i in 1..=pad_size {
        if i + 1 <= n {
            let idx = n - i - 1;
            result.push(signal[idx]);
        }
    }

    result
}

/// Remove padding from a previously padded signal.
///
/// Extracts the original signal region from a padded result.
///
/// # Arguments
///
/// * `padded` - The padded signal
/// * `original_len` - Length of the original unpadded signal
/// * `pad_size` - Size of padding that was applied
///
/// # Returns
///
/// A new vector containing only the original signal region
///
/// # Panics
///
/// Panics if the padded length doesn't match expectations
pub fn unpad_1d(padded: &[f64], original_len: usize, pad_size: usize) -> Vec<f64> {
    let expected_padded_len = original_len + 2 * pad_size;
    assert_eq!(
        padded.len(),
        expected_padded_len,
        "Padded length {} doesn't match expected {}",
        padded.len(),
        expected_padded_len
    );

    padded[pad_size..pad_size + original_len].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_symmetric_simple() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let padded = pad_symmetric_1d(&signal, 2);

        // Expected: left=[2, 1] + original=[1, 2, 3, 4] + right=[3, 2]
        // = [2, 1, 1, 2, 3, 4, 3, 2]
        assert_eq!(padded, vec![2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn test_pad_symmetric_single_point() {
        let signal = vec![5.0];
        let padded = pad_symmetric_1d(&signal, 1);

        // With single point and pad_size=1:
        // Left padding: i=0, idx=1-0-1=0, signal[0]=5.0 → [5.0]
        // Original: [5.0]
        // Right padding: i=1, check i+1<=1 (false), so nothing
        // Result: [5.0, 5.0]
        assert_eq!(padded.len(), 2);
        assert_eq!(padded, vec![5.0, 5.0]);
    }

    #[test]
    fn test_pad_unpad_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pad_size = 3;

        let padded = pad_symmetric_1d(&original, pad_size);
        assert_eq!(padded.len(), original.len() + 2 * pad_size);

        let recovered = unpad_1d(&padded, original.len(), pad_size);
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_unpad_preserves_center() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let padded = pad_symmetric_1d(&signal, 2);
        let unpadded = unpad_1d(&padded, signal.len(), 2);

        assert_eq!(unpadded, signal);
    }

    #[test]
    #[should_panic(expected = "pad_size 5 exceeds signal length 3")]
    fn test_pad_panic_oversized() {
        let signal = vec![1.0, 2.0, 3.0];
        let _ = pad_symmetric_1d(&signal, 5);
    }

    #[test]
    #[should_panic(expected = "Cannot pad empty signal")]
    fn test_pad_panic_empty() {
        let signal: Vec<f64> = vec![];
        let _ = pad_symmetric_1d(&signal, 1);
    }
}
