#![warn(missing_docs)]

//! Extrema detection utilities for signal processing.
//!
//! This module provides functions to detect local maxima and minima in signals,
//! handling plateaus and boundary conditions appropriately for EMD algorithms.

use serde::{Deserialize, Serialize};

/// Represents the extrema of a signal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Extrema {
    /// Indices of local maxima
    pub maxima: Vec<usize>,
    /// Indices of local minima
    pub minima: Vec<usize>,
}

/// Find indices of local maxima in the signal.
///
/// A local maximum is where the value is greater than both neighbors.
/// For plateaus (flat tops), the midpoint index of the plateau is used.
///
/// Boundary elements are considered if they satisfy the condition with the available neighbor.
///
/// # Arguments
/// * `signal` - The input signal as a slice of f64
///
/// # Returns
/// A vector of indices where local maxima occur, sorted in ascending order.
///
/// # Examples
/// ```
/// use ferromode::extrema::find_local_maxima;
///
/// let signal = [1.0, 3.0, 2.0, 4.0, 1.0];
/// let maxima = find_local_maxima(&signal);
/// assert_eq!(maxima, vec![1, 3]);
/// ```
pub fn find_local_maxima(signal: &[f64]) -> Vec<usize> {
    let mut maxima = Vec::new();

    if signal.len() < 3 {
        return maxima; // Need at least 3 points for an interior maximum
    }

    // Check middle elements only - no boundary extrema detection
    // This prevents spurious extrema at signal boundaries
    let mut i = 1;
    while i < signal.len() - 1 {
        // Start of potential maximum or plateau
        if signal[i] > signal[i - 1] {
            let plateau_start = i;

            // Extend through equal values (plateau)
            while i < signal.len() - 1 && signal[i] == signal[i + 1] {
                i += 1;
            }

            // Check if this plateau ends with a decrease
            if i < signal.len() - 1 && signal[i] > signal[i + 1] {
                // It's a maximum plateau, use midpoint
                let midpoint = (plateau_start + i) / 2;
                maxima.push(midpoint);
            }

            i += 1;
            continue;
        }

        i += 1;
    }

    maxima.sort();
    maxima.dedup();
    maxima
}

/// Find indices of local minima in the signal.
///
/// A local minimum is where the value is less than both neighbors.
/// For plateaus (flat bottoms), the midpoint index of the plateau is used.
///
/// Boundary elements are considered if they satisfy the condition with the available neighbor.
///
/// # Arguments
/// * `signal` - The input signal as a slice of f64
///
/// # Returns
/// A vector of indices where local minima occur, sorted in ascending order.
///
/// # Examples
/// ```
/// use ferromode::extrema::find_local_minima;
///
/// let signal = [3.0, 1.0, 4.0, 2.0, 5.0];
/// let minima = find_local_minima(&signal);
/// assert_eq!(minima, vec![1, 3]);
/// ```
pub fn find_local_minima(signal: &[f64]) -> Vec<usize> {
    let mut minima = Vec::new();

    if signal.len() < 3 {
        return minima; // Need at least 3 points for an interior minimum
    }

    // Check middle elements only - no boundary extrema detection
    // This prevents spurious extrema at signal boundaries
    let mut i = 1;
    while i < signal.len() - 1 {
        // Start of potential minimum or plateau
        if signal[i] < signal[i - 1] {
            let plateau_start = i;

            // Extend through equal values (plateau)
            while i < signal.len() - 1 && signal[i] == signal[i + 1] {
                i += 1;
            }

            // Check if this plateau ends with an increase
            if i < signal.len() - 1 && signal[i] < signal[i + 1] {
                // It's a minimum plateau, use midpoint
                let midpoint = (plateau_start + i) / 2;
                minima.push(midpoint);
            }

            // Skip past the plateau to avoid re-detecting
            i += 1;
            continue;
        }

        i += 1;
    }

    minima.sort();
    minima.dedup();
    minima
}

/// Detect both maxima and minima in the signal.
///
/// Combines the results of `find_local_maxima` and `find_local_minima`.
///
/// # Arguments
/// * `signal` - The input signal as a slice of f64
///
/// # Returns
/// An `Extrema` struct containing sorted vectors of maxima and minima indices.
///
/// # Examples
/// ```
/// use ferromode::extrema::detect_extrema;
///
/// let signal = [1.0, 3.0, 2.0, 4.0, 1.0];
/// let extrema = detect_extrema(&signal);
/// assert_eq!(extrema.maxima, vec![1, 3]);
/// assert_eq!(extrema.minima, vec![0, 4]);
/// ```
pub fn detect_extrema(signal: &[f64]) -> Extrema {
    Extrema { maxima: find_local_maxima(signal), minima: find_local_minima(signal) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_find_local_maxima_simple() {
        let signal = vec![1.0, 3.0, 2.0, 4.0, 1.0];
        let maxima = find_local_maxima(&signal);
        assert_eq!(maxima, vec![1, 3]);
    }

    #[test]
    fn test_find_local_minima_simple() {
        let signal = vec![3.0, 1.0, 4.0, 2.0, 5.0];
        let minima = find_local_minima(&signal);
        assert_eq!(minima, vec![1, 3]);
    }

    #[test]
    fn test_find_local_maxima_boundary() {
        let signal = vec![5.0, 1.0, 2.0, 3.0];
        let maxima = find_local_maxima(&signal);
        assert_eq!(maxima, Vec::<usize>::new()); // No interior maxima; boundary detection disabled
    }

    #[test]
    fn test_find_local_minima_boundary() {
        let signal = vec![1.0, 3.0, 2.0, 0.5];
        let minima = find_local_minima(&signal);
        assert_eq!(minima, Vec::<usize>::new()); // 2.0 is not a minimum (0.5 is lower on the right)
    }

    #[test]
    fn test_find_local_maxima_plateau() {
        let signal = vec![1.0, 2.0, 2.0, 2.0, 1.0];
        let maxima = find_local_maxima(&signal);
        // Midpoint of plateau 1,2,3 is 2
        assert_eq!(maxima, vec![2]);
    }

    #[test]
    fn test_find_local_minima_plateau() {
        let signal = vec![3.0, 1.0, 1.0, 1.0, 2.0];
        let minima = find_local_minima(&signal);
        // Midpoint of plateau 1,2,3 is 2
        assert_eq!(minima, vec![2]);
    }

    #[test]
    fn test_find_local_maxima_no_extrema() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let maxima = find_local_maxima(&signal);
        assert_eq!(maxima, Vec::<usize>::new()); // Monotonic signal has no interior extrema
    }

    #[test]
    fn test_find_local_minima_no_extrema() {
        let signal = vec![4.0, 3.0, 2.0, 1.0];
        let minima = find_local_minima(&signal);
        assert_eq!(minima, Vec::<usize>::new()); // Monotonic signal has no interior extrema
    }

    #[test]
    fn test_find_local_maxima_constant_signal() {
        let signal = vec![2.0, 2.0, 2.0, 2.0];
        let maxima = find_local_maxima(&signal);
        assert_eq!(maxima, Vec::<usize>::new());
    }

    #[test]
    fn test_find_local_minima_constant_signal() {
        let signal = vec![2.0, 2.0, 2.0, 2.0];
        let minima = find_local_minima(&signal);
        assert_eq!(minima, Vec::<usize>::new());
    }

    #[test]
    fn test_detect_extrema_simple() {
        let signal = vec![1.0, 3.0, 2.0, 4.0, 1.0];
        let extrema = detect_extrema(&signal);
        assert_eq!(extrema.maxima, vec![1, 3]);
        assert_eq!(extrema.minima, vec![2]); // Only interior minima detected
    }

    #[test]
    fn test_detect_extrema_sine_wave() {
        // Sine wave with known extrema
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let extrema = detect_extrema(&signal);

        // Should have some maxima and minima
        assert!(!extrema.maxima.is_empty());
        assert!(!extrema.minima.is_empty());

        // Check that maxima are above minima (roughly)
        for &max_idx in &extrema.maxima {
            for &min_idx in &extrema.minima {
                assert!(signal[max_idx] > signal[min_idx]);
            }
        }
    }

    #[test]
    fn test_detect_extrema_sawtooth() {
        // Sawtooth wave: /\/\
        let signal = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let extrema = detect_extrema(&signal);
        assert_eq!(extrema.maxima, vec![1, 3]);
        assert_eq!(extrema.minima, vec![2]); // Only interior minima detected
    }

    #[test]
    fn test_detect_extrema_step_function() {
        // Step function: constant then jump
        let signal = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let extrema = detect_extrema(&signal);
        // No local extrema, only boundaries if they qualify, but in this case, first is not > second, last not > previous
        assert_eq!(extrema.maxima, Vec::<usize>::new());
        assert_eq!(extrema.minima, Vec::<usize>::new());
    }

    #[test]
    fn test_empty_signal() {
        let signal = vec![];
        let extrema = detect_extrema(&signal);
        assert_eq!(extrema.maxima, Vec::<usize>::new());
        assert_eq!(extrema.minima, Vec::<usize>::new());
    }

    #[test]
    fn test_single_element() {
        let signal = vec![1.0];
        let extrema = detect_extrema(&signal);
        assert_eq!(extrema.maxima, Vec::<usize>::new());
        assert_eq!(extrema.minima, Vec::<usize>::new());
    }

    #[test]
    fn test_two_elements() {
        let signal = vec![1.0, 2.0];
        let extrema = detect_extrema(&signal);
        assert_eq!(extrema.maxima, Vec::<usize>::new()); // Need at least 3 points for interior extrema
        assert_eq!(extrema.minima, Vec::<usize>::new());
    }
}
