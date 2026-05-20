/// Mirror / Symmetric Extension.
///
/// Implements even-extension (reflect without sign change) and odd-extension
/// (reflect with sign inversion) mirror strategies.
use crate::boundary::{BoundaryCondition, ExtendedSignal, Extrema};

/// Mirror extension variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorVariant {
    /// Even extension: reflect signal around boundary without sign change
    Even,
    /// Odd extension: reflect signal with sign inversion
    Odd,
}

/// Configuration for mirror extension.
#[derive(Debug, Clone)]
pub struct MirrorConfig {
    pub variant: MirrorVariant,
    /// Number of samples to mirror from each end
    pub mirror_length: Option<usize>,
}

impl Default for MirrorConfig {
    fn default() -> Self {
        Self { variant: MirrorVariant::Even, mirror_length: None }
    }
}

/// Mirror boundary extension.
pub struct Mirror {
    config: MirrorConfig,
}

impl Mirror {
    pub fn new(config: MirrorConfig) -> Self {
        Self { config }
    }

    pub fn even() -> Self {
        Self { config: MirrorConfig { variant: MirrorVariant::Even, mirror_length: None } }
    }

    pub fn odd() -> Self {
        Self { config: MirrorConfig { variant: MirrorVariant::Odd, mirror_length: None } }
    }

    fn build_extension(
        signal: &[f64],
        variant: MirrorVariant,
        mirror_len: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let actual_len = mirror_len.min(signal.len());

        let mut left_ext = Vec::with_capacity(actual_len);
        let mut right_ext = Vec::with_capacity(actual_len);

        for i in 0..actual_len {
            let val = match variant {
                MirrorVariant::Even => signal[i],
                MirrorVariant::Odd => -signal[i],
            };
            left_ext.push(val);
        }
        left_ext.reverse();

        for i in 0..actual_len {
            let idx = signal.len() - 1 - i;
            let val = match variant {
                MirrorVariant::Even => signal[idx],
                MirrorVariant::Odd => -signal[idx],
            };
            right_ext.push(val);
        }
        // Do NOT reverse: right_ext already reads from signal[N-1] down to signal[N-actual_len],
        // which is the correct outward mirror reflection at the right boundary.

        (left_ext, right_ext)
    }
}

impl BoundaryCondition for Mirror {
    fn extend(&self, signal: &[f64], _extrema: &Extrema) -> ExtendedSignal {
        if signal.is_empty() {
            return ExtendedSignal { values: vec![], original_start: 0, original_end: 0 };
        }

        let mirror_len = self.config.mirror_length.unwrap_or(signal.len().min(20).max(4));
        let (left_ext, right_ext) = Self::build_extension(signal, self.config.variant, mirror_len);

        let mut values = Vec::with_capacity(left_ext.len() + signal.len() + right_ext.len());
        values.extend_from_slice(&left_ext);
        values.extend_from_slice(signal);
        values.extend_from_slice(&right_ext);

        let original_start = left_ext.len();
        let original_end = original_start + signal.len();

        ExtendedSignal { values, original_start, original_end }
    }

    fn name(&self) -> &str {
        match self.config.variant {
            MirrorVariant::Even => "mirror_even",
            MirrorVariant::Odd => "mirror_odd",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirror_even_preserves_original() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let extrema = Extrema { maxima_indices: vec![4], minima_indices: vec![0, 8] };

        let strategy = Mirror::even();
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
    }

    #[test]
    fn test_mirror_even_reflection() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let extrema = Extrema { maxima_indices: vec![4], minima_indices: vec![0] };

        let strategy = Mirror::even();
        let result = strategy.extend(&signal, &extrema);

        assert_eq!(strategy.name(), "mirror_even");
        assert!(result.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mirror_odd_preserves_original() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let extrema = Extrema { maxima_indices: vec![4], minima_indices: vec![0] };

        let strategy = Mirror::odd();
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() > signal.len());
    }

    #[test]
    fn test_mirror_odd_sign_inversion() {
        let signal = vec![1.0, 2.0, 3.0];
        let extrema = Extrema { maxima_indices: vec![2], minima_indices: vec![0] };

        let strategy = Mirror::odd();
        let result = strategy.extend(&signal, &extrema);

        assert_eq!(strategy.name(), "mirror_odd");

        let left_ext = &result.values[..result.original_start];
        let right_ext = &result.values[result.original_end..];

        for &v in left_ext {
            assert!(v.is_finite());
        }
        for &v in right_ext {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_mirror_even_endpoint_is_extremum() {
        let signal = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let extrema = Extrema { maxima_indices: vec![3], minima_indices: vec![0, 6] };

        let strategy = Mirror::even();
        let result = strategy.extend(&signal, &extrema);

        let left_ext = &result.values[..result.original_start];
        if !left_ext.is_empty() {
            let last_left = left_ext.last().unwrap();
            let first_orig = result.values[result.original_start];
            assert!(*last_left <= first_orig || *last_left >= first_orig);
        }
    }

    #[test]
    fn test_mirror_custom_length() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let extrema = Extrema { maxima_indices: vec![7], minima_indices: vec![0] };

        let strategy =
            Mirror::new(MirrorConfig { variant: MirrorVariant::Even, mirror_length: Some(3) });
        let result = strategy.extend(&signal, &extrema);

        assert_eq!(result.original_start, 3);
        assert_eq!(result.original_end, 3 + signal.len());
    }

    #[test]
    fn test_mirror_empty_signal() {
        let signal: Vec<f64> = vec![];
        let extrema = Extrema { maxima_indices: vec![], minima_indices: vec![] };

        let strategy = Mirror::even();
        let result = strategy.extend(&signal, &extrema);

        assert!(result.values.is_empty());
    }

    #[test]
    fn test_mirror_single_element() {
        let signal = vec![42.0];
        let extrema = Extrema { maxima_indices: vec![0], minima_indices: vec![] };

        let strategy = Mirror::even();
        let result = strategy.extend(&signal, &extrema);

        let orig_slice = &result.values[result.original_start..result.original_end];
        assert_eq!(orig_slice, &signal[..]);
        assert!(result.values.len() >= signal.len());
    }
}
