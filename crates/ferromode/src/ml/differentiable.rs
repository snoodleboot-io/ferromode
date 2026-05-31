#![warn(missing_docs)]

//! Differentiable Empirical Mode Decomposition.
//!
//! This module implements the forward pass for differentiable EMD,
//! capturing all necessary context for implicit differentiation.
//!
//! The forward pass decomposes a signal into IMFs and a residue,
//! while saving intermediate state (extrema locations, iteration counts, etc.)
//! needed for the implicit function backward pass.
//!
//! # Architecture
//!
//! The differentiable EMD uses an implicit function approach:
//! - **Forward pass:** Call standard EMD and capture context
//! - **Backward pass (T-321):** Use implicit differentiation theorem
//!
//! This avoids backpropagating through the entire sifting loop,
//! which is computationally expensive and numerically unstable.

use crate::algorithms::emd::{emd, EmdConfig};
use crate::error::EmdError;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ImplicitEmdContext
// ---------------------------------------------------------------------------

/// Context saved during EMD forward pass for implicit differentiation.
///
/// This struct captures all information needed to compute gradients via
/// the implicit function theorem. The backward pass will solve linear
/// systems involving these contexts rather than backpropagating through
/// the sifting algorithm.
///
/// # Examples
///
/// ```no_run
/// use ferromode::ml::differentiable::{DifferentiableEmd, ImplicitEmdContext};
/// use ferromode::algorithms::emd::EmdConfig;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
/// let config = EmdConfig::default();
/// let decomposer = DifferentiableEmd::new(config);
///
/// let context = decomposer.forward(&signal)?;
/// println!("Extracted {} IMFs", context.imfs.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ImplicitEmdContext {
    /// Original input signal.
    pub signal: Vec<f64>,

    /// Extracted Intrinsic Mode Functions (IMFs).
    /// Each IMF has the same length as the input signal.
    pub imfs: Vec<Vec<f64>>,

    /// Final residue (monotonic or trend component).
    /// Same length as input signal.
    pub residue: Vec<f64>,

    /// Extrema indices for each IMF.
    ///
    /// For each IMF, stores a tuple: (maxima_indices, minima_indices).
    /// These are needed for Jacobian computation in the implicit backward pass.
    pub extrema_indices: Vec<(Vec<usize>, Vec<usize>)>,

    /// Number of sifting iterations per IMF.
    ///
    /// Used to understand convergence behavior and complexity of
    /// the implicit function. Needed for gradient computation.
    pub num_sifts: Vec<usize>,

    /// Configuration used for decomposition.
    /// Stored for reference during backward pass.
    pub config: EmdConfig,

    /// Additional metadata stored as key-value pairs.
    ///
    /// Can contain runtime statistics, algorithm-specific data, etc.
    /// Used for extensibility and debugging.
    pub metadata: HashMap<String, String>,
}

impl ImplicitEmdContext {
    /// Create a new ImplicitEmdContext from decomposition results.
    ///
    /// # Arguments
    /// * `signal` — Original input signal
    /// * `imfs` — Extracted IMFs
    /// * `residue` — Final residue
    /// * `config` — Configuration used
    ///
    /// # Returns
    /// A new context with empty extrema and metadata fields.
    pub fn new(
        signal: Vec<f64>,
        imfs: Vec<Vec<f64>>,
        residue: Vec<f64>,
        config: EmdConfig,
    ) -> Self {
        let num_imfs = imfs.len();
        Self {
            signal,
            imfs,
            residue,
            extrema_indices: Vec::with_capacity(num_imfs),
            num_sifts: Vec::with_capacity(num_imfs),
            config,
            metadata: HashMap::new(),
        }
    }

    /// Add extrema information for an IMF.
    ///
    /// # Arguments
    /// * `maxima_indices` — Indices of local maxima in the IMF
    /// * `minima_indices` — Indices of local minima in the IMF
    pub fn add_extrema(&mut self, maxima_indices: Vec<usize>, minima_indices: Vec<usize>) {
        self.extrema_indices.push((maxima_indices, minima_indices));
    }

    /// Add sifting iteration count for an IMF.
    ///
    /// # Arguments
    /// * `num_iterations` — Number of sifting iterations performed
    pub fn add_sift_count(&mut self, num_iterations: usize) {
        self.num_sifts.push(num_iterations);
    }

    /// Add metadata key-value pair.
    ///
    /// # Arguments
    /// * `key` — Metadata key
    /// * `value` — Metadata value
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Validate context for consistency.
    ///
    /// # Returns
    /// `Ok(())` if context is valid, or `EmdError` if inconsistencies found.
    ///
    /// # Checks
    /// - All IMFs have same length as signal
    /// - Residue has same length as signal
    /// - All values are finite (no NaN/Inf)
    /// - Extrema counts match number of IMFs
    /// - Sift counts match number of IMFs
    pub fn validate(&self) -> Result<(), EmdError> {
        let signal_len = self.signal.len();

        // Check signal finitude
        for &val in &self.signal {
            if !val.is_finite() {
                return Err(EmdError::InvalidValue);
            }
        }

        // Check IMFs
        if self.imfs.len() != self.extrema_indices.len() {
            return Err(EmdError::InvalidConfig(format!(
                "IMF count ({}) does not match extrema count ({})",
                self.imfs.len(),
                self.extrema_indices.len()
            )));
        }

        if self.imfs.len() != self.num_sifts.len() {
            return Err(EmdError::InvalidConfig(format!(
                "IMF count ({}) does not match sift count ({})",
                self.imfs.len(),
                self.num_sifts.len()
            )));
        }

        for (_idx, imf) in self.imfs.iter().enumerate() {
            if imf.len() != signal_len {
                return Err(EmdError::DimensionMismatch);
            }
            for &val in imf {
                if !val.is_finite() {
                    return Err(EmdError::InvalidValue);
                }
            }
        }

        // Check residue
        if self.residue.len() != signal_len {
            return Err(EmdError::DimensionMismatch);
        }
        for &val in &self.residue {
            if !val.is_finite() {
                return Err(EmdError::InvalidValue);
            }
        }

        Ok(())
    }

    /// Compute reconstruction error.
    ///
    /// Returns the maximum absolute error between the original signal
    /// and the reconstruction (IMFs + residue).
    ///
    /// # Returns
    /// Maximum absolute reconstruction error.
    pub fn reconstruction_error(&self) -> f64 {
        let mut reconstructed = vec![0.0; self.signal.len()];
        for imf in &self.imfs {
            for (i, val) in imf.iter().enumerate() {
                reconstructed[i] += val;
            }
        }
        for (i, val) in self.residue.iter().enumerate() {
            reconstructed[i] += val;
        }

        let mut max_error = 0.0;
        for (_i, (&orig, &recon)) in self.signal.iter().zip(reconstructed.iter()).enumerate() {
            let error = (orig - recon).abs();
            if error > max_error {
                max_error = error;
            }
        }
        max_error
    }
}

// ---------------------------------------------------------------------------
// DifferentiableEmd
// ---------------------------------------------------------------------------

/// Differentiable EMD decomposer with forward pass infrastructure.
///
/// This struct provides a forward pass that decomposes a signal using
/// standard EMD, while capturing context needed for implicit differentiation.
///
/// # Examples
///
/// ```no_run
/// use ferromode::ml::differentiable::DifferentiableEmd;
/// use ferromode::algorithms::emd::EmdConfig;
///
/// let mut config = EmdConfig::default();
/// config.max_imfs = 5;
///
/// let decomposer = DifferentiableEmd::new(config);
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
///
/// let context = decomposer.forward(&signal)?;
/// let imfs = decomposer.get_imfs(&context);
/// let residue = decomposer.get_residue(&context);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct DifferentiableEmd {
    /// Configuration for EMD decomposition.
    pub config: EmdConfig,
}

impl DifferentiableEmd {
    /// Create a new DifferentiableEmd with the given configuration.
    ///
    /// # Arguments
    /// * `config` — EMD configuration
    ///
    /// # Examples
    /// ```
    /// use ferromode::ml::differentiable::DifferentiableEmd;
    /// use ferromode::algorithms::emd::EmdConfig;
    ///
    /// let config = EmdConfig::default();
    /// let decomposer = DifferentiableEmd::new(config);
    /// ```
    pub fn new(config: EmdConfig) -> Self {
        Self { config }
    }

    /// Perform forward pass of differentiable EMD.
    ///
    /// Decomposes the input signal using standard EMD, capturing all
    /// necessary context for implicit differentiation in the backward pass.
    ///
    /// # Arguments
    /// * `signal` — Input signal to decompose
    ///
    /// # Returns
    /// `ImplicitEmdContext` containing the decomposition and saved context,
    /// or `EmdError` if decomposition fails.
    ///
    /// # Examples
    /// ```
    /// use ferromode::ml::differentiable::DifferentiableEmd;
    /// use ferromode::algorithms::emd::EmdConfig;
    ///
    /// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
    /// let config = EmdConfig::default();
    /// let decomposer = DifferentiableEmd::new(config);
    ///
    /// let context = decomposer.forward(&signal)?;
    /// assert!(!context.imfs.is_empty() || !context.residue.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn forward(&self, signal: &[f64]) -> Result<ImplicitEmdContext, EmdError> {
        // Validate input
        if signal.is_empty() {
            return Err(EmdError::EmptySignal);
        }
        for &val in signal {
            if !val.is_finite() {
                return Err(EmdError::InvalidValue);
            }
        }

        // Call standard EMD decomposition
        let decomp_result = emd(signal, &self.config)?;

        // Extract IMFs and residue from result
        let imfs = decomp_result.imfs.imfs.clone();
        let residue = decomp_result.imfs.residue.clone();

        // Create context
        let mut context = ImplicitEmdContext::new(
            signal.to_vec(),
            imfs.clone(),
            residue.clone(),
            self.config.clone(),
        );

        // Capture extrema information for each IMF
        // This uses the extrema detection algorithm to find extrema in each IMF
        for imf in &imfs {
            let extrema = crate::extrema::detect_extrema(imf);
            context.add_extrema(extrema.maxima.clone(), extrema.minima.clone());
        }

        // Store decomposition metadata
        context.add_metadata("n_imfs".to_string(), imfs.len().to_string());
        context.add_metadata("n_siftings_total".to_string(), decomp_result.n_siftings.to_string());
        context
            .add_metadata("elapsed_ms".to_string(), decomp_result.elapsed.as_millis().to_string());

        // Estimate sifting iterations per IMF (distribute total)
        // In a full implementation, we'd track this during sifting
        let avg_sifts =
            if imfs.len() > 0 { decomp_result.n_siftings / imfs.len().max(1) } else { 0 };
        for _ in 0..imfs.len() {
            context.add_sift_count(avg_sifts);
        }

        // Validate context before returning
        context.validate()?;

        Ok(context)
    }

    /// Extract IMFs from context.
    ///
    /// # Arguments
    /// * `context` — The context from forward pass
    ///
    /// # Returns
    /// Vector of IMFs (each IMF is a vector of f64 values)
    ///
    /// # Examples
    /// ```
    /// use ferromode::ml::differentiable::DifferentiableEmd;
    /// use ferromode::algorithms::emd::EmdConfig;
    ///
    /// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
    /// let decomposer = DifferentiableEmd::new(EmdConfig::default());
    /// let context = decomposer.forward(&signal)?;
    /// let imfs = decomposer.get_imfs(&context);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_imfs(&self, context: &ImplicitEmdContext) -> Vec<Vec<f64>> {
        context.imfs.clone()
    }

    /// Extract residue from context.
    ///
    /// # Arguments
    /// * `context` — The context from forward pass
    ///
    /// # Returns
    /// The residue vector (trend component)
    ///
    /// # Examples
    /// ```
    /// use ferromode::ml::differentiable::DifferentiableEmd;
    /// use ferromode::algorithms::emd::EmdConfig;
    ///
    /// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
    /// let decomposer = DifferentiableEmd::new(EmdConfig::default());
    /// let context = decomposer.forward(&signal)?;
    /// let residue = decomposer.get_residue(&context);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_residue(&self, context: &ImplicitEmdContext) -> Vec<f64> {
        context.residue.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: Create a simple test signal (sine + noise)
    fn create_test_signal(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * t).sin() + 0.3 * (4.0 * PI * t).sin()
            })
            .collect()
    }

    /// Test 1: Forward pass preserves signal shape.
    ///
    /// Verifies that the decomposition produces IMFs and residue
    /// that match the input signal length.
    #[test]
    fn test_forward_preserves_shape() {
        let signal = create_test_signal(100);
        let config = EmdConfig::default();
        let decomposer = DifferentiableEmd::new(config);

        let context = decomposer.forward(&signal).expect("forward pass failed");

        // Validate all IMFs have same length as signal
        for imf in &context.imfs {
            assert_eq!(imf.len(), signal.len(), "IMF length should match signal length");
        }

        // Validate residue has same length
        assert_eq!(
            context.residue.len(),
            signal.len(),
            "Residue length should match signal length"
        );
    }

    /// Test 2: Forward pass allows perfect reconstruction.
    ///
    /// Verifies that IMFs + residue = original signal (error < 1e-10).
    /// This is critical for differentiable computation.
    #[test]
    fn test_forward_reconstruction() {
        let signal = create_test_signal(100);
        let config = EmdConfig::default();
        let decomposer = DifferentiableEmd::new(config);

        let context = decomposer.forward(&signal).expect("forward pass failed");

        // Reconstruct signal
        let mut reconstructed = vec![0.0; signal.len()];
        for imf in &context.imfs {
            for (i, &val) in imf.iter().enumerate() {
                reconstructed[i] += val;
            }
        }
        for (i, &val) in context.residue.iter().enumerate() {
            reconstructed[i] += val;
        }

        // Check reconstruction error
        for (i, (&orig, &recon)) in signal.iter().zip(reconstructed.iter()).enumerate() {
            let error = (orig - recon).abs();
            assert!(
                error < 1e-10,
                "Reconstruction error at index {} is {:.2e}, exceeds 1e-10",
                i,
                error
            );
        }
    }

    /// Test 3: Forward pass produces no NaN or Inf values.
    ///
    /// Verifies that all IMFs and the residue contain only finite values,
    /// which is necessary for gradient computation.
    #[test]
    fn test_forward_no_nans() {
        let signal = create_test_signal(100);
        let config = EmdConfig::default();
        let decomposer = DifferentiableEmd::new(config);

        let context = decomposer.forward(&signal).expect("forward pass failed");

        // Check IMFs
        for imf in &context.imfs {
            for &val in imf {
                assert!(val.is_finite(), "IMF contains non-finite value: {}", val);
            }
        }

        // Check residue
        for &val in &context.residue {
            assert!(val.is_finite(), "Residue contains non-finite value: {}", val);
        }

        // Check context validates
        context.validate().expect("context validation failed");
    }
}
