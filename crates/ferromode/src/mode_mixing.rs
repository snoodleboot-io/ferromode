#![warn(missing_docs)]

//! Mode Mixing Detection and Analysis
//!
//! Detects and quantifies mode mixing in EMD decomposition using frequency-domain
//! overlap metrics. Mode mixing occurs when two or more IMFs share overlapping
//! frequency content, indicating loss of unique frequency representation.
//!
//! # Frequency Domain Overlap Metric
//!
//! The overlap between two IMFs is computed by:
//! 1. Computing the FFT of each IMF to obtain power spectral density
//! 2. Normalizing spectra to [0, 1]
//! 3. Computing the intersection area (sum of minimum values at each frequency)
//! 4. Normalizing by mean spectrum energy to get overlap in [0, 1]
//!
//! # Interpretation Thresholds
//!
//! - **OK**: overlap < 0.1 — No significant mode mixing detected
//! - **Caution**: 0.1 ≤ overlap ≤ 0.3 — Moderate mixing, may indicate early stopping
//! - **Serious**: overlap > 0.3 — Significant mixing, likely sifting issue

use crate::error::EmdError;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// =========================================================================
// Data Structures
// =========================================================================

/// Result of mode mixing analysis for a decomposition
#[derive(Debug, Clone)]
pub struct ModeMixingAnalysis {
    /// Matrix of overlap scores between each IMF pair (symmetric)
    pub overlap_matrix: Vec<Vec<f64>>,
    /// Maximum overlap found in the matrix
    pub max_overlap: f64,
    /// List of IMF pairs with significant mixing (sorted by overlap descending)
    pub mixed_pairs: Vec<MixedPair>,
}

/// A pair of IMFs with identified mode mixing
#[derive(Debug, Clone, PartialEq)]
pub struct MixedPair {
    /// Index of first IMF
    pub imf_idx_1: usize,
    /// Index of second IMF
    pub imf_idx_2: usize,
    /// Overlap score in [0.0, 1.0]
    pub overlap: f64,
    /// Severity level of the mixing
    pub severity: MixingSeverity,
}

/// Severity level classification for mode mixing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MixingSeverity {
    /// overlap < 0.1: No significant mode mixing
    Ok,
    /// 0.1 ≤ overlap ≤ 0.3: Moderate mode mixing
    Caution,
    /// overlap > 0.3: Significant mode mixing
    Serious,
}

/// Interpretation of mode mixing with recommendations
#[derive(Debug, Clone)]
pub struct ModeMixingInterpretation {
    /// Overall severity assessment
    pub overall_severity: MixingSeverity,
    /// List of problematic pairs
    pub problematic_pairs: Vec<MixedPair>,
    /// Interpretation message
    pub interpretation: String,
    /// Recommended remediation steps
    pub remediation: Vec<String>,
}

// =========================================================================
// Core Functions: Overlap Metric Computation (T-336)
// =========================================================================

/// Compute mode mixing overlap matrix for a set of IMFs
///
/// Uses FFT-based power spectral density to measure frequency-domain overlap
/// between each pair of IMFs. Overlap is computed as the normalized intersection
/// area of the two power spectra.
///
/// # Arguments
/// * `imfs` — Collection of intrinsic mode functions (each is a time series)
///
/// # Returns
/// * `Ok(ModeMixingAnalysis)` — Analysis with overlap matrix and mixed pairs
/// * `Err(EmdError)` — If IMFs are empty or contain non-finite values
///
/// # Algorithm
/// 1. Compute FFT of each IMF
/// 2. Convert to power spectral density (magnitude² / N)
/// 3. Normalize each spectrum to unit energy
/// 4. For each pair: compute intersection area = Σ min(psd1[f], psd2[f])
/// 5. Classify mixing severity and return results
///
/// # Complexity
/// O(N × M log M) where N = number of IMFs, M = samples per IMF
pub fn compute_mode_mixing_overlap(imfs: &[Vec<f64>]) -> Result<ModeMixingAnalysis, EmdError> {
    // Validate inputs
    if imfs.is_empty() {
        return Ok(ModeMixingAnalysis {
            overlap_matrix: vec![],
            max_overlap: 0.0,
            mixed_pairs: vec![],
        });
    }

    let n_imfs = imfs.len();
    let n_samples = imfs[0].len();

    if n_samples == 0 {
        return Err(EmdError::EmptySignal);
    }

    // Validate all IMFs have same length
    for imf in imfs {
        if imf.len() != n_samples {
            return Err(EmdError::DimensionMismatch);
        }
        // Check for finite values
        if !imf.iter().all(|v| v.is_finite()) {
            return Err(EmdError::InvalidValue);
        }
    }

    // Compute power spectra for each IMF
    let mut spectra = Vec::with_capacity(n_imfs);
    for imf in imfs {
        let spectrum = compute_power_spectrum(imf)?;
        spectra.push(spectrum);
    }

    // Compute overlap matrix
    let mut overlap_matrix = vec![vec![0.0; n_imfs]; n_imfs];
    let mut max_overlap = 0.0;
    let mut mixed_pairs = Vec::new();

    for i in 0..n_imfs {
        for j in (i + 1)..n_imfs {
            let overlap = compute_spectrum_overlap(&spectra[i], &spectra[j]);
            overlap_matrix[i][j] = overlap;
            overlap_matrix[j][i] = overlap; // symmetric

            if overlap > max_overlap {
                max_overlap = overlap;
            }

            // If overlap is significant, add to mixed pairs
            if overlap > 0.05 {
                // Threshold for recording (slightly below caution threshold)
                let severity = classify_mixing_severity(overlap);
                mixed_pairs.push(MixedPair { imf_idx_1: i, imf_idx_2: j, overlap, severity });
            }
        }
    }

    // Sort mixed pairs by overlap descending
    mixed_pairs
        .sort_by(|a, b| b.overlap.partial_cmp(&a.overlap).unwrap_or(std::cmp::Ordering::Equal));

    Ok(ModeMixingAnalysis { overlap_matrix, max_overlap, mixed_pairs })
}

/// Compute power spectral density of a signal using FFT
fn compute_power_spectrum(signal: &[f64]) -> Result<Vec<f64>, EmdError> {
    if signal.is_empty() {
        return Err(EmdError::EmptySignal);
    }

    let n = signal.len();
    // Pad to next power of 2 for efficiency
    let n_fft = n.next_power_of_two();

    // Create complex input with padding
    let mut input: Vec<Complex<f64>> = signal.iter().map(|&v| Complex { re: v, im: 0.0 }).collect();
    input.resize(n_fft, Complex { re: 0.0, im: 0.0 });

    // Compute FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    fft.process(&mut input);

    // Compute power (magnitude squared) and normalize
    let mut power: Vec<f64> = input.iter().map(|c| (c.norm_sqr()) / n as f64).collect();

    // Keep only positive frequencies (0 to Nyquist)
    power.truncate(n_fft / 2 + 1);

    // Normalize to unit energy
    let total_energy: f64 = power.iter().sum();
    if total_energy > 1e-12 {
        for p in &mut power {
            *p /= total_energy;
        }
    }

    Ok(power)
}

/// Compute overlap between two power spectra
///
/// Overlap is computed as the sum of minimum values at each frequency:
/// overlap = Σ_f min(psd1[f], psd2[f])
///
/// This represents the intersection area of the two spectra and is
/// normalized to [0, 1] since both spectra are normalized to unit energy.
fn compute_spectrum_overlap(spectrum1: &[f64], spectrum2: &[f64]) -> f64 {
    let min_len = spectrum1.len().min(spectrum2.len());
    if min_len == 0 {
        return 0.0;
    }

    let overlap: f64 =
        spectrum1[..min_len].iter().zip(&spectrum2[..min_len]).map(|(s1, s2)| s1.min(*s2)).sum();

    overlap.max(0.0).min(1.0)
}

/// Classify mixing severity based on overlap value
fn classify_mixing_severity(overlap: f64) -> MixingSeverity {
    match overlap {
        o if o < 0.1 => MixingSeverity::Ok,
        o if o <= 0.3 => MixingSeverity::Caution,
        _ => MixingSeverity::Serious,
    }
}

// =========================================================================
// Heatmap Data Structure (T-337)
// =========================================================================

/// Heatmap representation of mode mixing for visualization
#[derive(Debug, Clone)]
pub struct ModeMixingHeatmap {
    /// The underlying analysis
    pub analysis: ModeMixingAnalysis,
}

impl ModeMixingHeatmap {
    /// Create a heatmap from raw overlap data
    pub fn new(analysis: ModeMixingAnalysis) -> Self {
        Self { analysis }
    }

    /// Get all overlap values as a flat vector
    pub fn get_all_overlaps(&self) -> Vec<f64> {
        let mut overlaps = Vec::new();
        for i in 0..self.analysis.overlap_matrix.len() {
            for j in (i + 1)..self.analysis.overlap_matrix[i].len() {
                overlaps.push(self.analysis.overlap_matrix[i][j]);
            }
        }
        overlaps
    }

    /// Get the N most mixed IMF pairs
    pub fn get_most_mixed_pairs(&self, n: usize) -> Vec<MixedPair> {
        self.analysis.mixed_pairs.iter().take(n).cloned().collect()
    }

    /// Generate human-readable severity report
    pub fn generate_severity_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Mode Mixing Analysis Report\n");
        report.push_str("===========================\n\n");

        report.push_str(&format!("Overall Max Overlap: {:.4}\n", self.analysis.max_overlap));

        if self.analysis.mixed_pairs.is_empty() {
            report.push_str("\n✓ No significant mode mixing detected.\n");
            return report;
        }

        report.push_str(&format!(
            "\nDetected {} problematic pairs:\n",
            self.analysis.mixed_pairs.len()
        ));

        let mut ok_count = 0;
        let mut caution_count = 0;
        let mut serious_count = 0;

        for pair in &self.analysis.mixed_pairs {
            let severity_str = match pair.severity {
                MixingSeverity::Ok => {
                    ok_count += 1;
                    "OK"
                }
                MixingSeverity::Caution => {
                    caution_count += 1;
                    "⚠️ CAUTION"
                }
                MixingSeverity::Serious => {
                    serious_count += 1;
                    "🚨 SERIOUS"
                }
            };

            report.push_str(&format!(
                "  IMF[{}] <-> IMF[{}]: {:.4} ({})\n",
                pair.imf_idx_1, pair.imf_idx_2, pair.overlap, severity_str
            ));
        }

        report.push_str(&format!(
            "\nSummary: {} OK, {} Caution, {} Serious\n",
            ok_count, caution_count, serious_count
        ));

        if serious_count > 0 {
            report.push_str(
                "\n⚠️  Serious mode mixing detected. Consider:\n\
                 - Adjusting sifting stopping criterion\n\
                 - Using different boundary condition method\n\
                 - Verifying signal preprocessing\n",
            );
        }

        report
    }

    /// Get the overlap matrix as a vector of vectors for export
    pub fn as_matrix(&self) -> &Vec<Vec<f64>> {
        &self.analysis.overlap_matrix
    }

    /// Number of IMFs analyzed
    pub fn n_imfs(&self) -> usize {
        self.analysis.overlap_matrix.len()
    }
}

// =========================================================================
// Integration Function: Analysis from Decomposition
// =========================================================================

/// Compute mode mixing analysis from a set of IMFs
///
/// Combines overlap computation and interpretation into a single operation.
pub fn analyze_mode_mixing(imfs: &[Vec<f64>]) -> Result<ModeMixingInterpretation, EmdError> {
    let analysis = compute_mode_mixing_overlap(imfs)?;

    let overall_severity = classify_mixing_severity(analysis.max_overlap);

    let problematic_pairs: Vec<MixedPair> =
        analysis.mixed_pairs.iter().filter(|p| p.severity != MixingSeverity::Ok).cloned().collect();

    let (interpretation, remediation) = generate_interpretation_text(
        overall_severity,
        analysis.max_overlap,
        problematic_pairs.len(),
    );

    Ok(ModeMixingInterpretation {
        overall_severity,
        problematic_pairs,
        interpretation,
        remediation,
    })
}

/// Generate interpretation text and remediation recommendations
fn generate_interpretation_text(
    severity: MixingSeverity,
    max_overlap: f64,
    num_problematic: usize,
) -> (String, Vec<String>) {
    let interpretation = match severity {
        MixingSeverity::Ok => {
            format!(
                "Decomposition quality is good. Max overlap of {:.4} indicates \
                 well-separated frequency components with minimal mode mixing.",
                max_overlap
            )
        }
        MixingSeverity::Caution => {
            format!(
                "Moderate mode mixing detected (max overlap {:.4}) in {} IMF pairs. \
                 Consider adjusting sifting parameters or boundary conditions for improved separation.",
                max_overlap, num_problematic
            )
        }
        MixingSeverity::Serious => {
            format!(
                "Significant mode mixing detected (max overlap {:.4}) affecting {} IMF pairs. \
                 This indicates loss of unique frequency content. Review sifting stopping criterion \
                 and boundary condition implementation.",
                max_overlap, num_problematic
            )
        }
    };

    let remediation = match severity {
        MixingSeverity::Ok => {
            vec!["Continue with current decomposition parameters for similar signals.".to_string()]
        }
        MixingSeverity::Caution => vec![
            "Increase EMD.minsiftiter (minimum sifting iterations) to ensure convergence."
                .to_string(),
            "Try alternative boundary conditions (e.g., SYMMETRIC, PERIODIC, PREDICTION)."
                .to_string(),
            "Verify that the signal preprocessing (filtering, normalization) is appropriate."
                .to_string(),
        ],
        MixingSeverity::Serious => vec![
            "Review the sifting stopping criterion - it may have terminated too early.".to_string(),
            "Check boundary condition implementation for edge artifacts.".to_string(),
            "Consider signal preprocessing: noise reduction, detrending, or resampling."
                .to_string(),
            "Try EEMD (Ensemble EMD) to reduce mode mixing via averaging.".to_string(),
            "Verify input signal validity - check for outliers or data corruption.".to_string(),
        ],
    };

    (interpretation, remediation)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_power_spectrum_valid() {
        let signal = vec![1.0, -1.0, 1.0, -1.0];
        let spectrum = compute_power_spectrum(&signal).unwrap();
        assert!(!spectrum.is_empty());
        assert!(spectrum.iter().all(|s| s.is_finite() && *s >= 0.0));
    }

    #[test]
    fn test_compute_power_spectrum_empty() {
        let signal = vec![];
        let result = compute_power_spectrum(&signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectrum_overlap_identical_signals() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let spectrum = compute_power_spectrum(&signal).unwrap();
        let overlap = compute_spectrum_overlap(&spectrum, &spectrum);
        assert!((overlap - 1.0).abs() < 0.01, "Identical spectra should overlap ~1.0");
    }

    #[test]
    fn test_spectrum_overlap_orthogonal_signals() {
        // Low frequency: single cycle in 4 samples
        let low_freq = vec![0.0, 1.0, 0.0, -1.0];
        // High frequency: two complete cycles
        let high_freq = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];

        let spectrum_low = compute_power_spectrum(&low_freq).unwrap();
        let spectrum_high = compute_power_spectrum(&high_freq).unwrap();
        let overlap = compute_spectrum_overlap(&spectrum_low, &spectrum_high);

        // Orthogonal signals should have low overlap
        assert!(overlap < 0.3, "Orthogonal spectra should have low overlap, got {}", overlap);
    }

    #[test]
    fn test_classify_mixing_severity() {
        assert_eq!(classify_mixing_severity(0.05), MixingSeverity::Ok);
        assert_eq!(classify_mixing_severity(0.1), MixingSeverity::Caution);
        assert_eq!(classify_mixing_severity(0.2), MixingSeverity::Caution);
        assert_eq!(classify_mixing_severity(0.3), MixingSeverity::Caution);
        assert_eq!(classify_mixing_severity(0.5), MixingSeverity::Serious);
    }

    #[test]
    fn test_mode_mixing_empty_imfs() {
        let imfs = vec![];
        let analysis = compute_mode_mixing_overlap(&imfs).unwrap();
        assert!(analysis.overlap_matrix.is_empty());
        assert_eq!(analysis.max_overlap, 0.0);
        assert!(analysis.mixed_pairs.is_empty());
    }

    #[test]
    fn test_mode_mixing_single_imf() {
        let imf = vec![1.0, 2.0, 3.0, 4.0];
        let analysis = compute_mode_mixing_overlap(&[imf]).unwrap();
        assert_eq!(analysis.overlap_matrix.len(), 1);
        assert_eq!(analysis.overlap_matrix[0].len(), 1);
        assert!(analysis.mixed_pairs.is_empty());
    }

    #[test]
    fn test_mode_mixing_dimension_mismatch() {
        let imf1 = vec![1.0, 2.0, 3.0];
        let imf2 = vec![1.0, 2.0];
        let result = compute_mode_mixing_overlap(&[imf1, imf2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_heatmap_basic() {
        let signal1 = vec![1.0, -1.0, 1.0, -1.0];
        let signal2 = vec![0.0, 1.0, 0.0, -1.0];
        let analysis = compute_mode_mixing_overlap(&[signal1, signal2]).unwrap();
        let heatmap = ModeMixingHeatmap::new(analysis);

        assert_eq!(heatmap.n_imfs(), 2);
        assert!(heatmap.analysis.max_overlap >= 0.0 && heatmap.analysis.max_overlap <= 1.0);
    }

    #[test]
    fn test_heatmap_most_mixed_pairs() {
        let signals = vec![
            vec![1.0, -1.0, 1.0, -1.0],
            vec![1.0, -1.0, 1.0, -1.0], // identical to first
            vec![0.0, 1.0, 0.0, -1.0],
        ];
        let analysis = compute_mode_mixing_overlap(&signals).unwrap();
        let heatmap = ModeMixingHeatmap::new(analysis);

        let top_pairs = heatmap.get_most_mixed_pairs(2);
        assert!(top_pairs.len() <= 2);
        if top_pairs.len() > 1 {
            assert!(top_pairs[0].overlap >= top_pairs[1].overlap);
        }
    }

    #[test]
    fn test_analyze_mode_mixing_interpretation() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let interpretation = analyze_mode_mixing(&[signal.clone()]).unwrap();

        assert!(!interpretation.interpretation.is_empty());
        assert!(!interpretation.remediation.is_empty());
    }
}
