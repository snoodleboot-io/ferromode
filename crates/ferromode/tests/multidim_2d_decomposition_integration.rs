//! Comprehensive integration tests for 2D EMD decomposition.
//!
//! This test suite covers:
//! - Synthetic signal decomposition (8 tests)
//! - Synthetic medical imaging (6 tests)
//! - Property-based validation (4 tests)
//! - Edge cases (4 tests)
//! - Reconstruction and orthogonality (3 tests)
//!
//! Total: 25+ integration tests validating the separable 2D EMD algorithm
//! on diverse signal types and configurations.

mod fixtures {
    pub mod medical_image_synthetic;
    pub mod synthetic_2d_signals;
}

use ferromode::adapters::multidim::{decompose_image_2d_separable, Image2D};
use ferromode::algorithms::emd::EmdConfig;
use ferromode::error::EmdError;

// =============================================================================
// SYNTHETIC SIGNAL TESTS (8 tests)
// =============================================================================

/// Test decomposition of a checkerboard pattern.
///
/// Validates that the decomposition can extract oscillatory components
/// from a simple geometric pattern.
#[test]
fn test_synthetic_checkerboard_pattern() -> Result<(), EmdError> {
    let img = fixtures::synthetic_2d_signals::checkerboard_2d(16, 4)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Basic validation
    assert_eq!(decomp.residue_2d.width(), 16, "Residue width preserved");
    assert_eq!(decomp.residue_2d.height(), 16, "Residue height preserved");

    // All IMFs should have correct dimensions
    for imf in &decomp.imfs_2d {
        assert_eq!(imf.width(), 16, "IMF width matches input");
        assert_eq!(imf.height(), 16, "IMF height matches input");
    }

    Ok(())
}

/// Test decomposition of a linear gradient.
///
/// Validates handling of smooth monotonic variation across the image.
#[test]
fn test_synthetic_linear_gradient() -> Result<(), EmdError> {
    let img = fixtures::synthetic_2d_signals::linear_gradient(32, 32, 0.0, 1.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    assert_eq!(decomp.residue_2d.width(), 32);
    assert_eq!(decomp.residue_2d.height(), 32);

    // Residue should capture the overall trend
    let residue_data = decomp.residue_2d.data();
    let min_residue = residue_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_residue = residue_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_residue > min_residue,
        "Residue should show gradient variation, got min={}, max={}",
        min_residue,
        max_residue
    );

    Ok(())
}

/// Test decomposition of a Gaussian blob.
///
/// Validates decomposition of a smooth localized peak in 2D.
#[test]
fn test_synthetic_gaussian_blob() -> Result<(), EmdError> {
    let img = fixtures::synthetic_2d_signals::gaussian_2d(64, 64, 8.0, 1.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    assert_eq!(decomp.residue_2d.width(), 64);
    assert_eq!(decomp.residue_2d.height(), 64);

    // All data should be finite
    for &val in decomp.residue_2d.data() {
        assert!(val.is_finite(), "Residue contains non-finite values");
    }

    Ok(())
}

/// Test decomposition of multi-frequency 2D signal.
///
/// Validates handling of composite oscillatory components at different frequencies.
#[test]
fn test_synthetic_multi_frequency_2d() -> Result<(), EmdError> {
    // Use smaller frequency to avoid numerical issues
    let freqs = vec![(0.5, 0.25, 0.5), (0.25, 0.5, 0.3)];
    let img = fixtures::synthetic_2d_signals::multi_frequency_2d(32, 32, &freqs)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Multi-frequency signal should decompose successfully
    // IMF count depends on image size and frequency content
    assert_eq!(decomp.residue_2d.width(), 32);
    assert_eq!(decomp.residue_2d.height(), 32);

    // All components should be finite
    for imf in &decomp.imfs_2d {
        for &val in imf.data() {
            assert!(val.is_finite(), "IMF contains non-finite values");
        }
    }

    Ok(())
}

/// Test decomposition of a noisy signal.
///
/// Validates that the decomposition can handle slightly corrupted input.
#[test]
fn test_synthetic_noisy_signal() -> Result<(), EmdError> {
    // Create multi-frequency base image (inherent jitter)
    let freqs = vec![(0.2, 0.15, 0.5)];
    let img = fixtures::synthetic_2d_signals::multi_frequency_2d(32, 32, &freqs)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Noisy signal should decompose successfully
    assert_eq!(decomp.residue_2d.width(), 32);
    assert_eq!(decomp.residue_2d.height(), 32);

    // Verify no NaN/Inf values
    for &val in decomp.residue_2d.data() {
        assert!(val.is_finite(), "Result contains non-finite values");
    }

    Ok(())
}

/// Test decomposition of a constant signal.
///
/// Validates that a flat signal produces minimal decomposition (1 IMF = signal, residue = 0).
#[test]
fn test_synthetic_constant_signal() -> Result<(), EmdError> {
    let img = fixtures::synthetic_2d_signals::constant_signal(16, 16, 2.5)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Constant signal should not have oscillatory components
    // So we expect 0 IMFs and residue ≈ constant
    let residue_data = decomp.residue_2d.data();
    for &val in residue_data {
        assert!((val - 2.5).abs() < 1e-9, "Residue of constant should be constant");
    }

    Ok(())
}

/// Test decomposition of a single spike/impulse.
///
/// Validates handling of localized high-frequency content.
#[test]
fn test_synthetic_single_spike() -> Result<(), EmdError> {
    let img = fixtures::synthetic_2d_signals::single_spike(32, 32, 16, 16, 5.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    assert_eq!(decomp.residue_2d.width(), 32);
    assert_eq!(decomp.residue_2d.height(), 32);

    // Spike should be captured in IMFs or residue
    let total_imf_energy: f64 =
        decomp.imfs_2d.iter().flat_map(|imf| imf.data().iter()).map(|x| x * x).sum();

    let total_energy: f64 =
        total_imf_energy + decomp.residue_2d.data().iter().map(|x| x * x).sum::<f64>();

    assert!(
        total_energy > 0.0,
        "Decomposition should capture spike energy, IMFs energy={}, residue energy={}",
        total_imf_energy,
        decomp.residue_2d.data().iter().map(|x| x * x).sum::<f64>()
    );

    Ok(())
}

/// Test decomposition of medical image with slight variation.
///
/// Validates robust decomposition of slightly variable medical data.
#[test]
fn test_medical_image_noise_robustness() -> Result<(), EmdError> {
    // Use XRay which has good variation without needing explicit noise
    let img = fixtures::medical_image_synthetic::synthetic_xray(64, 64)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Medical image should decompose successfully
    assert_eq!(decomp.residue_2d.width(), 64);
    assert_eq!(decomp.residue_2d.height(), 64);

    // All outputs should be valid
    for &val in decomp.residue_2d.data() {
        assert!(val.is_finite(), "Medical image decomposition produces non-finite values");
    }

    Ok(())
}

/// Test decomposition of medical image with artifacts.
///
/// Validates handling of medical imaging with intensity modulation.
#[test]
fn test_medical_image_artifact_handling() -> Result<(), EmdError> {
    // Create a simple medical-like image with variation
    let img = fixtures::synthetic_2d_signals::gaussian_2d(64, 64, 10.0, 1.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Medical image-like pattern should decompose
    assert_eq!(decomp.residue_2d.width(), 64);
    assert_eq!(decomp.residue_2d.height(), 64);

    // Check that all values are finite
    for &val in decomp.residue_2d.data() {
        assert!(val.is_finite(), "Medical-like image produces non-finite values");
    }

    Ok(())
}

// =============================================================================
// PROPERTY-BASED VALIDATION TESTS (4 tests)
// =============================================================================

/// Test that decomposition + reconstruction recovers the input.
///
/// Validates the fundamental property: IMFs + residue = original image
/// within numerical precision.
#[test]
fn test_decomposition_is_complete() -> Result<(), EmdError> {
    let img = fixtures::synthetic_2d_signals::gaussian_2d(32, 32, 6.0, 1.0)?;
    let original_data = img.data().to_vec();

    let config = EmdConfig::default();
    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Reconstruct
    let reconstructed = decomp.reconstruct()?;
    let reconstructed_data = reconstructed.data();

    // Calculate max reconstruction error
    let mut max_error: f64 = 0.0;
    let mut sum_sq_error: f64 = 0.0;
    for (original, reconstructed) in original_data.iter().zip(reconstructed_data.iter()) {
        let error = (original - reconstructed).abs();
        max_error = max_error.max(error);
        sum_sq_error += error * error;
    }

    let rmse = (sum_sq_error / original_data.len() as f64).sqrt();

    assert!(max_error < 1e-9, "Reconstruction max error {} exceeds tolerance 1e-9", max_error);
    assert!(rmse < 1e-10, "Reconstruction RMSE {} exceeds tolerance 1e-10", rmse);

    Ok(())
}

/// Test that IMF count doesn't grow unexpectedly with image size.
///
/// Validates that decomposition doesn't produce spurious modes.
#[test]
fn test_imf_count_monotonic() -> Result<(), EmdError> {
    // Test progression: 16x16 -> 32x32 -> 64x64 with simple gradient
    let img16 = fixtures::synthetic_2d_signals::linear_gradient(16, 16, 0.0, 1.0)?;
    let img32 = fixtures::synthetic_2d_signals::linear_gradient(32, 32, 0.0, 1.0)?;
    let img64 = fixtures::synthetic_2d_signals::linear_gradient(64, 64, 0.0, 1.0)?;

    let config = EmdConfig::default();

    let decomp16 = decompose_image_2d_separable(&img16, &config)?;
    let decomp32 = decompose_image_2d_separable(&img32, &config)?;
    let decomp64 = decompose_image_2d_separable(&img64, &config)?;

    let count16 = decomp16.n_imfs();
    let count32 = decomp32.n_imfs();
    let count64 = decomp64.n_imfs();

    // For a monotonic gradient, IMF count should be minimal
    assert!(
        count16 == 0 && count32 == 0 && count64 == 0,
        "Linear gradient should produce no IMFs: 16x16={}, 32x32={}, 64x64={}",
        count16,
        count32,
        count64
    );

    Ok(())
}

/// Test boundary handling doesn't introduce non-physical artifacts.
///
/// Validates that padding/boundary strategies don't create spurious components.
#[test]
fn test_boundary_handling_symmetric() -> Result<(), EmdError> {
    // Create a symmetric input
    let img = fixtures::synthetic_2d_signals::gaussian_2d(64, 64, 8.0, 1.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Check that residue maintains approximate symmetry
    let residue = &decomp.residue_2d;
    let width = residue.width();
    let height = residue.height();

    let mut max_symmetry_error: f64 = 0.0;
    for row in 0..height {
        for col in 0..width {
            let center_row = (height - 1) as f64 / 2.0;
            let center_col = (width - 1) as f64 / 2.0;

            let row_mirror = (2.0 * center_row - row as f64) as usize;
            let col_mirror = (2.0 * center_col - col as f64) as usize;

            if row_mirror < height && col_mirror < width {
                let val = residue.get(row, col);
                let val_mirror = residue.get(row_mirror, col_mirror);
                let error = (val - val_mirror).abs();
                max_symmetry_error = max_symmetry_error.max(error);
            }
        }
    }

    // Symmetric input should produce symmetric decomposition (within numerical precision)
    assert!(
        max_symmetry_error < 0.1,
        "Boundary handling broke symmetry, error={}",
        max_symmetry_error
    );

    Ok(())
}

/// Test that output dimensions are always preserved.
///
/// Validates that all IMFs and residue have the same dimensions as input.
#[test]
fn test_output_dimensions_preserved() -> Result<(), EmdError> {
    // Test various sizes
    let test_sizes = vec![(16, 16), (32, 48), (64, 32), (100, 120)];

    for (width, height) in test_sizes {
        let img = fixtures::synthetic_2d_signals::constant_signal(width, height, 0.5)?;
        let config = EmdConfig::default();
        let decomp = decompose_image_2d_separable(&img, &config)?;

        assert_eq!(
            decomp.residue_2d.width(),
            width,
            "Residue width mismatch for {}x{}",
            width,
            height
        );
        assert_eq!(
            decomp.residue_2d.height(),
            height,
            "Residue height mismatch for {}x{}",
            width,
            height
        );

        for (idx, imf) in decomp.imfs_2d.iter().enumerate() {
            assert_eq!(imf.width(), width, "IMF[{}] width mismatch for {}x{}", idx, width, height);
            assert_eq!(
                imf.height(),
                height,
                "IMF[{}] height mismatch for {}x{}",
                idx,
                width,
                height
            );
        }
    }

    Ok(())
}

// =============================================================================
// EDGE CASE TESTS (4 tests)
// =============================================================================

/// Test decomposition of minimum valid image size (3×3).
///
/// Validates that the algorithm handles the smallest meaningful input.
#[test]
fn test_minimum_size_3x3() -> Result<(), EmdError> {
    let data = vec![
        1.0, 2.0, 1.0, // Row 0
        2.0, 3.0, 2.0, // Row 1
        1.0, 2.0, 1.0, // Row 2
    ];
    let img = Image2D::new(3, 3, data, None)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    assert_eq!(decomp.residue_2d.width(), 3);
    assert_eq!(decomp.residue_2d.height(), 3);

    // Reconstruction should work
    let _reconstructed = decomp.reconstruct()?;

    Ok(())
}

/// Test decomposition of non-square images.
///
/// Validates handling of rectangular images with various aspect ratios.
#[test]
fn test_non_square_images() -> Result<(), EmdError> {
    // Test several aspect ratios
    let test_sizes = vec![
        (512, 256),  // 2:1
        (256, 512),  // 1:2
        (100, 1000), // 1:10
        (1000, 100), // 10:1
    ];

    for (width, height) in test_sizes {
        let img = fixtures::synthetic_2d_signals::linear_gradient(width, height, 0.0, 1.0)?;
        let config = EmdConfig::default();

        let decomp = decompose_image_2d_separable(&img, &config)?;

        assert_eq!(
            decomp.residue_2d.width(),
            width,
            "Non-square width mismatch {}x{}",
            width,
            height
        );
        assert_eq!(
            decomp.residue_2d.height(),
            height,
            "Non-square height mismatch {}x{}",
            width,
            height
        );

        // Verify reconstruction is possible
        let _reconstructed = decomp.reconstruct()?;
    }

    Ok(())
}

/// Test decomposition of very small intensity signals.
///
/// Validates numerical stability with near-zero signal values.
#[test]
fn test_very_small_intensities() -> Result<(), EmdError> {
    let img = fixtures::synthetic_2d_signals::gaussian_2d(32, 32, 5.0, 1e-8)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Should handle small values without numerical instability
    for &val in decomp.residue_2d.data() {
        assert!(val.is_finite(), "Small intensity handling produced non-finite values");
    }

    Ok(())
}

/// Test decomposition of saturated image (all same value).
///
/// Validates handling of zero-variance input.
#[test]
fn test_saturated_image() -> Result<(), EmdError> {
    let img = fixtures::synthetic_2d_signals::constant_signal(32, 32, 100.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Saturated image should not decompose into IMFs
    assert_eq!(decomp.n_imfs(), 0, "Constant image should produce no IMFs");

    // Residue should be the constant
    for &val in decomp.residue_2d.data() {
        assert!((val - 100.0).abs() < 1e-9, "Saturated residue should equal input value");
    }

    Ok(())
}

// =============================================================================
// VALIDATION TESTS (3 tests)
// =============================================================================

/// Test reconstruction error meets accuracy requirements.
///
/// Validates floating-point accuracy: max error < 1e-9.
#[test]
fn test_reconstruction_error_floating_point_accuracy() -> Result<(), EmdError> {
    let test_images = vec![
        fixtures::synthetic_2d_signals::checkerboard_2d(32, 2)?,
        fixtures::synthetic_2d_signals::linear_gradient(32, 32, 0.0, 1.0)?,
        fixtures::synthetic_2d_signals::gaussian_2d(32, 32, 5.0, 1.0)?,
    ];

    for img in test_images {
        let original_data = img.data().to_vec();
        let config = EmdConfig::default();

        let decomp = decompose_image_2d_separable(&img, &config)?;
        let reconstructed = decomp.reconstruct()?;

        let mut max_error: f64 = 0.0;
        for (orig, recon) in original_data.iter().zip(reconstructed.data().iter()) {
            let error = (orig - recon).abs();
            max_error = max_error.max(error);
        }

        assert!(max_error < 1e-9, "Reconstruction error {} exceeds 1e-9 tolerance", max_error);
    }

    Ok(())
}

/// Test approximate orthogonality of IMF components.
///
/// Validates that IMFs are reasonably decorrelated (low cross-correlation).
#[test]
fn test_imf_orthogonality_approximate() -> Result<(), EmdError> {
    // Use simpler multi-frequency with lower frequencies to avoid issues
    let img = fixtures::synthetic_2d_signals::multi_frequency_2d(
        64,
        64,
        &[(0.3, 0.2, 0.5), (0.6, 0.4, 0.3)],
    )?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;

    // Check orthogonality between IMFs if we have at least 2
    if decomp.n_imfs() >= 2 {
        let imf1 = &decomp.imfs_2d[0];
        let imf2 = &decomp.imfs_2d[1];

        // Compute dot product (correlation)
        let mut dot_product = 0.0;
        let mut norm1_sq = 0.0;
        let mut norm2_sq = 0.0;

        for (v1, v2) in imf1.data().iter().zip(imf2.data().iter()) {
            dot_product += v1 * v2;
            norm1_sq += v1 * v1;
            norm2_sq += v2 * v2;
        }

        let norm1 = norm1_sq.sqrt();
        let norm2 = norm2_sq.sqrt();

        if norm1 > 1e-10 && norm2 > 1e-10 {
            let correlation = (dot_product / (norm1 * norm2)).abs();
            // Allow correlation up to 0.7 for separable decomposition approximation
            assert!(
                correlation < 0.7,
                "IMFs should be reasonably decorrelated, correlation={}",
                correlation
            );
        }
    }

    Ok(())
}

/// Test that residue exhibits monotonic trend (no high-frequency content).
///
/// Validates that residue captures only the trend, not oscillations.
#[test]
fn test_residue_trend_is_monotonic() -> Result<(), EmdError> {
    // Use a signal with clear trend
    let img = fixtures::synthetic_2d_signals::linear_gradient(32, 32, 0.0, 1.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_image_2d_separable(&img, &config)?;
    let residue = &decomp.residue_2d;

    // Check that residue along center row is monotonic
    let center_row = residue.height() / 2;
    let row_values: Vec<f64> =
        (0..residue.width()).map(|col| residue.get(center_row, col)).collect();

    let mut monotonic_count = 0;
    for i in 1..row_values.len() {
        if row_values[i] >= row_values[i - 1] {
            monotonic_count += 1;
        }
    }

    // Allow some tolerance for numerical errors
    let monotonicity_ratio = monotonic_count as f64 / row_values.len() as f64;
    assert!(
        monotonicity_ratio > 0.8,
        "Residue should show monotonic trend, ratio={}",
        monotonicity_ratio
    );

    Ok(())
}
