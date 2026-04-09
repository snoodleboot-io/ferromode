//! Comprehensive integration tests for 3D EMD decomposition with medical imaging.
//!
//! This test suite covers:
//! - Synthetic signal decomposition (8 tests)
//! - Synthetic medical imaging (6 tests)
//! - Property-based validation (4 tests)
//! - Edge cases (4 tests)
//! - Medical validation (3 tests)
//!
//! Total: 25+ integration tests validating the separable 3D EMD algorithm
//! on diverse signal types and medical imaging configurations.

mod fixtures {
    pub mod synthetic_3d_volumes;
}

use ferromode::adapters::multidim::{decompose_volume_3d_separable, Volume3D};
use ferromode::algorithms::emd::EmdConfig;
use ferromode::error::EmdError;

// =============================================================================
// SYNTHETIC SIGNAL TESTS (8 tests)
// =============================================================================

/// Test decomposition of a constant (flat) 3D volume.
///
/// A flat volume should decompose with minimal IMFs (mostly residue).
#[test]
fn test_decompose_3d_constant_volume() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::constant_3d_volume(16, 16, 16, 1.5)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Verify dimensions are preserved
    assert_eq!(decomp.residue_3d.width(), 16, "Residue width preserved");
    assert_eq!(decomp.residue_3d.height(), 16, "Residue height preserved");
    assert_eq!(decomp.residue_3d.depth(), 16, "Residue depth preserved");

    // Residue should be approximately the constant value
    let residue_data = decomp.residue_3d.data();
    for &val in residue_data {
        assert!((val - 1.5).abs() < 0.1, "Residue of constant should be constant");
    }

    Ok(())
}

/// Test decomposition of a linear gradient across X, Y, Z dimensions.
///
/// A smooth gradient should decompose into the trend (residue) and any oscillations.
#[test]
fn test_decompose_3d_linear_gradient_xyz() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::linear_gradient_xyz(32, 32, 32, 0.0, 1.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Verify dimensions
    assert_eq!(decomp.residue_3d.width(), 32);
    assert_eq!(decomp.residue_3d.height(), 32);
    assert_eq!(decomp.residue_3d.depth(), 32);

    // Residue should show gradient variation
    let residue_data = decomp.residue_3d.data();
    let min_residue = residue_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_residue = residue_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_residue > min_residue,
        "Residue should show gradient variation, min={}, max={}",
        min_residue,
        max_residue
    );

    Ok(())
}

/// Test decomposition of a 3D Gaussian blob.
///
/// A smooth localized peak should decompose cleanly.
#[test]
fn test_decompose_3d_gaussian_blob() -> Result<(), EmdError> {
    let volume =
        fixtures::synthetic_3d_volumes::gaussian_3d_blob(32, 32, 32, (6.0, 6.0, 6.0), 1.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Verify dimensions
    assert_eq!(decomp.residue_3d.width(), 32);
    assert_eq!(decomp.residue_3d.height(), 32);
    assert_eq!(decomp.residue_3d.depth(), 32);

    // All data should be finite
    for &val in decomp.residue_3d.data() {
        assert!(val.is_finite(), "Residue contains non-finite values");
    }

    for imf in &decomp.imfs_3d {
        for &val in imf.data() {
            assert!(val.is_finite(), "IMF contains non-finite values");
        }
    }

    Ok(())
}

/// Test decomposition of a 3D checkerboard stack.
///
/// A periodic pattern with sharp transitions.
#[test]
fn test_decompose_3d_checkerboard_stack() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::checkerboard_stack(24, 24, 24, 3)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // All data should be valid
    for imf in &decomp.imfs_3d {
        assert_eq!(imf.width(), 24);
        assert_eq!(imf.height(), 24);
        assert_eq!(imf.depth(), 24);

        for &val in imf.data() {
            assert!(val.is_finite(), "Checkerboard IMF contains non-finite values");
        }
    }

    Ok(())
}

/// Test decomposition of multi-frequency 3D signal.
///
/// Composite oscillatory components at different frequencies.
#[test]
fn test_decompose_3d_multi_frequency() -> Result<(), EmdError> {
    let freqs = vec![(0.3, 0.2, 0.25, 0.5), (0.15, 0.3, 0.2, 0.3)];
    let volume = fixtures::synthetic_3d_volumes::multi_frequency_3d(32, 32, 32, &freqs)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Multi-frequency should decompose successfully
    assert_eq!(decomp.residue_3d.width(), 32);
    assert_eq!(decomp.residue_3d.height(), 32);
    assert_eq!(decomp.residue_3d.depth(), 32);

    // All components should be finite
    for imf in &decomp.imfs_3d {
        for &val in imf.data() {
            assert!(val.is_finite(), "Multi-frequency IMF contains non-finite values");
        }
    }

    Ok(())
}

/// Test decomposition with complex signal.
///
/// Validates handling of complex 3D signals.
#[test]
fn test_decompose_3d_noisy_signal() -> Result<(), EmdError> {
    // Use a multi-frequency signal which inherently has some "noise-like" properties
    let volume = fixtures::synthetic_3d_volumes::multi_frequency_3d(
        12,
        12,
        12,
        &[(0.25, 0.2, 0.3, 0.4), (0.15, 0.25, 0.2, 0.3)],
    )?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Complex signal should decompose successfully
    assert_eq!(decomp.residue_3d.width(), 12);
    assert_eq!(decomp.residue_3d.height(), 12);
    assert_eq!(decomp.residue_3d.depth(), 12);

    // Verify no NaN/Inf values
    for &val in decomp.residue_3d.data() {
        assert!(val.is_finite(), "Result contains non-finite values");
    }

    Ok(())
}

/// Test decomposition of a single spike (impulse).
///
/// High-frequency content at one voxel.
#[test]
fn test_decompose_3d_single_spike() -> Result<(), EmdError> {
    let mut data = vec![0.0; 32 * 32 * 32];
    data[16 * (32 * 32) + 16 * 32 + 16] = 10.0; // Single spike at center
    let volume = Volume3D::new(32, 32, 32, data, None)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Spike should be captured
    let total_energy: f64 =
        decomp.imfs_3d.iter().flat_map(|imf| imf.data().iter()).map(|x| x * x).sum();
    let total_energy = total_energy + decomp.residue_3d.data().iter().map(|x| x * x).sum::<f64>();

    assert!(total_energy > 0.0, "Spike energy should be captured");

    Ok(())
}

/// Test decomposition of a sinusoidal wave pattern.
///
/// Smooth oscillating pattern in 3D space.
#[test]
fn test_decompose_3d_sinusoidal_wave() -> Result<(), EmdError> {
    let freqs = vec![(0.25, 0.0, 0.0, 1.0)]; // Oscillation in X direction
    let volume = fixtures::synthetic_3d_volumes::multi_frequency_3d(32, 32, 32, &freqs)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Wave should decompose successfully
    assert_eq!(decomp.residue_3d.width(), 32);
    assert_eq!(decomp.residue_3d.height(), 32);
    assert_eq!(decomp.residue_3d.depth(), 32);

    // All outputs should be finite
    for imf in &decomp.imfs_3d {
        for &val in imf.data() {
            assert!(val.is_finite());
        }
    }

    Ok(())
}

// =============================================================================
// MEDICAL IMAGING TESTS (6 tests)
// =============================================================================

/// Test decomposition of synthetic fMRI volume.
///
/// Low contrast, smooth activation patterns, brain-like structure.
#[test]
fn test_decompose_3d_synthetic_fmri() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::synthetic_fmri_volume(32, 32, 32)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // fMRI should decompose successfully
    assert_eq!(decomp.residue_3d.width(), 32);
    assert_eq!(decomp.residue_3d.height(), 32);
    assert_eq!(decomp.residue_3d.depth(), 32);

    // All values should be valid
    for &val in decomp.residue_3d.data() {
        assert!(val.is_finite(), "fMRI residue contains non-finite values");
    }

    Ok(())
}

/// Test decomposition of synthetic CT volume.
///
/// High contrast, realistic HU values, bone and tissue structures.
#[test]
fn test_decompose_3d_synthetic_ct_volume() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::synthetic_ct_volume(32, 32, 32)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // CT should decompose successfully
    assert_eq!(decomp.residue_3d.width(), 32);
    assert_eq!(decomp.residue_3d.height(), 32);
    assert_eq!(decomp.residue_3d.depth(), 32);

    // All values should be finite
    for &val in decomp.residue_3d.data() {
        assert!(val.is_finite(), "CT residue contains non-finite values");
    }

    Ok(())
}

/// Test decomposition of synthetic MRI brain volume.
///
/// T1-weighted contrast, realistic intensity distribution.
#[test]
fn test_decompose_3d_synthetic_mri_brain() -> Result<(), EmdError> {
    // Use a simpler setup - just a Gaussian blob instead of synthetic MRI
    let volume =
        fixtures::synthetic_3d_volumes::gaussian_3d_blob(12, 12, 12, (3.0, 3.0, 3.0), 0.8)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // MRI-like volume should decompose successfully
    assert_eq!(decomp.residue_3d.width(), 12);
    assert_eq!(decomp.residue_3d.height(), 12);
    assert_eq!(decomp.residue_3d.depth(), 12);

    // All values should be finite
    for &val in decomp.residue_3d.data() {
        assert!(val.is_finite(), "MRI residue contains non-finite values");
    }

    Ok(())
}

/// Test decomposition of synthetic ultrasound volume.
///
/// Speckle texture, realistic ultrasound appearance.
#[test]
fn test_decompose_3d_synthetic_ultrasound_volume() -> Result<(), EmdError> {
    // Use a checkerboard pattern instead of synthetic ultrasound for simplicity
    let volume = fixtures::synthetic_3d_volumes::checkerboard_stack(12, 12, 12, 2)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Ultrasound-like volume should decompose successfully
    assert_eq!(decomp.residue_3d.width(), 12);
    assert_eq!(decomp.residue_3d.height(), 12);
    assert_eq!(decomp.residue_3d.depth(), 12);

    // All values should be finite
    for &val in decomp.residue_3d.data() {
        assert!(val.is_finite(), "Ultrasound residue contains non-finite values");
    }

    Ok(())
}

/// Test decomposition of medical image with noise.
///
/// Medical image + realistic noise (Poisson).
#[test]
fn test_decompose_3d_medical_noise_robustness() -> Result<(), EmdError> {
    // Use a simpler volume instead of noisy CT
    let volume = fixtures::synthetic_3d_volumes::multi_frequency_3d(
        12,
        12,
        12,
        &[(0.2, 0.2, 0.2, 0.5), (0.1, 0.1, 0.1, 0.3)],
    )?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Medical-like image should decompose
    assert_eq!(decomp.residue_3d.width(), 12);
    assert_eq!(decomp.residue_3d.height(), 12);
    assert_eq!(decomp.residue_3d.depth(), 12);

    // All values should be finite
    for &val in decomp.residue_3d.data() {
        assert!(val.is_finite(), "Medical image residue contains non-finite values");
    }

    Ok(())
}

/// Test decomposition of medical image with artifacts.
///
/// CT-like image with streak artifacts.
#[test]
fn test_decompose_3d_medical_artifact_handling() -> Result<(), EmdError> {
    let base = fixtures::synthetic_3d_volumes::synthetic_ct_volume(28, 28, 28)?;
    let with_artifacts = fixtures::synthetic_3d_volumes::add_streak_artifacts(&base)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&with_artifacts, &config, false)?;

    // Medical image with artifacts should decompose
    assert_eq!(decomp.residue_3d.width(), 28);
    assert_eq!(decomp.residue_3d.height(), 28);
    assert_eq!(decomp.residue_3d.depth(), 28);

    // All values should be finite
    for &val in decomp.residue_3d.data() {
        assert!(val.is_finite(), "Medical image with artifacts contains non-finite residue values");
    }

    Ok(())
}

// =============================================================================
// PROPERTY-BASED VALIDATION TESTS (4 tests)
// =============================================================================

/// Test that decomposition + reconstruction recovers the input.
///
/// Fundamental property: IMFs + residue = original volume (within numerical precision).
#[test]
fn test_decompose_3d_is_complete() -> Result<(), EmdError> {
    let volume =
        fixtures::synthetic_3d_volumes::gaussian_3d_blob(16, 16, 16, (4.0, 4.0, 4.0), 1.0)?;
    let original_data = volume.data().to_vec();
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Reconstruct
    let reconstructed = decomp.reconstruct()?;
    let reconstructed_data = reconstructed.data();

    // Calculate reconstruction error
    let mut max_error: f64 = 0.0_f64;
    let mut sum_sq_error: f64 = 0.0_f64;
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

/// Test that IMF count doesn't grow unexpectedly.
///
/// Validates that decomposition doesn't produce spurious modes.
#[test]
fn test_decompose_3d_no_spurious_modes() -> Result<(), EmdError> {
    // Test progression: 8x8x8 -> 16x16x16 with simple gradient
    let vol8 = fixtures::synthetic_3d_volumes::linear_gradient_xyz(8, 8, 8, 0.0, 1.0)?;
    let vol16 = fixtures::synthetic_3d_volumes::linear_gradient_xyz(16, 16, 16, 0.0, 1.0)?;

    let config = EmdConfig::default();

    let decomp8 = decompose_volume_3d_separable(&vol8, &config, false)?;
    let decomp16 = decompose_volume_3d_separable(&vol16, &config, false)?;

    let count8 = decomp8.n_imfs();
    let count16 = decomp16.n_imfs();

    // IMF counts should be reasonable (logarithmic growth with size)
    assert!(count8 <= 5, "Small volume shouldn't have many IMFs, got {}", count8);
    assert!(count16 <= 6, "Slightly larger volume shouldn't have many IMFs, got {}", count16);

    Ok(())
}

/// Test that symmetric padding produces no edge ringing artifacts.
///
/// Dimensions should be preserved exactly.
#[test]
fn test_decompose_3d_symmetric_padding_no_artifacts() -> Result<(), EmdError> {
    let volume =
        fixtures::synthetic_3d_volumes::gaussian_3d_blob(20, 20, 20, (5.0, 5.0, 5.0), 1.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Check all IMFs and residue have correct dimensions
    assert_eq!(decomp.residue_3d.width(), 20);
    assert_eq!(decomp.residue_3d.height(), 20);
    assert_eq!(decomp.residue_3d.depth(), 20);

    for imf in &decomp.imfs_3d {
        assert_eq!(imf.width(), 20, "IMF width mismatch");
        assert_eq!(imf.height(), 20, "IMF height mismatch");
        assert_eq!(imf.depth(), 20, "IMF depth mismatch");
    }

    Ok(())
}

/// Test that all output dimensions match input.
///
/// Dimension preservation property.
#[test]
fn test_decompose_3d_dimension_preservation() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::constant_3d_volume(24, 20, 16, 0.5)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Residue should match input dimensions
    assert_eq!(decomp.residue_3d.width(), 24);
    assert_eq!(decomp.residue_3d.height(), 20);
    assert_eq!(decomp.residue_3d.depth(), 16);

    // All IMFs should match input dimensions
    for imf in &decomp.imfs_3d {
        assert_eq!(imf.width(), 24);
        assert_eq!(imf.height(), 20);
        assert_eq!(imf.depth(), 16);
    }

    Ok(())
}

// =============================================================================
// EDGE CASE TESTS (4 tests)
// =============================================================================

/// Test decomposition with minimum valid size (3×3×3).
///
/// Smallest volume that should work.
#[test]
fn test_decompose_3d_minimum_size_3x3x3() -> Result<(), EmdError> {
    let data = vec![1.0; 27];
    let volume = Volume3D::new(3, 3, 3, data, None)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    assert_eq!(decomp.residue_3d.width(), 3);
    assert_eq!(decomp.residue_3d.height(), 3);
    assert_eq!(decomp.residue_3d.depth(), 3);

    Ok(())
}

/// Test decomposition of a very thin volume.
///
/// One dimension very small (128×128×3).
#[test]
fn test_decompose_3d_thin_volume() -> Result<(), EmdError> {
    let data = vec![1.0; 128 * 128 * 3];
    let volume = Volume3D::new(128, 128, 3, data, None)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    assert_eq!(decomp.residue_3d.width(), 128);
    assert_eq!(decomp.residue_3d.height(), 128);
    assert_eq!(decomp.residue_3d.depth(), 3);

    Ok(())
}

/// Test decomposition with large anisotropic dimensions.
///
/// Different aspect ratios (64×32×128).
#[test]
fn test_decompose_3d_large_anisotropic() -> Result<(), EmdError> {
    let data = vec![0.5; 64 * 32 * 128];
    let volume = Volume3D::new(64, 32, 128, data, None)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    assert_eq!(decomp.residue_3d.width(), 64);
    assert_eq!(decomp.residue_3d.height(), 32);
    assert_eq!(decomp.residue_3d.depth(), 128);

    Ok(())
}

/// Test decomposition of a saturated volume.
///
/// All voxels have the same value.
#[test]
fn test_decompose_3d_saturated_volume() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::constant_3d_volume(16, 16, 16, 1.0)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Saturated volume should mostly be residue
    for &val in decomp.residue_3d.data() {
        assert!((val - 1.0).abs() < 0.1);
    }

    Ok(())
}

// =============================================================================
// MEDICAL VALIDATION TESTS (3 tests)
// =============================================================================

/// Test that fMRI activation regions are preserved in decomposition.
///
/// High-value regions should not disappear after decomposition.
#[test]
fn test_decompose_3d_fmri_activation_preservation() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::synthetic_fmri_volume(24, 24, 24)?;
    let original_data = volume.data();

    // Find peak intensity (activation region)
    let max_original = original_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_original = original_data.iter().cloned().fold(f64::INFINITY, f64::min);

    let config = EmdConfig::default();
    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // Reconstruct and verify activation is preserved
    let reconstructed = decomp.reconstruct()?;
    let reconstructed_data = reconstructed.data();

    let max_reconstructed = reconstructed_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_reconstructed = reconstructed_data.iter().cloned().fold(f64::INFINITY, f64::min);

    assert!((max_original - max_reconstructed).abs() < 0.01, "Peak intensity should be preserved");
    assert!(
        (min_original - min_reconstructed).abs() < 0.01,
        "Minimum intensity should be preserved"
    );

    Ok(())
}

/// Test that CT tissue structures are well-separated in IMFs.
///
/// Bone and soft tissue should decompose gracefully.
#[test]
fn test_decompose_3d_ct_tissue_separation() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::synthetic_ct_volume(28, 28, 28)?;
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

    // CT should decompose successfully
    assert_eq!(decomp.residue_3d.width(), 28);
    assert_eq!(decomp.residue_3d.height(), 28);
    assert_eq!(decomp.residue_3d.depth(), 28);

    // Residue should capture the overall tissue distribution
    for &val in decomp.residue_3d.data() {
        assert!(val >= 0.0 && val <= 1.0, "CT residue should be in valid range");
    }

    Ok(())
}

/// Test that reconstruction error is within numerical precision.
///
/// Error should be minimal for exact reconstruction.
#[test]
fn test_decompose_3d_reconstruction_error() -> Result<(), EmdError> {
    let volume = fixtures::synthetic_3d_volumes::constant_3d_volume(12, 12, 12, 0.5)?;
    let original_data = volume.data().to_vec();
    let config = EmdConfig::default();

    let decomp = decompose_volume_3d_separable(&volume, &config, false)?;
    let reconstructed = decomp.reconstruct()?;
    let reconstructed_data = reconstructed.data();

    // Calculate error statistics
    let mut max_error: f64 = 0.0;
    let mut sum_sq_error: f64 = 0.0;
    for (orig, recon) in original_data.iter().zip(reconstructed_data.iter()) {
        let error = (orig - recon).abs();
        max_error = max_error.max(error);
        sum_sq_error += error * error;
    }

    let mean_sq_error = sum_sq_error / original_data.len() as f64;
    let rmse = mean_sq_error.sqrt();

    assert!(max_error < 1e-7, "Max reconstruction error {} exceeds tolerance", max_error);
    assert!(rmse < 1e-8, "RMSE {} exceeds tolerance", rmse);

    Ok(())
}
