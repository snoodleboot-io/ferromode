//! Synthetic 3D volume generators for testing medical imaging.
//!
//! This module provides functions to generate various synthetic 3D volumes
//! for testing 3D EMD decomposition, including geometric patterns, mathematical
//! functions, medical imaging simulators, and noise-corrupted variants.

use ferromode::adapters::multidim::Volume3D;
use ferromode::error::EmdError;
use std::f64::consts::PI;

/// Generate a constant (flat) 3D volume.
///
/// All voxels have the same value.
///
/// # Arguments
///
/// * `width` - Volume width (X dimension)
/// * `height` - Volume height (Y dimension)
/// * `depth` - Volume depth (Z dimension)
/// * `value` - Constant value for all voxels
///
/// # Returns
///
/// A Volume3D with all voxels set to the given value.
pub fn constant_3d_volume(
    width: usize,
    height: usize,
    depth: usize,
    value: f64,
) -> Result<Volume3D, EmdError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(EmdError::InvalidConfig("Dimensions must be > 0".to_string()));
    }
    if !value.is_finite() {
        return Err(EmdError::InvalidValue);
    }

    let data = vec![value; width * height * depth];
    Volume3D::new(width, height, depth, data, None)
}

/// Generate a linear gradient across all three dimensions.
///
/// Values vary linearly from (0,0,0) to (width-1, height-1, depth-1).
///
/// # Arguments
///
/// * `width` - Volume width
/// * `height` - Volume height
/// * `depth` - Volume depth
/// * `start` - Value at origin
/// * `end` - Value at far corner
///
/// # Returns
///
/// A Volume3D with smooth linear gradient variation.
pub fn linear_gradient_xyz(
    width: usize,
    height: usize,
    depth: usize,
    start: f64,
    end: f64,
) -> Result<Volume3D, EmdError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(EmdError::InvalidConfig("Dimensions must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height * depth);
    let diagonal_max = ((width - 1).pow(2) + (height - 1).pow(2) + (depth - 1).pow(2)) as f64;
    let diagonal_max = diagonal_max.sqrt();

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let distance =
                    ((x as f64).powi(2) + (y as f64).powi(2) + (z as f64).powi(2)).sqrt();
                let t = if diagonal_max > 0.0 { distance / diagonal_max } else { 0.0 };
                let val = start + t * (end - start);
                data.push(val);
            }
        }
    }

    Volume3D::new(width, height, depth, data, None)
}

/// Generate a 3D Gaussian blob centered in the volume.
///
/// # Arguments
///
/// * `width` - Volume width
/// * `height` - Volume height
/// * `depth` - Volume depth
/// * `sigma_xyz` - Standard deviations for each dimension (sigma_x, sigma_y, sigma_z)
/// * `amplitude` - Peak amplitude
///
/// # Returns
///
/// A Volume3D with a Gaussian peak in the center.
pub fn gaussian_3d_blob(
    width: usize,
    height: usize,
    depth: usize,
    sigma_xyz: (f64, f64, f64),
    amplitude: f64,
) -> Result<Volume3D, EmdError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(EmdError::InvalidConfig("Dimensions must be > 0".to_string()));
    }

    let (sigma_x, sigma_y, sigma_z) = sigma_xyz;
    if sigma_x <= 0.0
        || sigma_y <= 0.0
        || sigma_z <= 0.0
        || !sigma_x.is_finite()
        || !sigma_y.is_finite()
        || !sigma_z.is_finite()
    {
        return Err(EmdError::InvalidValue);
    }

    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;
    let center_z = (depth - 1) as f64 / 2.0;
    let sigma_x_sq = sigma_x * sigma_x;
    let sigma_y_sq = sigma_y * sigma_y;
    let sigma_z_sq = sigma_z * sigma_z;

    let mut data = Vec::with_capacity(width * height * depth);
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - center_x;
                let dy = y as f64 - center_y;
                let dz = z as f64 - center_z;
                let r_sq = dx * dx / sigma_x_sq + dy * dy / sigma_y_sq + dz * dz / sigma_z_sq;
                let val = amplitude * (-r_sq / 2.0).exp();
                data.push(val);
            }
        }
    }

    Volume3D::new(width, height, depth, data, None)
}

/// Generate a 3D checkerboard stack (stack of alternating checkerboard patterns).
///
/// # Arguments
///
/// * `width` - Volume width
/// * `height` - Volume height
/// * `depth` - Volume depth
/// * `block_size` - Size of each checkerboard block
///
/// # Returns
///
/// A Volume3D with alternating 1.0 and 0.0 values in a 3D checkerboard pattern.
pub fn checkerboard_stack(
    width: usize,
    height: usize,
    depth: usize,
    block_size: usize,
) -> Result<Volume3D, EmdError> {
    if width == 0 || height == 0 || depth == 0 || block_size == 0 {
        return Err(EmdError::InvalidConfig("Dimensions must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height * depth);
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let x_block = x / block_size;
                let y_block = y / block_size;
                let z_block = z / block_size;
                let val = if (x_block + y_block + z_block) % 2 == 0 { 1.0 } else { 0.0 };
                data.push(val);
            }
        }
    }

    Volume3D::new(width, height, depth, data, None)
}

/// Generate a multi-frequency 3D signal (sum of 3D sinusoids).
///
/// # Arguments
///
/// * `width` - Volume width
/// * `height` - Volume height
/// * `depth` - Volume depth
/// * `freqs` - List of (freq_x, freq_y, freq_z, amplitude) tuples
///
/// # Returns
///
/// A Volume3D with composite of 3D sinusoids.
pub fn multi_frequency_3d(
    width: usize,
    height: usize,
    depth: usize,
    freqs: &[(f64, f64, f64, f64)],
) -> Result<Volume3D, EmdError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(EmdError::InvalidConfig("Dimensions must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height * depth);
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let vx = x as f64 / width as f64;
                let vy = y as f64 / height as f64;
                let vz = z as f64 / depth as f64;
                let mut val = 0.0;
                for &(freq_x, freq_y, freq_z, amp) in freqs {
                    val += amp
                        * (2.0 * PI * freq_x * vx).sin()
                        * (2.0 * PI * freq_y * vy).sin()
                        * (2.0 * PI * freq_z * vz).sin();
                }
                data.push(val);
            }
        }
    }

    Volume3D::new(width, height, depth, data, None)
}

/// Add Gaussian noise to a 3D volume.
///
/// # Arguments
///
/// * `volume` - Input volume
/// * `sigma` - Standard deviation of Gaussian noise
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A new Volume3D with added Gaussian noise.
pub fn add_gaussian_noise_3d(
    volume: &Volume3D,
    sigma: f64,
    seed: u64,
) -> Result<Volume3D, EmdError> {
    if sigma < 0.0 || !sigma.is_finite() {
        return Err(EmdError::InvalidValue);
    }

    let mut rng_state = seed;
    let mut data = volume.data().to_vec();

    for val in &mut data {
        // Box-Muller transform
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let u1 = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;

        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;

        let u1_safe = (u1 * 0.999 + 0.0005).clamp(0.001, 0.999);
        let z = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * PI * u2).cos();
        *val += sigma * z;
    }

    // Validate all data is finite
    for &v in &data {
        if !v.is_finite() {
            return Err(EmdError::InvalidValue);
        }
    }

    Volume3D::new(volume.width(), volume.height(), volume.depth(), data, volume.voxel_spacing())
}

/// Add Poisson noise to a 3D volume.
///
/// # Arguments
///
/// * `volume` - Input volume (assumed to be non-negative)
/// * `lambda` - Poisson parameter (typical: lambda ~ mean intensity)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A new Volume3D with added Poisson noise.
pub fn add_poisson_noise_3d(
    volume: &Volume3D,
    lambda: f64,
    seed: u64,
) -> Result<Volume3D, EmdError> {
    if lambda < 0.0 || !lambda.is_finite() {
        return Err(EmdError::InvalidValue);
    }

    let mut rng_state = seed;
    let mut data = volume.data().to_vec();

    for val in &mut data {
        // Simple Poisson approximation using Gaussian for large lambda
        let poisson_std = lambda.sqrt();
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let u = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let v = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;
        let z = (-2.0 * (u + 1e-10).ln()).sqrt() * (2.0 * PI * v).cos();
        // Add noise but keep values in reasonable range
        let noisy = *val * (1.0 + lambda * z / 10.0);
        *val = noisy.max(-1.0).min(2.0); // Clamp to avoid extreme values
    }

    // Validate all data is finite
    for &v in &data {
        if !v.is_finite() {
            return Err(EmdError::InvalidValue);
        }
    }

    Volume3D::new(volume.width(), volume.height(), volume.depth(), data, volume.voxel_spacing())
}

/// Generate a synthetic fMRI volume (functional MRI).
///
/// Simulates:
/// - Low contrast (typical fMRI: ~1-2% signal change)
/// - Smooth activation patterns
/// - Brain-like spatial structure
/// - Realistic intensity distribution
///
/// # Arguments
///
/// * `width` - Volume width
/// * `height` - Volume height
/// * `depth` - Volume depth
///
/// # Returns
///
/// A Volume3D simulating fMRI data.
pub fn synthetic_fmri_volume(
    width: usize,
    height: usize,
    depth: usize,
) -> Result<Volume3D, EmdError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(EmdError::InvalidConfig("Dimensions must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height * depth);
    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;
    let center_z = (depth - 1) as f64 / 2.0;

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - center_x;
                let dy = y as f64 - center_y;
                let dz = z as f64 - center_z;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let max_r = ((width / 2) as f64).max((height / 2) as f64).max((depth / 2) as f64);

                // Base brain tissue
                let mut val = 0.5;

                // Brain-like structure
                if r < max_r * 0.7 {
                    val = 0.6;
                }

                // Activation region (low contrast)
                if r < max_r * 0.3 {
                    val += 0.02; // 2% signal increase (realistic fMRI)
                }

                // Add spatial smoothness (Gaussian modulation)
                let smooth = (-r * r / (2.0 * max_r * max_r)).exp();
                val = val * 0.8 + 0.6 * smooth;

                data.push(val);
            }
        }
    }

    Volume3D::new(width, height, depth, data, None)
}

/// Generate a synthetic CT (computed tomography) volume.
///
/// Simulates:
/// - High contrast (bone ~3000 HU, soft tissue ~40-60 HU)
/// - Realistic HU value distribution (normalized to [0, 1])
/// - Bone and soft tissue structures
/// - Sharp intensity transitions
///
/// # Arguments
///
/// * `width` - Volume width
/// * `height` - Volume height
/// * `depth` - Volume depth
///
/// # Returns
///
/// A Volume3D simulating CT data.
pub fn synthetic_ct_volume(
    width: usize,
    height: usize,
    depth: usize,
) -> Result<Volume3D, EmdError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(EmdError::InvalidConfig("Dimensions must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height * depth);
    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;
    let center_z = (depth - 1) as f64 / 2.0;

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - center_x;
                let dy = y as f64 - center_y;
                let dz = z as f64 - center_z;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let max_r = ((width / 2) as f64).max((height / 2) as f64).max((depth / 2) as f64);

                // Background (air)
                let mut val: f64 = 0.0;

                // Outer soft tissue
                if r < max_r * 0.85 {
                    val = 0.4;
                }

                // Inner bone structure
                if r < max_r * 0.3 {
                    val = 0.9;
                }

                // Additional bone structures
                let bone1_dist = ((dx - max_r * 0.3).powi(2)
                    + (dy - max_r * 0.3).powi(2)
                    + (dz - max_r * 0.3).powi(2))
                .sqrt();
                if bone1_dist < max_r * 0.15 {
                    val = 0.85;
                }

                let bone2_dist = ((dx + max_r * 0.3).powi(2)
                    + (dy + max_r * 0.3).powi(2)
                    + (dz + max_r * 0.3).powi(2))
                .sqrt();
                if bone2_dist < max_r * 0.15 {
                    val = 0.85;
                }

                val = val.clamp(0.0_f64, 1.0_f64);
                data.push(val);
            }
        }
    }

    Volume3D::new(width, height, depth, data, None)
}

/// Generate a synthetic T1-weighted MRI brain volume.
///
/// Simulates:
/// - T1 contrast (CSF dark, gray matter medium, white matter bright)
/// - Realistic intensity distribution
/// - Brain-like anatomy
/// - CSF ventricles
///
/// # Arguments
///
/// * `width` - Volume width
/// * `height` - Volume height
/// * `depth` - Volume depth
///
/// # Returns
///
/// A Volume3D simulating T1-weighted MRI data.
pub fn synthetic_mri_brain_volume(
    width: usize,
    height: usize,
    depth: usize,
) -> Result<Volume3D, EmdError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(EmdError::InvalidConfig("Dimensions must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height * depth);
    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;
    let center_z = (depth - 1) as f64 / 2.0;

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - center_x;
                let dy = y as f64 - center_y;
                let dz = z as f64 - center_z;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let max_r = ((width / 2) as f64).max((height / 2) as f64).max((depth / 2) as f64);

                // Skull (very low)
                let mut val = 0.1;

                // Gray matter (medium)
                if r < max_r * 0.85 {
                    val = 0.6;
                }

                // White matter (high)
                if r < max_r * 0.6 {
                    val = 0.8;
                }

                // CSF/ventricles (low)
                if r < max_r * 0.2 {
                    val = 0.3;
                }

                // Add anatomical variation
                let theta = dy.atan2(dx);
                let detail = 0.05 * (4.0 * theta).sin();
                val = (val + detail).max(0.0).min(1.0);

                data.push(val);
            }
        }
    }

    Volume3D::new(width, height, depth, data, None)
}

/// Generate a synthetic 3D ultrasound volume.
///
/// Simulates:
/// - Speckle noise pattern (granular texture)
/// - Depth-dependent attenuation
/// - Tissue layer patterns
///
/// # Arguments
///
/// * `width` - Volume width
/// * `height` - Volume height
/// * `depth` - Volume depth
///
/// # Returns
///
/// A Volume3D with ultrasound-like speckle and texture.
pub fn synthetic_ultrasound_volume(
    width: usize,
    height: usize,
    depth: usize,
) -> Result<Volume3D, EmdError> {
    if width == 0 || height == 0 || depth == 0 {
        return Err(EmdError::InvalidConfig("Dimensions must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height * depth);
    let mut rng_state: u64 = 12345;

    for z in 0..depth {
        for y in 0..height {
            for _x in 0..width {
                // Depth attenuation
                let depth_attenuation = 1.0 - (z as f64 / depth as f64) * 0.4;

                // Speckle pattern
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let speckle = ((rng_state >> 16) & 0xff) as f64 / 255.0;

                // Layer pattern
                let layer_intensity = 0.5 * (2.0 * PI * y as f64 / 20.0).sin() + 0.5;

                let val = depth_attenuation * (0.4 * speckle + 0.6 * layer_intensity);
                data.push(val);
            }
        }
    }

    Volume3D::new(width, height, depth, data, None)
}

/// Add medical imaging artifacts (streaks, beam hardening).
///
/// # Arguments
///
/// * `volume` - Input volume
///
/// # Returns
///
/// A new Volume3D with added artifacts.
pub fn add_streak_artifacts(volume: &Volume3D) -> Result<Volume3D, EmdError> {
    let width = volume.width();
    let height = volume.height();
    let depth = volume.depth();
    let mut data = volume.data().to_vec();

    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;

    // Add metal artifacts (streaks)
    for z in 0..depth {
        for y in 0..height {
            for _x in 0..width {
                let _dx = _x as f64 - center_x;
                let dy = y as f64 - center_y;

                // Streak pattern
                let streak_strength = 0.1 * ((dy * dy).sqrt() / (center_y)).sin();
                let idx = z * (width * height) + y * width + _x;
                data[idx] = (data[idx] + streak_strength).min(1.0).max(0.0);
            }
        }
    }

    Volume3D::new(width, height, depth, data, None)
}
