//! Synthetic medical image generators for testing.
//!
//! This module provides functions to generate synthetic medical imaging data
//! that mimics real CT, MRI, ultrasound, and X-ray images for testing EMD
//! decomposition on medical data.

use ferromode::adapters::multidim::Image2D;
use ferromode::error::EmdError;
use std::f64::consts::PI;

/// Generate a synthetic CT (computed tomography) slice.
///
/// Simulates a CT scan with:
/// - Bone structures (high intensity ~3000 HU equivalent)
/// - Soft tissue (medium intensity ~40-60 HU equivalent)
/// - Air/background (low intensity ~-1000 HU equivalent)
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
///
/// An Image2D simulating a CT scan with bone and tissue contrast.
pub fn synthetic_ct_slice(width: usize, height: usize) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height);
    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;

    for row in 0..height {
        for col in 0..width {
            let x = col as f64 - center_x;
            let y = row as f64 - center_y;
            let r = (x * x + y * y).sqrt();
            let max_r = ((width / 2) as f64).max((height / 2) as f64);
            if max_r == 0.0 {
                return Err(EmdError::InvalidConfig("Image too small".to_string()));
            }

            // Background (air-like, low intensity) - normalized to [0, 1]
            let mut val: f64 = 0.0;

            // Outer soft tissue ring (medium intensity)
            if r < max_r * 0.8 {
                val = 0.4;
            }

            // Inner bone-like structure (high intensity)
            if r < max_r * 0.3 {
                val = 0.9;
            }

            // Add small bone-like structures at different positions
            let bone1_dist = ((x - max_r * 0.3).powi(2) + (y - max_r * 0.3).powi(2)).sqrt();
            if bone1_dist < max_r * 0.15 {
                val = 0.8;
            }

            let bone2_dist = ((x + max_r * 0.3).powi(2) + (y + max_r * 0.3).powi(2)).sqrt();
            if bone2_dist < max_r * 0.15 {
                val = 0.8;
            }

            // Ensure finite and in range
            val = val.clamp(0.0, 1.0);
            data.push(val);
        }
    }

    Image2D::new(width, height, data, None)
}

/// Generate a synthetic T1-weighted MRI slice.
///
/// Simulates T1-weighted MRI contrast with:
/// - CSF (cerebrospinal fluid, low intensity)
/// - Gray matter (medium-high intensity)
/// - White matter (high intensity)
/// - Skull/bone (very low intensity)
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
///
/// An Image2D simulating T1-weighted MRI contrast.
pub fn synthetic_mri_slice(width: usize, height: usize) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height);
    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;

    for row in 0..height {
        for col in 0..width {
            let x = col as f64 - center_x;
            let y = row as f64 - center_y;
            let r = (x * x + y * y).sqrt();
            let max_r = ((width / 2) as f64).max((height / 2) as f64);
            let theta = y.atan2(x);

            // Background (skull/bone, very low)
            let mut val = 0.1;

            // Outer brain tissue (gray matter, medium)
            if r < max_r * 0.85 {
                val = 0.6;
            }

            // White matter regions (high intensity)
            if r < max_r * 0.6 {
                val = 0.8;
            }

            // Central CSF region (low intensity, like ventricles)
            if r < max_r * 0.2 {
                val = 0.3;
            }

            // Add some anatomical detail with sinusoidal variation (keep within bounds)
            let anatomical_detail = 0.05 * (4.0 * theta).sin();
            val = (val + anatomical_detail).max(0.0).min(1.0);

            data.push(val);
        }
    }

    Image2D::new(width, height, data, None)
}

/// Generate a synthetic B-mode ultrasound image.
///
/// Simulates ultrasound with:
/// - Speckle noise pattern (granular texture)
/// - Linear texture from beam direction
/// - Echo intensity variations
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
///
/// An Image2D with ultrasound-like speckle and texture.
pub fn synthetic_ultrasound(width: usize, height: usize) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height);
    let mut rng_state: u64 = 12345;

    for row in 0..height {
        for _col in 0..width {
            // Depth-dependent intensity (typical for ultrasound)
            let depth_attenuation = 1.0 - (row as f64 / height as f64) * 0.4;

            // Speckle pattern (pseudo-random)
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let speckle = ((rng_state >> 16) & 0xff) as f64 / 255.0;

            // Sinusoidal texture (simulating tissue layers)
            let layer_intensity = 0.5 * (2.0 * PI * row as f64 / 20.0).sin() + 0.5;

            let val = depth_attenuation * (0.4 * speckle + 0.6 * layer_intensity);
            data.push(val);
        }
    }

    Image2D::new(width, height, data, None)
}

/// Generate a synthetic radiograph (X-ray) image.
///
/// Simulates X-ray radiograph with:
/// - Bone structures (very bright)
/// - Soft tissue (medium brightness)
/// - Air (dark)
/// - Continuous intensity gradients
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
///
/// An Image2D simulating radiograph intensity distribution.
pub fn synthetic_xray(width: usize, height: usize) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height);
    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;

    for row in 0..height {
        for col in 0..width {
            let x = col as f64 - center_x;
            let y = row as f64 - center_y;
            let r = (x * x + y * y).sqrt();
            let max_r = ((width / 2) as f64).max((height / 2) as f64);

            // Background (air, dark)
            let mut val: f64 = 0.2;

            // Soft tissue (body outline, medium brightness)
            if r < max_r * 0.7 {
                val = 0.5;
            }

            // Bone structures (bright)
            if r < max_r * 0.15 {
                val = 0.95;
            }

            // Additional bone structures (femur-like)
            let bone_y = y.abs();
            if bone_y > max_r * 0.2 && bone_y < max_r * 0.5 && x.abs() < max_r * 0.1 {
                val = 0.9;
            }

            data.push(val.min(1.0).max(0.0)); // Clamp to [0, 1]
        }
    }

    Image2D::new(width, height, data, None)
}

/// Add realistic medical imaging noise to an image.
///
/// Simulates common artifacts in medical images:
/// - Gaussian thermal noise
/// - Poisson shot noise
/// - Speckle noise
///
/// # Arguments
///
/// * `image` - Input medical image
/// * `noise_type` - Type of noise: "gaussian", "poisson", or "speckle"
/// * `noise_level` - Noise intensity (0.0-0.1 typical)
/// * `seed` - Random seed
///
/// # Returns
///
/// A new Image2D with added medical imaging noise.
pub fn add_medical_noise(
    image: &Image2D,
    noise_type: &str,
    noise_level: f64,
    seed: u64,
) -> Result<Image2D, EmdError> {
    if noise_level < 0.0 || noise_level > 1.0 || !noise_level.is_finite() {
        return Err(EmdError::InvalidValue);
    }

    let mut data = image.data().to_vec();
    let mut rng_state = seed;

    match noise_type {
        "gaussian" => {
            // Gaussian noise
            for val in &mut data {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u1 = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u2 = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;

                let u1_safe = (u1 * 0.999 + 0.0005).clamp(0.001, 0.999);
                let z = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * PI * u2).cos();
                *val += noise_level * z;
            }
        }
        "poisson" => {
            // Approximated Poisson noise (Gaussian approximation for high counts)
            for val in &mut data {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let v = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;

                let u_safe = (u * 0.999 + 0.0005).clamp(0.001, 0.999);
                let z = (-2.0 * u_safe.ln()).sqrt() * (2.0 * PI * v).cos();
                *val += noise_level * (*val).sqrt() * z;
            }
        }
        "speckle" => {
            // Multiplicative speckle noise
            for val in &mut data {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u1 = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let u2 = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;

                let u1_safe = (u1 * 0.999 + 0.0005).clamp(0.001, 0.999);
                let z = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * PI * u2).cos();
                *val *= 1.0 + noise_level * z;
            }
        }
        _ => return Err(EmdError::InvalidConfig("Unknown noise type".to_string())),
    }

    Image2D::new(image.width(), image.height(), data, image.pixel_spacing())
}

/// Add common medical imaging artifacts.
///
/// Simulates artifacts like metal artifacts, beam hardening, etc.
///
/// # Arguments
///
/// * `image` - Input medical image
/// * `artifact_type` - Type of artifact: "metal" or "beam_hardening"
///
/// # Returns
///
/// A new Image2D with added artifacts.
pub fn add_medical_artifact(image: &Image2D, artifact_type: &str) -> Result<Image2D, EmdError> {
    let mut data = image.data().to_vec();
    let width = image.width();
    let height = image.height();

    match artifact_type {
        "metal" => {
            // Metal artifact: bright streaks from center
            let center_col = width / 2;
            let center_row = height / 2;

            for row in 0..height {
                for col in 0..width {
                    let distance_to_center = (((col as i32 - center_col as i32).abs() as f64)
                        .powi(2)
                        + ((row as i32 - center_row as i32).abs() as f64).powi(2))
                    .sqrt();

                    if distance_to_center < width as f64 / 6.0 {
                        let artifact_strength =
                            (0.3 * (1.0 - distance_to_center / (width as f64 / 6.0))).max(0.0);
                        let idx = row * width + col;
                        data[idx] = (data[idx] + artifact_strength).min(1.0);
                    }
                }
            }
        }
        "beam_hardening" => {
            // Beam hardening: intensity decreases from edges toward center
            let center_col = width as f64 / 2.0;
            let center_row = height as f64 / 2.0;
            let max_r = ((width / 2) as f64).max((height / 2) as f64);

            for row in 0..height {
                for col in 0..width {
                    let x = col as f64 - center_col;
                    let y = row as f64 - center_row;
                    let r = (x * x + y * y).sqrt();

                    let hardening_factor = 0.2 * (r / max_r).min(1.0);
                    let idx = row * width + col;
                    data[idx] *= 1.0 - hardening_factor;
                }
            }
        }
        _ => return Err(EmdError::InvalidConfig("Unknown artifact type".to_string())),
    }

    Image2D::new(width, height, data, image.pixel_spacing())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_ct_generation() {
        let img = synthetic_ct_slice(64, 64).unwrap();
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
        // All values should be finite
        for &val in img.data() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_synthetic_mri_generation() {
        let img = synthetic_mri_slice(64, 64).unwrap();
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_synthetic_ultrasound_generation() {
        let img = synthetic_ultrasound(64, 64).unwrap();
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_synthetic_xray_generation() {
        let img = synthetic_xray(64, 64).unwrap();
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_add_gaussian_noise() {
        let base = synthetic_ct_slice(32, 32).unwrap();
        let noisy = add_medical_noise(&base, "gaussian", 0.05, 42).unwrap();
        assert_eq!(noisy.width(), 32);
        assert_eq!(noisy.height(), 32);
    }

    #[test]
    fn test_add_artifact() {
        let base = synthetic_ct_slice(32, 32).unwrap();
        let with_artifact = add_medical_artifact(&base, "metal").unwrap();
        assert_eq!(with_artifact.width(), 32);
        assert_eq!(with_artifact.height(), 32);
    }
}
