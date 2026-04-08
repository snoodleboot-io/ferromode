//! Synthetic 2D signal generators for testing.
//!
//! This module provides functions to generate various synthetic 2D signals
//! for testing EMD decomposition, including geometric patterns, mathematical
//! functions, and noise-corrupted variants.

use ferromode::adapters::multidim::Image2D;
use ferromode::error::EmdError;
use std::f64::consts::PI;

/// Generate a checkerboard pattern with alternating values.
///
/// # Arguments
///
/// * `size` - Size of the checkerboard (size × size)
/// * `block_size` - Size of each checkerboard block
///
/// # Returns
///
/// An Image2D with alternating 1.0 and 0.0 values in a checkerboard pattern.
pub fn checkerboard_2d(size: usize, block_size: usize) -> Result<Image2D, EmdError> {
    if size == 0 || block_size == 0 {
        return Err(EmdError::InvalidConfig("Size and block_size must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(size * size);
    for row in 0..size {
        for col in 0..size {
            let row_block = row / block_size;
            let col_block = col / block_size;
            let val = if (row_block + col_block) % 2 == 0 { 1.0 } else { 0.0 };
            data.push(val);
        }
    }

    Image2D::new(size, size, data, None)
}

/// Generate a linear gradient from top-left to bottom-right.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `start` - Value at top-left
/// * `end` - Value at bottom-right
///
/// # Returns
///
/// An Image2D with smooth linear gradient variation.
pub fn linear_gradient(
    width: usize,
    height: usize,
    start: f64,
    end: f64,
) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height);
    let diagonal_max = ((width - 1).pow(2) + (height - 1).pow(2)) as f64;
    let diagonal_max = diagonal_max.sqrt();

    for row in 0..height {
        for col in 0..width {
            let distance = ((col as f64).powi(2) + (row as f64).powi(2)).sqrt();
            let t = if diagonal_max > 0.0 { distance / diagonal_max } else { 0.0 };
            let val = start + t * (end - start);
            data.push(val);
        }
    }

    Image2D::new(width, height, data, None)
}

/// Generate a 2D Gaussian blob centered in the image.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `sigma` - Standard deviation of Gaussian
/// * `amplitude` - Peak amplitude
///
/// # Returns
///
/// An Image2D with a Gaussian peak in the center.
pub fn gaussian_2d(
    width: usize,
    height: usize,
    sigma: f64,
    amplitude: f64,
) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }
    if sigma <= 0.0 || !sigma.is_finite() {
        return Err(EmdError::InvalidValue);
    }

    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;
    let sigma_sq = sigma * sigma;

    let mut data = Vec::with_capacity(width * height);
    for row in 0..height {
        for col in 0..width {
            let dx = col as f64 - center_x;
            let dy = row as f64 - center_y;
            let r_sq = dx * dx + dy * dy;
            let val = amplitude * (-r_sq / (2.0 * sigma_sq)).exp();
            data.push(val);
        }
    }

    Image2D::new(width, height, data, None)
}

/// Generate a multi-frequency 2D signal (sum of 2D sinusoids).
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `freqs` - List of (freq_x, freq_y, amplitude) tuples
///
/// # Returns
///
/// An Image2D with composite of 2D sinusoids.
pub fn multi_frequency_2d(
    width: usize,
    height: usize,
    freqs: &[(f64, f64, f64)],
) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }

    let mut data = Vec::with_capacity(width * height);
    for row in 0..height {
        for col in 0..width {
            let x = col as f64 / width as f64;
            let y = row as f64 / height as f64;
            let mut val = 0.0;
            for &(freq_x, freq_y, amp) in freqs {
                val += amp * (2.0 * PI * freq_x * x).sin() * (2.0 * PI * freq_y * y).sin();
            }
            data.push(val);
        }
    }

    Image2D::new(width, height, data, None)
}

/// Add Gaussian noise to an image.
///
/// # Arguments
///
/// * `image` - Input image
/// * `sigma` - Standard deviation of noise
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A new Image2D with added Gaussian noise.
pub fn add_gaussian_noise(image: &Image2D, sigma: f64, seed: u64) -> Result<Image2D, EmdError> {
    if sigma < 0.0 || !sigma.is_finite() {
        return Err(EmdError::InvalidValue);
    }

    // Simple deterministic pseudo-random generator (using seed for reproducibility)
    let mut rng_state = seed;
    let mut data = image.data().to_vec();

    for val in &mut data {
        // Linear congruential generator
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let u1 = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;

        // Box-Muller transform to convert uniform to normal
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = ((rng_state >> 16) & 0x7fff) as f64 / 32768.0;

        // Avoid ln(0) or ln(1)
        let u1_safe = (u1 * 0.999 + 0.0005).clamp(0.001, 0.999);
        let z = (-2.0 * u1_safe.ln()).sqrt() * (2.0 * PI * u2).cos();
        *val += sigma * z;
    }

    // Validate all data is finite before creating image
    for &v in &data {
        if !v.is_finite() {
            return Err(EmdError::InvalidValue);
        }
    }

    Image2D::new(image.width(), image.height(), data, image.pixel_spacing())
}

/// Generate a constant (flat) signal.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `value` - Constant value
///
/// # Returns
///
/// An Image2D with all pixels set to the same value.
pub fn constant_signal(width: usize, height: usize, value: f64) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }
    if !value.is_finite() {
        return Err(EmdError::InvalidValue);
    }

    let data = vec![value; width * height];
    Image2D::new(width, height, data, None)
}

/// Generate a single impulse (spike) at a given location.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `spike_row` - Row position of spike
/// * `spike_col` - Column position of spike
/// * `amplitude` - Spike amplitude
///
/// # Returns
///
/// An Image2D with a single spike at the specified location.
pub fn single_spike(
    width: usize,
    height: usize,
    spike_row: usize,
    spike_col: usize,
    amplitude: f64,
) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }
    if spike_row >= height || spike_col >= width {
        return Err(EmdError::InvalidConfig("Spike position out of bounds".to_string()));
    }
    if !amplitude.is_finite() {
        return Err(EmdError::InvalidValue);
    }

    let mut data = vec![0.0; width * height];
    data[spike_row * width + spike_col] = amplitude;

    Image2D::new(width, height, data, None)
}

/// Generate a ring/annular pattern.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `center_x` - Center X coordinate
/// * `center_y` - Center Y coordinate
/// * `inner_radius` - Inner radius
/// * `outer_radius` - Outer radius
/// * `amplitude` - Amplitude of ring
///
/// # Returns
///
/// An Image2D with a circular ring pattern.
pub fn ring_pattern(
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
    inner_radius: f64,
    outer_radius: f64,
    amplitude: f64,
) -> Result<Image2D, EmdError> {
    if width == 0 || height == 0 {
        return Err(EmdError::InvalidConfig("Width and height must be > 0".to_string()));
    }
    if inner_radius < 0.0 || outer_radius <= inner_radius || !amplitude.is_finite() {
        return Err(EmdError::InvalidValue);
    }

    let mut data = Vec::with_capacity(width * height);
    for row in 0..height {
        for col in 0..width {
            let dx = col as f64 - center_x;
            let dy = row as f64 - center_y;
            let r = (dx * dx + dy * dy).sqrt();

            let val = if r >= inner_radius && r <= outer_radius { amplitude } else { 0.0 };
            data.push(val);
        }
    }

    Image2D::new(width, height, data, None)
}

/// Generate a noisy synthetic signal (base signal + noise).
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `base_freqs` - List of (freq_x, freq_y, amplitude) for base signal
/// * `noise_sigma` - Standard deviation of noise
/// * `seed` - Random seed
///
/// # Returns
///
/// An Image2D with composite signal plus noise.
pub fn noisy_multi_frequency(
    width: usize,
    height: usize,
    base_freqs: &[(f64, f64, f64)],
    noise_sigma: f64,
    seed: u64,
) -> Result<Image2D, EmdError> {
    let base = multi_frequency_2d(width, height, base_freqs)?;
    add_gaussian_noise(&base, noise_sigma, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkerboard_generation() {
        let img = checkerboard_2d(4, 2).unwrap();
        assert_eq!(img.width(), 4);
        assert_eq!(img.height(), 4);
    }

    #[test]
    fn test_gaussian_2d_generation() {
        let img = gaussian_2d(32, 32, 5.0, 1.0).unwrap();
        assert_eq!(img.width(), 32);
        assert_eq!(img.height(), 32);
        // Center should have highest value
        let center = img.get(16, 16);
        let corner = img.get(0, 0);
        assert!(center > corner);
    }

    #[test]
    fn test_linear_gradient_generation() {
        let img = linear_gradient(10, 10, 0.0, 1.0).unwrap();
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 10);
    }

    #[test]
    fn test_constant_signal_generation() {
        let img = constant_signal(5, 5, 3.5).unwrap();
        assert_eq!(img.width(), 5);
        assert_eq!(img.height(), 5);
        for row in 0..5 {
            for col in 0..5 {
                assert!((img.get(row, col) - 3.5).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_single_spike_generation() {
        let img = single_spike(5, 5, 2, 2, 10.0).unwrap();
        assert_eq!(img.get(2, 2), 10.0);
        assert_eq!(img.get(0, 0), 0.0);
        assert_eq!(img.get(4, 4), 0.0);
    }
}
