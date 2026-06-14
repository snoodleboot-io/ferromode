#![warn(missing_docs)]

//! Utilities for extracting and reconstructing slices from 3D volumes.
//!
//! This module provides helper functions to convert between 3D volumes and
//! their 1D/2D projections (slices and columns) needed for the separable
//! 3D EMD decomposition algorithm.

use super::image_2d::Image2D;
use super::volume_3d::Volume3D;
use crate::error::EmdError;

/// Extract a single XY slice (2D image) at a given Z index from a 3D volume.
///
/// # Arguments
///
/// * `volume` - The 3D volume to slice
/// * `z` - The Z index to extract (must be < volume.depth())
///
/// # Returns
///
/// An `Image2D` containing the XY plane at the given Z index.
///
/// # Panics
///
/// Panics if `z >= volume.depth()`.
pub fn extract_z_slice(volume: &Volume3D, z: usize) -> Image2D {
    assert!(z < volume.depth(), "z {} >= depth {}", z, volume.depth());

    let width = volume.width();
    let height = volume.height();
    let _depth = volume.depth();

    // Extract the XY plane at index z
    // In row-major order: data[z * (width * height) + y * width + x]
    let mut slice_data = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let idx = z * (width * height) + y * width + x;
            slice_data.push(volume.data()[idx]);
        }
    }

    // Reconstruct Image2D - unwrap is safe because we control the data
    Image2D::new(width, height, slice_data, None).expect("slice construction failed")
}

/// Extract a single Z-column (1D signal) across all depth slices at a given (y, x) location.
///
/// # Arguments
///
/// * `images` - A collection of 2D images (typically from Phase 1 decomposition)
/// * `y` - The Y coordinate (row in each image, must be < image.height())
/// * `x` - The X coordinate (column in each image, must be < image.width())
///
/// # Returns
///
/// A 1D vector containing values at (x, y) across all images, indexed from z=0 to z=len(images)-1.
///
/// # Panics
///
/// Panics if any image has different dimensions or if coordinates are out of bounds.
pub fn extract_z_column(images: &[Image2D], y: usize, x: usize) -> Vec<f64> {
    if images.is_empty() {
        return Vec::new();
    }

    let width = images[0].width();
    let height = images[0].height();

    assert!(y < height, "y {} >= height {}", y, height);
    assert!(x < width, "x {} >= width {}", x, width);

    // Verify all images have consistent dimensions
    for img in images {
        assert_eq!(
            img.width(),
            width,
            "Image width mismatch: expected {}, got {}",
            width,
            img.width()
        );
        assert_eq!(
            img.height(),
            height,
            "Image height mismatch: expected {}, got {}",
            height,
            img.height()
        );
    }

    // Extract column values across depth
    let mut column = Vec::with_capacity(images.len());
    for img in images {
        column.push(img.get(y, x));
    }

    column
}

/// Reconstruct a 3D volume from a collection of 2D XY slices.
///
/// # Arguments
///
/// * `layers` - A collection of 2D images to stack as Z slices
/// * `depth` - Expected depth (must equal `layers.len()`)
///
/// # Returns
///
/// A reconstructed 3D volume.
///
/// # Errors
///
/// Returns `EmdError::DimensionMismatch` if:
/// - Number of layers doesn't match the specified depth
/// - All layers don't have the same dimensions
///
/// Returns `EmdError::InvalidValue` if any reconstructed data contains non-finite values.
pub fn construct_volume_from_layers(
    layers: &[Image2D],
    depth: usize,
) -> Result<Volume3D, EmdError> {
    if layers.len() != depth {
        return Err(EmdError::DimensionMismatch);
    }

    if layers.is_empty() {
        return Err(EmdError::InvalidConfig("layers collection is empty".to_string()));
    }

    let width = layers[0].width();
    let height = layers[0].height();

    // Verify all layers have consistent dimensions
    for (_z, layer) in layers.iter().enumerate() {
        if layer.width() != width || layer.height() != height {
            return Err(EmdError::DimensionMismatch);
        }
    }

    // Reconstruct volume by stacking layers in row-major order
    let mut volume_data = Vec::with_capacity(width * height * depth);

    for layer in layers {
        volume_data.extend_from_slice(layer.data());
    }

    // Construct volume - voxel_spacing is None for reconstructed volumes
    Volume3D::new(width, height, depth, volume_data, None)
}

/// Normalize IMF counts across multiple decompositions.
///
/// When decomposing slices or columns independently, the number of extracted
/// IMFs may vary slightly due to different stopping conditions. This function
/// ensures all decompositions produce the same number of IMFs by padding with
/// zeros if needed.
///
/// # Arguments
///
/// * `imfs` - A collection of IMF vectors, each with potentially different lengths
///
/// # Returns
///
/// A normalized collection where all IMF vectors have the same length
/// (the maximum from the input).
pub fn normalize_imf_counts(imfs: &[Vec<Image2D>]) -> Vec<Vec<Image2D>> {
    if imfs.is_empty() {
        return Vec::new();
    }

    // Find the maximum number of IMFs
    let max_imf_count = imfs.iter().map(|v| v.len()).max().unwrap_or(0);

    // If all already have the same count, return early
    if imfs.iter().all(|v| v.len() == max_imf_count) {
        return imfs.to_vec();
    }

    // Otherwise, pad shorter collections with zero-filled images
    let mut normalized = Vec::with_capacity(imfs.len());

    for imf_collection in imfs {
        let mut normalized_collection = imf_collection.clone();

        if normalized_collection.len() < max_imf_count {
            // Get dimensions from first IMF (all have same size)
            let (width, height) = if let Some(first) = normalized_collection.first() {
                (first.width(), first.height())
            } else {
                continue;
            };

            // Pad with zero-filled images
            let num_to_add = max_imf_count - normalized_collection.len();
            for _ in 0..num_to_add {
                let zero_img = Image2D::new(width, height, vec![0.0; width * height], None)
                    .expect("zero padding failed");
                normalized_collection.push(zero_img);
            }
        }

        normalized.push(normalized_collection);
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_z_slice_correctness() -> Result<(), EmdError> {
        // Create a simple 2x2x3 volume with distinguishable values per z
        let mut data = Vec::new();
        for z in 0..3 {
            for y in 0..2 {
                for x in 0..2 {
                    data.push((z * 4 + y * 2 + x) as f64);
                }
            }
        }

        let volume = Volume3D::new(2, 2, 3, data, None)?;

        // Extract each z slice and verify
        for z in 0..3 {
            let slice = extract_z_slice(&volume, z);
            assert_eq!(slice.width(), 2);
            assert_eq!(slice.height(), 2);

            // Verify values match expected pattern
            for y in 0..2 {
                for x in 0..2 {
                    let expected = (z * 4 + y * 2 + x) as f64;
                    assert_eq!(slice.get(y, x), expected);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_extract_z_column_consistency() -> Result<(), EmdError> {
        // Create three 2x2 images
        let img1 = Image2D::new(2, 2, vec![1.0, 2.0, 3.0, 4.0], None)?;
        let img2 = Image2D::new(2, 2, vec![5.0, 6.0, 7.0, 8.0], None)?;
        let img3 = Image2D::new(2, 2, vec![9.0, 10.0, 11.0, 12.0], None)?;

        let images = vec![img1, img2, img3];

        // Extract column at (y=0, x=0)
        let col = extract_z_column(&images, 0, 0);
        assert_eq!(col.len(), 3);
        assert_eq!(col[0], 1.0); // From img1
        assert_eq!(col[1], 5.0); // From img2
        assert_eq!(col[2], 9.0); // From img3

        // Extract column at (y=1, x=1)
        let col = extract_z_column(&images, 1, 1);
        assert_eq!(col.len(), 3);
        assert_eq!(col[0], 4.0); // From img1
        assert_eq!(col[1], 8.0); // From img2
        assert_eq!(col[2], 12.0); // From img3

        Ok(())
    }

    #[test]
    fn test_roundtrip_slicing() -> Result<(), EmdError> {
        // Create a 3x3x3 volume with simple sequential values
        let mut data = Vec::new();
        for i in 0..27 {
            data.push(i as f64);
        }

        let original_volume = Volume3D::new(3, 3, 3, data.clone(), None)?;

        // Extract all slices
        let mut slices = Vec::new();
        for z in 0..3 {
            slices.push(extract_z_slice(&original_volume, z));
        }

        // Reconstruct volume
        let reconstructed = construct_volume_from_layers(&slices, 3)?;

        // Verify dimensions
        assert_eq!(reconstructed.width(), 3);
        assert_eq!(reconstructed.height(), 3);
        assert_eq!(reconstructed.depth(), 3);

        // Verify all data matches
        assert_eq!(reconstructed.data(), original_volume.data());

        Ok(())
    }
}
