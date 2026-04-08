#![warn(missing_docs)]

//! 3D volumetric data structures and decomposition.
//!
//! This module provides basic 3D data handling structures. Full decomposition
//! algorithms are deferred to later tasks (T-311 onwards).

use super::image_2d::DecompositionMetadata;
use crate::error::EmdError;

/// A 3D volume stored in row-major order (Z, Y, X convention).
///
/// The underlying data is a flat `Vec<f64>` in row-major format with indexing:
/// `data[z * (width * height) + y * width + x]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Volume3D {
    /// Flattened volume data in row-major order (Z-Y-X)
    data: Vec<f64>,
    /// Volume width in voxels (X dimension)
    width: usize,
    /// Volume height in voxels (Y dimension)
    height: usize,
    /// Volume depth in voxels (Z dimension)
    depth: usize,
    /// Optional voxel spacing (dz, dy, dx) for physical correspondence
    voxel_spacing: Option<(f64, f64, f64)>,
}

impl Volume3D {
    /// Create a new Volume3D from dimensions and data.
    ///
    /// # Arguments
    ///
    /// * `width` - Volume width (X dimension, must be > 0)
    /// * `height` - Volume height (Y dimension, must be > 0)
    /// * `depth` - Volume depth (Z dimension, must be > 0)
    /// * `data` - Flattened row-major data (must have length = width × height × depth)
    /// * `voxel_spacing` - Optional (dz, dy, dx) physical spacing
    ///
    /// # Errors
    ///
    /// Returns `EmdError::InvalidDimensions` if dimensions don't match data length.
    /// Returns `EmdError::InvalidValue` if any data value is non-finite.
    pub fn new(
        width: usize,
        height: usize,
        depth: usize,
        data: Vec<f64>,
        voxel_spacing: Option<(f64, f64, f64)>,
    ) -> Result<Self, EmdError> {
        if width == 0 || height == 0 || depth == 0 {
            return Err(EmdError::InvalidConfig(
                "width, height, and depth must be > 0".to_string(),
            ));
        }

        let expected_len = width * height * depth;
        if data.len() != expected_len {
            return Err(EmdError::InvalidConfig(format!(
                "data length {} doesn't match dimensions {}x{}x{}",
                data.len(),
                width,
                height,
                depth
            )));
        }

        // Validate all data is finite
        for &val in &data {
            if !val.is_finite() {
                return Err(EmdError::InvalidValue);
            }
        }

        // Validate voxel spacing if provided
        if let Some((dz, dy, dx)) = voxel_spacing {
            if !dz.is_finite()
                || !dy.is_finite()
                || !dx.is_finite()
                || dz <= 0.0
                || dy <= 0.0
                || dx <= 0.0
            {
                return Err(EmdError::InvalidValue);
            }
        }

        Ok(Self { data, width, height, depth, voxel_spacing })
    }

    /// Get volume width (X dimension).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get volume height (Y dimension).
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get volume depth (Z dimension).
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Get the underlying data slice.
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get a voxel value at (x, y, z).
    ///
    /// # Panics
    ///
    /// Panics if coordinates are out of bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f64 {
        assert!(x < self.width, "x {} >= width {}", x, self.width);
        assert!(y < self.height, "y {} >= height {}", y, self.height);
        assert!(z < self.depth, "z {} >= depth {}", z, self.depth);
        self.data[z * (self.width * self.height) + y * self.width + x]
    }

    /// Set a voxel value at (x, y, z).
    ///
    /// # Panics
    ///
    /// Panics if coordinates are out of bounds.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: f64) {
        assert!(x < self.width, "x {} >= width {}", x, self.width);
        assert!(y < self.height, "y {} >= height {}", y, self.height);
        assert!(z < self.depth, "z {} >= depth {}", z, self.depth);
        let idx = z * (self.width * self.height) + y * self.width + x;
        self.data[idx] = value;
    }

    /// Get voxel spacing if available.
    #[inline]
    pub fn voxel_spacing(&self) -> Option<(f64, f64, f64)> {
        self.voxel_spacing
    }
}

/// Result of 3D EMD decomposition.
///
/// Contains 3D IMFs and residue from separable slicing decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct Volume3DDecomposition {
    /// Extracted 3D IMFs
    pub imfs_3d: Vec<Volume3D>,
    /// Residual volume
    pub residue_3d: Volume3D,
    /// Number of decomposition iterations
    pub num_iterations: usize,
    /// Metadata about the decomposition
    pub metadata: DecompositionMetadata,
}

impl Volume3DDecomposition {
    /// Get the number of 3D IMFs.
    #[inline]
    pub fn n_imfs(&self) -> usize {
        self.imfs_3d.len()
    }

    /// Reconstruct the original volume from IMFs and residue.
    pub fn reconstruct(&self) -> Result<Volume3D, EmdError> {
        let width = self.residue_3d.width();
        let height = self.residue_3d.height();
        let depth = self.residue_3d.depth();

        // Start with residue
        let mut reconstructed = self.residue_3d.data().to_vec();

        // Add all IMFs
        for imf in &self.imfs_3d {
            if imf.width() != width || imf.height() != height || imf.depth() != depth {
                return Err(EmdError::DimensionMismatch);
            }
            for i in 0..reconstructed.len() {
                reconstructed[i] += imf.data()[i];
            }
        }

        Volume3D::new(width, height, depth, reconstructed, self.residue_3d.voxel_spacing())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_3d_new() -> Result<(), EmdError> {
        let data = vec![1.0; 8];
        let vol = Volume3D::new(2, 2, 2, data, None)?;
        assert_eq!(vol.width(), 2);
        assert_eq!(vol.height(), 2);
        assert_eq!(vol.depth(), 2);
        Ok(())
    }

    #[test]
    fn test_volume_3d_get_set() -> Result<(), EmdError> {
        let data = vec![0.0; 8];
        let mut vol = Volume3D::new(2, 2, 2, data, None)?;

        vol.set(0, 0, 0, 5.0);
        assert_eq!(vol.get(0, 0, 0), 5.0);

        vol.set(1, 1, 1, 99.0);
        assert_eq!(vol.get(1, 1, 1), 99.0);
        Ok(())
    }

    #[test]
    fn test_volume_3d_voxel_spacing() -> Result<(), EmdError> {
        let data = vec![1.0; 8];
        let spacing = Some((1.0, 1.5, 2.0));
        let vol = Volume3D::new(2, 2, 2, data, spacing)?;

        assert_eq!(vol.voxel_spacing(), spacing);
        Ok(())
    }

    #[test]
    fn test_volume_3d_dimension_mismatch() {
        let result = Volume3D::new(2, 2, 2, vec![1.0, 2.0, 3.0], None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_decomposition_reconstruct() -> Result<(), EmdError> {
        let imf = Volume3D::new(2, 2, 2, vec![0.5; 8], None)?;
        let residue = Volume3D::new(2, 2, 2, vec![0.5; 8], None)?;

        let decomp = Volume3DDecomposition {
            imfs_3d: vec![imf],
            residue_3d: residue,
            num_iterations: 1,
            metadata: DecompositionMetadata::default(),
        };

        let reconstructed = decomp.reconstruct()?;
        assert_eq!(reconstructed.width(), 2);
        assert_eq!(reconstructed.height(), 2);
        assert_eq!(reconstructed.depth(), 2);
        Ok(())
    }
}
