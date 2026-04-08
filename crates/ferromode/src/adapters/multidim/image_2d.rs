#![warn(missing_docs)]

//! 2D image data structures and separable EMD decomposition.
//!
//! This module implements 2D image handling and the separable EMD algorithm,
//! which decomposes images by applying 1D EMD row-wise, then column-wise on
//! the resulting IMFs.

use crate::algorithms::emd::{emd, EmdConfig};
use crate::error::EmdError;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// DecompositionMetadata
// ---------------------------------------------------------------------------

/// Metadata about a 2D/3D decomposition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecompositionMetadata {
    /// Elapsed time for decomposition
    pub elapsed: Option<Duration>,
    /// Description of the decomposition
    pub description: Option<String>,
}

impl Default for DecompositionMetadata {
    fn default() -> Self {
        Self { elapsed: None, description: None }
    }
}

/// A 2D image stored in row-major order.
///
/// The underlying data is a flat `Vec<f64>` in row-major format, meaning
/// elements are indexed as: `data[row * width + col]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Image2D {
    /// Flattened image data in row-major order
    data: Vec<f64>,
    /// Image width in pixels
    width: usize,
    /// Image height in pixels
    height: usize,
    /// Optional pixel spacing (dy, dx) for physical correspondence
    pixel_spacing: Option<(f64, f64)>,
}

impl Image2D {
    /// Create a new Image2D from width, height, and data.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels (must be > 0)
    /// * `height` - Image height in pixels (must be > 0)
    /// * `data` - Flattened row-major data (must have length = width × height)
    /// * `pixel_spacing` - Optional (dy, dx) physical spacing
    ///
    /// # Errors
    ///
    /// Returns `EmdError::InvalidDimensions` if dimensions don't match data length.
    /// Returns `EmdError::InvalidValue` if any data value is non-finite.
    pub fn new(
        width: usize,
        height: usize,
        data: Vec<f64>,
        pixel_spacing: Option<(f64, f64)>,
    ) -> Result<Self, EmdError> {
        if width == 0 || height == 0 {
            return Err(EmdError::InvalidConfig("width and height must be > 0".to_string()));
        }

        if data.len() != width * height {
            return Err(EmdError::InvalidConfig(format!(
                "data length {} doesn't match dimensions {}x{}",
                data.len(),
                width,
                height
            )));
        }

        // Validate all data is finite
        for &val in &data {
            if !val.is_finite() {
                return Err(EmdError::InvalidValue);
            }
        }

        // Validate pixel spacing if provided
        if let Some((dy, dx)) = pixel_spacing {
            if !dy.is_finite() || !dx.is_finite() || dy <= 0.0 || dx <= 0.0 {
                return Err(EmdError::InvalidValue);
            }
        }

        Ok(Self { data, width, height, pixel_spacing })
    }

    /// Create an Image2D from a 2D array (Vec<Vec<f64>>).
    ///
    /// Converts a row-based representation to the internal row-major flat storage.
    /// All rows must have the same length.
    ///
    /// # Errors
    ///
    /// Returns `EmdError::InvalidDimensions` if rows have varying lengths.
    /// Returns `EmdError::InvalidValue` if any value is non-finite.
    pub fn from_2d_array(
        rows: &[Vec<f64>],
        pixel_spacing: Option<(f64, f64)>,
    ) -> Result<Self, EmdError> {
        if rows.is_empty() {
            return Err(EmdError::EmptySignal);
        }

        let height = rows.len();
        let width = rows[0].len();

        if width == 0 {
            return Err(EmdError::EmptySignal);
        }

        // Verify all rows have same length
        for row in rows {
            if row.len() != width {
                return Err(EmdError::DimensionMismatch);
            }
        }

        // Flatten to row-major
        let mut data = Vec::with_capacity(width * height);
        for row in rows {
            data.extend_from_slice(row);
        }

        Self::new(width, height, data, pixel_spacing)
    }

    /// Get image width.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get image height.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the underlying data slice.
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get a pixel value at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if row >= height or col >= width.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.height, "row {} >= height {}", row, self.height);
        assert!(col < self.width, "col {} >= width {}", col, self.width);
        self.data[row * self.width + col]
    }

    /// Set a pixel value at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if row >= height or col >= width.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.height, "row {} >= height {}", row, self.height);
        assert!(col < self.width, "col {} >= width {}", col, self.width);
        self.data[row * self.width + col] = value;
    }

    /// Get pixel spacing if available.
    #[inline]
    pub fn pixel_spacing(&self) -> Option<(f64, f64)> {
        self.pixel_spacing
    }

    /// Extract a single row as a 1D signal.
    ///
    /// # Panics
    ///
    /// Panics if row >= height.
    pub fn row(&self, row: usize) -> Vec<f64> {
        assert!(row < self.height, "row {} >= height {}", row, self.height);
        let start = row * self.width;
        let end = start + self.width;
        self.data[start..end].to_vec()
    }

    /// Extract a single column as a 1D signal.
    ///
    /// # Panics
    ///
    /// Panics if col >= width.
    pub fn column(&self, col: usize) -> Vec<f64> {
        assert!(col < self.width, "col {} >= width {}", col, self.width);
        let mut result = Vec::with_capacity(self.height);
        for row in 0..self.height {
            result.push(self.data[row * self.width + col]);
        }
        result
    }

    /// Convert to a 2D array (Vec<Vec<f64>>) representation.
    pub fn to_rows(&self) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(self.height);
        for row in 0..self.height {
            result.push(self.row(row));
        }
        result
    }

    /// Convert columns to a 2D array.
    pub fn to_columns(&self) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(self.width);
        for col in 0..self.width {
            result.push(self.column(col));
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Image2DDecomposition
// ---------------------------------------------------------------------------

/// Result of 2D EMD decomposition.
///
/// Contains all 2D IMFs and the residue from a separable decomposition,
/// plus metadata about the decomposition process.
#[derive(Debug, Clone, PartialEq)]
pub struct Image2DDecomposition {
    /// Extracted 2D IMFs
    pub imfs_2d: Vec<Image2D>,
    /// Residual image (lowest-frequency component)
    pub residue_2d: Image2D,
    /// Number of decomposition iterations
    pub num_iterations: usize,
    /// Metadata (timing, algorithm info, etc.)
    pub metadata: DecompositionMetadata,
}

impl Image2DDecomposition {
    /// Get the number of 2D IMFs.
    #[inline]
    pub fn n_imfs(&self) -> usize {
        self.imfs_2d.len()
    }

    /// Reconstruct the original image from IMFs and residue.
    ///
    /// Returns a new Image2D with the same dimensions as the original.
    pub fn reconstruct(&self) -> Result<Image2D, EmdError> {
        let width = self.residue_2d.width();
        let height = self.residue_2d.height();

        // Start with residue
        let mut reconstructed = self.residue_2d.data().to_vec();

        // Add all IMFs
        for imf in &self.imfs_2d {
            if imf.width() != width || imf.height() != height {
                return Err(EmdError::DimensionMismatch);
            }
            for i in 0..reconstructed.len() {
                reconstructed[i] += imf.data()[i];
            }
        }

        Image2D::new(width, height, reconstructed, self.residue_2d.pixel_spacing())
    }
}

// ---------------------------------------------------------------------------
// Separable Decomposition
// ---------------------------------------------------------------------------

/// Decompose a 2D image using separable (row-wise then column-wise) EMD.
///
/// This function implements the efficient separable approach for 2D EMD:
/// 1. Decompose each row independently using 1D EMD → N row-based IMFs
/// 2. For each row-based IMF, decompose each column independently → 2D IMFs
/// 3. Reconstruct 2D structure from results
///
/// # Arguments
///
/// * `image` - The input 2D image
/// * `config` - EMD configuration (max_imfs, boundary conditions, etc.)
///
/// # Returns
///
/// An `Image2DDecomposition` containing 2D IMFs and residue.
///
/// # Errors
///
/// Returns `EmdError` if:
/// - 1D decomposition fails for any row or column
/// - Image is too small (< 3×3 recommended)
/// - Configuration is invalid
///
/// # Algorithm Details
///
/// The separable approach reduces complexity from O(M×N) to O(M+N) passes
/// while maintaining good approximation of true 2D decomposition for
/// natural images and medical data.
///
/// Expected performance: 512×512 image in <5 seconds.
///
/// # Example
///
/// ```ignore
/// use ferromode::adapters::multidim::{Image2D, decompose_image_2d_separable};
/// use ferromode::algorithms::emd::EmdConfig;
///
/// let image = Image2D::new(10, 10, vec![1.0; 100], None)?;
/// let config = EmdConfig::default();
/// let decomp = decompose_image_2d_separable(&image, &config)?;
/// # Ok::<(), ferromode::error::EmdError>(())
/// ```
pub fn decompose_image_2d_separable(
    image: &Image2D,
    config: &EmdConfig,
) -> Result<Image2DDecomposition, EmdError> {
    let width = image.width();
    let height = image.height();
    let pixel_spacing = image.pixel_spacing();

    // Phase 1: Decompose rows
    let mut row_imfs: Vec<Vec<Vec<f64>>> = Vec::new(); // [imf_idx][row_idx]
    let mut row_residue: Vec<Vec<f64>> = Vec::new(); // [row_idx]

    for row_idx in 0..height {
        let row_data = image.row(row_idx);
        let decomp_result = emd(&row_data, config)?;

        // Initialize storage for first row
        if row_idx == 0 {
            for _ in &decomp_result.imfs.imfs {
                row_imfs.push(Vec::with_capacity(height));
            }
            row_residue.reserve(height);
        }

        // Verify consistent number of IMFs
        if decomp_result.imfs.imfs.len() != row_imfs.len() {
            return Err(EmdError::InvalidValue);
        }

        // Store decomposition
        for (imf_idx, imf) in decomp_result.imfs.imfs.iter().enumerate() {
            row_imfs[imf_idx].push(imf.clone());
        }
        row_residue.push(decomp_result.imfs.residue.clone());
    }

    // Phase 2: Decompose columns of each row-based IMF
    let mut image_2d_imfs: Vec<Image2D> = Vec::new();

    for row_based_imfs in row_imfs {
        // row_based_imfs[row] is the decomposition result for each row
        // Now decompose the columns of this row-based IMF
        let row_based_image = Image2D::from_2d_array(&row_based_imfs, pixel_spacing)?;

        for col_idx in 0..width {
            let col_data = row_based_image.column(col_idx);
            let col_decomp = emd(&col_data, config)?;

            // Initialize image IMFs storage on first column
            if col_idx == 0 {
                for _ in &col_decomp.imfs.imfs {
                    image_2d_imfs.push(Image2D::new(width, height, Vec::new(), pixel_spacing)?);
                }
            }

            // Store each column's decomposition
            for (imf_idx, imf) in col_decomp.imfs.imfs.iter().enumerate() {
                let mut imf_data = image_2d_imfs[imf_idx].data().to_vec();
                // Expand to full size if needed
                if imf_data.is_empty() {
                    imf_data = vec![0.0; width * height];
                }
                // Store column
                for (row, &val) in imf.iter().enumerate() {
                    imf_data[row * width + col_idx] = val;
                }
                image_2d_imfs[imf_idx] = Image2D::new(width, height, imf_data, pixel_spacing)?;
            }
        }
    }

    // Phase 3: Decompose columns of residue
    let residue_image = Image2D::from_2d_array(&row_residue, pixel_spacing)?;
    let mut residue_2d_data = vec![0.0; width * height];

    for col_idx in 0..width {
        let col_data = residue_image.column(col_idx);
        let col_decomp = emd(&col_data, config)?;

        // For residue, we only keep the final residue (no column IMFs)
        for (row, &val) in col_decomp.imfs.residue.iter().enumerate() {
            residue_2d_data[row * width + col_idx] = val;
        }
    }

    let residue_2d = Image2D::new(width, height, residue_2d_data, pixel_spacing)?;

    let metadata = DecompositionMetadata::default();

    Ok(Image2DDecomposition {
        imfs_2d: image_2d_imfs,
        residue_2d,
        num_iterations: 2, // Row decomposition + column decomposition
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_2d_new() -> Result<(), EmdError> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let img = Image2D::new(2, 2, data, None)?;
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        Ok(())
    }

    #[test]
    fn test_image_2d_get_set() -> Result<(), EmdError> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut img = Image2D::new(2, 2, data, None)?;

        assert_eq!(img.get(0, 0), 1.0);
        assert_eq!(img.get(1, 1), 4.0);

        img.set(0, 1, 99.0);
        assert_eq!(img.get(0, 1), 99.0);
        Ok(())
    }

    #[test]
    fn test_image_2d_row_column() -> Result<(), EmdError> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let img = Image2D::new(3, 2, data, None)?;

        let row0 = img.row(0);
        assert_eq!(row0, vec![1.0, 2.0, 3.0]);

        let col1 = img.column(1);
        assert_eq!(col1, vec![2.0, 5.0]);
        Ok(())
    }

    #[test]
    fn test_image_2d_from_2d_array() -> Result<(), EmdError> {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let img = Image2D::from_2d_array(&rows, None)?;

        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        assert_eq!(img.get(0, 0), 1.0);
        assert_eq!(img.get(1, 1), 4.0);
        Ok(())
    }

    #[test]
    fn test_image_2d_to_rows() -> Result<(), EmdError> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let img = Image2D::new(2, 2, data, None)?;
        let rows = img.to_rows();

        assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        Ok(())
    }

    #[test]
    fn test_image_2d_pixel_spacing() -> Result<(), EmdError> {
        let data = vec![1.0; 4];
        let spacing = Some((1.5, 1.5));
        let img = Image2D::new(2, 2, data, spacing)?;

        assert_eq!(img.pixel_spacing(), spacing);
        Ok(())
    }

    #[test]
    fn test_image_2d_dimension_mismatch() {
        let result = Image2D::new(2, 2, vec![1.0, 2.0, 3.0], None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_decomposition_reconstruct() -> Result<(), EmdError> {
        // Create a simple image
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let img = Image2D::new(2, 2, data, None)?;

        // Create a simple decomposition
        let imf1 = Image2D::new(2, 2, vec![0.5, 1.0, 1.5, 2.0], None)?;
        let residue = Image2D::new(2, 2, vec![0.5, 1.0, 1.5, 2.0], None)?;

        let decomp = Image2DDecomposition {
            imfs_2d: vec![imf1],
            residue_2d: residue,
            num_iterations: 1,
            metadata: DecompositionMetadata::default(),
        };

        let reconstructed = decomp.reconstruct()?;
        assert_eq!(reconstructed.width(), 2);
        assert_eq!(reconstructed.height(), 2);
        Ok(())
    }
}
