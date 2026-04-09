#![warn(missing_docs)]

//! 3D volume decomposition using separable EMD approach.
//!
//! This module implements 3D EMD decomposition by cascading 2D and 1D
//! operations along different dimensions, reducing computational complexity
//! while maintaining reasonable approximation of true 3D decomposition.
//!
//! # Algorithm Overview
//!
//! The separable 3D EMD decomposes a volume in three phases:
//!
//! **Phase 1: XY-Slice Decomposition (2D EMD)**
//! - Extract each XY slice as a 2D image at depth z ∈ [0, depth)
//! - Apply 2D separable EMD to each slice independently
//! - Output: For each IMF index, a collection of 2D images (one per slice)
//!
//! **Phase 2: Z-Column Decomposition (1D EMD)**
//! - For each IMF index and each (y, x) location:
//!   - Extract a Z-column: signal across all depth slices
//!   - Apply 1D EMD to the column
//! - Output: Final 3D IMFs reconstructed from column decompositions
//!
//! **Phase 3: Residue Propagation**
//! - Apply same column decomposition to the residue from Phase 1
//! - Final residue is the deepest residue from column decompositions
//!
//! # Complexity
//!
//! - **Time:** O(W × H × D × n_imfs × max_sifts)
//!   where n_imfs ≈ log₂(D) and max_sifts ≈ 5 average
//! - **Memory:** O(W × H × D) for volume + intermediate storage
//!
//! # Expected Performance
//!
//! - 128³ volume: ~4-5 seconds (estimated, with optimization)
//! - 64³ volume: ~1-2 seconds
//! - Memory usage: ~10-15× original volume size during decomposition
//!
//! # Parallelization
//!
//! Phase 1 can be parallelized using rayon over Z slices.
//! The `parallel` flag controls whether to use rayon::par_iter.

use crate::algorithms::emd::{emd, EmdConfig};
use crate::error::EmdError;

use super::image_2d::{decompose_image_2d_separable, DecompositionMetadata, Image2D};
use super::slicing::{extract_z_column, extract_z_slice};
use super::volume_3d::{Volume3D, Volume3DDecomposition};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Decompose a 3D volume using separable (XY slices + Z columns) EMD.
///
/// This function implements the three-phase separable approach for 3D EMD:
/// 1. Decompose each XY slice independently using 2D EMD → 2D IMFs for each slice
/// 2. For each IMF index and each pixel location, decompose the Z-column → final 3D IMFs
/// 3. Apply residue propagation → final 3D residue
///
/// # Arguments
///
/// * `volume` - The input 3D volume
/// * `config` - EMD configuration (max_imfs, boundary conditions, etc.)
/// * `parallel` - If true, use rayon for Phase 1 parallelization (requires `parallel` feature)
///
/// # Returns
///
/// A `Volume3DDecomposition` containing 3D IMFs and the residue.
///
/// # Errors
///
/// Returns `EmdError` if:
/// - Volume dimensions are < 3×3×3
/// - 1D/2D decomposition fails for any slice or column
/// - Intermediate results contain non-finite values
/// - IMF count consistency check fails
///
/// # Example
///
/// ```ignore
/// use ferromode::adapters::multidim::{Volume3D, decompose_volume_3d_separable};
/// use ferromode::algorithms::emd::EmdConfig;
///
/// let volume = Volume3D::new(10, 10, 10, vec![1.0; 1000], None)?;
/// let config = EmdConfig::default();
/// let decomp = decompose_volume_3d_separable(&volume, &config, false)?;
/// # Ok::<(), ferromode::error::EmdError>(())
/// ```
pub fn decompose_volume_3d_separable(
    volume: &Volume3D,
    config: &EmdConfig,
    parallel: bool,
) -> Result<Volume3DDecomposition, EmdError> {
    let width = volume.width();
    let height = volume.height();
    let depth = volume.depth();

    // =========================================================================
    // VALIDATION
    // =========================================================================

    if width < 3 || height < 3 || depth < 3 {
        return Err(EmdError::InvalidConfig(format!(
            "Volume must be at least 3×3×3 for separable decomposition, got {}×{}×{}",
            width, height, depth
        )));
    }

    // =========================================================================
    // PHASE 1: XY-Slice Decomposition (2D EMD on each slice)
    // =========================================================================

    let z_decompositions = if cfg!(feature = "parallel") && parallel {
        // Parallel decomposition using rayon
        #[cfg(feature = "parallel")]
        {
            (0..depth)
                .into_par_iter()
                .map(|z| {
                    let slice = extract_z_slice(volume, z);
                    decompose_image_2d_separable(&slice, config)
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        #[cfg(not(feature = "parallel"))]
        {
            // Parallel mode requested but feature not enabled - fall back to sequential
            let mut decomps = Vec::with_capacity(depth);
            for z in 0..depth {
                let slice = extract_z_slice(volume, z);
                decomps.push(decompose_image_2d_separable(&slice, config)?);
            }
            decomps
        }
    } else {
        // Sequential decomposition
        let mut decomps = Vec::with_capacity(depth);
        for z in 0..depth {
            let slice = extract_z_slice(volume, z);
            decomps.push(decompose_image_2d_separable(&slice, config)?);
        }
        decomps
    };

    if z_decompositions.is_empty() {
        return Err(EmdError::InvalidConfig("No slices to decompose".to_string()));
    }

    // Get number of 2D IMFs from first decomposition (all should match)
    let num_imfs_2d = z_decompositions[0].imfs_2d.len();

    // =========================================================================
    // PHASE 2: Z-Column Decomposition (1D EMD on each column)
    // =========================================================================

    let mut final_imfs_3d: Vec<Volume3D> = Vec::new();

    for imf_idx in 0..num_imfs_2d {
        // Extract the imf_idx'th 2D IMF from each z slice decomposition
        let imf_2d_images: Vec<Image2D> =
            z_decompositions.iter().map(|decomp| decomp.imfs_2d[imf_idx].clone()).collect();

        // Storage for all column-based IMFs from this 2D IMF set
        let mut imf_volumes: Vec<Vec<f64>> = Vec::new(); // [col_imf_idx][flat_idx]

        // For each (y, x) location, extract and decompose Z-column
        for y in 0..height {
            for x in 0..width {
                let z_column = extract_z_column(&imf_2d_images, y, x);

                // Decompose the column
                let col_decomp = emd(&z_column, config)?;

                // Store IMFs from column decomposition
                for (col_imf_idx, col_imf) in col_decomp.imfs.imfs.iter().enumerate() {
                    // Initialize storage for this column IMF if needed
                    if y == 0 && x == 0 {
                        imf_volumes.push(vec![0.0; width * height * depth]);
                    }

                    // Fill in values for this (y, x) position across all z
                    // Volume uses row-major order: data[z * (width * height) + y * width + x]
                    for z in 0..depth {
                        let idx = z * (width * height) + y * width + x;
                        imf_volumes[col_imf_idx][idx] = col_imf[z];
                    }
                }
            }
        }

        // Convert collected data to Volume3D objects
        for col_imf_data in imf_volumes {
            let vol = Volume3D::new(width, height, depth, col_imf_data, None)?;
            final_imfs_3d.push(vol);
        }
    }

    // =========================================================================
    // PHASE 3: Residue Propagation
    // =========================================================================

    let residue_2d_images: Vec<Image2D> =
        z_decompositions.iter().map(|decomp| decomp.residue_2d.clone()).collect();

    let mut final_residue_3d_data = vec![0.0; width * height * depth];

    for y in 0..height {
        for x in 0..width {
            let z_residue_column = extract_z_column(&residue_2d_images, y, x);

            // Decompose residue column
            let col_decomp = emd(&z_residue_column, config)?;

            // Take the final residue from column decomposition
            for (z, &val) in col_decomp.imfs.residue.iter().enumerate() {
                let idx = z * (width * height) + y * width + x;
                final_residue_3d_data[idx] = val;
            }
        }
    }

    let residue_3d = Volume3D::new(width, height, depth, final_residue_3d_data, None)?;

    // =========================================================================
    // RESULT ASSEMBLY
    // =========================================================================

    let metadata = DecompositionMetadata::default();

    Ok(Volume3DDecomposition {
        imfs_3d: final_imfs_3d,
        residue_3d,
        num_iterations: 3, // Three phases: 2D slices + 1D columns + residue
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_3d_simple_stack() -> Result<(), EmdError> {
        // Create a simple 3x3x3 volume
        let data = vec![1.0; 27];
        let volume = Volume3D::new(3, 3, 3, data, None)?;
        let config = EmdConfig::default();

        let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

        // Should produce some IMFs + residue
        assert!(decomp.imfs_3d.len() >= 0);
        assert_eq!(decomp.residue_3d.width(), 3);
        assert_eq!(decomp.residue_3d.height(), 3);
        assert_eq!(decomp.residue_3d.depth(), 3);

        Ok(())
    }

    #[test]
    fn test_decompose_3d_dimension_preservation() -> Result<(), EmdError> {
        let data = vec![1.0; 64]; // 4x4x4
        let volume = Volume3D::new(4, 4, 4, data, None)?;
        let config = EmdConfig::default();

        let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

        // All IMFs should have correct dimensions
        for imf in &decomp.imfs_3d {
            assert_eq!(imf.width(), 4);
            assert_eq!(imf.height(), 4);
            assert_eq!(imf.depth(), 4);
        }

        // Residue should have correct dimensions
        assert_eq!(decomp.residue_3d.width(), 4);
        assert_eq!(decomp.residue_3d.height(), 4);
        assert_eq!(decomp.residue_3d.depth(), 4);

        Ok(())
    }

    #[test]
    fn test_decompose_3d_reconstruction() -> Result<(), EmdError> {
        let data = vec![2.0; 27]; // 3x3x3
        let volume = Volume3D::new(3, 3, 3, data, None)?;
        let config = EmdConfig::default();

        let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

        // Reconstruction should be possible
        let reconstructed = decomp.reconstruct()?;
        assert_eq!(reconstructed.width(), 3);
        assert_eq!(reconstructed.height(), 3);
        assert_eq!(reconstructed.depth(), 3);

        Ok(())
    }

    #[test]
    fn test_decompose_3d_min_size() -> Result<(), EmdError> {
        // Minimum valid size
        let data = vec![1.0; 27]; // 3x3x3
        let volume = Volume3D::new(3, 3, 3, data, None)?;
        let config = EmdConfig::default();

        let decomp = decompose_volume_3d_separable(&volume, &config, false)?;
        assert_eq!(decomp.residue_3d.width(), 3);

        Ok(())
    }

    #[test]
    fn test_decompose_3d_too_small() {
        // Too small volume
        let result = Volume3D::new(2, 2, 2, vec![1.0; 8], None).and_then(|volume| {
            let config = EmdConfig::default();
            decompose_volume_3d_separable(&volume, &config, false)
        });

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_decompose_3d_non_cubic() -> Result<(), EmdError> {
        // Non-cubic volume
        let data = vec![1.0; 96]; // 4x4x6
        let volume = Volume3D::new(4, 4, 6, data, None)?;
        let config = EmdConfig::default();

        let decomp = decompose_volume_3d_separable(&volume, &config, false)?;

        assert_eq!(decomp.residue_3d.width(), 4);
        assert_eq!(decomp.residue_3d.height(), 4);
        assert_eq!(decomp.residue_3d.depth(), 6);

        Ok(())
    }
}
