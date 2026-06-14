#![warn(missing_docs)]

//! 3D extrema detection for volume data.
//!
//! This module provides utilities for finding local extrema (maxima and minima)
//! in 3D volumes. The implementation uses 26-neighborhood comparison (including
//! diagonal neighbors across all dimensions).
//!
//! # Algorithm
//!
//! A voxel at (x, y, z) is a local maximum if its value is strictly greater than
//! all 26 neighbors (3×3×3 - 1 = 26 comparisons).
//!
//! A voxel at (x, y, z) is a local minimum if its value is strictly less than
//! all 26 neighbors.
//!
//! Boundary voxels (at edges of the volume) are not considered as extrema
//! to avoid edge artifacts.
//!
//! # Note
//!
//! The separable 3D EMD decomposition (T-315) does not require explicit extrema
//! detection. This function is provided for future optimization passes and
//! analysis tools. Full 3D extrema-based decomposition is deferred to v2.4.
//!
//! # Complexity
//!
//! - **Time:** O(W × H × D × 26) = O(W × H × D)
//! - **Memory:** O(k) where k is the number of extrema found

use super::volume_3d::Volume3D;

/// Result type containing indices of maxima and minima.
#[derive(Debug, Clone, PartialEq)]
pub struct Extrema3D {
    /// Indices of local maxima as (x, y, z) tuples
    pub maxima: Vec<(usize, usize, usize)>,
    /// Indices of local minima as (x, y, z) tuples
    pub minima: Vec<(usize, usize, usize)>,
}

impl Extrema3D {
    /// Get the number of detected maxima.
    #[inline]
    pub fn n_maxima(&self) -> usize {
        self.maxima.len()
    }

    /// Get the number of detected minima.
    #[inline]
    pub fn n_minima(&self) -> usize {
        self.minima.len()
    }

    /// Get total number of extrema (maxima + minima).
    #[inline]
    pub fn n_total(&self) -> usize {
        self.maxima.len() + self.minima.len()
    }
}

/// Find all local extrema (maxima and minima) in a 3D volume.
///
/// A voxel is a local maximum if its value is strictly greater than all
/// 26 neighbors in the 3D neighborhood. A voxel is a local minimum if its
/// value is strictly less than all 26 neighbors.
///
/// Boundary voxels (x=0, x=width-1, y=0, y=height-1, z=0, z=depth-1) are
/// excluded from extrema detection to avoid spurious edge artifacts.
///
/// # Arguments
///
/// * `volume` - The 3D volume to analyze
///
/// # Returns
///
/// An `Extrema3D` struct containing indices of all detected maxima and minima.
///
/// # Algorithm
///
/// For each interior voxel (x, y, z) where 1 ≤ x < width-1, 1 ≤ y < height-1, 1 ≤ z < depth-1:
/// - Compare voxel value with all 26 neighbors
/// - If greater than all → maxima
/// - If less than all → minima
/// - Otherwise → non-extremum
///
/// # Example
///
/// ```ignore
/// use ferromode::adapters::multidim::{Volume3D, find_local_extrema_3d};
///
/// let volume = Volume3D::new(5, 5, 5, vec![1.0; 125], None)?;
/// let extrema = find_local_extrema_3d(&volume);
/// println!("Found {} maxima and {} minima", extrema.n_maxima(), extrema.n_minima());
/// # Ok::<(), ferromode::error::EmdError>(())
/// ```
pub fn find_local_extrema_3d(volume: &Volume3D) -> Extrema3D {
    let width = volume.width();
    let height = volume.height();
    let depth = volume.depth();

    let mut maxima = Vec::new();
    let mut minima = Vec::new();

    // Process only interior voxels (exclude boundaries)
    for z in 1..depth - 1 {
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center_val = volume.get(x, y, z);

                // Check all 26 neighbors
                let mut is_maximum = true;
                let mut is_minimum = true;

                'neighbor_loop: for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            // Skip the center voxel itself
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }

                            let nx = (x as i32 + dx) as usize;
                            let ny = (y as i32 + dy) as usize;
                            let nz = (z as i32 + dz) as usize;

                            let neighbor_val = volume.get(nx, ny, nz);

                            // Check if center is still potentially a maximum
                            if center_val <= neighbor_val {
                                is_maximum = false;
                            }

                            // Check if center is still potentially a minimum
                            if center_val >= neighbor_val {
                                is_minimum = false;
                            }

                            // Early exit if neither
                            if !is_maximum && !is_minimum {
                                break 'neighbor_loop;
                            }
                        }
                    }
                }

                // Record extrema
                if is_maximum {
                    maxima.push((x, y, z));
                } else if is_minimum {
                    minima.push((x, y, z));
                }
            }
        }
    }

    Extrema3D { maxima, minima }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EmdError;

    #[test]
    fn test_find_local_extrema_3d_simple() -> Result<(), EmdError> {
        // Create a 5x5x5 volume with a clear maximum at center and a clear minimum
        // at interior voxel (1, 1, 1).
        let mut data = vec![1.0; 125];

        // Set center voxel (2, 2, 2) to 100.0 — strict maximum
        let center_idx = 2 * (5 * 5) + 2 * 5 + 2;
        data[center_idx] = 100.0;

        // Set interior voxel (1, 1, 1) to 0.0 — strict minimum (all 26 neighbors = 1.0)
        let min_idx = 1 * (5 * 5) + 1 * 5 + 1;
        data[min_idx] = 0.0;

        let volume = Volume3D::new(5, 5, 5, data, None)?;
        let extrema = find_local_extrema_3d(&volume);

        // Should find the maximum at (2, 2, 2)
        assert_eq!(extrema.n_maxima(), 1);
        assert_eq!(extrema.maxima[0], (2, 2, 2));

        // Should find the minimum at (1, 1, 1)
        assert!(extrema.n_minima() > 0);
        assert!(extrema.minima.contains(&(1, 1, 1)));

        Ok(())
    }

    #[test]
    fn test_find_local_extrema_3d_boundaries() -> Result<(), EmdError> {
        // Create a 3x3x3 volume - boundary voxels should never be extrema
        let data = vec![1.0; 27];
        let volume = Volume3D::new(3, 3, 3, data, None)?;
        let extrema = find_local_extrema_3d(&volume);

        // With uniform data, no extrema should be found
        // (all voxels equal, so none are strictly greater/less)
        assert_eq!(extrema.n_maxima(), 0);
        assert_eq!(extrema.n_minima(), 0);

        Ok(())
    }
}
