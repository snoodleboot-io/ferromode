#![warn(missing_docs)]

//! Multidimensional EMD adapters for 2D and 3D data.
//!
//! This module provides efficient separable decomposition algorithms for
//! 2D images and 3D volumes by leveraging the core 1D EMD implementation.
//!
//! # Architecture
//!
//! The separable approach decomposes multi-dimensional data by cascading
//! 1D operations along each dimension:
//!
//! ```text
//! 2D Image (512×512)
//! ├── Phase 1: EMD on each row (512 1D ops)
//! └── Phase 2: EMD on each column of results (512 1D ops)
//! → 2D IMFs + Residue
//!
//! 3D Volume (128³)
//! ├── Phase 1: EMD on each XY plane (128 2D ops)
//! ├── Phase 2: EMD along Z for each pixel (128² 1D ops)
//! └── Phase 3: Residue propagation
//! → 3D IMFs + Residue
//! ```
//!
//! # Performance
//!
//! - **2D (512×512):** <5 seconds (estimated ~2-3 IMFs)
//! - **3D (128³):** <30 seconds (estimated ~2-3 IMFs, with optimization)
//!
//! # Module Organization
//!
//! - [`image_2d`] — 2D image types and separable decomposition
//! - [`volume_3d`] — 3D volume types (decomposition deferred)
//! - [`padding`] — Boundary padding utilities
//! - [`extrema_2d`] — 2D extrema detection (stub for T-310)

pub mod extrema_2d;
pub mod image_2d;
pub mod padding;
pub mod volume_3d;

// Re-export public API
pub use extrema_2d::{find_local_extrema_2d, Extrema2D};
pub use image_2d::{
    decompose_image_2d_separable, DecompositionMetadata, Image2D, Image2DDecomposition,
};
pub use padding::{calculate_optimal_padding_size, pad_periodic_1d, pad_symmetric_1d, unpad_1d};
pub use volume_3d::{Volume3D, Volume3DDecomposition};
