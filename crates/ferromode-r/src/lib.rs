//! Minimal R bindings for Ferromode EMD
//!
//! This module exposes the core EMD (Empirical Mode Decomposition) functionality
//! to R via extendr.

use extendr_api::prelude::*;
use ferromode::algorithms::emd::{emd, EmdConfig};
use ferromode::types::Signal;

/// Decompose a signal using Empirical Mode Decomposition (EMD)
///
/// @param signal A numeric vector representing the input signal
/// @param max_imfs Maximum number of IMFs to extract (default: 0 = automatic)
///
/// @return A list containing:
///   - `imfs`: List of Intrinsic Mode Functions (each as a numeric vector)
///   - `residue`: The residual component (numeric vector)
///
/// @examples
/// \dontrun{
/// signal <- sin(seq(0, 2*pi, length.out=100)) + rnorm(100, 0, 0.1)
/// result <- emd_decompose(signal, max_imfs = 5)
/// }
#[extendr]
fn emd_decompose(signal: Vec<f64>, max_imfs: i32) -> Result<List> {
    // Convert max_imfs to usize, treating negative as 0 (automatic)
    let max_imfs_usize = if max_imfs < 0 { 0 } else { max_imfs as usize };

    // Create minimal config
    let config = EmdConfig { max_imfs: max_imfs_usize, ..Default::default() };

    // Convert input to Signal type
    let sig = Signal::from_slice(&signal).map_err(|e| format!("Invalid signal: {}", e))?;

    // Run EMD decomposition
    let result =
        emd(sig.values(), &config).map_err(|e| format!("EMD decomposition failed: {}", e))?;

    // Convert IMFs to R list format (convert each Vec<f64> to Robj)
    let imfs_list = List::from_values(result.imfs.imfs.iter().map(|imf| r!(imf)));

    // Build and return result as R named list
    let result_list = list!(imfs = imfs_list, residue = r!(&result.imfs.residue));

    Ok(result_list)
}

// R module declaration
extendr_module! {
    mod ferromode_r;
    fn emd_decompose;
}
