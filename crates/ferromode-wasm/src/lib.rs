//! WASM bindings for Ferromode via wasm-bindgen.
//!
//! This module provides a pure-wrap contract: all WASM functions only marshal
//! data between JavaScript and Rust, with zero algorithm logic in the WASM layer.

mod config;
mod functions;
mod types;

use wasm_bindgen::prelude::*;

pub use config::{WasmEmdConfig, WasmEnsembleConfig, WasmVmdConfig};
pub use functions::{
    ceemd_wasm, ceemdan_wasm, eemd_wasm, emd_wasm, iceemdan_wasm, memd_wasm, namemd_wasm, vmd_wasm,
    WasmDecompositionResult, WasmHilbertResult, WasmImfCollection,
};
pub use types::{WasmAlgorithmType, WasmBoundaryCondition, WasmStoppingCriterion};

/// Return the version of the Ferromode WASM module.
#[wasm_bindgen]
pub fn ferromode_wasm_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let version = ferromode_wasm_version();
        assert!(!version.is_empty());
    }
}
