//! WASM bindings for Ferromode via wasm-bindgen.

use wasm_bindgen::prelude::*;

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
