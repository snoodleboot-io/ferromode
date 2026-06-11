# WASM Binding Guide

## Pure-Wrap Contract

The WASM bindings follow a **pure-wrap contract**: the WASM layer performs
**only marshalling** between JavaScript and Rust. No algorithm logic exists
in the WASM layer.

This means:
- All decomposition logic lives in `crates/ferromode/`
- WASM functions only: receive JS types → copy to Rust → call Rust → wrap result → return JS
- No algorithm parameters are computed in WASM
- No signal processing happens in WASM

## Architecture

```
JavaScript                    Rust (WASM)              Core Library
┌──────────────┐             ┌───────────────┐        ┌──────────────┐
│ Float64Array │ ──copy──→   │ &[f64]        │ ───→   │ emd()        │
│              │             │               │        │ eemd()       │
│ WasmEmdConfig│ ──convert─→ │ EmdConfig     │ ───→   │ ceemdan()    │
│              │             │               │        │ vmd()        │
│              │ ←──wrap───  │ Decomposition │ ←───   │ memd()       │
│              │   Float64   │ Result        │        │ namemd()     │
└──────────────┘             └───────────────┘        └──────────────┘
```

## Module Structure

| File | Purpose |
|------|---------|
| `lib.rs` | Module entry, re-exports all public types |
| `types.rs` | Enum wrappers: `WasmBoundaryCondition`, `WasmStoppingCriterion`, `WasmAlgorithmType` |
| `config.rs` | Config structs: `WasmEmdConfig`, `WasmEnsembleConfig`, `WasmVmdConfig` |
| `functions.rs` | All 8 algorithm functions + result types |

## Data Flow

### 1. Array Marshalling

JavaScript `Float64Array` is copied into Rust `Vec<f64>`:

```rust
let len = signal.length() as usize;
let mut data = vec![0.0f64; len];
signal.copy_to(&mut data);
```

For MEMD/NA-MEMD, an array of `Float64Array` is converted to `Vec<Vec<f64>>`.

### 2. Validation

Non-finite values (NaN, Inf) are rejected at the WASM boundary:

```rust
fn validate_signal(data: &[f64]) -> Result<(), JsValue> {
    for &val in data {
        if !val.is_finite() {
            return Err(JsValue::from_str("Signal contains non-finite value"));
        }
    }
    Ok(())
}
```

### 3. Error Mapping

Rust `EmdError` is converted to JavaScript `Error`:

```rust
fn error_to_js(err: EmdError) -> JsValue {
    js_sys::Error::new(&err.to_string()).into()
}
```

### 4. Result Accessors

Results are wrapped in `#[wasm_bindgen]` structs that return `Float64Array`
views into WASM linear memory:

```rust
pub fn get_imf(&self, n: usize) -> Result<Float64Array, JsValue> {
    self.inner.imfs.get(n)
        .map(|imf| Float64Array::from(imf.as_slice()))
        .ok_or_else(|| JsValue::from_str("IMF index out of bounds"))
}
```

## Available Functions

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `emd_wasm()` | `Float64Array`, `WasmEmdConfig` | `WasmDecompositionResult` | Empirical Mode Decomposition |
| `eemd_wasm()` | `Float64Array`, `WasmEnsembleConfig`, `WasmEmdConfig` | `WasmDecompositionResult` | Ensemble EMD |
| `ceemd_wasm()` | `Float64Array`, `WasmEnsembleConfig`, `WasmEmdConfig` | `WasmDecompositionResult` | Complementary EEMD |
| `ceemdan_wasm()` | `Float64Array`, `WasmEnsembleConfig`, `WasmEmdConfig` | `WasmDecompositionResult` | Complete EEMD with Adaptive Noise |
| `iceemdan_wasm()` | `Float64Array`, `WasmEnsembleConfig`, `WasmEmdConfig` | `WasmDecompositionResult` | Improved CEEMDAN |
| `vmd_wasm()` | `Float64Array`, `WasmVmdConfig` | `WasmDecompositionResult` | Variational Mode Decomposition |
| `memd_wasm()` | `Array<Float64Array>`, directions, max_imfs, seed | `WasmDecompositionResult` | Multivariate EMD |
| `namemd_wasm()` | `Array<Float64Array>`, directions, max_imfs, noise params | `WasmDecompositionResult` | Noise-Assisted MEMD |
