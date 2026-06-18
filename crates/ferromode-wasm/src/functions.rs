use ferromode::algorithms::ceemd::ceemd;
use ferromode::algorithms::ceemdan::ceemdan;
use ferromode::algorithms::eemd::eemd;
use ferromode::algorithms::emd::emd;
use ferromode::algorithms::hilbert::hilbert_imf;
use ferromode::algorithms::iceemdan::iceemdan;
use ferromode::algorithms::vmd::vmd;
use ferromode::error::EmdError;
use ferromode::multivariate::memd::memd;
use ferromode::multivariate::namemd::namemd;
use ferromode::types::{DecompositionResult, HilbertResult, ImfCollection};
use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

use crate::config::{
    WasmEmdConfig, WasmEnsembleConfig, WasmMemdConfig, WasmNaMemdConfig, WasmVmdConfig,
};

fn error_to_js(err: EmdError) -> JsValue {
    js_sys::Error::new(&err.to_string()).into()
}

fn validate_signal(data: &[f64]) -> Result<(), JsValue> {
    if data.is_empty() {
        return Err(JsValue::from_str("Signal must not be empty"));
    }
    for &val in data {
        if !val.is_finite() {
            return Err(JsValue::from_str("Signal contains non-finite value (NaN or Inf)"));
        }
    }
    Ok(())
}

#[wasm_bindgen]
pub struct WasmImfCollection {
    inner: ImfCollection,
}

#[wasm_bindgen]
impl WasmImfCollection {
    pub fn get_imf(&self, n: usize) -> Result<Float64Array, JsValue> {
        self.inner
            .imfs
            .get(n)
            .map(|imf| Float64Array::from(imf.as_slice()))
            .ok_or_else(|| JsValue::from_str(&format!("IMF index {} out of bounds", n)))
    }

    pub fn get_residue(&self) -> Float64Array {
        Float64Array::from(self.inner.residue.as_slice())
    }

    pub fn n_imfs(&self) -> usize {
        self.inner.n_imfs()
    }

    pub fn reconstruct(&self) -> Float64Array {
        Float64Array::from(self.inner.reconstruct().as_slice())
    }
}

#[wasm_bindgen]
pub struct WasmHilbertResult {
    inner: HilbertResult,
}

#[wasm_bindgen]
impl WasmHilbertResult {
    pub fn get_instantaneous_amplitude(&self, imf_idx: usize) -> Result<Float64Array, JsValue> {
        self.inner
            .instantaneous_amplitude
            .get(imf_idx)
            .map(|amp| Float64Array::from(amp.as_slice()))
            .ok_or_else(|| JsValue::from_str(&format!("IMF index {} out of bounds", imf_idx)))
    }

    pub fn get_instantaneous_frequency(&self, imf_idx: usize) -> Result<Float64Array, JsValue> {
        self.inner
            .instantaneous_frequency
            .get(imf_idx)
            .map(|freq| Float64Array::from(freq.as_slice()))
            .ok_or_else(|| JsValue::from_str(&format!("IMF index {} out of bounds", imf_idx)))
    }

    pub fn get_marginal_spectrum(&self) -> Float64Array {
        Float64Array::from(self.inner.marginal_spectrum.as_slice())
    }
}

#[wasm_bindgen]
pub struct WasmDecompositionResult {
    inner: DecompositionResult,
}

#[wasm_bindgen]
impl WasmDecompositionResult {
    pub fn imfs(&self) -> WasmImfCollection {
        WasmImfCollection {
            inner: ImfCollection {
                imfs: self.inner.imfs.imfs.clone(),
                residue: self.inner.imfs.residue.clone(),
            },
        }
    }

    pub fn hilbert(&self, sample_rate: f64) -> Result<WasmHilbertResult, JsValue> {
        let result = hilbert_imf(&self.inner.imfs.imfs, sample_rate).map_err(error_to_js)?;
        Ok(WasmHilbertResult { inner: result })
    }

    pub fn algorithm(&self) -> String {
        format!("{:?}", self.inner.algorithm)
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.inner.elapsed.as_secs_f64() * 1000.0
    }

    pub fn n_siftings(&self) -> usize {
        self.inner.n_siftings
    }

    pub fn config_snapshot(&self) -> String {
        self.inner.config_snapshot.clone()
    }
}

#[wasm_bindgen]
pub fn emd_wasm(
    signal: &Float64Array,
    config: &WasmEmdConfig,
) -> Result<WasmDecompositionResult, JsValue> {
    let len = signal.length() as usize;
    let mut data = vec![0.0f64; len];
    signal.copy_to(&mut data);
    validate_signal(&data)?;

    let rust_config = config.to_rust();
    let result = emd(&data, &rust_config).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}

#[wasm_bindgen]
pub fn eemd_wasm(
    signal: &Float64Array,
    ensemble_config: &WasmEnsembleConfig,
    emd_config: &WasmEmdConfig,
) -> Result<WasmDecompositionResult, JsValue> {
    let len = signal.length() as usize;
    let mut data = vec![0.0f64; len];
    signal.copy_to(&mut data);
    validate_signal(&data)?;

    let result =
        eemd(&data, &ensemble_config.to_rust(), &emd_config.to_rust()).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}

#[wasm_bindgen]
pub fn ceemd_wasm(
    signal: &Float64Array,
    ensemble_config: &WasmEnsembleConfig,
    emd_config: &WasmEmdConfig,
) -> Result<WasmDecompositionResult, JsValue> {
    let len = signal.length() as usize;
    let mut data = vec![0.0f64; len];
    signal.copy_to(&mut data);
    validate_signal(&data)?;

    let result =
        ceemd(&data, &ensemble_config.to_rust(), &emd_config.to_rust()).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}

#[wasm_bindgen]
pub fn ceemdan_wasm(
    signal: &Float64Array,
    ensemble_config: &WasmEnsembleConfig,
    emd_config: &WasmEmdConfig,
) -> Result<WasmDecompositionResult, JsValue> {
    let len = signal.length() as usize;
    let mut data = vec![0.0f64; len];
    signal.copy_to(&mut data);
    validate_signal(&data)?;

    let result =
        ceemdan(&data, &ensemble_config.to_rust(), &emd_config.to_rust()).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}

#[wasm_bindgen]
pub fn iceemdan_wasm(
    signal: &Float64Array,
    ensemble_config: &WasmEnsembleConfig,
    emd_config: &WasmEmdConfig,
) -> Result<WasmDecompositionResult, JsValue> {
    let len = signal.length() as usize;
    let mut data = vec![0.0f64; len];
    signal.copy_to(&mut data);
    validate_signal(&data)?;

    let result =
        iceemdan(&data, &ensemble_config.to_rust(), &emd_config.to_rust()).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}

#[wasm_bindgen]
pub fn vmd_wasm(
    signal: &Float64Array,
    config: &WasmVmdConfig,
) -> Result<WasmDecompositionResult, JsValue> {
    let len = signal.length() as usize;
    let mut data = vec![0.0f64; len];
    signal.copy_to(&mut data);
    validate_signal(&data)?;

    let result = vmd(&data, &config.to_rust()).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}

fn channels_from_js(channels: &js_sys::Array) -> Result<Vec<Vec<f64>>, JsValue> {
    if channels.length() == 0 {
        return Err(JsValue::from_str("Must provide at least one channel"));
    }
    let n_channels = channels.length() as usize;
    let mut signal: Vec<Vec<f64>> = Vec::with_capacity(n_channels);
    for i in 0..n_channels {
        let float_arr = js_sys::Float64Array::from(channels.get(i as u32));
        let len = float_arr.length() as usize;
        let mut data = vec![0.0f64; len];
        float_arr.copy_to(&mut data);
        validate_signal(&data)?;
        signal.push(data);
    }
    Ok(signal)
}

#[wasm_bindgen]
pub fn memd_wasm(
    channels: &js_sys::Array,
    config: &WasmMemdConfig,
) -> Result<WasmDecompositionResult, JsValue> {
    let signal = channels_from_js(channels)?;
    let result = memd(&signal, &config.to_rust()).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}

#[wasm_bindgen]
pub fn namemd_wasm(
    channels: &js_sys::Array,
    config: &WasmNaMemdConfig,
) -> Result<WasmDecompositionResult, JsValue> {
    let signal = channels_from_js(channels)?;
    let result = namemd(&signal, &config.to_rust()).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}

// ---------------------------------------------------------------------------
// Streaming decomposition
// ---------------------------------------------------------------------------

/// Streaming decomposer handle for chunk-wise decomposition.
#[wasm_bindgen]
pub struct WasmStreamingDecomposer {
    inner: ferromode::adapters::streaming::StreamingDecomposer,
}

#[wasm_bindgen]
impl WasmStreamingDecomposer {
    /// Create a streaming decomposer. `ar_order` 0 defaults to 3, `buffer_size`
    /// 0 to 4096.
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: &WasmEmdConfig,
        buffer_size: usize,
        ar_order: usize,
    ) -> Result<WasmStreamingDecomposer, JsValue> {
        let order = if ar_order == 0 { 3 } else { ar_order };
        let predictor = ferromode::adapters::streaming::ArModel::new(order)
            .map_err(error_to_js)?;
        let buf = if buffer_size == 0 { 4096 } else { buffer_size };
        let inner = ferromode::adapters::streaming::StreamingDecomposer::new(
            config.to_rust(),
            Box::new(predictor),
            buf,
        )
        .map_err(error_to_js)?;
        Ok(WasmStreamingDecomposer { inner })
    }

    /// Decompose one chunk; returns a flat object `{imfs, residue, n_imfs,
    /// spectral_entropy, stationarity_score, extrema_spacing_cv}`.
    pub fn decompose_chunk(&mut self, chunk: &Float64Array) -> Result<JsValue, JsValue> {
        let len = chunk.length() as usize;
        let mut data = vec![0.0f64; len];
        chunk.copy_to(&mut data);
        validate_signal(&data)?;
        let signal = ferromode::types::Signal::from_slice(&data).map_err(error_to_js)?;
        let res = self.inner.decompose_chunk(&signal).map_err(error_to_js)?;

        let imfs = js_sys::Array::new();
        for imf in &res.imfs {
            imfs.push(&Float64Array::from(imf.as_slice()));
        }
        let out = js_sys::Object::new();
        let set = |k: &str, v: &JsValue| {
            let _ = js_sys::Reflect::set(&out, &JsValue::from_str(k), v);
        };
        set("imfs", &imfs);
        set("residue", &Float64Array::from(res.remainder.as_slice()));
        set("n_imfs", &JsValue::from_f64(res.imfs.len() as f64));
        set("spectral_entropy", &JsValue::from_f64(res.metrics.spectral_entropy));
        set("stationarity_score", &JsValue::from_f64(res.metrics.stationarity_score));
        set("extrema_spacing_cv", &JsValue::from_f64(res.metrics.extrema_spacing_cv));
        Ok(out.into())
    }

    /// Reset streaming state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

// ---------------------------------------------------------------------------
// Differentiable EMD
// ---------------------------------------------------------------------------

/// Forward differentiable EMD result.
#[wasm_bindgen]
pub struct WasmEmdForwardResult {
    inner: ferromode::ml::differentiable::ImplicitEmdContext,
}

#[wasm_bindgen]
impl WasmEmdForwardResult {
    pub fn n_imfs(&self) -> usize {
        self.inner.imfs.len()
    }

    pub fn get_imf(&self, index: usize) -> Result<Float64Array, JsValue> {
        self.inner
            .imfs
            .get(index)
            .map(|imf| Float64Array::from(imf.as_slice()))
            .ok_or_else(|| JsValue::from_str("IMF index out of range"))
    }

    pub fn get_residue(&self) -> Float64Array {
        Float64Array::from(self.inner.residue.as_slice())
    }

    pub fn reconstruction_error(&self) -> f64 {
        self.inner.reconstruction_error()
    }
}

/// Differentiable EMD forward pass.
#[wasm_bindgen]
pub fn emd_forward(
    signal: &Float64Array,
    config: &WasmEmdConfig,
) -> Result<WasmEmdForwardResult, JsValue> {
    let len = signal.length() as usize;
    let mut data = vec![0.0f64; len];
    signal.copy_to(&mut data);
    validate_signal(&data)?;
    let diff = ferromode::ml::differentiable::DifferentiableEmd::new(config.to_rust());
    let ctx = diff.forward(&data).map_err(error_to_js)?;
    Ok(WasmEmdForwardResult { inner: ctx })
}

/// Differentiable EMD backward pass (placeholder: averages upstream gradients).
#[wasm_bindgen]
pub fn emd_backward(
    grad_imfs: &js_sys::Array,
    signal: &Float64Array,
) -> Result<Float64Array, JsValue> {
    let n = signal.length() as usize;
    if n == 0 {
        return Err(JsValue::from_str("Signal must not be empty"));
    }
    if grad_imfs.length() == 0 {
        return Err(JsValue::from_str("grad_imfs must not be empty"));
    }
    let n_imfs = grad_imfs.length() as usize;
    let mut out = vec![0.0f64; n];
    for i in 0..n_imfs {
        let arr = js_sys::Float64Array::from(grad_imfs.get(i as u32));
        if arr.length() as usize != n {
            return Err(JsValue::from_str("grad_imfs row length must equal signal length"));
        }
        let mut row = vec![0.0f64; n];
        arr.copy_to(&mut row);
        for (o, g) in out.iter_mut().zip(row.iter()) {
            *o += g;
        }
    }
    let nf = n_imfs as f64;
    for o in &mut out {
        *o /= nf;
    }
    Ok(Float64Array::from(out.as_slice()))
}
