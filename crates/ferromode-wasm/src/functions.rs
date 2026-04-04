use ferromode::algorithms::ceemd::ceemd;
use ferromode::algorithms::ceemdan::ceemdan;
use ferromode::algorithms::eemd::eemd;
use ferromode::algorithms::emd::emd;
use ferromode::algorithms::hilbert::hilbert_imf;
use ferromode::algorithms::iceemdan::iceemdan;
use ferromode::algorithms::vmd::vmd;
use ferromode::boundary::BoundaryConditionType;
use ferromode::error::EmdError;
use ferromode::multivariate::direction_sampling::DirectionConfig;
use ferromode::multivariate::memd::{memd, MemdConfig};
use ferromode::multivariate::namemd::{namemd, NaMemdConfig};
use ferromode::sifting::SiftingConfig;
use ferromode::types::{DecompositionResult, HilbertResult, ImfCollection};
use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

use crate::config::{WasmEmdConfig, WasmEnsembleConfig, WasmVmdConfig};

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

#[wasm_bindgen]
pub fn memd_wasm(
    channels: &js_sys::Array,
    num_directions: usize,
    max_imfs: usize,
    seed: Option<u64>,
) -> Result<WasmDecompositionResult, JsValue> {
    if channels.length() == 0 {
        return Err(JsValue::from_str("Must provide at least one channel"));
    }

    let n_channels = channels.length() as usize;
    let mut signal: Vec<Vec<f64>> = Vec::with_capacity(n_channels);

    for i in 0..n_channels {
        let js_arr = channels.get(i);
        let float_arr = js_sys::Float64Array::from(js_arr);
        let len = float_arr.length() as usize;
        let mut data = vec![0.0f64; len];
        float_arr.copy_to(&mut data);
        validate_signal(&data)?;
        signal.push(data);
    }

    let dir_config = match seed {
        Some(s) => DirectionConfig::new(num_directions).with_seed(s),
        None => DirectionConfig::new(num_directions),
    };

    let memd_config = MemdConfig {
        direction_config: dir_config,
        max_imfs,
        sifting_config: SiftingConfig::default(),
    };

    let result = memd(&signal, &memd_config).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}

#[wasm_bindgen]
pub fn namemd_wasm(
    channels: &js_sys::Array,
    num_directions: usize,
    max_imfs: usize,
    n_noise_channels: usize,
    noise_std: f64,
    seed: Option<u64>,
) -> Result<WasmDecompositionResult, JsValue> {
    if channels.length() == 0 {
        return Err(JsValue::from_str("Must provide at least one channel"));
    }

    let n_channels = channels.length() as usize;
    let mut signal: Vec<Vec<f64>> = Vec::with_capacity(n_channels);

    for i in 0..n_channels {
        let js_arr = channels.get(i);
        let float_arr = js_sys::Float64Array::from(js_arr);
        let len = float_arr.length() as usize;
        let mut data = vec![0.0f64; len];
        float_arr.copy_to(&mut data);
        validate_signal(&data)?;
        signal.push(data);
    }

    let dir_config = match seed {
        Some(s) => DirectionConfig::new(num_directions).with_seed(s),
        None => DirectionConfig::new(num_directions),
    };

    let base_config = MemdConfig {
        direction_config: dir_config,
        max_imfs,
        sifting_config: SiftingConfig::default(),
    };

    let namemd_config = NaMemdConfig { base_config, n_noise_channels, noise_std, seed };

    let result = namemd(&signal, &namemd_config).map_err(error_to_js)?;
    Ok(WasmDecompositionResult { inner: result })
}
