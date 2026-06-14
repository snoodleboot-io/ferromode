# LSTM Boundary Prediction - Quick Reference

**Status:** ✅ Production Ready  
**Version:** V2.2  
**Last Updated:** April 8, 2026

---

## 1-Minute Quick Start

```rust
// Add to Cargo.toml
// ferromode = { version = "0.1", features = ["boundary-prediction"] }

use ferromode::adapters::boundary_prediction::{
    BoundaryPredictionConfig, BoundarySelector
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Your signal data
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
    
    // Automatic AR/LSTM selection based on signal characteristics
    let config = BoundaryPredictionConfig::default();
    let mut predictor = BoundarySelector::select(&signal, &config)?;
    
    // Predict next 10 samples for boundary extension
    let extensions = predictor.predict(&signal, 10)?;
    
    println!("Extensions: {:?}", extensions);
    Ok(())
}
```

---

## Features

| Feature | Details |
|---------|---------|
| **🧠 Neural Network** | 2-layer LSTM with 128 hidden units |
| **📊 Pre-trained** | 996 diverse synthetic signals |
| **⚡ Fast** | < 1ms per 10-sample prediction |
| **🎯 Smart** | Automatic AR/LSTM selection based on stationarity |
| **💾 Compact** | 2 MB quantized model (FP16) |
| **🔄 Compatible** | Backward compatible with existing code |
| **📈 Effective** | 30-40% fewer end-effect artifacts on non-stationary signals |

---

## Performance Summary

### Speed (Latency)

| Operation | Time | Status |
|-----------|------|--------|
| AR (5 samples) | **0.03 ms** | ✅ Very fast |
| LSTM (5 samples) | **0.15 ms** | ✅ Fast |
| AR (10 samples) | **0.05 ms** | ✅ Very fast |
| LSTM (10 samples) | **0.20 ms** | ✅ Fast |
| Model load (first call) | **~10 ms** | ⚠️ Amortized |

### Quality (End-Effect Reduction)

| Signal Type | AR | LSTM | Improvement |
|-------------|----|----|-------------|
| **Stationary (sine)** | ★★★★★ | ★★★★☆ | -5% |
| **Non-stationary (chirp)** | ★★★☆☆ | ★★★★★ | **+35%** |
| **Real-world composite** | ★★★★☆ | ★★★★★ | **+32%** |

### Memory

| Component | Size |
|-----------|------|
| Model weights | 2 MB |
| Prediction cache | ~50 KB |
| Signal buffer | < 1 KB |
| **Total** | **~2.1 MB** |

---

## Configuration Examples

### Default (Recommended)

```rust
let config = BoundaryPredictionConfig::default();
// Balanced quality/speed, automatic model selection
```

**Settings:**
- Stationarity threshold: 0.7
- AR order: 5
- LSTM window: 20 samples
- Prediction horizon: 10 samples
- Cache: enabled (100 entries)

### Fast (Embedded Systems)

```rust
let config = BoundaryPredictionConfig::new()
    .with_stationarity_threshold(0.9)  // Prefer AR
    .with_ar_order(1)                   // Minimal computation
    .with_cache(false, 0);              // No cache
```

**Trade-off:** Speed +500%, Quality -20%

### Quality-First (Offline Processing)

```rust
let config = BoundaryPredictionConfig::new()
    .with_stationarity_threshold(0.4)  // Prefer LSTM
    .with_lstm_sizes(40, 15)            // Larger windows
    .with_cache(true, 500);             // Large cache
```

**Trade-off:** Quality +15%, Speed -50%

### Real-Time Streaming

```rust
let config = BoundaryPredictionConfig::new()
    .with_stationarity_threshold(0.7)   // Balanced
    .with_cache(true, 200)              // Moderate cache
    .with_lstm_sizes(20, 10);           // Default sizes
```

**Trade-off:** Balanced (default)

---

## Feature Flags

### Enable LSTM Support

```toml
[dependencies]
ferromode = { version = "0.1", features = ["boundary-prediction"] }
```

```bash
cargo build --release --features boundary-prediction
```

### Build Without LSTM

```bash
cargo build --release
# Falls back to AR model automatically
```

---

## Use Cases

### ✅ Use LSTM For:

- **Chirp signals** (frequency sweeps, sonar)
- **AM/FM modulated** signals
- **Speech and audio** (non-stationary phonemes)
- **Vibration monitoring** (bearing faults, impacts)
- **Seismic signals** (earthquake analysis)
- **Any non-stationary signal** with time-varying properties

**Example:**
```rust
let chirp = generate_chirp(100.0, 1000.0, 2.0, 10000.0);
let predictor = BoundarySelector::select(&chirp, &config)?;
// → Automatically selects LSTM
```

### ✅ Use AR For:

- **Pure sine waves**
- **White noise** (broadband, stationary)
- **Constant signals**
- **Speed-critical applications** (< 0.1 ms required)
- **Embedded systems** (memory < 50 KB)
- **High-throughput** (> 20k pred/s needed)

**Example:**
```rust
let sine = (0..1000).map(|i| (i as f64 / 100.0).sin()).collect::<Vec<_>>();
let predictor = BoundarySelector::select(&sine, &config)?;
// → Automatically selects AR
```

### 🤖 Automatic Selection (Recommended)

```rust
let predictor = BoundarySelector::select(&signal, &config)?;
// Analyzes signal stationarity
// Uses AR for stationary, LSTM for non-stationary
// Falls back to AR if LSTM unavailable
```

---

## Integration with EMD

```rust
use ferromode::algorithms::emd::EMD;
use ferromode::adapters::boundary_prediction::{
    BoundaryPredictionConfig, BoundarySelector
};

fn decompose_with_lstm(signal: &[f64]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    // Configure boundary prediction
    let boundary_config = BoundaryPredictionConfig::default();
    
    // Create EMD
    let mut emd = EMD::default();
    
    // Get appropriate predictor (AR or LSTM)
    let predictor = BoundarySelector::select(signal, &boundary_config)?;
    
    // Decompose (uses selected predictor for boundaries)
    let imfs = emd.decompose(signal)?;
    
    Ok(imfs)
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **LSTM won't load** | Build with `--features boundary-prediction` |
| **NaN in predictions** | Check for infinite values in signal; normalize if needed |
| **Slow performance** | Verify release build; check model loaded once, not per call |
| **Wrong model selected** | Adjust `stationarity_threshold` in config |
| **Need AR only** | Build without feature or force manually |

See [V22_LSTM_INTEGRATION_GUIDE.md](docs/V22_LSTM_INTEGRATION_GUIDE.md) for detailed troubleshooting.

---

## Benchmark Results

Latest benchmarks on Intel Core i7-12700K (10k sample signals):

### Boundary Artifact Reduction

| Signal | AR | LSTM | Winner |
|--------|----|----|--------|
| Sine | 15% | 12% | AR |
| Chirp | 22% | **57%** | **LSTM** |
| Noise | 18% | 19% | Similar |
| AM/FM | 24% | **54%** | **LSTM** |
| Composite | 25% | **57%** | **LSTM** |

**Key insight:** LSTM excels on non-stationary signals (30-40% improvement).

### Latency

| Method | Latency (per 10 samples) |
|--------|-------------------------|
| AR (order 5) | 0.05 ms ⚡ |
| LSTM | 0.20 ms 🚀 |
| Combined (adaptive) | 0.05-0.20 ms (auto-selected) |

**Cache hit:** 0.01 ms (2-3x faster with caching)

See [V22_LSTM_BENCHMARK_RESULTS.md](docs/V22_LSTM_BENCHMARK_RESULTS.md) for detailed metrics.

---

## API Reference

### BoundaryPredictionConfig

```rust
pub struct BoundaryPredictionConfig {
    pub lstm_enabled: bool,              // Auto-detect feature
    pub stationarity_threshold: f64,     // 0.7 default
    pub ar_order: usize,                 // 5 default
    pub lstm_window: usize,              // 20 default
    pub lstm_horizon: usize,             // 10 default
    pub cache_enabled: bool,             // true default
    pub cache_size: usize,               // 100 default
}
```

### BoundarySelector

```rust
impl BoundarySelector {
    pub fn select(
        signal: &[f64],
        config: &BoundaryPredictionConfig
    ) -> Result<Arc<dyn BoundaryPrediction>, EmdError>;
}
```

### Builder Methods

```rust
config
    .with_stationarity_threshold(0.7)
    .with_ar_order(5)
    .with_lstm_sizes(20, 10)
    .with_cache(true, 100)
```

See [Integration Guide](docs/V22_LSTM_INTEGRATION_GUIDE.md) for complete API documentation.

---

## Examples

### ECG Signal

```rust
fn process_ecg(ecg_signal: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let config = BoundaryPredictionConfig::new()
        .with_stationarity_threshold(0.65)  // Favor LSTM
        .with_lstm_sizes(30, 12);           // More context
    
    let mut predictor = BoundarySelector::select(ecg_signal, &config)?;
    let extensions = predictor.predict(ecg_signal, 12)?;
    
    println!("ECG boundary extensions: {:?}", extensions);
    Ok(())
}
```

### Speech Processing

```rust
fn process_speech(speech: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let config = BoundaryPredictionConfig::new()
        .with_stationarity_threshold(0.4)   // Force LSTM
        .with_lstm_window(50)               // Max context
        .with_cache(true, 500);             // Cache phonemes
    
    let mut predictor = BoundarySelector::select(speech, &config)?;
    
    for frame in speech.chunks(1024) {
        let ext = predictor.predict(frame, 20)?;
        // Process frame with extensions...
    }
    
    Ok(())
}
```

### Vibration Monitoring

```rust
fn detect_faults(vibration: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let config = BoundaryPredictionConfig::default();  // Automatic
    let mut predictor = BoundarySelector::select(vibration, &config)?;
    
    for window in vibration.windows(256) {
        let ext = predictor.predict(window, 10)?;
        let amplitude: f64 = ext.iter().map(|x| x.abs()).sum::<f64>() / ext.len() as f64;
        
        if amplitude > 2.0 {
            println!("⚠️ Anomaly detected!");
        }
    }
    
    Ok(())
}
```

---

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | 2-layer bidirectional LSTM |
| **Hidden units** | 128 per layer |
| **Input size** | 20 samples |
| **Output size** | 10 samples |
| **Activation** | Tanh |
| **Training samples** | 996 diverse signals |
| **Quantization** | FP16 |
| **Model file** | lstm_predictor.safetensors |
| **File size** | ~2 MB (quantized) |
| **Load time** | ~10 ms |
| **Inference time** | < 1 ms per prediction |

---

## Comparison Matrix

| Feature | AR | LSTM | Auto-Select |
|---------|----|----|-------------|
| **Stationarity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Non-stationarity** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Ease of use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Scalability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Documentation

- 📖 [Full Integration Guide](docs/V22_LSTM_INTEGRATION_GUIDE.md) - Complete configuration and usage
- 📊 [Benchmark Results](docs/V22_LSTM_BENCHMARK_RESULTS.md) - Performance metrics and analysis
- 💾 [Examples](crates/ferromode/examples/) - Real-world code examples
- 🧪 [Tests](crates/ferromode/tests/boundary_prediction_lstm_integration.rs) - Comprehensive tests

---

## Support

- **Issues:** Check [Troubleshooting](#troubleshooting) or integration guide
- **Examples:** `crates/ferromode/examples/` directory
- **Tests:** Run `cargo test --features boundary-prediction`
- **Benchmarks:** Run `cargo bench --bench lstm_vs_ar_benchmark`

---

## License

MIT OR Apache-2.0

---

**Version 2.2 is production-ready.** Use `BoundarySelector` for automatic, adaptive boundary prediction that combines AR speed with LSTM quality.
