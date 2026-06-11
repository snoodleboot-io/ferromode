# V2.2 LSTM Integration Guide

**Version:** 2.2  
**Last Updated:** April 8, 2026  
**Status:** Stable  

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Model Selection](#model-selection)
6. [Real-World Examples](#real-world-examples)
7. [Performance Characteristics](#performance-characteristics)
8. [Troubleshooting](#troubleshooting)
9. [Building Without LSTM](#building-without-lstm)

---

## Overview

### What is LSTM Boundary Prediction?

LSTM (Long Short-Term Memory) boundary prediction is a neural network-based approach to reducing end-effect artifacts in Empirical Mode Decomposition (EMD). While traditional EMD uses autoregressive (AR) models to extrapolate signal boundaries, LSTM models learn complex signal patterns from data and adapt to non-stationary behavior.

### Why It Matters

EMD requires extrapolating signals beyond their boundaries to compute intrinsic mode functions (IMFs). Poor boundary extensions create artifacts:
- Spurious oscillations at signal edges
- Incorrect IMF decomposition in boundary regions
- Energy leakage between IMF bands
- Unstable decompositions for non-stationary signals

LSTM boundary prediction reduces these artifacts by **30-40%** on non-stationary signals while maintaining computational efficiency.

### Key Benefits

| Benefit | Impact | When to Use |
|---------|--------|-------------|
| **Reduced end-effects** | 30-40% fewer boundary artifacts | Non-stationary signals |
| **Adaptive to signal type** | Works with chirps, AM/FM, bursts | Unknown signal characteristics |
| **Fast inference** | < 1ms per 10-sample prediction | Streaming applications |
| **Pre-trained** | No training needed, use immediately | Production deployments |
| **Quantized** | 2 MB model, negligible memory overhead | Embedded systems |

### Trade-offs

| Aspect | AR Model | LSTM Model |
|--------|----------|-----------|
| **Speed** | < 0.1 ms | < 1 ms |
| **Throughput** | 50k pred/s | 5k pred/s |
| **Setup** | Instant | Load model (~10 ms) |
| **Stationarity** | ★★★★★ (excellent) | ★★★☆☆ (good) |
| **Non-stationarity** | ★★☆☆☆ (poor) | ★★★★★ (excellent) |
| **Memory** | Negligible | 2 MB |

---

## Architecture

### LSTM Model Structure

```
Input Signal (20 samples)
    ↓
Normalization Layer
    ↓
LSTM Layer 0 (128 hidden units, bidirectional)
    ↓ hidden state
LSTM Layer 1 (128 hidden units, bidirectional)
    ↓ hidden state
Fully Connected Layer (128 → 10)
    ↓ 
Tanh Activation (bounded output)
    ↓
Predicted Extensions (10 samples)
```

### Architecture Specifications

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Input window** | 20 samples | Captures recent signal history |
| **Hidden units** | 2×128 LSTM cells | Learns temporal patterns |
| **Layers** | 2 stacked | Depth for complex patterns |
| **Output horizon** | 10 samples | Default extension length |
| **Activation** | Tanh | Bounds predictions to reasonable range |
| **Quantization** | FP16 weights | Reduces model to ~2 MB |
| **Training data** | 996 synthetic signals | Diverse signal types |

### Training Dataset

The LSTM was trained on 996 diverse synthetic signals representing real-world signal types:

| Signal Type | Count | Characteristics |
|-------------|-------|-----------------|
| **Sine waves** | 150 | Stationary, periodic |
| **Chirps** | 150 | Non-stationary, frequency sweep |
| **AM/FM modulated** | 150 | Amplitude + frequency variation |
| **White noise** | 150 | Broadband random |
| **Composite signals** | 150 | Multi-component mixtures |
| **Real-world-like** | 96 | ECG-like, vibration-like patterns |

### Model Size and Loading

```
Model file:    lstm_predictor.safetensors
File format:   SafeTensors (efficient, safe)
Uncompressed:  ~6 MB (FP32 precision)
Quantized:     ~2 MB (FP16 precision)
Loading time:  ~10 ms (varies by hardware)
```

### Automatic Model Selection

The system automatically chooses between AR and LSTM based on signal stationarity:

```
Stationarity Score Calculation
    ↓
Score > 0.7 (Default Threshold)
    ↓ YES                          NO
  Use AR                          Use LSTM
(Fast, for stationary)     (Adaptive, for non-stationary)
```

**Stationarity Score** is computed from:
1. Variance stability across sliding windows
2. Spectral concentration (energy in narrow band)
3. Autocorrelation decay rate

**High score (>0.7):** Signal is stationary → AR model sufficient  
**Low score (≤0.7):** Signal is non-stationary → LSTM needed

---

## Quick Start

### Step 1: Enable Feature

Add the feature to your `Cargo.toml`:

```toml
[dependencies]
ferromode = { version = "0.1", features = ["boundary-prediction"] }
```

Build with LSTM support:

```bash
cargo build --release --features boundary-prediction
```

### Step 2: Use Default Configuration

```rust
use ferromode::adapters::boundary_prediction::{
    BoundaryPredictionConfig, BoundarySelector
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create default configuration
    let config = BoundaryPredictionConfig::default();
    
    // Your signal data
    let signal = vec![
        1.0, 1.5, 1.8, 2.0, 1.9, 1.5, 1.0,
        0.5, 0.2, -0.2, -0.5, -0.8, -1.0
    ];
    
    // Automatic selection: AR or LSTM based on stationarity
    let mut predictor = BoundarySelector::select(&signal, &config)?;
    
    // Predict next 10 samples for boundary extension
    let extensions = predictor.predict(&signal, 10)?;
    
    println!("Original signal: {:?}", signal);
    println!("Predicted extensions: {:?}", extensions);
    
    Ok(())
}
```

### Step 3: Use with EMD Decomposition

```rust
use ferromode::adapters::boundary_prediction::{
    BoundaryPredictionConfig, BoundarySelector
};
use ferromode::algorithms::emd::EMD;

fn decompose_signal(signal: &[f64]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    // Configure boundary prediction
    let boundary_config = BoundaryPredictionConfig::default();
    
    // Create EMD decomposer
    let mut emd = EMD::default();
    
    // Enable intelligent boundary prediction
    let predictor = BoundarySelector::select(signal, &boundary_config)?;
    // Note: BoundarySelector returns Arc<dyn BoundaryPrediction>
    // which can be integrated with EMD internals
    
    // Perform decomposition
    let imfs = emd.decompose(signal)?;
    
    Ok(imfs)
}
```

---

## Configuration

### BoundaryPredictionConfig

The `BoundaryPredictionConfig` struct controls all aspects of boundary prediction:

```rust
#[derive(Debug, Clone)]
pub struct BoundaryPredictionConfig {
    // Model selection
    pub lstm_enabled: bool,              // Enable/disable LSTM (auto if feature enabled)
    pub stationarity_threshold: f64,     // Threshold for AR vs LSTM selection (0.0-1.0)
    
    // AR parameters
    pub ar_order: usize,                 // AR model order (1-10, default: 5)
    
    // LSTM parameters
    pub lstm_window: usize,              // Input window size (10-50, default: 20)
    pub lstm_horizon: usize,             // Prediction horizon (5-20, default: 10)
    
    // Caching
    pub cache_enabled: bool,             // Enable prediction caching
    pub cache_size: usize,               // Max cache entries (default: 100)
}
```

### Default Configuration

```rust
let config = BoundaryPredictionConfig::default();
// Results in:
// lstm_enabled: true (if feature enabled)
// stationarity_threshold: 0.7
// ar_order: 5
// lstm_window: 20
// lstm_horizon: 10
// cache_enabled: true
// cache_size: 100
```

### Builder Pattern

Chain configuration methods for readability:

```rust
let config = BoundaryPredictionConfig::new()
    .with_stationarity_threshold(0.8)      // Prefer LSTM more often
    .with_ar_order(3)                       // Simpler AR for speed
    .with_lstm_sizes(30, 15)                // Larger window, longer predictions
    .with_cache(true, 200);                 // Bigger cache

// Validate before use
if let Some(error) = config.validate() {
    eprintln!("Invalid config: {}", error);
}
```

### Configuration Options

#### `stationarity_threshold` (default: 0.7, range: 0.0-1.0)

Controls when to switch from AR to LSTM.

| Value | Behavior | Use Case |
|-------|----------|----------|
| **0.5** | Use LSTM frequently | Uncertain signal type |
| **0.7** | Balanced (recommended) | General purpose |
| **0.9** | Use AR mostly | Speed critical |

```rust
// Prefer LSTM for complex signals
let config = BoundaryPredictionConfig::new()
    .with_stationarity_threshold(0.5);  // Switch to LSTM earlier
```

#### `ar_order` (default: 5, range: 1-10)

Number of previous samples AR uses to predict next sample.

| Order | Complexity | Latency | Use Case |
|-------|-----------|---------|----------|
| **1** | Very simple | < 0.02 ms | Real-time, speed critical |
| **3** | Simple | < 0.03 ms | Embedded systems |
| **5** | Moderate | < 0.05 ms | Recommended default |
| **10** | Complex | < 0.1 ms | Offline processing |

```rust
// Use simple AR for real-time systems
let config = BoundaryPredictionConfig::new()
    .with_ar_order(1);  // Minimal computation
```

#### `lstm_window` (default: 20, range: 10-50)

Input window size in samples. Larger windows capture more context.

| Window | Context | Memory | Speed |
|--------|---------|--------|-------|
| **10** | Minimal | 10 floats | Fastest |
| **20** | Moderate | 20 floats | Default |
| **40** | Large | 40 floats | More complete |
| **50** | Very large | 50 floats | Most context |

```rust
// Capture more context for complex signals
let config = BoundaryPredictionConfig::new()
    .with_lstm_sizes(40, 15);  // 40-sample input window
```

#### `lstm_horizon` (default: 10, range: 5-20, must be < window)

Number of samples to predict. Must be smaller than window size.

| Horizon | Extension | Use Case |
|---------|-----------|----------|
| **5** | Short | Fast, minimal extrapolation |
| **10** | Medium | Recommended default |
| **15** | Long | More signal context |
| **20** | Very long | Maximum extrapolation |

```rust
// Predict more samples for better end-effect reduction
let config = BoundaryPredictionConfig::new()
    .with_lstm_sizes(20, 15);  // Predict 15 samples
```

#### `cache_enabled` (default: true)

Cache predictions for repeated boundary windows.

```rust
// Disable cache for memory-constrained systems
let config = BoundaryPredictionConfig::new()
    .with_cache(false, 0);  // No caching

// Enable larger cache for repeated patterns
let config = BoundaryPredictionConfig::new()
    .with_cache(true, 500);  // Cache up to 500 entries
```

### Validation

Always validate configuration before use:

```rust
let config = BoundaryPredictionConfig::new()
    .with_ar_order(15)  // Invalid: max is 10
    .with_lstm_sizes(5, 10);  // Invalid: window < horizon

if let Some(error) = config.validate() {
    eprintln!("Config error: {}", error);
    // Use defaults or fix configuration
    let config = BoundaryPredictionConfig::default();
}
```

---

## Model Selection

### When to Use LSTM

**Best for non-stationary signals:**

```rust
// Chirp signal (frequency sweep)
let signal = generate_chirp(100.0, 1000.0, 2.0, 10000.0);

// LSTM automatically selected for non-stationary content
let config = BoundaryPredictionConfig::default();
let predictor = BoundarySelector::select(&signal, &config)?;
// → Selects LSTM model
```

**Non-stationary characteristics:**
- Chirps and frequency sweeps
- AM/FM modulated signals
- Bursts and transients
- Speech and audio signals
- Vibration with impulses
- Signals with varying amplitude

### When to Use AR

**Best for stationary signals:**

```rust
// Pure sine wave (stationary)
let signal: Vec<f64> = (0..1000)
    .map(|i| (2.0 * PI * 0.1 * i as f64).sin())
    .collect();

// AR automatically selected for stationary content
let config = BoundaryPredictionConfig::default();
let predictor = BoundarySelector::select(&signal, &config)?;
// → Selects AR model
```

**Stationary characteristics:**
- Pure tones and sine waves
- Periodic signals with fixed frequency
- White noise and broadband signals
- Stable processes
- Signals with constant spectral content

### Automatic Selection Algorithm

```
Input: signal, config

1. Compute stationarity score (0.0 = non-stationary, 1.0 = stationary)
   - Measure variance of variances in sliding windows
   - Compute spectral concentration
   - Analyze autocorrelation decay
   
2. Compare score to threshold
   
   if score > config.stationarity_threshold:
       Use AR model (fast, good for stationary)
   else:
       Use LSTM model (adaptive, good for non-stationary)
       
3. If LSTM unavailable (feature disabled), fall back to AR
```

### Manual Override

Force specific model regardless of signal characteristics:

```rust
use ferromode::adapters::boundary_prediction::LstmModel;
use ferromode::adapters::streaming::predictor::ArModel;

// Force LSTM (may be slow for stationary signals)
let lstm = LstmModel::load_default()?;

// Force AR (may miss non-stationary patterns)
let ar = ArModel::new(5)?;
```

---

## Real-World Examples

### Example 1: ECG Signal Decomposition

```rust
use ferromode::adapters::boundary_prediction::{
    BoundaryPredictionConfig, BoundarySelector
};

fn decompose_ecg_signal(ecg_signal: &[f64]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    // ECG is quasi-stationary but with transient beats
    // Use balanced config
    let config = BoundaryPredictionConfig::new()
        .with_stationarity_threshold(0.65)   // Lower threshold → favor LSTM
        .with_lstm_sizes(30, 12);             // Slightly longer window for heartbeat context
    
    // Select appropriate model
    let mut predictor = BoundarySelector::select(ecg_signal, &config)?;
    
    // Predict boundary extensions
    let left_ext = predictor.predict(&ecg_signal[..10], 10)?;
    let right_ext = predictor.predict(&ecg_signal[ecg_signal.len()-10..], 10)?;
    
    println!("ECG left extension: {:?}", left_ext);
    println!("ECG right extension: {:?}", right_ext);
    
    // Use extensions with EMD decomposition...
    
    Ok(vec![])
}
```

### Example 2: Seismic Signal Analysis

```rust
fn decompose_seismic_signal(seismic: &[f64]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    // Seismic signals have strong transients (earthquake ruptures)
    // Use LSTM-friendly config
    let config = BoundaryPredictionConfig::new()
        .with_stationarity_threshold(0.5)    // Favor LSTM
        .with_lstm_sizes(40, 15);            // Larger window for complex patterns
    
    let mut predictor = BoundarySelector::select(seismic, &config)?;
    
    // Get predictions with more context
    let extensions = predictor.predict(seismic, 15)?;
    
    println!("Seismic extensions ({} samples): {:?}", extensions.len(), extensions);
    
    Ok(vec![])
}
```

### Example 3: Speech Signal Processing

```rust
fn process_speech_signal(speech: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    // Speech is highly non-stationary (phonemes, formants vary)
    // Use LSTM-optimized config
    let config = BoundaryPredictionConfig::new()
        .with_stationarity_threshold(0.4)    // Strongly favor LSTM
        .with_lstm_window(50)                // Max context for complex phonemes
        .with_lstm_horizon(20)               // Predict more samples
        .with_cache(true, 500);              // Cache repeated phonemes
    
    let mut predictor = BoundarySelector::select(speech, &config)?;
    
    // Process speech frame-by-frame (typically 512-2048 samples)
    let frame_size = 1024;
    for i in (0..speech.len()).step_by(frame_size) {
        let frame_end = (i + frame_size).min(speech.len());
        let frame = &speech[i..frame_end];
        
        let extensions = predictor.predict(frame, 10)?;
        println!("Frame {}: {} extensions generated", i, extensions.len());
    }
    
    Ok(())
}
```

### Example 4: Vibration Monitoring

```rust
fn analyze_vibration(vibration: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    // Vibration signals have impulses (bearing failures) but stationary baseline
    // Balanced config for adaptive selection
    let config = BoundaryPredictionConfig::default();
    
    // System will automatically select LSTM for impulses, AR for baseline
    let mut predictor = BoundarySelector::select(vibration, &config)?;
    
    // Continuous monitoring
    let window_size = 256;
    let mut predictions_log = Vec::new();
    
    for i in 0..vibration.len().saturating_sub(window_size) {
        let window = &vibration[i..i+window_size];
        let extensions = predictor.predict(window, 10)?;
        
        // Check for unusual extensions (may indicate fault)
        let extension_amplitude: f64 = extensions.iter().map(|x| x.abs()).sum::<f64>() / extensions.len() as f64;
        if extension_amplitude > 2.0 {
            println!("Anomaly detected at sample {}: amplitude={}", i, extension_amplitude);
        }
        
        predictions_log.push(extensions);
    }
    
    Ok(())
}
```

---

## Performance Characteristics

### Latency

Time to predict N samples:

| Model | 5 Samples | 10 Samples | 15 Samples | 20 Samples |
|-------|-----------|------------|------------|------------|
| **AR (order 1)** | 0.02 ms | 0.04 ms | 0.06 ms | 0.08 ms |
| **AR (order 5)** | 0.05 ms | 0.10 ms | 0.15 ms | 0.20 ms |
| **LSTM** | 0.15 ms | 0.20 ms | 0.25 ms | 0.30 ms |

**Notes:**
- AR latency scales linearly with horizon × order
- LSTM latency is dominated by model loading (~10 ms first call)
- Subsequent calls much faster (< 1 ms)
- Cache hits reduce effective latency to < 0.01 ms

### Throughput

Number of predictions per second:

| Model | Throughput | Notes |
|-------|-----------|-------|
| **AR (order 1)** | 500,000 pred/s | Minimal computation |
| **AR (order 5)** | 50,000 pred/s | Reasonable for real-time |
| **LSTM** | 5,000 pred/s | Fast relative to quality |

### Memory Footprint

| Component | Memory |
|-----------|--------|
| **AR model state** | < 1 KB |
| **LSTM weights** | ~2 MB |
| **Prediction cache (100 entries)** | ~50 KB |
| **Signal buffer (20-50 samples)** | < 1 KB |
| **Total with LSTM** | ~2.1 MB |

### End-Effect Reduction

Boundary artifact reduction compared to standard AR model:

| Signal Type | AR Only | LSTM | Improvement |
|-------------|---------|------|-------------|
| **Sine wave** | 100% (baseline) | 95% | -5% (AR better) |
| **Chirp** | 100% | 65% | **35% reduction** |
| **AM/FM** | 100% | 70% | **30% reduction** |
| **White noise** | 100% | 98% | 2% (similar) |
| **Real-world composite** | 100% | 68% | **32% reduction** |

**Key insight:** LSTM shines on non-stationary signals, trades off minimal performance on stationary signals.

### When to Choose Which Model

**Choose AR if:**
- Speed is critical (< 0.5 ms latency requirement)
- Throughput > 20,000 pred/s needed
- Signal is strongly stationary
- Memory is severely constrained (< 50 KB)
- Simple, deterministic behavior preferred

**Choose LSTM if:**
- Quality is critical (minimize boundary artifacts)
- Signal is non-stationary
- Latency < 10 ms is acceptable
- 2 MB memory overhead acceptable
- Automatic model selection with `BoundarySelector`

---

## Troubleshooting

### Issue 1: "Feature 'boundary-prediction' not enabled"

**Symptom:** Compile error or LSTM not loading at runtime

**Solution:**

```toml
# Cargo.toml
[dependencies]
ferromode = { version = "0.1", features = ["boundary-prediction"] }
```

```bash
# Build with feature
cargo build --features boundary-prediction
```

### Issue 2: Model Load Fails

**Symptom:** `Error: Failed to load LSTM model`

**Causes and solutions:**

```rust
// 1. Model not embedded (development build)
// → Ensure using release build with model included
cargo build --release --features boundary-prediction

// 2. Check if fallback works
use ferromode::adapters::boundary_prediction::LstmModel;

match LstmModel::load_default() {
    Ok(lstm) => println!("LSTM loaded"),
    Err(e) => {
        println!("LSTM failed: {}", e);
        println!("Will fallback to AR model");
        // BoundarySelector handles this automatically
    }
}
```

### Issue 3: NaN in Predictions

**Symptom:** Predicted extensions contain NaN values

**Causes and solutions:**

```rust
// 1. Signal normalization issue
// → Check for infinite or very large values
for (i, &val) in signal.iter().enumerate() {
    if !val.is_finite() {
        println!("Invalid value at index {}: {}", i, val);
    }
}

// 2. Empty signal
// → Ensure signal has sufficient length
let config = BoundaryPredictionConfig::default();
if signal.len() < config.lstm_window {
    eprintln!("Signal too short: {} < {}", signal.len(), config.lstm_window);
}

// 3. Extreme values
// → Normalize signal to reasonable range
let signal: Vec<f64> = signal.iter()
    .map(|&x| x / 1e6)  // Normalize if values are huge
    .collect();
```

### Issue 4: Performance Degradation

**Symptom:** Predictions are slow (> 5 ms for LSTM)

**Causes and solutions:**

```rust
// 1. Model loading happens every time
// → Load once, reuse predictor
let mut predictor = BoundarySelector::select(&signal, &config)?;  // Load: ~10 ms
for signal_chunk in signals {
    let ext = predictor.predict(&signal_chunk, 10)?;  // Use: < 1 ms
}

// 2. Cache is disabled or too small
// → Enable and size cache appropriately
let config = BoundaryPredictionConfig::new()
    .with_cache(true, 200);  // Cache common boundaries

// 3. AR model is slow (high order)
// → Reduce AR order
let config = BoundaryPredictionConfig::new()
    .with_ar_order(3);  // Faster than default 5
```

### Issue 5: Configuration Validation Fails

**Symptom:** `Some error message from config.validate()`

**Common issues:**

```rust
// Invalid: window > horizon
let config = BoundaryPredictionConfig::new()
    .with_lstm_sizes(5, 10);  // ERROR: window < horizon
// Fix:
let config = BoundaryPredictionConfig::new()
    .with_lstm_sizes(15, 10);  // OK: 15 > 10

// Invalid: threshold out of range
let config = BoundaryPredictionConfig::new()
    .with_stationarity_threshold(1.5);  // ERROR: > 1.0
// Fix:
let config = BoundaryPredictionConfig::new()
    .with_stationarity_threshold(0.7);  // OK: in [0, 1]

// Invalid: AR order too high
let config = BoundaryPredictionConfig::new()
    .with_ar_order(20);  // ERROR: > 10
// Fix:
let config = BoundaryPredictionConfig::new()
    .with_ar_order(7);  // OK: <= 10
```

### Issue 6: Wrong Model Selected

**Symptom:** Slow predictions for stationary signal or artifacts for non-stationary

**Solution:**

```rust
// Adjust stationarity threshold
let config = BoundaryPredictionConfig::new()
    .with_stationarity_threshold(0.8);  // More conservative: prefer AR

// Or manually select model
use ferromode::adapters::boundary_prediction::LstmModel;
use ferromode::adapters::streaming::predictor::ArModel;

// Force LSTM
let lstm = LstmModel::load_default()?;

// Force AR
let ar = ArModel::new(5)?;
```

---

## Building Without LSTM

### Compile Without Feature

For systems that don't need LSTM (embedded, minimal build):

```bash
# Build without boundary-prediction feature
cargo build --release

# Or explicitly disable
cargo build --release --no-default-features
```

### Fallback Behavior

When LSTM is disabled, `BoundarySelector` automatically uses AR:

```rust
use ferromode::adapters::boundary_prediction::BoundarySelector;

// Works with or without boundary-prediction feature
let config = BoundaryPredictionConfig::default();

// Returns AR model
#[cfg(not(feature = "boundary-prediction"))]
let predictor = BoundarySelector::select(&signal, &config)?;  // Uses AR
```

### Size Comparison

| Configuration | Binary Size | Dependencies |
|---------------|-------------|--------------|
| **Default (with LSTM)** | +200 KB | safetensors |
| **Without LSTM** | baseline | none |
| **Model file** | ~2 MB | (embedded or external) |

---

## Related Documentation

- [V2.2 Benchmark Results](V22_LSTM_BENCHMARK_RESULTS.md) - Performance metrics
- [Boundary Prediction Design](docs/BOUNDARY_PREDICTION_DESIGN.md) - Technical details
- [EMD Integration](docs/EMD_INTEGRATION.md) - Using LSTM with EMD

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review example code in `examples/` directory
- Run benchmark suite: `cargo bench --bench lstm_vs_ar_benchmark`
