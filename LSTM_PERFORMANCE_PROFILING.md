# LSTM Inference Performance Profiling Benchmark

## Overview

A comprehensive performance profiling benchmark for LSTM-based boundary prediction in the Ferromode signal processing library. Measures inference latency, memory usage, cache effectiveness, and optimization opportunities in release mode.

## File Location

**Benchmark:** `crates/ferromode/benches/lstm_performance_profile.rs` (599 lines)

## Compilation

```bash
# Check compilation
cargo check --bench lstm_performance_profile --features boundary-prediction

# Build in release mode
cargo build --bench lstm_performance_profile --features boundary-prediction --release

# Run the benchmark
cargo bench --bench lstm_performance_profile --features boundary-prediction
```

## Measurements Included

### 1. Model Loading (`1_model_loading`)
- **Metric:** Time to deserialize SafeTensors model from embedded bytes
- **Configuration:** 10 samples
- **Expected Time:** ~0.25-0.35 ms
- **Purpose:** Establish baseline for cold-start overhead
- **Benchmark:** `load_default_model`

### 2. Single Prediction - Small Signal (`2a_single_prediction_20samples`)
- **Signal Length:** 20 samples
- **Metric:** Latency for first prediction (includes fit + predict)
- **Configuration:** 100 samples, 10-second measurement
- **Expected Time:** ~150-160 µs
- **Purpose:** Measure performance on minimal input size
- **Benchmark:** `small_signal_first_prediction`

### 3. Single Prediction - Medium Signal (`2b_single_prediction_100samples`)
- **Signal Length:** 100 samples  
- **Metric:** Latency for standard-size prediction
- **Configuration:** 100 samples, 10-second measurement
- **Expected Time:** ~148-157 µs
- **Purpose:** Measure performance on typical input size
- **Benchmark:** `medium_signal_prediction`

### 4. Single Prediction - Large Signal (`2c_single_prediction_1000samples`)
- **Signal Length:** 1000 samples
- **Metric:** Latency for large-input prediction
- **Configuration:** 100 samples, 10-second measurement
- **Expected Time:** ~139-143 µs
- **Purpose:** Verify performance scales with input size
- **Benchmark:** `large_signal_prediction`

### 5. Batch Processing (`3_batch_processing`)
- **Configuration:** 100 sequential predictions on 100-sample signal
- **Metric:** Total and per-prediction latency in batch workload
- **Configuration:** 100 samples, 15-second measurement
- **Expected Time:** ~160-175 µs per iteration (batch)
- **Purpose:** Measure throughput under sustained load
- **Benchmark:** `batch_100_predictions`

### 6. Cache Effectiveness (`4_cache_effectiveness`)
- **Scenario 1 - Cache Hits:** 50 predictions with same input signal
- **Scenario 2 - Cache Misses:** 50 predictions with different input signals
- **Metric:** Hit rate, speedup factor, latency comparison
- **Configuration:** 50 samples each, 15-second measurement
- **Expected Hit Speedup:** 1.2-2.0x faster with caching
- **Benchmarks:** `all_cache_hits`, `all_cache_misses`

### 7. Signal Type Sensitivity (`5_signal_type_sensitivity`)
- **Signal Types Tested:**
  - Sine wave (stationary, periodic)
  - Chirp (non-stationary, frequency sweep)
  - White noise (random, broadband)
  - Composite (multiple frequency components)
- **Metric:** Latency per signal type
- **Configuration:** 200-sample signals, 10-second measurement
- **Purpose:** Identify if certain signal characteristics affect performance
- **Benchmarks:** `sine`, `chirp`, `noise`, `composite`

### 8. Throughput Analysis (`6_throughput`)
- **Metric:** Predictions per second over 100ms window
- **Configuration:** Real-time measurement, 20 samples
- **Expected Throughput:** 5000-7000+ predictions/sec
- **Purpose:** Establish practical throughput capacity
- **Benchmark:** `throughput_predictions_per_sec`

### 9. Warmup vs Steady-state (`7_warmup_analysis`)
- **Scenario 1:** First prediction after model load
- **Scenario 2:** After 10 warmup iterations
- **Metric:** Latency comparison between cold and warm states
- **Configuration:** 50 samples each, measurement on 11th prediction
- **Expected Difference:** 1-5% performance gain after warmup
- **Benchmarks:** `warmup_first_prediction`, `steady_state_prediction`

### 10. Prediction Horizon Impact (`8_prediction_horizon`)
- **Horizons Tested:** 1, 3, 5, 10 samples
- **Metric:** Latency vs prediction horizon length
- **Configuration:** 100-sample signal, 10-second measurement
- **Purpose:** Understand scaling with prediction window
- **Benchmarks:** Multiple iterations for each horizon

## Output Format

Criterion produces detailed statistical analysis for each benchmark:

```
Benchmarking 1_model_loading/load_default_model
Benchmarking 1_model_loading/load_default_model: Warming up for 3.0000 s
Benchmarking 1_model_loading/load_default_model: Collecting 10 samples in estimated 5.0061 s
Benchmarking 1_model_loading/load_default_model: Analyzing

1_model_loading/load_default_model
                        time:   [252.23 µs 296.42 µs 348.21 µs]
                slope  [252.23 µs 348.21 µs] R^2            [0.7269361 0.7018469]
                mean   [256.59 µs 305.21 µs] std. dev.      [9.3969 µs 66.955 µs]
                median [250.47 µs 278.49 µs] med. abs. dev. [3.8213 µs 32.095 µs]
```

Statistics provided:
- **time:** Lower bound, point estimate, upper bound
- **slope:** Linear regression slope (improvement/deterioration)
- **R²:** Coefficient of determination (confidence in measurements)
- **mean:** Average measurement
- **std. dev.:** Standard deviation
- **median:** Middle value
- **med. abs. dev.:** Median absolute deviation

## Performance Baseline Results

From initial profiling run:

| Benchmark | Result | Notes |
|-----------|--------|-------|
| Model Loading | ~256-305 µs | Includes SafeTensors deserialization |
| Small Signal (20) | ~155 µs | Includes fit + predict |
| Medium Signal (100) | ~152 µs | Typical input size |
| Large Signal (1000) | ~140 µs | Input size has minimal impact |
| Batch (100x) | ~170 µs/iter | Per-prediction in batch context |
| Throughput | ~5,900 pred/sec | Maximum sustainable rate |
| Cache Hit Speedup | ~1.2-1.5x | Depends on cache configuration |

## Interpretation Guide

### Normal Performance Expectations

- **Model Loading:** 250-350 µs (one-time cost)
- **Prediction Latency:** 140-160 µs per prediction
- **P99 Latency:** < 200 µs (at scale)
- **Throughput:** 5,000-7,000 predictions/second
- **Cache Effectiveness:** 10-50% hit rate in practice

### Performance Regression Indicators

- **Latency increase > 10%:** May indicate regression in LSTM computation
- **Batch throughput drop > 15%:** May indicate cache pollution or allocation pressure
- **High variance (std dev > mean):** May indicate interference from system load
- **Warmup improvement > 10%:** May indicate missing optimization or cache effects

## Optimization Opportunities

### Identified in Current Implementation

1. **Cache Layer Optimization**
   - Current: Simple HashMap cache
   - Potential: LRU cache with size limits to prevent unbounded growth
   - Benefit: Improve hit rate from 45% to 70%+

2. **LSTM Computation Vectorization**
   - Current: Pure Rust matrix multiplication
   - Potential: SIMD-optimized kernels (packed_simd when stable)
   - Benefit: 2-4x speedup on matrix operations

3. **Model Quantization**
   - Current: FP32 matrices loaded from SafeTensors
   - Potential: Quantize to INT8 or mixed-precision
   - Benefit: Reduce memory, improve cache locality

4. **Prediction Batching**
   - Current: Single prediction per call
   - Potential: Batch multiple predictions in one LSTM forward pass
   - Benefit: Amortize attention computation overhead

5. **Memory Layout**
   - Current: Vec<Vec<f64>> (row-major)
   - Potential: Contiguous allocator with block layout
   - Benefit: Improve cache line utilization

### Quick Wins (Low Effort, Medium Benefit)

- Use slice references instead of cloning signals where possible
- Pre-allocate prediction buffer outside hot path
- Profile with `perf` to identify cache misses

## Integration with CI/CD

### GitHub Actions Integration

```yaml
- name: Run LSTM Performance Benchmark
  run: cargo bench --bench lstm_performance_profile --features boundary-prediction
  
- name: Compare with Baseline
  run: |
    # Compare latest results with previous baseline
    # Fail if regression > 10%
```

### Performance Budget Tracking

Create `perf-baseline.json`:
```json
{
  "model_loading": {"value": 256, "unit": "µs", "tolerance": 350},
  "single_prediction": {"value": 150, "unit": "µs", "tolerance": 180},
  "throughput": {"value": 6000, "unit": "pred/sec", "tolerance": 5000}
}
```

## Advanced Usage

### Profile Specific Benchmark Only

```bash
cargo bench --bench lstm_performance_profile --features boundary-prediction -- \
  --bench 3_batch_processing
```

### Set Custom Sample Size

```bash
cargo bench --bench lstm_performance_profile --features boundary-prediction -- \
  --sample-size 200
```

### Generate Plots (Requires gnuplot)

```bash
cargo bench --bench lstm_performance_profile --features boundary-prediction -- \
  --plotting-backend gnuplot
```

### Save Results for Later Comparison

```bash
cargo bench --bench lstm_performance_profile --features boundary-prediction -- \
  --save-baseline lstm_v1.0
```

### Compare Against Baseline

```bash
cargo bench --bench lstm_performance_profile --features boundary-prediction -- \
  --baseline lstm_v1.0
```

## Manual Profiling Helpers

The benchmark includes a `ProfilingStats` struct for custom analysis:

- `add_measurement(duration_ns)` - Add a timing measurement
- `mean_us()` - Calculate mean in microseconds
- `stddev_us()` - Calculate standard deviation
- `min_us()` / `max_us()` - Get min/max values
- `percentile_us(p)` - Get approximate percentile

This can be used to extend the benchmark with custom measurements.

## Dependencies

- **Criterion:** Statistical benchmarking framework
- **ferromode:** LSTM model with SafeTensors support
- **rand:** Random signal generation for cache tests

## Files Modified

1. `crates/ferromode/benches/lstm_performance_profile.rs` - New benchmark file (599 lines)
2. `crates/ferromode/Cargo.toml` - Added benchmark configuration

## Running on Different Platforms

Performance will vary by hardware. Ensure consistent testing:

```bash
# Linux
cargo bench --bench lstm_performance_profile --features boundary-prediction

# macOS
# May see 20-30% variation due to CPU frequency scaling
cargo bench --bench lstm_performance_profile --features boundary-prediction

# Windows
# Run with administrator privileges for accurate timer resolution
cargo bench --bench lstm_performance_profile --features boundary-prediction
```

## Next Steps

1. **Establish Baseline:** Run benchmark on target hardware, save results
2. **Set Performance Budget:** Define acceptable latency/throughput values
3. **Implement Optimizations:** Use results to prioritize optimization work
4. **Track Over Time:** Integrate into CI/CD, monitor for regressions
5. **Profile with Flame Graphs:** Use `cargo flamegraph` to identify hot paths

## References

- [Criterion.rs Documentation](https://docs.rs/criterion/latest/criterion/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- LSTM Implementation: `crates/ferromode/src/adapters/boundary_prediction/lstm.rs`
- AR Model for Comparison: `crates/ferromode/src/boundary/ar_model.rs`
