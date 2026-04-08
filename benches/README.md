# Ferromode V2.2 Release Benchmark Suite

Comprehensive performance benchmarking and reporting tool for Ferromode V2.2.

## Quick Start

Run all benchmarks in release mode with comprehensive reporting:

```bash
./benches/run_release_benchmarks.sh
```

## What It Does

### 1. **Cleanup Phase**
- Removes old benchmark results
- Clears criterion cache for fresh measurements
- Creates output directories

### 2. **Build Phase**
- Builds all benchmarks in release mode
- Enables `boundary-prediction` feature
- Uses optimization flags: `-C opt-level=3 -C lto=thin`
- Verifies build succeeds

### 3. **Benchmark Execution**
- Runs `lstm_vs_ar_benchmark`
- Runs `lstm_performance_profile`
- Captures output to log files
- Displays real-time progress

### 4. **Results Collection**
- Extracts key metrics from logs
- Parses performance measurements
- Organizes results for reporting

### 5. **Report Generation**
- Creates interactive HTML report
- Generates markdown summary
- Prints console summary
- Verifies all targets met

## Performance Targets

All of these targets must pass:

| Metric | Target | Status |
|--------|--------|--------|
| LSTM Latency | < 1.0 ms | ✅ Required |
| Throughput | > 5,000 pred/sec | ✅ Required |
| Model Size | < 2.0 MB | ✅ Required |
| End-Effect Reduction | > 30% | ✅ Required |

## Output Files

After running, check these locations:

### Reports
- **Interactive HTML Report**: `target/benchmark_results/performance_report.html`
- **Markdown Summary**: `target/benchmark_results/v22_benchmark_summary.md`

### Logs
- **Build Log**: `target/benchmark_results/logs/build.log`
- **LSTM vs AR Log**: `target/benchmark_results/logs/lstm_vs_ar_benchmark.log`
- **LSTM Profile Log**: `target/benchmark_results/logs/lstm_performance_profile.log`

### Criterion Results
- **Statistical Analysis**: `target/criterion/report/index.html`

## Example Output

```
========================================
V2.2 Release Mode Performance Report
========================================

Build Configuration:
  Mode: Release
  Features: boundary-prediction
  Target: x86_64-unknown-linux-gnu
  Optimization: -C opt-level=3

LSTM Inference Latency:
  Single prediction (20 samples): 0.135 ms ✓ (< 1ms target)
  Single prediction (100 samples): 0.152 ms ✓
  Batch (100 predictions): 0.148 ms avg ✓
  p99 latency: 0.285 ms ✓

AR Inference Latency:
  Single prediction: 0.018 ms
  LSTM/AR ratio: 7.5x

Throughput:
  LSTM: 7,407 predictions/sec ✓
  AR: 55,556 predictions/sec

Model Loading:
  SafeTensors load: 0.52 ms

Memory:
  Model size: 0.802 MB
  Peak RSS: 12.4 MB

End-Effect Reduction:
  Sine wave: 35% improvement
  Chirp: 62% improvement
  AM/FM: 64% improvement
  Average: 54% improvement ✓

Status: ✅ ALL TARGETS MET
```

## Benchmark Details

### lstm_vs_ar_benchmark

Comprehensive comparison of LSTM and AR boundary prediction methods across diverse signal types:

- **Sine wave** - Stationary, periodic signal
- **Chirp** - Non-stationary frequency sweep
- **White noise** - Random broadband signal
- **AM/FM modulated** - Amplitude + frequency modulation
- **Real-world-like** - Composite signal with multiple components

Metrics:
- Reconstruction error in boundary region
- Endpoint smoothness
- IMF stability
- Computational latency

### lstm_performance_profile

Detailed profiling of LSTM inference performance:

- Latency distribution (min, p50, p95, p99, max)
- Throughput under various batch sizes
- Model initialization overhead
- Memory footprint during inference

## Requirements

- Rust 1.75+
- `cargo` with release profile
- SafeTensors support (boundary-prediction feature)
- Linux/Unix environment (uses bash)

## Troubleshooting

### Build fails
Check `target/benchmark_results/logs/build.log` for error details.

### Benchmarks timeout
Some benchmarks take several minutes. On slow systems, may need to adjust timeout.

### Missing output files
Ensure script has write permissions to `target/` directory.

## Performance Interpretation

### Latency
- **< 0.2 ms** - Excellent, well below target
- **0.2-0.5 ms** - Good, still far below target
- **0.5-1.0 ms** - Acceptable, at boundary
- **> 1.0 ms** - Failed, exceeds target

### Throughput
- **> 10,000 pred/sec** - Excellent
- **> 5,000 pred/sec** - Acceptable (target)
- **< 5,000 pred/sec** - Failed

### End-Effect Reduction
- **> 50%** - Excellent
- **30-50%** - Acceptable (target)
- **< 30%** - Failed

### Model Size
- **< 1.0 MB** - Excellent (target easily met)
- **1.0-2.0 MB** - Acceptable (target)
- **> 2.0 MB** - Failed

## Production Readiness

✅ Ferromode V2.2 is production-ready if:
- All targets pass
- No memory leaks or crashes
- Consistent performance across runs
- Suitable for your latency/throughput requirements

## Next Steps

1. Review the HTML report: `target/benchmark_results/performance_report.html`
2. Check detailed logs if any issues
3. Run in staging environment
4. Monitor production metrics
5. Consider GPU acceleration for higher throughput if needed

---

Generated by Ferromode V2.2 Release Benchmark Suite
