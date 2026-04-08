# Streaming Decomposition User Guide

## What is Streaming Decomposition?

Streaming decomposition enables real-time analysis of continuous signals without loading entire datasets into memory. Instead of decomposing a complete signal at once, the streaming adapter processes fixed-size chunks sequentially while maintaining state across boundaries.

### Key Benefits

- **Bounded Memory**: Process gigabytes of data with constant memory footprint (< 100 MB)
- **Low Latency**: Decompose 1000 samples in < 10ms p99
- **Continuous Operation**: Process endless data streams 24/7
- **Stateful Processing**: Maintain context across chunks for improved accuracy
- **Adaptive Algorithms**: Automatically select optimal algorithm (EMD, EEMD, CEEMDAN) based on signal properties

## When to Use Streaming vs Batch

### Use Streaming When:
- Processing continuous data (sensors, network traffic, audio streams)
- Signal is too large to fit in memory
- Real-time analysis required (low latency critical)
- Processing unending data streams
- Running on embedded/resource-constrained devices

### Use Batch When:
- Complete signal available upfront
- Non-time-critical analysis
- Need maximum accuracy (no boundary effects)
- Signal fits comfortably in memory
- One-off analysis tasks

## Core Concepts

### Chunks
A chunk is a fixed-size segment of samples processed at once. Typical chunk sizes: 256-2048 samples.

```
Original Signal: [0    100    200    300    400    500    600    700    800]
Chunks (size=256): [0-255] [256-511] [512-767] ...
```

### State
The streaming decomposer maintains state across chunks:
- **Ring Buffer**: Recent samples for context
- **History**: Previous envelopes for continuity
- **Predictor**: AR model for boundary prediction
- **Metrics**: Signal stationarity and complexity

### Boundary Prediction
At chunk boundaries, an AR predictor extends the signal to reduce end effects. Without prediction:

```
Signal: [.... actual data ....]
Result: [end-effects] [accurate IMFs] [end-effects]
```

With prediction:

```
Signal: [pred] [.... actual data ....] [pred]
Result: [better IMFs] [better IMFs] [better IMFs]
```

Boundary prediction typically reduces end-effect energy > 30%.

## Performance Characteristics

### Latency
- **Single chunk (1000 samples)**: 5-15ms typical
- **p99 latency**: < 10ms
- **Breakdown**: 80% sifting, 10% boundary prediction, 10% housekeeping

### Memory
- **Base overhead**: ~20 MB (ring buffer, state)
- **Per chunk**: O(chunk_size) temporary allocations
- **Max for 1-min window**: < 100 MB at 1000 Hz
- **Reclamation**: Automatic via ring buffer rotation

### Algorithm Selection
The streaming decomposer automatically selects algorithms based on signal stationarity:

| Stationarity Score | Algorithm | Reason |
|---|---|---|
| > 0.8 | EMD | Stationary, fewer IMFs needed |
| 0.5 - 0.8 | EEMD | Moderate non-stationarity |
| < 0.5 | CEEMDAN | High noise/non-stationarity |

## Configuration

### Creating a Decomposer

**Rust:**
```rust
use ferromode::adapters::streaming::{ArModel, StreamingDecomposer};
use ferromode::algorithms::emd::EmdConfig;

let mut config = EmdConfig::default();
config.max_imfs = 8;
config.boundary_condition = BoundaryConditionType::MirrorEven;

let predictor = Box::new(ArModel::new(3)?);
let mut decomposer = StreamingDecomposer::new(config, predictor, 4096)?;
```

**Python:**
```python
import ferromode_py

decomposer = ferromode_py.StreamingDecomposer(
    max_imfs=8,
    chunk_size=1024,
    buffer_size=4096,
    boundary_condition="mirror"  # or "periodic", "mirror_odd", "slope", etc.
)
```

### Parameters

- **max_imfs** (default 8): Maximum IMFs to extract (1-16). Higher = more detail, more computation
- **chunk_size** (default 1024): Samples per chunk. Larger = better quality, higher latency
- **buffer_size** (default 4096): Ring buffer capacity. Should be 2-4x chunk_size
- **boundary_condition** (default "mirror"): How to extend signal at chunk boundaries
  - "mirror": Mirror even around endpoints (recommended)
  - "mirror_odd": Mirror odd around endpoints
  - "periodic": Assume signal repeats
  - "slope": Linear extrapolation
  - "ar_model": AR model prediction
  - "wave_matching": Match waveform phase

## Tuning Guide

### For Real-Time Applications (Low Latency)

```rust
// Optimize for speed
let mut config = EmdConfig::default();
config.max_imfs = 3;  // Fewer IMFs = faster

let decomposer = StreamingDecomposer::new(
    config,
    Box::new(ArModel::new(2)?),  // Lower AR order
    2048  // Smaller buffer
)?;
```

**Expected latency**: 2-5ms per chunk

### For High-Quality Analysis (Low Noise)

```rust
// Optimize for accuracy
let mut config = EmdConfig::default();
config.max_imfs = 12;  // More IMFs = better detail

let decomposer = StreamingDecomposer::new(
    config,
    Box::new(ArModel::new(5)?),  // Higher AR order
    8192  // Larger buffer for context
)?;
```

**Expected latency**: 15-30ms per chunk

### For Memory-Constrained Devices

```rust
// Minimize memory footprint
let decomposer = StreamingDecomposer::new(
    EmdConfig::default(),
    Box::new(ArModel::new(2)?),
    1024  // Minimal buffer
)?;
```

**Expected memory**: 30-50 MB

## Example: Audio Stream Decomposition

Process real-time audio from a microphone:

```rust
use ferromode::adapters::streaming::StreamingDecomposer;
use ferromode::algorithms::emd::EmdConfig;
use ferromode::types::Signal;

fn process_audio_stream(audio_source: AudioSource) -> Result<()> {
    let mut decomposer = StreamingDecomposer::new(
        EmdConfig::default(),
        Box::new(ArModel::new(3)?),
        4096
    )?;

    for audio_chunk in audio_source.iter_chunks(1024) {
        // Convert audio samples to Signal
        let signal = Signal::from_slice(&audio_chunk)?;
        
        // Decompose
        let result = decomposer.decompose_chunk(&signal)?;
        
        // Check stationarity
        let entropy = result.metrics.spectral_entropy;
        
        if entropy > 0.7 {
            println!("High noise detected: {}", entropy);
        }
        
        // Process IMFs (e.g., feature extraction, filtering)
        for (i, imf) in result.imfs.iter().enumerate() {
            println!("IMF {}: {} samples", i, imf.len());
        }
    }
    
    Ok(())
}
```

## Troubleshooting

### Problem: High End Effects Near Boundaries

**Cause**: Boundary prediction not effective for signal type

**Solution**:
```rust
// Try stronger AR model
let predictor = Box::new(ArModel::new(5)?);  // Increase order

// Or switch boundary condition
config.boundary_condition = BoundaryConditionType::WaveformMatching;
```

### Problem: High Memory Usage

**Cause**: Buffer size too large or many IMFs being stored

**Solution**:
```rust
// Reduce buffer size
let decomposer = StreamingDecomposer::new(
    config,
    predictor,
    2048  // Was 8192
)?;

// Reduce max_imfs
config.max_imfs = 4;  // Was 8
```

### Problem: Inconsistent Results Across Chunks

**Cause**: State not being updated properly or signal discontinuity

**Solution**:
```rust
// Ensure no gaps between chunks
assert_eq!(chunk1.len() + chunk2.len(), expected_total);

// Check for signal discontinuities
if signal.has_large_jump() {
    decomposer.reset();  // Reset state if signal restarts
}
```

### Problem: Algorithm Not Adapting

**Cause**: Stationarity detection threshold mismatch

**Solution**:
Check metric values:
```rust
let result = decomposer.decompose_chunk(&signal)?;
println!("Entropy: {}", result.metrics.spectral_entropy);
println!("Stationarity: {}", result.metrics.stationarity_score);
println!("Extrema CV: {}", result.metrics.extrema_spacing_cv);
```

## Performance Benchmarks

Measured on Intel Core i7-8700K (reference platform):

### Latency (per chunk)

| Chunk Size | Mean | p50 | p99 |
|---|---|---|---|
| 256 samples | 1.2ms | 1.1ms | 2.3ms |
| 512 samples | 2.8ms | 2.6ms | 4.5ms |
| 1024 samples | 6.2ms | 5.9ms | 9.8ms |
| 2048 samples | 14.5ms | 14.1ms | 19.2ms |

### Memory

| Duration | Chunk Size | Peak RSS |
|---|---|---|
| 1 minute | 256 | 28 MB |
| 1 minute | 512 | 35 MB |
| 1 minute | 1024 | 45 MB |
| 1 minute | 2048 | 65 MB |

## API Reference

### Rust: StreamingDecomposer

```rust
pub struct StreamingDecomposer {
    pub fn new(
        base_config: EmdConfig,
        predictor: Box<dyn PredictorState>,
        buffer_size: usize,
    ) -> Result<Self, EmdError>
    
    pub fn decompose_chunk(&mut self, chunk: &Signal) -> Result<ChunkResult, EmdError>
    
    pub fn reset(&mut self)
    
    pub fn chunk_id(&self) -> u64
}

pub struct ChunkResult {
    pub imfs: Vec<Vec<f64>>,        // IMFs extracted from chunk
    pub remainder: Vec<f64>,         // Residual component
    pub metrics: IntermittencyMetrics,
}

pub struct IntermittencyMetrics {
    pub spectral_entropy: f64,       // 0..1, lower = stationary
    pub extrema_spacing_cv: f64,     // Coefficient of variation
    pub stationarity_score: f64,     // 0..1, > 0.8 = stationary
}
```

### Python: StreamingDecomposer

```python
class StreamingDecomposer:
    def __init__(
        self,
        max_imfs: int = 8,
        chunk_size: int = 1024,
        buffer_size: int = 4096,
        boundary_condition: str = "mirror"
    ) -> None
    
    def decompose_chunk(self, chunk: np.ndarray) -> dict:
        # Returns: {
        #     'imfs': np.ndarray,           # (n_imfs, len(chunk))
        #     'residue': np.ndarray,        # (len(chunk),)
        #     'metrics': {
        #         'spectral_entropy': float,
        #         'stationarity_score': float,
        #         'extrema_spacing_cv': float,
        #     }
        # }
    
    def reset(self) -> None
```

## Further Reading

- **Boundary Prediction**: See `docs/STREAMING_V2_STATUS.md` for technical details
- **Intermittency Detection**: Algorithm selection rules in streaming decomposer source
- **Performance Tuning**: Benchmark results in `crates/ferromode/benches/streaming.rs`
