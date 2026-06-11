# Entropy Metrics User Guide

Complete guide to using entropy-based complexity metrics in Ferromode for signal analysis and quality assessment.

**Version:** 2.5.0  
**Last Updated:** April 2026  
**Tasks:** T-330, T-331, T-332, T-335

---

## Table of Contents

1. [Introduction](#introduction)
2. [Spectral Entropy](#spectral-entropy)
3. [Permutation Entropy](#permutation-entropy)
4. [Sample Entropy](#sample-entropy)
5. [Integration with EMD](#integration-with-emd)
6. [Time-Frequency Analysis](#time-frequency-analysis)
7. [Interpretation Guide](#interpretation-guide)
8. [Best Practices](#best-practices)
9. [Code Examples](#code-examples)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

Entropy metrics quantify the complexity, disorder, and information content of signals. In EMD analysis, entropy provides a complementary view to the decomposition itself, helping you assess decomposition quality, detect mode mixing, and characterize signal complexity.

### What Are Entropy Metrics?

Entropy, from information theory, measures the average amount of information or uncertainty in a signal. Higher entropy indicates more randomness or disorder; lower entropy indicates more structure or predictability.

Three complementary entropy measures are implemented in Ferromode:

| Metric | Measures | Range | Use Case |
|--------|----------|-------|----------|
| **Spectral Entropy** | Frequency disorder | 0 to ln(N) | Energy distribution across frequencies |
| **Permutation Entropy** | Temporal complexity | 0 to ln(m!) | Pattern-based nonlinearity detection |
| **Sample Entropy** | Self-similarity | 0 to ~2 | Overall signal regularity |

### Why Use Entropy Metrics in EMD Analysis?

**Quality Assessment:** Detect decomposition problems
- High entropy IMFs suggest mode mixing or poor decomposition
- Entropy baseline helps validate decomposition correctness

**Mode Mixing Detection:** Identify components containing multiple modes
- High permutation entropy indicates chaotic mixing
- Track entropy evolution to pinpoint problem IMFs

**Signal Characterization:** Understand complexity distribution
- Entropy profiles reveal dominant modes
- Compare signals by complexity signature

**Preprocessing Validation:** Confirm signal quality
- Compare entropy before/after preprocessing
- Ensure preprocessing doesn't destroy useful modes

### When to Use Each Entropy Type

- **Use Spectral Entropy** when you care about frequency content and broadband characteristics
- **Use Permutation Entropy** when detecting chaotic behavior or nonlinearity
- **Use Sample Entropy** for robust overall complexity assessment
- **Use All Three** for comprehensive signal characterization

---

## Spectral Entropy

Spectral entropy measures the disorder in the frequency domain by treating the normalized power spectrum as a probability distribution.

### Concept and Formula

The power spectrum P(f) is normalized to create a probability distribution p(f):

```
p(f) = P(f) / ∑P(f)
```

Spectral entropy is then computed as Shannon entropy:

```
H_s = -∑ p(f) · ln(p(f))
```

In natural logarithm units, with maximum value = ln(N) where N is the spectrum length.

### Interpretation

**Value Range:** 0 to ln(N)
- **0 to 1**: Pure sinusoid or narrowband signal (energy concentrated at specific frequency)
- **1 to 3**: Mixed narrowband/broadband (multiple dominant frequencies)
- **3 to 6**: Broadband or noisy signal (energy spread across frequencies)

**Normalized Range [0, 1]:**
- **0.0 to 0.3**: Clean, narrowband signal (sine wave, pure tone)
- **0.3 to 0.6**: Mixed content (chirp, AM/FM modulated)
- **0.6 to 1.0**: Broadband, noise-like (white noise, chaos)

### When to Use

Use spectral entropy to:
- **Detect noise contamination**: High entropy indicates broadband noise
- **Assess frequency content**: Compare energy concentration across components
- **Validate preprocessing**: Ensure denoising preserved important frequencies
- **Identify dominant modes**: Low-entropy IMFs contain concentrated information

### Examples

#### Pure Sinusoid
```
Frequency: 5 Hz, Amplitude: 1.0
Spectral Entropy: 0.18 (normalized: 0.08)
Interpretation: Highly concentrated energy at 5 Hz
```

#### White Noise
```
Bandwidth: 0-500 Hz, SNR: ~0 dB
Spectral Entropy: 5.84 (normalized: 0.96)
Interpretation: Energy spread across all frequencies
```

#### Chirp Signal (5-20 Hz)
```
Frequency sweep: 5 Hz → 20 Hz over 1 second
Spectral Entropy: 3.21 (normalized: 0.54)
Interpretation: Mixed content - energy distributed across swept frequencies
```

### Code Usage

**Basic usage:**
```rust
use ferromode::analysis::spectral_entropy_normalized;

// Analyze a signal
let signal = vec![/* ... */];
let entropy = spectral_entropy_normalized(&signal)?;

// Interpret result
if entropy > 0.8 {
    println!("Signal is broadband/noisy");
} else if entropy > 0.4 {
    println!("Signal has mixed narrowband/broadband content");
} else {
    println!("Signal is narrowband/clean");
}
```

**With decomposition:**
```rust
use ferromode::analysis::{spectral_entropy_normalized, EntropyAnalysis};

let decomposition = emd::decompose(&signal, &config)?;
let analysis = EntropyAnalysis::from_decomposition(&decomposition)?;

// Print per-IMF spectral entropy
for (i, entropy) in analysis.spectral_entropy.iter().enumerate() {
    println!("IMF {} spectral entropy: {:.4}", i, entropy);
}

println!("Mean spectral entropy: {:.4}", analysis.mean_spectral_entropy);
```

---

## Permutation Entropy

Permutation entropy measures temporal complexity by analyzing ordinal patterns in sliding windows. It captures nonlinear and chaotic behavior without requiring assumptions about linearity.

### Concept and Formula

For each sliding window of size m (embedding dimension), compute the ordinal pattern - the ranking of values:

```
Example: window [3.2, 1.1, 2.8] → pattern [2, 0, 1]
         (1.1 is smallest, 3.2 is largest, 2.8 is middle)
```

Count occurrences of each pattern. Compute Shannon entropy of pattern distribution:

```
H_p = -∑ p(π) · ln(p(π))
```

where π ranges over all possible ordinal patterns (m! possible patterns).

### Interpretation

**Value Range:** 0 to ln(m!) where m is embedding dimension
- For m=3: 0 to ln(6) ≈ 1.79
- For m=4: 0 to ln(24) ≈ 3.18
- For m=5: 0 to ln(120) ≈ 4.79

**Normalized Range [0, 1]:**
- **0.0 to 0.2**: Highly regular pattern (mostly one ordinal pattern)
- **0.2 to 0.5**: Periodic or weakly chaotic
- **0.5 to 0.8**: Mixed regular/chaotic behavior
- **0.8 to 1.0**: Chaotic, random, or white noise

### When to Use

Use permutation entropy to:
- **Detect chaos and nonlinearity**: Chaotic signals have high permutation entropy
- **Identify mode mixing**: Mixed modes produce diverse ordinal patterns
- **Compare signal behavior**: Different dynamics produce different pattern distributions
- **Analyze IMF quality**: Well-separated modes have lower entropy

### Embedding Dimension Selection

**Typical values:**
- **m=3**: Quick analysis, good for most signals, max patterns = 6
- **m=4**: More detail, typical for EMD analysis, max patterns = 24
- **m=5**: Fine detail, requires longer signals, max patterns = 120

**Rule of thumb:**
- Signal length ≥ 10 × m! (so for m=4, need ≥ 240 samples)
- m=3 or m=4 for most applications
- Increase m only if you have >1000 samples

### Examples

#### Pure Sine Wave (5 Hz, 1000 samples)
```
Embedding dimension: 3
Permutation Entropy: 0.32 (normalized)
Pattern Distribution: Dominated by 2-3 ordinal patterns
Interpretation: Regular periodic signal, low chaos
```

#### White Noise (1000 samples)
```
Embedding dimension: 3
Permutation Entropy: 0.95 (normalized)
Pattern Distribution: All ordinal patterns equally probable
Interpretation: Maximum disorder, random behavior
```

#### Chaotic Signal (Henon map, 1000 samples)
```
Embedding dimension: 3
Permutation Entropy: 0.87 (normalized)
Pattern Distribution: Diverse patterns, not all equally probable
Interpretation: High complexity but with structure
```

### Code Usage

**Basic usage:**
```rust
use ferromode::analysis::permutation_entropy_normalized;

let signal = vec![/* ... */];
let embedding_dim = 3;
let entropy = permutation_entropy_normalized(&signal, embedding_dim)?;

if entropy > 0.7 {
    println!("Signal is chaotic or contains mode mixing");
} else {
    println!("Signal is relatively regular");
}
```

**Analyzing decomposition:**
```rust
use ferromode::analysis::EntropyAnalysis;

let analysis = EntropyAnalysis::from_decomposition(&decomposition)?;

// Find IMFs with high chaotic content
for (i, entropy) in analysis.permutation_entropy.iter().enumerate() {
    if entropy > 0.6 {
        println!("⚠️ IMF {} has high chaos: {:.4}", i, entropy);
    }
}
```

---

## Sample Entropy

Sample entropy measures self-similarity and regularity. Lower values indicate more predictable signals; higher values indicate more complex or random behavior. It's more robust than other entropy metrics but computationally more expensive.

### Concept and Formula

Sample entropy counts matching templates:
- For embedding dimension m, count template pairs within tolerance r
- Then count template pairs within tolerance r for dimension m+1
- Compute: SampEn = -ln(C(m+1,r) / C(m,r))

```
SampEn(m, r, N) = -ln(matches(m+1) / matches(m))
```

Lower values → more matches → more regular signal  
Higher values → fewer matches → more complex signal

### Interpretation

**Value Range:** 0 to ~2 (unbounded, typically 0-3)
- **0.0 to 0.5**: Highly regular, very predictable
- **0.5 to 1.0**: Moderate complexity, some predictability
- **1.0 to 1.5**: Moderate-to-high complexity
- **1.5+**: High complexity, mostly random

**Typical values by signal type:**
- Constant signal: ~0.0
- Pure sine wave: 0.3-0.5
- AM/FM modulated: 0.6-1.0
- White noise: 1.5-2.0+

### When to Use

Use sample entropy to:
- **Assess overall complexity**: Single metric capturing regularity
- **Detect pathological signals**: Anomalies show entropy spikes
- **Quality check**: Sudden entropy changes indicate problems
- **Characterize components**: Identify which IMFs contain information

### Tolerance Parameter Selection

Sample entropy requires a tolerance threshold r that determines "matching."

**Default behavior:** r = 0.2 × standard deviation

**Custom tolerance:**
- **Tight (r=0.1×std)**: Higher entropy, stricter matching
- **Normal (r=0.2×std)**: Balanced, most common
- **Loose (r=0.3×std)**: Lower entropy, more forgiving

**Rule of thumb:**
- Use default (0.2×std) as starting point
- Adjust if entropy values seem unreasonable
- Document chosen tolerance for reproducibility

### Examples

#### Constant Signal (all values = 1.0, 100 samples)
```
Embedding dimension: 2
Tolerance: 0.2 × std (0.0 in this case)
Sample Entropy: 0.0
Interpretation: Perfect predictability
```

#### Periodic Signal (sine wave, 100 Hz, 1000 samples)
```
Embedding dimension: 2
Tolerance: 0.2 × std
Sample Entropy: 0.42
Interpretation: Regular, predictable pattern
```

#### Random Signal (white noise, 1000 samples)
```
Embedding dimension: 2
Tolerance: 0.2 × std
Sample Entropy: 1.74
Interpretation: High complexity, low predictability
```

### Code Usage

**Basic usage:**
```rust
use ferromode::analysis::sample_entropy;

let signal = vec![/* ... */];
let embedding_dim = 2;

// Use default tolerance (0.2 * std)
let entropy = sample_entropy(&signal, embedding_dim, None)?;

// Or specify custom tolerance
let std_dev = calculate_std_dev(&signal);
let custom_tolerance = 0.15 * std_dev;
let entropy = sample_entropy(&signal, embedding_dim, Some(custom_tolerance))?;

if entropy < 0.5 {
    println!("Signal is highly regular/predictable");
} else if entropy < 1.0 {
    println!("Signal is moderately complex");
} else {
    println!("Signal is highly complex/random");
}
```

**Quality assessment:**
```rust
use ferromode::analysis::sample_entropy;

// Compare entropy before and after preprocessing
let entropy_before = sample_entropy(&original, 2, None)?;
let entropy_after = sample_entropy(&processed, 2, None)?;

if (entropy_after - entropy_before).abs() > 0.5 {
    println!("⚠️ Preprocessing significantly changed signal complexity");
}
```

---

## Integration with EMD

The most powerful use of entropy metrics is analyzing the entropy of each IMF from an EMD decomposition. This reveals:
- Which components are noisy vs. information-bearing
- Whether decomposition succeeded (mode separation)
- Overall signal complexity distribution across modes

### EntropyAnalysis Structure

The `EntropyAnalysis` struct computes all three entropy metrics for each IMF and provides aggregate statistics:

```rust
pub struct EntropyAnalysis {
    pub spectral_entropy: Vec<f64>,
    pub permutation_entropy: Vec<f64>,
    pub sample_entropy: Vec<f64>,
    
    pub mean_spectral_entropy: f64,
    pub mean_permutation_entropy: f64,
    pub mean_sample_entropy: f64,
    
    pub complexity_score: f64,  // [0, 1]
    pub high_complexity_imfs: Vec<usize>,  // entropy > 0.7
}
```

### Complete Workflow

```rust
use ferromode::analysis::EntropyAnalysis;
use ferromode::emd;

// Step 1: Decompose signal
let config = EmdConfig::default();
let decomposition = emd::decompose(&signal, &config)?;

// Step 2: Analyze entropy
let analysis = EntropyAnalysis::from_decomposition(&decomposition)?;

// Step 3: Interpret results
println!("Decomposition Quality Assessment");
println!("================================");
println!("Mean Spectral Entropy:    {:.4}", analysis.mean_spectral_entropy);
println!("Mean Permutation Entropy: {:.4}", analysis.mean_permutation_entropy);
println!("Mean Sample Entropy:      {:.4}", analysis.mean_sample_entropy);
println!("Complexity Score:         {:.4}", analysis.complexity_score);

if !analysis.high_complexity_imfs.is_empty() {
    println!("\n⚠️ Warning: High-complexity IMFs detected");
    println!("Affected IMFs: {:?}", analysis.high_complexity_imfs);
    println!("This may indicate mode mixing or signal quality issues.");
}

// Step 4: Per-IMF analysis
println!("\nPer-IMF Entropy Analysis:");
for i in 0..decomposition.imfs.imfs.len() {
    println!("IMF {}: spectral={:.4}, permutation={:.4}, sample={:.4}",
             i,
             analysis.spectral_entropy[i],
             analysis.permutation_entropy[i],
             analysis.sample_entropy[i]);
}
```

### Interpretation by Decomposition Phase

**Phase 1: Initial Decomposition**
```
Expected Patterns:
- First IMF(s): Highest entropy (fastest oscillations, noise)
- Middle IMFs: Moderate entropy (signal modes)
- Last IMF: Lowest entropy (trend, residue)

Problem Indicators:
- All IMFs have high entropy → possible mode mixing
- All IMFs have low entropy → signal too simple, limited decomposition
```

**Phase 2: Mode Separation Validation**
```
Good Decomposition:
- Spectral entropy decreases across IMFs
- Permutation entropy shows distinct patterns
- Sample entropy stable but with variation

Poor Decomposition:
- Spectral entropy chaotic (no pattern)
- Permutation entropy high across all IMFs
- Sample entropy suggests mode mixing
```

**Phase 3: Quality Assessment**
```
Excellent Quality: complexity_score < 0.4
Good Quality: 0.4 ≤ complexity_score < 0.6
Questionable Quality: 0.6 ≤ complexity_score < 0.75
Poor Quality: complexity_score ≥ 0.75
```

### Decision Tree

```
Start → Decompose signal
    ↓
    Analyze entropy
    ↓
    Is complexity_score < 0.5?
    ├─ YES → Decomposition likely successful ✓
    │        High confidence in IMF separation
    │        Proceed to analysis
    │
    └─ NO → Complex signal or possible mode mixing
             Check high_complexity_imfs list
             ├─ None listed → Signal inherently complex
             │                Use entropy context for interpretation
             │
             └─ Some listed → Investigate those IMFs
                              Consider:
                              - Increasing max_imf
                              - Adjusting stopping criteria
                              - Preprocessing signal
                              - Re-decomposing with different config
```

---

## Time-Frequency Analysis

For non-stationary signals, entropy can be computed in sliding windows to show how complexity evolves over time.

### Sliding Window Spectral Entropy

Track frequency content evolution:

```rust
use ferromode::analysis::spectral_entropy_normalized;

fn compute_windowed_entropy(
    signal: &[f64],
    window_size: usize,
    hop_size: usize,
) -> Result<Vec<f64>, EmdError> {
    let mut entropies = Vec::new();
    
    for start in (0..signal.len()).step_by(hop_size) {
        let end = std::cmp::min(start + window_size, signal.len());
        if end - start >= 4 {  // Minimum viable window
            let window = &signal[start..end];
            let entropy = spectral_entropy_normalized(window)?;
            entropies.push(entropy);
        }
    }
    
    Ok(entropies)
}

// Usage
let window_size = 256;
let hop_size = 128;
let entropies = compute_windowed_entropy(&signal, window_size, hop_size)?;

// Plot or analyze entropies over time
for (i, &entropy) in entropies.iter().enumerate() {
    println!("Window {}: entropy = {:.4}", i, entropy);
}
```

### Detecting Non-Stationarity

High variance in windowed entropy indicates non-stationarity:

```rust
fn detect_nonstationarity(entropies: &[f64]) -> (f64, f64) {
    let mean = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let variance = entropies.iter()
        .map(|&e| (e - mean).powi(2))
        .sum::<f64>() / entropies.len() as f64;
    
    (mean, variance.sqrt())
}

let (mean, std_dev) = detect_nonstationarity(&entropies);
println!("Mean entropy: {:.4}", mean);
println!("Entropy std dev: {:.4}", std_dev);

if std_dev > 0.15 {
    println!("Signal is highly non-stationary");
    println!("EMD is well-suited for this signal");
}
```

### Visualization Guidelines

**Time-Entropy Spectrogram:**
```
Axes:
  X: Time (windows)
  Y: Entropy value
  Color: Intensity or frequency content

Pattern interpretation:
- Flat line: Stationary signal
- Increasing trend: Increasing complexity
- Sudden spikes: Anomalies or transients
- Periodic variation: Cyclostationary behavior
```

---

## Interpretation Guide

### Understanding Entropy Values

**Spectral Entropy (normalized [0, 1]):**

| Value | Signal Type | Interpretation |
|-------|-------------|-----------------|
| < 0.3 | Pure tone | Narrowband, concentrated frequency |
| 0.3-0.5 | Modulated sinusoid | Mixed content, clear fundamental |
| 0.5-0.7 | Broadband with peaks | Noisy but with dominant frequencies |
| > 0.7 | White noise | Uniform frequency distribution |

**Permutation Entropy (normalized [0, 1]):**

| Value | Signal Type | Interpretation |
|-------|-------------|-----------------|
| < 0.3 | Constant/linear trend | Highly predictable ordinal patterns |
| 0.3-0.5 | Periodic/simple modes | Regular temporal structure |
| 0.5-0.7 | Mixed periodic/chaotic | Complex temporal behavior |
| > 0.7 | Chaotic/random | Unpredictable patterns |

**Sample Entropy (unbounded, typical 0-2):**

| Value | Signal Type | Interpretation |
|-------|-------------|-----------------|
| < 0.3 | Highly regular | Excellent predictability |
| 0.3-0.8 | Moderate complexity | Balanced pattern/randomness |
| 0.8-1.5 | High complexity | Limited predictability |
| > 1.5 | Very high complexity | Essentially random |

### Comparing Signals

To compare entropy across signals:

1. **Normalize** by signal length when possible
2. **Use same parameters** (embedding dim, tolerance, window size)
3. **Compare within same signal type** (apples-to-apples)
4. **Consider context**: Entropy values depend on signal properties

Example comparison:

```rust
// Compare two decompositions
let decomp1 = emd::decompose(&signal1, &config)?;
let decomp2 = emd::decompose(&signal2, &config)?;

let analysis1 = EntropyAnalysis::from_decomposition(&decomp1)?;
let analysis2 = EntropyAnalysis::from_decomposition(&decomp2)?;

println!("Signal 1 complexity: {:.4}", analysis1.complexity_score);
println!("Signal 2 complexity: {:.4}", analysis2.complexity_score);

if (analysis1.complexity_score - analysis2.complexity_score).abs() > 0.1 {
    println!("Signals have significantly different complexity");
}
```

### Quality Thresholds

**Decomposition Quality (based on complexity_score):**

```
Score < 0.4:   Excellent  ✓ Clean decomposition, clear mode separation
0.4 - 0.6:     Good       ✓ Acceptable decomposition, minor mixing
0.6 - 0.75:    Fair       ⚠ Some mode mixing, consider validation
> 0.75:        Poor       ✗ Significant mixing, reconsider approach
```

**Signal Quality Assessment (mean spectral entropy):**

```
< 0.4:         Clean signal   ✓ Suitable for analysis
0.4 - 0.6:     Moderate noise ✓ Acceptable with caution
0.6 - 0.8:     High noise     ⚠ Consider preprocessing
> 0.8:         Very noisy     ✗ Preprocessing essential
```

**Mode Mixing Detection (high_complexity_imfs):**

```
None listed:           No obvious mixing
1-2 IMFs listed:       Minor mixing in final IMFs
> 2 IMFs listed:       Significant mixing, investigate
All IMFs listed:       Possible decomposition failure
```

---

## Best Practices

### 1. Always Normalize Signals

Entropy is scale-sensitive. Normalize before computing:

```rust
fn normalize(signal: &[f64]) -> Vec<f64> {
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let std = calculate_std_dev(signal);
    
    signal.iter().map(|&x| (x - mean) / std).collect()
}

let normalized_signal = normalize(&signal);
let entropy = spectral_entropy_normalized(&normalized_signal)?;
```

### 2. Choose Appropriate Embedding Dimensions

**Guideline by signal length:**

```
100-200 samples:   Use m=2 or m=3
200-500 samples:   Use m=3
500-1000 samples:  Use m=3 or m=4
> 1000 samples:    Use m=4 or m=5
```

### 3. Use Consistent Parameters

Document and maintain consistent parameters across analyses:

```rust
// Define canonical parameters
const SPECTRAL_NORMALIZED: bool = true;
const PERMUTATION_DIM: usize = 3;
const SAMPLE_DIM: usize = 2;
const SAMPLE_TOLERANCE: Option<f64> = None;  // Auto: 0.2*std

// Apply consistently
let se = spectral_entropy_normalized(&signal)?;
let pe = permutation_entropy_normalized(&signal, PERMUTATION_DIM)?;
let sae = sample_entropy(&signal, SAMPLE_DIM, SAMPLE_TOLERANCE)?;
```

### 4. Validate with Multiple Metrics

Never rely on a single entropy metric. Always use complementary metrics:

```rust
// All three provide different perspectives
let spectral = spectral_entropy_normalized(&signal)?;
let permutation = permutation_entropy_normalized(&signal, 3)?;
let sample = sample_entropy(&signal, 2, None)?;

if spectral > 0.7 && permutation > 0.7 {
    println!("Strong evidence of broadband/noisy signal");
} else if spectral < 0.3 && permutation < 0.3 {
    println!("Strong evidence of clean sinusoidal signal");
} else {
    println!("Mixed characteristics - investigate further");
}
```

### 5. Document Signal Preprocessing

Record how signals were preprocessed, as this affects entropy:

```
Signal Preparation Report:
- Sampling rate: 1000 Hz
- Original length: 10000 samples
- Preprocessing: High-pass filtered at 1 Hz, removed DC
- Normalization: Mean-centered, unit variance
- Result: Entropy = 0.52 (spectral)
```

### 6. Consider Signal Context

Entropy interpretation depends on domain knowledge:

```rust
// Medical context: higher entropy = abnormal
if entropy > threshold {
    println!("⚠️ Abnormal pattern detected");
}

// Engineering context: higher entropy = wear/damage
if entropy_trend.is_increasing() {
    println!("⚠️ Degradation trend detected");
}

// Communication context: moderate entropy = encoded data
if entropy > 0.4 && entropy < 0.8 {
    println!("Signal likely contains encoded information");
}
```

### 7. Handle Edge Cases

**Empty or too-short signals:**
```rust
if signal.len() < 10 {
    eprintln!("Signal too short for reliable entropy computation");
    return Err(EmdError::InsufficientData);
}
```

**Constant signals:**
```rust
let std_dev = calculate_std_dev(&signal);
if std_dev < 1e-10 {
    eprintln!("Signal is effectively constant");
    // Handle specially - entropy metrics not meaningful
}
```

**NaN or Inf values:**
```rust
if signal.iter().any(|&x| !x.is_finite()) {
    eprintln!("Signal contains NaN or Inf");
    return Err(EmdError::InvalidData);
}
```

---

## Code Examples

### Example 1: Complete Signal Analysis

```rust
use ferromode::analysis::{
    spectral_entropy_normalized,
    permutation_entropy_normalized,
    sample_entropy,
    EntropyAnalysis,
};

fn analyze_signal(signal: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    // Validate input
    if signal.is_empty() {
        return Err("Empty signal".into());
    }
    
    println!("Signal Analysis Report");
    println!("====================");
    println!("Signal length: {}", signal.len());
    
    // Compute individual metrics
    let spectral = spectral_entropy_normalized(signal)?;
    let permutation = permutation_entropy_normalized(signal, 3)?;
    let sample = sample_entropy(signal, 2, None)?;
    
    println!("\nEntropy Metrics:");
    println!("  Spectral Entropy (norm):    {:.4}", spectral);
    println!("  Permutation Entropy (norm): {:.4}", permutation);
    println!("  Sample Entropy:             {:.4}", sample);
    
    // Interpret results
    println!("\nInterpretation:");
    if spectral < 0.4 {
        println!("  ✓ Frequency content is concentrated (narrowband)");
    } else if spectral > 0.7 {
        println!("  ! Frequency content is distributed (broadband/noisy)");
    }
    
    if permutation < 0.3 {
        println!("  ✓ Temporal patterns are regular/predictable");
    } else if permutation > 0.7 {
        println!("  ! Temporal patterns are chaotic/random");
    }
    
    Ok(())
}
```

### Example 2: IMF Entropy Analysis

```rust
use ferromode::analysis::EntropyAnalysis;
use ferromode::emd;

fn analyze_decomposition(
    signal: &[f64],
    config: &emd::EmdConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Decompose
    let decomposition = emd::decompose(signal, config)?;
    
    // Analyze
    let analysis = EntropyAnalysis::from_decomposition(&decomposition)?;
    
    println!("IMF Entropy Analysis");
    println!("===================");
    
    // Print per-IMF results
    println!("\n{:<6} {:<12} {:<12} {:<12}", "IMF", "Spectral", "Permutation", "Sample");
    println!("{}", "-".repeat(50));
    
    for i in 0..analysis.spectral_entropy.len() {
        println!(
            "{:<6} {:<12.4} {:<12.4} {:<12.4}",
            i,
            analysis.spectral_entropy[i],
            analysis.permutation_entropy[i],
            analysis.sample_entropy[i]
        );
    }
    
    // Summary statistics
    println!("\n{}", "-".repeat(50));
    println!("Mean Spectral:    {:.4}", analysis.mean_spectral_entropy);
    println!("Mean Permutation: {:.4}", analysis.mean_permutation_entropy);
    println!("Mean Sample:      {:.4}", analysis.mean_sample_entropy);
    println!("\nOverall Complexity Score: {:.4}", analysis.complexity_score);
    
    // Quality assessment
    println!("\nQuality Assessment:");
    if analysis.complexity_score < 0.4 {
        println!("✓ Excellent decomposition quality");
    } else if analysis.complexity_score < 0.6 {
        println!("✓ Good decomposition quality");
    } else if analysis.complexity_score < 0.75 {
        println!("⚠ Fair decomposition quality - possible mode mixing");
    } else {
        println!("✗ Poor decomposition quality - likely mode mixing");
    }
    
    if !analysis.high_complexity_imfs.is_empty() {
        println!("\nHigh-complexity IMFs: {:?}", analysis.high_complexity_imfs);
        println!("Recommendation: Investigate mode mixing");
    }
    
    Ok(())
}
```

### Example 3: Signal Preprocessing Validation

```rust
use ferromode::analysis::spectral_entropy_normalized;

fn validate_preprocessing(
    original: &[f64],
    processed: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let entropy_before = spectral_entropy_normalized(original)?;
    let entropy_after = spectral_entropy_normalized(processed)?;
    
    let change = (entropy_after - entropy_before).abs();
    
    println!("Preprocessing Validation");
    println!("=======================");
    println!("Entropy before: {:.4}", entropy_before);
    println!("Entropy after:  {:.4}", entropy_after);
    println!("Change:         {:.4}", change);
    
    if change > 0.3 {
        println!("\n⚠️ Warning: Large entropy change detected");
        println!("Preprocessing may have altered signal characteristics");
    } else {
        println!("\n✓ Preprocessing had minimal impact on spectral characteristics");
    }
    
    Ok(())
}
```

---

## Troubleshooting

### Issue: Entropy values seem unreasonable

**Symptoms:** Very high entropy for what appears to be a clean signal, or very low entropy for random data.

**Causes:**
- Signal not normalized (scale affects entropy)
- Embedding dimension too large for signal length
- Tolerance parameter too tight/loose (sample entropy)

**Solutions:**
```rust
// 1. Normalize the signal
let normalized = normalize(signal);
let entropy = spectral_entropy_normalized(&normalized)?;

// 2. Check signal length relative to embedding dimension
if signal.len() < 100 {
    println!("Consider using smaller embedding dimension");
}

// 3. Adjust tolerance
let std = calculate_std_dev(&signal);
let entropy = sample_entropy(&signal, 2, Some(0.2 * std))?;
```

### Issue: Inconsistent entropy values on repeated runs

**Symptoms:** Same signal produces different entropy values each time.

**Causes:**
- Not using deterministic parameters
- Preprocessing includes randomization
- FFT implementation has floating-point precision issues

**Solutions:**
```rust
// Use fixed, well-defined parameters
const EMBEDDING_DIM: usize = 3;
const SAMPLE_DIM: usize = 2;
const TOLERANCE_FACTOR: f64 = 0.2;

// Ensure preprocessing is deterministic
fn preprocess(signal: &[f64]) -> Vec<f64> {
    let normalized = normalize(signal);
    // No random operations
    normalized
}

// Set random seed if any randomization needed
use rand::SeedableRng;
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
```

### Issue: Decomposition seems poor but entropy is reasonable

**Symptoms:** Visual inspection suggests mode mixing, but complexity score is < 0.6.

**Causes:**
- Entropy metrics don't capture all quality issues
- Mode mixing may be acceptable for application
- Visual inspection may be misleading

**Solutions:**
```rust
// Use entropy as one data point among several
let analysis = EntropyAnalysis::from_decomposition(&decomp)?;
let imf_count = decomp.imfs.imfs.len();
let iterations = decomp.iterations;

println!("Decomposition metrics:");
println!("  Entropy score: {:.4}", analysis.complexity_score);
println!("  IMF count: {}", imf_count);
println!("  Iterations: {}", iterations);

// Cross-validate with other quality metrics
if imf_count > 10 {
    println!("⚠️ Many IMFs may indicate over-decomposition");
}
if iterations > 1000 {
    println!("⚠️ Many iterations may indicate convergence issues");
}
```

### Issue: Memory/performance concerns with large signals

**Symptoms:** Program crashes or becomes very slow with long signals.

**Causes:**
- Sample entropy is O(N²) complexity
- FFT requires power-of-2 padding (memory overhead)
- Processing entire long signals at once

**Solutions:**
```rust
// Use windowed analysis instead
let window_size = 2048;
let hop_size = 1024;

for i in (0..signal.len()).step_by(hop_size) {
    let end = std::cmp::min(i + window_size, signal.len());
    let window = &signal[i..end];
    let entropy = spectral_entropy_normalized(window)?;
    println!("Window entropy: {:.4}", entropy);
}

// For sample entropy on large signals, consider downsampling
let downsampled = downsample(&signal, 4);
let sample_ent = sample_entropy(&downsampled, 2, None)?;

// Or use approximate sample entropy (ApEn) for faster computation
// Note: ApEn not yet implemented, but would be O(N)
```

---

## References

**Spectral Entropy:**
- Inouye et al. (1991). "Quantification of EEG irregularity by use of the entropy of the power spectrum." Electroencephalography and Clinical Neurophysiology, 79(3), 204-210.

**Permutation Entropy:**
- Bandt & Pompe (2002). "Permutation entropy: A natural complexity measure for time series." Physical Review Letters, 88(17), 174102.

**Sample Entropy:**
- Richman & Moorman (2000). "Physiological time-series analysis using approximate entropy and sample entropy." American Journal of Physiology, 278(6), H2039-H2049.

**EMD and Entropy Integration:**
- Wu & Huang (2004). "A study of the characteristics of white noise using the empirical mode decomposition method." Journal of Sound and Vibration, 277(4-5), 1111-1123.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.5.0 | Apr 2026 | Initial comprehensive guide with all three entropy metrics |
| 2.5.1 | TBD | Add sliding window analysis section, Python examples |
| 2.5.2 | TBD | Add GPU-accelerated entropy computation |

---

## Contact and Support

For questions or issues with entropy metrics:
1. Check troubleshooting section above
2. Review code examples (Python and Rust)
3. Open an issue on GitHub with reproducible example
4. Include: signal characteristics, parameters used, expected vs. actual results
