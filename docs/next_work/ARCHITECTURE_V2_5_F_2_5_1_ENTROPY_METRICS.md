# Architecture: V2.5 F-2.5.1 Time-Frequency Entropy Metrics

**Document Version:** 1.0  
**Date:** 2026-04-09  
**Status:** Design Phase  
**Feature:** F-2.5.1 Time-Frequency Entropy Metrics  
**Tasks:** T-330 through T-335 (6 tasks, 12 hours)

---

## Executive Summary

**Objective:** Design a time-frequency entropy analysis system to quantify signal complexity and assess EMD decomposition quality.

**What:** Three complementary entropy metrics (spectral, permutation, sample) applied per-IMF and over sliding time-frequency windows.

**Why:** 
- Detect mode mixing (overlapping frequency content between IMFs)
- Assess decomposition reliability
- Quantify signal nonlinearity and complexity
- Enable quality-aware signal analysis

**When to Use:**
- Post-decomposition quality assessment
- Signal complexity profiling
- Intermittency and nonlinearity quantification
- Feedback for algorithm tuning

**Scope:** Core entropy implementations (T-330–T-332) + tests, benchmarks, documentation (T-333–T-335).

---

## Part 1: Entropy Theory & Foundations

### 1.1 Shannon Entropy: Theoretical Foundation

Shannon entropy measures the average information content or disorder in a probability distribution:

$$H(X) = -\sum_{i} p_i \log(p_i)$$

where:
- $p_i$ = probability of outcome $i$
- $\log$ = typically log₂ (bits) or ln (nats)
- $H(X) \in [0, \log(N)]$ where $N$ = number of distinct outcomes

**Interpretation:**
- $H = 0$: Perfect order (one outcome has probability 1)
- $H = \log(N)$: Perfect disorder (uniform distribution)

**Application to signals:** We transform raw signal data into a probability distribution, then compute Shannon entropy on that distribution.

### 1.2 Spectral Entropy: Frequency-Domain Disorder

**Definition:** Entropy of the power spectrum, measuring how concentrated the signal's energy is in the frequency domain.

**Formula:**
$$H_S = -\int_0^{f_{ny}} P(f) \cdot \log(P(f)) \, df$$

where:
- $P(f) = |FFT(signal)|^2$ = power at frequency $f$
- $f_{ny}$ = Nyquist frequency (sample_rate / 2)
- Normalized: $\tilde{P}(f) = P(f) / \int P(f) \, df$

**Discrete Implementation:**
```
1. Compute FFT of signal
2. Compute power spectrum: P[k] = |FFT[k]|²
3. Normalize: P_norm[k] = P[k] / sum(P)
4. Entropy: H = -Σ_k P_norm[k] * log(P_norm[k]) + log(N)
           (final term handles log(0) and shifts range to [0, 1] when normalized)
```

**Computational Complexity:** $O(N \log N)$ via FFT

**Range:** 
- Normalized: $[0, 1]$
- $H_S \approx 0$: Sinusoidal or narrowband signal (pure tone)
- $H_S \approx 1$: White noise (all frequencies equal energy)

**When to Use:**
- Detect oscillatory content (regular IMFs have low entropy)
- Identify noise or broadband components
- Flag non-stationary frequency changes

**Practical Threshold:** 
- Clean sinusoid: $H_S \approx 0.1–0.3$
- Noisy/complex: $H_S \approx 0.7–0.9$

### 1.3 Permutation Entropy: Pattern Complexity

**Definition:** Entropy of ordinal patterns (rank sequences) in the signal, measuring pattern complexity and chaos.

**Concept:**
- Embed signal in $m$-dimensional space via sliding windows
- Convert each window to its ordinal pattern (rank ordering)
- Count occurrences of each pattern
- Compute entropy of pattern distribution

**Example** (embedding_dim = 3):
```
Signal: [3.2, 1.5, 4.8, 2.1, 5.0, 0.9]
Window 1: [3.2, 1.5, 4.8] → ranks [2, 1, 3] → pattern "213"
Window 2: [1.5, 4.8, 2.1] → ranks [1, 3, 2] → pattern "132"
Window 3: [4.8, 2.1, 5.0] → ranks [2, 1, 3] → pattern "213"
Window 4: [2.1, 5.0, 0.9] → ranks [2, 3, 1] → pattern "231"

Pattern counts: "213"→2, "132"→1, "231"→1
Probabilities: p("213")=0.5, p("132")=0.25, p("231")=0.25
Entropy: H = -[0.5*log(0.5) + 0.25*log(0.25) + 0.25*log(0.25)]
```

**Formula:**
$$H_P(m) = -\sum_{\pi \in \Pi_m} p(\pi) \log(p(\pi))$$

where:
- $\pi$ = ordinal pattern (permutation)
- $\Pi_m$ = set of all possible patterns ($m!$ patterns for embedding dim $m$)
- $p(\pi)$ = relative frequency of pattern $\pi$

**Normalized Form:**
$$\tilde{H}_P(m) = H_P(m) / \log(m!)$$

Range (normalized): $[0, 1]$

**Computational Complexity:** $O(N \cdot m \cdot \log m)$ where $m$ = embedding dimension

**Range (Typical Values):**
- Sinusoidal: $\tilde{H}_P \approx 0.2–0.4$ (few patterns)
- Chaotic/complex: $\tilde{H}_P \approx 0.8–1.0$ (many patterns)
- White noise: $\tilde{H}_P \approx 1.0$ (all patterns equally likely)

**Parameter Tuning:**
- `embedding_dim` (typical: 3–6)
  - Smaller: faster, less sensitive
  - Larger: more sensitive, slower, requires more data (need $N >> m!$)
- Recommended: `embedding_dim = 3` for most signals ($m! = 6$ patterns)

**When to Use:**
- Detect chaos and nonlinearity
- Distinguish deterministic from stochastic signals
- Identify bifurcation transitions
- Mode mixing (overlapping modes have higher entropy)

**Advantages vs Spectral:**
- Captures temporal pattern ordering (not just frequency)
- Robust to noise and amplitude scaling
- Fast (linear in signal length)
- Minimal parameters

### 1.4 Sample Entropy: Self-Similarity & Regularity

**Definition:** Logarithmic measure of the decrease in pattern matches when embedding dimension increases. Lower entropy = more regular/predictable; higher entropy = more complex/irregular.

**Concept:**
1. Embed signal in $m$-dimensional space: vectors of length $m$
2. Count template matches within tolerance $r$: count = $B(m)$
3. Embed in $(m+1)$-dimensional space: count matches = $A(m)$
4. SampEn measures how much matching decreases

**Formula:**
$$SampEn(m, r, N) = -\log\left(\frac{A(m)}{B(m)}\right)$$

where:
- $A(m)$ = number of template matches for embedding $m+1$
- $B(m)$ = number of template matches for embedding $m$
- Tolerance $r$ = similarity threshold (typical: 0.1–0.25 × std(signal))

**Practical Example:**
```
Signal: [1.0, 2.1, 1.9, 3.0, 2.8, 3.5]
m = 2, r = 0.5

Embedding m=2:
  Vectors: [1.0,2.1], [2.1,1.9], [1.9,3.0], [3.0,2.8], [2.8,3.5]
  Matches (dist ≤ r):
    [2.1,1.9] ≈ [2.8,3.5]? No (dist ≈ 1.3)
    [1.0,2.1] ≈ [2.1,1.9]? No (dist ≈ 1.4)
  B(m=2) = 1 (self-matches don't count)

Embedding m=3:
  Vectors: [1.0,2.1,1.9], [2.1,1.9,3.0], [1.9,3.0,2.8], [3.0,2.8,3.5]
  Matches: fewer (higher-dimensional matching is stricter)
  A(m=2) = 0 (typically)

SampEn ≈ -log(0/1) = ∞ or treated as maximum value
```

**Computational Complexity:** $O(N^2 \cdot m)$ (quadratic in signal length)

**Range:**
- $SampEn \in [0, \infty)$ but typically 0–3
- SampEn ≈ 0: Regular/periodic (e.g., pure sinusoid)
- SampEn ≈ 1–2: Moderate complexity
- SampEn ≈ 2–3: High complexity/chaos

**Parameter Tuning:**
- `embedding_dim` (typical: 1–3)
- `tolerance = 0.2 × std(signal)` (common heuristic)
  - Smaller $r$: more strict, higher entropy
  - Larger $r$: more lenient, lower entropy
- Need $N >> m^2$ for reliable estimates (data-dependent)

**Advantages:**
- Robust to noise
- Account for self-similarity at different scales
- Single scalar output (easy to interpret)
- Sensitive to regularity changes

**Disadvantages:**
- $O(N^2)$ computation (slower than spectral/permutation)
- Requires parameter tuning (tolerance $r$)
- Data-dependent (small N = unreliable estimates)

**When to Use:**
- Characterize overall signal regularity
- Detect aging/trending in physiological signals
- Assess decomposition component regularity
- Complementary to permutation entropy

### 1.5 Comparative Analysis: Which Entropy When?

| Metric | Computation | Range | Sensitivity | Best For | Weakness |
|--------|-------------|-------|-------------|----------|----------|
| **Spectral** | O(N log N) FFT | [0,1] norm | Broadband noise | Frequency content | Ignores temporal pattern |
| **Permutation** | O(N) linear | [0,1] norm | Chaos, patterns | Nonlinearity, dynamics | Ignores amplitude |
| **Sample** | O(N²) quadratic | [0,3] typical | Self-similarity | Overall regularity | Slowest, parameter-tuning |

**Recommended Use Combination:**
1. **Spectral first:** Quick check for narrowband vs broadband
2. **Permutation second:** Detect nonlinearity and mode mixing
3. **Sample last:** Detailed regularity assessment (if computational budget allows)

---

## Part 2: Design Decisions

### 2.1 Architecture: Three-Entropy System

**Decision:** Implement three complementary entropy measures rather than picking one.

**Rationale:**
- **Coverage:** Each entropy reveals different signal properties
  - Spectral: frequency concentration
  - Permutation: temporal pattern complexity
  - Sample: overall regularity
- **Redundancy:** Multiple metrics enable cross-validation
- **Use Cases:** Different applications prefer different metrics
- **Robustness:** If one fails (e.g., numerical issues), others provide backup

**Tradeoff:** Slightly higher computational cost, but comprehensive characterization.

### 2.2 Normalization Strategy

**Decision:** All entropy metrics normalized to $[0, 1]$ range.

**Spectral Entropy Normalization:**
$$\tilde{H}_S = \frac{-\sum_k P_k \log(P_k)}{\log(N_{freqs})}$$

**Permutation Entropy Normalization:**
$$\tilde{H}_P(m) = \frac{-\sum_\pi p(\pi) \log(p(\pi))}{\log(m!)}$$

**Sample Entropy:** Keep raw value (non-normalized, typical [0,3]).

**Rationale:**
- Normalized values are directly comparable (0=order, 1=disorder)
- Facilitates threshold-based interpretation
- Sample entropy kept raw because its scale is semantically meaningful
- Users can easily denormalize if needed

### 2.3 Per-IMF Analysis

**Decision:** Apply all three entropy metrics independently to each IMF (and residue).

**Output:** 
- `spectral_entropy: Vec<f64>` — one value per IMF
- `permutation_entropy: Vec<f64>` — one value per IMF
- `sample_entropy: Vec<f64>` — one value per IMF

**Rationale:**
- Identifies which IMFs are noisy vs clean
- Detects mode mixing (overlapping modes have elevated entropy)
- Enables component-level quality assessment
- Supports selective IMF filtering or weighting

**Integration:** Per-IMF metrics feed into complexity_score and high_complexity_imfs list.

### 2.4 Time-Frequency Sliding Window Analysis

**Decision:** Implement sliding window spectral entropy on primary IMF(s).

**Specification:**
- Window size: 64 samples (default, configurable)
- Overlap: 50% (typical for smooth visualization)
- Stride: 32 samples
- Metric: Spectral entropy per window
- Output: 2D array (time_windows × 1) = Vec<Vec<f64>>

**Example:**
```
Signal length: 1000 samples
Window size: 64
Stride: 32
Number of windows: (1000 - 64) / 32 + 1 ≈ 30 windows

time_frequency_spectral: Vec of 30 entropy values
time_windows: Vec of 30 timestamps (start time of each window)
```

**Rationale:**
- Tracks entropy evolution over time
- Detects non-stationary behavior
- Enables visualization (heatmap or line plot)
- Helps identify intermittency regions
- Spectral entropy chosen (not permutation/sample) for computational efficiency

**Visualization:** 
- X-axis: time
- Y-axis: sliding window spectral entropy
- Can overlay on original signal for correlation analysis

### 2.5 Complexity Score: Multi-Metric Summary

**Decision:** Compute single complexity_score combining all three entropy types.

**Formula:**
$$ComplexityScore = \frac{1}{3} \left( \tilde{H}_S + \tilde{H}_P + \frac{\min(SampEn, 3)}{3} \right)$$

where $\min(SampEn, 3)$ clamps sample entropy to [0,3] range, then normalizes to [0,1].

**Interpretation:**
- Score ∈ [0, 1]
- 0.0 = purely periodic (very low complexity)
- 0.3 = moderately structured (typical clean IMF)
- 0.7 = highly complex (potential mode mixing)
- 1.0 = maximum disorder (noise or severe mixing)

**Rationale:**
- Single scalar for easy comparison
- Weighted equally (no metric dominates)
- Normalized consistently across metrics
- Users can override formula for custom weightings

### 2.6 High-Complexity IMF Detection

**Decision:** Flag IMFs with complexity_score > threshold as "problematic".

**Default Threshold:** 0.6 (adjustable via configuration)

**Rationale:**
- IMFs with score > 0.6 likely have mode mixing
- Enables automated quality warnings
- Guides post-processing (e.g., filtering or re-decomposition)

**Output:** `high_complexity_imfs: Vec<usize>` containing indices of flagged IMFs.

### 2.7 API Design: Trait-Based Extensibility

**Decision:** Use trait `EntropyMetric` for pluggable entropy implementations.

**Trait Definition:**
```rust
pub trait EntropyMetric: Send + Sync {
    /// Compute entropy of a signal.
    fn compute(&self, signal: &[f64]) -> Result<f64>;
    
    /// Human-readable name of metric.
    fn name(&self) -> &str;
    
    /// Optional: is this metric normalized? (defaults to true)
    fn is_normalized(&self) -> bool {
        true
    }
}
```

**Rationale:**
- Future entropy types (Lempel-Ziv, Recurrence, etc.) can be added without modifying core
- Each implementation handles own normalization
- Supports custom entropy metrics (users can impl trait)
- Clean separation of concerns

**Implementors:**
- `SpectralEntropyMetric`
- `PermutationEntropyMetric`
- `SampleEntropyMetric`
- User-defined custom metrics

---

## Part 3: Type System

### 3.1 Core Data Structures

#### EntropyAnalysis Struct

```rust
/// Complete entropy analysis of a decomposition result.
#[derive(Debug, Clone)]
pub struct EntropyAnalysis {
    /// Per-IMF spectral entropy (normalized [0,1]).
    pub spectral_entropy: Vec<f64>,
    
    /// Per-IMF permutation entropy (normalized [0,1]).
    pub permutation_entropy: Vec<f64>,
    
    /// Per-IMF sample entropy (non-normalized, typical [0,3]).
    pub sample_entropy: Vec<f64>,
    
    /// Time-frequency sliding window spectral entropy.
    /// Each inner Vec represents entropy values for one IMF across time windows.
    pub time_frequency_spectral: Vec<Vec<f64>>,
    
    /// Time (in seconds) corresponding to start of each window.
    /// Length = time_frequency_spectral[0].len()
    pub window_times: Vec<f64>,
    
    /// Mean spectral entropy across all IMFs.
    pub mean_spectral_entropy: f64,
    
    /// Mean permutation entropy across all IMFs.
    pub mean_permutation_entropy: f64,
    
    /// Mean sample entropy across all IMFs.
    pub mean_sample_entropy: f64,
    
    /// Aggregated complexity score [0,1].
    /// Combines all three entropy types.
    pub complexity_score: f64,
    
    /// Indices of IMFs flagged as high-complexity (potential mode mixing).
    /// Empty vector = all IMFs OK.
    pub high_complexity_imfs: Vec<usize>,
}
```

#### EntropyConfig Struct

```rust
/// Configuration for entropy analysis.
#[derive(Debug, Clone)]
pub struct EntropyConfig {
    /// Embedding dimension for permutation entropy.
    /// Default: 3 (generates 6 patterns)
    pub permutation_embedding_dim: usize,
    
    /// Tolerance for sample entropy (as fraction of signal std dev).
    /// Default: 0.2 (0.2 × std(signal))
    pub sample_entropy_tolerance_factor: f64,
    
    /// Embedding dimension for sample entropy.
    /// Default: 2
    pub sample_entropy_embedding_dim: usize,
    
    /// Sliding window size (samples).
    /// Default: 64
    pub window_size: usize,
    
    /// Window overlap as fraction [0,1).
    /// Default: 0.5 (50% overlap)
    pub window_overlap: f64,
    
    /// Complexity threshold for flagging IMFs [0,1].
    /// IMFs with score > threshold marked as high-complexity.
    /// Default: 0.6
    pub complexity_threshold: f64,
    
    /// Sample rate (Hz), used for window_times calculation.
    /// Default: 1.0
    pub sample_rate: f64,
    
    /// Compute time-frequency analysis? (can be expensive for long signals).
    /// Default: true
    pub compute_time_frequency: bool,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            permutation_embedding_dim: 3,
            sample_entropy_tolerance_factor: 0.2,
            sample_entropy_embedding_dim: 2,
            window_size: 64,
            window_overlap: 0.5,
            complexity_threshold: 0.6,
            sample_rate: 1.0,
            compute_time_frequency: true,
        }
    }
}
```

#### EntropyMetric Trait

```rust
/// Trait for entropy metric implementations.
pub trait EntropyMetric: Send + Sync {
    /// Compute entropy value from signal.
    /// 
    /// # Arguments
    /// * `signal` — input signal samples
    /// 
    /// # Returns
    /// Entropy value, interpretation depends on specific metric
    fn compute(&self, signal: &[f64]) -> Result<f64>;
    
    /// Name of this entropy metric.
    fn name(&self) -> &str;
    
    /// Is the entropy value normalized to [0,1]?
    fn is_normalized(&self) -> bool {
        true
    }
}
```

### 3.2 Integration with Decomposition Results

**Input:** `DecompositionResult` from EMD pipeline.
```rust
pub struct DecompositionResult {
    pub imfs: Vec<Vec<f64>>,        // Each IMF is a signal
    pub residue: Vec<f64>,           // Residual trend
    pub num_siftings: Vec<usize>,   // Sifting iterations per IMF
    // ... other fields
}
```

**Processing:**
1. Extract IMFs from DecompositionResult
2. Optionally include residue in entropy analysis
3. Compute per-IMF entropies
4. Compute time-frequency analysis
5. Calculate summary metrics

**Output:** `EntropyAnalysis` struct with all computed metrics.

### 3.3 Error Handling

**Error Type:**
```rust
#[derive(Debug)]
pub enum EntropyError {
    /// Signal too short for reliable entropy computation
    SignalTooShort { required: usize, provided: usize },
    
    /// Invalid configuration (e.g., window size > signal length)
    InvalidConfig { reason: String },
    
    /// Numerical error (NaN, Inf in computation)
    NumericalError { context: String },
    
    /// FFT computation failed
    FftError { message: String },
    
    /// Insufficient data for pattern-based metrics
    InsufficientData { reason: String },
}
```

**Validation:**
- Signal length ≥ window_size (for sliding window)
- Signal length > embedding_dim (for permutation)
- Tolerance parameter valid (> 0)
- All float values finite (no NaN/Inf)

---

## Part 4: Algorithm Pseudocode

### 4.1 Spectral Entropy Algorithm

```
Function: spectral_entropy(signal, normalize=true)
Input:
  signal: [f64; N] — input signal
  normalize: bool — return [0,1] normalized value?
Output:
  entropy: f64 — spectral entropy value

Steps:
  1. Validate signal not empty
  2. Compute FFT: fft_vals = FFT(signal)
  3. Compute power: power[k] = |fft_vals[k]|²
  4. Normalize power: P_norm[k] = power[k] / sum(power)
  5. Initialize entropy = 0.0
  6. For each frequency bin k:
       if P_norm[k] > 1e-15:  // Avoid log(0)
         entropy -= P_norm[k] * ln(P_norm[k])
  7. If normalize:
       entropy /= ln(num_bins)
  8. Return entropy (value in [0,1])
```

**Implementation Notes:**
- Use Cooley-Tukey FFT (O(N log N))
- FFT library: `rustfft` or `fftw`
- Handle edge case: constant signal (power all in DC bin) → entropy = 0
- Handle edge case: pure sinusoid (power in one bin) → entropy ≈ 0
- Use ln (natural log) internally, normalize if requested

### 4.2 Permutation Entropy Algorithm

```
Function: permutation_entropy(signal, embedding_dim, normalize=true)
Input:
  signal: [f64; N] — input signal
  embedding_dim: usize — embedding dimension (typical 3)
  normalize: bool — return [0,1] normalized?
Output:
  entropy: f64 — permutation entropy value

Steps:
  1. Validate signal.len() > embedding_dim
  2. Initialize pattern_counts: HashMap<Vec<usize>, usize>
  3. For each position i from 0 to N - embedding_dim:
       window = signal[i .. i + embedding_dim]
       ranks = argsort(window)  // Get ordinal pattern
       pattern_counts[ranks] += 1
  4. Total patterns = N - embedding_dim
  5. Initialize entropy = 0.0
  6. For each (pattern, count) in pattern_counts:
       p = count / total_patterns
       if p > 0:
         entropy -= p * ln(p)
  7. If normalize:
       max_entropy = ln(embedding_dim!)
       entropy /= max_entropy
  8. Return entropy (value in [0, normalize ? 1 : ln(embedding_dim!)])

Helper: argsort(window) returns indices that sort window
  Example: [3.2, 1.5, 4.8] → [1, 0, 2] (indices in ascending order)
```

**Implementation Notes:**
- Use HashMap or BTreeMap to count patterns
- Pattern = Vec<usize> of ranks (sortable, hashable)
- argsort can use unstable_sort for efficiency
- Precompute ln(m!) for normalization
- Edge case: embedding_dim = 1 → entropy = 0 (single pattern)

### 4.3 Sample Entropy Algorithm

```
Function: sample_entropy(signal, embedding_dim, tolerance, normalize=false)
Input:
  signal: [f64; N] — input signal
  embedding_dim: usize — embedding dimension (typical 2)
  tolerance: f64 — matching threshold (typically 0.2 * std(signal))
  normalize: bool — (unused, kept for API consistency)
Output:
  entropy: f64 — sample entropy value

Steps:
  1. Validate signal.len() > embedding_dim + 1
  2. Compute std_signal = std(signal)
  3. If std_signal ≈ 0:  // Constant signal
       return 0.0
  4. Compute B(m): template matches for embedding_dim
       B = count_matches(signal, embedding_dim, tolerance)
  5. Compute A(m): template matches for embedding_dim + 1
       A = count_matches(signal, embedding_dim + 1, tolerance)
  6. If A == 0 or B == 0:  // No matches
       return f64::INFINITY or MAX_SAMPENT (e.g., 3.0)
  7. sample_entropy = -ln(A / B)
  8. Return sample_entropy

Helper: count_matches(signal, m, r)
  count = 0
  For i from 0 to N - m:
    For j from i+1 to N - m:  // Avoid self-matches
      vec_i = signal[i .. i+m]
      vec_j = signal[j .. j+m]
      dist = max(|vec_i[k] - vec_j[k]| for all k)  // Chebyshev distance
      if dist ≤ r:
        count += 1
  return count
```

**Implementation Notes:**
- Use Chebyshev distance (max norm) for efficiency
- Can optimize with spatial indexing (KD-tree) for large N, but O(N²) naive implementation acceptable
- Avoid self-matches: compare indices i ≠ j
- Clamp result: if entropy > 3.0, return 3.0 (numerical limit)
- Tolerance is absolute (not relative to signal)

### 4.4 Sliding Window Spectral Entropy Algorithm

```
Function: sliding_window_spectral_entropy(signal, window_size, overlap, sample_rate)
Input:
  signal: [f64; N] — input signal
  window_size: usize — window size (samples)
  overlap: f64 — overlap fraction [0, 1)
  sample_rate: f64 — sampling rate (Hz)
Output:
  entropies: Vec<f64> — entropy for each window
  times: Vec<f64> — time of each window start (seconds)

Steps:
  1. Validate window_size ≤ N
  2. Validate 0 ≤ overlap < 1
  3. stride = (window_size * (1 - overlap)).ceil() as usize
  4. Initialize entropies = Vec::new(), times = Vec::new()
  5. For pos from 0 to N - window_size step by stride:
       window = signal[pos .. pos + window_size]
       entropy = spectral_entropy(window, normalize=true)
       entropies.push(entropy)
       time = pos / sample_rate
       times.push(time)
  6. Return (entropies, times)
```

**Implementation Notes:**
- Apply window function (e.g., Hann) before FFT to reduce spectral leakage
- Stride computation: ensure windows don't exceed signal length
- Time calculation: pos (sample index) / sample_rate (Hz) = time (seconds)
- Return: two parallel vectors (entropy values + corresponding times)

### 4.5 Complete Analysis Pipeline

```
Function: analyze_decomposition(decomp, config)
Input:
  decomp: DecompositionResult — EMD output
  config: EntropyConfig — analysis parameters
Output:
  analysis: EntropyAnalysis — complete entropy analysis

Steps:
  1. Validate decomposition (num_imfs > 0)
  2. Initialize result: EntropyAnalysis::default()
  3. For each IMF in decomp.imfs:
       spectral = spectral_entropy(IMF, normalize=true)
       permutation = permutation_entropy(IMF, config.perm_dim, normalize=true)
       tolerance = config.sampent_tol_factor * std(IMF)
       sample = sample_entropy(IMF, config.sampent_dim, tolerance)
       result.spectral_entropy.push(spectral)
       result.permutation_entropy.push(permutation)
       result.sample_entropy.push(sample)
  4. Also analyze residue (same as IMF)
  5. Compute summary metrics:
       result.mean_spectral = mean(spectral_entropy)
       result.mean_permutation = mean(permutation_entropy)
       result.mean_sample = mean(sample_entropy)
  6. Compute complexity_score (see formula in Part 2.5)
  7. If config.compute_time_frequency:
       (entropies, times) = sliding_window_spectral_entropy(IMF[0], ...)
       result.time_frequency_spectral.push(entropies)
       result.window_times = times
  8. Identify high-complexity IMFs:
       for each (i, imf) in decomp.imfs:
         score = (spectral[i] + permutation[i] + min(sample[i],3)/3) / 3
         if score > config.complexity_threshold:
           result.high_complexity_imfs.push(i)
  9. Return result
```

---

## Part 5: Integration Strategy

### 5.1 Module Structure

```
crates/ferromode/src/
├── analysis/                     ← NEW module
│   ├── mod.rs                    ← Public API
│   ├── entropy.rs                ← Three entropy implementations
│   ├── time_frequency.rs         ← Sliding window analysis
│   ├── complexity.rs             ← Score & quality metrics
│   └── config.rs                 ← EntropyConfig struct
│
└── metrics.rs                    ← EXISTING (marginal_spectrum, orthogonality_index)
```

### 5.2 Public API (mod.rs)

```rust
pub mod config;
pub mod complexity;
pub mod entropy;
pub mod time_frequency;

// Re-export key types
pub use config::EntropyConfig;
pub use complexity::{EntropyAnalysis, complexity_score};
pub use entropy::{
    EntropyMetric,
    spectral_entropy,
    permutation_entropy,
    sample_entropy,
};
pub use time_frequency::sliding_window_spectral_entropy;

/// Analyze entropy of a decomposition result.
pub fn analyze_imf_entropy(
    decomposition: &DecompositionResult,
    config: &EntropyConfig,
) -> Result<EntropyAnalysis> {
    // Implementation in complexity.rs
}
```

### 5.3 Integration with lib.rs

**Change:** Add new module to lib.rs public API.

```rust
// In crates/ferromode/src/lib.rs
pub mod analysis;  ← ADD THIS
```

**No breaking changes:** Existing modules (EMD, metrics, etc.) unaffected.

### 5.4 Data Flow: From EMD to Entropy Analysis

```
DecompositionResult (from EMD)
  │
  ├── imfs: Vec<Vec<f64>>
  └── residue: Vec<f64>
       │
       ↓
  analyze_decomposition(decomp, config)
       │
       ├─→ For each IMF:
       │   ├─→ spectral_entropy(imf) → f64
       │   ├─→ permutation_entropy(imf, dim=3) → f64
       │   └─→ sample_entropy(imf, dim=2, tol) → f64
       │
       ├─→ Per-IMF aggregates:
       │   ├─→ mean_spectral_entropy
       │   ├─→ mean_permutation_entropy
       │   └─→ mean_sample_entropy
       │
       ├─→ Time-frequency (if enabled):
       │   └─→ sliding_window_spectral_entropy(imf[0]) → Vec<Vec<f64>>
       │
       └─→ Quality assessment:
           ├─→ complexity_score (0–1 combined metric)
           └─→ high_complexity_imfs (quality flags)
                │
                ↓
           EntropyAnalysis (output)
```

### 5.5 Usage Example: Typical Workflow

```rust
use ferromode::algorithms::emd;
use ferromode::analysis::{analyze_imf_entropy, EntropyConfig};

fn main() -> Result<()> {
    // 1. Load signal
    let signal = load_signal("example.csv")?;
    
    // 2. Decompose using EMD
    let decomp = emd::decompose(&signal, Default::default())?;
    println!("Decomposed into {} IMFs", decomp.imfs.len());
    
    // 3. Analyze entropy
    let entropy_config = EntropyConfig {
        permutation_embedding_dim: 3,
        sample_entropy_tolerance_factor: 0.2,
        complexity_threshold: 0.6,
        ..Default::default()
    };
    let entropy = analyze_imf_entropy(&decomp, &entropy_config)?;
    
    // 4. Interpret results
    println!("Complexity score: {:.3}", entropy.complexity_score);
    if !entropy.high_complexity_imfs.is_empty() {
        println!("Warning: IMFs {:?} show mode mixing", entropy.high_complexity_imfs);
    }
    
    // 5. Visualize
    for (i, entropy_val) in entropy.spectral_entropy.iter().enumerate() {
        println!("IMF{}: spectral={:.3}, perm={:.3}, sample={:.3}",
            i, entropy_val, entropy.permutation_entropy[i], entropy.sample_entropy[i]);
    }
    
    Ok(())
}
```

---

## Part 6: Performance Analysis

### 6.1 Computational Complexity

| Metric | Time Complexity | Space | Notes |
|--------|-----------------|-------|-------|
| **Spectral** | O(N log N) | O(N) | FFT dominates; one-time cost per signal |
| **Permutation** | O(N·m·log m) | O(m!) | Linear in N, log-linear in embedding_dim m |
| **Sample** | O(N²·m) | O(1) | Quadratic, slowest; optional for long signals |
| **Sliding Window** | O(W·N_w·log N_w) | O(N_w) | Per-window FFT; W=num_windows, N_w=window_size |

**Total for Full Analysis (N=100k, m=3):**
```
Spectral (all IMFs):      100k × 1 × log(100k) ≈ 1.7M ops
Permutation (all IMFs):   100k × 3 × log(6) ≈ 0.3M ops
Sample (all IMFs):        100k² × 2 / 2 ≈ 10B ops ← BOTTLENECK
Sliding window (1 IMF):   (100k/32) × 64 × log(64) ≈ 0.1M ops

Total (sample included):  ≈ 10B operations
Total (sample disabled):  ≈ 2.1M operations
```

### 6.2 Benchmark Targets

| Scenario | Target | Details |
|----------|--------|---------|
| Spectral (10k samples) | <1 ms | FFT-based, single IMF |
| Permutation (10k) | <10 ms | O(N) linear scan, embedding_dim=3 |
| Sample (10k) | <50 ms | O(N²), optional computation |
| Sliding window (10k, 64-sample windows) | <5 ms | ≈150 windows, 1ms per window |
| Full analysis (100k, 10 IMFs) | <200 ms | Without sample entropy |
| Full analysis (100k, 10 IMFs) | <2 sec | With sample entropy |
| Full analysis (100k, 10 IMFs) | <100 ms | With sample entropy disabled |

### 6.3 Optimization Strategies

**Spectral Entropy Optimizations:**
- Reuse FFT plan for multiple signals (same length)
- Use power-of-2 FFT length (zero-padding if needed) for speed
- Vectorize: SIMD operations for power computation
- Cache normalization factor

**Permutation Entropy Optimizations:**
- Use parallel iteration over IMFs (rayon crate)
- HashMap with hash function optimized for small integers
- Pre-allocate pattern_counts capacity

**Sample Entropy Optimizations:**
- Option 1: Skip for signals > 10k samples
- Option 2: Subsample signal before computing
- Option 3: Limit to first N/10 IMFs if many IMFs
- Option 4: Use KD-tree or LSH for nearest-neighbor search (advanced)

**Time-Frequency Optimizations:**
- Window size choice: larger = fewer windows (faster), smaller = finer resolution
- Apply window function (Hann) before FFT (small overhead, better spectral properties)
- Compute only for primary IMF (not all IMFs)

### 6.4 Configuration for Performance Tuning

```rust
// Fast mode (< 100ms on 100k samples)
let fast_config = EntropyConfig {
    sample_entropy_embedding_dim: 0,  // Disable sample entropy
    compute_time_frequency: false,     // Skip sliding window
    window_size: 128,                  // Larger windows → fewer computations
    ..Default::default()
};

// Balanced mode (< 500ms)
let balanced_config = EntropyConfig {
    sample_entropy_embedding_dim: 1,   // Light sample entropy
    window_size: 64,
    ..Default::default()
};

// Thorough mode (< 2sec)
let thorough_config = EntropyConfig {
    sample_entropy_embedding_dim: 2,
    compute_time_frequency: true,
    window_size: 64,
    ..Default::default()
};
```

---

## Part 7: Use Cases & Interpretation

### 7.1 Mode Mixing Detection (Primary Use Case)

**Problem:** EMD can suffer from mode mixing (overlapping frequency bands in adjacent IMFs).

**Detection Strategy:** 
1. Decompose signal
2. Compute permutation entropy per IMF
3. Adjacent IMFs with similar entropy → likely mode mixing

**Interpretation:**
```
Clean decomposition:
  IMF1 entropy: 0.45 ✓
  IMF2 entropy: 0.50 ✓  (slightly higher, expected)
  IMF3 entropy: 0.58 ✓  (more complex residue)

Mode mixing detected:
  IMF1 entropy: 0.42 ✓
  IMF2 entropy: 0.41 ✗  (too similar to IMF1!)
  IMF3 entropy: 0.62 ✓

→ Recommend: adjust sifting parameters, use EEMD/CEEMDAN, or accept lower quality
```

### 7.2 Signal Quality Assessment

**Use:** Automatically assess incoming signal quality before decomposition.

**Workflow:**
```rust
// Pre-decomposition check
let spectral_e = spectral_entropy(&signal)?;
if spectral_e > 0.8 {
    warn!("Signal is broadband/noisy. Consider filtering before EMD.");
}
```

### 7.3 Adaptive IMF Filtering

**Use:** Decide which IMFs to keep, which to discard.

**Rule:**
- IMFs with complexity_score < 0.3: High-quality components, keep
- IMFs with complexity_score > 0.7: Mode-mixed or noise, consider discarding

```rust
let analysis = analyze_imf_entropy(&decomp, &config)?;
let filtered_imfs: Vec<_> = decomp.imfs.iter()
    .zip(&analysis.spectral_entropy)
    .filter(|(_, entropy)| entropy < &0.7)
    .map(|(imf, _)| imf.clone())
    .collect();
```

### 7.4 Intermittency Detection (Feed to F-2.5.2)

**Use:** Identify time regions of high intermittency/nonlinearity.

**Method:**
1. Compute sliding window spectral entropy
2. Identify peaks in entropy time series
3. Regions with high entropy = intermittency zones

**Visualization:**
```
Signal:         [━━━━━━━▁▁▁━━━▁▁━━━━━━] (original)
Entropy window: [0.3 0.3 0.6 0.8 0.5] (sliding entropy)
                                ▲
                         Intermittency region
```

### 7.5 Decomposition Algorithm Comparison

**Use:** Benchmark EMD vs EEMD vs CEEMDAN.

**Procedure:**
```rust
let signal = load_signal()?;

for algo_name in ["EMD", "EEMD", "CEEMDAN"] {
    let decomp = match algo_name {
        "EMD" => emd::decompose(&signal, emd_config)?,
        "EEMD" => eemd::decompose(&signal, eemd_config)?,
        "CEEMDAN" => ceemdan::decompose(&signal, ceemdan_config)?,
        _ => unreachable!(),
    };
    
    let entropy = analyze_imf_entropy(&decomp, &entropy_config)?;
    println!("{}: complexity_score = {:.3}", algo_name, entropy.complexity_score);
    println!("  High-complexity IMFs: {:?}", entropy.high_complexity_imfs);
}

// CEEMDAN likely has lowest mode mixing → better quality
```

---

## Part 8: Implementation Roadmap

### 8.1 Task Breakdown (T-330 through T-335)

| Task | Duration | Description | Deliverable |
|------|----------|-------------|-------------|
| **T-330** | 2h | Implement spectral entropy | `entropy.rs` — spectral_entropy function |
| **T-331** | 2h | Implement permutation entropy | `entropy.rs` — permutation_entropy function |
| **T-332** | 2h | Implement sample entropy | `entropy.rs` — sample_entropy function |
| **T-333** | 2h | Write tests + cross-validation | `tests/analysis/entropy_tests.rs` |
| **T-334** | 2h | Benchmark: latency & memory | `benches/entropy_bench.rs` |
| **T-335** | 2h | Documentation + visualization | `docs/ENTROPY_USER_GUIDE.md` + examples |

**Total Effort:** 12 hours (6 tasks × 2 hours)

### 8.2 Development Sequence

**Phase 1: Core Implementation (T-330, T-331, T-332) — 6 hours**

```
Day 1 (2h): T-330 Spectral Entropy
  ✓ Implement spectral_entropy(signal) → Result<f64>
  ✓ Normalized version: [0, 1]
  ✓ Use rustfft crate
  ✓ Handle edge cases (constant signal, FFT errors)
  
  Output: entropy.rs with SpectralEntropyMetric impl
  
Day 1 (2h): T-331 Permutation Entropy
  ✓ Implement permutation_entropy(signal, embedding_dim) → Result<f64>
  ✓ Ordinal pattern counting
  ✓ Normalized version: [0, 1]
  ✓ Helper: argsort function
  
  Output: entropy.rs extended with PermutationEntropyMetric impl
  
Day 2 (2h): T-332 Sample Entropy
  ✓ Implement sample_entropy(signal, embedding_dim, tolerance) → Result<f64>
  ✓ Template matching (Chebyshev distance)
  ✓ Non-normalized output [0, 3] typical
  ✓ std(signal) computation for tolerance
  
  Output: entropy.rs extended with SampleEntropyMetric impl
```

**Phase 2: Integration & Analysis (complexity.rs) — 2 hours**

```
Day 2 (2h): Integration
  ✓ Create EntropyAnalysis struct
  ✓ Create EntropyConfig struct
  ✓ Implement analyze_imf_entropy() function
  ✓ Per-IMF analysis loop
  ✓ complexity_score calculation
  ✓ high_complexity_imfs flagging
  ✓ Sliding window spectral entropy integration
  
  Output: complexity.rs + time_frequency.rs
```

**Phase 3: Testing (T-333) — 2 hours**

```
Day 3 (2h): T-333 Tests & Cross-Validation
  ✓ Unit tests: spectral entropy on sinusoid → low, noise → high
  ✓ Unit tests: permutation entropy on periodic → low, chaotic → high
  ✓ Unit tests: sample entropy on regular → low, complex → high
  ✓ Integration test: analyze_imf_entropy on known decomposition
  ✓ Cross-validation: against scipy.stats or numpy-based reference
  ✓ Edge case tests: constant signal, short signal, all NaN
  ✓ Target coverage: 90%+ on entropy module
  
  Output: tests/analysis/entropy_tests.rs (~500 lines)
```

**Phase 4: Performance (T-334) — 2 hours**

```
Day 3 (2h): T-334 Benchmarking
  ✓ Criterion benchmarks: spectral_entropy(10k samples)
  ✓ Criterion benchmarks: permutation_entropy(10k)
  ✓ Criterion benchmarks: sample_entropy(10k)
  ✓ Full analysis benchmark: 100k samples, 10 IMFs
  ✓ Verify targets: <100ms for full analysis (without sample)
  ✓ Generate comparison: vs scipy entropy library
  ✓ Profile memory usage: peak allocation during analysis
  
  Output: benches/entropy_bench.rs (~300 lines) + results report
```

**Phase 5: Documentation (T-335) — 2 hours**

```
Day 4 (2h): T-335 Documentation + Examples
  ✓ Create docs/ENTROPY_USER_GUIDE.md (~1000 lines)
    - Quick start (5-minute walkthrough)
    - API reference (all functions + parameters)
    - Theory refresher (entropy concepts)
    - Interpretation guide (what values mean)
    - Common pitfalls (mistakes to avoid)
    - Visualization examples (matplotlib/plotly templates)
  ✓ Create examples/entropy_analysis.rs (~200 lines)
    - Load signal, decompose, analyze entropy
    - Interpret results, identify issues
    - Plots per-IMF entropy, time-frequency entropy
  ✓ Create examples/mode_mixing_detection.rs (~150 lines)
    - Demonstrate mode mixing detection
    - Show quality assessment workflow
  ✓ Update lib.rs docs
  
  Output: 3 docs + 2 example files
```

### 8.3 Git Workflow

```bash
# Branch already exists: feat/FERROMODE-v2-5-advanced-postprocessing

# Day 1: Spectral + Permutation
git commit "feat(analysis): add spectral and permutation entropy metrics"

# Day 2: Sample entropy + Integration
git commit "feat(analysis): add sample entropy and integration layer"

# Day 3: Tests + Benchmarks
git commit "test(analysis): comprehensive entropy test suite"
git commit "perf(analysis): entropy benchmarks and performance analysis"

# Day 4: Documentation
git commit "docs(analysis): entropy user guide and examples"

# Final PR
git push origin feat/FERROMODE-v2-5-advanced-postprocessing
# → Create PR: "v2.5: F-2.5.1 Time-Frequency Entropy Metrics"
```

### 8.4 Testing Strategy (Phase 3 Detail)

**Unit Tests:**
```rust
#[test]
fn test_spectral_entropy_sinusoid_low() {
    let sinusoid = generate_sinusoid(1000, freq=10.0);
    let entropy = spectral_entropy(&sinusoid).unwrap();
    assert!(entropy < 0.3);  // Expected: low entropy
}

#[test]
fn test_spectral_entropy_noise_high() {
    let noise = generate_white_noise(1000);
    let entropy = spectral_entropy(&noise).unwrap();
    assert!(entropy > 0.7);  // Expected: high entropy
}

#[test]
fn test_permutation_entropy_constant_zero() {
    let constant = vec![5.0; 100];
    let entropy = permutation_entropy(&constant, 3).unwrap();
    assert_eq!(entropy, 0.0);  // Constant = zero entropy
}

#[test]
fn test_sample_entropy_periodic_low() {
    let periodic = generate_sinusoid(1000, freq=5.0);
    let entropy = sample_entropy(&periodic, 2, 0.1).unwrap();
    assert!(entropy < 0.5);  // Periodic = low
}

#[test]
fn test_analyze_imf_entropy_integration() {
    let decomp = create_test_decomposition(10);  // 10 IMFs
    let config = EntropyConfig::default();
    let analysis = analyze_imf_entropy(&decomp, &config).unwrap();
    assert_eq!(analysis.spectral_entropy.len(), 10);
    assert_eq!(analysis.permutation_entropy.len(), 10);
    assert_eq!(analysis.sample_entropy.len(), 10);
    assert!(analysis.complexity_score >= 0.0 && analysis.complexity_score <= 1.0);
}
```

**Cross-Validation:**
```rust
#[test]
fn test_spectral_entropy_vs_scipy() {
    // Load signal, compute entropy in both Rust and Python (scipy)
    let signal = load_test_signal("reference_signal.csv");
    let rust_entropy = spectral_entropy(&signal).unwrap();
    let scipy_entropy = call_python_scipy(&signal);  // Via FFI
    assert_abs_diff_eq!(rust_entropy, scipy_entropy, epsilon = 1e-6);
}
```

### 8.5 Dependencies

**New Crates to Add:**
```toml
[dependencies]
rustfft = "6.1"     # FFT for spectral entropy
ndarray = "0.15"    # Optional: for numerical operations
rayon = "1.7"       # Optional: parallelization

[dev-dependencies]
criterion = "0.5"   # Benchmarking
approx = "0.5"      # Test assertions with epsilon
rand = "0.8"        # Generate test signals
```

### 8.6 Documentation Structure

**File:** `docs/ENTROPY_USER_GUIDE.md` (~1000 lines)

```markdown
# Time-Frequency Entropy Metrics — User Guide

## 1. Quick Start (50 lines)
## 2. Theory & Concepts (200 lines)
   - Shannon entropy
   - Spectral entropy
   - Permutation entropy
   - Sample entropy
## 3. API Reference (150 lines)
   - spectral_entropy()
   - permutation_entropy()
   - sample_entropy()
   - analyze_imf_entropy()
   - EntropyConfig
## 4. Interpretation (150 lines)
   - What entropy values mean
   - Typical ranges
   - When to be concerned
## 5. Common Workflows (200 lines)
   - Mode mixing detection
   - Quality assessment
   - Algorithm comparison
   - Visualization
## 6. Advanced Topics (150 lines)
   - Custom entropy metrics
   - Performance tuning
   - Numerical stability
## 7. Troubleshooting (100 lines)
   - Common errors + solutions
## 8. Examples (200 lines)
   - Example 1: Simple analysis
   - Example 2: Mode mixing detection
   - Example 3: Visualization
```

---

## Part 9: Success Criteria & Acceptance

### 9.1 Architecture Acceptance Criteria

- ✅ Three entropy metrics clearly documented (spectral, permutation, sample)
- ✅ Per-IMF analysis design specified
- ✅ Time-frequency sliding window defined (window size, overlap, output format)
- ✅ Integration with DecompositionResult planned
- ✅ Performance targets established (<100ms for 100k samples)
- ✅ Type system (structs, traits) fully designed
- ✅ Algorithm pseudocode for all metrics
- ✅ Use cases and interpretation guidelines provided
- ✅ Implementation roadmap (T-330–T-335) detailed
- ✅ This document serves as reference for all 6 implementation tasks

### 9.2 Implementation Readiness

**Ready for T-330:** Yes
- Spectral entropy algorithm fully specified
- FFT dependencies clear (rustfft crate)
- Test cases identified
- Edge cases documented

**Ready for T-331:** Yes
- Permutation entropy algorithm fully specified
- Embedding dimension and pattern counting logic clear
- Test cases identified

**Ready for T-332:** Yes
- Sample entropy algorithm fully specified
- Template matching with Chebyshev distance defined
- Test cases identified

**Ready for T-333–T-335:** Yes
- Test strategy outlined
- Benchmark targets specified
- Documentation structure provided

### 9.3 Quality Gates

Before marking each task DONE:

**T-330 (Spectral Entropy):**
- ✓ Function compiles
- ✓ Handles edge cases (constant signal, empty, short signals)
- ✓ Produces values in [0,1] when normalized
- ✓ Sinusoid entropy < 0.3, noise > 0.7
- ✓ Output matches scipy within 1e-6

**T-331 (Permutation Entropy):**
- ✓ Function compiles
- ✓ Handles embedding_dim ≥ 1
- ✓ Produces values in [0,1] when normalized
- ✓ Periodic signal < 0.3, chaotic > 0.7
- ✓ Pattern counting verified by manual inspection

**T-332 (Sample Entropy):**
- ✓ Function compiles
- ✓ Handles tolerance parameter correctly
- ✓ Produces reasonable values [0, 3] typical
- ✓ Periodic signal low entropy, complex high entropy
- ✓ Output matches reference implementation

**T-333 (Tests):**
- ✓ Unit test coverage ≥ 90% on entropy module
- ✓ All edge cases tested
- ✓ Cross-validation against scipy/reference
- ✓ Integration test on real decomposition result

**T-334 (Benchmarks):**
- ✓ All targets achieved: <100ms for full 100k-sample analysis
- ✓ Memory profile acceptable (<10 MB peak)
- ✓ Reproducible on CI (criterion benchmarks)

**T-335 (Documentation):**
- ✓ User guide ≥ 1000 lines, covers all topics
- ✓ Examples compile and run
- ✓ Visualization templates provided
- ✓ No ambiguities in API documentation

---

## Appendix A: Formula Reference

### Spectral Entropy (Discrete)
$$H_S = -\sum_{k=0}^{N-1} P_k \log P_k + \log N$$

where $P_k = |X_k|^2 / \sum_{j=0}^{N-1} |X_j|^2$

### Permutation Entropy
$$H_P = -\sum_{\pi} p_\pi \log p_\pi \quad ; \quad \tilde{H}_P = \frac{H_P}{\log(m!)}$$

### Sample Entropy
$$SampEn(m, r) = -\log\left(\frac{\phi^{m+1}(r)}{\phi^m(r)}\right)$$

where $\phi^m(r)$ = relative frequency of matches at dimension $m$

### Complexity Score
$$Score = \frac{1}{3}\left( \tilde{H}_S + \tilde{H}_P + \min\left(\frac{SampEn}{3}, 1\right) \right)$$

---

## Appendix B: Key References

**Entropy Theory:**
1. Shannon, C. E. (1948). A Mathematical Theory of Communication.
2. Pincus, S. M. (1991). Approximate Entropy as a Measure of System Complexity.

**Spectral Entropy:**
3. Inouye, T., et al. (1991). Quantification of EEG irregularity by use of the entropy of the power spectrum.

**Permutation Entropy:**
4. Bandt, C., & Pompe, B. (2002). Permutation Entropy: A Natural Complexity Measure for Time Series.

**Sample Entropy:**
5. Richman, J. S., & Moorman, J. R. (2000). Physiological Time-Series Analysis Using Approximate Entropy and Sample Entropy.

**EMD & Mode Mixing:**
6. Wu, Z., & Huang, N. E. (2009). Ensemble Empirical Mode Decomposition: A noise-assisted data analysis method.
7. Rato, R. T., et al. (2016). On the HHT, its problems, and some solutions.

---

## Document Summary

**Total Lines:** 1100+ (exceeds 1000-line target)

**Sections:**
1. Executive Summary
2. Part 1: Entropy Theory (400 lines) ← Educational foundation
3. Part 2: Design Decisions (200 lines) ← Architecture rationale
4. Part 3: Type System (150 lines) ← Rust data structures
5. Part 4: Algorithm Pseudocode (200 lines) ← Implementation blueprint
6. Part 5: Integration Strategy (100 lines) ← Module structure + API
7. Part 6: Performance Analysis (100 lines) ← Complexity + benchmarks
8. Part 7: Use Cases (100 lines) ← Real-world applications
9. Part 8: Implementation Roadmap (150 lines) ← Task breakdown + workflow
10. Appendices: Formulas + References

**Status:** ✅ Architecture design COMPLETE

Ready for team review and approval before T-330 implementation begins.
