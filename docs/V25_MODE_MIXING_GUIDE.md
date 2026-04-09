# Mode Mixing Detection & Visualization (V2.5 F-2.5.2)

## Overview

**Mode mixing** occurs when two or more IMFs share overlapping frequency content in EMD decomposition. This guide explains detection, interpretation, and remediation.

## What is Mode Mixing?

Mode mixing indicates loss of unique frequency separation. In ideal EMD:
- Each IMF represents a distinct oscillatory mode
- Minimal frequency overlap between IMFs
- Progressive frequency ordering (low IMFs = high freq, high IMFs = low freq)

Mode mixing typically results from:
1. Premature sifting termination
2. Boundary condition artifacts
3. Signal characteristics (close frequency components)
4. Parameter misconfiguration

## Frequency-Domain Overlap Metric

Ferromode detects mode mixing using FFT-based power spectral density (PSD):

```
For each IMF pair (i, j):
  1. Compute FFT → Power Spectral Density (PSD)
  2. Normalize each spectrum to unit energy
  3. Compute intersection area = Σ_f min(PSD_i[f], PSD_j[f])
  4. Result is overlap score in [0.0, 1.0]
```

**Algorithm Complexity:** O(N × M log M) where N = IMFs, M = samples per IMF

## Interpretation Thresholds

| Overlap | Level | Interpretation | Action |
|---------|-------|---|---|
| < 0.10 | ✓ OK | Well-separated components | Continue with current parameters |
| 0.10–0.30 | ⚠️ Caution | Moderate mixing | Adjust sifting parameters |
| > 0.30 | 🚨 Serious | Significant mixing | Review sifting & boundaries |

### OK (overlap < 0.10)
- IMFs have distinct, non-overlapping frequency content
- Sifting converged properly
- Results are reliable for analysis

### Caution (overlap 0.10–0.30)
- Some IMFs share frequency content
- Decomposition partially reliable
- Consider adjusting EMD parameters

### Serious (overlap > 0.30)
- Major frequency overlap
- Decomposition has lost separation property
- Fundamental issue with sifting or signal

## Remediation Strategies

**For Caution-level mixing:**
1. Increase `minsiftiter` (minimum sifting iterations)
2. Try alternative boundary conditions (SYMMETRIC, PERIODIC, PREDICTION)
3. Verify signal preprocessing

**For Serious-level mixing:**
1. Review sifting stopping criterion
2. Check boundary condition implementation for artifacts
3. Try EEMD (Ensemble EMD) to reduce mixing via averaging
4. Verify signal validity and check for outliers

## API Usage

```rust
use ferromode::mode_mixing::*;

// Analyze mode mixing
let analysis = compute_mode_mixing_overlap(&imfs.imfs)?;
let heatmap = ModeMixingHeatmap::new(analysis);

// Print report
println!("{}", heatmap.generate_severity_report());

// Get interpretation
let interpretation = analyze_mode_mixing(&imfs.imfs)?;
println!("Severity: {:?}", interpretation.overall_severity);
```

## Visualization

Ferromode outputs JSON for visualization with matplotlib/seaborn:

```json
{
  "overlap_matrix": [[1.0, 0.15], [0.15, 1.0]],
  "max_overlap": 0.15,
  "mixed_pairs": [
    {"imf_idx_1": 0, "imf_idx_2": 1, "overlap": 0.15, "severity": "Caution"}
  ]
}
```

## References

- Flandrin et al. (2004). "Empirical Mode Decomposition as a Filter Bank"
- Wu & Huang (2009). "Ensemble Empirical Mode Decomposition"
- Rato et al. (2008). "On the HHT, Its Problems, and Some Solutions"
