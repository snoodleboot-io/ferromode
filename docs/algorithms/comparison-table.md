# Algorithm Comparison

| Property | EMD | EEMD | CEEMD | CEEMDAN | ICEEMDAN | VMD | MEMD | NA-MEMD |
|----------|-----|------|-------|---------|----------|-----|------|---------|
| **Signal Type** | Univariate | Univariate | Univariate | Univariate | Univariate | Univariate | Multivariate | Multivariate |
| **Mode Mixing** | Yes | Reduced | Reduced | No | No | No | Reduced | Reduced |
| **Reconstruction** | Good | Approximate | Approximate | Exact | Exact | Exact | Good | Good |
| **Speed** | Fast | Slow | Slower | Medium | Medium | Fast | Medium | Medium |
| **Parameters** | Few | Few+M | Few+M | Few+M | Few+M | K,α,τ | Direction | Dir+Noise |
| **Noise Robust** | No | Yes | Yes | Yes | Yes | Yes | Partial | Yes |
| **Recursive** | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes |
| **Adaptive** | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes |
| **Parallelizable** | No | Yes | Yes | Yes | Yes | Yes | Partial | Partial |

## Parameter Summary

| Algorithm | Required Parameters | Tuning Difficulty |
|-----------|-------------------|------------------|
| EMD | `sd_threshold`, `s_number` | Easy |
| EEMD | + `num_ensembles`, `noise_std` | Medium |
| CEEMD | Same as EEMD | Medium |
| CEEMDAN | Same as EEMD | Medium |
| ICEEMDAN | Same as EEMD | Medium |
| VMD | `n_modes`, `alpha`, `tau` | Hard |
| MEMD | `num_directions` | Easy |
| NA-MEMD | + `n_noise_channels`, `noise_std` | Medium |

## Boundary Condition Comparison

| Strategy | Extension Method | Spline Type | Relative Speed | Best For |
|----------|-----------------|-------------|----------------|----------|
| **MirrorEven** | Reflect signal at endpoints | Natural | 1× (baseline) | Zero-mean, symmetric signals |
| **MirrorOdd** | Antisymmetric reflection | Natural | 1× | Signals with zero endpoints |
| **Periodic** | Wrap-around | Periodic | 1× | Genuinely periodic signals |
| **Slope** | Linear extrapolation from endpoint slope | Natural | 1× | Smooth, slowly varying signals |
| **AR Model** | Autoregressive prediction | Natural | ~1.5× | Stationary or near-stationary |
| **Characteristic Wave** | Wave-based extension | Natural | ~1.5× | Oscillatory signals |
| **Waveform Matching** | Pattern-matched extension | Natural | ~2× | Quasi-periodic signals |
| **PalindromeCyclic** | Even-symmetric palindrome (2N-1) + periodic spline | Periodic | ~2× | Signals with trends or non-zero endpoints |

PalindromeCyclic is the most stable option when signals have a non-zero mean,
a trend, or non-zero endpoint values. The palindrome pre-extension eliminates
the discontinuity that other strategies must handle by extrapolation, and exact
reconstruction is guaranteed.

## When to Choose Each

- **EMD**: Baseline, fast, good for clean signals
- **EEMD**: When EMD shows mode mixing, acceptable slowdown
- **CEEMD**: Better noise cancellation than EEMD with same parameters
- **CEEMDAN**: Best balance of quality and speed for noisy signals
- **ICEEMDAN**: Highest quality, biomedical/precision applications
- **VMD**: Known number of modes, need non-recursive decomposition
- **MEMD**: Multiple correlated channels, need mode alignment
- **NA-MEMD**: MEMD with improved mode separation via noise assistance
