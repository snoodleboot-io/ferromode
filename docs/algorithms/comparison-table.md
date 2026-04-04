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

## When to Choose Each

- **EMD**: Baseline, fast, good for clean signals
- **EEMD**: When EMD shows mode mixing, acceptable slowdown
- **CEEMD**: Better noise cancellation than EEMD with same parameters
- **CEEMDAN**: Best balance of quality and speed for noisy signals
- **ICEEMDAN**: Highest quality, biomedical/precision applications
- **VMD**: Known number of modes, need non-recursive decomposition
- **MEMD**: Multiple correlated channels, need mode alignment
- **NA-MEMD**: MEMD with improved mode separation via noise assistance
