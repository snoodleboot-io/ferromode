# Intermittency Handling in Ferromode: Current State & Future Roadmap

This document clarifies which algorithms in Ferromode handle intermittency/non-stationarity, and what is planned for future iterations.

## Current State: What We've Implemented (v1.x)

Ferromode v1.x already includes several algorithms specifically designed to handle intermittency and non-stationarity in signals:

### ✅ Ensemble Methods (M3: Ensemble Methods Beta)
These algorithms combat mode mixing caused by intermittency through noise-assisted trials:

1. **EEMD** (Ensemble Empirical Mode Decomposition) - T-067 to T-071
   - Adds Gaussian white noise to signal, performs EMD multiple times, averages results
   - Effective for signals with intermittent components
   - Reference: Wu & Huang (2009)

2. **CEEMD** (Complementary Ensemble EMD) - T-072 to T-074  
   - Uses paired noise trials (+ε and -ε) for better noise cancellation than EEMD
   - Superior performance for highly intermittent signals
   - Reference: Yeh et al. (2010)

3. **CEEMDAN** (Complete Ensemble EMD with Adaptive Noise) - T-075 to T-080
   - Stage-wise noise adaptation: noise amplitude scales with residual standard deviation
   - Further reduces mode mixing and improves IMF separation
   - Reference: Torres et al. (2011)

4. **ICEEMDAN** (Improved CEEMDAN) - T-081 to T-083
   - Uses EMD of pure noise to generate more appropriate noise-assisted trials
   - Best-in-class for separating closely spaced frequencies in non-stationary signals
   - Reference: Colominas et al. (2014)

### ✅ Boundary Condition Strategies (M1: Spline & Boundary Foundation)
Proper boundary handling reduces end effects that can masquerade as intermittency:

- Characteristic Wave (Huang et al. 1998) - T-029/T-030/T-031
- Mirror/Even-Odd extension - T-032/T-033/T-034  
- Periodic/Cyclical extension (Zeng & He 2004) - T-035/T-036/T-037/T-038
- Slope-based extrapolation - T-039/T-040/T-041
- AR model extrapolation - T-042/T-043/T-044/T-045
- Waveform matching/cross-correlation extension - T-046/T-047/T-048/T-049

### ✅ Intermittency Detection (Pre-check) - T-064
Optional pre-decomposition intermittency test based on extrema spacing variability (coefficient of variation).

## Planned for Future Work (v2.x)

While the core ensemble methods already handle intermittency well, future work will focus on:

### 1. Advanced Noise-Assisted Methods
- **Adaptive Noise EMD (AN-EMD)**: Noise characteristics adapt based on local signal properties
- **GA-Optimized EEMD**: Genetic algorithm to determine optimal noise amplitude and trial count
- **PSO-CEEMDAN**: Particle swarm optimization for CEEMDAN parameter tuning
- **ICA-Based Noise Generation**: Use independent components as noise sources instead of Gaussian

### 2. Hybrid Decomposition Frameworks  
- **EMD-SVR**: Use Support Vector Regression to extract trends before EMD, reducing intermittency effects
- **Wavelet-EMD Hybrid**: Wavelet preprocessing to isolate oscillatory components
- **EEMD + Complementary Ensemble Lag**: Time-delayed noise trials for better phase preservation
- **Adaptive Basis EMD**: Data-driven basis functions instead of fixed sifting

### 3. Multivariate Extensions for Intermittency
- **JADE-MEMD**: Joint Approximate Diagonalization of Eigenvalues for multivariate EMD
- **MUSIC-Assisted MEMD**: Use MUSIC algorithm to guide direction selection in MEMD
- **Copula-Based MEMD**: Model dependence structure between channels with copulas

### 4. Machine Learning Approaches
- **LSTM-Predicted Boundary Conditions**: Neural networks predict optimal signal extensions
- **IMF Selection Networks**: Learn which IMFs to retain for downstream tasks
- **Differentiable EMD Layers**: End-to-end trainable EMD for deep learning pipelines
- **Reinforcement Learning for Sifting**: Learn optimal stopping policies per signal type

### 5. Advanced Signal Models
- **Fractional EMD (FEMD)**: Incorporate fractional calculus for better handling of power-law signals
- **HT-EMD**: Hilbert Transform-guided EMD for improved frequency separation
- **EEMD based on Local Characteristic Wave Decomposition (LCWD)**: Local adaptive basis functions
- **EMD with Shapley Value Attribution**: Fair attribution of signal components to IMFs

### 6. Real-Time & Adaptive Processing
- **Online Intermittency Detection**: Sliding window coefficient of variation monitoring
- **Adaptive Algorithm Selection**: Switch between EMD/EEMD/CEEMDAN based on real-time intermittency metrics
- **Buffer-Adaptive Processing**: Variable chunk sizes based on detected stationarity
- **Predictive Boundary Extension**: Use ARIMA/LSTM to predict boundary conditions for streaming data

### 7. Validation & Benchmarking
- **Intermittency-Specific Test Suite**: Standardized signals with known intermittency properties
- **Cross-Algorithm Comparison**: Benchmark EEMD/CEEMDAN/ICEEMDAN/AN-EMD/etc. on intermittent signals
- **Physiological Signal Validation**: ECG, EEG, fMRI applications with known intermittent events
- **Financial Time Series Testing**: Volatility clustering and regime-change detection

## Relationship to Existing Features

The user's confusion may stem from conflating two different concepts:

1. **Intermittency Detection** (what we have as T-064): A pre-check that measures signal non-stationarity via extrema spacing variability
2. **Intermittency Handling Algorithms** (what we have in M3): Algorithms like EEMD/CEEMDAN/ICEEMDAN that are specifically designed to decompose intermittent signals effectively

Ferromode v1.x already implements **#2** (the algorithms that handle intermittency) and has a basic version of **#1** (the detection mechanism).

Future work will enhance both:
- More sophisticated intermittency detection (using higher-order statistics, machine learning)
- More advanced intermittency-handling algorithms (beyond the basic ensemble methods)
- Better integration between detection and algorithm selection/adaptation

## Immediate Next Steps (Post v1.0)

For v1.x maintenance and polishing:
1. Add intermittency detection to all language bindings (currently only in Rust core)
2. Expose intermittency test results in DecompositionResult across all bindings
3. Add cross-language validation tests for intermittency detection
4. Consider making intermittency check run during decomposition (not just pre-check) for adaptive sifting

For v2.x planning:
The intermittency handling algorithms listed above represent the primary frontier for advancement beyond the solid foundation laid in v1.x.

---
*This document clarifies that intermittency handling is already well-addressed in v1.x through the ensemble methods (EEMD, CEEMD, CEEMDAN, ICEEMDAN), with future work focusing on advanced variants and integration techniques.*