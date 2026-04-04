# Ensemble Empirical Mode Decomposition (EEMD)

## Mathematical Description

EEMD addresses the **mode mixing** problem in EMD by adding white noise to the
signal and averaging across multiple trials:

$$\bar{c}_i(t) = \frac{1}{M} \sum_{j=1}^{M} c_{i,j}(t)$$

where $c_{i,j}(t)$ is the $i$-th IMF from the $j$-th trial, and $M$ is the
number of ensemble trials.

### Algorithm

1. Add Gaussian white noise $n_j(t)$ with standard deviation $\varepsilon$ to the signal:
   $x_j(t) = x(t) + \varepsilon \cdot n_j(t)$
2. Run EMD on $x_j(t)$ to obtain IMFs $c_{i,j}(t)$
3. Repeat for $M$ trials with different noise realizations
4. Average corresponding IMFs across all trials

### Noise Properties

The added noise populates the whole time-frequency space uniformly. Because white
noise has zero mean, the ensemble average cancels the added noise while preserving
the signal structure:

$$\lim_{M \to \infty} \frac{1}{M} \sum_{j=1}^{M} \varepsilon \cdot n_j(t) = 0$$

### Parameters

- **num_ensembles** ($M$): Number of trials. Recommended: 50-200. Wu & Huang
  recommend 100.
- **noise_std** ($\varepsilon$): Noise amplitude as fraction of signal std.
  Typical: 0.01 to 0.4. Recommended: 0.2.
- **seed**: Optional RNG seed for reproducibility.

### Complexity

- **Time**: $O(M \cdot n \cdot k \cdot s)$ — $M$ times slower than EMD
- **Space**: $O(n \cdot k)$ — same as EMD (IMFs averaged incrementally)

## References

- Wu, Z., & Huang, N. E. (2009). "Ensemble Empirical Mode Decomposition:
  A Noise-Assisted Data Analysis Method." *Advances in Adaptive Data Analysis*,
  1(1), 1-41.
