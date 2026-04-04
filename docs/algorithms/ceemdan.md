# Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN)

## Mathematical Description

CEEMDAN improves upon EEMD by adding noise **stage-wise** rather than to the
original signal. At each stage $k$, adaptive noise is added to the current
residue, and only the first IMF is extracted via EMD.

### Algorithm

1. **Stage 0**: For each trial $j$:
   - $x_j(t) = x(t) + \varepsilon_0 \cdot n_j(t)$
   - Extract first IMF: $c_{1,j} = \text{EMD}_1(x_j(t))$
   - Average: $\bar{c}_1(t) = \frac{1}{M} \sum_{j=1}^{M} c_{1,j}(t)$

2. **Compute residue**: $r_1(t) = x(t) - \bar{c}_1(t)$

3. **Stage $k$** ($k \geq 1$): For each trial $j$:
   - Adaptive noise scale: $\varepsilon_k = \varepsilon \cdot \text{std}(r_k) / \text{std}(n)$
   - $r_{k,j}(t) = r_k(t) + \varepsilon_k \cdot n_j(t)$
   - Extract first IMF: $c_{k+1,j} = \text{EMD}_1(r_{k,j}(t))$
   - Average: $\bar{c}_{k+1}(t) = \frac{1}{M} \sum_{j=1}^{M} c_{k+1,j}(t)$

4. **Update residue**: $r_{k+1}(t) = r_k(t) - \bar{c}_{k+1}(t)$

5. Repeat until residue has < 2 extrema

### Key Advantages

- **Complete reconstruction**: $x(t) = \sum_i \bar{c}_i(t) + r_{\text{final}}(t)$
  with negligible error
- **No mode mixing**: Noise is added adaptively at each stage
- **Better spectral separation** than EEMD

### Parameters

Same as EEMD: `num_ensembles`, `noise_std`, `seed`.

### Complexity

- **Time**: $O(K \cdot M \cdot n \cdot s)$ where $K$ = number of stages (IMFs)
- **Space**: $O(n \cdot k)$

## References

- Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P. (2011).
  "A Complete Ensemble Empirical Mode Decomposition with Adaptive Noise."
  *IEEE International Conference on Acoustics, Speech and Signal Processing
  (ICASSP)*, 4144-4147.
