# Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (ICEEMDAN)

## Mathematical Description

ICEEMDAN improves upon CEEMDAN by using the **$k$-th IMF of a noise-only signal**
instead of raw white noise at each stage. This produces cleaner IMFs with less
residual noise.

### Algorithm

1. **Pre-compute**: Run EMD on pure noise signals to get noise IMFs for each stage.
   For each trial $j$, decompose pure noise $n_j(t)$ to get IMFs $\text{IMF}_k(n_j)$.

2. **Stage 0**:
   - $x_j(t) = x(t) + \varepsilon \cdot \text{IMF}_1(n_j(t))$
   - $\bar{c}_1(t) = \frac{1}{M} \sum_{j=1}^{M} \text{EMD}_1(x_j(t))$

3. **Stage $k$** ($k \geq 1$):
   - $\varepsilon_k = \varepsilon \cdot \text{std}(r_k) / \text{std}(n)$
   - $r_{k,j}(t) = r_k(t) + \varepsilon_k \cdot \text{IMF}_k(n_j(t))$
   - $\bar{c}_{k+1}(t) = \frac{1}{M} \sum_{j=1}^{M} \text{EMD}_1(r_{k,j}(t))$

4. **Update residue**: $r_{k+1}(t) = r_k(t) - \bar{c}_{k+1}(t)$

### Key Difference from CEEMDAN

| CEEMDAN | ICEEMDAN |
|---------|----------|
| Adds raw white noise at each stage | Adds $k$-th IMF of noise at stage $k$ |
| More residual noise in IMFs | Cleaner IMFs with less noise |
| Faster to compute | Slower (pre-computation of noise IMFs) |

### Advantages

- Less residual noise in IMFs compared to CEEMDAN
- Better mode separation
- Complete reconstruction with negligible error

### Complexity

- **Time**: $O(K \cdot M \cdot n \cdot s)$ — similar to CEEMDAN but with
  pre-computation overhead for noise IMFs
- **Space**: $O(K \cdot M \cdot n)$ — stores pre-computed noise IMFs

## References

- Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014).
  "Improved Complete Ensemble EMD: A Suitable Technique for Biomedical
  Signal Processing." *Biomedical Signal Processing and Control*, 14, 19-29.
