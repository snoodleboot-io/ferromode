# Variational Mode Decomposition (VMD)

## Mathematical Description

VMD decomposes a signal into a set of band-limited modes by solving a
**variational optimization problem** using the Alternating Direction Method
of Multipliers (ADMM) in the frequency domain.

### Optimization Problem

$$\min_{\{u_k\}, \{\omega_k\}} \left\{ \sum_k \left\| \partial_t \left[ \left(\delta(t) + \frac{j}{\pi t}\right) * u_k(t) \right] e^{-j\omega_k t} \right\|_2^2 \right\}$$

subject to: $\sum_k u_k(t) = f(t)$

where:
- $u_k(t)$ are the modes (band-limited signals)
- $\omega_k$ are the center frequencies
- $\alpha$ is the bandwidth penalty parameter

### ADMM Algorithm

The constrained problem is solved iteratively:

1. **Mode update** (Wiener filtering in frequency domain):
   $$\hat{u}_k^{n+1}(\omega) = \frac{\hat{f}(\omega) - \sum_{l \neq k} \hat{u}_l(\omega) + \frac{\hat{\lambda}(\omega)}{2}}{1 + 2\alpha(\omega - \omega_k)^2}$$

2. **Center frequency update** (spectral centroid):
   $$\omega_k^{n+1} = \frac{\int_0^\infty \omega |\hat{u}_k(\omega)|^2 d\omega}{\int_0^\infty |\hat{u}_k(\omega)|^2 d\omega}$$

3. **Lagrange multiplier update** (dual ascent):
   $$\hat{\lambda}^{n+1}(\omega) = \hat{\lambda}^n(\omega) + \tau \left(\hat{f}(\omega) - \sum_k \hat{u}_k^{n+1}(\omega)\right)$$

4. **Convergence**: Stop when $\sum_k \|u_k^{n+1} - u_k^n\|_2^2 / \|u_k^n\|_2^2 < \varepsilon$

### Parameters

- `n_modes` ($K$): Number of modes to extract
- `alpha` ($\alpha$): Bandwidth penalty. Higher = narrower modes. Default: 2000
- `tau` ($\tau$): Dual ascent step size. 0 = strict constraint. Default: 0
- `tol` ($\varepsilon$): Convergence tolerance. Default: 1e-7
- `max_iterations`: Maximum ADMM iterations. Default: 500

### Advantages over EMD

- **Non-recursive**: All modes extracted simultaneously
- **Well-defined**: Solves a clear optimization problem
- **Robust**: Not sensitive to noise or sampling
- **Efficient**: FFT-based, converges in tens of iterations

### Complexity

- **Time**: $O(K \cdot I \cdot n \log n)$ where $I$ = iterations to converge
- **Space**: $O(K \cdot n)$

## References

- Dragomiretskiy, K., & Zosso, D. (2014). "Variational Mode Decomposition."
  *IEEE Transactions on Signal Processing*, 62(3), 531-544.
