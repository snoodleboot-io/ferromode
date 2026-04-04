# Complementary Ensemble Empirical Mode Decomposition (CEEMD)

## Mathematical Description

CEEMD improves upon EEMD by using **complementary noise pairs**. For each trial $j$,
it computes both $\text{EMD}(x(t) + \varepsilon \cdot n_j(t))$ and
$\text{EMD}(x(t) - \varepsilon \cdot n_j(t))$, then averages the pair:

$$\bar{c}_i(t) = \frac{1}{2M} \sum_{j=1}^{M} \left[c_{i,j}^+(t) + c_{i,j}^-(t)\right]$$

where $c_{i,j}^+$ and $c_{i,j}^-$ are IMFs from the positive and negative noise trials.

### Why Complementary Pairs?

In EEMD, the residual noise in the averaged IMFs decreases as $1/\sqrt{M}$. CEEMD
achieves faster noise cancellation because the positive and negative noise components
tend to cancel out more effectively when paired:

$$\varepsilon \cdot n_j(t) + (-\varepsilon \cdot n_j(t)) = 0$$

This means CEEMD can achieve the same noise reduction with fewer ensemble trials
than EEMD.

### Algorithm

1. For each trial $j = 1, \ldots, M$:
   - Generate noise $n_j(t)$
   - Compute $c_{i,j}^+ = \text{EMD}(x(t) + \varepsilon \cdot n_j(t))$
   - Compute $c_{i,j}^- = \text{EMD}(x(t) - \varepsilon \cdot n_j(t))$
   - Average pair: $\bar{c}_{i,j} = \frac{1}{2}(c_{i,j}^+ + c_{i,j}^-)$
2. Average all pair results: $\bar{c}_i = \frac{1}{M} \sum_{j=1}^{M} \bar{c}_{i,j}$

### Parameters

Same as EEMD: `num_ensembles`, `noise_std`, `seed`.

### Complexity

- **Time**: $O(2M \cdot n \cdot k \cdot s)$ — 2x EEMD (each trial runs twice)
- **Space**: $O(n \cdot k)$ — same as EEMD

## References

- Yeh, J.-R., Huang, N. E., & Wu, Z. (2010). "Complementary Ensemble Empirical
  Mode Decomposition: A Novel Noise Assisted Data Analysis Method."
  *Advances in Adaptive Data Analysis*, 2(4), 417-431.
