# Multivariate Empirical Mode Decomposition (MEMD)

## Mathematical Description

MEMD extends EMD to $n$-variate signals by projecting the multivariate signal
onto direction vectors on the $n$-sphere and averaging the results.

### Algorithm

1. **Generate direction vectors**: Sample $K$ direction vectors $\mathbf{x}^\theta$
   uniformly on the $(n-1)$-sphere using a Hammersley sequence.

2. **Project signal**: For each direction $\mathbf{x}^\theta$, compute the
   1D projection: $p^\theta(t) = \mathbf{s}(t) \cdot \mathbf{x}^\theta$

3. **Find extrema**: Detect extrema of each projection $p^\theta(t)$

4. **Interpolate envelopes**: For each channel, interpolate envelopes through
   the extrema positions of each projection

5. **Compute local mean**: Average envelopes across all directions:
   $$\mathbf{m}(t) = \frac{1}{K} \sum_{\theta=1}^{K} \mathbf{e}^\theta(t)$$

6. **Sift**: $\mathbf{h}(t) = \mathbf{s}(t) - \mathbf{m}(t)$, repeat until
   multivariate stopping criterion is met

### Direction Sampling

Direction vectors are generated using:
- **Hammersley sequence**: Low-discrepancy sequence for uniform coverage
- **Number of directions**: Typically 8-64 depending on dimensionality

### Mode Alignment

A key property of MEMD is **mode alignment**: corresponding IMFs across
channels represent the same oscillatory mode, which is not guaranteed by
applying EMD independently to each channel.

### Parameters

- `num_directions`: Number of direction vectors (default: 16)
- `max_imfs`: Maximum IMFs to extract (0 = auto)
- `sifting_config`: Sifting parameters

### Complexity

- **Time**: $O(n \cdot k \cdot s \cdot K)$ where $K$ = number of directions
- **Space**: $O(n \cdot k \cdot L)$ where $L$ = signal length per channel

## References

- Rehman, N., & Mandic, D. P. (2010). "Multivariate Empirical Mode
  Decomposition." *Proceedings of the Royal Society A*, 466(2117), 1291-1302.
