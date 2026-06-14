# Empirical Mode Decomposition (EMD)

## Mathematical Description

EMD decomposes a signal $x(t)$ into a finite set of Intrinsic Mode Functions (IMFs)
plus a residue:

$$x(t) = \sum_{i=1}^{n} c_i(t) + r_n(t)$$

where $c_i(t)$ are the IMFs and $r_n(t)$ is the final residue.

### Sifting Process

Each IMF is extracted through an iterative sifting process:

1. **Detect extrema**: Find all local maxima and minima of the signal
2. **Construct envelopes**: Interpolate upper envelope through maxima and lower
   envelope through minima using cubic splines
3. **Compute mean**: $m(t) = \frac{u(t) + l(t)}{2}$
4. **Extract detail**: $h(t) = x(t) - m(t)$
5. **Check stopping criterion**: If $h(t)$ satisfies the IMF conditions, stop;
   otherwise repeat with $h(t)$ as the new signal

### IMF Conditions

A function $c(t)$ is an IMF if:

1. The number of extrema and zero-crossings differ by at most one
2. The mean value of the upper and lower envelopes is zero at every point

### Stopping Criteria

- **SD Threshold**: Stop when $\text{SD} = \frac{\sum |h_{k-1}(t) - h_k(t)|^2}{\sum |h_{k-1}(t)|^2} < \epsilon$
- **S-Number**: Stop after $S$ consecutive iterations where extrema and zero-crossings
  differ by at most 1
- **Energy Difference**: Stop when relative energy change falls below threshold

## Boundary Conditions

EMD suffers from end effects during envelope interpolation. Ferromode supports:

- **Mirror Even/Odd**: Reflect signal at boundaries
- **Periodic**: Assume signal is periodic
- **Slope**: Extend using slope at endpoints
- **AR Model**: Autoregressive extrapolation
- **Characteristic Wave**: Wave-based boundary extension
- **Waveform Matching**: Match waveform patterns at boundaries
- **PalindromeCyclic**: Pre-extends the signal into a `2N-1` even-symmetric
  palindrome before sifting begins, then uses a periodic (cyclic) cubic spline
  for all envelope interpolation throughout the sifting loop. IMFs and residue
  are trimmed back to the original N samples after decomposition. This is the
  most stable option for signals with trends or non-zero endpoints because the
  palindrome extension guarantees C¹ continuity at the seam; exact
  reconstruction is preserved. It is approximately 2× slower than MirrorEven
  because sifting operates on the 2N-1 extended signal internally.

## Spline Types

The cubic spline used for envelope interpolation is controlled by the
`SplineType` field on `SiftingConfig`:

- **Natural** (default): Zero second derivative at endpoints. Standard choice
  for most boundary conditions.
- **Periodic**: First and second derivatives match at the two endpoints.
  Selected automatically when `PalindromeCyclic` is active; can also be set
  manually.
- **NotAKnot**: C³ continuity enforced at the first and last interior knots.
  Useful when the signal is smooth and you want a less constrained spline near
  the boundaries.

## Complexity

- **Time**: $O(n \cdot k \cdot s)$ where $n$ = signal length, $k$ = number of IMFs,
  $s$ = average sifting iterations per IMF
- **Space**: $O(n \cdot k)$ to store all IMFs

## References

- Huang, N. E., et al. (1998). "The empirical mode decomposition and the Hilbert
  spectrum for nonlinear and non-stationary time series analysis."
  *Proceedings of the Royal Society of London A*, 454(1971), 903-995.
