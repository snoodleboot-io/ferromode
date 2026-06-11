# PalindromeCyclic Boundary Condition

## Overview

The `PalindromeCyclic` boundary condition is Ferromode's most stable option for
suppressing end effects in EMD. It pre-extends the signal into a `2N-1`
even-symmetric palindrome before the sifting loop begins, then uses a periodic
(cyclic) cubic spline for all envelope interpolation. After decomposition, IMFs
and residue are trimmed back to the original `N` samples.

Exact reconstruction is preserved regardless of boundary strategy:
`Σ imf[:N] + residue[:N] == original_signal`.

---

## The End-Effect Problem

EMD builds upper and lower envelopes by fitting cubic splines through signal
extrema. Near the start and end of the signal the spline must extrapolate
beyond the outermost extrema — there is no data to constrain it. This produces
artifacts in the first and last IMFs known as **end effects**.

Standard mitigations reflect a short window at each boundary (`MirrorEven`,
`MirrorOdd`). These work well for zero-mean, near-periodic signals but
struggle when:

- The signal has a rising or falling trend (endpoints are far from zero)
- The signal has a large non-zero value at the boundary
- The first or last extremum is close to the boundary

`PalindromeCyclic` avoids these issues entirely by making the full signal
periodic before sifting starts.

---

## How It Works

```
Original (N = 8 samples)
  s₀  s₁  s₂  s₃  s₄  s₅  s₆  s₇

Palindrome (2N-1 = 15 samples)
  s₀  s₁  s₂  s₃  s₄  s₅  s₆  s₇  s₆  s₅  s₄  s₃  s₂  s₁  s₀
  └────────── original ──────────┘└─────── mirror (no duplicate endpoint) ─────┘
```

The palindrome is even-symmetric around `s_{N-1}`. This means:

1. Both endpoints equal `s₀` → the extended signal is naturally periodic
2. The derivative at the seam is zero by symmetry → C¹ continuity at the join

Sifting runs on the `2N-1` signal using `SplineType::Periodic` (cyclic cubic
spline), which correctly handles the periodicity. Within each sifting
iteration, `MirrorEven` is applied as a small stability margin at the
boundaries of the already-extended signal.

After all IMFs are extracted, each is truncated to `[:N]` samples.

---

## When to Use PalindromeCyclic

| Signal Characteristic | Recommended Strategy |
|---|---|
| Zero-mean, near-periodic | `MirrorEven` (default, fastest) |
| Truly periodic (circular data) | `Periodic` |
| Rising or falling trend | **`PalindromeCyclic`** |
| Large non-zero values at endpoints | **`PalindromeCyclic`** |
| Stochastic, known AR structure | `ARModel` |
| Quasi-periodic, matched waveform | `WaveformMatching` |
| Most stable choice needed | **`PalindromeCyclic`** |

**Rule of thumb:** if you observe spurious high-amplitude oscillations at the
start or end of IMF 1, switch to `PalindromeCyclic`.

---

## Code Examples

### Rust

```rust
use ferromode::algorithms::emd::{emd, EmdConfig};
use ferromode::boundary::BoundaryConditionType;
use ferromode::sifting::SiftingConfig;

let config = EmdConfig {
    sifting_config: SiftingConfig {
        boundary_condition: BoundaryConditionType::PalindromeCyclic,
        ..SiftingConfig::default()
    },
    validate_reconstruction: true,
    reconstruction_tolerance: 1e-10,
    ..EmdConfig::default()
};

let result = emd(&signal, &config)?;

// All IMFs and residue are trimmed to signal.len() — no extra handling needed.
for imf in &result.imfs.imfs {
    assert_eq!(imf.len(), signal.len());
}
```

### Python

```python
import ferromode

result = ferromode.emd(
    signal,
    boundary_condition="palindrome_cyclic",
)
```

Streaming decomposer:

```python
decomposer = ferromode.StreamingDecomposer(
    chunk_size=512,
    boundary_condition="palindrome_cyclic",
)
```

### JavaScript / WASM

```javascript
import init, { emd } from 'ferromode';
await init();

const result = emd(signal, {
  boundary_condition: 'PalindromeCyclic',
});
```

### MATLAB / Octave (MEX)

```matlab
result = ferromode_emd(signal, struct( ...
    'boundary_condition', 'palindrome', ...
    'max_imfs', 0 ...
));
```

### C++

```cpp
#include "ferromode.h"

CEmdConfig config = ferromode_default_emd_config();
config.boundary_condition = 7;  // PalindromeCyclic
CDecompositionResult* result = ferromode_emd(signal, n, &config);
```

---

## SplineType Control (Advanced)

`PalindromeCyclic` selects `SplineType::Periodic` automatically. You can also
control the spline type independently for other boundary strategies:

```rust
use ferromode::spline::SplineType;

let config = EmdConfig {
    sifting_config: SiftingConfig {
        boundary_condition: BoundaryConditionType::MirrorEven,
        spline_type: SplineType::NotAKnot,  // override spline type
        ..SiftingConfig::default()
    },
    ..EmdConfig::default()
};
```

| SplineType | Endpoint constraint | Use when |
|---|---|---|
| `Natural` (default) | Second derivative = 0 at endpoints | Most signals |
| `Periodic` | First and second derivatives match at endpoints | Signal is periodic or palindrome-extended |
| `NotAKnot` | C³ continuity at first/last interior knots | Smooth signals, no artificial boundary constraint |

---

## The `build_palindrome` Utility

The palindrome pre-processing is also available as a standalone function:

```rust
use ferromode::boundary::build_palindrome;

let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let palindrome = build_palindrome(&signal);
// [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
assert_eq!(palindrome.len(), 2 * signal.len() - 1);
```

---

## Performance

`PalindromeCyclic` operates on a `2N-1` sample signal internally, so each
sifting iteration is approximately twice as expensive as `MirrorEven`. For most
signals this overhead is acceptable. Boundary artifacts are typically more
expensive to deal with after the fact than to prevent here.

| Strategy | Relative cost | Stability |
|---|---|---|
| `MirrorEven` | 1× (baseline) | Good for periodic/zero-mean |
| `ARModel` | ~1.2× | Good for stochastic |
| `PalindromeCyclic` | ~2× | Best for general signals |

---

## Running the Example

A complete demonstration is included:

```bash
cargo run --example palindrome_cyclic_boundary --release
```

This example:
1. Runs EMD on a pure sine wave with `PalindromeCyclic`
2. Stress-tests a signal with a rising trend + fast oscillation (hard boundary case)
3. Compares boundary RMS of IMF endpoints against `MirrorEven`
4. Demonstrates `max_imfs` limiting with correct output lengths
5. Shows manual `SplineType` control independent of boundary strategy

---

## See Also

- [EMD Algorithm Reference](algorithms/emd.md)
- [Boundary Condition Comparison](algorithms/comparison-table.md)
- [Boundary Strategy Decision Tree](algorithms/decision-tree.md)
- `palindrome_cyclic_boundary` example (`crates/ferromode/examples/`)
