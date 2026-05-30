# PyEMD Bug: Spurious Minimum Inserted at Signal-Start Plateau via Off-by-One Index Wrap

## Summary

PyEMD's `find_extrema()` contains an off-by-one error in its flat-plateau detection
that causes it to misclassify the **opening flat region of a signal** as a local minimum.
The midpoint of that region is inserted into `indmin`, producing a spurious extremum
that does not exist in the signal. This corrupts every subsequent sifting iteration
and the resulting IMF frequency content.

## Affected File

`PyEMD/EMD.py` — `find_extrema()` method, plateau handling block.

## Reproduction

```python
import numpy as np
from PyEMD import EMD

# MIT-BIH Arrhythmia Database record 100 (PhysioNet, 360 Hz)
# First 8 samples are flat: all 0.17492 mV
import wfdb, os
wfdb.dl_database("mitdb", dl_dir="/tmp/mitdb", records=["100"])
rec = wfdb.rdrecord("/tmp/mitdb/100", sampfrom=0, sampto=3600, channels=[0])
sig = rec.p_signal[:, 0].astype(float) - rec.p_signal[:, 0].mean()

emd = EMD()
T   = np.arange(len(sig))
result = emd.find_extrema(T, sig)
local_min_pos = result[2]

print("First interior minimum index:", local_min_pos[0])  # prints 4
print("Signal values 0-8:", sig[0:9])                     # all equal for 0-7
```

**Expected:** no minimum in the flat region `[0, 7]` — all values identical.  
**Actual:** minimum inserted at index `4` (midpoint of plateau).

## Root Cause

In the plateau handling block (around line 636 of `EMD.py`):

```python
d = np.diff(S)
bad = d == 0
dd = np.diff(np.concatenate(([0], bad, [0])))
debs = np.nonzero(dd == 1)[0]   # plateau start indices into d
fins = np.nonzero(dd == -1)[0]  # plateau end indices into d

# ...strip leading check (debs[0] == 1, NOT debs[0] == 0)...

if len(debs) > 0:
    d_before = d[debs - 1]   # <-- BUG when debs[0] == 0
    d_after  = d[fins]
```

When a plateau begins at the very first sample (`debs[0] == 0`), `d[debs[0] - 1]`
becomes `d[-1]` — Python's negative-index wrap-around returns the **last element**
of the diff array (the slope at the *end* of the signal). If that value is negative,
the condition `d_before < 0 and d_after > 0` is satisfied and a spurious minimum
is recorded at the midpoint of the plateau.

**The strip check does not catch this:** the guard condition is `debs[0] == 1`
(plateau starting at `d[1]`, i.e. second sample), not `debs[0] == 0`.

## Fix

Add a guard for `debs[0] == 0` alongside the existing `debs[0] == 1` check:

```python
# Before (line ~649):
if len(debs) > 0 and debs[0] == 1:

# After:
if len(debs) > 0 and debs[0] <= 1:
```

Or equivalently, clamp `d_before` for the boundary case:

```python
d_before = np.where(debs > 0, d[debs - 1], 0.0)
```

## Impact

- Signals with a flat (quantised or DC-offset) opening segment receive one spurious
  minimum at the plateau midpoint.
- This shifts the lower envelope upward near the left boundary.
- All IMFs extracted from the signal carry a frequency error traceable to this first
  sifting iteration; dominant frequency of IMF1 was measured 27.9 Hz (PyEMD) vs
  15.1 Hz (Ferromode, which correctly excludes the spurious extremum).
- The effect propagates through the entire decomposition cascade.

## Dataset Used for Verification

- **MIT-BIH Arrhythmia Database**, Record 100, Lead MLII
- PhysioNet: https://physionet.org/content/mitdb/1.0.0/
- 3600 samples at 360 Hz (10 seconds)
- Reference: Moody & Mark (2001) *Circulation* 101(23):e215–e220

## PyEMD Version

Tested against `emd-signal 1.9.0` (PyPI package `EMD-signal`).

## Related

- Original EMD paper: Huang et al. (1998) Proc. R. Soc. Lond. A 454, 903–995
- Comparison plots: `gut_check_sidebyside.png`, `gut_check_overlay.png`
