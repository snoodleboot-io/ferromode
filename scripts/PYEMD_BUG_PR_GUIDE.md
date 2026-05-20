# Guide: Automating the PyEMD Bug Issue and Fix PR

## Prerequisites

```bash
# Install gh CLI if not present
# https://cli.github.com/

# Ensure you have a PyEMD fork or write access to laszukdawid/PyEMD
gh auth login
```

## Step 1: Fork the Repo

```bash
gh repo fork laszukdawid/PyEMD --clone --remote
cd PyEMD
```

## Step 2: Create a Fix Branch

```bash
git checkout -b fix/plateau-boundary-index-wrap
```

## Step 3: Apply the Fix

In `PyEMD/EMD.py`, find the plateau stripping check (search for `debs[0] == 1`).
Change:

```python
if len(debs) > 0 and debs[0] == 1:
```

to:

```python
if len(debs) > 0 and debs[0] <= 1:
```

One-liner patch (run from the `PyEMD/` repo root):

```bash
sed -i 's/if len(debs) > 0 and debs\[0\] == 1:/if len(debs) > 0 and debs[0] <= 1:/' PyEMD/EMD.py
```

Verify the change:

```bash
grep -n "debs\[0\]" PyEMD/EMD.py
```

## Step 4: Write a Regression Test

Add to `PyEMD/tests/test_emd.py`:

```python
def test_find_extrema_no_spurious_minimum_at_flat_start(self):
    """Flat opening segment must not produce a spurious minimum (index-0 wrap bug)."""
    S = np.array([0.5] * 8 + [1.0, 0.8, 0.5, 0.3, 0.1, 0.3, 0.5])
    T = np.arange(len(S))
    emd = EMD()
    result = emd.find_extrema(T, S)
    local_min_pos = result[2]
    # No minimum should fall in the flat region [0, 7]
    self.assertTrue(
        all(p > 7 for p in local_min_pos),
        f"Spurious minimum in flat start region: {local_min_pos}"
    )
```

Run the test suite:

```bash
pip install -e .
python -m pytest PyEMD/tests/test_emd.py -v
```

## Step 5: Commit

```bash
git add PyEMD/EMD.py PyEMD/tests/test_emd.py
git commit -m "fix: prevent spurious minimum at signal-start flat plateau

When a plateau begins at the very first sample (debs[0] == 0),
d[debs[0] - 1] wraps to d[-1] via Python negative indexing,
reading the slope at the end of the signal instead of the
beginning. This causes the plateau to be misclassified as a
local minimum.

Extend the existing strip guard from debs[0] == 1 to debs[0] <= 1
to cover this boundary case.

Fixes: plateau-at-index-0 produces spurious indmin entry
Verified against: MIT-BIH record 100 (PhysioNet, 360 Hz)
"
```

## Step 6: Push and Open PR

```bash
git push -u origin fix/plateau-boundary-index-wrap

gh pr create \
  --repo laszukdawid/PyEMD \
  --title "fix: prevent spurious minimum at signal-start flat plateau (index-0 wrap)" \
  --body "$(cat << 'EOF'
## Problem

When a signal begins with a flat (zero-diff) plateau, `find_extrema()` wraps
`d[debs[0] - 1]` to `d[-1]` via Python negative indexing when `debs[0] == 0`.
This reads the slope at the **end** of the signal, not the beginning, causing the
plateau to be incorrectly classified as a local minimum.

The existing guard (`if len(debs) > 0 and debs[0] == 1`) does not cover the
`debs[0] == 0` case.

## Fix

Change the guard from `== 1` to `<= 1`:

```python
# Before
if len(debs) > 0 and debs[0] == 1:

# After
if len(debs) > 0 and debs[0] <= 1:
```

## Verification

Tested on MIT-BIH Arrhythmia Database record 100 (PhysioNet, 360 Hz, 3600 samples).
The spurious minimum at sample 4 (midpoint of the 8-sample flat opening) is eliminated.
IMF dominant frequencies now match reference implementations.

Regression test added to `test_emd.py`.

## References

- MIT-BIH database: Moody & Mark (2001) *Circulation* 101(23):e215
- Huang et al. (1998) Proc. R. Soc. Lond. A 454, 903–995
EOF
)"
```

## Step 7: Open the Issue (optional, if PR not accepted immediately)

```bash
gh issue create \
  --repo laszukdawid/PyEMD \
  --title "Bug: spurious minimum inserted at signal-start flat plateau (index-0 wrap in find_extrema)" \
  --body-file /path/to/ferromode/scripts/PYEMD_BUG_REPORT.md \
  --label "bug"
```

## Files Produced by This Investigation

| File | Description |
|------|-------------|
| `gut_check_compare.py` | Runs both PyEMD and Ferromode, prints NRMS diff table |
| `gut_check_plot.py` | Generates the two PNG comparison plots |
| `gut_check_sidebyside.png` | Side-by-side IMF panels |
| `gut_check_overlay.png` | Overlaid IMFs with NRMS annotation |
| `PYEMD_BUG_REPORT.md` | Full bug report (use as issue body) |
| `PYEMD_BUG_PR_GUIDE.md` | This file |
