#!/usr/bin/env python3
"""
Compare first-iteration envelope from PyEMD vs ferromode on MIT-BIH signal.
Prints the mean envelope for the first 20 samples so we can see where they diverge.
"""
import numpy as np
from PyEMD import EMD
from scipy.interpolate import CubicSpline

sig = np.loadtxt("data/mitdb/signal.csv")
T   = np.arange(len(sig))

# ── PyEMD first-iteration envelope ───────────────────────────────────────────
emd = EMD()
upper_py, lower_py, max_ext, min_ext = emd.extract_max_min_spline(T, sig)
mean_py = (upper_py + lower_py) / 2

print("PyEMD knot positions (maxima, first 6):", max_ext[0][:6])
print("PyEMD knot values    (maxima, first 6):", max_ext[1][:6].round(5))
print()
print(f"{'i':>4}  {'sig':>10}  {'upper_py':>10}  {'lower_py':>10}  {'mean_py':>10}")
for i in range(20):
    print(f"{i:>4}  {sig[i]:>10.5f}  {upper_py[i]:>10.5f}  {lower_py[i]:>10.5f}  {mean_py[i]:>10.5f}")

# ── Reproduce ferromode knots: same interior extrema + reflected boundary ─────
def find_maxima(s):
    m = []
    i = 1
    while i < len(s) - 1:
        if s[i] > s[i-1]:
            ps = i
            while i < len(s)-1 and s[i] == s[i+1]:
                i += 1
            if i < len(s)-1 and s[i] > s[i+1]:
                m.append((ps+i)//2)
            i += 1; continue
        i += 1
    if s[0] > s[1]:   m = [0] + m
    if s[-1] > s[-2]: m = m + [len(s)-1]
    return sorted(set(m))

def find_minima(s):
    m = []
    i = 1
    while i < len(s) - 1:
        if s[i] < s[i-1]:
            ps = i
            while i < len(s)-1 and s[i] == s[i+1]:
                i += 1
            if i < len(s)-1 and s[i] < s[i+1]:
                m.append((ps+i)//2)
            i += 1; continue
        i += 1
    if s[0] < s[1]:   m = [0] + m
    if s[-1] < s[-2]: m = m + [len(s)-1]
    return sorted(set(m))

mx = np.array(find_maxima(sig), dtype=float)
mn = np.array(find_minima(sig), dtype=float)

# Reflect 2 outermost extrema across each boundary (same as ExtremasMirror)
n = len(sig)
def reflect(idx, left):
    if left:
        return sorted(-idx[:2][::-1])
    else:
        return sorted(2*(n-1) - idx[-2:][::-1])

lmax = reflect(mx, True);  rmax = reflect(mx, False)
lmin = reflect(mn, True);  rmin = reflect(mn, False)

max_idx = np.array(sorted(lmax + list(mx) + rmax))
max_val = np.array([sig[abs(int(x))] if x <= 0 else (sig[int(x)] if x < n else sig[int(2*(n-1)-x)]) for x in max_idx])
min_idx = np.array(sorted(lmin + list(mn) + rmin))
min_val = np.array([sig[abs(int(x))] if x <= 0 else (sig[int(x)] if x < n else sig[int(2*(n-1)-x)]) for x in min_idx])

cs_max = CubicSpline(max_idx, max_val)  # not-a-knot (scipy default)
cs_min = CubicSpline(min_idx, min_val)

upper_fe = cs_max(T)
lower_fe = cs_min(T)
mean_fe  = (upper_fe + lower_fe) / 2

print()
print(f"{'i':>4}  {'upper_fe':>10}  {'lower_fe':>10}  {'mean_fe':>10}  {'Δmean':>10}")
for i in range(20):
    print(f"{i:>4}  {upper_fe[i]:>10.5f}  {lower_fe[i]:>10.5f}  {mean_fe[i]:>10.5f}  {mean_fe[i]-mean_py[i]:>10.5f}")

print()
print("Max |Δmean| over full signal:", np.max(np.abs(mean_fe - mean_py)).round(6))
print("Mean |Δmean| over full signal:", np.mean(np.abs(mean_fe - mean_py)).round(6))
