# Which Algorithm Should I Use?

## Decision Tree

```
Is your signal univariate or multivariate?
│
├── Univariate (single channel)
│   │
│   │  Do you need fast, simple decomposition?
│   ├── Yes → EMD
│   │
│   │  Does your signal have mode mixing?
│   │  (multiple frequency components in one IMF)
│   ├── Yes →
│   │   │
│   │   │  Do you need the best reconstruction accuracy?
│   │   ├── Yes → ICEEMDAN
│   │   │
│   │   │  Do you need a balance of speed and quality?
│   │   ├── Yes → CEEMDAN
│   │   │
│   │   │  Do you want complementary noise cancellation?
│   │   ├── Yes → CEEMD
│   │   │
│   │   └── Default → EEMD
│   │
│   └── No →
│       │
│       │  Do you know the number of modes in advance?
│       ├── Yes → VMD
│       │
│       └── No → EMD
│
└── Multivariate (multiple channels)
    │
    │  Do channels share common oscillatory modes?
    ├── Yes → MEMD
    │
    │  Does MEMD show mode mixing across channels?
    ├── Yes → NA-MEMD
    │
    └── Default → MEMD
```

## Quick Reference

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Quick exploration | EMD |
| Mode mixing in univariate signal | CEEMDAN |
| Best reconstruction accuracy | ICEEMDAN |
| Known number of modes | VMD |
| Multivariate with mode alignment | MEMD |
| Multivariate with mode mixing | NA-MEMD |
| Noisy signals | EEMD / CEEMD |
| Biomedical signals | ICEEMDAN |
| Real-time / speed critical | EMD or VMD |

## Choosing a Boundary Condition

```
Does your signal have a non-zero mean or trend?
│
├── Yes → PalindromeCyclic
│         (most stable; ~2× slower; exact reconstruction guaranteed)
│
└── No
    │
    │  Is the signal genuinely periodic?
    ├── Yes → Periodic
    │
    │  Do you need the fastest option?
    ├── Yes → MirrorEven or MirrorOdd
    │
    │  Does the signal have smooth, slowly-varying endpoints?
    ├── Yes → Slope
    │
    └── Default → MirrorEven
```

**PalindromeCyclic** is the recommended starting point when you are uncertain
about the boundary behavior of your signal. It eliminates the most common
source of end-effect artifacts (discontinuity at boundaries) at the cost of
approximately 2× processing time. See `docs/PALINDROME_CYCLIC_GUIDE.md` for
full details and per-language code examples.
