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
