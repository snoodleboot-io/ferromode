# Ferromode MEX Bindings

MATLAB/Octave MEX bindings for Ferromode signal decomposition algorithms.

## Building

### Prerequisites

- Rust 1.75+
- MATLAB R2017b+ or GNU Octave 5.0+
- A C compiler (for the MEX build system)

### MATLAB Build

Set the `MATLAB_ROOT` environment variable to your MATLAB installation:

```bash
export MATLAB_ROOT=/usr/local/MATLAB/R2024a
cargo build -p ferromode-mex --release
```

The output will be `libferromode_mex.mexa64` (Linux), `libferromode_mex.mexmaci64` (macOS), or `libferromode_mex.mexw64` (Windows).

### Octave Build

Set the `FERROMODE_OCTAVE` environment variable:

```bash
export FERROMODE_OCTAVE=1
cargo build -p ferromode-mex --release
```

The output will be `libferromode_mex.oct`.

## Usage

Add the `m/` directory and the compiled MEX binary to your MATLAB/Octave path:

```matlab
addpath('/path/to/ferromode/crates/ferromode-mex/m');
addpath('/path/to/ferromode/target/release');
```

### Basic Usage

```matlab
% Create a test signal
t = linspace(0, 1, 1000);
x = sin(2*pi*5*t) + 0.5*sin(2*pi*20*t);

% EMD decomposition
result = ferromode_emd(x);
plot(result.imfs');
title(sprintf('EMD: %d IMFs extracted', result.n_imfs));

% EEMD with custom parameters
result = ferromode_eemd(x, 'NumEnsembles', 50, 'Seed', 42);

% VMD with 2 modes
result = ferromode_vmd(x, 'NModes', 2);

% Multivariate EMD (bivariate signal)
ch1 = sin(2*pi*5*t);
ch2 = sin(2*pi*5*t + pi/4);
mv_signal = [ch1', ch2'];
result = ferromode_memd(mv_signal, 'NumDirections', 32);

% Reconstruct signal from decomposition
reconstructed = ferromode_reconstruct(result);
max_error = max(abs(x - reconstructed));
fprintf('Reconstruction error: %.2e\n', max_error);
```

### Configurable Parameters

| Parameter | Algorithms | Default | Description |
|-----------|-----------|---------|-------------|
| `MaxIMFs` | EMD, MEMD | 0 (auto) | Maximum IMFs to extract |
| `SDThreshold` | EMD, MEMD | 0.2 | Sifting stopping criterion |
| `SNumber` | EMD, MEMD | 5 | Consecutive stable iterations |
| `MaxSiftingIterations` | EMD, MEMD | 100 | Max sifting iterations per IMF |
| `BoundaryCondition` | EMD | 'mirror' | Boundary handling strategy |
| `NumEnsembles` | EEMD, CEEMD, CEEMDAN, ICEEMDAN | 100 | Number of ensemble trials |
| `NoiseStd` | EEMD, CEEMD, CEEMDAN, ICEEMDAN, NA-MEMD | 0.2 | Noise amplitude fraction |
| `Seed` | All ensemble methods | [] | Random seed for reproducibility |
| `NumDirections` | MEMD, NA-MEMD | 16 | Direction vectors for projection |
| `NNoiseChannels` | NA-MEMD | 2 | Number of noise channels |
| `NModes` | VMD | 3 | Number of modes to extract |
| `Alpha` | VMD | 2000 | Bandwidth penalty |
| `Tau` | VMD | 0 | Dual ascent step size |
| `Tol` | VMD | 1e-7 | Convergence tolerance |
| `MaxIterations` | VMD | 500 | Maximum ADMM iterations |

### Result Structure

All decomposition functions return a struct with:

| Field | Type | Description |
|-------|------|-------------|
| `imfs` | n_imfs x n_samples | Intrinsic Mode Functions |
| `residue` | 1 x n_samples | Final residue |
| `n_imfs` | scalar | Number of IMFs extracted |
| `algorithm` | string | Algorithm name |
| `elapsed_ms` | scalar | Computation time (ms) |
| `n_siftings` | scalar | Total sifting iterations |

## Algorithms

- **EMD** — Empirical Mode Decomposition (Huang et al., 1998)
- **EEMD** — Ensemble EMD (Wu & Huang, 2009)
- **CEEMD** — Complementary EEMD (Yeh et al., 2010)
- **CEEMDAN** — Complete Ensemble EMD with Adaptive Noise (Torres et al., 2011)
- **ICEEMDAN** — Improved CEEMDAN (Colominas et al., 2014)
- **MEMD** — Multivariate EMD (Rehman & Mandic, 2010)
- **NA-MEMD** — Noise-Assisted MEMD (Rehman & Mandic, 2011)
- **VMD** — Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014)

## Testing

```matlab
cd crates/ferromode-mex/tests
test_ferromode
```

## Architecture

The MEX layer follows a pure-wrap contract:
- **No algorithm logic** — all computation is in the Rust `ferromode` crate
- **Marshalling only** — converts MATLAB matrices to Rust slices and back
- **Validation at boundary** — checks for non-double, complex, empty, non-finite inputs
- **Error propagation** — Rust `EmdError` messages forwarded to MATLAB via `mexErrMsgIdAndTxt`
