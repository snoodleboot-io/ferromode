# JavaScript/TypeScript Tutorial

## Installation

```bash
npm install ferromode-js
```

## Quick Start

```javascript
import init, {
  emd_wasm,
  WasmEmdConfig,
  WasmBoundaryCondition,
} from "ferromode-js";

// Initialize the WASM module
await init();

// Create a signal
const signal = new Float64Array(1000);
for (let i = 0; i < signal.length; i++) {
  signal[i] = Math.sin(2 * Math.PI * 0.05 * i) +
              0.5 * Math.sin(2 * Math.PI * 0.2 * i);
}

// Configure EMD
const config = new WasmEmdConfig(
  0.2,    // sd_threshold
  5,      // s_number
  100,    // max_sifting_iterations
  0,      // max_imfs (0 = auto)
  WasmBoundaryCondition.MirrorEven,
  true,   // validate_reconstruction
  1e-12,  // reconstruction_tolerance
);

// Run decomposition
const result = emd_wasm(signal, config);

// Access IMFs
const imfs = result.imfs();
console.log(`Extracted ${imfs.n_imfs()} IMFs`);

for (let i = 0; i < imfs.n_imfs(); i++) {
  const imf = imfs.get_imf(i);
  console.log(`IMF ${i}: length=${imf.length}`);
}

// Reconstruct signal
const reconstructed = imfs.reconstruct();
```

## Hilbert Analysis

```javascript
// Get Hilbert-Huang transform
const hilbert = result.hilbert(1000.0); // sample_rate = 1000 Hz

// Instantaneous amplitude (envelope)
const amplitude = hilbert.get_instantaneous_amplitude(0);

// Instantaneous frequency
const frequency = hilbert.get_instantaneous_frequency(0);

// Marginal spectrum
const spectrum = hilbert.get_marginal_spectrum();
```

## Ensemble Methods

```javascript
import {
  ceemdan_wasm,
  WasmEnsembleConfig,
} from "ferromode-js";

const ensembleConfig = new WasmEnsembleConfig(
  100,    // num_ensembles
  0.2,    // noise_std
  42n,    // seed (BigInt for u64)
);

const result = ceemdan_wasm(signal, ensembleConfig, emdConfig);
```

## VMD

```javascript
import { vmd_wasm, WasmVmdConfig } from "ferromode-js";

const vmdConfig = new WasmVmdConfig(
  3,       // n_modes
  2000.0,  // alpha (bandwidth penalty)
  0.0,     // tau (dual ascent step)
  1e-7,    // tol (convergence)
  500,     // max_iterations
);

const result = vmd_wasm(signal, vmdConfig);
```

## Multivariate (MEMD)

```javascript
import { memd_wasm } from "ferromode-js";

// Multiple channels as array of Float64Arrays
const ch1 = new Float64Array(500);
const ch2 = new Float64Array(500);
// ... populate channels ...

const result = memd_wasm(
  [ch1, ch2],  // channels
  16,          // num_directions
  5,           // max_imfs
  42n,         // seed
);
```

## Error Handling

```javascript
try {
  const badSignal = new Float64Array([1.0, NaN, 3.0]);
  emd_wasm(badSignal, config);
} catch (err) {
  console.error("Decomposition failed:", err.message);
}
```
