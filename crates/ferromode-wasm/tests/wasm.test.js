import { describe, it, expect } from "vitest";
import init, {
  emd_wasm,
  eemd_wasm,
  ceemd_wasm,
  ceemdan_wasm,
  iceemdan_wasm,
  vmd_wasm,
  memd_wasm,
  namemd_wasm,
  emd_forward,
  WasmEmdConfig,
  WasmEnsembleConfig,
  WasmVmdConfig,
  WasmMemdConfig,
  WasmNaMemdConfig,
  WasmBoundaryCondition,
  WasmSplineType,
  WasmStreamingDecomposer,
  ferromode_wasm_version,
} from "../pkg/ferromode_wasm.js";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

// The `--target web` build initializes by fetching the .wasm over HTTP, which
// isn't available under Node/vitest. Load the bytes from disk and hand them to
// init() directly instead.
const wasmBytes = await readFile(
  fileURLToPath(new URL("../pkg/ferromode_wasm_bg.wasm", import.meta.url)),
);
await init(wasmBytes);

// Generate a simple sine wave
function sineWave(n, freq, sampleRate) {
  const arr = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    arr[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate);
  }
  return arr;
}

function defaultEmdConfig(maxImfs = 0) {
  return new WasmEmdConfig(
    0.2, // sd_threshold
    5, // s_number
    100, // max_sifting_iterations
    maxImfs, // max_imfs (0 = auto)
    WasmBoundaryCondition.MirrorEven,
    true, // validate_reconstruction
    1e-12, // reconstruction_tolerance
    1e-6, // energy_threshold
    -1n, // fixed_iterations (negative = unset)
    WasmSplineType.Natural,
    -1.0, // intermittency_cv (negative = disabled)
    0, // intermittency_min_intervals
  );
}

function defaultEnsembleConfig() {
  return new WasmEnsembleConfig(10, 0.2, 42n);
}

function defaultVmdConfig() {
  return new WasmVmdConfig(3, 2000.0, 0.0, 1e-7, 500);
}

describe("ferromode-wasm", () => {
  describe("version", () => {
    it("returns a non-empty version string", () => {
      const version = ferromode_wasm_version();
      expect(version).toBeTruthy();
      expect(version.length).toBeGreaterThan(0);
    });
  });

  describe("emd_wasm", () => {
    it("decomposes a sine wave and returns WasmDecompositionResult", () => {
      const signal = sineWave(200, 10, 200);
      const config = defaultEmdConfig();
      const result = emd_wasm(signal, config);

      expect(result).toBeDefined();
      expect(typeof result.algorithm()).toBe("string");
      expect(result.algorithm()).toContain("EMD");
      expect(result.elapsed_ms()).toBeGreaterThanOrEqual(0);
    });

    it("returns Float64Array IMFs", () => {
      const signal = sineWave(200, 10, 200);
      const config = defaultEmdConfig();
      const result = emd_wasm(signal, config);
      const imfs = result.imfs();

      expect(imfs.n_imfs()).toBeGreaterThan(0);
      const imf0 = imfs.get_imf(0);
      expect(imf0 instanceof Float64Array).toBe(true);
      expect(imf0.length).toBe(200);
    });

    it("returns Float64Array residue", () => {
      const signal = sineWave(200, 10, 200);
      const config = defaultEmdConfig();
      const result = emd_wasm(signal, config);
      const imfs = result.imfs();
      const residue = imfs.get_residue();

      expect(residue instanceof Float64Array).toBe(true);
      expect(residue.length).toBe(200);
    });

    it("reconstructs signal from IMFs + residue", () => {
      const signal = sineWave(200, 10, 200);
      const config = defaultEmdConfig();
      const result = emd_wasm(signal, config);
      const imfs = result.imfs();
      const reconstructed = imfs.reconstruct();

      expect(reconstructed instanceof Float64Array).toBe(true);
      expect(reconstructed.length).toBe(200);
    });

    it("rejects NaN values", () => {
      const signal = new Float64Array([1.0, NaN, 3.0, 4.0, 5.0]);
      const config = defaultEmdConfig();

      expect(() => emd_wasm(signal, config)).toThrow();
    });

    it("rejects empty signal", () => {
      const signal = new Float64Array(0);
      const config = defaultEmdConfig();

      expect(() => emd_wasm(signal, config)).toThrow();
    });
  });

  describe("eemd_wasm", () => {
    it("decomposes with ensemble averaging", () => {
      const signal = sineWave(200, 10, 200);
      const ensembleCfg = defaultEnsembleConfig();
      const emdCfg = defaultEmdConfig();
      const result = eemd_wasm(signal, ensembleCfg, emdCfg);

      expect(result).toBeDefined();
      expect(result.algorithm()).toContain("EEMD");
      const imfs = result.imfs();
      expect(imfs.n_imfs()).toBeGreaterThan(0);
    });
  });

  describe("ceemd_wasm", () => {
    it("decomposes with complementary ensemble", () => {
      const signal = sineWave(200, 10, 200);
      const ensembleCfg = defaultEnsembleConfig();
      const emdCfg = defaultEmdConfig();
      const result = ceemd_wasm(signal, ensembleCfg, emdCfg);

      expect(result).toBeDefined();
      expect(result.algorithm()).toContain("CEEMD");
    });
  });

  describe("ceemdan_wasm", () => {
    it("decomposes with adaptive noise", () => {
      const signal = sineWave(200, 10, 200);
      const ensembleCfg = defaultEnsembleConfig();
      // Cap max_imfs: unbounded CEEMDAN stages are pathologically slow
      // on this signal (see FIXME in crates/ferromode/src/algorithms/ceemdan.rs).
      const emdCfg = defaultEmdConfig(4);
      const result = ceemdan_wasm(signal, ensembleCfg, emdCfg);

      expect(result).toBeDefined();
      expect(result.algorithm()).toContain("CEEMDAN");
    });
  });

  describe("iceemdan_wasm", () => {
    it("decomposes with improved adaptive noise", () => {
      const signal = sineWave(200, 10, 200);
      const ensembleCfg = defaultEnsembleConfig();
      const emdCfg = defaultEmdConfig();
      const result = iceemdan_wasm(signal, ensembleCfg, emdCfg);

      expect(result).toBeDefined();
      expect(result.algorithm()).toContain("ICEEMDAN");
    });
  });

  describe("vmd_wasm", () => {
    it("decomposes with variational mode decomposition", () => {
      const signal = sineWave(200, 10, 200);
      const config = defaultVmdConfig();
      const result = vmd_wasm(signal, config);

      expect(result).toBeDefined();
      expect(result.algorithm()).toContain("VMD");
      const imfs = result.imfs();
      expect(imfs.n_imfs()).toBeGreaterThan(0);
    });
  });

  describe("memd_wasm", () => {
    it("decomposes multivariate signal", () => {
      const ch1 = sineWave(200, 10, 200);
      const ch2 = sineWave(200, 10, 200);
      const channels = [ch1, ch2];

      const result = memd_wasm(channels, new WasmMemdConfig(16, 3, 0.2, 5, 100));

      expect(result).toBeDefined();
      expect(result.algorithm()).toContain("MEMD");
    });
  });

  describe("namemd_wasm", () => {
    it("decomposes with noise-assisted multivariate EMD", () => {
      const ch1 = sineWave(200, 10, 200);
      const ch2 = sineWave(200, 10, 200);
      const channels = [ch1, ch2];

      const result = namemd_wasm(
        channels,
        new WasmNaMemdConfig(16, 3, 100, 2, 0.1, 42n),
      );

      expect(result).toBeDefined();
      expect(result.algorithm()).toContain("NAMEMD");
    });
  });

  describe("streaming + differentiable", () => {
    it("streams chunks through WasmStreamingDecomposer", () => {
      const dec = new WasmStreamingDecomposer(defaultEmdConfig(4), 2048, 3);
      const chunk = sineWave(256, 5, 256);
      const out = dec.decompose_chunk(chunk);
      expect(out.n_imfs).toBeGreaterThanOrEqual(1);
      expect(out.imfs.length).toBe(out.n_imfs);
      expect(typeof out.spectral_entropy).toBe("number");
      dec.reset();
    });

    it("runs differentiable forward + backward", () => {
      const signal = sineWave(200, 5, 200);
      const fwd = emd_forward(signal, defaultEmdConfig(4));
      expect(fwd.n_imfs()).toBeGreaterThanOrEqual(1);
      expect(fwd.get_imf(0).length).toBe(200);
      expect(Number.isFinite(fwd.reconstruction_error())).toBe(true);

      const grads = [];
      for (let i = 0; i < fwd.n_imfs(); i++) grads.push(new Float64Array(200).fill(1.0));
      const grad = fwd.backward(grads); // exact VJP via the saved forward context
      expect(grad.length).toBe(200);
      expect(Array.from(grad).every(Number.isFinite)).toBe(true);
    });
  });

  describe("hilbert analysis", () => {
    it("computes Hilbert transform on decomposition result", () => {
      const signal = sineWave(200, 10, 200);
      const config = defaultEmdConfig();
      const result = emd_wasm(signal, config);
      const hilbert = result.hilbert(200.0);

      expect(hilbert).toBeDefined();

      const amp = hilbert.get_instantaneous_amplitude(0);
      expect(amp instanceof Float64Array).toBe(true);

      const freq = hilbert.get_instantaneous_frequency(0);
      expect(freq instanceof Float64Array).toBe(true);

      const spectrum = hilbert.get_marginal_spectrum();
      expect(spectrum instanceof Float64Array).toBe(true);
    });
  });
});
