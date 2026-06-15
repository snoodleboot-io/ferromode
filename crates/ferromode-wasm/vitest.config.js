import { defineConfig } from "vitest/config";

// WASM decompositions run single-threaded and are slower on CI runners, so give
// the per-test timeout headroom above vitest's 5s default.
export default defineConfig({
  test: {
    testTimeout: 30000,
    hookTimeout: 30000,
  },
});
