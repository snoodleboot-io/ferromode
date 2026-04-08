/*
 * HIP Kernels for GPU-Accelerated EMD/CEEMDAN (AMD ROCm Backend)
 *
 * This file contains the core computational kernels for EEMD and CEEMDAN
 * (Ensemble Empirical Mode Decomposition) algorithms, ported from CUDA
 * to HIP for AMD GPU support (RDNA, RDNA2, GCN architectures).
 *
 * Three main kernels:
 * 1. generate_noise_kernel - Parallel Gaussian noise generation
 * 2. add_signal_kernel - Element-wise signal + scaled noise
 * 3. find_extrema_kernel - Parallel extrema detection for spline basis
 *
 * Compiled with hipcc and linked against HIP runtime.
 *
 * HIP/CUDA Differences:
 * - `hipLaunchKernelGGL()` replaces `<<<grid, block>>>`
 * - `hiprandState_t` replaces `curandState_t` for RNG
 * - `hiprand_normal_double()` replaces `curand_normal_double()`
 * - `__global__`, `__device__` are compatible
 * - Atomic operations: `atomicInc()` works in both
 *
 * Build Architecture Flags:
 * - RDNA:  --offload-arch=gfx90a (MI250, MI250X)
 * - RDNA2: --offload-arch=gfx1030 (RX 6000 series)
 * - GCN:   --offload-arch=gfx906 (MI50, MI60, Radeon VII)
 */

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

/* Forward declarations */
struct ExtremaResult;

/**
 * HIP error checking utility macro.
 * Used in host code only (not in kernels).
 */
#define HIP_CHECK(err)                                                      \
  do {                                                                      \
    hipError_t hip_err = (err);                                            \
    if (hip_err != hipSuccess) {                                           \
      return hip_err;                                                      \
    }                                                                       \
  } while (0)

/**
 * Kernel 1: generate_noise_kernel (HIP)
 *
 * Generate independent Gaussian (normal distribution) noise in parallel.
 * Ported from CUDA with HIP-specific RNG handling.
 *
 * Algorithm:
 * - One thread per noise sample
 * - Uses hiprand for parallel random number generation
 * - Box-Muller transform for Gaussian sampling
 *
 * Grid configuration:
 * - 1D grid: (N + 255) / 256 blocks
 * - 1D blocks: 256 threads per block
 * - 1 thread per noise sample
 *
 * Time complexity: O(N) parallel
 * Work per thread: 1 Box-Muller pair generation per 2 samples
 *
 * @param seed Initial seed for RNG state
 * @param scale Standard deviation scaling factor
 * @param output Device pointer to output array (f64, length N)
 * @param size Number of noise samples to generate
 *
 * @returns HIP error code (hipSuccess if successful)
 */
__global__ void generate_noise_kernel(unsigned long long seed, double scale,
                                      double *output, unsigned int size) {
  // Thread index in 1D grid
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Guard: check bounds
  if (idx >= size) {
    return;
  }

  // Initialize RNG state for this thread
  // Each thread has its own state for thread-safe generation
  hiprandState state;
  hiprand_init(seed, idx, 0, &state);

  // Generate standard normal using Box-Muller transform
  // hiprand_normal_double returns N(0, 1)
  double noise = hiprand_normal_double(&state);

  // Scale by requested standard deviation
  output[idx] = noise * scale;
}

/**
 * Kernel 2: add_signal_kernel (HIP)
 *
 * Add scaled noise to signal: output[i] = signal[i] + scale * noise[i]
 *
 * Algorithm:
 * - Element-wise addition with scaling
 * - One thread per sample
 * - Bandwidth-limited: high arithmetic intensity
 *
 * Grid configuration:
 * - 1D grid: (N + 255) / 256 blocks
 * - 1D blocks: 256 threads per block
 *
 * Time complexity: O(N) parallel, bandwidth-limited
 * Arithmetic intensity: 1 multiply + 1 add per 3 loads (signal, noise, output)
 *
 * @param signal Device pointer to signal array (f64, length N)
 * @param noise Device pointer to noise array (f64, length N)
 * @param noise_scale Scaling factor for noise (typically 0.001 to 1.0)
 * @param output Device pointer to output array (f64, length N)
 * @param size Number of elements to process
 *
 * @returns HIP error code
 */
__global__ void add_signal_kernel(const double *signal, const double *noise,
                                  double noise_scale, double *output,
                                  unsigned int size) {
  // Thread index in 1D grid
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Guard: check bounds
  if (idx >= size) {
    return;
  }

  // Element-wise addition with scaling
  output[idx] = signal[idx] + noise_scale * noise[idx];
}

/**
 * Helper: Compare values for extrema detection
 *
 * Checks if point idx is a local maximum: signal[idx-1] < signal[idx] >
 * signal[idx+1]
 *
 * @param signal Signal array
 * @param idx Index to check
 * @param size Total size of signal
 *
 * @returns 1 if local maximum, 0 otherwise
 */
__device__ int is_local_max(const double *signal, int idx, int size) {
  // Boundary check: need both neighbors
  if (idx <= 0 || idx >= size - 1) {
    return 0;
  }

  // Check local maximum: signal[idx-1] < signal[idx] > signal[idx+1]
  return (signal[idx - 1] < signal[idx]) && (signal[idx + 1] < signal[idx]);
}

/**
 * Helper: Check if point is a local minimum
 *
 * Checks if point idx is a local minimum: signal[idx-1] > signal[idx] <
 * signal[idx+1]
 *
 * @param signal Signal array
 * @param idx Index to check
 * @param size Total size of signal
 *
 * @returns 1 if local minimum, 0 otherwise
 */
__device__ int is_local_min(const double *signal, int idx, int size) {
  // Boundary check: need both neighbors
  if (idx <= 0 || idx >= size - 1) {
    return 0;
  }

  // Check local minimum: signal[idx-1] > signal[idx] < signal[idx+1]
  return (signal[idx - 1] > signal[idx]) && (signal[idx + 1] < signal[idx]);
}

/**
 * Kernel 3: find_extrema_kernel (HIP)
 *
 * Find local extrema (maxima and minima) in the signal.
 * Required for spline basis computation in EMD algorithm.
 *
 * Algorithm:
 * - One thread per sample (parallel extrema detection)
 * - Each thread checks if its point is a local max or min
 * - Use atomic operations to append to result arrays
 * - Track counts of extrema found
 *
 * Grid configuration:
 * - 1D grid: (N + 255) / 256 blocks
 * - 1D blocks: 256 threads per block
 *
 * Time complexity: O(N) parallel with atomic operations
 * Synchronization: Minimal (only atomic increments)
 *
 * @param signal Device pointer to signal array (f64, length size)
 * @param max_indices Output array for maxima indices (u32, length size)
 * @param min_indices Output array for minima indices (u32, length size)
 * @param max_count Output scalar: number of maxima found (u32)
 * @param min_count Output scalar: number of minima found (u32)
 * @param size Number of samples in signal
 *
 * @returns HIP error code
 */
__global__ void find_extrema_kernel(const double *signal,
                                    unsigned int *max_indices,
                                    unsigned int *min_indices,
                                    unsigned int *max_count,
                                    unsigned int *min_count,
                                    unsigned int size) {
  // Thread index in 1D (flattened grid)
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Guard: check bounds
  if (idx >= size) {
    return;
  }

  // Check if this point is a local maximum
  if (is_local_max(signal, idx, size)) {
    // Atomically increment counter and get position
    unsigned int pos = atomicInc(max_count, size);

    // Append index to result array (safe because pos < size)
    if (pos < size) {
      max_indices[pos] = idx;
    }
  }

  // Check if this point is a local minimum
  if (is_local_min(signal, idx, size)) {
    // Atomically increment counter and get position
    unsigned int pos = atomicInc(min_count, size);

    // Append index to result array (safe because pos < size)
    if (pos < size) {
      min_indices[pos] = idx;
    }
  }
}

/**
 * Host function: Launch noise generation kernel (HIP)
 *
 * Wrapper that calls the generate_noise kernel with proper error handling.
 * Uses hipLaunchKernelGGL for portable kernel launch syntax.
 *
 * Grid configuration:
 * - blocks = (size + 255) / 256
 * - threads = 256
 *
 * @param seed RNG seed for reproducibility
 * @param scale Standard deviation of Gaussian noise
 * @param output Device pointer to output array (preallocated, size*sizeof(f64)
 * bytes)
 * @param size Number of noise samples to generate
 *
 * @returns HIP error code
 */
extern "C" hipError_t launch_generate_noise_hip(unsigned long long seed,
                                                double scale, double *output,
                                                unsigned int size) {
  // Validate input
  if (output == nullptr || size == 0) {
    return hipErrorInvalidValue;
  }

  // Calculate grid configuration
  unsigned int block_size = 256;
  unsigned int num_blocks = (size + block_size - 1) / block_size;

  // Launch kernel using hipLaunchKernelGGL
  // Syntax: hipLaunchKernelGGL(kernel, numBlocks, blockSize, 0, 0, args...)
  hipLaunchKernelGGL(generate_noise_kernel, dim3(num_blocks),
                     dim3(block_size), 0, 0, seed, scale, output, size);

  // Check for launch errors
  return hipGetLastError();
}

/**
 * Host function: Launch signal addition kernel (HIP)
 *
 * Wrapper that calls the add_signal kernel.
 * Uses hipLaunchKernelGGL for portable kernel launch syntax.
 *
 * Grid configuration:
 * - blocks = (size + 255) / 256
 * - threads = 256
 *
 * @param signal Device pointer to signal array
 * @param noise Device pointer to noise array
 * @param noise_scale Scaling factor (typically small, e.g. 0.1)
 * @param output Device pointer to output array (preallocated)
 * @param size Number of elements
 *
 * @returns HIP error code
 */
extern "C" hipError_t launch_add_signal_hip(const double *signal,
                                            const double *noise,
                                            double noise_scale,
                                            double *output,
                                            unsigned int size) {
  // Validate input
  if (signal == nullptr || noise == nullptr || output == nullptr ||
      size == 0) {
    return hipErrorInvalidValue;
  }

  // Calculate grid configuration
  unsigned int block_size = 256;
  unsigned int num_blocks = (size + block_size - 1) / block_size;

  // Launch kernel using hipLaunchKernelGGL
  hipLaunchKernelGGL(add_signal_kernel, dim3(num_blocks),
                     dim3(block_size), 0, 0, signal, noise, noise_scale,
                     output, size);

  // Check for launch errors
  return hipGetLastError();
}

/**
 * Host function: Launch extrema finding kernel (HIP)
 *
 * Wrapper that calls the find_extrema kernel.
 * Uses hipLaunchKernelGGL for portable kernel launch syntax.
 *
 * Note: Output arrays must be pre-initialized to 0.
 *       Counters must be pre-initialized to 0.
 *
 * Grid configuration:
 * - blocks = (size + 255) / 256
 * - threads = 256
 *
 * @param signal Device pointer to signal array
 * @param max_indices Output array for maxima indices (must be preallocated,
 *                    size >= num_maxima)
 * @param min_indices Output array for minima indices (must be preallocated,
 *                    size >= num_minima)
 * @param max_count Output scalar: count of maxima found
 * @param min_count Output scalar: count of minima found
 * @param size Number of samples in signal
 *
 * @returns HIP error code
 */
extern "C" hipError_t launch_find_extrema_hip(
    const double *signal, unsigned int *max_indices,
    unsigned int *min_indices, unsigned int *max_count,
    unsigned int *min_count, unsigned int size) {
  // Validate input
  if (signal == nullptr || max_indices == nullptr ||
      min_indices == nullptr || max_count == nullptr ||
      min_count == nullptr || size == 0) {
    return hipErrorInvalidValue;
  }

  // Calculate grid configuration
  unsigned int block_size = 256;
  unsigned int num_blocks = (size + block_size - 1) / block_size;

  // Launch kernel using hipLaunchKernelGGL
  hipLaunchKernelGGL(find_extrema_kernel, dim3(num_blocks),
                     dim3(block_size), 0, 0, signal, max_indices,
                     min_indices, max_count, min_count, size);

  // Check for launch errors
  return hipGetLastError();
}

/*
 * GPU Architecture Support (hipcc compilation flags):
 *
 * Gencode flags for different GPU architectures:
 * --offload-arch=gfx906    # AMD CDNA (MI50, MI60, Radeon VII)
 * --offload-arch=gfx90a    # AMD CDNA2 (MI250, MI250X)
 * --offload-arch=gfx1030   # AMD RDNA2 (RX 6000 series)
 *
 * These flags enable JIT compilation for forward compatibility.
 *
 * Features used:
 * - hiprand (random number generation)
 * - Atomic operations (atomicInc)
 * - Standard warp operations (64-wide SIMD on RDNA, 64-wide on CDNA)
 *
 * Performance characteristics:
 * - Memory bandwidth limited: depends on GPU memory bandwidth
 * - Compute limited: extrema detection uses minimal computation
 * - Latency hidden by massive parallelism
 *
 * Key HIP Differences from CUDA:
 * - hiprand_init() vs curand_init()
 * - hiprand_normal_double() vs curand_normal_double()
 * - hipLaunchKernelGGL() vs <<<>>> syntax
 * - atomicInc() works identically in HIP and CUDA
 * - AMD wave width can be 64-wide (vs NVIDIA 32-wide warps)
 */
