/*
 * CUDA Kernels for GPU-Accelerated EMD/CEEMDAN
 *
 * This file contains the core computational kernels for EEMD and CEEMDAN
 * (Ensemble Empirical Mode Decomposition) algorithms.
 *
 * Three main kernels:
 * 1. generate_noise_kernel - Parallel Gaussian noise generation
 * 2. add_signal_kernel - Element-wise signal + scaled noise
 * 3. find_extrema_kernel - Parallel extrema detection for spline basis
 *
 * Compiled with NVCC and linked against CUDA runtime.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

/* Forward declarations */
struct ExtremaResult;

/**
 * CUDA error checking utility macro.
 * Used in host code only (not in kernels).
 */
#define CUDA_CHECK(err)                                                     \
  do {                                                                      \
    cudaError_t cuda_err = (err);                                          \
    if (cuda_err != cudaSuccess) {                                         \
      return cuda_err;                                                     \
    }                                                                       \
  } while (0)

/**
 * Kernel 1: generate_noise_kernel
 *
 * Generate independent Gaussian (normal distribution) noise in parallel.
 *
 * Algorithm:
 * - One thread per noise sample
 * - Uses cuRAND for parallel random number generation
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
 * @returns CUDA error code (cudaSuccess if successful)
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
  curandState state;
  curand_init(seed, idx, 0, &state);

  // Generate standard normal using Box-Muller transform
  // curand_normal returns N(0, 1)
  double noise = curand_normal_double(&state);

  // Scale by requested standard deviation
  output[idx] = noise * scale;
}

/**
 * Kernel 2: add_signal_kernel
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
 * @returns CUDA error code
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
 * Kernel 3: find_extrema_kernel
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
 * - 2D grid for better load distribution
 * - 16x16 threads per block (256 total)
 * - Grid size: (N + 255) / 256 blocks
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
 * @returns CUDA error code
 */
__global__ void find_extrema_kernel(const double *signal,
                                    unsigned int *max_indices,
                                    unsigned int *min_indices,
                                    unsigned int *max_count,
                                    unsigned int *min_count, unsigned int size) {
  // Thread index in 1D (flatten 2D grid)
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
 * Host function: Launch noise generation kernel
 *
 * Wrapper that calls the generate_noise kernel with proper error handling.
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
 * @returns CUDA error code
 */
extern "C" cudaError_t launch_generate_noise(unsigned long long seed,
                                             double scale, double *output,
                                             unsigned int size) {
  // Validate input
  if (output == nullptr || size == 0) {
    return cudaErrorInvalidValue;
  }

  // Calculate grid configuration
  unsigned int block_size = 256;
  unsigned int num_blocks = (size + block_size - 1) / block_size;

  // Launch kernel
  generate_noise_kernel<<<num_blocks, block_size>>>(seed, scale, output, size);

  // Check for launch errors
  return cudaGetLastError();
}

/**
 * Host function: Launch signal addition kernel
 *
 * Wrapper that calls the add_signal kernel.
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
 * @returns CUDA error code
 */
extern "C" cudaError_t launch_add_signal(const double *signal,
                                         const double *noise,
                                         double noise_scale, double *output,
                                         unsigned int size) {
  // Validate input
  if (signal == nullptr || noise == nullptr || output == nullptr ||
      size == 0) {
    return cudaErrorInvalidValue;
  }

  // Calculate grid configuration
  unsigned int block_size = 256;
  unsigned int num_blocks = (size + block_size - 1) / block_size;

  // Launch kernel
  add_signal_kernel<<<num_blocks, block_size>>>(signal, noise, noise_scale,
                                                output, size);

  // Check for launch errors
  return cudaGetLastError();
}

/**
 * Host function: Launch extrema finding kernel
 *
 * Wrapper that calls the find_extrema kernel.
 *
 * Note: Output arrays must be pre-initialized to 0.
 *       Counters must be pre-initialized to 0.
 *
 * Grid configuration:
 * - blocks = (size + 255) / 256
 * - threads = 256 (16x16)
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
 * @returns CUDA error code
 */
extern "C" cudaError_t launch_find_extrema(
    const double *signal, unsigned int *max_indices,
    unsigned int *min_indices, unsigned int *max_count,
    unsigned int *min_count, unsigned int size) {
  // Validate input
  if (signal == nullptr || max_indices == nullptr ||
      min_indices == nullptr || max_count == nullptr ||
      min_count == nullptr || size == 0) {
    return cudaErrorInvalidValue;
  }

  // Calculate grid configuration
  unsigned int block_size = 256;
  unsigned int num_blocks = (size + block_size - 1) / block_size;

  // Launch kernel
  find_extrema_kernel<<<num_blocks, block_size>>>(
      signal, max_indices, min_indices, max_count, min_count, size);

  // Check for launch errors
  return cudaGetLastError();
}

/*
 * GPU Architecture Support (nvcc compilation flags):
 *
 * Gencode flags for different GPU architectures:
 * -gencode arch=compute_70,code=sm_70   # NVIDIA V100, Titan V
 * -gencode arch=compute_80,code=sm_80   # NVIDIA A100
 * -gencode arch=compute_86,code=sm_86   # NVIDIA RTX 30xx series
 * -gencode arch=compute_89,code=sm_89   # NVIDIA RTX 40xx series
 *
 * These flags enable JIT compilation for forward compatibility.
 *
 * Features used:
 * - cuRAND (random number generation)
 * - Atomic operations (atomicInc)
 * - Standard warp operations (32-wide SIMD)
 *
 * Performance characteristics:
 * - Memory bandwidth limited: depends on GPU memory bandwidth
 * - Compute limited: extrema detection uses minimal computation
 * - Latency hidden by massive parallelism
 */
