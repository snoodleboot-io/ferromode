# Generated R wrappers for ferromodeR extendr bindings.
# Each function delegates to the compiled Rust implementation via .Call().
# Config is an R named list; results are named lists.

#' Decompose a signal using Empirical Mode Decomposition
#' @param signal Numeric vector
#' @param config Named list of config options (max_imfs)
#' @return Named list: n_imfs, n_samples, algorithm, elapsed_ms, imfs, residue
#' @export
emd <- function(signal, config) .Call(wrap__emd, signal, config)

#' Decompose a signal using Ensemble EMD
#' @param signal Numeric vector
#' @param config Named list: num_ensembles, noise_std, seed
#' @return Named list: n_imfs, n_samples, algorithm, elapsed_ms, imfs, residue
#' @export
eemd <- function(signal, config) .Call(wrap__eemd, signal, config)

#' Decompose a signal using Complementary EEMD
#' @param signal Numeric vector
#' @param config Named list: num_ensembles, noise_std, seed
#' @return Named list: n_imfs, n_samples, algorithm, elapsed_ms, imfs, residue
#' @export
ceemd <- function(signal, config) .Call(wrap__ceemd, signal, config)

#' Decompose a signal using Complete EEMD with Adaptive Noise
#' @param signal Numeric vector
#' @param config Named list: num_ensembles, noise_std, seed
#' @return Named list: n_imfs, n_samples, algorithm, elapsed_ms, imfs, residue
#' @export
ceemdan <- function(signal, config) .Call(wrap__ceemdan, signal, config)

#' Decompose a signal using Improved CEEMDAN
#' @param signal Numeric vector
#' @param config Named list: num_ensembles, noise_std, seed
#' @return Named list: n_imfs, n_samples, algorithm, elapsed_ms, imfs, residue
#' @export
iceemdan <- function(signal, config) .Call(wrap__iceemdan, signal, config)

#' Decompose a signal using Variational Mode Decomposition
#' @param signal Numeric vector
#' @param config Named list: n_modes, alpha, tol, max_iterations
#' @return Named list: n_imfs, n_samples, algorithm, elapsed_ms, imfs, residue
#' @export
vmd <- function(signal, config) .Call(wrap__vmd, signal, config)

#' Decompose multivariate channels using MEMD
#' @param channels List of numeric vectors (one per channel)
#' @param config Named list: num_directions, max_imfs, sd_threshold, s_number, max_sifting_iterations
#' @return Named list: n_imfs, n_samples, algorithm, elapsed_ms, imfs, residue
#' @export
memd <- function(channels, config) .Call(wrap__memd, channels, config)

#' Decompose multivariate channels using Noise-Assisted MEMD
#' @param channels List of numeric vectors (one per channel)
#' @param config Named list: num_directions, max_imfs, n_noise_channels, noise_std, seed
#' @return Named list: n_imfs, n_samples, algorithm, elapsed_ms, imfs, residue
#' @export
namemd <- function(channels, config) .Call(wrap__namemd, channels, config)

#' Compute the Hilbert spectrum of a set of IMFs
#' @param imfs List of numeric vectors (one per IMF)
#' @param sample_rate Numeric sampling rate
#' @return Named list: instantaneous_amplitude, instantaneous_frequency, marginal_spectrum
#' @export
hilbert <- function(imfs, sample_rate) .Call(wrap__hilbert, imfs, sample_rate)

#' Reconstruct a signal from a decomposition result
#' @param result Named list returned by any decomposition function
#' @return Numeric vector (sum of IMFs and residue)
#' @export
reconstruct <- function(result) .Call(wrap__reconstruct, result)

#' Create a streaming decomposer handle
#' @param config Named list of EMD config options
#' @param buffer_size Integer ring-buffer size (<=0 = default 4096)
#' @param ar_order Integer AR predictor order (<=0 = default 3)
#' @return External pointer handle
#' @export
streaming_new <- function(config, buffer_size = 4096L, ar_order = 3L) .Call(wrap__streaming_new, config, buffer_size, ar_order)

#' Decompose one chunk through a streaming handle
#' @param handle Handle from streaming_new
#' @param chunk Numeric vector
#' @return Named list: imfs, residue, spectral_entropy, stationarity_score, extrema_spacing_cv
#' @export
streaming_decompose_chunk <- function(handle, chunk) .Call(wrap__streaming_decompose_chunk, handle, chunk)

#' Reset a streaming handle's state
#' @param handle Handle from streaming_new
#' @export
streaming_reset <- function(handle) invisible(.Call(wrap__streaming_reset, handle))

#' Differentiable EMD forward pass
#' @param signal Numeric vector
#' @param config Named list of EMD config options
#' @return Named list: imfs, residue, num_sifts, reconstruction_error, handle
#' @export
emd_forward <- function(signal, config) .Call(wrap__emd_forward, signal, config)

#' Differentiable EMD backward pass (exact implicit differentiation)
#' @param handle The `handle` returned by `emd_forward`
#' @param grad_imfs List of numeric gradient vectors (one per IMF)
#' @return Numeric gradient vector w.r.t. the input signal
#' @export
emd_backward <- function(handle, grad_imfs) .Call(wrap__emd_backward, handle, grad_imfs)

#' Return the ferromodeR package version string
#' @return Character string
#' @export
ferromode_r_version <- function() .Call(wrap__ferromode_r_version)
