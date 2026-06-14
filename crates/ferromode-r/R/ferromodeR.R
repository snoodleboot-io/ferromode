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

#' Reconstruct a signal from a decomposition result
#' @param result Named list returned by any decomposition function
#' @return Numeric vector (sum of IMFs and residue)
#' @export
reconstruct <- function(result) .Call(wrap__reconstruct, result)

#' Return the ferromodeR package version string
#' @return Character string
#' @export
ferromode_r_version <- function() .Call(wrap__ferromode_r_version)
