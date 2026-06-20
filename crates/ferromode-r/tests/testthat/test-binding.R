test_that("emd decomposes a sine wave", {
  signal <- sin(seq(0, 2 * pi, length.out = 100))
  result <- emd(signal, list())
  expect_type(result, "list")
  expect_true(result$n_imfs >= 1)
  expect_equal(result$n_samples, 100L)
  expect_equal(result$algorithm, "emd")
  expect_true(result$elapsed_ms >= 0)
})

test_that("emd respects max_imfs", {
  signal <- sin(seq(0, 2 * pi, length.out = 100))
  result <- emd(signal, list(max_imfs = 1L))
  expect_equal(result$n_imfs, 1L)
})

test_that("eemd produces results", {
  signal <- sin(seq(0, 2 * pi, length.out = 100))
  result <- eemd(signal, list(num_ensembles = 5L, noise_std = 0.2, seed = 42L))
  expect_type(result, "list")
  expect_true(result$n_imfs >= 1)
  expect_equal(result$algorithm, "eemd")
})

test_that("ceemd produces results", {
  signal <- sin(seq(0, 2 * pi, length.out = 100))
  result <- ceemd(signal, list(num_ensembles = 5L, noise_std = 0.2, seed = 42L))
  expect_type(result, "list")
  expect_true(result$n_imfs >= 1)
  expect_equal(result$algorithm, "ceemd")
})

test_that("ceemdan produces results", {
  signal <- sin(seq(0, 2 * pi, length.out = 100))
  result <- ceemdan(signal, list(num_ensembles = 5L, noise_std = 0.2, seed = 42L))
  expect_type(result, "list")
  expect_true(result$n_imfs >= 1)
  expect_equal(result$algorithm, "ceemdan")
})

test_that("iceemdan produces results", {
  signal <- sin(seq(0, 2 * pi, length.out = 100))
  result <- iceemdan(signal, list(num_ensembles = 5L, noise_std = 0.2, seed = 42L))
  expect_type(result, "list")
  expect_true(result$n_imfs >= 1)
  expect_equal(result$algorithm, "iceemdan")
})

test_that("vmd decomposes a two-tone signal", {
  t <- seq(0, 1, length.out = 200)
  signal <- sin(2 * pi * 5 * t) + 0.5 * sin(2 * pi * 20 * t)
  result <- vmd(signal, list(n_modes = 2L, alpha = 2000.0, tol = 1e-7, max_iterations = 500L))
  expect_type(result, "list")
  expect_equal(result$n_imfs, 2L)
  expect_equal(result$algorithm, "vmd")
})

test_that("reconstruct recovers original signal", {
  signal <- sin(seq(0, 2 * pi, length.out = 100))
  result <- emd(signal, list())
  reconstructed <- reconstruct(result)
  expect_equal(length(reconstructed), length(signal))
  expect_true(max(abs(signal - reconstructed)) < 1e-6)
})

test_that("ferromode_r_version returns a string", {
  ver <- ferromode_r_version()
  expect_type(ver, "character")
  expect_true(nchar(ver) > 0)
})

test_that("emd honors boundary + full config (palindrome_cyclic)", {
  signal <- sin(seq(0, 4 * pi, length.out = 128))
  result <- emd(signal, list(max_imfs = 4L, boundary_condition = "palindrome_cyclic",
                             spline_type = "periodic", energy_threshold = 1e-7))
  expect_true(result$n_imfs >= 1)
  expect_equal(result$algorithm, "emd")
})

test_that("memd decomposes multivariate channels", {
  ch1 <- sin(seq(0, 4 * pi, length.out = 128))
  ch2 <- cos(seq(0, 4 * pi, length.out = 128))
  result <- memd(list(ch1, ch2), list(num_directions = 16L, max_imfs = 3L))
  expect_equal(result$algorithm, "memd")
  expect_true(result$n_imfs >= 1)
})

test_that("namemd decomposes multivariate channels", {
  ch1 <- sin(seq(0, 4 * pi, length.out = 128))
  ch2 <- cos(seq(0, 4 * pi, length.out = 128))
  result <- namemd(list(ch1, ch2), list(num_directions = 16L, max_imfs = 3L,
                                        n_noise_channels = 2L, noise_std = 0.1, seed = 42L))
  expect_equal(result$algorithm, "namemd")
  expect_true(result$n_imfs >= 1)
})

test_that("hilbert returns spectra", {
  signal <- sin(seq(0, 8 * pi, length.out = 128))
  result <- emd(signal, list(max_imfs = 4L))
  h <- hilbert(result$imfs, 100.0)
  expect_equal(length(h$instantaneous_amplitude), result$n_imfs)
  expect_true(length(h$marginal_spectrum) > 0)
})

test_that("streaming decomposes chunks", {
  handle <- streaming_new(list(max_imfs = 4L), 2048L, 3L)
  chunk <- sin(seq(0, 4 * pi, length.out = 256))
  out <- streaming_decompose_chunk(handle, chunk)
  expect_true(length(out$imfs) >= 1)
  expect_true(is.numeric(out$spectral_entropy))
  streaming_reset(handle)
})

test_that("differentiable forward + backward", {
  signal <- sin(seq(0, 4 * pi, length.out = 200)) + 0.3 * seq(0, 1, length.out = 200)
  fwd <- emd_forward(signal, list(max_imfs = 4L))
  expect_true(length(fwd$imfs) >= 1)
  expect_true(is.finite(fwd$reconstruction_error))
  grads <- lapply(seq_along(fwd$imfs), function(i) rep(1.0, 200))
  grad <- emd_backward(grads, signal)
  expect_equal(length(grad), 200L)
  expect_true(abs(grad[1] - 1.0) < 1e-12)
})

test_that("emd rejects non-finite values", {
  signal <- c(1.0, NaN, 3.0, 4.0, 5.0)
  expect_error(emd(signal, list()), "non-finite")
})

test_that("emd rejects too-short signal", {
  signal <- c(1.0, 2.0)
  expect_error(emd(signal, list()), "insufficient data")
})
