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

test_that("emd rejects non-finite values", {
  signal <- c(1.0, NaN, 3.0, 4.0, 5.0)
  expect_error(emd(signal, list()), "non-finite")
})

test_that("emd rejects too-short signal", {
  signal <- c(1.0, 2.0)
  expect_error(emd(signal, list()), "at least 3")
})
