#!/usr/bin/env Rscript
# Run R EMD on a signal CSV and write IMFs + residue to output CSV.
# Usage: Rscript run_r_emd.R <input.csv> <output.csv> <max_imfs>

library(EMD)

args     <- commandArgs(trailingOnly = TRUE)
in_path  <- args[1]
out_path <- args[2]
max_imfs <- as.integer(args[3])

sig <- as.numeric(readLines(in_path))

result <- emd(xt = sig, max.imf = max_imfs, boundary = "wave")

imf_mat  <- result$imf        # matrix: nrow=length(sig), ncol=n_imfs
residue  <- result$residue    # vector length(sig)

n_imfs_out <- ncol(imf_mat)

# Build header
header <- paste(c(paste0("imf", seq_len(n_imfs_out)), "residue"), collapse = ",")

# Build rows
lines <- c(header)
for (i in seq_along(sig)) {
    vals <- c(imf_mat[i, ], residue[i])
    lines <- c(lines, paste(vals, collapse = ","))
}

writeLines(lines, out_path)
