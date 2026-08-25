#!/usr/bin/env Rscript
#
# maxent_predict.R — run MaxEnt inference on arbitrary input data
#
# Usage:
#   Rscript maxent_predict.R <input.h5> <output.h5> <model.rds>
#
# input.h5  : HDF5 file with dataset "X" of shape (N_samples, N_features),
#             written by Python in row-major (C) order.
# output.h5 : HDF5 file; predictions written to dataset "predictions" (N_samples,).
# model.rds : MaxEnt model saved with saveRDS() during training.
#
# Column naming follows the same V1, V2, ... convention used in maxent_train.R
# so that the saved model can find its expected feature names.

library(maxnet)
library(hdf5r)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript maxent_predict.R input.h5 output.h5 model.rds")
}

infile    <- args[1]
outfile   <- args[2]
model_rds <- args[3]

cat("Loading predictors from:", infile, "\n")
h5_in <- H5File$new(infile, mode = "r")
on.exit(h5_in$close_all(), add = TRUE)

# Read and transpose — same as maxent_train.R read_2d()
# HDF5 stores data in row-major (C); R reads in column-major (Fortran),
# so we transpose to restore (N_samples, N_features) orientation.
X_raw <- h5_in[["X"]]$read()
if (is.vector(X_raw)) {
  X_raw <- matrix(X_raw, ncol = 1)
} else if (!is.matrix(X_raw)) {
  X_raw <- as.matrix(X_raw)
}
X_raw <- t(X_raw)   # now (N_samples, N_features)

X <- as.data.frame(X_raw)
X[] <- lapply(X, as.numeric)

# Apply the same column naming used during training (V1, V2, ...)
colnames(X) <- paste0("V", seq_len(ncol(X)))

cat("Predictor matrix:", nrow(X), "samples x", ncol(X), "features\n")

cat("Loading MaxEnt model from:", model_rds, "\n")
m <- readRDS(model_rds)

cat("Running predictions (cloglog) …\n")
preds <- tryCatch(
  predict(m, X, type = "cloglog"),
  error = function(e) stop("Prediction failed: ", conditionMessage(e))
)
preds <- as.numeric(preds)

cat("Writing", length(preds), "predictions to:", outfile, "\n")
h5_out <- H5File$new(outfile, mode = "w")
on.exit(h5_out$close_all(), add = TRUE)
h5_out[["predictions"]] <- preds

cat("Done.\n")
