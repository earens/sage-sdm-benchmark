#!/usr/bin/env Rscript

library(maxnet)
library(hdf5r)


args <- commandArgs(trailingOnly = TRUE)
infile     <- args[1]  
outfile    <- args[2]  
model_path <- args[3]  

m <- readRDS(model_path)

# Read HDF5 datasets
f <- H5File$new(infile, mode = "r")
on.exit(f$close_all())

# Helper function to safely read and ensure 2D matrix
read_2d <- function(h5_dataset) {
  data <- h5_dataset$read()
  if (is.vector(data)) {
    data <- matrix(data, ncol = 1)
  } else if (!is.matrix(data)) {
    data <- as.matrix(data)
  }
  # Transpose if needed (HDF5 uses row-major, R uses column-major)
  data <- t(data)
  
  return(data)
}

X_val  <- read_2d(f[["X_val"]])
X_test <- read_2d(f[["X_test"]])

# Convert to data.frame and numeric
X_val  <- as.data.frame(X_val)
X_test <- as.data.frame(X_test)

model_vars <- all.vars(m$formula)

#print len model vars
cat("num cols:", ncol(X_val), "\n")
cat("num rows:", nrow(X_val), "\n")
missing <- setdiff(model_vars, colnames(X_val))

if (length(missing) > 0) {
  stop("Missing predictors: ", paste(missing, collapse = ", "))
}

# Set column names to match what the model expects (V1, V2, V3, ...)
n_features <- ncol(X_val)
col_names <- paste0("V", seq_len(n_features))
colnames(X_val) <- col_names
colnames(X_test) <- col_names

X_val[]  <- lapply(X_val, as.numeric)
X_test[] <- lapply(X_test, as.numeric)

val_pred  <- predict(m, X_val,  type = "cloglog")
test_pred <- predict(m, X_test, type = "cloglog")

# Write predictions to HDF5
if (file.exists(outfile)) file.remove(outfile)
f_out <- H5File$new(outfile, mode = "w")
on.exit(f_out$close_all(), add = TRUE)

f_out[["val"]]  <- val_pred
f_out[["test"]] <- test_pred

