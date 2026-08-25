#!/usr/bin/env Rscript

library(maxnet)
library(hdf5r)


args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 6) {
  stop("Expected 6 arguments: infile, outfile, model_path, time_log, species_id, seed")
}

infile     <- args[1]
outfile    <- args[2]
model_path <- args[3]
time_log   <- args[4]
species_id <- as.integer(args[5])
seed       <- as.integer(args[6])

set.seed(seed)

cat("Loading training and test data from:", infile, "\n")
cat("Using seed:", seed, "\n")

# ------------------------------
# Read HDF5 datasets
# ------------------------------
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
  # Transpose: HDF5 uses row-major (C), R uses column-major (Fortran)
  data <- t(data)
  return(data)
}

X_train <- read_2d(f[["X_train"]])
y_train <- as.numeric(f[["y_train"]]$read())
X_val   <- read_2d(f[["X_val"]])
X_test  <- read_2d(f[["X_test"]])

# Ensure y is binary 0/1
y_train[y_train != 0] <- 1

# Convert X to data.frames
X_train <- as.data.frame(X_train)
X_val   <- as.data.frame(X_val)
X_test  <- as.data.frame(X_test)

# Set consistent column names (V1, V2, V3, ...)
n_features <- ncol(X_train)
col_names <- paste0("V", seq_len(n_features))
colnames(X_train) <- col_names
colnames(X_val) <- col_names
colnames(X_test) <- col_names

X_train[] <- lapply(X_train, as.numeric)
X_val[]   <- lapply(X_val, as.numeric)
X_test[]  <- lapply(X_test, as.numeric)

# Feature class selection
n_presences <- sum(y_train == 1)
n_features  <- ncol(X_train)

if (n_presences < 10) {
  feature_classes <- "l"
} else if (n_presences < 15) {
  feature_classes <- "lq"
} else if (n_presences < 100) {
  feature_classes <- "lqp"
} else {
  feature_classes <- "lqpht"
}

cat("Training MaxNet model with feature classes:", feature_classes, "\n")
log_training_info <- function(time_log, species_id, n_samples, n_presences, n_features, 
                              feature_classes, train_time = NA, success = FALSE) {
  cat("Logging training info to:", time_log, "at species_id", species_id, "\n")
  
  df <- read.csv(time_log)
  
  row_idx <- species_id + 1  # Python 0-based indexing
  row_idx <- which(df$species_id == species_id)

  
  df$n_samples[row_idx]       <- n_samples
  df$n_presences[row_idx]     <- n_presences
  df$n_features[row_idx]      <- n_features
  df$feature_classes[row_idx] <- feature_classes
  
  if (!is.na(train_time)) {
    df$train_time_seconds[row_idx] <- train_time
  }
  
  write.csv(df, time_log, row.names = FALSE)
}

train_success <- FALSE
train_time    <- NA

tryCatch({
  train_start <- Sys.time()
  
  f_formula <- maxnet.formula(p = y_train, data = X_train, classes = feature_classes)

  m <- maxnet(p = y_train, data = X_train, f = f_formula, regmult = 1)
  
  train_end <- Sys.time()
  train_time <- as.numeric(difftime(train_end, train_start, units = "secs"))
  
  cat("Training time:", train_time, "seconds\n")
  cat("Saving model to:", model_path, "\n")
  saveRDS(m, file = model_path)
 
  val_pred  <- predict(m, X_val, type = "cloglog")
  test_pred <- predict(m, X_test, type = "cloglog")
  
  cat("Writing predictions to HDF5...\n")
  
  if (file.exists(outfile)) file.remove(outfile)
  f_out <- H5File$new(outfile, mode = "w")
  on.exit(f_out$close_all(), add = TRUE)
  
  f_out[["val"]]  <- val_pred
  f_out[["test"]] <- test_pred
  
  train_success <- TRUE
  
}, error = function(e) {
  cat("ERROR during training:", conditionMessage(e), "\n")
  train_success <- FALSE
})

log_training_info(
  time_log,
  species_id,
  nrow(X_train),
  n_presences,
  n_features,
  feature_classes,
  train_time,
  train_success
)

if (!train_success) {
  cat("Training failed for species", species_id, "\n")
  quit(status = 1)
}
