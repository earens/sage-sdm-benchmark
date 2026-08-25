# install.R — R package dependencies for the SDM benchmark.
#
# Run once:  Rscript install.R
#
# NOTE: prefer `environment-r.yml` (conda) — it pulls pre-built binaries and the
# system libraries (GDAL/GEOS/PROJ, libxml2) that sf/rgbif/tidyverse need. This
# CRAN-source route requires those system libs to already be present, or the
# geospatial + tidyverse builds will fail.
#
# R is used for: the R evaluation tool (scripts/evaluate_predictions.R), the
# MaxEnt baselines (scripts/traditional_sdms/*.R), and parts of data prep
# (scripts/data_prep_scripts/*.r). Install only the group(s) you need.
#
# maxent_train.py invokes `Rscript` from your PATH, so installing these packages
# into whichever R that resolves to keeps the Python↔R handoff working (conda not
# required — a system R with these packages works too).
#
# The paper's results were produced with R 4.5.3 and the versions below. This
# script installs the current CRAN versions; for an exact match,
# pin with remotes::install_version(pkg, version), e.g.
#   remotes::install_version("maxnet", "0.1.4")
#
#   maxnet 0.1.4        glmnet 5.0         hdf5r 1.3.12       rWCVP 1.3.0
#   kewr 0.6.1          rgbif 3.8.5        sf 1.1-2           CoordinateCleaner 3.0.1
#   countrycode 1.8.0   data.table 1.18.4  dplyr 1.2.1        future.apply 1.20.2
#   pbapply 1.7-4       tidyverse 2.0.0    ggplot2 4.0.3

repos <- "https://cloud.r-project.org"

# --- CRAN packages, grouped by what needs them --------------------------------
groups <- list(
  # R evaluation tool: metrics + stratified table + delta heatmap
  eval      = c("hdf5r", "ggplot2"),
  # MaxEnt / traditional baselines
  baselines = c("maxnet", "data.table", "dplyr", "pbapply", "future.apply"),
  # Occurrence cleaning / taxonomy / spatial (data prep)
  dataprep  = c("sf", "CoordinateCleaner", "countrycode", "rgbif", "tidyverse")
)

cran <- unique(unlist(groups))
missing <- cran[!vapply(cran, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  cat("Installing from CRAN:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, repos = repos)
} else {
  cat("All CRAN packages already installed.\n")
}

# --- GitHub-only packages (data prep: POWO ranges via Kew) ---------------------
# rWCVP and kewr are not on CRAN; install them explicitly if you run the
# native-range / taxonomy prep scripts:
#
#   install.packages("remotes")
#   remotes::install_github("matildabrown/rWCVP")
#   remotes::install_github("barnabywalker/kewr")

cat("\nDone. (For data-prep taxonomy: also install rWCVP + kewr from GitHub — see comments.)\n")
