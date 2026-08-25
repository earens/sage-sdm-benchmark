#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Evaluate species-distribution predictions in R.
#
# A lightweight R counterpart of scripts/evaluate_predictions.py: reads a
# predictions HDF5, computes per-species metrics (AUROC / AUPRC / F1), a
# stratified breakdown over the two property axes, and — with a reference model
# — a coarse delta heatmap (the paper's "one average per compartment" figure).
#
# Continental maps are Python-only (they need the geospatial pipeline); this
# script covers metrics + stratification + the delta heatmap.
#
# Usage:
#   Rscript scripts/evaluate_predictions.R \
#       --predictions preds.h5 [--reference ref.h5] \
#       --proxies species_properties.csv --output-dir out [--metrics auroc,auprc,f1]
#
# Prediction HDF5 format (same as the Python toolkit):
#   probabilities [N_locations, N_species]  in [0, 1]
#   targets       [N_locations, N_species]  {0, 1}, -1 = ignore
# ---------------------------------------------------------------------------

suppressMessages(library(ggplot2))

# ---- tiny base-R argument parser (avoids an optparse dependency) ----------
parse_args <- function(args) {
  out <- list(metrics = "auroc,auprc,f1", reference = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    out[[key]] <- args[i + 1]
    i <- i + 2
  }
  out
}

# ---- HDF5 read (mirrors scripts/traditional_sdms/maxent_eval.R) -----------
read_2d <- function(dset, n_species) {
  m <- dset$read()
  if (is.null(dim(m))) m <- matrix(m, nrow = 1)
  # hdf5r may return the transpose of the (locations, species) layout; orient
  # so columns index species.
  if (ncol(m) != n_species && nrow(m) == n_species) m <- t(m)
  m
}

read_predictions <- function(path, n_species) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) stop("the 'hdf5r' package is required to read HDF5 predictions (see install.R)")
  H5File <- hdf5r::H5File
  f <- H5File$new(path, mode = "r")
  on.exit(f$close_all())
  probs <- read_2d(f[["probabilities"]], n_species)
  targets <- if (f$exists("targets")) read_2d(f[["targets"]], n_species) else NULL
  list(probabilities = probs, targets = targets)
}

# ---- per-species metrics (ignore targets == -1) ---------------------------
# AUROC via the Mann-Whitney U statistic (rank-based; matches torchmetrics).
auroc_binary <- function(scores, labels) {
  pos <- scores[labels == 1]; neg <- scores[labels == 0]
  np <- length(pos); nn <- length(neg)
  if (np == 0 || nn == 0) return(NA_real_)
  r <- rank(c(pos, neg))            # average ranks -> handles ties
  (sum(r[seq_len(np)]) - np * (np + 1) / 2) / (np * nn)
}

# Average precision (step-wise, matches torchmetrics AveragePrecision).
auprc_binary <- function(scores, labels) {
  np <- sum(labels == 1)
  if (np == 0) return(NA_real_)
  ord <- order(scores, decreasing = TRUE)
  l <- labels[ord]
  tp <- cumsum(l == 1); fp <- cumsum(l == 0)
  precision <- tp / (tp + fp)
  recall <- tp / np
  sum(c(recall[1], diff(recall)) * precision)
}

f1_binary <- function(scores, labels, thr = 0.5) {
  pred <- as.integer(scores >= thr)
  tp <- sum(pred == 1 & labels == 1)
  fp <- sum(pred == 1 & labels == 0)
  fn <- sum(pred == 0 & labels == 1)
  denom <- 2 * tp + fp + fn
  if (denom == 0) return(NA_real_)
  2 * tp / denom
}

METRIC_FUNS <- list(auroc = auroc_binary, auprc = auprc_binary, f1 = f1_binary)

per_species_metrics <- function(probs, targets, metrics) {
  S <- ncol(probs)
  res <- setNames(lapply(metrics, function(m) rep(NA_real_, S)), metrics)
  for (s in seq_len(S)) {
    t <- targets[, s]; keep <- t >= 0
    if (!any(keep)) next
    sc <- probs[keep, s]; lb <- t[keep]
    for (m in metrics) res[[m]][s] <- METRIC_FUNS[[m]](sc, lb)
  }
  res
}

# ---- stratification regions (mirrors src/evaluation/regions.py) -----------
region_masks <- function(se_rank, tp_rank, N, decile = 10) {
  M <- N %/% 2; L <- N %/% decile
  in_corner <- se_rank <= L & tp_rank <= L
  diag_left <- in_corner & se_rank <  tp_rank
  diag_bot  <- in_corner & se_rank >= tp_rank
  list(
    bottom_left  = se_rank <= M & tp_rank <= M,
    bottom_right = se_rank >  M & tp_rank <= M,
    top_left     = se_rank <= M & tp_rank >  M,
    top_right    = se_rank >  M & tp_rank >  M,
    L1 = tp_rank <= L & se_rank <= M & !diag_left,
    L2 = tp_rank <= L & se_rank >  M,
    L3 = se_rank <= L & tp_rank <= M & !diag_bot,
    L4 = se_rank <= L & tp_rank >  M
  )
}

AXIS_LABELS <- list(
  sampling_effort = c("sparse", "dense"),
  relative_prevalence = c("infrequent", "frequent")
)
axis_labels <- function(col) if (!is.null(AXIS_LABELS[[col]])) AXIS_LABELS[[col]] else c("low", "high")

# Mean (or delta-vs-reference) of a metric over a species mask, in percent.
cell_value <- function(vals, ref, mask) {
  d <- if (is.null(ref)) vals[mask] else (vals - ref)[mask]
  d <- d[is.finite(d)]
  if (length(d) == 0) NA_real_ else mean(d) * 100
}

# ---- coarse delta heatmap (ggplot2) ---------------------------------------
# Violet <-> green diverging palette (matches _sdm_cmap("H") in Python).
HEATMAP_COLORS <- c("#5C3578", "#A87BC0", "#F5F2EC", "#4E9448", "#1E5020")

delta_heatmap_coarse <- function(deltas, xl, xh, yl, yh, metric_label, vmax = 5, out_path) {
  Lc <- 0.1; Mc <- 0.5
  # Quadrant rectangles.
  quads <- data.frame(
    xmin = c(0, Mc, 0, Mc), xmax = c(Mc, 1, Mc, 1),
    ymin = c(0, 0, Mc, Mc), ymax = c(Mc, Mc, 1, 1),
    delta = c(deltas$bottom_left, deltas$bottom_right, deltas$top_left, deltas$top_right),
    lx = c(Mc / 2, (Mc + 1) / 2, Mc / 2, (Mc + 1) / 2),
    ly = c(Mc / 2, Mc / 2, (Mc + 1) / 2, (Mc + 1) / 2)
  )
  # L-strip polygons (p10 on one axis, split at the median on the other).
  strip_poly <- function(id) {
    v <- switch(id,
      L1 = list(x = c(0, Mc, Mc, Lc, Lc), y = c(0, 0, Lc, Lc, 0)),        # bottom, sparse half
      L2 = list(x = c(Mc, 1, 1, Mc), y = c(0, 0, Lc, Lc)),               # bottom, dense half
      L3 = list(x = c(0, Lc, Lc, 0), y = c(Lc, Lc, Mc, Mc)),             # left, infrequent half
      L4 = list(x = c(0, Lc, Lc, 0), y = c(Mc, Mc, 1, 1)))              # left, frequent half
    data.frame(x = v$x, y = v$y, id = id)
  }
  strips <- do.call(rbind, lapply(c("L1", "L2", "L3", "L4"), strip_poly))
  strip_delta <- c(L1 = deltas$L1, L2 = deltas$L2, L3 = deltas$L3, L4 = deltas$L4)
  strips$delta <- strip_delta[strips$id]
  strip_lab <- data.frame(
    id = c("L1", "L2", "L3", "L4"),
    lx = c(Mc / 2, (Mc + 1) / 2, Lc / 2, Lc / 2),
    ly = c(Lc / 2, Lc / 2, (Lc + Mc) / 2, (Mc + 1) / 2),
    delta = strip_delta,
    angle = c(0, 0, 90, 90)
  )
  lab_col <- function(d) ifelse(abs(d) > vmax * 0.4, "white", "black")

  ggplot() +
    geom_rect(data = quads, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = delta), color = NA) +
    geom_polygon(data = strips, aes(x = x, y = y, group = id, fill = delta), color = NA) +
    geom_text(data = quads, aes(lx, ly, label = sprintf("%+.1f", delta)), color = lab_col(quads$delta), fontface = "bold", size = 6) +
    geom_text(data = strip_lab, aes(lx, ly, label = sprintf("%+.1f", delta), angle = angle), color = lab_col(strip_lab$delta), fontface = "bold", size = 4) +
    geom_segment(aes(x = Mc, xend = Mc, y = 0, yend = 1)) +
    geom_segment(aes(x = 0, xend = 1, y = Mc, yend = Mc)) +
    geom_segment(aes(x = Lc, xend = 1, y = Lc, yend = Lc), linetype = "dashed", color = "grey60", linewidth = 0.3) +
    geom_segment(aes(x = Lc, xend = Lc, y = Lc, yend = 1), linetype = "dashed", color = "grey60", linewidth = 0.3) +
    scale_fill_gradientn(colors = HEATMAP_COLORS, limits = c(-vmax, vmax), oob = scales::squish,
                         name = sprintf("Δ %s (%%)", metric_label)) +
    scale_x_continuous(breaks = c(0.25, 0.75), labels = c(xl, xh), expand = c(0, 0)) +
    scale_y_continuous(breaks = c(0.25, 0.75), labels = c(yl, yh), expand = c(0, 0)) +
    coord_equal() +
    labs(x = "Sampling effort", y = "Relative prevalence") +
    theme_minimal(base_size = 14) +
    theme(panel.grid = element_blank(),
          axis.text = element_text(face = "italic"),
          axis.title = element_text(face = "bold"))
  ggsave(out_path, width = 7, height = 6, dpi = 200)
}

# =========================================================================
main <- function(args) {
  a <- parse_args(args)
  metrics <- strsplit(a$metrics, ",")[[1]]
  dir.create(a$`output-dir`, showWarnings = FALSE, recursive = TRUE)

  proxies <- read.csv(a$proxies)
  n_species <- nrow(proxies)

  cat(sprintf("[eval] metrics for %s ...\n", a$predictions))
  pred <- read_predictions(a$predictions, n_species)
  if (is.null(pred$targets)) stop("predictions file has no 'targets' dataset")
  res <- per_species_metrics(pred$probabilities, pred$targets, metrics)

  ref <- NULL
  if (!is.null(a$reference)) {
    cat(sprintf("[eval] metrics for reference %s ...\n", a$reference))
    refp <- read_predictions(a$reference, n_species)
    ref <- per_species_metrics(refp$probabilities, pred$targets, metrics)
  }

  # Per-species CSV.
  df <- data.frame(species_id = seq_len(n_species) - 1)
  for (m in metrics) {
    df[[m]] <- res[[m]]
    if (!is.null(ref)) { df[[paste0(m, "_reference")]] <- ref[[m]]; df[[paste0(m, "_delta")]] <- res[[m]] - ref[[m]] }
  }
  write.csv(df, file.path(a$`output-dir`, "species_metrics.csv"), row.names = FALSE)

  # Macro summary.
  cat("\nmetric        macro(%)", if (!is.null(ref)) "   reference   delta" else "", "\n")
  for (m in metrics) {
    mv <- mean(res[[m]], na.rm = TRUE) * 100
    line <- sprintf("%-12s  %7.3f", m, mv)
    if (!is.null(ref)) { rv <- mean(ref[[m]], na.rm = TRUE) * 100; line <- paste0(line, sprintf("   %8.3f  %+6.3f", rv, mv - rv)) }
    cat(line, "\n")
  }

  # Stratification.
  xcol <- "sampling_effort"; ycol <- "relative_prevalence"
  if (all(c(xcol, ycol) %in% names(proxies))) {
    x <- proxies[[xcol]]; y <- proxies[[ycol]]
    fin <- is.finite(x) & is.finite(y)
    se_rank <- rank(x[fin], ties.method = "first")
    tp_rank <- rank(y[fin], ties.method = "first")
    sub <- region_masks(se_rank, tp_rank, sum(fin), decile = 10)
    full <- function(sm) { z <- rep(FALSE, n_species); z[which(fin)[sm]] <- TRUE; z }
    lab <- axis_labels(xcol); xl <- lab[1]; xh <- lab[2]
    lab <- axis_labels(ycol); yl <- lab[1]; yh <- lab[2]

    strata <- list(
      c(sprintf("%s-%s", xl, yl), "bottom_left"), c(sprintf("%s-%s", xl, yh), "top_left"),
      c(sprintf("%s-%s", xh, yl), "bottom_right"), c(sprintf("%s-%s", xh, yh), "top_right"),
      c(sprintf("%s(p10) x %s-half", xl, yl), "L3"), c(sprintf("%s(p10) x %s-half", xl, yh), "L4"),
      c(sprintf("%s(p10) x %s-half", yl, xl), "L1"), c(sprintf("%s(p10) x %s-half", yl, xh), "L2")
    )
    cat(sprintf("\nStratified %s (%%) — x=%s, y=%s\n", if (is.null(ref)) "mean" else "delta vs reference", xcol, ycol))
    hm_metric <- metrics[1]
    strat_delta <- list()
    for (st in strata) {
      mask <- full(sub[[st[2]]])
      cells <- sapply(metrics, function(m) cell_value(res[[m]], if (is.null(ref)) NULL else ref[[m]], mask))
      strat_delta[[st[2]]] <- cells[[hm_metric]]
      cat(sprintf("  %-30s %6d  %s\n", st[1], sum(mask),
                  paste(sprintf("%8.2f", cells), collapse = "")))
    }

    # Coarse delta heatmap (needs a reference).
    if (!is.null(ref)) {
      delta_heatmap_coarse(strat_delta, xl, xh, yl, yh, toupper(hm_metric),
                           out_path = file.path(a$`output-dir`, "delta_heatmap.png"))
      cat(sprintf("\n[delta heatmap] %s: wrote %s\n", hm_metric, file.path(a$`output-dir`, "delta_heatmap.png")))
    }
  }
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) main(args)
