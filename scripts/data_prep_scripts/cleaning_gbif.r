library(CoordinateCleaner)
library(countrycode)
library(dplyr)
library(ggplot2)
library(rgbif)
library(sf)
library(data.table)
library(pbapply)
library(rWCVP)
library(kewr)
library(future.apply)  

#NOTE if called from outside (i.e. within python script)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript cleaning_gbif.r /path/to/species_folder /path/to/species_of_interest.txt")
}
species_folder <- args[1]
species_of_interest_file <- args[2]
clean_species_folder <- file.path(dirname(species_folder), "gbif_clean")

# wcvp_distribution() below matches a name against every WCVP row carrying it and unions
# their distributions, including rows whose taxon_status is not Accepted. Where a homonym is
# an Unplaced/Illegitimate name for a different plant, that union hands the species range it
# does not have (e.g. Erysimum diffusum picking up Switzerland from an Unplaced homonym),
# and occurrences there survive cleaning. The WCVP tables downloaded by splotopen.py carry
# taxon_status, so restrict each species to the regions of its ACCEPTED row -- the same rule
# gbif.py uses to build the native-range shapefiles, keeping cleaning and the range masks on
# one definition instead of two.
wcvp_dir <- if (length(args) >= 3) args[3] else file.path(dirname(species_folder), "powo")
wcvp_names <- fread(file.path(wcvp_dir, "wcvp_names.csv"), sep = "|", quote = "")
wcvp_dist <- fread(file.path(wcvp_dir, "wcvp_distribution.csv"), sep = "|", quote = "")

accepted_ids <- wcvp_names[taxon_rank == "Species" & taxon_status == "Accepted",
                           .(plant_name_id, taxon_name)]
accepted_dist <- merge(
  wcvp_dist[introduced == 0 & extinct == 0 & location_doubtful == 0,
            .(plant_name_id, area_code_l3)],
  accepted_ids, by = "plant_name_id"
)
accepted_l3 <- split(accepted_dist$area_code_l3, accepted_dist$taxon_name)

# Which column of wcvp_distribution()'s output holds the L3 code is an assumption we would
# rather not bake in, so identify it by content: pick the column whose values are actually
# WCVP L3 codes. Fails loudly if none is, instead of silently leaving the range unfiltered.
all_l3_codes <- unique(wcvp_dist$area_code_l3)
find_l3_column <- function(x) {
  for (col in names(x)) {
    v <- x[[col]]
    if (!is.character(v) && !is.factor(v)) next
    v <- unique(as.character(v))
    if (length(v) > 0 && all(v %in% all_l3_codes)) return(col)
  }
  NULL
}

#only process missing speciess
skip_existing <- TRUE 

#NOTE expects per species csvs, where the species name is the file name
process_species_data <- function(species_name) {
  
  species_csv <- species_csv <- file.path(species_folder, paste0(species_name, ".csv"))
  dat <- read.csv(species_csv)
  dat <- dat %>%
    dplyr::select(species, decimalLongitude, decimalLatitude, countryCode, individualCount,
                  gbifID, family, taxonRank, coordinateUncertaintyInMeters,
                  year, basisOfRecord, institutionCode, datasetKey) %>%
    filter(!is.na(decimalLongitude), !is.na(decimalLatitude))

  dat$countryCode <- countrycode(as.character(dat$countryCode), origin =  'iso2c', destination = 'iso3c')
  dat$decimalLatitude <- as.numeric(dat$decimalLatitude)
  dat$decimalLongitude <- as.numeric(dat$decimalLongitude)

  # Native Range
  native_range <- tryCatch({
    wcvp_distribution(species_name, taxon_rank = "species", introduced = FALSE, extinct = FALSE, location_doubtful = FALSE)
  }, error = function(e) {
    cat("Error accessing native range for species:", species_name, "\n")
    log_erroneous_species(species_name, "Error accessing native range")
    return(NULL)
  })

  if (is.null(native_range)) {
    return(NULL) 
  }

  if (is.null(native_range) || nrow(native_range) == 0 || !inherits(native_range, "sf")) {
    cat("No valid native range for species:", species_name, "\n")
    log_erroneous_species(species_name, "No valid native range")
    return(NULL)
  }

  # Drop regions contributed only by non-Accepted homonyms (see accepted_l3 above).
  keep_l3 <- accepted_l3[[species_name]]
  l3_col <- find_l3_column(sf::st_drop_geometry(native_range))
  if (is.null(l3_col)) {
    # Never skip silently: an unfiltered range is exactly the bug this guards against.
    stop("wcvp_distribution() returned no column of WCVP L3 codes; cannot filter to the ",
         "accepted taxon. Columns: ", paste(names(native_range), collapse = ", "))
  }
  if (is.null(keep_l3)) {
    cat("No accepted WCVP row for species:", species_name, "\n")
    log_erroneous_species(species_name, "No accepted WCVP row")
    return(NULL)
  }
  {
    range_l3 <- as.character(native_range[[l3_col]])
    dropped <- setdiff(unique(range_l3), keep_l3)
    if (length(dropped) > 0) {
      cat("Dropping non-accepted region(s) for", species_name, ":",
          paste(dropped, collapse = ", "), "\n")
    }
    native_range <- native_range[range_l3 %in% keep_l3, ]
    if (nrow(native_range) == 0) {
      cat("No accepted native range for species:", species_name, "\n")
      log_erroneous_species(species_name, "No accepted native range")
      return(NULL)
    }
  }

  # Basic Cleaning
  flags <- tryCatch({
    clean_coordinates(
      dat,
      lon = "decimalLongitude",
      lat = "decimalLatitude",
      species = "species",
      countries = "countryCode",
      test = c("capitals", "centroids", "duplicates", "equal", "gbif", "institutions", "seas", "urban", "validity", "zeros")
    )
  }, error = function(e) {
    cat("Error during cleaning coordinates for species:", species_name, "\n")
    log_erroneous_species(species_name, "Error during cleaning coordinates")
    return(NULL)
  })

  if (is.null(flags)) {
    return(NULL)
  }

  clean <- dat[flags$.summary,]
  # Coordinate Uncertainty and Basis of Record
  clean$coordinateUncertaintyInMeters <- as.numeric(clean$coordinateUncertaintyInMeters)
  clean <- clean %>% filter(coordinateUncertaintyInMeters / 1000 <= 1 | is.na(coordinateUncertaintyInMeters))
  clean <- clean %>% filter(basisOfRecord %in% c("HUMAN_OBSERVATION", "OBSERVATION", "PRESERVED_SPECIMEN"))

  if (nrow(clean) == 0) {
    cat("No valid data points after cleaning for species:", species_name, "\n")
    log_erroneous_species(species_name, "No valid data points after cleaning")
    return(NULL)
  }
  
  # Native Range Cleaning
  coords <- st_as_sf(clean, coords = c("decimalLongitude", "decimalLatitude"), crs = 4326)
  buffered_dist <- native_range 

  intersects_result <- st_intersects(coords,st_union(buffered_dist), sparse = FALSE)

  if (is.null(intersects_result) || length(intersects_result) == 0) {
    cat("No intersection found for species:", species_name, "\n")
    log_erroneous_species(species_name, "No valid data points in range")
    return(NULL)
  }
  clean$native <- intersects_result[, 1]
  clean <- filter(clean, native == TRUE)

  if (nrow(clean) == 0) {
    cat("No native occurrences found for species:", species_name, "\n")
    log_erroneous_species(species_name, "No valid data points after cleaning")
    return(NULL)
  }

  # Save the cleaned data
  write.csv(clean, file = paste0(clean_species_folder, "/", species_name, ".csv"), row.names = FALSE)
  return(clean)
}

# log failed species to disk
log_erroneous_species <- function(species_name, matched_name) {

  erronous_species_file <- file.path(dirname(species_folder), "gbif_clean", "erronous_species.csv")
  if (!file.exists(erronous_species_file)) {
    write.csv(data.frame(original_name = character(), matched_name = character(), stringsAsFactors = FALSE), 
              erronous_species_file, row.names = FALSE)
  }
  erronous_entry <- data.frame(original_name = species_name, matched_name = matched_name, stringsAsFactors = FALSE)
  write.table(erronous_entry, file = erronous_species_file, append = TRUE, col.names = FALSE, row.names = FALSE, sep = ",")
}

num_workers <- 16
plan(multisession, workers = num_workers)

if (!dir.exists(clean_species_folder)) {
  dir.create(clean_species_folder, recursive = TRUE, showWarnings = FALSE)
}

species_of_interest <- species_of_interest_file
# Read species of interest
species_of_interest <- readLines(species_of_interest)

species_files <- list.files(species_folder, pattern = "\\.csv$", full.names = FALSE)
species_names <- gsub("\\.csv$", "", species_files)

# Filter species names based on species of interest
species_names <- intersect(species_names, species_of_interest)#[1:10]

#print number of species to process
#cat(length(species_names), "species to process\n")


if (skip_existing) {
  cleaned_files <- list.files(clean_species_folder, pattern = "\\.csv$", full.names = FALSE)
  cleaned_species <- gsub("\\.csv$", "", cleaned_files)
  species_names <- setdiff(species_names, cleaned_species)
  cat(length(species_names), "species to process\n")

} 

results <- future_lapply(species_names, process_species_data)

plan(sequential)
