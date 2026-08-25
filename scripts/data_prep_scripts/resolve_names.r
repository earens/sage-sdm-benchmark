# Resolve GBIF species names to WCVP accepted names
# ---------------------------------------------------------------------------
# We resolve ALL GBIF scientificName values ourselves against WCVP, independent
# of GBIF's backbone taxonomy. This ensures taxonomic consistency with our
# evaluation data (sPlot), which is also resolved against WCVP.
#
# Strategy:
#   Input: list of unique scientificName values from GBIF per-species files.
#   1. Parse each name into taxon name + author string
#   2. Deduplicate by parsed taxon name (32k unique from 83k filenames)
#   3. Exact match with author info first (fast), strict fuzzy on remainder
#   4. Only accept homotypic synonyms 
#   5. Map original filenames → WCVP accepted name via parsed taxon name
# ---------------------------------------------------------------------------

library(tidyverse)
library(rWCVP)

# ---------------------------------------------------------------------------
# 1. Read species names (the per-species filenames = GBIF scientificName values)
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
  species_path <- args[1]
} else {
  stop("Usage: Rscript resolve_names.r <path/to/gbif_original_species.txt>")
}
output_dir <- dirname(species_path)

raw_names <- readLines(species_path)
cat(length(raw_names), "scientificName values read from file\n")

species_data <- data.frame(
  original_name = raw_names,
  stringsAsFactors = FALSE
)

# ---------------------------------------------------------------------------
# 2. Parse each scientificName into taxon name + author
#    Examples:
#      "Abies alba Mill."                         → name="Abies alba", author="Mill."
#      "Abies alba subsp. alba"                   → name="Abies alba subsp. alba", author=NA
#      "Abies alba var. brevifolia Mattf."         → name="Abies alba var. brevifolia", author="Mattf."
#      "Abies alba f. pyramidalis (Carrière) Voss" → name="Abies alba f. pyramidalis", author="(Carrière) Voss"
# ---------------------------------------------------------------------------
parse_scientific_name <- function(name) {
  parts <- str_split(name, "\\s+")[[1]]

  if (length(parts) <= 2) {
    return(data.frame(parsed_name = name, parsed_author = NA_character_,
                      stringsAsFactors = FALSE))
  }

  # Check for infraspecific rank marker at position 3
  has_infra <- length(parts) >= 4 &&
    str_detect(parts[3], "^(subsp|var|f|subvar|forma)\\.?$")

  if (has_infra && length(parts) >= 4) {
    taxon_name <- paste(parts[1:4], collapse = " ")
    author <- if (length(parts) > 4) paste(parts[5:length(parts)], collapse = " ") else NA_character_
  } else {
    taxon_name <- paste(parts[1:2], collapse = " ")
    author <- paste(parts[3:length(parts)], collapse = " ")
  }

  data.frame(parsed_name = taxon_name, parsed_author = author,
             stringsAsFactors = FALSE)
}

cat("Parsing scientific names...\n")
parsed <- bind_rows(lapply(species_data$original_name, parse_scientific_name))
species_data$parsed_name   <- parsed$parsed_name
species_data$parsed_author <- parsed$parsed_author

# ---------------------------------------------------------------------------
# 3. Deduplicate by parsed taxon name for matching
#    Many filenames differ only in author string — rWCVP matches by taxon name
#    and uses author for disambiguation, so we deduplicate by taxon name and
#    keep ONE representative author string per name (prefer non-NA).
# ---------------------------------------------------------------------------
unique_taxa <- species_data %>%
  group_by(parsed_name) %>%
  summarise(
    # Pick the most informative author (non-NA, longest string)
    parsed_author = {
      authors <- parsed_author[!is.na(parsed_author)]
      if (length(authors) > 0) authors[which.max(nchar(authors))] else NA_character_
    },
    n_original = n(),
    .groups = "drop"
  )

cat(nrow(unique_taxa), "unique taxon names to resolve (from",
    nrow(species_data), "original names)\n")

# ---------------------------------------------------------------------------
# 4. STEP 1 — Exact matching (fast) with author info
# ---------------------------------------------------------------------------
cat("\n--- Step 1: Exact matching ---\n")

match_df <- data.frame(
  splot_name   = unique_taxa$parsed_name,
  splot_author = unique_taxa$parsed_author,
  stringsAsFactors = FALSE
)

exact_results <- wcvp_match_names(
  match_df,
  name_col   = "splot_name",
  author_col = "splot_author",
  fuzzy = FALSE,
  progress_bar = TRUE
)

matched_exact   <- exact_results %>% filter(!is.na(wcvp_id))
unmatched_exact <- exact_results %>% filter(is.na(wcvp_id)) %>%
  distinct(splot_name, splot_author)

cat("  Exact hits:    ", nrow(matched_exact), "\n")
cat("  Still unmatched:", nrow(unmatched_exact), "\n")

# ---------------------------------------------------------------------------
# 5. STEP 2 — Strict fuzzy matching on unmatched names only
#    Only accept: edit_distance == 1 AND similarity >= 0.95
# ---------------------------------------------------------------------------
if (nrow(unmatched_exact) > 0) {
  cat("\n--- Step 2: Fuzzy matching on", nrow(unmatched_exact), "unmatched names ---\n")

  fuzzy_results <- wcvp_match_names(
    unmatched_exact,
    name_col   = "splot_name",
    author_col = "splot_author",
    fuzzy = TRUE,
    progress_bar = TRUE
  )

  # Separate true fuzzy from any extra exact matches
  fuzzy_only <- fuzzy_results %>%
    filter(is.na(match_type) | str_detect(match_type, "Fuzzy"))

  extra_exact <- fuzzy_results %>%
    filter(!is.na(match_type) & !str_detect(match_type, "Fuzzy"))

  # Strict filter: only 1 character edit + very high similarity
  good_fuzzy <- fuzzy_only %>%
    filter(!is.na(match_type),
           match_edit_distance == 1,
           match_similarity >= 0.95)

  cat("  Extra exact matches:", nrow(extra_exact), "\n")
  cat("  Good fuzzy matches (ed=1, sim>=0.95):", nrow(good_fuzzy), "\n")
  cat("  Rejected fuzzy:",
      sum(!is.na(fuzzy_only$match_type)) - nrow(good_fuzzy), "\n")

  all_matches <- bind_rows(matched_exact, extra_exact, good_fuzzy)
} else {
  all_matches <- matched_exact
}

cat("\nTotal matched rows before dedup:", nrow(all_matches), "\n")

# ---------------------------------------------------------------------------
# 6. Resolve multiple matches — homotypic synonyms only
# ---------------------------------------------------------------------------
cat("Resolving multiple matches (homotypic synonyms only)...\n")

resolve_multi <- function(df) {
  if (nrow(df) == 1) return(df)

  valid <- filter(df, !is.na(match_similarity))
  if (nrow(valid) == 0) return(head(df, 1))

  # 1) Prefer Accepted names
  accepted <- valid %>% filter(wcvp_status == "Accepted")
  if (nrow(accepted) >= 1) {
    return(accepted %>% arrange(desc(match_similarity)) %>% head(1))
  }

  # 2) Only accept homotypic synonyms
  homo_syns <- valid %>%
    filter(wcvp_status %in% c("Synonym", "Orthographic"),
           wcvp_homotypic == TRUE)
  if (nrow(homo_syns) >= 1) {
    return(homo_syns %>% arrange(desc(match_similarity)) %>% head(1))
  }

  # 3) If all remaining point to the same accepted_id, accept it
  if (n_distinct(valid$wcvp_accepted_id, na.rm = TRUE) == 1) {
    return(head(valid, 1))
  }

  # 4) Truly ambiguous — mark as unresolvable
  final <- head(valid, 1)
  final$match_type <- "Could not resolve multiple matches"
  final$wcvp_accepted_id <- NA_integer_
  final
}

resolved <- all_matches %>%
  nest_by(splot_name) %>%
  mutate(data = list(resolve_multi(data))) %>%
  unnest(col = data) %>%
  ungroup()

# ---------------------------------------------------------------------------
# 7. Look up accepted names in WCVP — homotypic filter
# ---------------------------------------------------------------------------
cat("Looking up accepted taxon names in WCVP...\n")

wcvp_lookup <- rWCVPdata::wcvp_names %>%
  select(plant_name_id, taxon_name, taxon_authors, taxon_rank, taxon_status,
         parent_plant_name_id)

accepted_matches <- resolved %>%
  filter(!is.na(wcvp_accepted_id)) %>%
  left_join(wcvp_lookup, by = c("wcvp_accepted_id" = "plant_name_id")) %>%
  filter(taxon_status == "Accepted") %>%
  filter(taxon_rank %in% c("Species", "Subspecies", "Variety")) %>%
  # Only keep homotypic synonyms (Accepted status always passes through)
  filter(wcvp_status == "Accepted" | wcvp_homotypic == TRUE)

# For subspecies/varieties, look up the parent species via WCVP's own
# parent_plant_name_id link.
parent_lookup <- wcvp_lookup %>%
  filter(taxon_rank == "Species", taxon_status == "Accepted") %>%
  select(parent_id = plant_name_id, parent_species_name = taxon_name)

accepted_matches <- accepted_matches %>%
  left_join(parent_lookup, by = c("parent_plant_name_id" = "parent_id")) %>%
  mutate(
    parent_species_name = case_when(
      # Already a species — use directly
      taxon_rank == "Species" ~ taxon_name,
      # Subspecies/variety with a resolved parent species — use it
      !is.na(parent_species_name) ~ parent_species_name,
      # Fallback: regex strip as last resort
      TRUE ~ sub("\\s+(var|subsp|f|subvar|forma)\\.?\\s+.*", "", taxon_name)
    )
  )

# Report how many used the parent link vs fallback
n_parent_link <- sum(accepted_matches$taxon_rank %in% c("Subspecies", "Variety") &
                     !is.na(accepted_matches$parent_plant_name_id))
n_fallback    <- sum(accepted_matches$taxon_rank %in% c("Subspecies", "Variety") &
                     is.na(accepted_matches$parent_plant_name_id))
cat("  Subspecies→parent via WCVP link:", n_parent_link, "\n")
cat("  Subspecies→parent via fallback:  ", n_fallback, "\n")
cat("  Accepted matches:", nrow(accepted_matches), "\n")

# ---------------------------------------------------------------------------
# 8. Build output — map back to ALL original GBIF filenames
#    Join via parsed_name (taxon name without author)
# ---------------------------------------------------------------------------
resolved_lookup <- accepted_matches %>%
  select(
    parsed_name  = splot_name,       # the deduplicated taxon name we matched on
    match_name   = wcvp_name,
    match_status = wcvp_status,
    match_type,
    wcvp_homotypic,
    accepted_plant_name_id = wcvp_accepted_id,
    accepted_taxon_name    = taxon_name,
    accepted_taxon_authors = taxon_authors,
    accepted_taxon_rank    = taxon_rank,
    parent_species_name
  )

# Join back to ALL original names via parsed_name
final_output <- species_data %>%
  left_join(resolved_lookup, by = "parsed_name") %>%
  filter(!is.na(accepted_taxon_name))

# splot_name column expected by downstream gbif.py
final_output$splot_name <- final_output$original_name

# Format for downstream compatibility
output_compat <- final_output %>%
  select(
    splot_name,
    match_name,
    match_status,
    match_type,
    accepted_plant_name_id,
    # Use parent species for subspecies/varieties (downstream expects binomial)
    accepted_taxon_name    = parent_species_name,
    accepted_taxon_authors,
    accepted_taxon_rank,
    full_accepted_taxon_name = accepted_taxon_name
  )

# Unmatched original names
unmatched <- species_data %>%
  filter(!original_name %in% output_compat$splot_name) %>%
  select(splot_name = original_name)

# ---------------------------------------------------------------------------
# 9. Summary & output
# ---------------------------------------------------------------------------
cat("\n=== Resolution Summary ===\n")
cat("Total input names:          ", nrow(species_data), "\n")
cat("Unique taxon names matched: ", nrow(unique_taxa), "\n")
cat("Resolved to WCVP:          ", n_distinct(resolved_lookup$parsed_name), "\n")
cat("Original names mapped:     ", n_distinct(output_compat$splot_name), "/", nrow(species_data), "\n")
cat("Unique accepted species:   ", n_distinct(output_compat$accepted_taxon_name), "\n")
cat("Unresolved original names: ", nrow(unmatched), "\n")

# Homotypic-only stats
homo_count <- resolved %>%
  filter(wcvp_status %in% c("Synonym", "Orthographic"), wcvp_homotypic == TRUE) %>%
  nrow()
hetero_rejected <- resolved %>%
  filter(wcvp_status %in% c("Synonym", "Orthographic"),
         wcvp_homotypic == FALSE | is.na(wcvp_homotypic)) %>%
  nrow()
cat("Homotypic synonyms used:   ", homo_count, "\n")
cat("Heterotypic synonyms dropped:", hetero_rejected, "\n")

write.csv(output_compat, file = file.path(output_dir, "matched_names.csv"), row.names = FALSE)
write.csv(unmatched,     file = file.path(output_dir, "unmatched_names.csv"), row.names = FALSE)

cat("Results written to", output_dir, "\n")
