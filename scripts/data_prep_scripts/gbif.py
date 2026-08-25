import rootutils

import os
import pandas as pd
import yaml
from tqdm import tqdm
from pygbif import species as species
from pygbif import occurrences as occ
import time
import subprocess
import glob
import geopandas as gpd
from shapely.geometry import MultiPolygon

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

#get root
root = os.getenv("PROJECT_ROOT")

cleaning_gbif_script_path = f"{root}/scripts/data_prep_scripts/cleaning_gbif.r"

with open(f"{root}/configs/local_default.yaml") as stream:
    data_dir = yaml.safe_load(stream)['data_dir']
    data_dir = data_dir.split("/")[:-2]
    data_dir = "/".join(data_dir) + "/" + "data_raw/"


# Function to process the chunks and write per species CSV files
def process_csv(input_csv, chunk_size):
    total_rows = 332_216_666

    # Keep file handles open in a dictionary
    file_handles = {}

    chunks = pd.read_csv(
        input_csv,
        chunksize=chunk_size,
        sep="\t",
        low_memory=False,
        on_bad_lines='skip'
    )

    with tqdm(total=total_rows) as pbar:
        for chunk in chunks:
            # Extract species name
            chunk['species'] = (
                chunk['scientificName']
            )

            # Sort by species so writes are in contiguous blocks
            chunk.sort_values("species", inplace=True)

            # Now group by species but only iterate keys (faster after sort)
            for species, species_group in chunk.groupby("species", sort=False):
                if not species:  # skip empty
                    continue

                species_file = f"{data_dir}gbif/per_species/{species}.csv"

                # Open handle once, reuse
                if species not in file_handles:
                    file_handles[species] = open(species_file, "a", encoding="utf-8")

                    # If new file, write header
                    if os.stat(species_file).st_size == 0:
                        species_group.to_csv(file_handles[species], header=True, index=False)
                        continue

                # Append without header
                species_group.to_csv(file_handles[species], header=False, index=False)

            pbar.update(len(chunk))

    # Close all open handles
    for f in file_handles.values():
        f.close()


skip = False
#check if file exists
if not os.path.exists(f"{data_dir}splotopen/splot_metadata/final_species.txt"):
    print("Species file missing, please run the splotopen preprocessing script first")
else:
    splot_species = pd.read_csv(
        f"{data_dir}splotopen/splot_metadata/final_species.txt", header=None, names=["Species"]
    ).values.flatten()


if not os.path.exists(f"{data_dir}gbif/gbif_metadata"):
    os.makedirs(f"{data_dir}gbif/gbif_metadata")

# only looking for species name
gbif_metadata_path = f"{data_dir}gbif/gbif_metadata/gbif_metadata.csv"
if not os.path.exists(gbif_metadata_path) and not skip:
    for i, s in tqdm(enumerate(splot_species), total=len(splot_species)):
        data = species.name_backbone(name=s)
        if i == 0:
            columns = list(data.keys())
            df_gbif = pd.DataFrame(columns=columns)
        data = pd.DataFrame(data, index=[0])
        df_gbif = pd.concat([df_gbif,data], ignore_index=True)
    df_gbif["extraction_name"] = splot_species
    df_gbif.to_csv(gbif_metadata_path, index=False)
elif os.path.exists(gbif_metadata_path):
    df_gbif = pd.read_csv(gbif_metadata_path)
else:
    print(f"GBIF metadata file not found at {gbif_metadata_path} and skip=True. "
          "Set skip=False to download from GBIF backbone.")
    exit(1)

skip = True


#extracting by speciesKey
print(f"Number of species found in GBIF backbone: {df_gbif.shape[0]}")
subset = df_gbif[["extraction_name", "speciesKey"]]
subset = subset[subset["speciesKey"].isnull()]
print(f"Number of species not found in GBIF backbone: {subset.shape[0]}")
subset.to_csv(f"{data_dir}gbif/gbif_metadata/missing.csv", index=False)



df_gbif = df_gbif.dropna(subset=["speciesKey"])



# Check if there is a file with numbers ending in .zip
zip_files = glob.glob(f"{data_dir}gbif/*[0-9].zip")
if zip_files and not skip:
    key = os.path.basename(zip_files[0]).split('.')[0]
else:
    key = None



if not key and not skip:
    s = df_gbif['speciesKey'].astype(int)
    s = [str(i) for i in s]

    # GBIF downloads require an account: https://www.gbif.org/user/profile
    # Provide credentials via environment variables:
    #   export GBIF_USER=...  GBIF_PWD=...  GBIF_EMAIL=...
    gbif_user = os.environ.get("GBIF_USER")
    gbif_pwd = os.environ.get("GBIF_PWD")
    gbif_email = os.environ.get("GBIF_EMAIL")
    if not (gbif_user and gbif_pwd and gbif_email):
        raise RuntimeError(
            "GBIF credentials not found. Set the GBIF_USER, GBIF_PWD and GBIF_EMAIL "
            "environment variables (create a free account at https://www.gbif.org/)."
        )
    key = occ.download(f"taxonKey in {s}", user=gbif_user, pwd=gbif_pwd, email=gbif_email)
    print(f"Download key: {key}")
    while True:
        try:
            occ.download_get(key, path=f"{data_dir}gbif/")
            print("Download completed.")
            break  # Exit the loop once successful
        except Exception:
            print("Download not ready yet. Retrying in 60 seconds... ")

        time.sleep(60)


skip = True
key = "0033085-260208012135463"
#occ.download_get(key, path=f"{data_dir}gbif/")
#unzip
if not os.path.exists(f"{data_dir}gbif/{key}.csv") and not skip:
    os.system(f"unzip {data_dir}gbif/{key}.zip -d {data_dir}gbif/")
if not os.path.exists(f"{data_dir}gbif/per_species") and not skip:
    os.makedirs(f"{data_dir}gbif/per_species")

    #split into per species csv files by scientificName
    print("Splitting the data into per-species csv files")
    process_csv(f"{data_dir}gbif/{key}.csv", chunk_size = 100000)


#get all species in gbif and write to file
species_files = os.listdir(f"{data_dir}gbif/per_species")
species_files = [os.path.splitext(x)[0] for x in species_files]
species_files = sorted(set(species_files))
#write to file
with open(f"{data_dir}gbif/gbif_metadata/gbif_original_species.txt", 'w') as f:
    for s in species_files:
        clean = str(s).strip().replace("\t", " ")  # strip whitespace + remove tabs
        f.write(clean + "\n")

########################################
# Resolve GBIF species names to WCVP accepted names
resolve_names_script_path = f"{root}/scripts/data_prep_scripts/resolve_names.r"
if not os.path.exists(f"{data_dir}gbif/gbif_metadata/matched_names.csv"):
    print("Resolving GBIF species names to WCVP")
    result = subprocess.run(
        ["conda", "run", "-n", "sdm", "Rscript", resolve_names_script_path,
         f"{data_dir}gbif/gbif_metadata/gbif_original_species.txt"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"resolve_names.r failed with return code {result.returncode}. Aborting.")
        exit(1)

########################################

if not os.path.exists(f"{data_dir}gbif/gbif_resolved"):
    os.makedirs(f"{data_dir}gbif/gbif_resolved")

    print("Resolving names and merging per species files")
    #resolve names
    resolved_names = pd.read_csv(f"{data_dir}gbif/gbif_metadata/matched_names.csv")
    resolved_names = resolved_names.groupby("accepted_taxon_name")["splot_name"].apply(list).reset_index()
    #NOTE only keep species in splotopen
    resolved_names = resolved_names[resolved_names.accepted_taxon_name.isin(splot_species)]


    for i, (index, row) in tqdm(enumerate(resolved_names.iterrows()), total=len(resolved_names)):
        splot_names = row["splot_name"]
        accepted_taxon_name = row["accepted_taxon_name"]

        # Create a new DataFrame to hold the concatenated data
        combined_df = pd.DataFrame()

        for splot_name in splot_names:
            file_path = f"{data_dir}gbif/per_species/{splot_name}.csv"
            if os.path.exists(file_path):
                temp_df = pd.read_csv(file_path, low_memory=False)
                temp_df["gbif_original_name"] = splot_name
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

        combined_df["species"] = accepted_taxon_name
        # Save the combined DataFrame to a new CSV file
        output_file = f"{data_dir}gbif/gbif_resolved/{accepted_taxon_name}.csv"
        combined_df.to_csv(output_file, index=False)


#remove files if they are in subset (species not found in GBIF backbone)
subset = pd.read_csv(f"{data_dir}gbif/gbif_metadata/missing.csv")
missing_species = subset["extraction_name"].values.flatten()
for s in missing_species:
    file_path = f"{data_dir}gbif/gbif_resolved/{s}.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed file {file_path} as species could not be found in GBIF backbone")

#write final species names to file
full_names = os.listdir(f"{data_dir}gbif/gbif_resolved")
full_names = [os.path.splitext(x)[0] for x in full_names]
full_names = sorted(set(full_names))
with open(f"{data_dir}gbif/gbif_metadata/final_species.txt", 'w') as f:
    for s in full_names:
        clean = str(s).strip().replace("\t", " ")  # strip whitespace + remove tabs
        f.write(clean + "\n")

#compare the species in the species file with the species in the gbif folder
#and report for how many we could not find a match
species_files = os.listdir(f"{data_dir}gbif/per_species")
species_files = [os.path.splitext(x)[0] for x in species_files]
species_files = set(species_files)
#compare to full_names
missing_species = set(full_names) - species_files
print(f"Could not find a match for {len(missing_species)} species.")


if not os.path.exists(f"{data_dir}gbif/gbif_clean"):
    os.makedirs(f"{data_dir}gbif/gbif_clean")
    result = subprocess.run(
        ["conda", "run", "-n", "sdm", "Rscript", cleaning_gbif_script_path,
         f"{data_dir}gbif/gbif_resolved", f"{data_dir}gbif/gbif_metadata/final_species.txt"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"R script failed with return code {result.returncode}. Removing gbif_clean to allow retry.")
        import shutil
        shutil.rmtree(f"{data_dir}gbif/gbif_clean")
        exit(1)



if not os.path.exists(f"{data_dir}gbif/gbif_powo_all.csv"):
    files = os.listdir(f"{data_dir}gbif/gbif_clean/")
    files = [x for x in files if x != "erronous_species.csv"]
    print("Merging cleaned files and writing to disk")
    for i, file in enumerate(tqdm(files)):
        df = pd.read_csv(f"{data_dir}gbif/gbif_clean/{file}")
        if i == 0:
            df.to_csv(f"{data_dir}gbif/gbif_powo_all.csv", index=False)
        else:
            df.to_csv(f"{data_dir}gbif/gbif_powo_all.csv", mode='a', header=False, index=False)


# Filter splotopen_processed.csv to only keep species present in GBIF
# and recover empty locations (like in splotopen.py)
from geo_utils import latlon_to_grid_cell, grid_cell_to_latlon

splotopen_processed_path = f"{data_dir}splotopen/splotopen_processed.csv"
if os.path.exists(splotopen_processed_path):
    print("\nFiltering splotopen data to only keep species present in GBIF...")
    df_splot = pd.read_csv(splotopen_processed_path)

    # Read final GBIF species list from gbif_clean (post-cleaning, excludes erroneous species)
    gbif_clean_files = os.listdir(f"{data_dir}gbif/gbif_clean/")
    gbif_clean_files = [f for f in gbif_clean_files if f != "erronous_species.csv"]
    gbif_species = {os.path.splitext(f)[0] for f in gbif_clean_files}

    # Save all unique locations before filtering (including empty ones)
    df_all_locations = df_splot[["Latitude", "Longitude"]].drop_duplicates()

    # Remove species entries not found in GBIF
    splot_species_before = df_splot.Species.dropna().nunique()
    df_splot_filtered = df_splot[df_splot.Species.isna() | df_splot.Species.isin(gbif_species)]
    splot_species_after = df_splot_filtered.Species.dropna().nunique()
    print(f"Removed {splot_species_before - splot_species_after} species not found in GBIF "
          f"({splot_species_before} -> {splot_species_after})")

    # Recover locations where all species were removed (all-negative vectors for test set)
    # Use 1km grid cells to avoid duplicates, consistent with splotopen.py
    # Grid all original locations
    all_locs = df_all_locations[["Longitude", "Latitude"]].values
    all_grid_x, all_grid_y = latlon_to_grid_cell(all_locs[:, 0], all_locs[:, 1])
    all_center_lon, all_center_lat = grid_cell_to_latlon(all_grid_x, all_grid_y)

    df_locations_grid = pd.DataFrame({
        "Latitude": all_center_lat,
        "Longitude": all_center_lon,
        "grid_cell": list(zip(all_grid_x, all_grid_y)),
    })
    df_locations_unique = df_locations_grid.drop_duplicates(subset=["grid_cell"])

    # Find grid cells already covered by the filtered data
    final_locs = df_splot_filtered[["Longitude", "Latitude"]].dropna().values
    final_grid_x, final_grid_y = latlon_to_grid_cell(final_locs[:, 0], final_locs[:, 1])
    final_grid_cells = set(zip(final_grid_x, final_grid_y))

    # Keep only grid cells not already in df_splot_filtered
    missing_mask = ~df_locations_unique["grid_cell"].isin(final_grid_cells)
    missing_locations = df_locations_unique.loc[missing_mask, ["Latitude", "Longitude"]].copy()

    print(f"Adding {len(missing_locations)} empty locations (unique 1km cells with no surviving species)")
    df_splot_filtered = pd.concat([df_splot_filtered, missing_locations], ignore_index=True)

    # Save as new final version (does not overwrite the original)
    splotopen_final_path = f"{data_dir}splotopen/splotopen_processed_final.csv"
    df_splot_filtered.to_csv(splotopen_final_path, index=False, sep=",")
    print(f"Saved {splotopen_final_path}")
    print(f"Final: {df_splot_filtered.Species.dropna().nunique()} species, {len(df_splot_filtered)} rows")

    # Write final species list (does not overwrite the original)
    splot_final_species = sorted([s for s in df_splot_filtered.Species.unique() if pd.notna(s)])
    final_species_path = f"{data_dir}splotopen/splot_metadata/final_species_gbif_filtered.txt"
    with open(final_species_path, "w", encoding="utf-8") as f:
        for s in splot_final_species:
            clean = str(s).strip().replace("\t", " ")
            f.write(clean + "\n")
    print(f"Saved {final_species_path} with {len(splot_final_species)} species")
else:
    print(f"Warning: {splotopen_processed_path} not found. Run splotopen.py first.")


# Generate missing POWO range shapefiles for GBIF species
# splotopen.py generates range maps only for sPlot-resolved species.
# The GBIF pipeline resolves names independently via WCVP — some accepted
# names may differ

# data_dir in gbif.py points to data_raw — derive the processed data dir
with open(f"{root}/configs/local_default.yaml") as stream:
    processed_data_dir = yaml.safe_load(stream)['data_dir']

range_output_dir = os.path.join(processed_data_dir, "native_ranges")
os.makedirs(range_output_dir, exist_ok=True)

# Also output to data_raw/powo for consistency with splotopen.py
raw_range_dir = f"{data_dir}powo/native_ranges/"
os.makedirs(raw_range_dir, exist_ok=True)

# Final GBIF species list (from cleaned files)
gbif_clean_dir = f"{data_dir}gbif/gbif_clean/"
gbif_clean_files = [f for f in os.listdir(gbif_clean_dir) if f != "erronous_species.csv" and f.endswith(".csv")]
all_gbif_species = sorted({os.path.splitext(f)[0] for f in gbif_clean_files})

# Find species missing from the native_ranges directory
existing_ranges = {os.path.splitext(f)[0] for f in os.listdir(range_output_dir) if f.endswith(".shp")}
missing_species = [s for s in all_gbif_species if s not in existing_ranges]

if not missing_species:
    print("\nAll GBIF species already have range shapefiles.")
else:
    print(f"\nGenerating {len(missing_species)} missing range shapefiles from WCVP...")

    # Load WCVP data (same as splotopen.py)
    wcvp_dist_path = f"{data_dir}powo/wcvp_distribution.csv"
    wcvp_names_path = f"{data_dir}powo/wcvp_names.csv"
    l3_shp_path = f"{data_dir}maps/level3/level3.shp"

    wcvp_dist = pd.read_csv(wcvp_dist_path, delimiter="|")
    wcvp_names = pd.read_csv(wcvp_names_path, delimiter="|")
    wcvp_names = wcvp_names[(wcvp_names.taxon_rank == "Species") & (wcvp_names.taxon_status == "Accepted")]
    country_gdf = gpd.read_file(l3_shp_path)

    clean_dist = wcvp_dist[
        (wcvp_dist.introduced == 0) &
        (wcvp_dist.extinct == 0) &
        (wcvp_dist.location_doubtful == 0)
    ]

    generated = 0
    skipped_no_range = 0
    for s in tqdm(missing_species, desc="Generating range shapefiles"):
        names_data = wcvp_names[wcvp_names.taxon_name == s]
        if names_data.empty:
            skipped_no_range += 1
            continue

        plant_ids = names_data.plant_name_id.values
        areas = clean_dist[clean_dist.plant_name_id.isin(plant_ids)].area_code_l3.unique().tolist()
        if not areas:
            skipped_no_range += 1
            continue

        polys = country_gdf[country_gdf.LEVEL3_COD.isin(areas)].geometry
        if polys.empty:
            skipped_no_range += 1
            continue

        merged = polys.union_all()

        if merged.geom_type == "GeometryCollection":
            merged = MultiPolygon([geom for geom in merged.geoms if geom.geom_type in ["Polygon", "MultiPolygon"]])
        elif merged.geom_type == "Polygon":
            merged = MultiPolygon([merged])

        gdf = gpd.GeoDataFrame(index=[0], crs=country_gdf.crs, geometry=[merged])

        # Save to both directories
        gdf.to_file(os.path.join(range_output_dir, f"{s}.shp"), driver="ESRI Shapefile")
        gdf.to_file(os.path.join(raw_range_dir, f"{s}.shp"), driver="ESRI Shapefile")
        generated += 1

    print(f"Generated {generated} new range shapefiles, {skipped_no_range} species had no WCVP range data")


