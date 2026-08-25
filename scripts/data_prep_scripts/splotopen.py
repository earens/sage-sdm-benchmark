

#trigger download from https://idata.idiv.de/ddm/Data/DownloadZip/3474?version=5806
import rootutils

import os
import pandas as pd
import yaml
import subprocess
from shapely.geometry import MultiPolygon
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from geo_utils import latlon_to_grid_cell, grid_cell_to_latlon



link = "https://idata.idiv.de/ddm/Data/DownloadZip/3474?version=5806"
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

#get root
root = os.getenv("PROJECT_ROOT")
resolve_names_script_path = f"{root}/scripts/data_prep_scripts/resolve_names.r"


with open(f"{root}/configs/local_default.yaml") as stream:
    data_dir = yaml.safe_load(stream)['data_dir']
    data_dir = data_dir.split("/")[:-2]
    data_dir = "/".join(data_dir) + "/" + "data_raw/"


#download data into data_raw/splotopen
if not os.path.exists(f"{data_dir}splotopen"):
    os.makedirs(f"{data_dir}splotopen")
    os.makedirs(f"{data_dir}splotopen/orig_data")
if not os.path.exists(f"{data_dir}splotopen/orig_data/splotopen.zip"):
    os.system(f"wget -O {data_dir}splotopen/orig_data/splotopen.zip {link}")

if not os.path.exists(f"{data_dir}splotopen/orig_data/sPlotOpen.RData"):

    os.system(f"unzip {data_dir}splotopen/orig_data/splotopen.zip -d {data_dir}splotopen/orig_data")

#read data
#sPlotOpen_CWM_CWV.csv contains the traits
#sPlotOpen_DT.csv species + cover
#sPlotOpen_header.csv plot information
#sPlotOpen_metadata.csv metadata
if not os.path.exists(f"{data_dir}splotopen/splotopen.csv"):

    df_names = ["DT", "header"]

    files = os.listdir(f"{data_dir}splotopen/orig_data")
    files = [x for x in files if x.endswith(".csv") or x.endswith(".txt")]

    #find the files that contain the names above
    files = [f"{data_dir}splotopen/orig_data/{file}" for name in df_names for file in files if name in file]

    dfs = [pd.read_csv(x, sep="\t",  encoding="ISO-8859-1") for x in files]
    df = pd.merge(dfs[0], dfs[1], on="PlotObservationID")

    #NOTE first cleaning steps
    #as close as we can get to all plants recorded
    df = df[df.Plant_recorded.isin(["Not specified", "All vascular plants"])]
    df = df[df.Location_uncertainty <= 1000]

    df.to_csv(f"{data_dir}splotopen/splotopen.csv", index=False, sep=";")
else:
    df = pd.read_csv(f"{data_dir}splotopen/splotopen.csv", sep=";", low_memory=False)

#NOTE to recover empty locations
df_locations = df[["Latitude", "Longitude"]].drop_duplicates()

if not os.path.exists(f"{data_dir}splotopen/splot_metadata"):
    os.makedirs(f"{data_dir}splotopen/splot_metadata")

    original_species = sorted(df.Original_species.unique())
    original_species_file = f"{data_dir}splotopen/splot_metadata/original_species.txt"

    # write species list to txt (cleaned, one per line)
    with open(original_species_file, "w", encoding="utf-8") as f:
        for s in original_species:
            clean = str(s).strip().replace("\t", " ")  # strip whitespace + remove tabs
            f.write(clean + "\n")

else:
    original_species_file = f"{data_dir}splotopen/splot_metadata/original_species.txt"
    with open(original_species_file) as f:
        original_species = [line.strip() for line in f.readlines()]

#check if resolved_species.txt exists
if not os.path.exists(f"{data_dir}splotopen/splot_metadata/matched_names.csv"):
    print("Resolving species names. This may take a while.")
    result = subprocess.run(
        ["Rscript", resolve_names_script_path, original_species_file],
        capture_output=True, text=True,
    )

    print("R Script Output:")
    print(result.stdout)
    if result.stderr:
        print("R Script Errors:")
        print(result.stderr)

#drop species that could not be resolved
unmatched_names = pd.read_csv(f"{data_dir}splotopen/splot_metadata/unmatched_names.csv")


df_resolved = df[~df.Original_species.isin(unmatched_names.splot_name)]

#rename resolved species
resolved_species = pd.read_csv(f"{data_dir}splotopen/splot_metadata/matched_names.csv")
#overwrite Species column with by getting the accepted_taxon_name from resolved_species (original name -> accepted name)
df_resolved = pd.merge(df_resolved, resolved_species, left_on="Original_species", right_on="splot_name")
df_resolved = df_resolved.drop(columns=["Species", "splot_name"])
df_resolved = df_resolved.rename(columns={"accepted_taxon_name": "Species"})
df_resolved = df_resolved.drop_duplicates(subset=["Species","Latitude", "Longitude"])


df_resolved_species = df_resolved.Species.unique()
#write species list to txt
with open(f"{data_dir}splotopen/splot_metadata/resolved_species.txt", "w", encoding="utf-8") as f:
    for s in sorted(df_resolved_species):
        clean = str(s).strip().replace("\t", " ")  # strip whitespace + remove tabs
        f.write(clean + "\n")

######### POWO - RANGES ##############################
# Auto-download TDWG Level 3 shapefile if missing
# Download entire level3 folder from GitHub
if not os.path.exists(f"{data_dir}maps"):
    os.makedirs(f"{data_dir}maps")
if not os.path.exists(f"{data_dir}maps/level3/level3.shp"):
    print("Downloading TDWG Level 3 shapefile folder...")
    os.makedirs(f"{data_dir}maps/level3", exist_ok=True)
    # Use GitHub's archive download for the entire level3 folder
    os.system(
        f"cd {data_dir}maps && git clone --depth 1 --filter=blob:none --sparse "
        "https://github.com/tdwg/wgsrpd.git temp_wgsrpd"
    )
    os.system(f"cd {data_dir}maps/temp_wgsrpd && git sparse-checkout set level3")
    os.system(f"cp -r {data_dir}maps/temp_wgsrpd/level3/* {data_dir}maps/level3/")
    os.system(f"rm -rf {data_dir}maps/temp_wgsrpd")

# Auto-download Natural Earth countries shapefile if missing
ne_countries_link = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
if not os.path.exists(f"{data_dir}maps/ne_110m_admin_0_countries.shp"):
    print("Downloading Natural Earth 110m countries shapefile...")
    os.system(f"wget -O {data_dir}maps/ne_110m_admin_0_countries.zip {ne_countries_link}")
    os.system(f"unzip {data_dir}maps/ne_110m_admin_0_countries.zip -d {data_dir}maps")

# Auto-download WCVP data if missing (just like splot)
wcvp_link = "https://sftp.kew.org/pub/data-repositories/WCVP/wcvp.zip"
if not os.path.exists(f"{data_dir}powo"):
    os.makedirs(f"{data_dir}powo")
if not os.path.exists(f"{data_dir}powo/wcvp.zip"):
    print("Downloading WCVP data...")
    os.system(f"wget -O {data_dir}powo/wcvp.zip {wcvp_link}")
if not os.path.exists(f"{data_dir}powo/wcvp_distribution.csv") or not os.path.exists(f"{data_dir}powo/wcvp_names.csv"):
    print("Extracting WCVP data...")
    os.system(f"unzip {data_dir}powo/wcvp.zip -d {data_dir}powo")

if not os.path.exists(f"{data_dir}powo/native_ranges/"):
    metadata = f"{data_dir}powo/wcvp_distribution.csv"
    names = f"{data_dir}powo/wcvp_names.csv"
    country_shapefile_dir = f"{data_dir}maps/level3/level3.shp"
    species = f"{data_dir}splotopen/splot_metadata/resolved_species.txt"

    metadata = pd.read_csv(metadata, delimiter="|")
    names = pd.read_csv(names, delimiter="|")
    names = names[names.taxon_rank == "Species"]
    names = names[names.taxon_status == "Accepted"]
    country_gdf = gpd.read_file(country_shapefile_dir)
    with open(species) as f:
        species = [line.strip() for line in f.readlines()]

    clean_metadata = metadata[metadata.introduced == 0]
    clean_metadata = clean_metadata[clean_metadata.extinct == 0]
    clean_metadata = clean_metadata[clean_metadata.location_doubtful == 0]

    destination = f"{data_dir}powo/native_ranges/"
    os.makedirs(destination, exist_ok=True)

    for s in tqdm(species):
        names_data = names[names.taxon_name == s]
        id = names_data.plant_name_id.values
        areas = clean_metadata[clean_metadata.plant_name_id.isin(id)].area_code_l3.unique().tolist()
        polys = country_gdf[country_gdf.LEVEL3_COD.isin(areas)].geometry
        merged = polys.union_all()
        # If it's a GeometryCollection, filter only Polygon-like geometries
        if merged.geom_type == "GeometryCollection":
            merged = MultiPolygon([geom for geom in merged.geoms if geom.geom_type in ["Polygon", "MultiPolygon"]])

        # If it's just a Polygon, wrap it
        elif merged.geom_type == "Polygon":
            merged = MultiPolygon([merged])
        gdf = gpd.GeoDataFrame(index=[0], crs=country_gdf.crs, geometry=[merged])
        gdf.to_file(f"{destination}{s}.shp", driver="ESRI Shapefile")

species_with_ranges = [f.split(".")[0] for f in os.listdir(f"{data_dir}powo/native_ranges/") if f.endswith(".shp")]
num_species = df_resolved.Species.nunique()
df_resolved = df_resolved[df_resolved.Species.isin(species_with_ranges)]
print(f"Number of species with ranges: {df_resolved.Species.nunique()} out of {num_species}")
print(f"Before range filtering: {df_resolved.Species.nunique()} species, {len(df_resolved)} occurrences")



# Convert occurrences to GeoDataFrame once
points_gdf = gpd.GeoDataFrame(
    df_resolved,
    geometry=gpd.points_from_xy(df_resolved.Longitude, df_resolved.Latitude),
    crs="EPSG:4326"
)

range_dir = f"{data_dir}powo/native_ranges/"
shapefiles = [f for f in os.listdir(range_dir) if f.endswith(".shp")]

results = []

for shp_file in tqdm(shapefiles[:], desc="Species ranges"):
    species_name = shp_file.replace(".shp", "")

    # subset points of this species only
    pts = points_gdf[points_gdf.Species == species_name]
    if pts.empty:
        continue

    # load this species' native range
    gdf = gpd.read_file(os.path.join(range_dir, shp_file))
    gdf["Species"] = species_name

    # spatial join: only keep points inside the range
    joined = gpd.sjoin(pts, gdf[["Species", "geometry"]],
                       how="inner", predicate="within")
    #rename species_left to Species
    joined = joined.rename(columns={"Species_left": "Species", "index_left": "index"})
    #remove species_right
    joined = joined.drop(columns=["Species_right", "index_right"])
    results.append(joined.drop(columns="geometry"))

df_resolved = pd.concat(results, ignore_index=True)

print(f"After range filtering: {df_resolved.Species.nunique()} species, {len(df_resolved)} occurrences")

# Aggregate locations to 1km grid before filtering
print("\nAggregating to 1km grid...")
locations = df_resolved[["Longitude", "Latitude"]].values
grid_x, grid_y = latlon_to_grid_cell(locations[:, 0], locations[:, 1])
grid_xy = np.stack([grid_x, grid_y], axis=1)

unique_cells, inverse_indices = np.unique(grid_xy, axis=0, return_inverse=True)
df_resolved["grid_cell_idx"] = inverse_indices

# Compute cell center coordinates (consistent with aggregate_to_1km_grid.py)
center_lon, center_lat = grid_cell_to_latlon(unique_cells[:, 0], unique_cells[:, 1])
cell_center_map = {i: (center_lon[i], center_lat[i]) for i in range(len(unique_cells))}

# For each grid cell and species, keep only one occurrence
df_aggregated = df_resolved.groupby(["Species", "grid_cell_idx"]).first().reset_index()

# Replace lat/lon with grid cell centers
df_aggregated["Longitude"] = df_aggregated["grid_cell_idx"].map(lambda i: cell_center_map[i][0])
df_aggregated["Latitude"] = df_aggregated["grid_cell_idx"].map(lambda i: cell_center_map[i][1])
df_aggregated = df_aggregated.drop(columns=["grid_cell_idx"])

print(f"After 1km aggregation: {df_aggregated.Species.nunique()} species, {len(df_aggregated)} occurrences")

# Now filter for species with at least 30 grid cells
print("\nRemoving species with less than 30 grid cells")
species_counts = df_aggregated.groupby("Species").size()
species_counts = species_counts[species_counts >= 30]
df_final = df_aggregated[df_aggregated.Species.isin(species_counts.index)]

print(f"After filtering: {df_final.Species.nunique()} species, {len(df_final)} occurrences")

# Recover locations where no species survived filtering (all-negative vectors for test set)
# Use 1km grid cells to avoid duplicates within the same cell
all_locs = df_locations[["Longitude", "Latitude"]].values
all_grid_x, all_grid_y = latlon_to_grid_cell(all_locs[:, 0], all_locs[:, 1])

# Compute cell centers for all original locations
all_center_lon, all_center_lat = grid_cell_to_latlon(all_grid_x, all_grid_y)

# Deduplicate original locations to unique 1km cells using cell center coords
df_locations_grid = pd.DataFrame({
    "Latitude": all_center_lat,
    "Longitude": all_center_lon,
    "grid_cell": list(zip(all_grid_x, all_grid_y)),
})
df_locations_unique = df_locations_grid.drop_duplicates(subset=["grid_cell"])

# Find grid cells already covered by df_final
final_locs = df_final[["Longitude", "Latitude"]].values
final_grid_x, final_grid_y = latlon_to_grid_cell(final_locs[:, 0], final_locs[:, 1])
final_grid_cells = set(zip(final_grid_x, final_grid_y))

# Keep only grid cells not already in df_final
missing_mask = ~df_locations_unique["grid_cell"].isin(final_grid_cells)
missing_locations = df_locations_unique.loc[missing_mask, ["Latitude", "Longitude"]].copy()

print(f"Adding {len(missing_locations)} empty locations (unique 1km cells with no surviving species)")
df_final = pd.concat([df_final, missing_locations], ignore_index=True)



print("Writing to disk")
df_final.to_csv(f"{data_dir}splotopen/splotopen_processed.csv", index=False, sep=",")
print(f"Number of species: {len(df_final.Species.unique())}")
print(f"Number of plots: {len(df_final.PlotObservationID.unique())}")

#write species list to txt
species = list(df_final.Species.unique())
#astype str to avoid issues with nan
#sort species
species = sorted([s for s in species if pd.notna(s)])
with open(f"{data_dir}splotopen/splot_metadata/final_species.txt", "w", encoding="utf-8") as f:
    for s in species:
        clean = str(s).strip().replace("\t", " ")  # strip whitespace + remove tabs
        f.write(clean + "\n")


