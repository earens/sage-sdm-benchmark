import os

from sklearn.preprocessing import MultiLabelBinarizer
import rootutils
import yaml
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp


# --- Setup project root and config ---
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root = os.getenv("PROJECT_ROOT")
with open(f"{root}/configs/local_default.yaml") as stream:
    data_dir = yaml.safe_load(stream)["data_dir"]

tqdm.pandas()


#TODO mkdir from raw data
# --- Define data paths ---
occurrence_data_dir = os.path.join(data_dir, "occurrence_data/")
targets_dir = os.path.join(data_dir, "targets/")

def _compute_species_indices(targets_csr: sp.csr_matrix):
    """Compute per-species location index lookup from a CSR targets matrix.

    Returns:
        target_location_indices: flat array of location ids (one block per species)
        target_location_indices_lookup: cumulative offsets into the flat array (len = num_species + 1)
    """
    targets_csc = targets_csr.tocsc()
    num_species = targets_csr.shape[1]

    all_location_indices = []
    lengths = np.zeros(num_species, dtype=np.int32)

    for species_id in tqdm(range(num_species), desc="Building species indices"):
        start = targets_csc.indptr[species_id]
        end = targets_csc.indptr[species_id + 1]
        location_ids = targets_csc.indices[start:end]
        values = targets_csc.data[start:end]
        presence_indices = location_ids[values == 1]
        all_location_indices.extend(presence_indices.tolist())
        lengths[species_id] = len(presence_indices)

    target_location_indices_lookup = np.concatenate([[0], np.cumsum(lengths)])
    target_location_indices = np.array(all_location_indices, dtype=np.int32)
    return target_location_indices, target_location_indices_lookup


def save_targets_to_h5(targets_path, targets):
    targets = targets.tocsr()
    targets.data[:] = 1
    targets.data = targets.data.astype(np.uint8)

    # Pre-compute species-indexed lookup so it's always available
    target_location_indices, target_location_indices_lookup = _compute_species_indices(targets)

    with h5py.File(targets_path, "w") as f:
        g = f.create_group("csr")
        g.create_dataset("data", data=targets.data, compression="gzip", shuffle=True)
        g.create_dataset("indices", data=targets.indices.astype(np.int32), compression="gzip", shuffle=True)
        g.create_dataset("indptr", data=targets.indptr.astype(np.int32), compression="gzip", shuffle=True)
        g.attrs["shape"] = targets.shape
        g.attrs["format"] = "csr"

        f.create_dataset("target_location_indices", data=target_location_indices, compression="gzip")
        f.create_dataset("target_location_indices_lookup", data=target_location_indices_lookup, compression="gzip")

def load_targets_from_h5(targets_path):
    with h5py.File(targets_path, "r") as f:
        g = f["csr"]
        targets = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                               shape=tuple(g.attrs["shape"]))
    return targets


def add_species_indices_to_h5(targets_path):
    """Add species index lookup to an existing targets HDF5 file (if missing).

    Kept for backward compatibility with older HDF5 files.
    New files created by save_targets_to_h5 already include these datasets.
    """
    with h5py.File(targets_path, 'r') as f:
        if "target_location_indices" in f.keys():
            return
        dset = f['csr']
        targets_csr = sp.csr_matrix(
            (dset["data"][:], dset["indices"][:], dset["indptr"][:]),
            shape=tuple(dset.attrs["shape"])
        )

    target_location_indices, target_location_indices_lookup = _compute_species_indices(targets_csr)

    with h5py.File(targets_path, 'a') as f:
        f.create_dataset('target_location_indices_lookup', data=target_location_indices_lookup, compression='gzip')
        f.create_dataset('target_location_indices', data=target_location_indices, compression='gzip')


def main():

    print("Loading occurrence data...")
    splotopen_data = pd.read_csv(os.path.join(occurrence_data_dir, "splotopen.csv"), low_memory=False)
    gbif_powo_data = pd.read_csv(os.path.join(occurrence_data_dir, "gbif.csv"))

    print("Creating species index mapping...")
    species_list = gbif_powo_data["species"].unique()
    species_list = np.sort(species_list)
    species2ind = {species: i for i, species in enumerate(species_list)}
    species_list_path = os.path.join(targets_dir, "species_names.csv")
    pd.DataFrame(list(species2ind.items()), columns=['Species Name', 'Index']).to_csv(species_list_path, index=False)
    print(f"Saved species list to {species_list_path}")

    mlb = MultiLabelBinarizer(classes=range(len(species2ind)), sparse_output=True)

    print("Processing splotopen targets...")
    # Group by grid cell coordinates (data is already aggregated to 1km grid in splotopen.py)
    # Empty locations (NaN Species) become all-zero rows in the target matrix
    grouped = splotopen_data.groupby(["Longitude", "Latitude"])["Species"].progress_apply(
        lambda x: [species2ind[s] for s in x if pd.notna(s) and s in species2ind]
    ).reset_index()["Species"]
    splotopen_targets = mlb.fit_transform(grouped)
    splotopen_targets_path = os.path.join(targets_dir, "eval_targets.h5")
    save_targets_to_h5(splotopen_targets_path, splotopen_targets)
    print(f"Saved splotopen targets to {splotopen_targets_path}")

    print("Processing GBIF-POWO targets...")
    grouped = (
        gbif_powo_data.groupby(["decimalLongitude", "decimalLatitude"])["species"]
        .progress_apply(list).reset_index()["species"]
    )
    grouped = grouped.apply(lambda x: [species2ind[species] for species in x if species in species2ind])
    gbif_powo_targets = mlb.transform(grouped)
    gbif_powo_targets_path = os.path.join(targets_dir, "train_targets.h5")
    save_targets_to_h5(gbif_powo_targets_path, gbif_powo_targets)
    print(f"Saved GBIF-POWO targets to {gbif_powo_targets_path}")


if __name__ == "__main__":
    main()
