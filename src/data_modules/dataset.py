"""Dataset classes backing the SDM DataModule.

Lazily read the HDF5 stores and assemble training/eval samples: ``PODataset``
(presence-only occurrences with deterministic epoch-based background pairing),
``PADataset`` (presence-absence eval plots), plus thin wrappers for predictors,
targets and range masks.
"""

import os

import h5py
import numpy as np
import torch

try:
    import hdf5plugin  # noqa: F401  (imported for its side effect: registers Blosc/LZ4 HDF5 filters)
except ImportError:
    pass

from torch.utils.data import Dataset, Subset


class PADataset(Dataset):
    """Presence-absence evaluation dataset wrapping sPlotOpen vegetation plot data.

    Each item is a single plot location with multi-label presence/absence targets
    over all species (1 = present, 0 = absent).  Used for val and test only.

    Args:
        observation_locations_dataset: Dataset returning (lon, lat) tensors.
        observation_predictors_datasets: List of predictor datasets; concatenated on retrieval.
        observation_targets_dataset: TargetsDataset returning binary multi-label targets.
        transforms: Optional transform applied to the concatenated predictor tensor.
        range_mask: Optional MaskDataset; if provided, out-of-range species are masked during eval.
        sampled_area_dataset: Optional dataset returning per-cell sampled plot area (m^2),
            aligned 1:1 with the target rows; used to mask/weight unreliable small-plot absences.
    """
    def __init__(self, observation_locations_dataset, observation_predictors_datasets,
                 observation_targets_dataset, transforms=None, range_mask=None,
                 sampled_area_dataset=None):
        self.observation_locations = observation_locations_dataset
        self.observation_predictors = observation_predictors_datasets
        self.observation_targets = observation_targets_dataset
        self.transforms = transforms
        self.range_mask = range_mask
        self.sampled_area_dataset = sampled_area_dataset  # optional per-cell plot area (m^2)

    def __len__(self):
        return len(self.observation_targets)

    def __getitem__(self, idx):

        predictors = [ds[idx] for ds in self.observation_predictors]
        predictors = torch.cat(predictors, dim=-1) if len(predictors) > 1 else predictors[0]

        item = {
            "observation_locations": self.observation_locations[idx],
            "observation_predictors": predictors,
            "observation_targets": self.observation_targets[idx],
        }

        if self.range_mask is not None:
            item["range_mask"] = self.range_mask[idx]
        if self.sampled_area_dataset is not None:
            item["sampled_area_m2"] = self.sampled_area_dataset[idx]
        if self.transforms:
            item["observation_predictors"] = self.transforms(item["observation_predictors"])
        return item

class PODataset(Dataset):
    """Presence-only training dataset pairing GBIF occurrences with background points.

    Each item returns one occurrence location together with a pseudo-randomly
    chosen background (pseudo-absence) location.  The bg pairing is deterministic
    given (sample_index, epoch) so results are fully reproducible across workers
    and machines, but the pairing changes every epoch to act as data augmentation.

    Args:
        observation_locations_dataset: Dataset returning (lon, lat) for occurrences.
        observation_predictors_datasets: List of predictor datasets for occurrences.
        observation_targets_dataset: TargetsDataset of multi-label occurrence targets.
        bg_locations_dataset: Dataset returning (lon, lat) for background points.
        bg_predictors_datasets: List of predictor datasets for background points.
        transforms: Optional transform applied to predictor tensors.
        observation_range_mask: Optional per-occurrence range mask.
        bg_range_mask: Optional per-background range mask.
        subsample_indices: If given, only these row indices of the occurrence datasets are used.
    """
    def __init__(self, observation_locations_dataset, observation_predictors_datasets,
                 observation_targets_dataset, bg_locations_dataset, bg_predictors_datasets,
                 transforms=None, observation_range_mask=None, bg_range_mask=None,
                 subsample_indices=None, load_bg=True):
        self.observation_locations_dataset = observation_locations_dataset
        self.observation_predictors_datasets = observation_predictors_datasets
        self.observation_targets_dataset = observation_targets_dataset
        self.bg_locations_dataset = bg_locations_dataset
        self.bg_predictors_datasets = bg_predictors_datasets
        self.transforms = transforms
        self.observation_range_mask = observation_range_mask
        self.bg_range_mask = bg_range_mask
        self.load_bg = load_bg

        if subsample_indices is not None:
            self.observation_locations_dataset = Subset(self.observation_locations_dataset, subsample_indices)
            self.observation_targets_dataset = Subset(self.observation_targets_dataset, subsample_indices)
            self.observation_predictors_datasets = [
                Subset(ds, subsample_indices) for ds in self.observation_predictors_datasets
            ]
            if self.observation_range_mask is not None:
                self.observation_range_mask = Subset(self.observation_range_mask, subsample_indices)

        # bg may be absent when the background stream is disabled (load_bg=False).
        self._n_bg = len(self.bg_locations_dataset) if self.bg_locations_dataset is not None else 0
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch counter so bg pairings change each epoch."""
        self._epoch = epoch

    @staticmethod
    def _hash_bg_idx(idx: int, epoch: int, n_bg: int) -> int:
        """Deterministic, zero-memory hash mapping (idx, epoch) -> bg index.

        Uses a Stafford variant of the Murmur finaliser (mix13) for good
        uniformity.  Same (idx, epoch) always gives the same result
        regardless of worker, process or machine.
        """
        x = idx * 2654435761 + epoch * 2246822519
        x = ((x >> 16) ^ x) * 0x45D9F3B
        x = ((x >> 16) ^ x) * 0x45D9F3B
        x = (x >> 16) ^ x
        return x % n_bg


    def __len__(self):
        return len(self.observation_targets_dataset)

    def __getitem__(self, idx):

        observation_predictors = [ds[idx] for ds in self.observation_predictors_datasets]
        observation_predictors = (
            torch.cat(observation_predictors, dim=-1)
            if len(observation_predictors) > 1 else observation_predictors[0]
        )

        item = {
            "observation_locations": self.observation_locations_dataset[idx],
            "observation_predictors": observation_predictors,
            "observation_targets": self.observation_targets_dataset[idx],
        }
        if self.observation_range_mask is not None:
            item["range_mask"] = self.observation_range_mask[idx]
        if self.transforms:
            item["observation_predictors"] = self.transforms(item["observation_predictors"])

        if self.load_bg:
            # Deterministic random bg pairing: every observation gets a
            # pseudo-random bg point that changes each epoch (data augmentation)
            # but is reproducible across workers/machines.
            bg_idx = self._hash_bg_idx(idx, self._epoch, self._n_bg)
            bg_predictors = [ds[bg_idx] for ds in self.bg_predictors_datasets]
            bg_predictors = torch.cat(bg_predictors, dim=-1) if len(bg_predictors) > 1 else bg_predictors[0]
            item["bg_locations"] = self.bg_locations_dataset[bg_idx]
            item["bg_predictors"] = bg_predictors
            if self.observation_range_mask is not None:
                item["bg_range_mask"] = self.bg_range_mask[bg_idx]
            if self.transforms:
                item["bg_predictors"] = self.transforms(item["bg_predictors"])
        return item


class PredictorsDataset(Dataset):
    """HDF5-backed predictor dataset optimized for random row access.

    Args:
        h5_path: Path to HDF5 file.
        key: Dataset key inside the file.
        indices: Column indices to select (None = all columns).
        in_memory: If True, load entire dataset into RAM as a torch tensor
                   for maximum speed. Recommended for files < ~2 GB.
                   If None (default), auto-detect based on file size.
        in_memory_threshold_mb: Auto-load into memory if file < this size (MB).
    """
    IN_MEMORY_THRESHOLD_MB = 2048  # 2 GB default

    def __init__(self, h5_path: str, key: str, indices, in_memory: bool | None = None,
                 in_memory_threshold_mb: float | None = None):
        self.h5_path = h5_path
        self.key = key
        self.indices = indices
        self._h5 = None
        self._dset = None
        self._data_tensor = None  # holds in-memory data if loaded

        threshold = in_memory_threshold_mb or self.IN_MEMORY_THRESHOLD_MB

        if in_memory is None:
            # Auto-detect: load into memory if file is small enough
            file_mb = os.path.getsize(h5_path) / (1024 * 1024)
            in_memory = file_mb < threshold

        if in_memory:
            self._load_into_memory()

    def _load_into_memory(self):
        """Load the entire dataset into a contiguous float32 torch tensor
        placed in shared memory so forked DataLoader workers can read it
        without copy-on-write page faults."""
        # Use a large rdcc cache to speed up the sequential read
        with h5py.File(self.h5_path, "r", swmr=False, locking=False,
                       rdcc_nbytes=64 * 1024 * 1024) as f:
            dset = f[self.key]
            if self.indices is None:
                data = dset[()]
            else:
                data = dset[:, self.indices]
        self._data_tensor = (
            torch.tensor(data, dtype=torch.float32)
            .contiguous()
            .share_memory_()
        )
        self._len = self._data_tensor.shape[0]

    def _ensure_open(self):
        if self._h5 is None and self._data_tensor is None:
            # Use rdcc (raw data chunk cache): keep modest to avoid OOM
            # when multiple DataLoader workers each open their own copy
            self._h5 = h5py.File(
                self.h5_path, "r", swmr=False, locking=False,
                rdcc_nbytes=16 * 1024 * 1024,  # 16 MB chunk cache per worker
                rdcc_nslots=10007,
            )
            self._dset = self._h5[self.key]

    def __len__(self):
        if self._data_tensor is not None:
            return self._len
        self._ensure_open()
        return self._dset.shape[0]

    def __getitem__(self, idx):
        if self._data_tensor is not None:
            return self._data_tensor[idx]

        self._ensure_open()
        if self.indices is None:
            x = self._dset[idx]
        else:
            x = self._dset[idx, self.indices]
        return torch.tensor(x, dtype=torch.float32)

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass



class TargetsDataset(Dataset):
    """Sparse target dataset optimised for multi-worker DataLoader access.

    The sparse CSR components (indptr, indices, data) are stored as
    shared-memory torch tensors so forked workers can read them without
    copy-on-write page faults or per-sample scipy overhead.
    """
    def __init__(self, targets_data, h5_path: str = None):
        self.targets_data = targets_data          # keep for .sum() etc. in setup()
        self._n_rows = targets_data.shape[0]
        self._n_cols = targets_data.shape[1]

        # Pre-extract CSR arrays into shared-memory tensors ─────────────────
        if hasattr(targets_data, 'indptr'):       # scipy CSR
            csr = targets_data
        elif hasattr(targets_data, 'tocsr'):      # other sparse
            csr = targets_data.tocsr()
        else:
            csr = None

        if csr is not None:
            self._indptr  = torch.from_numpy(
                csr.indptr.astype(np.int64)).share_memory_()
            self._indices = torch.from_numpy(
                csr.indices.astype(np.int64)).share_memory_()
            self._values  = torch.from_numpy(
                csr.data.astype(np.int8)).share_memory_()
            self._fast = True
        else:
            self._fast = False

    def __len__(self):
        return self._n_rows

    def __getitem__(self, idx):
        if self._fast:
            start = self._indptr[idx].item()
            end   = self._indptr[idx + 1].item()
            row   = torch.zeros(self._n_cols, dtype=torch.int8)
            if start < end:
                row[self._indices[start:end]] = self._values[start:end]
            return row

        # Dense fallback (never reached for CSR)
        row = self.targets_data[idx]
        if hasattr(row, "toarray"):
            row = row.toarray().ravel()
        return torch.tensor(row, dtype=torch.int64)


class MaskDataset(Dataset):
    """Range mask dataset stored as sparse (cumulative_lengths + species_index) in HDF5.

    - ``cumulative_lengths`` is small and loaded into a shared-memory tensor
      so forked DataLoader workers can read it without copy-on-write faults.
    - ``species_index`` can be huge (>100 GB uncompressed for GBIF) and is
      therefore read lazily from the HDF5 file.  Each worker opens its own
      handle with a **small** chunk cache (16 MB) to avoid the previous OOM
      issue caused by 256 MB caches × many workers.
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._h5 = None          # lazy, opened per worker
        self._row_to_cell = None  # set only in indirect mode

        with h5py.File(h5_path, "r", swmr=False, locking=False) as f:
            self.num_species = int(f.attrs["num_species"])
            if "row_to_cell" in f:
                # Indirect mode: this file only stores a per-row -> base-row index
                # map; the (cumulative_lengths, species_index) live in a shared base
                # mask (attr "base_mask", same dir). Lets many rows reuse one cell's
                # mask without replicating species_index (which otherwise bloats the
                # file and thrashes the page cache).
                self._row_to_cell = torch.from_numpy(
                    f["row_to_cell"][()].astype(np.int64)).share_memory_()
                self._length = len(self._row_to_cell)
                self._data_path = os.path.join(os.path.dirname(h5_path), f.attrs["base_mask"])
                with h5py.File(self._data_path, "r", swmr=False, locking=False) as bf:
                    cum_lengths = bf["cumulative_lengths"][()]
            else:
                self._length = len(f["lengths"])
                self._data_path = h5_path
                cum_lengths = f["cumulative_lengths"][()]

        # cumulative_lengths is small → shared-memory tensor
        self._cum_lengths = torch.from_numpy(
            cum_lengths.astype(np.int64)).share_memory_()

    def _ensure_open(self):
        """Open the HDF5 file lazily with a small chunk cache."""
        if self._h5 is None:
            self._h5 = h5py.File(
                self._data_path, "r", swmr=False, locking=False,
                rdcc_nbytes=16 * 1024 * 1024,   # 16 MB – enough for sequential slices
                rdcc_nslots=10007,
            )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._ensure_open()
        row = idx if self._row_to_cell is None else int(self._row_to_cell[idx].item())
        start = self._cum_lengths[row].item()
        end = self._cum_lengths[row + 1].item()

        mask = torch.zeros(self.num_species, dtype=torch.int8)
        if start < end:
            species = self._h5["species_index"][start:end]
            mask[torch.from_numpy(species.astype(np.int64))] = 1
        return mask

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
