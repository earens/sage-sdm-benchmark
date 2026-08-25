"""LightningDataModule for the SDM benchmark.

Lazily loads the HDF5 data: presence-only GBIF occurrences with target-group
background points for training, and sPlotOpen presence-absence plots (with a
cached multilabel-stratified val/test split) for evaluation. Handles predictor
normalization, optional 1 km aggregation, range masking, per-species loss
weighting, and occurrence subsampling.
"""

import logging
from typing import Any

import os
import hashlib
import h5py
import hdf5plugin  # noqa: F401  — registers the Blosc filters the data is written with
import torch
import numpy as np
import pytorch_lightning as pl
import scipy.sparse as sp
import pandas as pd

from torch.utils.data import DataLoader, Dataset, Subset

from src.data_modules.dataset import PADataset, PODataset, PredictorsDataset, TargetsDataset, MaskDataset

logger = logging.getLogger(__name__)


os.environ.setdefault("MALLOC_ARENA_MAX", "2")


def _seed_worker(worker_id):
    """Ensure each DataLoader worker has a deterministic, unique seed,
    and immediately trim malloc arenas so freed memory is returned to the OS."""
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np
    import random
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# Statistics for normalization computed on the background dataset
predictors_means = {
    "chelsa": [2824.0169542731446, 93.38653362671955, 391.2680259137375, 6968.906313157404,
               2971.6945167055223, 2682.7656807675507, 288.92883593797154, 2866.8668306478808,
               2792.1953059226576, 2912.935608133659, 2735.6805789842847, 865.9032954120382,
               1455.134938569275, 239.87797431199667, 641.8433599201815, 3850.686253880019,
               851.851895195238, 2417.160027285874, 1796.1783381804803],
    "soilgrids": [4365.917522129043, 1318.5694785458047, 21.60208549182663, 22.137968624481488,
                     38.07227636875869, 63.5544675237916, 27.229620989358864, 50.630357692095515],
    "topography": [619.7443285141475, 33.419758832701994, 10.66458030919444, -0.22247168348931837,
                   0.0016219606847754395, -940.2019371102409, -940.1974591823076, 2.858178995864766,
                   0.004146200120381549, 0.002062762382972026, -1.944424205694818e-05, 1.2468076533129604e-05,
                   6.905668068741409e-05, -6.801284902712589e-05, -3.8055759944752435e-06, -3.836312194598671e-06],
    "human_footprint": [5.88157229010372, 0.1890322464145157, 0.9554335298746456, 0.3633100255057543,
                        0.09480501586825876, 0.4733537750346771, 2.325746700922552, 0.15359821763160428,
                        1.3263011024553293],
    # WorldCover landcover, stored ALREADY one-hot (11 classes). Identity stats so the
    # z-score transform is a no-op — never standardize one-hot columns.
    "landcover": [0.0] * 11,
}
predictors_stds = {
    "chelsa": [185.76626062980648, 30.475922579370433, 199.41701842581662, 4613.91750357326,
               160.23486229363832, 212.62921235532806, 123.9651045288516, 184.82280599979944,
               209.4990281806562, 153.44182091108402, 219.27971319854768, 870.2433206280394,
               1385.6865630004456, 401.4742898034511, 365.0283367532856, 3708.401092847289,
               1345.1983027972492, 2517.8318510730605, 2822.73873675296],
    "soilgrids": [4605.537702178613, 188.49874670745467, 13.880032870389766, 9.289732980643576,
                     55.21526320108318, 11.943800427536448, 13.273330468616372, 17.226076443858194],
    "topography": [844.59493693205, 56.927531736831405, 18.027133262178456, 4.499406585489329,
                   0.005019668385351001, 2918.414795726486, 2918.4162215742977, 4.932460451408292,
                   0.2642792560404825, 0.3090120110426653, 0.0001487962791438103, 0.0001297414559412208,
                   0.07354855190604426, 0.06830419931792198, 0.00018091755108364197, 0.00013801311133432398],
    "human_footprint": [6.86166205071386, 1.3618330564755063, 2.4031607268193547, 1.53128114639278,
                        0.5068428958710283, 0.9439213486312761, 2.5339118418699798, 1.0978129752383377,
                        2.2378756433529485],
    "landcover": [1.0] * 11,
}


class DataModule(pl.LightningDataModule):
    """DataModule pairing presence-only GBIF training data (with target-group
    background) against sPlotOpen presence-absence evaluation plots.

    Handles predictor normalization, optional 1 km aggregation, PO/PA/background
    range masking, per-species loss weighting, and occurrence subsampling.

    Args:
        train_dataset: Name of the presence-only training dataset (used as the
            prefix for its predictor/target HDF5 files, e.g. ``"train"``).
        aggregated: If True, use the 1 km-aggregated PO training data (files with
            the ``_1km`` suffix); background points are never aggregated.
        predictors_subset: Mapping of predictor group name (``"location"``,
            ``"chelsa"``, ``"soilgrids"``, ``"topography"``, ``"human_footprint"``)
            to the list of column indices to load from that group. Defaults to all
            columns of every group.
        batch_size: Batch size for the train/val/test dataloaders.
        num_workers: Number of DataLoader worker processes (val/test are capped at 2).
        pin_memory: Whether the dataloaders pin host memory for faster GPU transfer.
        data_dir: Root data directory; all HDF5/CSV paths are resolved relative to it.
        po_range_mask: If True, apply POWO native-range masks to the presence-only
            observation and background points during training.
        pa_range_mask: If True, apply the POWO native-range mask to the sPlotOpen
            presence-absence evaluation plots.
        subsample_indices_file: Optional filename (under ``targets/``) of a saved
            index array selecting a subset of PO training rows; None uses all rows.
    """

    def __init__(
        self,
        train_dataset: str = "train",
        aggregated: bool = True,
        predictors_subset: dict[str, list[int]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_dir: str = None,
        po_range_mask: bool | None = False,
        pa_range_mask: bool | None = False,
        subsample_indices_file: str = None,
        use_bg_stream: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['_class_path', 'class_path'])

        # When False, the random-background stream is fully off: no background_*.h5
        # predictor/mask files are opened, and the bg loss term (gradient-dead under
        # BCE.lambda_2=1.0) is skipped. Lets predictor sets without a bg sample (e.g.
        # landcover) train without a dummy background file.
        self.use_bg_stream = use_bg_stream

        if predictors_subset is None:
            predictors_subset = {
                "location": [0, 1],
                "chelsa": list(range(19)),
                "soilgrids": list(range(8)),
                "topography": list(range(16)),
                "human_footprint": list(range(9)),
            }

        self.predictors_subset = predictors_subset
        self.train_dataset = train_dataset
        self.aggregated = aggregated
        self.suffix = "_1km" if aggregated else ""

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.po_range_mask = po_range_mask
        self.pa_range_mask = pa_range_mask

        self.subsample_indices_file = subsample_indices_file
        self.subsample_indices = None


    @property
    def num_classes(self):
        # Use cached value if available (set after setup frees the CSR)
        if hasattr(self, '_num_classes'):
            return self._num_classes
        # Try getting from targets shape first (works during setup)
        if self.data_train is not None:
            return self.data_train.observation_targets_dataset._n_cols
        # Fallback: read from canonical species list
        species_csv = os.path.join(self.hparams.data_dir, "targets", "species_names.csv")
        if os.path.exists(species_csv):
            return len(pd.read_csv(species_csv))
        raise ValueError("Cannot determine num_classes: no training data loaded and species_names.csv not found.")

    def setup(self, stage: str = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # Load range masks. PO masks follow the aggregation (range_masks_1km/ or
        # range_masks/); the background points are never aggregated, and the sPlotOpen
        # evaluation plots are always scored on the 1 km grid.
        po_dir = f"range_masks{self.suffix}"
        bg_dir = "range_masks"
        pa_dir = "range_masks_1km"
        po_mask_path = f"{self.hparams.data_dir}{po_dir}/train_range_mask.h5"
        po_range_mask = (
            MaskDataset(po_mask_path)
            if self.hparams.po_range_mask else None
        )
        bg_range_mask = (
            MaskDataset(f"{self.hparams.data_dir}{bg_dir}/background_range_mask.h5")
            if self.hparams.po_range_mask and self.use_bg_stream else None
        )
        pa_range_mask = (
            MaskDataset(f"{self.hparams.data_dir}{pa_dir}/eval_range_mask.h5")
            if self.hparams.pa_range_mask else None
        )


        # Load test data
        self.test_locations_dataset = None
        self.test_predictors_datasets = []

        for name, indices in self.predictors_subset.items():
            path = f"{self.hparams.data_dir}predictors/eval_{name}.h5"
            if name == "location":
                self.test_locations_dataset = PredictorsDataset(path, "predictor_data", indices)
            else:
                self.test_predictors_datasets.append(PredictorsDataset(path, "predictor_data", indices))

        with h5py.File(f"{self.hparams.data_dir}targets/eval_targets.h5", 'r') as f:
            dset = f['csr']
            dset = sp.csr_matrix((dset["data"][:], dset["indices"][:], dset["indptr"][:]),
                            shape=tuple(dset.attrs["shape"]))
            self.test_targets_dataset = TargetsDataset(dset)

        # Optional per-cell sampled plot area (m^2), aligned 1:1 with the eval rows.
        # Used to down-weight / mask unreliable small-plot absences during eval.
        area_path = f"{self.hparams.data_dir}predictors/eval_sampled_area.h5"
        self.test_sampled_area_dataset = (
            PredictorsDataset(area_path, "predictor_data", None)
            if os.path.exists(area_path) else None
        )

        testset = PADataset(
            observation_locations_dataset=self.test_locations_dataset,
            observation_predictors_datasets=self.test_predictors_datasets,
            observation_targets_dataset=self.test_targets_dataset,
            transforms=self._make_transforms(),
            range_mask=pa_range_mask,
            sampled_area_dataset=self.test_sampled_area_dataset,
        )
        # Evaluation uses a multilabel-stratified 80/20 split of the sPlotOpen
        # presence-absence plots into validation and test. The split is cached
        # to disk.
        val_cache = f"{self.hparams.data_dir}val_indices.npy"
        test_cache = f"{self.hparams.data_dir}test_indices.npy"
        if os.path.exists(val_cache) and os.path.exists(test_cache):
            val_indices = np.load(val_cache).tolist()
            test_indices = np.load(test_cache).tolist()
            logger.info(f"[Split] Loaded cached split: val={len(val_indices)}, test={len(test_indices)}")
        else:
            val_indices, test_indices = self._multilabel_stratified_split(
                targets_sparse=self.test_targets_dataset.targets_data,
                val_fraction=0.2,
                min_test_occurrences=20,
            )
            np.save(val_cache, np.array(val_indices))
            np.save(test_cache, np.array(test_indices))
        self.data_val = Subset(testset, val_indices)
        self.data_test = Subset(testset, test_indices)

        # Load training data
        if stage == "fit":
            self.observation_locations_dataset = None
            self.observation_predictors_datasets = []
            self.bg_locations_dataset = None
            self.bg_predictors_datasets = []

            for name, indices in self.predictors_subset.items():
                path_observation = f"{self.hparams.data_dir}predictors/{self.train_dataset}_{name}{self.suffix}.h5"
                if name == "location":
                    self.observation_locations_dataset = PredictorsDataset(path_observation, "predictor_data", indices)
                else:
                    self.observation_predictors_datasets.append(
                        PredictorsDataset(path_observation, "predictor_data", indices)
                    )
                # Background predictors only loaded when the bg stream is on.
                if self.use_bg_stream:
                    path_bg = f"{self.hparams.data_dir}predictors/background_{name}.h5"
                    if name == "location":
                        self.bg_locations_dataset = PredictorsDataset(path_bg, "predictor_data", indices)
                    else:
                        self.bg_predictors_datasets.append(PredictorsDataset(path_bg, "predictor_data", indices))

            with h5py.File(f"{self.hparams.data_dir}targets/{self.train_dataset}_targets{self.suffix}.h5", 'r') as f:
                dset = f['csr']
                dset = sp.csr_matrix((dset["data"][:], dset["indices"][:], dset["indptr"][:]),
                               shape=tuple(dset.attrs["shape"]))
            self.observation_targets_dataset = TargetsDataset(dset)

            if self.subsample_indices_file is not None:
                self.subsample_indices = np.load(f"{self.hparams.data_dir}targets/{self.subsample_indices_file}")

            self.data_train = PODataset(
                observation_locations_dataset=self.observation_locations_dataset,
                observation_predictors_datasets=self.observation_predictors_datasets,
                observation_targets_dataset=self.observation_targets_dataset,
                bg_locations_dataset=self.bg_locations_dataset,
                bg_predictors_datasets=self.bg_predictors_datasets,
                transforms=self._make_transforms(),
                observation_range_mask=po_range_mask,
                bg_range_mask=bg_range_mask,
                subsample_indices=self.subsample_indices,
                load_bg=self.use_bg_stream,
            )

            if self.trainer is not None and self.trainer.model.species_weighting["method"] != "uniform":
                # Compute per-species loss weights from the training data and
                # masks, and set them on the model's loss function.

                effective_targets = dset[self.subsample_indices] if self.subsample_indices is not None else dset

                obs_mask_path = (
                    f"{self.hparams.data_dir}{po_dir}/train_range_mask.h5"
                    if self.hparams.po_range_mask else None
                )
                bg_mask_path = (
                    f"{self.hparams.data_dir}{bg_dir}/background_range_mask.h5"
                    if self.hparams.po_range_mask and self.use_bg_stream else None
                )
                num_bg_rows = len(self.bg_locations_dataset) if self.bg_locations_dataset is not None else 0

                pos_counts, tgb_counts, bg_counts = self._get_or_compute_loss_denominators(
                    targets_sparse=effective_targets,
                    observation_mask_path=obs_mask_path,
                    bg_mask_path=bg_mask_path,
                    num_bg_rows=num_bg_rows,
                    selected_indices=self.subsample_indices,
                )

                species_weighting = self.trainer.model.species_weighting
                method = species_weighting["method"]
                # Clip bounds only apply to the 'inversely_proportional_clipped'
                # method; ignored otherwise.
                clip_min = species_weighting.get("clip_min", 0.05)
                clip_max = species_weighting.get("clip_max", 20.0)
                obs_species_weights = self._compute_species_weights(
                    effective_counts=pos_counts,
                    total_samples=effective_targets.shape[0],
                    species_weights_method=method,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
                tgb_species_weights = self._compute_species_weights(
                    effective_counts=tgb_counts,
                    total_samples=effective_targets.shape[0],
                    species_weights_method=method,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
                if num_bg_rows > 0:
                    rb_species_weights = self._compute_species_weights(
                        effective_counts=bg_counts,
                        total_samples=num_bg_rows,
                        species_weights_method=method,
                        clip_min=clip_min,
                        clip_max=clip_max,
                    )
                else:
                    # bg stream off → rb term is unused (BCE.lambda_2=1.0); mirror obs weights.
                    rb_species_weights = obs_species_weights

                self.species_weights = {
                    "obs": obs_species_weights,
                    "tgb": tgb_species_weights,
                    "rb": rb_species_weights,
                }

                self.trainer.model.loss.set_species_weights(
                    torch.tensor(obs_species_weights, dtype=torch.float32),
                    torch.tensor(tgb_species_weights, dtype=torch.float32),
                    torch.tensor(rb_species_weights, dtype=torch.float32),
                )

            # Cache num_classes and free the heavyweight scipy CSR matrix.
            # Only the shared-memory tensors inside TargetsDataset are needed
            # for training; the CSR was only required for .sum() above.
            self._num_classes = self.observation_targets_dataset.targets_data.shape[1]
            self.observation_targets_dataset.targets_data = None
            # Same for test targets (only used for split computation above)
            if hasattr(self, 'test_targets_dataset') and self.test_targets_dataset is not None:
                self._test_num_classes = self.test_targets_dataset.targets_data.shape[1]
                self.test_targets_dataset.targets_data = None


        else:
            self.data_train = None

    def _multilabel_stratified_split(
        self,
        targets_sparse: sp.csr_matrix,
        val_fraction: float = 0.2,
        min_test_occurrences: int = 20,
    ):
        """Split indices into val/test ensuring every species has at least
        `min_test_occurrences` occurrences in the test set.

        Strategy (iterative stratification, rarest-species-first):
        1. Compute per-species occurrence counts.
        2. Process species from rarest to most common.
        3. For each species, iterate over locations containing it (shuffled).
           Assign each location to whichever split (val or test) is furthest
           below its desired proportion *for that species*, while hard-reserving
           enough locations to guarantee the test minimum.
        4. Locations already assigned by an earlier (rarer) species keep their
           assignment — later species just "inherit" them.
        5. Any locations not yet assigned (e.g. all-zero rows = empty locations)
           are distributed randomly at the end.
        """
        n_samples = targets_sparse.shape[0]
        n_species = targets_sparse.shape[1]

        # Per-species column sums (total occurrences in the full dataset)
        species_counts = np.asarray(targets_sparse.sum(axis=0)).ravel()  # (n_species,)

        # Sort species indices by count ascending (rare first)
        species_order = np.argsort(species_counts)

        # 0 = unassigned, 1 = val, 2 = test
        assignment = np.zeros(n_samples, dtype=np.int8)

        rng = np.random.default_rng(seed=42)

        # Track current counts per species in each split
        val_counts = np.zeros(n_species, dtype=np.int64)
        test_counts = np.zeros(n_species, dtype=np.int64)

        for sp_idx in species_order:
            total = species_counts[sp_idx]
            if total == 0:
                continue

            # Rows (locations) that contain this species
            row_indices = targets_sparse[:, sp_idx].nonzero()[0]
            rng.shuffle(row_indices)

            # Desired counts
            desired_test = max(min_test_occurrences, int(np.round(total * (1 - val_fraction))))
            desired_val = total - desired_test

            # Separate already-assigned vs unassigned for this species
            assigned_mask = assignment[row_indices] != 0
            unassigned = row_indices[~assigned_mask]

            # Current counts for this species
            cur_test = test_counts[sp_idx]
            cur_val = val_counts[sp_idx]

            # How many more does each split need for this species?
            need_test = max(0, desired_test - cur_test)
            need_val = max(0, desired_val - cur_val)

            for ri in unassigned:
                # Hard-reserve: if remaining unassigned are just enough for
                # the test minimum, force-assign to test
                remaining_unassigned = need_test + need_val
                if remaining_unassigned <= 0:
                    # Both splits satisfied for this species — assign by
                    # global proportion
                    if rng.random() < val_fraction:
                        choice = 1
                    else:
                        choice = 2
                else:
                    # Assign to whichever split needs more
                    test_deficit = need_test / max(remaining_unassigned, 1)
                    if rng.random() < test_deficit:
                        choice = 2
                    else:
                        choice = 1

                assignment[ri] = choice

                # Update counts for ALL species present at this location
                row_species = targets_sparse[ri].indices  # CSR gives column indices
                if choice == 1:
                    val_counts[row_species] += 1
                    need_val = max(0, need_val - 1)
                else:
                    test_counts[row_species] += 1
                    need_test = max(0, need_test - 1)

        # Assign any remaining unassigned locations randomly
        unassigned_mask = assignment == 0
        n_unassigned = unassigned_mask.sum()
        if n_unassigned > 0:
            random_assignments = rng.choice(
                [1, 2], size=n_unassigned, p=[val_fraction, 1 - val_fraction]
            )
            assignment[unassigned_mask] = random_assignments

        val_indices = np.where(assignment == 1)[0].tolist()
        test_indices = np.where(assignment == 2)[0].tolist()

        # Log diagnostics
        final_test_counts = np.asarray(
            targets_sparse[test_indices].sum(axis=0)
        ).ravel()
        species_with_occ = species_counts > 0
        min_in_test = final_test_counts[species_with_occ].min()
        n_below = (final_test_counts[species_with_occ] < min_test_occurrences).sum()
        logger.info(
            f"[Stratified split] val={len(val_indices)}, test={len(test_indices)} | "
            f"Min test occ per species: {min_in_test} | "
            f"Species below {min_test_occurrences} in test: {n_below}/{species_with_occ.sum()}"
        )

        return val_indices, test_indices

    def _make_transforms(self, normalize: bool = True):
        """Create normalization transforms that respect the predictor subset.

        Args:
            normalize: If True, apply z-score normalization after NaN imputation.
                       If False, only impute NaNs with predictor means (no scaling).
        """
        selected_means = []
        selected_stds = []

        for name, indices in self.predictors_subset.items():
            if name == "location":
                continue

            # Get the full statistics for this predictor type
            full_means = predictors_means[name]
            full_stds = predictors_stds[name]

            # Select only the indices specified in predictors_subset
            for idx in indices:
                selected_means.append(full_means[idx])
                selected_stds.append(full_stds[idx])

        if normalize:
            return NormalizeTabular(means=selected_means, stds=selected_stds)
        else:
            return ImputeNaN(means=selected_means)


    def _shuffle_generator(self):
        """A dedicated RNG for the training shuffle, seeded from the run seed.

        Without this, ``shuffle=True`` draws from the GLOBAL torch RNG, so the batch
        order depends on how many random numbers were consumed before training started
        — i.e. on how many parameters the model constructed.
        """
        if getattr(self, "_shuffle_gen", None) is None:
            seed = int(os.environ.get("PL_GLOBAL_SEED", 0))
            self._shuffle_gen = torch.Generator()
            self._shuffle_gen.manual_seed(seed)
        return self._shuffle_gen

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=False,
            worker_init_fn=_seed_worker,
            generator=self._shuffle_generator(),
        )

    def val_dataloader(self):
        # Val dataset is small (~11k samples) — use fewer workers and
        # don't persist them to avoid accumulating COW-fragmented memory
        # across epochs.
        n_val_workers = min(self.hparams.num_workers, 2)
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=n_val_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=False,
            worker_init_fn=_seed_worker,
        )

    def test_dataloader(self):
        n_test_workers = min(self.hparams.num_workers, 2)
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=n_test_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=False,
            worker_init_fn=_seed_worker,
        )

    def teardown(self, stage: str = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def _count_species_from_mask_file(
        self,
        mask_path: str,
        num_species: int,
        selected_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Count how many selected rows are valid for each species from a sparse mask HDF5."""
        with h5py.File(mask_path, "r") as f:
            if selected_indices is None:
                # Stream the species index in chunks and accumulate the bincount,
                # so the full (potentially multi-GB) array is never held in RAM —
                # the background mask index alone is ~1.9 GB, which OOMs modest hosts.
                species_index = f["species_index"]
                counts = np.zeros(num_species, dtype=np.int64)
                n = int(species_index.shape[0])
                chunk = 20_000_000  # ~80 MB per read at int32
                for start in range(0, n, chunk):
                    counts += np.bincount(species_index[start:start + chunk], minlength=num_species)
                return counts

            cumulative_lengths = f["cumulative_lengths"][:]
            species_index = f["species_index"]
            counts = np.zeros(num_species, dtype=np.int64)

            for idx in np.asarray(selected_indices, dtype=np.int64):
                start = cumulative_lengths[idx]
                end = cumulative_lengths[idx + 1]
                if start < end:
                    counts[np.asarray(species_index[start:end], dtype=np.int64)] += 1

        return counts

    def _compute_training_species_totals(
        self,
        num_species: int,
        num_observation_rows: int,
        observation_mask_path: str | None = None,
        selected_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute valid per-species observation counts seen by the training loss."""
        if observation_mask_path is None:
            return np.full(num_species, num_observation_rows, dtype=np.int64)

        return self._count_species_from_mask_file(
            observation_mask_path,
            num_species,
            selected_indices=selected_indices,
        )

    def _compute_positive_counts_with_mask(
        self,
        targets_sparse: sp.csr_matrix,
        mask_path: str,
        selected_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Count per-species presences that survive the optional observation mask."""
        num_species = targets_sparse.shape[1]
        positive_counts = np.zeros(num_species, dtype=np.int64)

        with h5py.File(mask_path, "r") as f:
            cumulative_lengths = f["cumulative_lengths"][:]
            species_index = f["species_index"]

            row_indices = (
                np.asarray(selected_indices, dtype=np.int64)
                if selected_indices is not None
                else np.arange(targets_sparse.shape[0], dtype=np.int64)
            )

            for local_row_idx, global_row_idx in enumerate(row_indices):
                present_species = targets_sparse.indices[
                    targets_sparse.indptr[local_row_idx]:targets_sparse.indptr[local_row_idx + 1]
                ]
                if present_species.size == 0:
                    continue

                start = cumulative_lengths[global_row_idx]
                end = cumulative_lengths[global_row_idx + 1]
                if start >= end:
                    continue

                valid_species = np.asarray(species_index[start:end], dtype=np.int64)
                matched_species = np.intersect1d(present_species, valid_species, assume_unique=False)
                if matched_species.size > 0:
                    positive_counts[matched_species] += 1

        return positive_counts

    def _compute_loss_denominators(
        self,
        targets_sparse: sp.csr_matrix,
        observation_mask_path: str | None,
        bg_mask_path: str | None,
        num_bg_rows: int,
        selected_indices: np.ndarray | None = None,
    ):
        """Compute per-species denominators for the three loss terms."""
        num_species = targets_sparse.shape[1]

        if observation_mask_path is None:
            pos_counts = np.asarray(targets_sparse.sum(axis=0)).ravel().astype(np.int64)
            valid_obs_counts = np.full(num_species, targets_sparse.shape[0], dtype=np.int64)
        else:
            pos_counts = self._compute_positive_counts_with_mask(
                targets_sparse,
                observation_mask_path,
                selected_indices=selected_indices,
            )
            valid_obs_counts = self._compute_training_species_totals(
                num_species=num_species,
                num_observation_rows=targets_sparse.shape[0],
                observation_mask_path=observation_mask_path,
                selected_indices=selected_indices,
            )

        tgb_counts = np.maximum(valid_obs_counts - pos_counts, 0)

        if bg_mask_path is None:
            bg_counts = np.full(num_species, num_bg_rows, dtype=np.int64)
        else:
            bg_counts = self._count_species_from_mask_file(bg_mask_path, num_species)

        return pos_counts, tgb_counts, bg_counts

    def _get_or_compute_loss_denominators(
        self,
        targets_sparse: sp.csr_matrix,
        observation_mask_path: str | None,
        bg_mask_path: str | None,
        num_bg_rows: int,
        selected_indices: np.ndarray | None = None,
    ):
        """Cache expensive denominator counts so repeated runs start quickly."""
        cache_dir = os.path.join(self.hparams.data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        subsample_tag = self.subsample_indices_file or "full"
        obs_mask_tag = os.path.basename(observation_mask_path) if observation_mask_path else "no_obs_mask"
        bg_mask_tag = os.path.basename(bg_mask_path) if bg_mask_path else "no_bg_mask"
        cache_key = "|".join([
            self.train_dataset,
            self.suffix,
            subsample_tag,
            obs_mask_tag,
            bg_mask_tag,
            str(targets_sparse.shape[0]),
            str(targets_sparse.shape[1]),
            str(num_bg_rows),
        ])
        cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"loss_denominators_{cache_hash}.npz")

        if os.path.exists(cache_path):
            logger.info(f"[Species weights] Loading cached loss denominators from {cache_path}")
            cached = np.load(cache_path)
            return cached["pos_counts"], cached["tgb_counts"], cached["bg_counts"]

        logger.info(f"[Species weights] Computing loss denominators and caching them to {cache_path}")
        pos_counts, tgb_counts, bg_counts = self._compute_loss_denominators(
            targets_sparse=targets_sparse,
            observation_mask_path=observation_mask_path,
            bg_mask_path=bg_mask_path,
            num_bg_rows=num_bg_rows,
            selected_indices=selected_indices,
        )
        np.savez_compressed(
            cache_path,
            pos_counts=pos_counts,
            tgb_counts=tgb_counts,
            bg_counts=bg_counts,
        )
        return pos_counts, tgb_counts, bg_counts

    def _compute_species_weights(self, effective_counts, total_samples, species_weights_method,
                                 clip_min=0.05, clip_max=20.0):
        """
        Compute species weights from a total-samples / effective-count ratio.

        `clip_min` / `clip_max` bound the weights for the
        'inversely_proportional_clipped' method and are ignored otherwise.
        """
        effective_counts = np.asarray(effective_counts, dtype=np.float64)
        safe_counts = np.clip(effective_counts, 1.0, None)
        base_weights = float(total_samples) / safe_counts

        if species_weights_method == "inversely_proportional":
            species_weights = base_weights

        elif species_weights_method == "inversely_proportional_clipped":
            species_weights = np.clip(base_weights, clip_min, clip_max)

        elif species_weights_method == "inversely_proportional_sqrt":
            species_weights = np.sqrt(base_weights)

        elif species_weights_method == "uniform":
            species_weights = np.ones(len(effective_counts))

        elif species_weights_method == "inversely_proportional_not_normalized":
            species_weights = 1.0 / safe_counts

        else:
            raise ValueError(
                "species_weights_method must be one of: 'inversely_proportional', "
                "'inversely_proportional_clipped', 'inversely_proportional_sqrt', "
                "'inversely_proportional_not_normalized', 'uniform'"
            )

        return species_weights


class NormalizeTabular:
    def __init__(self, means, stds):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def __call__(self, x):
        # Impute NaNs with means before normalization
        x = torch.where(torch.isnan(x), self.means, x)
        return (x - self.means) / self.stds


class ImputeNaN:
    """Replace NaN values with predictor means (no scaling)."""
    def __init__(self, means):
        self.means = torch.tensor(means, dtype=torch.float32)

    def __call__(self, x):
        return torch.where(torch.isnan(x), self.means, x)


