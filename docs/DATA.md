# Data

Full reference for the SAGE benchmark data: what it contains, where to get it, the on-disk layout, the environmental predictors, the raw sources and their terms, and the pipeline that rebuilds the processed files from scratch.

For a short overview, see the [main README](../README.md#data).

## Download

The processed data (targets, predictors, and range masks as HDF5) is hosted on **Zenodo
(DOI: [10.5281/zenodo.21297133](https://doi.org/10.5281/zenodo.21297133))**. The trained
checkpoints, per-plot predictions, and per-species metrics are a **separate** model record
(see [Evaluation framework](../README.md#evaluation-framework)).

Download the archive, extract it, and point `data_dir` at the extracted folder (not the `.tar`):

```bash
mkdir -p /path/to/data && cd /path/to/data
wget -c https://zenodo.org/records/21297133/files/sage_benchmark_data.tar?download=1 -O sage_benchmark_data.tar
tar xf sage_benchmark_data.tar        # core: targets/ predictors/ range_masks_1km/ range_masks/ maps/
```

That archive is everything needed to train and evaluate. Two optional extras extract into the same `data_dir`:

```bash
# per-species POWO range polygons — for the map figures and exploration tutorials:
tar xzf native_ranges.tar.gz                                          # -> native_ranges/
# non-aggregated presence-only records — needed for the `baseline_deepsdm` model
# (aggregated: false), or to apply your own aggregation:
tar xf sage_po_records_nonaggregated.tar                              # adds targets/, predictors/, range_masks/
```

Set `data_dir` in `configs/local_default.yaml` to that directory.
## Layout

```text
<data_dir>/
├── targets/
│   ├── species_names.csv                 # canonical species order (defines num_classes)
│   ├── species_properties_1km.csv        # per-species sampling effort × relative prevalence
│   ├── train_targets_1km.h5              # sparse presence-only training targets
│   ├── eval_targets.h5                   # presence-absence evaluation targets (all plots)
│   └── eval_test_targets.h5              # test-split ground truth (default for evaluate_predictions.py)
├── predictors/                           # one HDF5 per source × predictor group
│   ├── train_<group>_1km.h5              #   PO training (1 km-aggregated)
│   ├── background_<group>.h5             #   background points (never aggregated)
│   └── eval_<group>.h5                   #   eval plots  (<group> = location/chelsa/soilgrids/topography/human_footprint)
├── range_masks_1km/                      # POWO native-range masks — training + evaluation
├── range_masks/                          # background_range_mask.h5 (bg points; used in training)
├── maps/                                 # ne_110m_admin_0_countries.* Natural Earth basemap
├── native_ranges/                        # per-species POWO polygons (from native_ranges.tar.gz)
├── val_indices.npy                       # the fixed validation split of the evaluation plots
├── test_indices.npy                      # the fixed test split (matches the released predictions)
└── eval_plot_metadata.csv                # plot-level sPlotOpen attributes + join key to the eval rows
```

`targets/subsample_indices/` holds the pre-computed per-species occurrence subsamples
(`train_indices_<N>_1km.npy`) that `subsample_indices_file` selects.

> **Masks.** `range_masks_1km/` is a single frozen generation of the POWO native-range masks — the one every released run was trained and evaluated with. `targets/species_properties_1km.csv` is derived from it (`sampling_effort = n_cells_visited_in_range / n_cells_in_range * 100`, `relative_prevalence = n_species_obs / n_cells_visited_in_range * 100`), so the stratification axes, the masks, and the released predictions are all on the same basis. Rebuilding the masks with `generate_masks.py` shifts those axes and makes the per-group numbers no longer comparable to the paper; recompute the properties CSV with `scripts/model_evaluation/compute_species_ranks.py` if you do.

The evaluation set comprises 53,336 vegetation plots, split once into a fixed validation set (11,068 plots) and test set (42,268 plots). The split is provided as `val_indices.npy` / `test_indices.npy`. The released `test_predictions.h5` and `targets/eval_test_targets.h5` are aligned to `test_indices.npy`. Training pairs 89.8 M GBIF occurrences with background points, all aggregated to a 1 km equal-area grid (EPSG:6933).

## Environmental predictors

52 environmental predictors, plus an optional 2-D location group:

| Group | # Features | Source |
|-------|-----------|--------|
| `location` | 2 | Longitude, latitude (optional; off by default) |
| `chelsa` | 19 | CHELSA bioclimatic variables |
| `soilgrids` | 8 | SoilGrids soil properties |
| `topography` | 16 | EarthEnv terrain/elevation derivatives |
| `human_footprint` | 9 | Human Footprint Index components |

## Data config (`configs/data/datamodule.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_dataset` | `"train"` | Filename prefix of the training data to load, e.g. `train_targets_1km.h5` (`"train"` is the released benchmark set) |
| `aggregated` | `true` | Whether to use 1 km grid-aggregated data |
| `predictors_subset` | all | Dict of predictor groups → feature indices to include |
| `po_range_mask` | `true` | Apply range masks to presence-only training data |
| `pa_range_mask` | `true` | Apply range masks to presence-absence test data |
| `subsample_indices_file` | `subsample_indices/train_indices_10000_1km.npy` | Pre-computed occurrence subsample indices (from `subsample_targets.py`); `null` = use all occurrences |
| `batch_size` | `512` | Training batch size |
| `num_workers` | `14` | DataLoader workers |

## Raw sources and licensing

The raw sources and the license under which each may be redistributed are listed below; where a derived layer can't be redistributed, use the prep scripts to rebuild it from the raw source.

| Source | Layer | Access |
|--------|-------|--------|
| [GBIF](https://www.gbif.org/) | Occurrences (presence-only training) | Free account + API key ([setup](https://www.gbif.org/user/profile)) |
| [sPlotOpen](https://doi.org/10.1111/geb.13346) | Vegetation plots (presence-absence eval) | Public download |
| [CHELSA](https://chelsa-climate.org/) | 19 bioclimatic predictors | Public rasters |
| [SoilGrids](https://soilgrids.org/) | 8 soil-property predictors | Public rasters |
| [EarthEnv](https://www.earthenv.org/topography) | 16 topography predictors | Public rasters |
| [Human Footprint](https://doi.org/10.1038/sdata.2016.67) | 9 human-pressure predictors | Public rasters |
| [POWO](https://powo.science.kew.org/) / [TDWG](https://www.tdwg.org/standards/wgsrpd/) | Native-range masks | Public (per-source terms apply) |

## Pipeline (rebuild from raw)
Run the data-preparation scripts in order:

1. **`scripts/data_prep_scripts/gbif.py`** — Download the GBIF occurrence records for the species list and resolve their taxonomy
2. **`scripts/data_prep_scripts/cleaning_gbif.r`** — Coordinate cleaning and native-range filtering against WCVP. Each species is restricted to the L3 regions of its Accepted WCVP row.
3. **`scripts/data_prep_scripts/splotopen.py`** — Download and preprocess sPlotOpen vegetation data (taxonomic name resolution, range filtering, duplicate removal)
4. **`scripts/data_prep_scripts/targets.py`** — Convert occurrence CSVs into sparse multi-label target matrices (HDF5)
5. **`scripts/data_prep_scripts/predictors.py`** — Extract environmental predictor values from rasters (CHELSA, SoilGrids, Topography, Human Footprint) at occurrence locations and generate background points
6. **`scripts/data_prep_scripts/aggregate_to_1km_grid.py`** — Aggregate targets and predictors onto a 1 km equal-area grid (EPSG:6933)
7. **`scripts/data_prep_scripts/generate_masks.py`** — Build species range masks from POWO native range shapefiles and TDWG Level 3 botanical regions.
8. **`scripts/data_prep_scripts/subsample_targets.py`** — Cap occurrences per species (the released benchmark uses `--num_samples_per_species 1000`, and `configs/data/datamodule.yaml` selects the result via `subsample_indices_file`). The draw is taken from each species' in-range cells, so the per-species quota holds after the loss re-applies the range mask

Steps 2 and 3 need R; see [Installation](../README.md#installation). Steps 2 and 7 carry the two corrections that define the released data — run them as written and the pipeline lands on the shipped files directly (`train_targets_1km.h5` with 57,084,622 stored presences, and subsample indices identical to the released `train_indices_<N>_1km.npy`).
