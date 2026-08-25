# Tutorials


Before you start, install the package with the `tutorials` extra and set your data path in `configs/local_default.yaml` (see the [main README](../README.md)):

```bash
python -m pip install -e ".[tutorials]"
```


Read the notebooks in order for a full tour, or jump to the one you need.

The **map notebooks (01 and 04)** additionally need the per-species range polygons. If you extracted only the core `sage_benchmark_data.tar`, add them into the same `data_dir`:

```bash
tar xzf native_ranges.tar.gz          # -> native_ranges/  (extract inside your data_dir)
```

Three cells depend on an optional download or extra. Each degrades to a printed note rather than an error, so the rest of the notebook still runs:

| Notebook | Cell | Needs | Without it |
|---|---|---|---|
| 01 | occurrence distributions | `targets/train_targets.h5`, from `sage_po_records_nonaggregated.tar` | the **Full** (pre-aggregation) panel is dropped |
| 02 | Random Forest baseline | the `baselines` extra — `pip install -e ".[baselines]"` | the baseline demo is skipped |
| 04 | confusion overlay | the run's `test_predictions.h5` from the model record | the prediction map still renders; only the overlay is skipped |

Notebook 04 also needs one checkpoint (`optimized_deepsdm_s2`) from the model record in order to predict at all, and says so up front if it is missing.

| # | Notebook | What you'll do | Data |
|---|----------|----------------|------|
| 01 | [**Explore the Data**](01_explore_the_data.ipynb) | inspect species, occurrences, predictors, and native ranges; see the sampling-effort × relative-prevalence axes the evaluation is built on | Zenodo |
| 02 | [**Train a Model**](02_train_a_model.ipynb) | the layered-config system, predictor and loss choices, launching training, checkpoints/resuming, and testing | Zenodo |
| 03 | [**Evaluate Models**](03_evaluate_models.ipynb) | per-species metrics, the stratified report, and delta heatmaps — for the framework's models and your own predictions | Zenodo / `examples/` |
| 04 | [**Generate Maps**](04_generate_maps.ipynb) | continental prediction maps, threshold sliders, and confusion overlays against sPlotOpen | Zenodo |

For installation, data download, reproducing the baselines, and the prediction-file format, see the [main README](../README.md).
