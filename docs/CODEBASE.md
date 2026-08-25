# Codebase

Overview for insights on Architecture, file structure, framework configuration, debugging, and testing. For training models and their configurations see [`MODELS.md`](MODELS.md); for the data see [`DATA.md`](DATA.md).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                │
│   Merges layered YAML configs → launches CustomLightningCLI     │
└───────────────┬─────────────────────────────┬───────────────────┘
                │                             │
    ┌───────────▼───────────┐     ┌───────────▼────────────┐
    │    LightningModule    │     │   LightningDataModule  │
    │  (src/modules/)       │     │  (src/data_modules/)   │
    │  - Training loop      │     │  - PO train data       │
    │  - Metric management  │     │  - PA test data        │
    │  - Map generation     │     │  - Normalization       │
    └──┬──────┬─────┬───────┘     └──────┬──────┬──────────┘
       │      │     │                    │      │
   ┌───▼──┐ ┌─▼──┐ ┌▼────────┐   ┌───────▼─┐ ┌──▼─────────┐
   │Models│ │Loss│ │Metrics  │   │Datasets │ │Range Masks │
   └──────┘ └────┘ │AUROC,   │   │PO, PA   │ └────────────┘
                   │AUPRG    │   └─────────┘ 
                   └─────────┘
```

## File structure

```text
├── configs/                                # Layered YAML configuration
│   ├── default.yaml                        # Trainer, logger, callbacks, global settings
│   ├── local_default.yaml.example          # Template for machine-specific paths (copy → local_default.yaml)
│   ├── paper_runs.yaml                     # Paper models and their run folders (see Evaluation framework)
│   ├── data/
│   │   ├── datamodule.yaml                 # SDM data config (dataset, split, predictors)
│   │   └── demo.yaml                       # 50-species demo subset (for the tutorials)
│   ├── model/
│   │   ├── resnet.yaml                  # ResNet model + loss + metrics config (main DeepSDM)
│   │   ├── simple_mlp.yaml                 # Plain MLP variant
│   │   ├── demo_mlp.yaml                   # Lightweight MLP for the demo/tutorials
│   │   ├── FFTransformer.yaml              # Feature-tokenizer transformer variant
│   │   ├── lr.yaml                         # Logistic-regression baseline
│   │   ├── lr_with_maxent_features.yaml    # LR on MaxEnt-style features
│   │   ├── maxent.yaml                     # MaxEnt baseline
│   │   ├── randomforest.yaml               # Down-sampled Random Forest (Valavi 2021)
│   │   ├── glm.yaml / gam.yaml / brt.yaml  # GLM / GAM / BRT single-species baselines
│   │   └── sklearn.yaml                    # Generic scikit-learn baseline template
│   └── experiments/
│       ├── test.yaml                       # Reference experiment (references data + model configs)
│       ├── demo.yaml                       # Runnable demo experiment (CPU, 50 species)
│       ├── baseline_deepsdm.yaml           # Main-table ablation ladder: baseline DeepSDM
│       ├── aggregation.yaml                #   + 1 km aggregation
│       ├── subsampling.yaml                #   + subsampling (N=1000)
│       ├── architecture.yaml               #   + ResNet architecture
│       └── optimized_deepsdm.yaml          #   + tuned loss / reweighting (best model)
├── src/                                    # Core source code
│   ├── main.py                             # Entry point: merges configs, launches CLI
│   ├── models/                             # resnet.py (main), simple_mlp.py, fftransformer.py,
│   │                                       #   linear.py, location_encoding.py
│   ├── modules/
│   │   └── lightning_module.py             # SDM LightningModule (train/val/test/maps)
│   ├── data_modules/
│   │   ├── datamodule.py                   # SDM DataModule (PO train + PA test)
│   │   └── dataset.py                      # PODataset, PADataset, PredictorsDataset, etc.
│   ├── criteria/
│   │   └── bce.py                          # Weighted BCE loss (observation + background)
│   ├── metrics/
│   │   ├── precision_recall_gain.py        # AUPRG metric (Flach & Kull 2015)
│   │   └── utils.py                        # Metric logging
│   ├── evaluation/                         # Trainer-independent evaluation 
│   │   ├── prediction_io.py                # Prediction-file contract (load HDF5/CSV predictions)
│   │   ├── metrics.py                      # Per-species metrics from a predictions file
│   │   ├── regions.py                      # Stratification regions (quadrants + extreme deciles)
│   │   └── stratify.py                     # Stratified score tables
│   ├── visualizations/
│   │   ├── data_stats.py                   # Occurrence / prevalence distributions
│   │   ├── delta_heatmap.py                # Model-vs-reference delta heatmap (paper figure)
│   │   ├── property_space.py               # Species property-space figure (SE × RP)
│   │   ├── species_map.py                  # Per-species occurrence map
│   │   └── maps/                           # Continental prediction maps (paper palette)
│   │       ├── grid.py                     # Grid build + model-agnostic prediction (torch/sklearn/maxent)
│   │       ├── render.py                   # Plotly figure builders + colorschemes + save
│   │       ├── slider.py                   # Interactive threshold slider (confusion points)
│   │       └── wandb_maps.py               # Per-epoch wandb + test-time HTML map generation
│   ├── callbacks/
│   │   └── timing.py                       # Epoch/step timing callback
│   └── utils/
│       └── cli.py                          # CustomLightningCLI with WandB integration
├── scripts/                                # Standalone processing scripts
│   ├── evaluate_predictions.py             # Score any predictions file
│   ├── generate_maps.py                    # Continental maps from a checkpoint / predictions file
│   ├── data_prep_scripts/                  # Data preprocessing pipeline (see DATA.md)
│   ├── model_evaluation/
│   │   └── compute_species_ranks.py        # Build per-species proxies (sampling effort, relative prevalence)
│   └── traditional_sdms/                   # Single-species baselines (see MODELS.md)
├── tutorials/                              # Hands-on notebooks: explore, train, evaluate, generate maps
└── examples/                               # Offline worked example for the evaluation toolkit
```

## Configuration reference


### Metrics (per stage: `train`, `val`, `test`)

Metrics are fully configurable via `torchmetrics` class paths:

```yaml
metrics:
  test:
    auroc:
      class_path: torchmetrics.classification.MultilabelAUROC
      init_args:
        num_labels: 0        # Dynamically set from species_names.csv
        average: null        # null = per-species, "macro" = averaged
        ignore_index: -1     # Ignore masked (out-of-range) species
    auprg:
      class_path: src.metrics.precision_recall_gain.MultilabelAUPRG
      init_args:
        num_labels: 0
        average: null
        ignore_index: -1
```

### Species distribution maps (`continent_map_configs`)

Maps are automatically cropped to each species' native range (from POWO shapefiles) with a configurable margin. All countries intersecting the bounding box are shown for context.

```yaml
continent_map_configs:
  - species_ids: [4425, 6568]  # Species indices to generate maps for
    continent: "europe"        # Optional tag for file naming
```

### Trainer (`configs/default.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed_everything` | `0` | Global random seed |
| `trainer.max_epochs` | `-1` | Maximum training epochs |
| `trainer.accelerator` | `"auto"` | Device: `auto` picks CUDA GPU / Apple MPS / CPU; force with `cpu`, `gpu`, `tpu` |
| `trainer.precision` | `"32-true"` | Training precision (`16-mixed`, `32-true`, etc.) |
| `trainer.logger` | `WandbLogger` | Experiment logger |
| `test_after_fit` | `true` | Automatically run test after training completes |
| `ignore_warnings` | `false` | Suppress Python warnings |

**Built-in callbacks** (configured in `default.yaml`):
- `ModelCheckpoint` — Saves top-k models by `val/auroc`
- `EarlyStopping` — Stops training when `val/auroc` plateaus
- `LearningRateMonitor` — Logs learning rate per epoch

## Debugging

```bash
# Quick sanity check (single batch)
python src/main.py --exp test fit --trainer.fast_dev_run true

# Attach a debugger (VS Code / PyCharm)
DEBUG=1 python src/main.py --exp test fit
```

## Testing

`tests/test_models.py` smoke-tests every model's forward pass and the BCE loss on synthetic tensors — CPU-only, no dataset files, no network, a few seconds.

```bash
pytest tests/test_models.py -v
```

For an end-to-end check of the real entry points, run the demo experiment on the 50-species subset in `examples/demo_data/`:

```bash
python src/main.py --exp demo fit --trainer.fast_dev_run true
python scripts/evaluate_predictions.py --predictions examples/example_predictions.h5 \
    --reference examples/example_reference.h5 --proxies examples/example_species_properties.csv \
    --output-dir examples/output
```

## Key design decisions

- **Presence-only training, presence-absence testing.** The model trains on GBIF occurrence data (presence-only with background points) and is evaluated on sPlotOpen vegetation plot data (true presence-absence).
- **Range masking.** Species are only evaluated at locations within their known native range (POWO). Out-of-range locations are masked with `ignore_index=-1` in metrics.
- **Weighted BCE loss.** The loss separates observation locations from background points with configurable weights (`lambda_1`, `lambda_2`), and supports per-species weighting to handle class imbalance.
- **1 km aggregation.** Occurrence data is aggregated to a 1 km equal-area grid (EPSG:6933) to reduce spatial autocorrelation and duplicate reporting.
