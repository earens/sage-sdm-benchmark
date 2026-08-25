# Models

Training reference for both model families the benchmark compares: the multi-species DeepSDM (one network over all species) and the single-species baselines (one model per species — MaxEnt, Random Forest, GLM, GAM, BRT).

## Multi-species DeepSDM

A single residual network predicts all species jointly, trained on presence-only data with background points.

```bash
python src/main.py --exp optimized_deepsdm fit
```

The five main-table DeepSDM configurations each have their own experiment config, runnable with `python src/main.py --exp <name> fit`: `baseline_deepsdm`, `aggregation`, `subsampling`, `architecture`, `optimized_deepsdm` (see [`configs/paper_runs.yaml`](../configs/paper_runs.yaml)).

> **Data note.** All runs except `baseline_deepsdm` train on the 1 km-aggregated data in `sage_benchmark_data.tar`. `baseline_deepsdm` sets `aggregated: false` and trains on the non-aggregated presence-only records — to run it, also extract `sage_po_records_nonaggregated.tar` into the same `data_dir` (it merges alongside the 1 km files; see [DATA.md](DATA.md#download)).

### Model (`configs/model/resnet.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `net.class_path` | `models.ResNet` | Model class |
| `net.init_args.num_classes` | Auto | Number of species (read from `species_names.csv`) |
| `net.init_args.hidden_dim` | `512` | Hidden layer dimension |
| `net.init_args.num_blocks` | `8` | Number of residual blocks |
| `net.init_args.use_location` | `false` | Enable location encoding (sin/cos features) |
| `net.init_args.dropout1` | `0.1` | Dropout rate (main) |
| `net.init_args.dropout2` | `0.0` | Dropout rate (secondary) |

### Loss and species weighting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss.class_path` | `criteria.bce.BCE` | Loss function class |
| `loss.init_args.lambda_1` | `1024.0` | Observation loss weight |
| `loss.init_args.lambda_2` | `0.5` | Target-group background loss weight |
| `species_weighting.method` | `"uniform"` | `uniform`, `inversely_proportional`, `inversely_proportional_clipped`, `inversely_proportional_sqrt`, `inversely_proportional_not_normalized` |
| `species_weighting.clip_min` / `clip_max` | `0.05` / `20.0` | Weight clip bounds (only used by `inversely_proportional_clipped`) |

Alternative architectures ship as their own model configs: `simple_mlp.yaml` (plain MLP), `FFTransformer.yaml` (feature-tokenizer transformer), `lr.yaml` (logistic regression).

## Single-species baselines

Traditional models are fit one species at a time: Random Forest, GLM, GAM, and BRT share a single sklearn-based script, and MaxEnt runs through R's `maxnet`. All of them train on the same presence-only + target-group-background data as the DeepSDM and are evaluated against the same presence-absence test set.

### Random Forest / GLM / GAM / BRT

The model and its hyperparameters are set in the chosen `--config`:

```bash
# Train species 0–99 (inclusive) — swap the config to change the model
python scripts/traditional_sdms/sklearn_train.py 0 100 --config configs/model/randomforest.yaml
python scripts/traditional_sdms/sklearn_train.py 0 100 --config configs/model/glm.yaml
```

- `start end` — inclusive range of species indices into `targets/species_names.csv`.
- `--force` — retrain even if a species' model file exists (default: skip existing, so interrupted runs resume cheaply).
- `--data-dir` — point at an alternative data root.

Once models are trained, score them and merge parallel jobs:

```bash
python scripts/traditional_sdms/sklearn_eval.py 0 100 --config configs/model/randomforest.yaml    # per-species metrics
python scripts/traditional_sdms/merge_sklearn_results.py --config configs/model/randomforest.yaml # merge parallel jobs
```

The **full paper protocol** (per model, target-group background) is documented at the top of [`configs/model/sklearn.yaml`](../configs/model/sklearn.yaml) and runs in four steps:

1. **Grid search** — three reference runs at seed 0 over the model's hyperparameter grid (RF `max_features` 7/17/26, GLM `C` 0.001/0.01/0.1, GAM `lam` 1/10/100, BRT `tc` 1/3/5), each with its own `config_tag`.
2. **Select** — `select_reg_param.py --model randomforest` picks, per species, the grid value with the best validation AUROC → `<model>_tgb_param.csv`.
3. **Final runs** — five seeds (0–4) with `per_species_param_file` set to that CSV.
4. **Aggregate** — `aggregate_seed_results.py --model randomforest` → mean ± std test metrics.

### MaxEnt (R `maxnet`)

MaxEnt is fit per-species through a Python launcher over an R `maxnet` backend:

```bash
python scripts/traditional_sdms/maxent_train.py 0 100     # train
python scripts/traditional_sdms/merge_maxent_results.py   # merge parallel jobs
python scripts/traditional_sdms/maxent_eval.py 0 100      # evaluate with range masks
```

| `configs/model/maxent.yaml` | Default | Description |
|-----------|---------|-------------|
| `model` | `"maxent"` | Model type |
| `bg_type` | `"tgb"` | Background sampling: `tgb` (target-group) or `random` |
| `seed` | `0` | Random seed |
| `train_cap` | `10000` | Max training points per species |
| `skip_existing` | `true` | Skip already-trained species |

### Running on a cluster (SLURM)

Each baseline prepares an sbatch array script that computes its species slice from `SLURM_ARRAY_TASK_ID`, spreading all 5,771 species across the array:

```bash
sbatch --array=0-115 scripts/sklearn.sh        # train  (50 species / task)
sbatch --array=0-11  scripts/sklearn_eval.sh   # eval   (500 species / task)
sbatch --array=0-115 scripts/maxent.sh         # MaxEnt train
sbatch --array=0-11  scripts/maxent_eval.sh    # MaxEnt eval
```

The `.py` scripts also run single-machine — just pass a species range on the command line.

## The released runs (model record)

The trained runs are a **separate Zenodo record** from the data: **DOI [10.5281/zenodo.21335338](https://doi.org/10.5281/zenodo.21335338)**. 

### What the record contains

48.3 GB in four tarballs plus the shared ground truth. Each tarball expands to per-run folders named `<model>_s<seed>` (seeds 0–4) that mirror the framework's `logs/` layout.

| File | Size | Expands to | Runs |
|------|------|-----------|------|
| `predictions.tar` | 42.7 GB | `<run>/test_predictions.h5` | all 50 |
| `checkpoints.tar` | 5.6 GB | `<run>/checkpoints/best.ckpt` (+ `ORIGIN.txt`) | 25 deep |
| `configs.tar` | 184 KB | `<run>/config.yaml` — required to load a checkpoint | 25 deep |
| `metrics.tar` | 3.6 MB | `<run>/metrics_by_species.h5` | all 50 |
| `eval_test_targets.h5` | 43 MB | shared ground truth — `targets`, `locations`, `sampled_area_m2` | — |

There are 50 runs: the 5 DeepSDM configurations (`baseline_deepsdm`, `aggregation`, `subsampling`, `architecture`, `optimized_deepsdm`) and the 5 single-species baselines (`glm`, `gam`, `brt`, `randomforest`, `maxent`), each with 5 seeds.

All 50 are provided with predictions and metrics. Only the 25 deep runs additionally come with a checkpoint and a config — the single-species baselines are thousands of fitted per-species model objects, too large to release, so they are reusable through their predictions rather than their weights.

Each deep run's `checkpoints/ORIGIN.txt` records the filename training originally wrote (`epoch=NN-val_auroc=XX.XXXX.ckpt`) and the W&B run id; `best.ckpt` is that same file under a stable name.

### Extracting

Everything goes into your `log_dir` (the one in `configs/local_default.yaml`). The tarballs already contain the `<run>/` folders, so extract at the top of `log_dir` and the layout comes out right. `eval_test_targets.h5` is copied in alongside them.

You rarely need all 48 GB. Adding a run-folder name at the end of a `tar x` command extracts only that folder, so pulling one run costs a few hundred MB instead of tens of GB:

```bash
tar xf predictions.tar -C "$LOG_DIR" optimized_deepsdm_s2
```

Write the folder name literally. A glob like `'optimized_deepsdm_s2/*'` is rejected by GNU tar on Linux (`Use --wildcards to enable pattern matching`), whereas the bare name works the same on Linux and macOS. 

The commands below follow that pattern:

```bash
LOG_DIR=$(python -c "import yaml;print(yaml.safe_load(open('configs/local_default.yaml'))['log_dir'])")

wget -c "https://zenodo.org/records/21335338/files/checkpoints.tar?download=1" -O checkpoints.tar
wget -c "https://zenodo.org/records/21335338/files/configs.tar?download=1"     -O configs.tar
wget -c "https://zenodo.org/records/21335338/files/metrics.tar?download=1"     -O metrics.tar
wget -c "https://zenodo.org/records/21335338/files/predictions.tar?download=1" -O predictions.tar
wget -c "https://zenodo.org/records/21335338/files/eval_test_targets.h5?download=1" -O eval_test_targets.h5

# Check the transfers before unpacking (SHA256SUMS is in the record):
#   wget -c "https://zenodo.org/records/21335338/files/SHA256SUMS?download=1" -O SHA256SUMS && sha256sum -c SHA256SUMS

# One run's checkpoint (~475 MB) + its config (~8 KB). Both are needed to load a deep
# model: the checkpoint stores weights, config.yaml rebuilds the network around them.
tar xf checkpoints.tar -C "$LOG_DIR" optimized_deepsdm_s2
tar xf configs.tar     -C "$LOG_DIR" optimized_deepsdm_s2

# That run's predictions (~926 MB deep, ~443 MB for a baseline) — to score it or use it as --reference
tar xf predictions.tar -C "$LOG_DIR" optimized_deepsdm_s2

# Per-species metrics for every model including the baselines (3.6 MB — just take all of it)
tar xf metrics.tar -C "$LOG_DIR"

# The shared ground truth (43 MB) — this copy carries plot coordinates, the data record's does not
cp eval_test_targets.h5 "$LOG_DIR"/
```

To take the whole record (~48 GB):

```bash
for t in predictions checkpoints configs metrics; do tar xf "$t.tar" -C "$LOG_DIR"; done
cp eval_test_targets.h5 "$LOG_DIR"/
```

Check what a tarball holds before pulling from it with `tar tf predictions.tar | head`.

### The resulting layout

```text
<log_dir>/
├── eval_test_targets.h5               # shared ground truth (targets + locations)
├── optimized_deepsdm_s2/              # a deep run — all four artifacts
│   ├── checkpoints/
│   │   ├── best.ckpt
│   │   └── ORIGIN.txt
│   ├── config.yaml                    # needed to rebuild the network before loading the checkpoint
│   ├── test_predictions.h5
│   └── metrics_by_species.h5
├── randomforest_s0/                   # a baseline run — predictions + metrics
│   ├── test_predictions.h5
│   └── metrics_by_species.h5
└── ...
```

[`configs/paper_runs.yaml`](../configs/paper_runs.yaml) maps each model in the paper to its five run folders.

## Add your own model

To benchmark a new architecture inside the framework, you need three pieces:

1. **A network** — a `torch.nn.Module` in `src/models/` whose `forward` maps a predictor batch to per-species logits of shape `[batch, num_species]`.  `num_species` is injected automatically from `species_names.csv`.  `src/models/simple_mlp.py` is the smallest example to copy.
2. **A model config** — `configs/model/<name>.yaml`. Copy `resnet.yaml` and point `net.class_path` at your class (with its `init_args`); the loss, metrics, and species-weighting blocks can stay unchanged.
3. **An experiment config** — set `model_config: <name>` in a `configs/experiments/<exp>.yaml` (alongside `data_config` and any overrides).

Then train it exactly like the built-in models:

```bash
python src/main.py --exp <exp> fit
```


## Notes

- **Device.** Training defaults to `accelerator: auto` (`configs/default.yaml`) — it uses a CUDA GPU when present, Apple MPS on Apple Silicon, or the CPU otherwise. Force a device with `--trainer.accelerator=cpu` (or `gpu`) `--trainer.devices=1`. Full DeepSDM training is GPU-scale; on CPU use it only for a smoke test (add `--trainer.fast_dev_run true`).

- **Networked filesystems.** The traditional-baseline scripts set `HDF5_USE_FILE_LOCKING=FALSE` so many parallel workers can read the shared HDF5 stores on a cluster filesystem. If you see HDF5 file-locking errors when reading the data from your own scripts, export the same variable.
