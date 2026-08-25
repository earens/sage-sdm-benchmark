"""Entry point for the SDM benchmark.

Merges the layered YAML configs (local_default -> default -> data -> model ->
experiment), injects ``num_species``/``num_features`` from ``species_names.csv``,
builds the WandB run name, and launches the ``CustomLightningCLI``.
Run: ``python src/main.py --exp <experiment> {fit,test,validate,predict}``.
"""

import os
import logging
import pyrootutils
import argparse
import re
from omegaconf import OmegaConf
import tempfile
import sys
import signal


pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False)
from src.utils import CustomLightningCLI  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

config_dir = "configs/"

if __name__ == "__main__":

    # Handle Ctrl+C (SIGINT) and termination (SIGTERM) — restore default after first signal
    def signal_handler(signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        logger.info(f"Received signal {signum}, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if os.environ.get("DEBUG", False):
        import debugpy

        debugpy.listen(5678)
        logger.info("Waiting for debugger attach")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", required=True, help="Basename of experiment config.")
    args, unknown = parser.parse_known_args()
    exp_config_name = args.exp

    local_default_cfg = OmegaConf.load(os.path.join(config_dir, "local_default.yaml"))
    log_dir = local_default_cfg["log_dir"]

    experiment_cfg = OmegaConf.load(os.path.join(config_dir, "experiments/", f"{exp_config_name}.yaml"))
    default_cfg = OmegaConf.load(os.path.join(config_dir, "default.yaml"))
    data_cfg = OmegaConf.load(os.path.join(config_dir, "data/", f"{experiment_cfg.data_config}.yaml"))
    model_cfg = OmegaConf.load(os.path.join(config_dir, "model/", f"{experiment_cfg.model_config}.yaml"))

    # data_dir: the experiment config may override the default from local_default.yaml
    if "data_dir" in experiment_cfg:
        data_dir = experiment_cfg.pop("data_dir")
    else:
        data_dir = local_default_cfg["data_dir"]

    default_cfg.trainer.logger.init_args.save_dir = log_dir
    data_cfg.data.init_args.data_dir = data_dir
    experiment_cfg.pop("data_config")
    experiment_cfg.pop("model_config")

    merged_cfg = OmegaConf.merge(
        default_cfg,
        data_cfg,
        model_cfg,
        experiment_cfg
    )

    # Read num_species from the canonical species list
    import pandas as pd
    species_df = pd.read_csv(os.path.join(data_dir, "targets", "species_names.csv"))
    num_species = len(species_df)
    logger.info(f"Read {num_species} species from species_names.csv")

    # Compute num_features from predictor subset (excludes location, which is handled via use_location)
    predictor_subset = merged_cfg.data.init_args.get("predictors_subset", {})
    num_features = sum(len(v) for k, v in predictor_subset.items() if k != "location")
    logger.info(f"Computed num_features={num_features} from predictor subset")

    # Inject num_species and num_features into model config
    merged_cfg.model.init_args.net.init_args.num_classes = num_species
    # Only fill num_features for nets that declared it as a placeholder (e.g.
    # FTTransformer/FFTransformer: `num_features: null`). Nets that infer their
    # input dim lazily (e.g. ResNet's LazyLinear) omit the key entirely and must
    # not have it injected, or jsonargparse rejects the unexpected arg.
    net_init = merged_cfg.model.init_args.net.init_args
    if "num_features" in net_init and net_init.get("num_features") is None:
        net_init.num_features = num_features
    for stage in ["train", "val", "test"]:
        if stage in merged_cfg.model.init_args.metrics:
            for metric_name in merged_cfg.model.init_args.metrics[stage]:
                metric = merged_cfg.model.init_args.metrics[stage][metric_name]
                if "init_args" in metric and "num_labels" in metric.init_args:
                    metric.init_args.num_labels = num_species

    # Build wandb run name: TR/TE_<predictors>_PW<number>_A/F_M/NM[_S<number>]
    # Determine mode from CLI subcommand (fit=TR, test=TE)
    is_test = "test" in unknown
    mode_str = "TE" if is_test else "TR"

    # For test mode, load the saved config from the checkpoint run directory
    if is_test:
        # Find --ckpt_path in unknown args
        ckpt_path = None
        for i, arg in enumerate(unknown):
            if arg.startswith("--ckpt_path="):
                ckpt_path = arg.split("=", 1)[1]
            elif arg == "--ckpt_path" and i + 1 < len(unknown):
                ckpt_path = unknown[i + 1]
        if ckpt_path:
            # checkpoint is in <run_dir>/checkpoints/<ckpt>.ckpt -> go up to <run_dir>
            run_dir = os.path.dirname(os.path.dirname(ckpt_path))
            saved_config_path = os.path.join(run_dir, "config.yaml")
            if os.path.exists(saved_config_path):
                name_cfg = OmegaConf.load(saved_config_path)
                logger.info(f"Loaded saved config from {saved_config_path} for run naming")
            else:
                logger.warning(f"No config.yaml found at {saved_config_path}, using current config")
                name_cfg = merged_cfg
        else:
            logger.warning("No --ckpt_path provided for test, using current config")
            name_cfg = merged_cfg
    else:
        name_cfg = merged_cfg

    # Helper to extract naming values from a config (handles both saved and merged formats)
    def get_name_values(cfg):
        """Extract values needed for WandB run naming from either saved or merged config."""
        # Predictors: L=location (from use_location), C=chelsa, S=soilgrids, T=topography, H=human_footprint
        pred_map = {
            "chelsa": "C",
            "soilgrids": "S",
            "topography": "T",
            "human_footprint": "H",
        }

        # Location depends on use_location model flag, not predictors_subset
        try:
            use_loc = cfg.model.init_args.net.init_args.get("use_location", False)
        except Exception:
            use_loc = False
        pred_str = "L" if use_loc else ""

        try:
            pred_subset = cfg.data.init_args.get("predictors_subset", {})
        except Exception:
            pred_subset = {}
        pred_str += "".join(
            letter for key, letter in pred_map.items()
            if key in pred_subset and pred_subset[key]
        )

        # Positive weight from loss lambda_1
        try:
            pw = cfg.model.init_args.loss.init_args.get("lambda_1", 1.0)
        except Exception:
            pw = 1.0

        # Aggregated (A) or Full (F)
        try:
            agg = cfg.data.init_args.get("aggregated", False)
        except Exception:
            agg = False
        agg_str = "A" if agg else "F"

        # PO range mask
        try:
            po_mask = cfg.data.init_args.get("po_range_mask", False)
        except Exception:
            po_mask = False
        mask_str = "M" if po_mask else "NM"

        # Subsampling
        try:
            sub_file = cfg.data.init_args.get("subsample_indices_file", None)
        except Exception:
            sub_file = None
        sub_str = ""
        if sub_file:
            match = re.search(r'_(\d+)(?:_1km)?\.npy', str(sub_file))
            sub_str = f"_S{match.group(1)}" if match else "_S"

        return pred_str, pw, agg_str, mask_str, sub_str

    pred_str, pw, agg_str, mask_str, sub_str = get_name_values(name_cfg)
    pw_str = f"PW{int(pw)}" if float(pw) == int(float(pw)) else f"PW{pw}"
    run_name = f"{mode_str}_{pred_str}_{pw_str}_{agg_str}_{mask_str}{sub_str}"
    merged_cfg.trainer.logger.init_args.name = run_name
    logger.info(f"Wandb run name: {run_name}")

    # Extract dot-notation overrides (e.g. --model.init_args.loss.init_args.lambda_1=2048)
    # and merge them via OmegaConf instead of passing to LightningCLI,
    # which cannot resolve deeply nested init_args.
    passthrough_args = []
    dotlist_entries = []
    # Top-level scalar keys that should be merged into OmegaConf (not passed through to LightningCLI)
    TOP_LEVEL_KEYS = {'seed_everything'}
    for arg in unknown:
        # Match --some.dotted.key=value or --top_level_key=value
        m = re.match(r'^--([\w.]+)=(.+)$', arg)
        if m:
            key, val = m.group(1), m.group(2)
            if '.' in key or key in TOP_LEVEL_KEYS:
                dotlist_entries.append(f"{key}={val}")
            else:
                passthrough_args.append(arg)
        else:
            passthrough_args.append(arg)

    if dotlist_entries:
        logger.info(f"Applying CLI overrides via OmegaConf: {dotlist_entries}")
        cli_overrides = OmegaConf.from_dotlist(dotlist_entries)
        merged_cfg = OmegaConf.merge(merged_cfg, cli_overrides)

    # Recompute run name after overrides (lambda_1 or seed_everything may have changed)
    pred_str, pw, agg_str, mask_str, sub_str = get_name_values(merged_cfg)
    pw_str = f"PW{int(pw)}" if float(pw) == int(float(pw)) else f"PW{pw}"
    run_name = f"{mode_str}_{pred_str}_{pw_str}_{agg_str}_{mask_str}{sub_str}"
    merged_cfg.trainer.logger.init_args.name = run_name
    if dotlist_entries:
        logger.info(f"Updated wandb run name: {run_name}")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(merged_cfg, f.name)
        merged_config_path = f.name

    sys.argv = [sys.argv[0]]

    if is_test and ckpt_path:
        # Robust checkpoint reload for `test --ckpt_path`: build the model, datamodule
        # and trainer from the (--exp) config, then run test with the checkpoint providing
        # WEIGHTS ONLY. This deliberately bypasses LightningCLI's own `--ckpt_path` handling,
        # which rebuilds the model from the checkpoint's saved hyperparameters and breaks
        # across pytorch-lightning / jsonargparse versions (e.g. pl >= 2.5 rejects the
        # LightningModule's subclass kwargs). Lightning's restore — invoked by
        # trainer.test(..., ckpt_path=...) — does the torch.load in a way that is correct
        # for the installed torch version and calls the module's on_load_checkpoint hook.
        logger.info(f"Loading checkpoint weights from {ckpt_path} into a config-built model")
        cli = CustomLightningCLI(args=["-c", merged_config_path], run=False)
        cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    else:
        cli_args = passthrough_args + ["-c", merged_config_path]
        cli = CustomLightningCLI(args=cli_args)
