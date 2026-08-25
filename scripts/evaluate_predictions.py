#!/usr/bin/env python
"""Evaluate model predictions against the SDM benchmark test set.

Bring your own model's predictions (in the benchmark prediction format, see
``src/evaluation/prediction_io.py``) and reproduce the benchmark's per-species
metrics — AUROC and AUPRG by default — without running the training loop. The
numbers match the paper's evaluation exactly (same metric definitions).

Examples
--------
Evaluate your own probabilities against the benchmark test set:

    python scripts/evaluate_predictions.py \
        --predictions my_model_probs.h5 \
        --data-dir /path/to/data --output-dir out/my_model

Compare against a reference run (adds per-species and macro deltas)::

    python scripts/evaluate_predictions.py \
        --predictions my_model_probs.h5 \
        --reference logs/optimized_deepsdm_s0/test_predictions.h5 \
        --data-dir /path/to/data --output-dir out/my_model_vs_deepsdm

A predictions file that already bundles ``targets`` (e.g. a benchmark
``test_predictions.h5``) is scored as-is; ``--data-dir`` is then only used for
species names and stratification.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.evaluation import (  # noqa: E402
    compute_species_metrics,
    load_predictions,
    macro_summary,
    save_metrics_h5,
)
from src.evaluation.stratify import DEFAULT_X, DEFAULT_Y, stratified_report  # noqa: E402
from src.visualizations.delta_heatmap import plot_delta_heatmaps  # noqa: E402

DEFAULT_METRICS = ["auroc", "auprg"]


def _default_targets_path(data_dir: str | None) -> str | None:
    """Benchmark test-split ground truth shipped with the data (``targets/eval_test_targets.h5``).

    Used when ``--targets`` is omitted, so a user who downloaded the data does not
    have to point at the labels explicitly.
    """
    if data_dir:
        path = os.path.join(data_dir, "targets", "eval_test_targets.h5")
        if os.path.exists(path):
            return path
    return None


def _load_species_names(data_dir: str | None) -> pd.DataFrame | None:
    if not data_dir:
        return None
    path = os.path.join(data_dir, "targets", "species_names.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_proxies(data_dir: str | None, proxies_path: str | None) -> pd.DataFrame | None:
    """Load species-property proxies (compute_species_ranks output) for stratification."""
    if proxies_path:
        return pd.read_csv(proxies_path)
    if not data_dir:
        return None
    for name in ("species_properties_1km.csv", "species_properties.csv"):
        path = os.path.join(data_dir, "targets", name)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def _metrics_dataframe(results: dict, species_names: pd.DataFrame | None) -> pd.DataFrame:
    n_species = len(next(iter(results.values())))
    df = pd.DataFrame({"species_index": np.arange(n_species)})
    if species_names is not None and len(species_names) == n_species:
        df["species_name"] = species_names["Species Name"].to_numpy()
    for name, scores in results.items():
        df[name] = scores
    return df


def _evaluate(path: str, targets_path: str | None, metrics: list, batch_size: int, device: str) -> dict:
    data = load_predictions(path, targets_path=targets_path)
    if data["targets"] is None:
        raise SystemExit(
            f"No ground-truth targets for '{path}'. Pass --data-dir (which ships "
            "targets/eval_test_targets.h5), give --targets explicitly, or use a predictions file "
            "that bundles a 'targets' dataset."
        )
    return compute_species_metrics(
        data["probabilities"], data["targets"], metrics=metrics, batch_size=batch_size, device=device
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="Predictions file (HDF5 or CSV).")
    p.add_argument("--targets", default=None,
                   help="Ground-truth targets file (HDF5). "
                        "Default: <data-dir>/targets/eval_test_targets.h5.")
    p.add_argument("--reference", default=None, help="Optional reference predictions file for delta comparison.")
    p.add_argument("--output-dir", required=True, help="Directory for metric outputs.")
    p.add_argument("--data-dir", default=None,
                   help="Benchmark data dir; supplies the ground-truth targets, species names, "
                        "and stratification proxies.")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   help=f"Metrics to compute (default: {DEFAULT_METRICS}).")
    p.add_argument("--batch-size", type=int, default=8192, help="Locations per metric update (memory knob).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device for metric state.")
    p.add_argument("--proxies", default=None,
                   help="Species-property CSV for stratification "
                        "(default: <data-dir>/targets/species_properties_1km.csv).")
    p.add_argument("--x-axis", default=DEFAULT_X, help=f"Stratification x property (default: {DEFAULT_X}).")
    p.add_argument("--y-axis", default=DEFAULT_Y, help=f"Stratification y property (default: {DEFAULT_Y}).")
    p.add_argument("--extreme-pct", type=float, default=10.0,
                   help="Percentile for the extreme outer bins (default: 10).")
    p.add_argument("--no-stratify", action="store_true", help="Skip the quadrant / extreme-bin breakdown.")
    p.add_argument("--heatmap-window", type=int, default=850,
                   help="Species per moving window for a 'fine' delta heatmap (default: 850).")
    p.add_argument("--heatmap-mode", default="coarse", choices=["coarse", "fine"],
                   help="Delta heatmap style: 'coarse' = one average per region (paper main figure), "
                        "'fine' = moving-window surface.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    species_names = _load_species_names(args.data_dir)
    targets_path = args.targets or _default_targets_path(args.data_dir)

    print(f"[eval] computing metrics for {args.predictions} ...")
    results = _evaluate(args.predictions, targets_path, args.metrics, args.batch_size, args.device)
    save_metrics_h5(results, os.path.join(args.output_dir, "metrics_by_species.h5"))
    df = _metrics_dataframe(results, species_names)

    summary = macro_summary(results)
    ref_summary = None
    ref_results = None
    if args.reference:
        print(f"[eval] computing metrics for reference {args.reference} ...")
        ref_results = _evaluate(args.reference, targets_path, args.metrics, args.batch_size, args.device)
        ref_summary = macro_summary(ref_results)
        for name in args.metrics:
            df[f"{name}_reference"] = ref_results[name]
            df[f"{name}_delta"] = results[name] - ref_results[name]

    csv_path = os.path.join(args.output_dir, "species_metrics.csv")
    df.to_csv(csv_path, index=False)

    # Human-readable macro summary.
    lines = ["metric        macro(%)" + ("   reference   delta" if ref_summary else "")]
    for name in args.metrics:
        line = f"{name:<12}  {summary[name]:7.3f}"
        if ref_summary:
            line += f"   {ref_summary[name]:8.3f}  {summary[name] - ref_summary[name]:+6.3f}"
        lines.append(line)
        # AUPRG: report the robust median alongside the (outlier-sensitive) mean.
        if name == "auprg" and "auprg_median" in summary:
            mline = f"{'auprg(median)':<12}  {summary['auprg_median']:7.3f}"
            if ref_summary and "auprg_median" in ref_summary:
                mline += (f"   {ref_summary['auprg_median']:8.3f}  "
                          f"{summary['auprg_median'] - ref_summary['auprg_median']:+6.3f}")
            lines.append(mline)
    report = "\n".join(lines)

    # Stratified breakdown: quadrants + extreme bins on the delta-heatmap axes.
    if not args.no_stratify:
        proxies = _load_proxies(args.data_dir, args.proxies)
        if proxies is None:
            report += "\n\n[stratified] no proxies CSV found (pass --data-dir or --proxies) — skipped."
        else:
            strat = stratified_report(
                results, proxies, x_col=args.x_axis, y_col=args.y_axis,
                reference=ref_results, extreme_pct=args.extreme_pct, metrics=args.metrics,
            )
            report += "\n\n" + strat

            # Delta heatmap (the paper's 2-D moving-window figure): needs a reference.
            if ref_results is not None and args.x_axis in proxies.columns and args.y_axis in proxies.columns:
                hm_metric = args.metrics[0]
                idx = proxies.set_index("species_id")
                nsp = len(results[hm_metric])
                hx = np.where(np.isfinite(idx[args.x_axis].reindex(range(nsp)).to_numpy(dtype=float)),
                              idx[args.x_axis].reindex(range(nsp)).to_numpy(dtype=float), np.nan)
                hy = np.where(np.isfinite(idx[args.y_axis].reindex(range(nsp)).to_numpy(dtype=float)),
                              idx[args.y_axis].reindex(range(nsp)).to_numpy(dtype=float), np.nan)
                fin = np.isfinite(hx) & np.isfinite(hy)
                hm_prox = pd.DataFrame({args.x_axis: hx[fin], args.y_axis: hy[fin]})
                pairs = [{
                    "m1": ref_results[hm_metric][fin],   # reference
                    "m2": results[hm_metric][fin],       # this run
                    "label": "this run − reference",
                }]
                # Unbounded metrics (e.g. AUPRG) use the robust median aggregation.
                agg = "median" if "auprg" in hm_metric.lower() else "mean"
                window = min(args.heatmap_window, max(2, int(fin.sum()) // 2))
                hm_path = os.path.join(args.output_dir, "delta_heatmap.png")
                plot_delta_heatmaps(
                    hm_prox, args.x_axis, args.y_axis, pairs, mode=args.heatmap_mode,
                    window_size=window, agg=agg, metric_label=hm_metric.upper(), out_path=hm_path)
                report += (
                    f"\n\n[delta heatmap] {hm_metric} "
                    f"(green = this run better, purple = reference better): wrote {hm_path}"
                )

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(report + "\n")

    print(f"[eval] wrote {csv_path}")
    print(f"[eval] wrote {os.path.join(args.output_dir, 'metrics_by_species.h5')}")
    print("\n" + report)


if __name__ == "__main__":
    main()
